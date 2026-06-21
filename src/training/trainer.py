"""
trainer.py — Training loop for Bangla OCR V2 (ViT + XLM-RoBERTa).

Key improvements over V1:
  1. Phased training: encoder is frozen for the first N epochs (warm-up),
     then un-frozen with a small separate LR group (fine-tuning).
  2. Linear warm-up → CosineAnnealingLR schedule (same as V1 but now the
     warm-up is handled explicitly so it doesn't fight the cosine decay).
  3. Checkpoint strategy: always overwrite 'last_model.pt'; only write
     'best_model.pt' when CER improves.  This is disk-safe for Kaggle.
  4. Gradient accumulation support (accumulate_steps > 1) so you can
     effectively train with a larger batch on a single 16 GB GPU.
  5. TensorBoard logging of loss, LR, CER, WER, and exact-match accuracy.
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.training.metrics import evaluate_model
from src.models.ocr_model import unfreeze_encoder


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        config: dict,
        device: torch.device,
        output_dir: str = "/kaggle/working/checkpoints",
        accumulate_steps: int = 1,
    ):
        self.model            = model
        self.train_loader     = train_loader
        self.val_loader       = val_loader
        self.tokenizer        = tokenizer
        self.config           = config
        self.device           = device
        self.output_dir       = output_dir
        self.accumulate_steps = max(1, accumulate_steps)

        os.makedirs(output_dir, exist_ok=True)

        # Optimizer — only trainable params (encoder may be frozen initially)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        # LR schedule: linear warm-up then cosine annealing
        total_steps  = (len(train_loader) // self.accumulate_steps) * config["training"]["epochs"]
        warmup_steps = config["training"]["warmup_steps"]

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Mixed precision
        self.use_amp  = config["training"]["mixed_precision"]
        self.scaler   = GradScaler("cuda", enabled=self.use_amp)

        # Label-smoothed loss (Suggestion 6) — computed from logits manually
        # because VisionEncoderDecoderModel's internal loss does not support
        # label_smoothing.  We use ignore_index=-100 (the padding sentinel).
        label_smoothing = config["model"].get("label_smoothing", 0.1)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=-100,
        )
        print(f"  Label smoothing : {label_smoothing}")

        # Early stopping state
        self.best_cer         = float("inf")
        self.patience_counter = 0
        self.patience         = config["training"]["early_stop_patience"]

        self.global_step           = 0
        self.freeze_encoder_epochs = config["model"].get("freeze_encoder_epochs", 5)
        self.encoder_unfrozen      = False

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
        except ImportError:
            self.writer = None

    # ──────────────────────────────────────────
    # Training epoch
    # ──────────────────────────────────────────

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss  = 0.0
        step_loss   = 0.0
        progress    = tqdm(self.train_loader, desc=f"Epoch {epoch:03d}", leave=True)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress):
            pixel_values = batch["pixel_values"].to(self.device)
            labels       = batch["labels"].to(self.device)

            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(
                    pixel_values=pixel_values,
                    labels=labels,
                )
                # Use our label-smoothed criterion instead of outputs.loss.
                # logits shape: (B, seq_len, vocab_size)
                # labels shape: (B, seq_len)  with -100 at pad positions
                logits = outputs.logits  # (B, T, V)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),  # (B*T, V)
                    labels.reshape(-1),                   # (B*T,)
                ) / self.accumulate_steps

            self.scaler.scale(loss).backward()
            step_loss += loss.item()

            # Update weights every `accumulate_steps` mini-batches
            if (batch_idx + 1) % self.accumulate_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["gradient_clip"],
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if self.writer and self.global_step % self.config["training"]["log_every"] == 0:
                    self.writer.add_scalar("Loss/train", step_loss, self.global_step)
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("LR", lr, self.global_step)

                total_loss += step_loss
                progress.set_postfix({"loss": f"{step_loss:.4f}"})
                step_loss = 0.0
                self.global_step += 1

        steps = max(1, len(self.train_loader) // self.accumulate_steps)
        return total_loss / steps

    # ──────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────

    def validate(self, epoch: int) -> dict:
        metrics = evaluate_model(
            model=self.model,
            dataloader=self.val_loader,
            tokenizer=self.tokenizer,
            device=self.device,
            num_beams=self.config["training"].get("num_beams", 4),
        )
        if self.writer:
            self.writer.add_scalar("CER/val",      metrics["cer"],                 epoch)
            self.writer.add_scalar("WER/val",      metrics["wer"],                 epoch)
            self.writer.add_scalar("Accuracy/val", metrics["exact_match_accuracy"], epoch)
        return metrics

    # ──────────────────────────────────────────
    # Checkpointing (disk-safe: overwrite, never accumulate)
    # ──────────────────────────────────────────

    def _save(self, path: str, epoch: int, metrics: dict):
        torch.save(
            {
                "epoch":              epoch,
                "model_state_dict":   self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict":  self.scaler.state_dict(),
                "metrics":            metrics,
                "config":             self.config,
            },
            path,
        )

    # ──────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────

    def run(self) -> float:
        cfg = self.config["training"]

        for epoch in range(1, cfg["epochs"] + 1):

            # ── Phase transition: un-freeze encoder ──
            if (not self.encoder_unfrozen
                    and epoch > self.freeze_encoder_epochs):
                print(f"\n[Epoch {epoch}] Un-freezing encoder for fine-tuning...")
                unfreeze_encoder(
                    self.model,
                    self.optimizer,
                    encoder_lr=self.config["training"].get("encoder_lr", 5e-6),
                )
                self.encoder_unfrozen = True

            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            cer_val = val_metrics["cer"]
            wer_val = val_metrics["wer"]
            acc_val = val_metrics["exact_match_accuracy"]

            print(
                f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
                f"CER={cer_val:.4f} | WER={wer_val:.4f} | "
                f"Acc={acc_val:.4f}"
            )

            # Always overwrite last checkpoint (disk-safe)
            last_path = os.path.join(self.output_dir, "last_model.pt")
            self._save(last_path, epoch, val_metrics)

            # Save best only on improvement
            if cer_val < self.best_cer:
                self.best_cer         = cer_val
                self.patience_counter = 0
                best_path = os.path.join(self.output_dir, "best_model.pt")
                self._save(best_path, epoch, val_metrics)
                print(f"  ✓ New best CER={cer_val:.4f} — saved best_model.pt")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.patience})")
                if self.patience_counter >= self.patience:
                    print(f"  Early stopping triggered after {epoch} epochs.")
                    break

        if self.writer:
            self.writer.close()

        return self.best_cer
