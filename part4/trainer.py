"""
Training utilities.
Example submission.
"""

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import time
import sys
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import cross_entropy, gradient_clipping


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    batch_size: int = 8
    log_interval: int = 10
    save_interval: int = 500
    checkpoint_dir: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False
    patience: Optional[int] = None
    use_wandb: bool = False
    val_per_steps: Optional[int] = None  


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        compute_loss_fn: Optional[Callable] = None,
    ):
        self.model = model.to(config.device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.compute_loss_fn = compute_loss_fn or self._default_lm_loss
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        total_steps = len(train_dataloader) * config.num_epochs
        if config.warmup_steps > 0:
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=config.warmup_steps,
            )
            main = CosineAnnealingLR(
                self.optimizer, T_max=total_steps - config.warmup_steps
            )
            self.scheduler = SequentialLR(
                self.optimizer, [warmup, main], milestones=[config.warmup_steps]
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0   
        self.train_losses = []
        self.val_losses = []
        self.scaler = GradScaler("cuda", enabled=config.use_amp)

    def _default_lm_loss(
        self, batch: Dict[str, torch.Tensor], model: nn.Module
    ) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)
        logits = model(input_ids)
        batch_size, seq_len, vocab_size = logits.shape
        return cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(self.train_dataloader, desc="  Batches", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=self.config.use_amp):
                loss = self.compute_loss_fn(batch, self.model)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            gradient_clipping(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=self.global_step)

            if self.config.use_wandb and _WANDB_AVAILABLE and wandb.run is not None:
                current_lr = self.scheduler.get_last_lr()[0]
                wandb.log(
                    {
                        "train/step_loss": loss.item(),
                        "train/lr": current_lr,
                    },
                    step=self.global_step,
                )

            if (
                self.config.val_per_steps is not None
                and self.val_dataloader is not None
                and self.global_step % self.config.val_per_steps == 0
            ):
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    val_loss=f"{val_loss:.4f}",
                    step=self.global_step,
                )
                if self.config.use_wandb and _WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({"val/loss": val_loss}, step=self.global_step)
                self.model.train()  

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_dataloader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Validating", unit="batch"):
            loss = self.compute_loss_fn(batch, self.model)
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(self) -> Dict[str, Any]:
        print("Begin to train llm!")
        for epoch in tqdm(range(self.config.num_epochs), desc="Epochs", unit="epoch"):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            epoch_log: Dict[str, Any] = {"train/epoch_loss": train_loss, "epoch": epoch + 1}

            if self.val_dataloader:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                epoch_log["val/loss"] = val_loss

                if self.config.patience is not None:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.epochs_no_improve = 0
                    else:
                        self.epochs_no_improve += 1
                        if self.epochs_no_improve >= self.config.patience:
                            if self.config.use_wandb and _WANDB_AVAILABLE and wandb.run is not None:
                                wandb.log(epoch_log, step=self.global_step)
                            break

            if self.config.use_wandb and _WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(epoch_log, step=self.global_step)

        return {"train_losses": self.train_losses, "val_losses": self.val_losses}


def compute_qa_loss(
    batch: Dict[str, torch.Tensor], model: nn.Module, device: str = "cuda"
) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids, attention_mask)
    return cross_entropy(logits, labels)


def create_qa_loss_fn(device: str = "cuda") -> Callable:
    return lambda batch, model: compute_qa_loss(batch, model, device)
