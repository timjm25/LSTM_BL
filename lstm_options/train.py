"""Training loop for LSTM Options Pricing model."""
import math
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from .config import OptionsModelConfig, OptionsTrainConfig
from .dataset import make_options_dataloaders
from .model import OptionsLSTM, build_options_model


class OptionsPricingLoss(nn.Module):
    """
    Combined loss for options pricing:
      - MSE on normalised price (main signal)
      - Penalise negative predicted prices (put-call-parity regularisation)
    """

    def __init__(self, no_arb_weight: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.no_arb_weight = no_arb_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        mse_loss = self.mse(pred.squeeze(-1), target)
        # No-arbitrage: predicted (unnormalised) price should be >= 0
        no_arb_penalty = torch.relu(-pred).mean()
        total = mse_loss + self.no_arb_weight * no_arb_penalty
        return {"loss": total, "mse": mse_loss, "no_arb": no_arb_penalty}


class OptionsTrainer:
    def __init__(
        self,
        train_cfg: OptionsTrainConfig,
        model_cfg: OptionsModelConfig,
        option_type: str = "call",
    ):
        self.tcfg = train_cfg
        self.mcfg = model_cfg

        if train_cfg.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(train_cfg.device)

        torch.manual_seed(train_cfg.seed)

        self.model = build_options_model(model_cfg).to(self.device)
        self.loss_fn = OptionsPricingLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.train_dl, self.val_dl, self.test_dl = make_options_dataloaders(
            train_cfg, model_cfg, option_type=option_type
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg.epochs - train_cfg.warmup_epochs,
        )
        Path(train_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "val_rmse": [], "lr": []
        }

    def _warmup_lr(self, epoch: int):
        if epoch < self.tcfg.warmup_epochs:
            lr = self.tcfg.learning_rate * (epoch + 1) / self.tcfg.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n = 0
        for feats, labels in self.train_dl:
            feats = feats.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            pred, _ = self.model(feats)
            losses = self.loss_fn(pred, labels)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
            self.optimizer.step()
            total_loss += losses["loss"].item() * len(labels)
            n += len(labels)
        return {"loss": total_loss / n}

    @torch.no_grad()
    def eval_epoch(self, loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_pred, all_true = [], []
        for feats, labels in loader:
            feats = feats.to(self.device)
            labels = labels.to(self.device)
            pred, _ = self.model(feats)
            losses = self.loss_fn(pred, labels)
            total_loss += losses["loss"].item() * len(labels)
            all_pred.append(pred.squeeze(-1).cpu())
            all_true.append(labels.cpu())
        preds = torch.cat(all_pred)
        trues = torch.cat(all_true)
        rmse = ((preds - trues) ** 2).mean().sqrt().item()
        n = sum(len(l) for l in all_true)
        return {"loss": total_loss / n, "rmse": rmse}

    def save_checkpoint(self, epoch: int, val_loss: float):
        path = Path(self.tcfg.checkpoint_dir) / "best_options.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "model_config": self.mcfg,
            "history": self.history,
        }, path)

    def run(self) -> Dict[str, list]:
        print(f"[options-train] Device: {self.device}  Params: {self.model.num_parameters:,}")
        for epoch in range(self.tcfg.epochs):
            self._warmup_lr(epoch)
            tr = self.train_epoch()
            vl = self.eval_epoch(self.val_dl)

            if epoch >= self.tcfg.warmup_epochs:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(tr["loss"])
            self.history["val_loss"].append(vl["loss"])
            self.history["val_rmse"].append(vl["rmse"])
            self.history["lr"].append(lr)

            if vl["loss"] < self.best_val_loss:
                self.best_val_loss = vl["loss"]
                self.save_checkpoint(epoch, vl["loss"])

            if epoch % 10 == 0 or epoch == self.tcfg.epochs - 1:
                print(
                    f"  Epoch {epoch:3d}/{self.tcfg.epochs} | "
                    f"train={tr['loss']:.5f} | "
                    f"val={vl['loss']:.5f} | rmse={vl['rmse']:.5f} | "
                    f"lr={lr:.2e}"
                )

        test_metrics = self.eval_epoch(self.test_dl)
        print(f"\n[options-train] Test loss={test_metrics['loss']:.5f}  RMSE={test_metrics['rmse']:.5f}")
        return self.history
