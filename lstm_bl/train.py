"""Training loop for LSTM Black-Litterman portfolio model."""
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import BLModelConfig, BLTrainConfig
from .dataset import make_bl_dataloaders
from .model import BLPortfolioLSTM, build_bl_model


class PortfolioLoss(nn.Module):
    """
    Multi-objective portfolio loss:
      1. Weight MSE         — match target BL weights
      2. Sharpe proxy       — maximise risk-adjusted returns on validation data
      3. Diversification    — penalise concentration (entropy regularisation)
      4. Turnover           — penalise large weight changes (optional)
    """

    def __init__(
        self,
        sharpe_weight: float = 0.5,
        diversification_weight: float = 0.1,
        turnover_weight: float = 0.05,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.sharpe_w = sharpe_weight
        self.div_w = diversification_weight
        self.to_w = turnover_weight
        self.prev_weights = None

    def forward(
        self,
        pred_weights: torch.Tensor,
        target_weights: torch.Tensor,
        returns_window: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_weights   : (B, n) predicted portfolio weights
            target_weights : (B, n) target BL weights
            returns_window : (B, T, n) asset returns from sequence
        """
        mse_loss = self.mse(pred_weights, target_weights)

        # Sharpe proxy: maximise mean / std of portfolio returns
        port_returns = (pred_weights.unsqueeze(1) * returns_window).sum(-1)  # (B, T)
        mean_ret = port_returns.mean(dim=1)   # (B,)
        std_ret = port_returns.std(dim=1) + 1e-8
        sharpe_loss = -(mean_ret / std_ret).mean()

        # Diversification: maximise entropy of weights
        entropy = -(pred_weights * (pred_weights + 1e-8).log()).sum(-1).mean()
        div_loss = -entropy

        # Turnover penalty
        to_loss = torch.tensor(0.0, device=pred_weights.device)
        if self.prev_weights is not None and self.prev_weights.shape == pred_weights.shape:
            to_loss = (pred_weights - self.prev_weights.detach()).abs().sum(-1).mean()
        self.prev_weights = pred_weights.detach().clone()

        total = (
            mse_loss
            + self.sharpe_w * sharpe_loss
            + self.div_w * div_loss
            + self.to_w * to_loss
        )
        return {
            "loss": total,
            "mse": mse_loss,
            "sharpe_loss": sharpe_loss,
            "div_loss": div_loss,
            "to_loss": to_loss,
        }


def _sharpe(weights: torch.Tensor, returns: torch.Tensor) -> float:
    """Annualised Sharpe ratio for a batch (mean over batch)."""
    port_r = (weights * returns[:, -1, :]).sum(-1)   # approximate: use last day
    mean = port_r.mean().item() * 252
    std = port_r.std().item() * np.sqrt(252) + 1e-8
    return mean / std


class BLTrainer:
    def __init__(self, train_cfg: BLTrainConfig, model_cfg: BLModelConfig):
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

        self.model = build_bl_model(model_cfg).to(self.device)
        self.loss_fn = PortfolioLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.train_dl, self.val_dl, self.test_dl = make_bl_dataloaders(train_cfg, model_cfg)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg.epochs - train_cfg.warmup_epochs,
        )
        Path(train_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "val_sharpe": [], "lr": []
        }

    def _warmup_lr(self, epoch: int):
        if epoch < self.tcfg.warmup_epochs:
            lr = self.tcfg.learning_rate * (epoch + 1) / self.tcfg.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.loss_fn.prev_weights = None
        total_loss = 0.0
        n = 0
        for feats, targets in self.train_dl:
            feats = feats.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            pred_w, Q, omega = self.model(feats)
            losses = self.loss_fn(pred_w, targets, feats[:, :, :self.mcfg.n_assets])
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
            self.optimizer.step()
            total_loss += losses["loss"].item() * len(feats)
            n += len(feats)
        return {"loss": total_loss / n}

    @torch.no_grad()
    def eval_epoch(self, loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_sharpe = []
        n = 0
        prev_w = None
        for feats, targets in loader:
            feats = feats.to(self.device)
            targets = targets.to(self.device)
            pred_w, Q, omega = self.model(feats)
            losses = self.loss_fn(pred_w, targets, feats[:, :, :self.mcfg.n_assets])
            total_loss += losses["loss"].item() * len(feats)
            all_sharpe.append(_sharpe(pred_w, feats[:, :, :self.mcfg.n_assets]))
            n += len(feats)
        return {"loss": total_loss / n, "sharpe": np.mean(all_sharpe)}

    def save_checkpoint(self, epoch: int, val_loss: float):
        path = Path(self.tcfg.checkpoint_dir) / "best_bl.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "model_config": self.mcfg,
            "history": self.history,
        }, path)

    def run(self) -> Dict[str, list]:
        print(f"[bl-train] Device: {self.device}  Params: {self.model.num_parameters:,}")
        for epoch in range(self.tcfg.epochs):
            self._warmup_lr(epoch)
            tr = self.train_epoch()
            vl = self.eval_epoch(self.val_dl)

            if epoch >= self.tcfg.warmup_epochs:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(tr["loss"])
            self.history["val_loss"].append(vl["loss"])
            self.history["val_sharpe"].append(vl["sharpe"])
            self.history["lr"].append(lr)

            if vl["loss"] < self.best_val_loss:
                self.best_val_loss = vl["loss"]
                self.save_checkpoint(epoch, vl["loss"])

            if epoch % 15 == 0 or epoch == self.tcfg.epochs - 1:
                print(
                    f"  Epoch {epoch:3d}/{self.tcfg.epochs} | "
                    f"train={tr['loss']:.5f} | "
                    f"val={vl['loss']:.5f} | sharpe={vl['sharpe']:.3f} | "
                    f"lr={lr:.2e}"
                )

        test = self.eval_epoch(self.test_dl)
        print(f"\n[bl-train] Test loss={test['loss']:.5f}  Sharpe={test['sharpe']:.3f}")
        return self.history
