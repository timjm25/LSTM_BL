"""
LSTM Options Pricing Model.

Architecture
------------
Input sequence  (B, T, input_dim)
        │
        ▼
┌──────────────────┐
│  LSTM Stack       │   num_layers stacked LSTM cells with dropout
│  (3 layers)       │   Hidden state captures path-dependent dynamics
└──────┬───────────┘
       │ last hidden state (B, hidden_dim)
       ▼
┌──────────────────┐
│  Attention Gate   │   Soft attention over T timesteps — lets model
│                   │   weight recent vs historical price information
└──────┬───────────┘
       │ context vector (B, hidden_dim)
       ▼
┌──────────────────┐
│  MLP Decoder      │   LayerNorm + GELU + residual, outputs scalar price
└──────┬───────────┘
       │
       ▼
 Option price (B, 1)

Key design choices
------------------
- Attention over timesteps addresses the "which part of history matters?" problem
  that plain LSTM last-state misses for long maturities.
- BSM Greeks fed as features so the LSTM can learn residual corrections to BSM
  rather than learning the full pricing function from scratch.
- Residual connection in decoder stabilises training on normalised targets.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OptionsModelConfig


class TemporalAttention(nn.Module):
    """Additive attention over the time dimension of LSTM outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_out: (B, T, hidden_dim)
        Returns:
            context: (B, hidden_dim) — attention-weighted sum
        """
        weights = self.score(lstm_out)          # (B, T, 1)
        weights = F.softmax(weights, dim=1)     # normalise over T
        return (weights * lstm_out).sum(dim=1)  # (B, hidden_dim)


class OptionsLSTM(nn.Module):
    """
    LSTM-based option pricer.

    Can be used in two modes:
      1. BSM-residual mode  : predict the difference from the BSM price
         (bsm_residual=True and pass ``bsm_price`` to forward/predict; the output is
         bsm_price + learned correction, in the same normalised units as the label)
      2. Direct mode        : predict the full option price directly
         (bsm_residual=False, or call forward without ``bsm_price``)
    """

    def __init__(self, cfg: OptionsModelConfig, bsm_residual: bool = True):
        super().__init__()
        self.cfg = cfg
        self.bsm_residual = bsm_residual

        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.attention = TemporalAttention(cfg.hidden_dim)

        # MLP decoder with residual
        self.decoder = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, cfg.hidden_dim // 4),
            nn.GELU(),
        )
        self.output_layer = nn.Linear(cfg.hidden_dim // 4, cfg.output_dim)
        self.residual_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 4)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget gate bias → 1
                n = param.size(0) // 4
                param.data[n:2*n].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        bsm_price: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x         : (B, T, input_dim)
            hidden    : optional (h0, c0) for online inference
            bsm_price : optional (B,) BSM baseline (normalised like the label);
                        added to the output when the model is in residual mode

        Returns:
            price  : (B, output_dim)
            hidden : updated (hn, cn) for streaming
        """
        lstm_out, hidden_out = self.lstm(x, hidden)  # (B, T, hidden_dim)

        context = self.attention(lstm_out)           # (B, hidden_dim)

        # Residual MLP decoder
        h = self.decoder(context)                    # (B, hidden_dim//4)
        h = h + self.residual_proj(context)          # residual skip

        price = self.output_layer(h)                 # (B, output_dim)
        if self.bsm_residual and bsm_price is not None:
            price = price + bsm_price.reshape(-1, 1).to(price.dtype)
        return price, hidden_out

    def predict(self, x: torch.Tensor, bsm_price: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inference wrapper — returns (B,) price predictions."""
        with torch.no_grad():
            price, _ = self.forward(x, bsm_price=bsm_price)
        return price.squeeze(-1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_options_model(cfg: Optional[OptionsModelConfig] = None, bsm_residual: bool = True) -> OptionsLSTM:
    return OptionsLSTM(cfg or OptionsModelConfig(), bsm_residual=bsm_residual)


def load_options_checkpoint(path: str, device: str = "cpu"):
    """Load a checkpoint; the model comes back in eval mode (dropout off).

    Checkpoints written by OptionsTrainer also carry ``label_mean`` / ``label_std``
    (to turn normalised outputs back into prices) and ``bsm_residual``.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", OptionsModelConfig())
    model = build_options_model(cfg, bsm_residual=ckpt.get("bsm_residual", True))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
