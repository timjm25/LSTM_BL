"""
LSTM Black-Litterman Portfolio Model.

Architecture
------------
Input: (B, T, n_assets + macro_features)
         daily returns  +  macro/sentiment signals
         │
         ▼
┌────────────────────────────────────┐
│  Dual-stream LSTM Encoder           │
│  ├─ Asset Stream (return dynamics)  │   Separate LSTMs for assets vs macro
│  └─ Macro Stream (regime signals)   │   allow independent learning rates
└────────────┬───────────────────────┘
             │ concatenated hidden states
             ▼
┌────────────────────────────────────┐
│  Cross-Stream Attention             │   Macro hidden state attends over
│                                     │   asset hidden states — learns which
│                                     │   assets respond most to each regime
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  View Generation Head               │   Predicts Q: expected excess returns
│  (BL analyst views)                 │   per asset (k = n_assets views)
└────────────┬───────────────────────┘
             │  Q (n_assets,)
             ▼
┌────────────────────────────────────┐
│  Uncertainty Head                   │   Predicts diagonal of Ω (view errors)
│  (BL confidence matrix)             │   softplus-activated to stay positive
└────────────┬───────────────────────┘
             │  Omega diag (n_assets,)
             ▼
        Black-Litterman Solver (differentiable)
             │
             ▼
┌────────────────────────────────────┐
│  Portfolio Weight Head              │   Applies softmax for long-only
│  (Softmax + risk aversion scaling)  │   simplex constraint ∑w=1
└────────────────────────────────────┘
             │
             ▼
    Portfolio weights w* (B, n_assets)
"""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BLModelConfig
from .black_litterman import black_litterman, mv_weights, covariance_from_returns


class CrossStreamAttention(nn.Module):
    """
    Macro stream attends over asset-stream hidden states.

    query  = macro hidden state  (B, macro_hidden)
    keys   = asset hidden state  (B, n_assets, asset_hidden)
    output = (B, asset_hidden)
    """

    def __init__(self, query_dim: int, key_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, key_dim)
        self.scale = key_dim ** -0.5

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(query).unsqueeze(1)         # (B, 1, key_dim)
        scores = (q * keys).sum(-1, keepdim=True) * self.scale  # (B, n, 1)
        weights = F.softmax(scores, dim=1)
        return (weights * keys).sum(dim=1)           # (B, key_dim)


class DifferentiableBLSolver(nn.Module):
    """
    Differentiable approximation of the BL weight computation.

    Rather than calling numpy linalg inside the forward pass (not
    differentiable), we implement BL as a matrix operation in PyTorch.

    For simplicity we use diagonal P = I (one view per asset) and
    compute the posterior analytically in closed form.
    """

    def __init__(self, n_assets: int, tau: float = 0.05, delta: float = 2.5):
        super().__init__()
        self.n = n_assets
        self.tau = tau
        self.delta = delta
        # Learnable risk aversion (initialised to delta)
        self.log_delta = nn.Parameter(torch.tensor(float(np.log(delta))))

    def forward(
        self,
        Q: torch.Tensor,
        omega_diag: torch.Tensor,
        Sigma: torch.Tensor,
        w_mkt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            Q          : (B, n)  LSTM view predictions (expected excess returns)
            omega_diag : (B, n)  positive uncertainty diagonal
            Sigma      : (B, n, n) asset covariance
            w_mkt      : (B, n) market weights

        Returns:
            weights : (B, n) softmax portfolio weights
        """
        B, n = Q.shape
        delta = torch.exp(self.log_delta)

        # Equilibrium returns  Π = δ Σ w_mkt
        Pi = delta * torch.bmm(Sigma, w_mkt.unsqueeze(-1)).squeeze(-1)  # (B, n)

        # τΣ  and its inverse
        tauSigma = self.tau * Sigma
        # Regularise for numerical stability
        tauSigma = tauSigma + 1e-6 * torch.eye(n, device=Q.device).unsqueeze(0)
        tauSigma_inv = torch.linalg.inv(tauSigma)

        # Ω⁻¹ = diag(1/omega_diag)
        Omega_inv_diag = 1.0 / (omega_diag + 1e-8)  # (B, n)
        # P = I, so Pᵀ Ω⁻¹ P = diag(omega_inv_diag)
        PtOmegaInvP = torch.diag_embed(Omega_inv_diag)  # (B, n, n)

        # M = (τΣ)⁻¹ + Pᵀ Ω⁻¹ P
        M = tauSigma_inv + PtOmegaInvP  # (B, n, n)
        M = M + 1e-6 * torch.eye(n, device=Q.device).unsqueeze(0)
        M_inv = torch.linalg.inv(M)

        # μ_BL = M⁻¹ [(τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q]
        rhs = (
            torch.bmm(tauSigma_inv, Pi.unsqueeze(-1)).squeeze(-1)   # (B, n)
            + Omega_inv_diag * Q                                      # (B, n)
        )
        mu_bl = torch.bmm(M_inv, rhs.unsqueeze(-1)).squeeze(-1)     # (B, n)

        # Σ_BL = Σ + M⁻¹
        Sigma_bl = Sigma + M_inv

        # w* = (δ Σ_BL)⁻¹ μ_BL
        lhs = delta * Sigma_bl
        lhs = lhs + 1e-6 * torch.eye(n, device=Q.device).unsqueeze(0)
        w_raw = torch.linalg.solve(lhs, mu_bl.unsqueeze(-1)).squeeze(-1)  # (B, n)

        # Long-only softmax constraint
        w_opt = F.softmax(w_raw, dim=-1)
        return w_opt


class BLPortfolioLSTM(nn.Module):
    """
    Full LSTM Black-Litterman portfolio model.

    The model learns to generate analyst-quality views (Q) and confidence
    estimates (Ω) from recent price/macro history, then passes these through
    the analytical BL solver to produce portfolio weights.
    """

    def __init__(self, cfg: BLModelConfig):
        super().__init__()
        self.cfg = cfg
        n = cfg.n_assets
        macro_dim = cfg.macro_features
        h = cfg.hidden_dim

        # Asset return encoder
        self.asset_lstm = nn.LSTM(
            input_size=n,
            hidden_size=h,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        # Macro / regime encoder
        self.macro_lstm = nn.LSTM(
            input_size=macro_dim,
            hidden_size=h // 2,
            num_layers=max(1, cfg.num_layers - 1),
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 2 else 0.0,
        )

        # Cross-stream attention: macro queries asset
        self.cross_attn = CrossStreamAttention(query_dim=h // 2, key_dim=h)

        # Fusion
        fused_dim = h + h   # asset final state + cross-attn output
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # View generation head — predicts Q  (per-asset expected returns)
        self.view_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, n),
        )

        # Uncertainty head — predicts Ω diag  (positive via softplus)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, n),
            nn.Softplus(),    # guarantees Ω > 0
        )

        # Differentiable BL solver
        self.bl_solver = DifferentiableBLSolver(n_assets=n, tau=cfg.tau)

        # Learnable covariance estimator weights (exponential smoothing)
        self.ew_logit = nn.Parameter(torch.zeros(1))  # sigmoid → smoothing factor

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name and "lstm" in name:
                nn.init.zeros_(param)
                n = param.size(0) // 4
                param.data[n:2*n].fill_(1.0)

    def _estimate_covariance(self, returns: torch.Tensor) -> torch.Tensor:
        """Exponentially weighted covariance from return window.

        Args:
            returns: (B, T, n_assets)
        Returns:
            Sigma: (B, n, n)
        """
        B, T, n = returns.shape
        alpha = torch.sigmoid(self.ew_logit)  # smoothing factor
        weights = alpha * (1 - alpha) ** torch.arange(T - 1, -1, -1, device=returns.device, dtype=returns.dtype)
        weights = weights / weights.sum()       # (T,)
        weights = weights.view(1, T, 1)

        # Weighted mean
        mu = (weights * returns).sum(dim=1, keepdim=True)  # (B, 1, n)
        centered = returns - mu                             # (B, T, n)
        # Weighted covariance
        w_sqrt = weights.sqrt()
        centered_w = w_sqrt * centered                      # (B, T, n)
        Sigma = torch.bmm(centered_w.transpose(1, 2), centered_w)  # (B, n, n)
        # Diagonal regularisation
        Sigma = Sigma + 1e-5 * torch.eye(n, device=returns.device).unsqueeze(0)
        return Sigma

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, n_assets + macro_features)

        Returns:
            weights  : (B, n_assets) — BL portfolio weights
            Q        : (B, n_assets) — predicted views
            omega    : (B, n_assets) — predicted uncertainty diagonal
        """
        B, T, _ = x.shape
        n = self.cfg.n_assets
        macro_dim = self.cfg.macro_features

        asset_in = x[:, :, :n]                   # (B, T, n)
        macro_in = x[:, :, n:]                   # (B, T, macro_dim)

        # Encode asset returns
        asset_out, (asset_hn, _) = self.asset_lstm(asset_in)
        asset_h = asset_hn[-1]                    # (B, h) — last layer hidden

        # Encode macro features
        _, (macro_hn, _) = self.macro_lstm(macro_in)
        macro_h = macro_hn[-1]                    # (B, h//2)

        # Cross-stream attention: macro queries over asset timestep outputs
        # Reshape asset_out to (B, T, h) for attention
        cross_context = self.cross_attn(macro_h, asset_out)  # (B, h)

        # Fuse
        fused = self.fusion(torch.cat([asset_h, cross_context], dim=-1))  # (B, h)

        # BL view predictions
        Q = self.view_head(fused)                  # (B, n)
        omega_diag = self.uncertainty_head(fused) + 1e-6  # (B, n), positive

        # Estimate covariance from asset return window
        Sigma = self._estimate_covariance(asset_in)  # (B, n, n)

        # Equal-weight market portfolio
        w_mkt = torch.ones(B, n, device=x.device) / n

        # BL solver → portfolio weights
        weights = self.bl_solver(Q, omega_diag, Sigma, w_mkt)  # (B, n)

        return weights, Q, omega_diag

    @torch.no_grad()
    def predict_weights(self, x: torch.Tensor) -> np.ndarray:
        """Inference wrapper. Returns numpy portfolio weights."""
        self.eval()
        weights, _, _ = self.forward(x)
        # Use list-conversion to avoid NumPy 2.x / PyTorch ABI incompatibility
        return np.array(weights.cpu().detach().tolist(), dtype=np.float32)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_bl_model(cfg: Optional[BLModelConfig] = None) -> BLPortfolioLSTM:
    return BLPortfolioLSTM(cfg or BLModelConfig())


def load_bl_checkpoint(path: str, device: str = "cpu"):
    """Load a checkpoint; the model comes back in eval mode (dropout off).

    Checkpoints written by BLTrainer also carry ``asset_names`` / ``macro_names``
    (the column order the model expects).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", BLModelConfig())
    model = build_bl_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
