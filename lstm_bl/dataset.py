"""
Dataset for LSTM Black-Litterman model.

Each sample is a (sequence_length × total_input_dim) window of:
  - Asset daily returns   (n_assets columns)
  - Macro/sentiment features  (macro_features columns):
      [VIX, yield_spread, credit_spread, momentum_factor,
       value_factor, quality_factor, FX_vol, commodity_idx]

Target: the optimal BL portfolio weights for the NEXT period (n_assets,).
"""
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import BLModelConfig, BLTrainConfig
from .black_litterman import (
    black_litterman,
    mv_weights,
    covariance_from_returns,
)


class SyntheticBLDataset(Dataset):
    """
    Synthetic multi-asset return series with regime-switching.

    Two regimes are simulated:
      Bull: moderate positive drift, lower vol, positive correlation
      Bear: near-zero drift, higher vol, higher correlation (flight-to-quality)

    Macro features co-move with regime — VIX rises, yield spread widens in Bear.
    """

    def __init__(
        self,
        n_series: int = 500,
        series_len: int = 504,    # ~2 years of daily data
        cfg: Optional[BLModelConfig] = None,
        seed: int = 42,
    ):
        self.cfg = cfg or BLModelConfig()
        T = self.cfg.sequence_length
        rng = np.random.default_rng(seed)
        n = self.cfg.n_assets
        macro_dim = self.cfg.macro_features

        self.samples: list = []

        for _ in range(n_series):
            returns, macro = self._simulate_series(rng, series_len, n, macro_dim)

            for start in range(series_len - T - 1):
                ret_window = returns[start: start + T]     # (T, n)
                mac_window = macro[start: start + T]       # (T, macro_dim)
                feats = np.concatenate([ret_window, mac_window], axis=1).astype(np.float32)

                # Target: BL weights using the next 21-day forward realisation as "views"
                future_returns = returns[start + T: start + T + 21]
                if len(future_returns) < 5:
                    continue

                # Estimated covariance from window
                Sigma = covariance_from_returns(ret_window)
                w_mkt = np.ones(n) / n   # equal-weight market portfolio

                # LSTM "views": forward-looking expected returns (simulated analyst signal)
                # In production the LSTM replaces this with its own predictions
                Q = future_returns.mean(axis=0) * 252
                P = np.eye(n)
                Omega = np.diag(np.var(future_returns, axis=0) * 252 + 1e-6)

                try:
                    mu_bl, Sigma_bl = black_litterman(Sigma, w_mkt, P, Q, Omega, tau=self.cfg.tau)
                    w_opt = mv_weights(mu_bl, Sigma_bl)
                    if not np.all(np.isfinite(w_opt)):
                        continue
                    self.samples.append((feats, w_opt.astype(np.float32)))
                except np.linalg.LinAlgError:
                    continue

    @staticmethod
    def _simulate_series(rng, T: int, n: int, macro_dim: int):
        """Simulate regime-switching GBM returns and correlated macro features."""
        # Asset correlation structure
        base_corr = rng.uniform(0.2, 0.5, (n, n))
        base_corr = (base_corr + base_corr.T) / 2
        np.fill_diagonal(base_corr, 1.0)
        # Make PSD via clip-eigenvalue trick
        eigvals, eigvecs = np.linalg.eigh(base_corr)
        eigvals = np.clip(eigvals, 0.01, None)
        base_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T

        vols = rng.uniform(0.10, 0.35, n) / np.sqrt(252)
        cov_bull = np.outer(vols, vols) * base_corr
        cov_bear = np.outer(vols * 1.5, vols * 1.5) * (base_corr * 0.5 + 0.5)

        drift_bull = rng.uniform(0.05, 0.20, n) / 252
        drift_bear = rng.uniform(-0.10, 0.03, n) / 252

        returns = np.zeros((T, n), dtype=np.float32)
        macro = np.zeros((T, macro_dim), dtype=np.float32)
        regime = 1  # 1=bull, 0=bear
        regime_len = 0

        for t in range(T):
            # Regime transition
            if regime_len > 20:
                switch_prob = 0.02 if regime == 1 else 0.03
                if rng.random() < switch_prob:
                    regime = 1 - regime
                    regime_len = 0
            regime_len += 1

            if regime == 1:
                mu = drift_bull
                cov = cov_bull
                vix = rng.normal(15, 2)
                yld = rng.normal(1.0, 0.2)
            else:
                mu = drift_bear
                cov = cov_bear
                vix = rng.normal(28, 5)
                yld = rng.normal(2.0, 0.5)

            chol = np.linalg.cholesky(cov + 1e-8 * np.eye(n))
            returns[t] = mu + chol @ rng.standard_normal(n)

            # Macro features: VIX, yield_spread, credit_spread, factors, FX_vol, commodity...
            credit = max(0, rng.normal(1.5 if regime == 0 else 0.8, 0.3))
            n_factors = max(0, macro_dim - 5)
            factors = rng.normal(0.0 if regime == 0 else 0.02, 0.01, n_factors)
            full_macro = np.concatenate([
                [vix / 50, yld / 5, credit / 5],
                factors,
                [rng.normal(0.08 if regime == 0 else 0.05, 0.01)],  # FX vol
                [rng.normal(0.0, 0.02)],                              # commodity
            ])
            # Pad or truncate to exactly macro_dim
            if len(full_macro) < macro_dim:
                full_macro = np.pad(full_macro, (0, macro_dim - len(full_macro)))
            macro[t] = full_macro[:macro_dim]

        return returns, macro

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, weights = self.samples[idx]
        return (
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
        )


class CSVBLDataset(Dataset):
    """
    Loads a CSV of daily returns + macro features.
    Expected columns: date, asset_0, ..., asset_{n-1}, vix, yield_spread, ...
    """

    def __init__(self, csv_path: str, cfg: Optional[BLModelConfig] = None):
        import pandas as pd
        self.cfg = cfg or BLModelConfig()
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        n = self.cfg.n_assets
        T = self.cfg.sequence_length

        asset_cols = [c for c in df.columns if c.startswith("asset_")][:n]
        macro_cols = [c for c in df.columns if not c.startswith("asset_")][:self.cfg.macro_features]

        returns = df[asset_cols].values.astype(np.float32)
        macro = df[macro_cols].values.astype(np.float32)

        self.samples = []
        for start in range(len(df) - T - 21):
            ret_w = returns[start: start + T]
            mac_w = macro[start: start + T]
            feats = np.concatenate([ret_w, mac_w], axis=1)

            future = returns[start + T: start + T + 21]
            Sigma = covariance_from_returns(ret_w)
            w_mkt = np.ones(n) / n
            Q = future.mean(axis=0) * 252
            P = np.eye(n)
            Omega = np.diag(np.var(future, axis=0) * 252 + 1e-6)

            try:
                mu_bl, Sigma_bl = black_litterman(Sigma, w_mkt, P, Q, Omega, tau=self.cfg.tau)
                w_opt = mv_weights(mu_bl, Sigma_bl)
                if np.all(np.isfinite(w_opt)):
                    self.samples.append((feats.astype(np.float32), w_opt.astype(np.float32)))
            except np.linalg.LinAlgError:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, weights = self.samples[idx]
        return torch.tensor(feats), torch.tensor(weights)


def make_bl_dataloaders(
    train_cfg: BLTrainConfig,
    model_cfg: BLModelConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_path = Path(train_cfg.data_dir)
    csv_files = list(data_path.glob("*.csv")) if data_path.exists() else []

    if csv_files:
        print(f"[bl] Loading real data from {csv_files[0]}")
        full_ds = CSVBLDataset(str(csv_files[0]), model_cfg)
    else:
        print("[bl] No data found — using SyntheticBLDataset.")
        full_ds = SyntheticBLDataset(n_series=200, cfg=model_cfg)

    n = len(full_ds)
    n_test = max(1, int(n * train_cfg.test_split))
    n_val = max(1, int(n * train_cfg.val_split))
    n_train = n - n_val - n_test

    g = torch.Generator()
    g.manual_seed(train_cfg.seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val, n_test], generator=g
    )

    kw = dict(
        batch_size=train_cfg.batch_size,
        num_workers=min(train_cfg.num_workers, os.cpu_count() or 1),
        pin_memory=train_cfg.pin_memory,
    )
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )
