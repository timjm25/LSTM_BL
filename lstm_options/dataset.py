"""
Dataset generation for options pricing LSTM.

Two sources:
  1. SyntheticOptionsDataset — Monte-Carlo GBM paths labelled with BSM prices.
     Used when no real market data is available.
  2. CSVOptionsDataset — loads pre-downloaded CBOE/Bloomberg option chains.
     Expected columns: date, S, K, T, r, sigma, call_price (or put_price).
"""
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import OptionsModelConfig, OptionsTrainConfig
from .black_scholes import bsm_price, bsm_greeks


# ── Feature engineering ────────────────────────────────────────────────────────

def build_feature_vector(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> np.ndarray:
    """
    10-dim feature vector fed into LSTM at each timestep.

    Features
    --------
    [0]  log(S/K)       — log-moneyness
    [1]  T              — time to expiry (years)
    [2]  r              — risk-free rate
    [3]  sigma          — volatility
    [4]  S              — normalised spot price (divided by K)
    [5]  delta          — BSM delta
    [6]  gamma          — BSM gamma (scaled)
    [7]  vega           — BSM vega (scaled)
    [8]  theta          — BSM theta (scaled)
    [9]  moneyness_flag — +1 ITM, 0 ATM ±5%, -1 OTM
    """
    greeks = bsm_greeks(S, K, T, r, sigma)
    log_moneyness = np.log(S / K) if K > 0 else 0.0
    moneyness = S / K
    if moneyness > 1.05:
        flag = 1.0
    elif moneyness < 0.95:
        flag = -1.0
    else:
        flag = 0.0

    return np.array([
        log_moneyness,
        T,
        r,
        sigma,
        moneyness,
        greeks["delta"],
        greeks["gamma"] * 10,     # scale up tiny gamma values
        greeks["vega"],
        greeks["theta"] * 10,
        flag,
    ], dtype=np.float32)


# ── Synthetic dataset ──────────────────────────────────────────────────────────

class SyntheticOptionsDataset(Dataset):
    """
    Generates GBM (Geometric Brownian Motion) price paths for an underlying
    and labels each window with BSM call/put prices.

    The LSTM sees a rolling window of `sequence_length` daily feature vectors
    and predicts the option price at the end of the window.
    """

    def __init__(
        self,
        n_paths: int = 2000,
        path_len: int = 252,
        cfg: Optional[OptionsModelConfig] = None,
        option_type: str = "call",
        seed: int = 42,
    ):
        self.cfg = cfg or OptionsModelConfig()
        self.option_type = option_type
        T = self.cfg.sequence_length
        rng = np.random.default_rng(seed)

        self.samples: list = []  # (features (T,10), price_label)

        for _ in range(n_paths):
            # Random contract parameters
            S0 = rng.uniform(50, 500)
            K = S0 * rng.uniform(0.8, 1.2)
            total_T = rng.uniform(0.1, 2.0)    # years
            r = rng.uniform(0.01, 0.08)
            sigma = rng.uniform(0.10, 0.60)

            # Simulate GBM daily path
            dt = total_T / path_len
            log_returns = rng.normal(
                (r - 0.5 * sigma ** 2) * dt,
                sigma * np.sqrt(dt),
                path_len,
            )
            prices = S0 * np.exp(np.cumsum(log_returns))
            prices = np.concatenate([[S0], prices])  # length path_len+1

            for start in range(len(prices) - T):
                window_prices = prices[start: start + T + 1]
                t_remaining = max(total_T - start * dt, 1e-4)

                # Stochastic vol along path (GARCH-lite: rolling realised vol)
                if start >= 10:
                    realised = np.std(np.diff(np.log(prices[max(0, start-10):start+1])))
                    vol = realised * np.sqrt(252) if realised > 0 else sigma
                else:
                    vol = sigma

                feats = np.stack([
                    build_feature_vector(p, K, max(t_remaining - i * dt, 1e-4), r, vol)
                    for i, p in enumerate(window_prices[:T])
                ])  # (T, 10)

                label = bsm_price(
                    window_prices[T], K, max(t_remaining - T * dt, 1e-4),
                    r, vol, self.option_type,
                )
                self.samples.append((feats.astype(np.float32), np.float32(label)))

        # Global normalisation of labels (divide by ATM price for scale invariance)
        labels = np.array([s[1] for s in self.samples])
        self.label_mean = float(labels.mean())
        self.label_std = float(labels.std()) + 1e-8

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, label = self.samples[idx]
        feats_t = torch.tensor(feats, dtype=torch.float32)
        label_t = torch.tensor(
            (label - self.label_mean) / self.label_std, dtype=torch.float32
        )
        return feats_t, label_t


# ── CSV dataset (real market data) ────────────────────────────────────────────

class CSVOptionsDataset(Dataset):
    """
    Loads an option chain CSV with columns:
        date, S, K, T, r, sigma, price

    Builds rolling sequence windows for the LSTM.
    """

    def __init__(self, csv_path: str, cfg: Optional[OptionsModelConfig] = None):
        import pandas as pd
        self.cfg = cfg or OptionsModelConfig()
        df = pd.read_csv(csv_path)
        required = {"S", "K", "T", "r", "sigma", "price"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        T = self.cfg.sequence_length
        self.samples = []

        for i in range(len(df) - T):
            rows = df.iloc[i: i + T + 1]
            feats = np.stack([
                build_feature_vector(
                    row.S, row.K, row.T, row.r, row.sigma
                )
                for _, row in rows.iloc[:T].iterrows()
            ]).astype(np.float32)
            label = np.float32(rows.iloc[T]["price"])
            self.samples.append((feats, label))

        labels = np.array([s[1] for s in self.samples])
        self.label_mean = float(labels.mean())
        self.label_std = float(labels.std()) + 1e-8

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, label = self.samples[idx]
        return (
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor((label - self.label_mean) / self.label_std, dtype=torch.float32),
        )


# ── DataLoader factory ─────────────────────────────────────────────────────────

def make_options_dataloaders(
    train_cfg: OptionsTrainConfig,
    model_cfg: OptionsModelConfig,
    option_type: str = "call",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_path = Path(train_cfg.data_dir)
    csv_files = list(data_path.glob("*.csv")) if data_path.exists() else []

    if csv_files:
        print(f"[options] Loading real data from {csv_files[0]}")
        full_ds = CSVOptionsDataset(str(csv_files[0]), model_cfg)
    else:
        print("[options] No CSV found — using SyntheticOptionsDataset.")
        full_ds = SyntheticOptionsDataset(
            n_paths=1000, path_len=252, cfg=model_cfg,
            option_type=option_type,
        )

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
