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
from .black_scholes import bsm_price, bsm_greeks, implied_volatility


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
        # Synthetic labels *are* BSM prices, so the BSM baseline equals the label here.
        self.bsm_baselines = labels.astype(np.float32)
        self.return_bsm = False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        feats, label = self.samples[idx]
        feats_t = torch.tensor(feats, dtype=torch.float32)
        label_t = torch.tensor(
            (label - self.label_mean) / self.label_std, dtype=torch.float32
        )
        if self.return_bsm:
            bsm = (self.bsm_baselines[idx] - self.label_mean) / self.label_std
            return feats_t, label_t, torch.tensor(bsm, dtype=torch.float32)
        return feats_t, label_t


# ── Loading your own data ─────────────────────────────────────────────────────

class DataValidationError(ValueError):
    """Raised when an input file fails validation; ``errors`` lists each problem."""

    def __init__(self, errors):
        self.errors = list(errors)
        shown = "\n".join(f"  - {e}" for e in self.errors[:20])
        more = f"\n  ... and {len(self.errors) - 20} more" if len(self.errors) > 20 else ""
        super().__init__(f"Data file failed validation ({len(self.errors)} problem(s)):\n{shown}{more}")


OPTIONS_REQUIRED = ("S", "K", "T", "r", "price")
OPTIONS_OPTIONAL = ("date", "contract_id", "option_type", "sigma")

# Accepted alternative headers (matched case-insensitively, spaces/dashes = underscores).
OPTIONS_ALIASES = {
    "s": "S", "spot": "S", "underlying": "S", "underlying_price": "S", "stock_price": "S",
    "k": "K", "strike": "K", "strike_price": "K",
    "t": "T", "tte": "T", "time_to_expiry": "T", "expiry_years": "T", "years_to_expiry": "T",
    "rate": "r", "risk_free_rate": "r", "rf": "r",
    "vol": "sigma", "volatility": "sigma", "iv": "sigma", "implied_vol": "sigma",
    "implied_volatility": "sigma",
    "option_price": "price", "mid": "price", "mid_price": "price", "call_price": "price",
    "put_price": "price", "market_price": "price",
    "contract": "contract_id", "symbol": "contract_id", "option_symbol": "contract_id",
    "type": "option_type", "cp": "option_type", "call_put": "option_type",
}


def read_table(path):
    """Read a CSV, Excel (.xlsx) or JSON (list of records) file into a DataFrame."""
    import pandas as pd

    path = str(path)
    if not os.path.exists(path):
        raise DataValidationError([f"File not found: {path}"])
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".csv", ".txt"):
            return pd.read_csv(path)
        if ext in (".xlsx", ".xlsm"):
            return pd.read_excel(path)  # needs openpyxl
        if ext == ".json":
            return pd.read_json(path, orient="records")
    except ImportError as e:
        raise DataValidationError([f"Reading {ext} files needs an extra package: {e}"])
    except Exception as e:  # pandas raises many parser-specific exceptions
        raise DataValidationError([f"Could not read {path}: {e}"])
    raise DataValidationError([f"Unsupported file type '{ext}'. Use .csv, .xlsx or .json."])


def _canon(name, aliases):
    key = str(name).strip()
    low = key.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    return aliases.get(low, low)


def load_options_table(path_or_df, option_type: str = "call"):
    """Load and validate an options file.

    Columns (header names are case-insensitive; aliases in OPTIONS_ALIASES):
      required  S, K, T (years), r, price
      optional  sigma (annualised vol; implied from price when missing),
                date, contract_id, option_type (call/put)

    Rows are grouped by contract_id (one contract when absent) and sorted by date.
    Returns a cleaned DataFrame with canonical column names.
    """
    import pandas as pd

    df = path_or_df.copy() if isinstance(path_or_df, pd.DataFrame) else read_table(path_or_df)
    df = df.rename(columns={c: _canon(c, OPTIONS_ALIASES) for c in df.columns})
    errors = []
    dupes = sorted({c for c in df.columns if list(df.columns).count(c) > 1})
    if dupes:
        raise DataValidationError([f"More than one column maps to: {', '.join(dupes)}."])
    missing = [c for c in OPTIONS_REQUIRED if c not in df.columns]
    if missing:
        raise DataValidationError([
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(OPTIONS_REQUIRED)} (+ optional {', '.join(OPTIONS_OPTIONAL)})."
        ])
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        raise DataValidationError(["The file has no data rows."])

    numeric = list(OPTIONS_REQUIRED) + (["sigma"] if "sigma" in df.columns else [])
    for col in numeric:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = df.index[converted.isna() & df[col].notna()].tolist()
        blank = df.index[df[col].isna()].tolist()
        for i in bad[:5]:
            errors.append(f"Row {i + 2}: {col} '{df.at[i, col]}' is not a number.")
        if blank and col != "sigma":
            errors.append(f"Column {col} is blank on {len(blank)} row(s), first at row {blank[0] + 2}.")
        df[col] = converted
    if errors:
        raise DataValidationError(errors)

    checks = [
        ("S", df["S"] <= 0, "must be > 0"),
        ("K", df["K"] <= 0, "must be > 0"),
        ("T", df["T"] <= 0, "must be > 0 (years to expiry)"),
        ("price", df["price"] < 0, "must be >= 0"),
        ("r", (df["r"] < -0.1) | (df["r"] > 1.0), "should be a decimal rate like 0.05 (not 5)"),
    ]
    if "sigma" in df.columns:
        checks.append(("sigma", df["sigma"] <= 0, "must be > 0 (annualised, e.g. 0.25)"))
    for col, mask, msg in checks:
        rows = df.index[mask.fillna(False)].tolist()
        if rows:
            errors.append(f"{col} {msg}: {len(rows)} row(s), first at row {rows[0] + 2}.")
    if (df["T"] > 30).any():
        errors.append("T has values above 30 — T is in years; divide days by 365.")

    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].fillna(option_type).astype(str).str.strip().str.lower()
        df["option_type"] = df["option_type"].replace({"c": "call", "p": "put"})
        bad = df.index[~df["option_type"].isin(["call", "put"])].tolist()
        if bad:
            errors.append(f"option_type must be call/put (or C/P): row {bad[0] + 2} has '{df.at[bad[0], 'option_type']}'.")
    else:
        df["option_type"] = option_type

    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        bad = df.index[parsed.isna()].tolist()
        if bad:
            errors.append(f"date could not be read on {len(bad)} row(s), first at row {bad[0] + 2} ('{df.at[bad[0], 'date']}').")
        df["date"] = parsed
    if errors:
        raise DataValidationError(errors)

    if "sigma" not in df.columns:
        df["sigma"] = np.nan
    need_iv = df["sigma"].isna()
    if need_iv.any():
        df.loc[need_iv, "sigma"] = [
            implied_volatility(row.price, row.S, row.K, row.T, row.r, row.option_type)
            for row in df.loc[need_iv].itertuples()
        ]
        bad = df.index[~np.isfinite(df["sigma"]) | (df["sigma"] <= 1e-5) | (df["sigma"] > 5)].tolist()
        if bad:
            raise DataValidationError([
                f"sigma is blank and could not be implied from price on {len(bad)} row(s), "
                f"first at row {bad[0] + 2}; check price vs. intrinsic value or supply sigma."
            ])

    if "contract_id" not in df.columns:
        df["contract_id"] = "contract"
    df["contract_id"] = df["contract_id"].fillna("contract").astype(str)
    sort_cols = ["contract_id"] + (["date"] if "date" in df.columns else [])
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return df


class CSVOptionsDataset(Dataset):
    """
    Options history from your own file (CSV / .xlsx / .json) — see load_options_table
    for the column schema.

    For each contract, a window of ``sequence_length`` consecutive rows is the input and
    the market ``price`` on the following row is the label. Windows never span two
    contracts. ``bsm_baselines`` holds the textbook BSM price for each label row.
    """

    def __init__(self, csv_path, cfg: Optional[OptionsModelConfig] = None, option_type: str = "call"):
        self.cfg = cfg or OptionsModelConfig()
        df = load_options_table(csv_path, option_type=option_type)
        T = self.cfg.sequence_length

        self.samples = []
        baselines = []
        self.index_rows = []  # (contract_id, label row position in df)
        too_short = []
        for contract_id, grp in df.groupby("contract_id", sort=False):
            if len(grp) < T + 1:
                too_short.append(contract_id)
                continue
            feats_all = np.stack([
                build_feature_vector(r.S, r.K, r.T, r.r, r.sigma) for r in grp.itertuples()
            ]).astype(np.float32)
            prices = grp["price"].to_numpy(dtype=np.float32)
            rows = list(grp.itertuples())
            for i in range(len(grp) - T):
                tgt = rows[i + T]
                self.samples.append((feats_all[i: i + T], prices[i + T]))
                baselines.append(bsm_price(tgt.S, tgt.K, tgt.T, tgt.r, tgt.sigma, tgt.option_type))
                self.index_rows.append((contract_id, tgt.Index, getattr(tgt, "date", None)))
        if not self.samples:
            raise DataValidationError([
                f"No contract has at least {T + 1} rows (sequence_length {T} + 1 label row); "
                f"add more history or lower --seq_len. Contracts found: {len(too_short)}."
            ])
        self.table = df
        self.source = csv_path if isinstance(csv_path, (str, os.PathLike)) else "dataframe"
        self.bsm_baselines = np.asarray(baselines, dtype=np.float32)
        labels = np.array([s[1] for s in self.samples])
        self.label_mean = float(labels.mean())
        self.label_std = float(labels.std()) + 1e-8
        self.return_bsm = False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        feats, label = self.samples[idx]
        out = (
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor((label - self.label_mean) / self.label_std, dtype=torch.float32),
        )
        if self.return_bsm:
            bsm = (self.bsm_baselines[idx] - self.label_mean) / self.label_std
            out = out + (torch.tensor(bsm, dtype=torch.float32),)
        return out


def write_demo_options_csv(path, n_contracts: int = 4, days: int = 120, seed: int = 0):
    """Write a synthetic options history (GBM spot, BSM price + smile noise) for trying the CSV path."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_contracts):
        S = rng.uniform(80, 120)
        K = round(S * rng.uniform(0.9, 1.1))
        T0 = rng.uniform(0.6, 1.5)
        r = 0.04
        sigma = rng.uniform(0.18, 0.35)
        opt = "call" if c % 2 == 0 else "put"
        for d in range(days):
            T = max(T0 - d / 252, 1e-3)
            price = bsm_price(S, K, T, r, sigma, opt) * (1 + 0.03 * abs(np.log(S / K)) * 10) + rng.normal(0, 0.02)
            rows.append({
                "date": (pd.Timestamp("2025-01-02") + pd.tseries.offsets.BDay(d)).date().isoformat(),
                "contract_id": f"DEMO_{opt.upper()}_{K}",
                "option_type": opt, "S": round(S, 4), "K": K, "T": round(T, 6),
                "r": r, "sigma": round(sigma, 4), "price": round(max(price, 0.0), 4),
            })
            S *= float(np.exp(rng.normal(-0.5 * sigma ** 2 / 252, sigma / np.sqrt(252))))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ── DataLoader factory ─────────────────────────────────────────────────────────

def find_data_file(data_dir):
    """First .csv / .xlsx / .json file in ``data_dir`` (sorted), or None."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        return None
    files = sorted(p for p in data_path.iterdir() if p.suffix.lower() in (".csv", ".xlsx", ".json"))
    return files[0] if files else None


def _split(full_ds, train_cfg, chronological: bool):
    n = len(full_ds)
    n_test = max(1, int(n * train_cfg.test_split))
    n_val = max(1, int(n * train_cfg.val_split))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise DataValidationError([f"Only {n} training windows — not enough to split into train/val/test."])
    if chronological:
        # Real data: hold out the latest windows so overlapping windows don't leak into val/test.
        order = list(range(n))
        rows = getattr(full_ds, "index_rows", None)
        if rows:
            has_dates = all(r[2] is not None for r in rows)
            order.sort(key=lambda i: (rows[i][2], rows[i][1]) if has_dates else rows[i][1])
        return (
            torch.utils.data.Subset(full_ds, order[:n_train]),
            torch.utils.data.Subset(full_ds, order[n_train:n_train + n_val]),
            torch.utils.data.Subset(full_ds, order[n_train + n_val:]),
        )
    g = torch.Generator()
    g.manual_seed(train_cfg.seed)
    return torch.utils.data.random_split(full_ds, [n_train, n_val, n_test], generator=g)


def make_options_dataset(train_cfg: OptionsTrainConfig, model_cfg: OptionsModelConfig, option_type: str = "call"):
    """Resolve the training data: explicit data_file, else first file in data_dir, else synthetic."""
    data_file = getattr(train_cfg, "data_file", None)
    synthetic = getattr(train_cfg, "synthetic", False)
    if not synthetic and not data_file:
        data_file = find_data_file(train_cfg.data_dir)
    if data_file and not synthetic:
        print(f"[options] Loading your data from {data_file}")
        return CSVOptionsDataset(str(data_file), model_cfg, option_type=option_type), True
    if not synthetic:
        print(f"[options] No data file in {train_cfg.data_dir} — using synthetic demo data "
              f"(pass --data FILE to train on your own, or --synthetic to silence this).")
    ds = SyntheticOptionsDataset(
        n_paths=getattr(train_cfg, "synthetic_paths", 1000), path_len=252, cfg=model_cfg,
        option_type=option_type,
    )
    return ds, False


def make_options_dataloaders(
    train_cfg: OptionsTrainConfig,
    model_cfg: OptionsModelConfig,
    option_type: str = "call",
    dataset=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if dataset is None:
        dataset, is_real = make_options_dataset(train_cfg, model_cfg, option_type)
    else:
        is_real = isinstance(dataset, CSVOptionsDataset)
    dataset.return_bsm = bool(getattr(train_cfg, "bsm_residual", False))
    train_ds, val_ds, test_ds = _split(dataset, train_cfg, chronological=is_real)

    kw = dict(
        batch_size=train_cfg.batch_size,
        num_workers=min(train_cfg.num_workers, os.cpu_count() or 1),
        pin_memory=train_cfg.pin_memory and torch.cuda.is_available(),
    )
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )
