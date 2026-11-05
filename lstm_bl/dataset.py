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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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


class DataValidationError(ValueError):
    """Raised when an input file fails validation; ``errors`` lists each problem."""

    def __init__(self, errors):
        self.errors = list(errors)
        shown = "\n".join(f"  - {e}" for e in self.errors[:20])
        more = f"\n  ... and {len(self.errors) - 20} more" if len(self.errors) > 20 else ""
        super().__init__(f"Data file failed validation ({len(self.errors)} problem(s)):\n{shown}{more}")


FORWARD_DAYS = 21   # horizon of the forward returns used to build the BL target weights


@dataclass
class BLTable:
    """Validated portfolio history: aligned daily asset returns and macro features."""
    returns: np.ndarray          # (T, n_assets) daily simple returns
    macro: np.ndarray            # (T, n_macro)
    asset_names: List[str]
    macro_names: List[str]
    dates: Optional[list] = None
    notes: List[str] = field(default_factory=list)

    @property
    def n_assets(self) -> int:
        return self.returns.shape[1]

    @property
    def n_macro(self) -> int:
        return self.macro.shape[1]


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


def load_bl_table(
    path_or_df,
    asset_cols: Optional[Sequence[str]] = None,
    macro_cols: Optional[Sequence[str]] = None,
    asset_prefix: str = "asset_",
    prices: bool = False,
    min_rows: int = 0,
) -> BLTable:
    """Load and validate a wide portfolio file (one row per day).

    Columns:
      date        optional; parsed and used to sort rows
      assets      ``asset_cols`` if given, else every column starting with ``asset_prefix``
                  — daily returns as decimals (0.01 = 1%), or prices with ``prices=True``
      macro       ``macro_cols`` if given, else every other numeric column
                  (e.g. vix, yield_spread, credit_spread, ...)

    Missing macro values are forward-filled; rows still incomplete are dropped and noted.
    """
    import pandas as pd

    df = path_or_df.copy() if isinstance(path_or_df, pd.DataFrame) else read_table(path_or_df)
    df.columns = [str(c).strip() for c in df.columns]
    errors, notes = [], []
    lower = {c.lower(): c for c in df.columns}
    date_col = lower.get("date")
    df = df.dropna(how="all")

    if asset_cols:
        missing = [c for c in asset_cols if c not in df.columns]
        if missing:
            raise DataValidationError([f"Asset column(s) not in file: {', '.join(missing)}."])
        assets = list(asset_cols)
    else:
        assets = [c for c in df.columns if c.lower().startswith(asset_prefix.lower())]
        if not assets:
            raise DataValidationError([
                f"No asset columns found. Name them with the prefix '{asset_prefix}' "
                f"(e.g. {asset_prefix}SPY) or pass asset_cols / --assets."
            ])
    if len(assets) < 2:
        errors.append("At least 2 asset columns are needed to build a portfolio.")

    if macro_cols is not None:
        missing = [c for c in macro_cols if c not in df.columns]
        if missing:
            raise DataValidationError([f"Macro column(s) not in file: {', '.join(missing)}."])
        macros = list(macro_cols)
    else:
        macros = [c for c in df.columns if c not in assets and c != date_col]

    for col in assets + macros:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = df.index[converted.isna() & df[col].notna()].tolist()
        if bad:
            pos = df.index.get_loc(bad[0]) + 2
            errors.append(f"Column {col}: '{df.at[bad[0], col]}' on row {pos} is not a number.")
        df[col] = converted
    if errors:
        raise DataValidationError(errors)

    dates = None
    if date_col:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        bad = df.index[parsed.isna()].tolist()
        if bad:
            raise DataValidationError([
                f"date could not be read on {len(bad)} row(s), first at row {df.index.get_loc(bad[0]) + 2}."
            ])
        df[date_col] = parsed
        if parsed.duplicated().any():
            raise DataValidationError([f"date has duplicates, e.g. {parsed[parsed.duplicated()].iloc[0].date()}."])
        df = df.sort_values(date_col, kind="stable")

    if prices:
        px = df[assets]
        if (px <= 0).any().any():
            raise DataValidationError(["Prices must be > 0 to convert to returns."])
        df[assets] = px.pct_change()
        df = df.iloc[1:]
        notes.append("Converted asset prices to daily simple returns (first row dropped).")

    if macros:
        df[macros] = df[macros].ffill()
    before = len(df)
    df = df.dropna(subset=assets + macros)
    if len(df) < before:
        notes.append(f"Dropped {before - len(df)} row(s) with missing asset returns or macro values.")

    big = [c for c in assets if (df[c].abs() > 1.0).any()]
    if big:
        raise DataValidationError([
            f"Asset column(s) {', '.join(big[:5])} have daily moves above 100% — these look like "
            f"prices, not returns. Pass prices=True (CLI: --prices), or give returns as decimals."
        ])
    if len(df) < max(min_rows, 2):
        raise DataValidationError([
            f"Only {len(df)} usable rows; need at least {max(min_rows, 2)} "
            f"(sequence_length + {FORWARD_DAYS} forward days + 1)."
        ])

    returns = df[assets].to_numpy(dtype=np.float32)
    if macros:
        macro = df[macros].to_numpy(dtype=np.float32)
    else:
        macro = np.zeros((len(df), 1), dtype=np.float32)
        macros = ["_no_macro"]
        notes.append("No macro columns found — using a single zero placeholder feature.")
    return BLTable(
        returns=returns, macro=macro, asset_names=assets, macro_names=macros,
        dates=list(df[date_col]) if date_col else None, notes=notes,
    )


def config_for_table(table: BLTable, cfg: Optional[BLModelConfig] = None) -> BLModelConfig:
    """Copy of ``cfg`` with n_assets / macro_features / total_input_dim set from the data."""
    cfg = cfg or BLModelConfig()
    return replace(
        cfg, n_assets=table.n_assets, macro_features=table.n_macro,
        total_input_dim=table.n_assets + table.n_macro,
    )


def bl_target_weights(window_returns: np.ndarray, future_returns: np.ndarray, tau: float) -> Optional[np.ndarray]:
    """BL target weights for one window, using realised forward returns as the views."""
    n = window_returns.shape[1]
    Sigma = covariance_from_returns(window_returns)
    w_mkt = np.ones(n) / n
    Q = future_returns.mean(axis=0) * 252
    P = np.eye(n)
    Omega = np.diag(np.var(future_returns, axis=0) * 252 + 1e-6)
    try:
        mu_bl, Sigma_bl = black_litterman(Sigma, w_mkt, P, Q, Omega, tau=tau)
        w_opt = mv_weights(mu_bl, Sigma_bl)
    except np.linalg.LinAlgError:
        return None
    return w_opt if np.all(np.isfinite(w_opt)) else None


class CSVBLDataset(Dataset):
    """
    Portfolio history from your own file (CSV / .xlsx / .json) or a BLTable — see
    load_bl_table for the column schema. Each sample is a ``sequence_length`` window of
    [asset returns | macro features]; the target is the BL weights implied by the next
    21 days. ``cfg.n_assets`` / ``cfg.macro_features`` must match the data — use
    ``config_for_table`` to build a matching config.
    """

    def __init__(self, csv_path, cfg: Optional[BLModelConfig] = None, **load_kwargs):
        self.cfg = cfg or BLModelConfig()
        T = self.cfg.sequence_length
        if isinstance(csv_path, BLTable):
            table = csv_path
        else:
            table = load_bl_table(csv_path, min_rows=T + FORWARD_DAYS + 1, **load_kwargs)
        if table.n_assets != self.cfg.n_assets or table.n_macro != self.cfg.macro_features:
            raise DataValidationError([
                f"The data has {table.n_assets} assets and {table.n_macro} macro features but the "
                f"model config expects {self.cfg.n_assets} and {self.cfg.macro_features}; "
                f"build the config with config_for_table()."
            ])
        if len(table.returns) < T + FORWARD_DAYS + 1:
            raise DataValidationError([
                f"Only {len(table.returns)} rows; need at least {T + FORWARD_DAYS + 1} "
                f"(sequence_length {T} + {FORWARD_DAYS} forward days + 1)."
            ])
        self.table = table
        self.source = csv_path if isinstance(csv_path, (str, os.PathLike)) else "table"
        returns, macro = table.returns, table.macro

        self.samples = []
        self.window_ends = []
        for start in range(len(returns) - T - FORWARD_DAYS):
            ret_w = returns[start: start + T]
            feats = np.concatenate([ret_w, macro[start: start + T]], axis=1)
            w_opt = bl_target_weights(ret_w, returns[start + T: start + T + FORWARD_DAYS], self.cfg.tau)
            if w_opt is not None:
                self.samples.append((feats.astype(np.float32), w_opt.astype(np.float32)))
                self.window_ends.append(start + T)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, weights = self.samples[idx]
        return torch.tensor(feats), torch.tensor(weights)


def write_demo_bl_csv(path, n_assets: int = 5, days: int = 400, seed: int = 0):
    """Write a synthetic regime-switching portfolio history in the CSV schema, for trying the file path."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    returns, macro = SyntheticBLDataset._simulate_series(rng, days, n_assets, 8)
    names = ["vix", "yield_spread", "credit_spread", "momentum", "value", "quality", "fx_vol", "commodity"]
    df = pd.DataFrame(returns, columns=[f"asset_DEMO{i}" for i in range(n_assets)])
    for j, name in enumerate(names):
        df[name] = macro[:, j]
    df.insert(0, "date", pd.bdate_range("2024-01-02", periods=days).date)
    df.round(6).to_csv(path, index=False)
    return path


def find_data_file(data_dir):
    """First .csv / .xlsx / .json file in ``data_dir`` (sorted), or None."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        return None
    files = sorted(p for p in data_path.iterdir() if p.suffix.lower() in (".csv", ".xlsx", ".json"))
    return files[0] if files else None


def resolve_bl_data(train_cfg: BLTrainConfig, model_cfg: BLModelConfig):
    """Pick the data source and a model config that matches it.

    Returns (table_or_None, model_cfg). ``None`` means synthetic data.
    """
    data_file = getattr(train_cfg, "data_file", None)
    if getattr(train_cfg, "synthetic", False):
        return None, model_cfg
    if not data_file:
        data_file = find_data_file(train_cfg.data_dir)
    if not data_file:
        print(f"[bl] No data file in {train_cfg.data_dir} — using synthetic demo data "
              f"(pass --data FILE to train on your own, or --synthetic to silence this).")
        return None, model_cfg
    print(f"[bl] Loading your data from {data_file}")
    table = load_bl_table(
        data_file,
        asset_cols=getattr(train_cfg, "asset_cols", None),
        macro_cols=getattr(train_cfg, "macro_cols", None),
        prices=getattr(train_cfg, "prices", False),
        min_rows=model_cfg.sequence_length + FORWARD_DAYS + 1,
    )
    table.source = str(data_file)
    for note in table.notes:
        print(f"[bl] {note}")
    return table, config_for_table(table, model_cfg)


def make_bl_dataloaders(
    train_cfg: BLTrainConfig,
    model_cfg: BLModelConfig,
    table: Optional[BLTable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build loaders. With ``table`` (or a data file), uses your data and a chronological split."""
    if table is None:
        table, model_cfg = resolve_bl_data(train_cfg, model_cfg)
    if table is not None:
        full_ds = CSVBLDataset(table, model_cfg)
        full_ds.source = getattr(table, "source", "table")
    else:
        full_ds = SyntheticBLDataset(n_series=getattr(train_cfg, "synthetic_series", 200), cfg=model_cfg)

    n = len(full_ds)
    n_test = max(1, int(n * train_cfg.test_split))
    n_val = max(1, int(n * train_cfg.val_split))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise DataValidationError([f"Only {n} training windows — not enough to split into train/val/test."])

    if table is not None:
        # Real data: hold out the latest windows so overlapping windows don't leak.
        idx = list(range(n))
        train_ds = torch.utils.data.Subset(full_ds, idx[:n_train])
        val_ds = torch.utils.data.Subset(full_ds, idx[n_train:n_train + n_val])
        test_ds = torch.utils.data.Subset(full_ds, idx[n_train + n_val:])
    else:
        g = torch.Generator()
        g.manual_seed(train_cfg.seed)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val, n_test], generator=g
        )

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
