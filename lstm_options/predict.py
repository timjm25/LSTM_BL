"""Price your own options history with a trained checkpoint."""
import numpy as np
import torch

from .black_scholes import bsm_price  # noqa: F401  (re-exported for convenience)
from .dataset import CSVOptionsDataset, DataValidationError
from .model import load_options_checkpoint


def predict_options(checkpoint_path, data_path, option_type: str = None, device: str = "cpu"):
    """Return a DataFrame with one row per priceable row of ``data_path``.

    Each row needs ``sequence_length`` earlier rows of the same contract. Columns:
    contract_id, date (if given), option_type, market_price, bsm_price, lstm_price.
    """
    import pandas as pd

    model, ckpt = load_options_checkpoint(checkpoint_path, device=device)
    if "label_mean" not in ckpt or "label_std" not in ckpt:
        raise DataValidationError([
            "This checkpoint has no label_mean/label_std (it predates this version); retrain it "
            "with run_options.py so predictions can be converted back to prices."
        ])
    mean, std = float(ckpt["label_mean"]), float(ckpt["label_std"])
    ds = CSVOptionsDataset(data_path, model.cfg, option_type=option_type or ckpt.get("option_type", "call"))
    feats = torch.tensor(np.stack([s[0] for s in ds.samples]), dtype=torch.float32, device=device)
    bsm_norm = torch.tensor((ds.bsm_baselines - mean) / std, dtype=torch.float32, device=device)
    preds = []
    for i in range(0, len(feats), 256):
        bsm = bsm_norm[i:i + 256] if model.bsm_residual else None
        preds.append(model.predict(feats[i:i + 256], bsm_price=bsm).cpu().numpy())
    lstm_price = np.concatenate(preds) * std + mean

    table = ds.table
    rows = []
    for (contract_id, pos, date), market, bsm, pred in zip(
        ds.index_rows, (s[1] for s in ds.samples), ds.bsm_baselines, lstm_price
    ):
        row = {"contract_id": contract_id}
        if date is not None:
            row["date"] = pd.Timestamp(date).date().isoformat()
        row.update({
            "option_type": table.at[pos, "option_type"],
            "market_price": float(market),
            "bsm_price": float(bsm),
            "lstm_price": float(pred),
        })
        rows.append(row)
    return pd.DataFrame(rows)
