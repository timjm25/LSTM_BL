"""Turn your latest market window into Black-Litterman weights with a trained checkpoint."""
import numpy as np
import torch

from .dataset import DataValidationError, load_bl_table
from .model import load_bl_checkpoint


def predict_bl_weights(checkpoint_path, data_path, prices: bool = False, device: str = "cpu"):
    """Weights, views (Q) and view uncertainty (omega) for the last ``sequence_length`` rows.

    The file must contain the same asset and macro columns the model was trained on
    (names are stored in checkpoints written by BLTrainer). Returns a DataFrame with
    one row per asset: asset, weight, view_Q, view_uncertainty.
    """
    import pandas as pd

    model, ckpt = load_bl_checkpoint(checkpoint_path, device=device)
    cfg = model.cfg
    assets = ckpt.get("asset_names")
    macros = ckpt.get("macro_names")
    if ckpt.get("data_source", "synthetic") == "synthetic":
        assets = macros = None  # names are placeholders; fall back to the asset_ prefix
    if macros == ["_no_macro"]:
        macros = []
    table = load_bl_table(
        data_path, asset_cols=assets, macro_cols=macros, prices=prices, min_rows=cfg.sequence_length,
    )
    if table.n_assets != cfg.n_assets or table.n_macro != cfg.macro_features:
        raise DataValidationError([
            f"The file has {table.n_assets} assets / {table.n_macro} macro features; the model "
            f"was trained on {cfg.n_assets} / {cfg.macro_features}."
        ])
    T = cfg.sequence_length
    x = np.concatenate([table.returns[-T:], table.macro[-T:]], axis=1)[None]
    x = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        weights, Q, omega = model(x)
    out = pd.DataFrame({
        "asset": table.asset_names,
        "weight": weights[0].cpu().numpy().astype(float),
        "view_Q": Q[0].cpu().numpy().astype(float),
        "view_uncertainty": omega[0].cpu().numpy().astype(float),
    })
    if table.dates is not None:
        out.attrs["as_of"] = pd.Timestamp(table.dates[-1]).date().isoformat()
    return out
