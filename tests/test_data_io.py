"""Tests for loading your own data, BSM-residual mode, checkpoints metadata and prediction.

Everything here uses tiny files and models so the suite stays fast on CPU.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch

from lstm_bl.config import BLModelConfig, BLTrainConfig
from lstm_bl.dataset import (
    CSVBLDataset,
    DataValidationError as BLDataError,
    config_for_table,
    load_bl_table,
    make_bl_dataloaders,
    write_demo_bl_csv,
)
from lstm_bl.predict import predict_bl_weights
from lstm_bl.train import BLTrainer
from lstm_options.black_scholes import bsm_price
from lstm_options.config import OptionsModelConfig, OptionsTrainConfig
from lstm_options.dataset import (
    CSVOptionsDataset,
    DataValidationError as OptDataError,
    load_options_table,
    make_options_dataloaders,
    write_demo_options_csv,
)
from lstm_options.model import build_options_model, load_options_checkpoint
from lstm_options.predict import predict_options
from lstm_options.train import OptionsTrainer

ROOT = Path(__file__).parent.parent
TEMPLATES = ROOT / "data" / "templates"


def _opt_cfg(seq=5):
    return OptionsModelConfig(hidden_dim=8, num_layers=1, sequence_length=seq)


def _bl_cfg(seq=10):
    return BLModelConfig(hidden_dim=8, num_layers=1, sequence_length=seq)


# ── Options: loading & validation ─────────────────────────────────────────────

def test_options_template_loads_and_implies_missing_sigma():
    df = load_options_table(TEMPLATES / "options_template.csv")
    assert list(df["contract_id"].unique()) == ["SAMPLE_CALL_100", "SAMPLE_PUT_95"]
    assert df["sigma"].notna().all()  # the blank sigma was implied from price
    assert set(df["option_type"]) == {"call", "put"}


def test_options_aliases_and_cp_flags():
    raw = pd.DataFrame({
        "Spot": [100.0], "Strike": [100.0], "TTE": [0.5], "Rate": [0.05],
        "IV": [0.2], "Mid": [6.9], "Call/Put": ["P"],
    })
    df = load_options_table(raw)
    assert {"S", "K", "T", "r", "sigma", "price"} <= set(df.columns)
    assert df.at[0, "option_type"] == "put"


def test_implied_vol_matches_known_sigma():
    price = bsm_price(100, 105, 0.75, 0.03, 0.31, "call")
    df = load_options_table(pd.DataFrame({"S": [100], "K": [105], "T": [0.75], "r": [0.03], "price": [price]}))
    assert abs(df.at[0, "sigma"] - 0.31) < 1e-3


def test_options_missing_column_message():
    with pytest.raises(OptDataError) as e:
        load_options_table(pd.DataFrame({"S": [1], "K": [1], "T": [1], "r": [0.0]}))
    assert "price" in str(e.value)


def test_options_bad_values_reported():
    raw = pd.DataFrame({
        "S": [100, -5, "abc"], "K": [100, 100, 100], "T": [0.5, 0.5, 0.5],
        "r": [0.05, 0.05, 0.05], "sigma": [0.2, 0.2, 0.2], "price": [5, 5, 5],
    })
    with pytest.raises(OptDataError) as e:
        load_options_table(raw)
    assert "Row 4" in str(e.value) and "abc" in str(e.value)


@pytest.mark.parametrize("col,value,hint", [("T", 90, "years"), ("r", 5.0, "decimal"), ("sigma", 0.0, "> 0")])
def test_options_unit_hints(col, value, hint):
    raw = {"S": [100], "K": [100], "T": [0.5], "r": [0.05], "sigma": [0.2], "price": [5]}
    raw[col] = [value]
    with pytest.raises(OptDataError) as e:
        load_options_table(pd.DataFrame(raw))
    assert hint in str(e.value)


def test_options_windows_never_span_contracts(tmp_path):
    path = write_demo_options_csv(tmp_path / "o.csv", n_contracts=3, days=12)
    ds = CSVOptionsDataset(str(path), _opt_cfg(seq=5))
    assert len(ds) == 3 * (12 - 5)
    assert all(cid.startswith("DEMO_") for cid, _, _ in ds.index_rows)


def test_options_too_little_history(tmp_path):
    path = write_demo_options_csv(tmp_path / "o.csv", n_contracts=2, days=4)
    with pytest.raises(OptDataError) as e:
        CSVOptionsDataset(str(path), _opt_cfg(seq=5))
    assert "seq_len" in str(e.value)


def test_options_reads_xlsx_and_json(tmp_path):
    df = pd.read_csv(write_demo_options_csv(tmp_path / "o.csv", n_contracts=1, days=8))
    df.to_excel(tmp_path / "o.xlsx", index=False)
    (tmp_path / "o.json").write_text(json.dumps(df.to_dict(orient="records")))
    assert len(load_options_table(tmp_path / "o.xlsx")) == 8
    assert len(load_options_table(tmp_path / "o.json")) == 8
    with pytest.raises(OptDataError):
        load_options_table(tmp_path / "o.parquet")


# ── Options: BSM-residual mode ────────────────────────────────────────────────

def test_residual_mode_adds_bsm_baseline():
    model = build_options_model(_opt_cfg(), bsm_residual=True).eval()
    x = torch.randn(3, 5, 10)
    bsm = torch.tensor([0.5, -1.0, 2.0])
    with torch.no_grad():
        direct, _ = model(x)
        resid, _ = model(x, bsm_price=bsm)
    assert torch.allclose(resid.squeeze(-1) - direct.squeeze(-1), bsm, atol=1e-6)
    off = build_options_model(_opt_cfg(), bsm_residual=False).eval()
    with torch.no_grad():
        a, _ = off(x)
        b, _ = off(x, bsm_price=bsm)
    assert torch.allclose(a, b)


def test_dataloader_returns_bsm_when_residual(tmp_path):
    path = write_demo_options_csv(tmp_path / "o.csv", n_contracts=2, days=20)
    tcfg = OptionsTrainConfig(data_file=str(path), bsm_residual=True, batch_size=4, num_workers=0)
    train_dl, val_dl, _ = make_options_dataloaders(tcfg, _opt_cfg())
    batch = next(iter(train_dl))
    assert len(batch) == 3 and batch[2].shape == (4,)
    # Chronological split: every validation window ends after every training window.
    ds = train_dl.dataset.dataset
    train_dates = [ds.index_rows[i][2] for i in train_dl.dataset.indices]
    val_dates = [ds.index_rows[i][2] for i in val_dl.dataset.indices]
    assert max(train_dates) <= min(val_dates)


def test_options_train_checkpoint_and_predict(tmp_path):
    path = write_demo_options_csv(tmp_path / "o.csv", n_contracts=2, days=20)
    tcfg = OptionsTrainConfig(
        epochs=1, warmup_epochs=0, batch_size=8, num_workers=0, device="cpu",
        data_file=str(path), bsm_residual=True, checkpoint_dir=str(tmp_path / "ck"),
    )
    OptionsTrainer(tcfg, _opt_cfg()).run()
    ckpt_path = tmp_path / "ck" / "best_options.pt"
    model, ckpt = load_options_checkpoint(str(ckpt_path))
    assert not model.training and model.bsm_residual
    assert ckpt["label_std"] > 0 and ckpt["data_source"] == str(path)
    preds = predict_options(str(ckpt_path), str(path))
    assert list(preds.columns) == ["contract_id", "date", "option_type", "market_price", "bsm_price", "lstm_price"]
    assert len(preds) == 2 * (20 - 5)
    assert np.isfinite(preds["lstm_price"]).all()


# ── Black-Litterman: loading & validation ─────────────────────────────────────

def test_bl_template_loads_by_prefix():
    t = load_bl_table(TEMPLATES / "bl_template.csv")
    assert t.asset_names == ["asset_SAMPLE_A", "asset_SAMPLE_B", "asset_SAMPLE_C"]
    assert t.macro_names == ["vix", "yield_spread", "credit_spread"]
    assert t.returns.shape == (3, 3)


def test_bl_explicit_columns_and_sorting():
    raw = pd.DataFrame({
        "date": ["2025-01-03", "2025-01-02"], "SPY": [0.01, 0.02], "TLT": [-0.01, 0.0], "vix": [15, 16], "junk": [1, 2],
    })
    t = load_bl_table(raw, asset_cols=["SPY", "TLT"], macro_cols=["vix"])
    assert t.returns[0].tolist() == pytest.approx([0.02, 0.0])  # sorted by date
    assert t.macro_names == ["vix"]


def test_bl_prices_converted_and_price_like_returns_rejected():
    raw = pd.DataFrame({"asset_A": [100, 101, 99.99], "asset_B": [50, 50.5, 51], "vix": [15, 16, 17]})
    with pytest.raises(BLDataError) as e:
        load_bl_table(raw)
    assert "--prices" in str(e.value)
    t = load_bl_table(raw, prices=True)
    assert t.returns.shape == (2, 2)
    assert t.returns[0, 0] == pytest.approx(0.01)


def test_bl_validation_messages():
    with pytest.raises(BLDataError) as e:
        load_bl_table(pd.DataFrame({"SPY": [0.01], "vix": [15]}))
    assert "asset_" in str(e.value)
    with pytest.raises(BLDataError) as e:
        load_bl_table(pd.DataFrame({"asset_A": [0.01, "x"], "asset_B": [0.0, 0.0]}))
    assert "row 3" in str(e.value)
    with pytest.raises(BLDataError) as e:
        load_bl_table(pd.DataFrame({"date": ["2025-01-02"] * 2, "asset_A": [0, 0], "asset_B": [0, 0]}))
    assert "duplicates" in str(e.value)


def test_bl_missing_macro_ffilled_and_placeholder_when_none():
    raw = pd.DataFrame({"asset_A": [0.01, 0.02, 0.0], "asset_B": [0.0, 0.01, None], "vix": [15, None, 17]})
    t = load_bl_table(raw)
    assert t.macro[1, 0] == 15  # forward-filled
    assert len(t.returns) == 2 and any("Dropped 1" in n for n in t.notes)
    t2 = load_bl_table(raw.drop(columns=["vix"]).dropna())
    assert t2.macro_names == ["_no_macro"]


def test_bl_dataset_requires_matching_config(tmp_path):
    t = load_bl_table(write_demo_bl_csv(tmp_path / "b.csv", n_assets=3, days=60))
    with pytest.raises(BLDataError):
        CSVBLDataset(t, _bl_cfg())  # default config expects 10 assets
    cfg = config_for_table(t, _bl_cfg())
    assert (cfg.n_assets, cfg.macro_features, cfg.total_input_dim) == (3, 8, 11)
    ds = CSVBLDataset(t, cfg)
    feats, w = ds[0]
    assert feats.shape == (10, 11) and w.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_bl_too_few_rows(tmp_path):
    path = write_demo_bl_csv(tmp_path / "b.csv", n_assets=3, days=20)
    tcfg = BLTrainConfig(data_file=str(path), num_workers=0)
    with pytest.raises(BLDataError) as e:
        make_bl_dataloaders(tcfg, _bl_cfg())
    assert "forward days" in str(e.value)


def test_bl_train_checkpoint_and_predict(tmp_path):
    path = write_demo_bl_csv(tmp_path / "b.csv", n_assets=4, days=80)
    tcfg = BLTrainConfig(
        epochs=1, warmup_epochs=0, batch_size=8, num_workers=0, device="cpu",
        data_file=str(path), checkpoint_dir=str(tmp_path / "ck"),
    )
    trainer = BLTrainer(tcfg, _bl_cfg())
    assert trainer.mcfg.n_assets == 4  # config adapted to the file
    trainer.run()
    weights = predict_bl_weights(str(tmp_path / "ck" / "best_bl.pt"), str(path))
    assert list(weights["asset"]) == [f"asset_DEMO{i}" for i in range(4)]
    assert weights["weight"].sum() == pytest.approx(1.0, abs=1e-5)
    assert (weights["view_uncertainty"] > 0).all()
    assert weights.attrs["as_of"]


def test_bl_predict_rejects_wrong_columns(tmp_path):
    path = write_demo_bl_csv(tmp_path / "b.csv", n_assets=4, days=80)
    tcfg = BLTrainConfig(epochs=1, warmup_epochs=0, batch_size=8, num_workers=0, device="cpu",
                         data_file=str(path), checkpoint_dir=str(tmp_path / "ck"))
    BLTrainer(tcfg, _bl_cfg()).run()
    other = write_demo_bl_csv(tmp_path / "other.csv", n_assets=3, days=80)
    with pytest.raises(BLDataError):
        predict_bl_weights(str(tmp_path / "ck" / "best_bl.pt"), str(other))


# ── CLI entry points ───────────────────────────────────────────────────────────

def test_predict_cli_reports_bad_file(tmp_path, capsys):
    import predict

    bad = tmp_path / "bad.csv"
    bad.write_text("S,K\n1,2\n")
    ckpt = tmp_path / "ck.pt"
    model = build_options_model(_opt_cfg())
    torch.save({"model_state_dict": model.state_dict(), "model_config": _opt_cfg(),
                "label_mean": 1.0, "label_std": 1.0}, ckpt)
    assert predict.main(["options", "--checkpoint", str(ckpt), "--data", str(bad)]) == 2
    assert "Missing required column" in capsys.readouterr().err
    assert predict.main(["bl", "--checkpoint", str(tmp_path / "none.pt"), "--data", str(bad)]) == 1


def test_make_demo_data(tmp_path):
    import make_demo_data

    make_demo_data.main(["--out_dir", str(tmp_path)])
    assert len(load_options_table(tmp_path / "options_demo.csv")) > 0
    assert load_bl_table(tmp_path / "bl_demo.csv").n_assets == 5
