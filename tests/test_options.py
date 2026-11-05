"""Unit tests for LSTM Options Pricing model."""
import sys
import math
import tempfile
import pathlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from lstm_options.config import OptionsModelConfig, OptionsTrainConfig
from lstm_options.black_scholes import bsm_price, bsm_greeks, implied_volatility
from lstm_options.dataset import SyntheticOptionsDataset, build_feature_vector
from lstm_options.model import OptionsLSTM, build_options_model, load_options_checkpoint
from lstm_options.train import OptionsPricingLoss


# ── BSM tests ──────────────────────────────────────────────────────────────────

def test_bsm_call_put_parity():
    """Put-call parity: C - P = S - K*e^(-rT)"""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    call = bsm_price(S, K, T, r, sigma, "call")
    put = bsm_price(S, K, T, r, sigma, "put")
    lhs = call - put
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-6, f"Put-call parity violated: {lhs:.6f} != {rhs:.6f}"


def test_bsm_call_positive():
    call = bsm_price(100, 100, 1.0, 0.05, 0.20)
    assert call > 0


def test_bsm_put_positive():
    put = bsm_price(100, 110, 1.0, 0.05, 0.20, "put")
    assert put > 0


def test_bsm_greeks_delta_bounds():
    greeks = bsm_greeks(100, 100, 1.0, 0.05, 0.20, "call")
    assert 0 <= greeks["delta"] <= 1.0


def test_bsm_greeks_gamma_positive():
    greeks = bsm_greeks(100, 100, 1.0, 0.05, 0.20, "call")
    assert greeks["gamma"] > 0


def test_implied_vol_roundtrip():
    """IV solve should recover the original volatility."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.25
    price = bsm_price(S, K, T, r, sigma)
    iv = implied_volatility(price, S, K, T, r)
    assert abs(iv - sigma) < 1e-4, f"IV mismatch: got {iv:.6f}, expected {sigma}"


# ── Feature engineering tests ──────────────────────────────────────────────────

def test_feature_vector_shape():
    feat = build_feature_vector(100, 100, 1.0, 0.05, 0.20)
    assert feat.shape == (10,)
    assert feat.dtype == np.float32


def test_feature_itm_flag():
    feat = build_feature_vector(110, 100, 1.0, 0.05, 0.20)
    assert feat[-1] == 1.0   # ITM

def test_feature_otm_flag():
    feat = build_feature_vector(85, 100, 1.0, 0.05, 0.20)
    assert feat[-1] == -1.0  # OTM


# ── Dataset tests ──────────────────────────────────────────────────────────────

def test_synthetic_dataset_length():
    cfg = OptionsModelConfig(sequence_length=10)
    ds = SyntheticOptionsDataset(n_paths=10, path_len=50, cfg=cfg, seed=0)
    assert len(ds) > 0


def test_synthetic_dataset_item_shapes():
    cfg = OptionsModelConfig(sequence_length=10, input_dim=10)
    ds = SyntheticOptionsDataset(n_paths=5, path_len=30, cfg=cfg, seed=1)
    feats, label = ds[0]
    assert feats.shape == (10, 10)
    assert label.shape == ()
    assert feats.dtype == torch.float32


# ── Model tests ────────────────────────────────────────────────────────────────

def _small_options_cfg():
    return OptionsModelConfig(
        input_dim=10,
        hidden_dim=32,
        num_layers=2,
        sequence_length=10,
        output_dim=1,
    )


def test_options_model_forward_shape():
    cfg = _small_options_cfg()
    model = build_options_model(cfg)
    x = torch.randn(4, cfg.sequence_length, cfg.input_dim)
    price, hidden = model(x)
    assert price.shape == (4, 1)


def test_options_model_predict_scalar():
    cfg = _small_options_cfg()
    model = build_options_model(cfg)
    x = torch.randn(2, cfg.sequence_length, cfg.input_dim)
    pred = model.predict(x)
    assert pred.shape == (2,)


def test_options_model_hidden_state():
    """Hidden state should be different from zero-init after a forward pass."""
    cfg = _small_options_cfg()
    model = build_options_model(cfg)
    x = torch.randn(1, cfg.sequence_length, cfg.input_dim)
    _, (hn, cn) = model(x)
    assert not torch.equal(hn, torch.zeros_like(hn))


def test_options_gradients_flow():
    cfg = _small_options_cfg()
    model = build_options_model(cfg)
    model.train()
    x = torch.randn(2, cfg.sequence_length, cfg.input_dim)
    target = torch.rand(2)
    loss_fn = OptionsPricingLoss()
    pred, _ = model(x)
    losses = loss_fn(pred, target)
    losses["loss"].backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(not np.isnan(g) for g in grad_norms)


def test_options_parameter_count():
    model = build_options_model(_small_options_cfg())
    assert model.num_parameters > 0
    print(f"  Options small model: {model.num_parameters:,} parameters")


def test_options_checkpoint_roundtrip(tmp_path):
    cfg = _small_options_cfg()
    model = build_options_model(cfg)
    path = str(tmp_path / "opts.pt")
    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "val_loss": 0.1,
        "model_config": cfg,
        "history": {},
    }, path)
    loaded, ckpt = load_options_checkpoint(path)
    assert ckpt["epoch"] == 1
    x = torch.randn(1, cfg.sequence_length, cfg.input_dim)
    model.eval()  # dropout off on both sides; the loader returns an eval-mode model
    assert not loaded.training
    with torch.no_grad():
        out_orig, _ = model(x)
        out_load, _ = loaded(x)
    assert torch.allclose(out_orig, out_load)


# ── Loss tests ─────────────────────────────────────────────────────────────────

def test_options_loss_positive():
    loss_fn = OptionsPricingLoss()
    pred = torch.rand(4, 1)
    target = torch.rand(4)
    losses = loss_fn(pred, target)
    assert losses["loss"].item() >= 0
    assert "mse" in losses
    assert "no_arb" in losses


if __name__ == "__main__":
    tests = [
        test_bsm_call_put_parity,
        test_bsm_call_positive,
        test_bsm_put_positive,
        test_bsm_greeks_delta_bounds,
        test_bsm_greeks_gamma_positive,
        test_implied_vol_roundtrip,
        test_feature_vector_shape,
        test_feature_itm_flag,
        test_feature_otm_flag,
        test_synthetic_dataset_length,
        test_synthetic_dataset_item_shapes,
        test_options_model_forward_shape,
        test_options_model_predict_scalar,
        test_options_model_hidden_state,
        test_options_gradients_flow,
        test_options_parameter_count,
        test_options_loss_positive,
    ]
    for fn in tests:
        name = fn.__name__
        try:
            if "checkpoint" in name:
                fn(tmp_path=pathlib.Path(tempfile.mkdtemp()))
            else:
                fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            raise
    print(f"\nAll {len(tests)} options tests passed.")
