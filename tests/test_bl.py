"""Unit tests for LSTM Black-Litterman portfolio model."""
import sys
import tempfile
import pathlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from lstm_bl.config import BLModelConfig, BLTrainConfig
from lstm_bl.black_litterman import (
    reverse_optimise,
    black_litterman,
    mv_weights,
    portfolio_stats,
    covariance_from_returns,
)
from lstm_bl.dataset import SyntheticBLDataset
from lstm_bl.model import (
    BLPortfolioLSTM,
    DifferentiableBLSolver,
    build_bl_model,
    load_bl_checkpoint,
)
from lstm_bl.train import PortfolioLoss


# ── Analytical BL tests ────────────────────────────────────────────────────────

def test_reverse_optimise_shape():
    n = 5
    w = np.ones(n) / n
    Sigma = np.eye(n) * 0.04
    Pi = reverse_optimise(w, Sigma)
    assert Pi.shape == (n,)


def test_bl_posterior_shape():
    n = 4
    Sigma = np.eye(n) * 0.04
    w_mkt = np.ones(n) / n
    P = np.eye(n)
    Q = np.array([0.05, 0.03, -0.01, 0.07])
    Omega = np.eye(n) * 0.01
    mu_bl, Sigma_bl = black_litterman(Sigma, w_mkt, P, Q, Omega)
    assert mu_bl.shape == (n,)
    assert Sigma_bl.shape == (n, n)


def test_bl_weights_sum_to_one():
    n = 4
    Sigma = np.eye(n) * 0.04
    w_mkt = np.ones(n) / n
    P = np.eye(n)
    Q = np.zeros(n)
    Omega = np.eye(n) * 0.01
    mu_bl, Sigma_bl = black_litterman(Sigma, w_mkt, P, Q, Omega)
    w = mv_weights(mu_bl, Sigma_bl)
    assert abs(w.sum() - 1.0) < 1e-6


def test_bl_long_only_weights():
    n = 4
    Sigma = np.eye(n) * 0.04
    w_mkt = np.ones(n) / n
    P = np.eye(n)
    Q = np.zeros(n)
    Omega = np.eye(n) * 0.01
    mu_bl, Sigma_bl = black_litterman(Sigma, w_mkt, P, Q, Omega)
    w = mv_weights(mu_bl, Sigma_bl, long_only=True)
    assert (w >= 0).all()


def test_portfolio_stats():
    n = 4
    w = np.ones(n) / n
    mu = np.array([0.10, 0.08, 0.06, 0.12]) / 252
    Sigma = np.eye(n) * (0.20 ** 2 / 252)
    stats = portfolio_stats(w, mu, Sigma)
    assert "return" in stats
    assert "volatility" in stats
    assert "sharpe" in stats
    assert stats["volatility"] > 0


def test_covariance_psd():
    """Covariance matrix should be positive semi-definite."""
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((100, 5)) * 0.01
    Sigma = covariance_from_returns(returns)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert (eigvals >= -1e-10).all(), "Covariance not PSD"


# ── Differentiable BL solver tests ────────────────────────────────────────────

def test_diff_bl_solver_output_shape():
    B, n = 4, 5
    solver = DifferentiableBLSolver(n_assets=n)
    Q = torch.randn(B, n)
    omega = torch.abs(torch.randn(B, n)) + 0.01
    Sigma = torch.eye(n).unsqueeze(0).repeat(B, 1, 1) * 0.04
    w_mkt = torch.ones(B, n) / n
    weights = solver(Q, omega, Sigma, w_mkt)
    assert weights.shape == (B, n)


def test_diff_bl_solver_sums_to_one():
    B, n = 4, 5
    solver = DifferentiableBLSolver(n_assets=n)
    Q = torch.randn(B, n)
    omega = torch.abs(torch.randn(B, n)) + 0.01
    Sigma = torch.eye(n).unsqueeze(0).repeat(B, 1, 1) * 0.04
    w_mkt = torch.ones(B, n) / n
    weights = solver(Q, omega, Sigma, w_mkt)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5)


# ── Dataset tests ──────────────────────────────────────────────────────────────

def test_synthetic_bl_dataset_nonzero():
    cfg = BLModelConfig(n_assets=4, macro_features=4, total_input_dim=8, sequence_length=30)
    ds = SyntheticBLDataset(n_series=10, series_len=100, cfg=cfg, seed=0)
    assert len(ds) > 0


def test_synthetic_bl_item_shapes():
    n = 4
    macro = 4
    cfg = BLModelConfig(n_assets=n, macro_features=macro, total_input_dim=n+macro, sequence_length=20)
    ds = SyntheticBLDataset(n_series=5, series_len=80, cfg=cfg, seed=1)
    feats, weights = ds[0]
    assert feats.shape == (20, n + macro)
    assert weights.shape == (n,)


def test_synthetic_bl_weights_sum_to_one():
    cfg = BLModelConfig(n_assets=4, macro_features=4, total_input_dim=8, sequence_length=20)
    ds = SyntheticBLDataset(n_series=5, series_len=80, cfg=cfg, seed=2)
    for i in range(min(10, len(ds))):
        _, w = ds[i]
        assert abs(w.sum().item() - 1.0) < 1e-5


# ── Full model tests ───────────────────────────────────────────────────────────

def _small_bl_cfg():
    return BLModelConfig(
        n_assets=4,
        macro_features=4,
        total_input_dim=8,
        hidden_dim=32,
        num_layers=2,
        sequence_length=20,
    )


def test_bl_model_forward_shape():
    cfg = _small_bl_cfg()
    model = build_bl_model(cfg)
    B, T = 2, cfg.sequence_length
    x = torch.randn(B, T, cfg.n_assets + cfg.macro_features)
    weights, Q, omega = model(x)
    assert weights.shape == (B, cfg.n_assets)
    assert Q.shape == (B, cfg.n_assets)
    assert omega.shape == (B, cfg.n_assets)


def test_bl_model_weights_simplex():
    """Predicted weights should sum to 1 and be non-negative."""
    cfg = _small_bl_cfg()
    model = build_bl_model(cfg)
    model.eval()
    x = torch.randn(4, cfg.sequence_length, cfg.n_assets + cfg.macro_features)
    with torch.no_grad():
        weights, _, _ = model(x)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert (weights >= 0).all()


def test_bl_model_gradients_flow():
    cfg = _small_bl_cfg()
    model = build_bl_model(cfg)
    model.train()
    x = torch.randn(2, cfg.sequence_length, cfg.n_assets + cfg.macro_features)
    target_w = torch.ones(2, cfg.n_assets) / cfg.n_assets
    loss_fn = PortfolioLoss()
    weights, Q, omega = model(x)
    losses = loss_fn(weights, target_w, x[:, :, :cfg.n_assets])
    losses["loss"].backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(not np.isnan(g) for g in grad_norms)


def test_bl_model_parameter_count():
    model = build_bl_model(_small_bl_cfg())
    assert model.num_parameters > 0
    print(f"  BL small model: {model.num_parameters:,} parameters")


def test_bl_model_predict_weights():
    cfg = _small_bl_cfg()
    model = build_bl_model(cfg)
    x = torch.randn(1, cfg.sequence_length, cfg.n_assets + cfg.macro_features)
    w_np = model.predict_weights(x)
    assert w_np.shape == (1, cfg.n_assets)
    assert abs(w_np.sum() - 1.0) < 1e-5


def test_bl_checkpoint_roundtrip(tmp_path):
    cfg = _small_bl_cfg()
    model = build_bl_model(cfg)
    path = str(tmp_path / "bl.pt")
    torch.save({
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "val_loss": 0.05,
        "model_config": cfg,
        "history": {},
    }, path)
    loaded, ckpt = load_bl_checkpoint(path)
    assert ckpt["epoch"] == 3
    x = torch.randn(1, cfg.sequence_length, cfg.n_assets + cfg.macro_features)
    with torch.no_grad():
        w1, _, _ = model(x)
        w2, _, _ = loaded(x)
    assert torch.allclose(w1, w2)


# ── Loss tests ─────────────────────────────────────────────────────────────────

def test_portfolio_loss_positive():
    n = 4
    loss_fn = PortfolioLoss()
    pred = torch.ones(2, n) / n
    target = torch.ones(2, n) / n
    returns = torch.randn(2, 20, n) * 0.01
    losses = loss_fn(pred, target, returns)
    assert "loss" in losses
    assert "mse" in losses


if __name__ == "__main__":
    tests = [
        test_reverse_optimise_shape,
        test_bl_posterior_shape,
        test_bl_weights_sum_to_one,
        test_bl_long_only_weights,
        test_portfolio_stats,
        test_covariance_psd,
        test_diff_bl_solver_output_shape,
        test_diff_bl_solver_sums_to_one,
        test_synthetic_bl_dataset_nonzero,
        test_synthetic_bl_item_shapes,
        test_synthetic_bl_weights_sum_to_one,
        test_bl_model_forward_shape,
        test_bl_model_weights_simplex,
        test_bl_model_gradients_flow,
        test_bl_model_parameter_count,
        test_bl_model_predict_weights,
        test_portfolio_loss_positive,
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
    print(f"\nAll {len(tests)} BL tests passed.")
