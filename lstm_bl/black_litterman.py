"""
Black-Litterman (BL) portfolio model — analytical engine.

The LSTM augments the BL model in two ways:
  1. View generation  — LSTM predicts the P·μ vector (expected excess returns
     for each view) instead of hand-crafting analyst views.
  2. Confidence matrix — LSTM predicts diagonal of Ω (view uncertainty) based
     on recent market regime features.

Classical BL recap
------------------
Given:
  Π  = δ Σ w_mkt          equilibrium returns (reverse-optimised from market cap)
  P  = view pick matrix    (k × n)
  Q  = view returns        (k,)
  Ω  = view uncertainty    (k × k diagonal)
  τ  = scalar uncertainty on prior

Posterior expected returns:
  μ_BL = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ · [(τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q]

Posterior covariance:
  M    = (τΣ)⁻¹ + Pᵀ Ω⁻¹ P
  Σ_BL = Σ + M⁻¹

Optimal portfolio weights (mean-variance):
  w* = (δ Σ_BL)⁻¹ μ_BL    (then normalise to sum=1)
"""
import numpy as np
from typing import Optional, Tuple


def reverse_optimise(
    w_mkt: np.ndarray,
    Sigma: np.ndarray,
    delta: float = 2.5,
) -> np.ndarray:
    """
    Reverse-optimise equilibrium excess returns from market-cap weights.

    Π = δ · Σ · w_mkt
    """
    return delta * Sigma @ w_mkt


def black_litterman(
    Sigma: np.ndarray,
    w_mkt: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05,
    delta: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytical Black-Litterman posterior.

    Parameters
    ----------
    Sigma  : (n, n)  asset covariance matrix
    w_mkt  : (n,)    market-cap weights (sum=1)
    P      : (k, n)  view pick matrix
    Q      : (k,)    view expected returns
    Omega  : (k, k)  view uncertainty (diagonal)
    tau    : scalar  uncertainty of prior
    delta  : scalar  risk-aversion coefficient

    Returns
    -------
    mu_bl    : (n,)  posterior expected returns
    Sigma_bl : (n,n) posterior covariance
    """
    n = Sigma.shape[0]
    Pi = reverse_optimise(w_mkt, Sigma, delta)

    tauSigma_inv = np.linalg.inv(tau * Sigma)
    Omega_inv = np.linalg.inv(Omega)

    M = tauSigma_inv + P.T @ Omega_inv @ P       # (n, n)
    M_inv = np.linalg.inv(M)

    mu_bl = M_inv @ (tauSigma_inv @ Pi + P.T @ Omega_inv @ Q)
    Sigma_bl = Sigma + M_inv

    return mu_bl, Sigma_bl


def mv_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    delta: float = 2.5,
    long_only: bool = True,
) -> np.ndarray:
    """
    Mean-variance optimal weights.

    w* = (δ Σ)⁻¹ μ,  then normalised.
    If long_only: clip negatives to 0 before normalising.
    """
    raw = np.linalg.solve(delta * Sigma, mu)
    if long_only:
        raw = np.clip(raw, 0, None)
    total = raw.sum()
    if total <= 1e-8:
        return np.ones(len(mu)) / len(mu)
    return raw / total


def portfolio_stats(
    w: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
) -> dict:
    """Compute annualised return, volatility, and Sharpe (rf=0)."""
    ret = float(w @ mu) * 252
    vol = float(np.sqrt(w @ Sigma @ w)) * np.sqrt(252)
    sharpe = ret / vol if vol > 0 else 0.0
    return {"return": ret, "volatility": vol, "sharpe": sharpe}


def covariance_from_returns(returns: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """
    Sample covariance with Ledoit-Wolf style constant-correlation shrinkage.

    Parameters
    ----------
    returns   : (T, n) daily returns matrix
    shrinkage : blend weight toward diagonal (reduces estimation error)
    """
    S = np.cov(returns.T)
    target = np.diag(np.diag(S))
    return (1 - shrinkage) * S + shrinkage * target
