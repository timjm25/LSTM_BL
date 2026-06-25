"""
Black-Scholes-Merton (BSM) analytical pricer and Greeks.

Used as:
  1. Target labels when training on synthetic data
  2. Baseline benchmark to compare LSTM against
  3. Feature engineering (BSM implied-vol as an input feature)
"""
import math
import numpy as np
from scipy.stats import norm


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    """Compute d1, d2 for BSM formula.  Returns (nan, nan) if inputs are invalid."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan"), float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Analytical Black-Scholes-Merton option price.

    Args:
        S           : current underlying price
        K           : strike price
        T           : time to expiry in years
        r           : continuously compounded risk-free rate
        sigma       : implied/historical volatility (annualised)
        option_type : 'call' or 'put'

    Returns:
        option price (float)
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if math.isnan(d1):
        return 0.0
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    """Compute first-order Greeks for BSM model."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if math.isnan(d1):
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    sqrt_T = math.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    disc = math.exp(-r * T)

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        rho = K * T * disc * norm.cdf(d2) / 100
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            - r * K * disc * norm.cdf(d2)
        ) / 365
    else:
        delta = norm.cdf(d1) - 1
        rho = -K * T * disc * norm.cdf(-d2) / 100
        theta = (
            -(S * pdf_d1 * sigma) / (2 * sqrt_T)
            + r * K * disc * norm.cdf(-d2)
        ) / 365

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """Newton-Raphson implied volatility solver."""
    if T <= 0:
        return float("nan")
    sigma = 0.2  # initial guess
    for _ in range(max_iter):
        price = bsm_price(S, K, T, r, sigma, option_type)
        d1, _ = _d1_d2(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * math.sqrt(T)
        if abs(vega) < 1e-10:
            break
        diff = price - market_price
        if abs(diff) < tol:
            break
        sigma -= diff / vega
        sigma = max(sigma, 1e-6)
    return sigma


# ── Vectorised wrappers ────────────────────────────────────────────────────────

def bsm_price_vec(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    option_type: str = "call",
) -> np.ndarray:
    """Vectorised BSM price over arrays of inputs."""
    return np.array([
        bsm_price(s, k, t, ri, sig, option_type)
        for s, k, t, ri, sig in zip(S, K, T, r, sigma)
    ], dtype=np.float32)
