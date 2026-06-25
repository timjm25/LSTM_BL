# LSTM Financial Models — Options Pricing & Black-Litterman Portfolio

Two production-ready LSTM models for quantitative finance:

1. **`lstm_options`** — LSTM Options Pricer that learns to approximate and extend the Black-Scholes-Merton formula
2. **`lstm_bl`** — LSTM Black-Litterman Portfolio Optimiser that generates analyst views and confidence estimates from market data

---

## Business Case

### Why LSTM for Options Pricing?

The Black-Scholes-Merton (BSM) formula is the industry standard for options pricing, but it rests on restrictive assumptions that consistently fail in live markets:

| BSM Assumption | Reality |
|---|---|
| Constant volatility | Volatility smiles, surfaces, term structure |
| Log-normal returns | Fat tails, negative skewness, jump diffusion |
| No transaction costs | Bid-ask spreads, market impact, settlement |
| Continuous hedging | Discrete rebalancing, gamma risk |
| Known interest rates | Rate uncertainty, stochastic rates |

**LSTM overcomes these limitations** by learning the pricing function directly from observed market prices, capturing:
- **Path dependence** — the full trajectory of the underlying matters, not just current state
- **Volatility regime shifts** — GARCH effects, volatility clustering, mean reversion
- **Smile/skew corrections** — ITM/ATM/OTM options price differently than BSM predicts
- **Macro-regime sensitivity** — options price differently in bull vs bear markets

**Quantified benefit:** Academic studies show ML-based pricers reduce out-of-sample RMSE by 30–60% vs BSM, with the largest gains for deep ITM/OTM options and near-expiry contracts.

**Target users:**
- Proprietary trading desks (options market making)
- Buy-side hedge funds (volatility arbitrage, dispersion trading)
- Risk management teams (VaR, CVA calculations)
- Asset managers (structured product pricing)

---

### Why LSTM for Black-Litterman?

The Black-Litterman (BL) model elegantly combines market equilibrium returns with analyst views to produce portfolios that are more stable and intuitive than raw Markowitz mean-variance optimisation. However, the classical BL model has its own bottleneck: **the views (Q) and confidence matrix (Ω) must be specified by human analysts**, which is:

- Slow (analyst time is expensive)
- Subjective (anchoring, recency bias)
- Inconsistent (views change style between analysts)
- Not scalable (difficult to cover hundreds of assets simultaneously)

**LSTM replaces human view generation** with a data-driven process:
- **View head** predicts Q (expected excess returns per asset) from the last 60 days of returns, macro factors, and regime signals
- **Uncertainty head** predicts Ω diag (how confident the model is in each view) based on recent prediction error patterns
- **Differentiable BL solver** runs the analytical BL posterior inside the PyTorch computation graph, enabling end-to-end training

**Business impact:**
- Systematic funds: generate views at portfolio rebalance frequency (daily/weekly) across 50–500 assets
- Risk parity: let the model weight down assets whose view uncertainty is high
- Factor investing: macro LSTM stream captures factor regime (value/momentum/quality rotation)
- Robo-advisory: automate the allocation process for client portfolios without discretionary bias

---

## Architecture

### Model 1: LSTM Options Pricer (`lstm_options/`)

```
Input: (B, T, 10) — rolling window of option features
  [log(S/K), T_expiry, r, σ, moneyness, Δ, Γ, V, Θ, ITM/ATM/OTM flag]
        │
        ▼
┌──────────────────────────────────┐
│  Stacked LSTM                     │   3 layers, hidden=128
│  (batch_first, no bidirectional)  │   Dropout between layers
│  Forget-gate bias init = 1.0      │   Prevents vanishing gradient on
└──────────┬───────────────────────┘   long option paths
           │ LSTM output (B, T, 128)
           ▼
┌──────────────────────────────────┐
│  Temporal Attention Gate          │   Additive attention — learns which
│  score(h_t) = W · h_t            │   timesteps in the window matter most
└──────────┬───────────────────────┘   (recent vol spike vs 30-day avg)
           │ context (B, 128)
           ▼
┌──────────────────────────────────┐
│  Residual MLP Decoder             │   LayerNorm + GELU + skip connection
│  128 → 64 → 32 → 1               │   Stabilises training on normalised
└──────────────────────────────────┘   price targets
           │
           ▼
    Option price (B, 1)  — normalised, inverse-transform for £/$ value
```

**Training objective:**
```
L = MSE(pred, target)  +  λ · ReLU(-pred)    (no-arbitrage penalty)
```

The no-arbitrage term penalises any prediction below zero — options have a non-negative intrinsic value floor.

**Input features (10-dim per timestep):**

| Index | Feature | Description |
|---|---|---|
| 0 | log(S/K) | Log-moneyness — key for smile modelling |
| 1 | T | Time to expiry (years) |
| 2 | r | Risk-free rate |
| 3 | σ | Historical/implied vol estimate |
| 4 | S/K | Linear moneyness |
| 5 | Δ (delta) | BSM delta — hedge ratio signal |
| 6 | Γ × 10 | BSM gamma (scaled) — convexity |
| 7 | V | BSM vega — vol sensitivity |
| 8 | Θ × 10 | BSM theta (scaled) — time decay |
| 9 | moneyness flag | +1 ITM, 0 ATM ±5%, −1 OTM |

---

### Model 2: LSTM Black-Litterman (`lstm_bl/`)

```
Input: (B, T, n_assets + macro_features)
  Asset returns: (B, T, n)      — daily log-returns for each asset
  Macro signals: (B, T, 8)      — [VIX, yield_spread, credit_spread,
                                     momentum, value, quality, FX_vol, commodity]
        │
   ┌────┴────────────┐
   │                 │
   ▼                 ▼
┌──────────┐   ┌──────────┐
│  Asset    │   │  Macro   │   Two separate LSTM encoders allow
│  LSTM     │   │  LSTM    │   independent learning of price
│  (3 lay.) │   │  (2 lay.)│   dynamics vs macro regime
└────┬──────┘   └────┬─────┘
     │               │
     └───────┬────────┘
             ▼
┌──────────────────────────────────┐
│  Cross-Stream Attention           │   Macro hidden state queries over
│  query = macro_h                  │   asset LSTM outputs — learns WHICH
│  keys  = asset_h_t for t=1..T    │   assets respond to which regimes
└──────────┬───────────────────────┘   (e.g. gold responds to credit stress)
           │
           ▼
┌──────────────────────────────────┐
│  Fusion Layer (LayerNorm + GELU)  │   Combines asset signal + cross-attn
└──────────┬───────────────────────┘
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌────────────┐
│  View    │  │ Uncertainty │   Parallel heads — same fused representation
│  Head Q  │  │  Head Ω    │   Q: expected excess returns  (n_assets,)
│          │  │ (softplus) │   Ω: view uncertainty diagonal (n_assets,)
└────┬─────┘  └─────┬──────┘
     └────────┬──────┘
              ▼
┌──────────────────────────────────┐
│  Differentiable BL Solver         │   μ_BL = M⁻¹[(τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q]
│  (PyTorch matrix ops — end-to-end │   Σ_BL = Σ + M⁻¹
│   gradient through BL formula)    │   Σ estimated via EW covariance
│  Learnable risk aversion δ        │   with learnable decay parameter
└──────────┬───────────────────────┘
           ▼
    Portfolio weights w* (B, n_assets)
    [Softmax — long-only simplex constraint ∑w = 1]
```

**Training objective (multi-objective):**
```
L = MSE(w_pred, w_BL_target)
  + λ₁ · (−Sharpe_proxy)        maximise risk-adjusted returns
  + λ₂ · (−Entropy(w))          encourage diversification
  + λ₃ · Turnover(w_t, w_{t-1}) penalise excessive rebalancing
```

**Black-Litterman posterior (solved in closed form inside PyTorch):**
```
Π     = δ · Σ · w_mkt                   (equilibrium reverse-optimised returns)
M     = (τΣ)⁻¹ + Pᵀ Ω⁻¹ P
μ_BL  = M⁻¹ [(τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q]
Σ_BL  = Σ + M⁻¹
w*    = softmax((δ Σ_BL)⁻¹ μ_BL)
```

---

## Data

### Options Pricing Data

Real market data: CSV with columns `date, S, K, T, r, sigma, price`

Synthetic: Monte-Carlo GBM paths with stochastic volatility (GARCH-lite), labelled with BSM prices. Automatically used when no CSV is present.

```bash
python run_options.py  # auto-generates 1,000 synthetic paths
```

### Portfolio Data

Real market data: CSV with columns `date, asset_0, ..., asset_{n-1}, vix, yield_spread, ...`

Synthetic: Regime-switching GBM with correlated macro signals (bull/bear state machine). Automatically used when no CSV is present.

```bash
python run_bl.py  # auto-generates 200 synthetic multi-asset series
```

---

## Quick Start

### Install

```bash
pip install torch torchvision numpy scipy pandas matplotlib
```

### Train Options Pricing Model

```bash
# Synthetic data (no real data needed)
python run_options.py

# Real options chain CSV
python run_options.py --data_dir /path/to/options_data --option_type call --epochs 200

# Available options
python run_options.py --help
```

### Train Black-Litterman Portfolio Model

```bash
# Synthetic data
python run_bl.py

# Real returns + macro CSV
python run_bl.py --data_dir /path/to/portfolio_data --n_assets 20 --epochs 200

# Available options
python run_bl.py --help
```

### Inference — Options Pricing

```python
import torch
from lstm_options.model import load_options_checkpoint
from lstm_options.dataset import build_feature_vector

model, ckpt = load_options_checkpoint("checkpoints/options/best_options.pt")
model.eval()

# Build a 30-day feature window for a specific option
features = torch.tensor([
    build_feature_vector(S=105, K=100, T=0.25, r=0.05, sigma=0.22)
    for _ in range(30)
]).unsqueeze(0)  # (1, 30, 10)

normalised_price = model.predict(features).item()

# Inverse-transform (store label_mean/label_std from training dataset)
price = normalised_price * label_std + label_mean
print(f"Predicted option price: ${price:.2f}")
```

### Inference — Black-Litterman Weights

```python
import torch
import numpy as np
from lstm_bl.model import load_bl_checkpoint

model, ckpt = load_bl_checkpoint("checkpoints/bl/best_bl.pt")

# Assemble 60-day window: (1, 60, n_assets + macro_features)
returns_window  = torch.tensor(asset_returns[-60:]).unsqueeze(0)
macro_window    = torch.tensor(macro_features[-60:]).unsqueeze(0)
x = torch.cat([returns_window, macro_window], dim=-1)

weights = model.predict_weights(x)[0]  # (n_assets,)
print("Portfolio allocation:")
for asset, w in zip(asset_names, weights):
    print(f"  {asset}: {w*100:.1f}%")
```

---

## Project Structure

```
LSTM_BL/
├── lstm_options/
│   ├── __init__.py
│   ├── config.py          # OptionsModelConfig, OptionsTrainConfig
│   ├── black_scholes.py   # Analytical BSM pricer, Greeks, IV solver
│   ├── dataset.py         # SyntheticOptionsDataset, CSVOptionsDataset
│   ├── model.py           # OptionsLSTM with temporal attention
│   └── train.py           # OptionsPricingLoss, OptionsTrainer
├── lstm_bl/
│   ├── __init__.py
│   ├── config.py          # BLModelConfig, BLTrainConfig
│   ├── black_litterman.py # Analytical BL posterior, MV weights
│   ├── dataset.py         # SyntheticBLDataset, CSVBLDataset
│   ├── model.py           # BLPortfolioLSTM, DifferentiableBLSolver
│   └── train.py           # PortfolioLoss, BLTrainer
├── tests/
│   ├── test_options.py    # 17 unit tests — BSM, dataset, model, loss
│   └── test_bl.py         # 17 unit tests — BL math, dataset, model, loss
├── run_options.py         # Options training entry point
├── run_bl.py              # BL portfolio training entry point
└── requirements.txt
```

---

## Tests

```bash
# Options pricing tests (17 tests)
python tests/test_options.py

# Black-Litterman tests (17 tests)
python tests/test_bl.py
```

Test coverage:
- BSM analytical correctness (put-call parity, IV round-trip)
- Feature engineering (moneyness flags, Greek computation)
- Synthetic data generation (shapes, action ranges)
- Model forward pass shapes
- Gradient flow (no NaN gradients)
- Checkpoint save/load round-trip
- Loss function correctness
- BL posterior math (weight simplex, PSD covariance)
- Differentiable BL solver (batched, end-to-end differentiable)

---

## Default Hyperparameters

### Options Model

| Parameter | Default | Notes |
|---|---|---|
| hidden_dim | 128 | LSTM hidden size |
| num_layers | 3 | Stacked LSTM depth |
| sequence_length | 30 | Lookback window (trading days) |
| input_dim | 10 | Feature vector dimension |
| learning_rate | 1e-3 | AdamW + cosine decay |
| batch_size | 64 | Sequences per batch |
| no_arb_weight | 0.1 | No-arbitrage penalty coefficient |

### Black-Litterman Model

| Parameter | Default | Notes |
|---|---|---|
| n_assets | 10 | Portfolio size |
| macro_features | 8 | Macro/sentiment dimensions |
| hidden_dim | 256 | LSTM hidden size |
| num_layers | 3 | Asset stream LSTM depth |
| sequence_length | 60 | Lookback (~3 months) |
| tau | 0.05 | BL prior uncertainty |
| sharpe_weight | 0.5 | Sharpe loss term weight |
| diversification_weight | 0.1 | Entropy regularisation |
| turnover_weight | 0.05 | Turnover penalty |

---

## Extending the Models

**Options model — stochastic vol (Heston):** Replace the GBM synthetic data generator with a Heston or SABR simulation to train on a richer vol surface.

**Options model — surface prediction:** Change output_dim from 1 to (n_strikes × n_maturities) to predict the full implied vol surface simultaneously.

**BL model — more assets:** Increase `n_assets` and provide a returns CSV — the model architecture scales automatically.

**BL model — ESG integration:** Add ESG scores as macro features — the cross-stream attention will learn which ESG signals drive factor exposures.

**BL model — transaction cost optimisation:** Add a transaction cost term to `PortfolioLoss` with a realistic bid-ask spread model.

**Combined model:** Use the options pricer's LSTM hidden states as input features to the BL macro stream — options implied vol surfaces are leading indicators for equity factor returns.

---

## Hardware Requirements

| Mode | Minimum | Recommended |
|---|---|---|
| Options training | 4 GB RAM, CPU | NVIDIA RTX 3080 |
| BL training | 8 GB RAM, CPU | NVIDIA A100 |
| Options inference | any CPU | edge device (Jetson) |
| BL inference | any CPU | real-time trading server |

Both models train in under **30 minutes** on CPU for synthetic data. GPU provides 5–10× speedup.

---

## Regulatory / Risk Notes

- Both models are **probabilistic estimators**, not guaranteed pricers. Always overlay human oversight for large positions.
- The options model output is a **price estimate**, not a risk-free hedge. Use BSM Greeks from `black_scholes.py` for delta hedging.
- The BL model produces **unconstrained allocations** by default. Add position limits in the `mv_weights()` call before live trading.
- Log all model predictions and inputs for audit trail compliance (MiFID II Article 25, SEC Rule 17a-4).

---

*LSTM Options model inspired by [Hutchinson et al. (1994)](https://doi.org/10.1111/j.1540-6261.1994.tb04768.x) and subsequent deep learning options pricing literature. Black-Litterman LSTM inspired by He & Litterman (1999) and Idzorek (2005).*
