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

## Quick Start

```bash
pip install -r requirements.txt

# 1. Try it on synthetic demo data (no files needed)
python run_options.py --synthetic --epochs 20
python run_bl.py --synthetic --epochs 20

# 2. Or write demo files in the real-data format and train from them
python make_demo_data.py                      # -> data/demo/options_demo.csv, data/demo/bl_demo.csv
python run_options.py --data data/demo/options_demo.csv --epochs 20
python run_bl.py      --data data/demo/bl_demo.csv      --epochs 20

# 3. Predict with the trained checkpoints
python predict.py options --data data/demo/options_demo.csv --out option_prices.csv
python predict.py bl      --data data/demo/bl_demo.csv      --out weights.csv

# Tests (fast, CPU only)
python3 -m pytest -q
```

Checkpoints are written to `checkpoints/options/` and `checkpoints/bl/` (git-ignored; change with
`--checkpoint_dir`). `--help` on any script lists every option. With no `--data`, the first
`.csv`/`.xlsx`/`.json` file in `--data_dir` (default `data/options`, `data/bl`) is used, and if there
is none the scripts fall back to synthetic data and say so.

---

## Bring your own data

Copy a template from [`data/templates/`](data/templates/) (the rows marked `SAMPLE` are examples to
replace), fill in your history, and pass it with `--data`. CSV, Excel (`.xlsx`, needs `openpyxl`)
and JSON (a list of records) are accepted. Each file is checked before training; problems are
reported with row numbers and nothing is trained on a bad file.

### Options file (`options_template.csv`)

One row per contract per day. Header names are case-insensitive and common aliases work
(`spot`, `strike`, `tte`, `iv`, `mid`, `call/put`, ...).

| Column | Required | Meaning |
|---|---|---|
| `S` | yes | underlying price (> 0) |
| `K` | yes | strike (> 0) |
| `T` | yes | time to expiry in **years** (> 0; divide days by 365) |
| `r` | yes | risk-free rate as a decimal (0.045, not 4.5) |
| `price` | yes | observed option price (the training label) |
| `sigma` | no | annualised volatility; blank cells are implied from `price` with the BSM IV solver |
| `contract_id` | no | groups rows into contracts; windows never span two contracts |
| `date` | no | sorts rows within a contract and drives the chronological train/val/test split |
| `option_type` | no | `call`/`put` (or `C`/`P`); default from `--option_type` |

Each training sample is `--seq_len` consecutive rows of one contract; the label is the price on
the next row. With `--bsm_residual`, the model learns the correction to the textbook BSM price of
that row instead of the raw price (on synthetic data the labels *are* BSM prices, so the residual
is zero there; it is meant for market data).

### Portfolio file (`bl_template.csv`)

One row per trading day, wide format.

| Column | Meaning |
|---|---|
| `date` | optional; parsed and used to sort rows (duplicates are an error) |
| `asset_*` | one column per asset: daily returns as decimals (0.01 = 1%). Use `--assets SPY,TLT,GLD` to pick columns with other names, and `--prices` if the columns are prices — they are converted to returns |
| everything else | macro/regime features (VIX, spreads, factor returns, ...). Use `--macro vix,credit_spread` to choose them; missing values are forward-filled |

The number of assets and macro features is taken from the file, so `--n_assets` only applies to
synthetic data. You need at least `--seq_len` + 22 rows (the target weights are built from the next
21 days). Real data is split chronologically (latest windows held out for validation and test) so
overlapping windows don't leak.

### Synthetic data

* Options: Monte-Carlo GBM paths with a rolling realised-vol estimate, labelled with BSM prices.
* Portfolio: regime-switching (bull/bear) correlated returns with co-moving macro signals.

---

## Inference

`predict.py` (above) is the simplest route. From Python:

```python
from lstm_options.predict import predict_options
from lstm_bl.predict import predict_bl_weights

prices = predict_options("checkpoints/options/best_options.pt", "my_options.csv")
# columns: contract_id, date, option_type, market_price, bsm_price, lstm_price

weights = predict_bl_weights("checkpoints/bl/best_bl.pt", "my_portfolio.csv")
# one row per asset: asset, weight, view_Q, view_uncertainty (for the latest seq_len window)
```

Checkpoints store what inference needs: the options checkpoint keeps `label_mean`/`label_std`
(to turn the normalised output back into a price) and whether it was trained in residual mode;
the BL checkpoint keeps the asset and macro column names in training order. Both loaders
(`load_options_checkpoint`, `load_bl_checkpoint`) return the model in eval mode.

---

## Project Structure

```
LSTM_BL/
├── lstm_options/
│   ├── config.py          # OptionsModelConfig, OptionsTrainConfig
│   ├── black_scholes.py   # Analytical BSM pricer, Greeks, IV solver
│   ├── dataset.py         # load_options_table (validation), CSVOptionsDataset, SyntheticOptionsDataset
│   ├── model.py           # OptionsLSTM with temporal attention + BSM-residual mode
│   ├── predict.py         # predict_options: price a file with a checkpoint
│   └── train.py           # OptionsPricingLoss, OptionsTrainer
├── lstm_bl/
│   ├── config.py          # BLModelConfig, BLTrainConfig
│   ├── black_litterman.py # Analytical BL posterior, MV weights
│   ├── dataset.py         # load_bl_table (validation), CSVBLDataset, SyntheticBLDataset
│   ├── model.py           # BLPortfolioLSTM, DifferentiableBLSolver
│   ├── predict.py         # predict_bl_weights: weights for the latest window
│   └── train.py           # PortfolioLoss, BLTrainer
├── data/templates/        # options_template.csv, bl_template.csv (column schema + SAMPLE rows)
├── tests/
│   ├── test_options.py    # BSM, features, dataset, model, loss
│   ├── test_bl.py         # BL math, dataset, model, loss
│   └── test_data_io.py    # file loading/validation, residual mode, training smoke tests, predict
├── run_options.py         # Options training entry point
├── run_bl.py              # BL portfolio training entry point
├── predict.py             # Inference CLI for both models
├── make_demo_data.py      # Writes synthetic demo files in the real-data schema
└── requirements.txt
```

---

## Tests

```bash
python3 -m pytest -q      # 61 tests, ~10 s on CPU
```

Test coverage:
- BSM analytical correctness (put-call parity, IV round-trip)
- Feature engineering (moneyness flags, Greek computation)
- Synthetic data generation (shapes, action ranges)
- Model forward pass shapes, gradient flow, checkpoint save/load round-trip
- Loss function correctness
- BL posterior math (weight simplex, PSD covariance) and the differentiable BL solver
- Loading your own files: CSV/xlsx/JSON, header aliases, unit and value checks with row numbers,
  implied vol for missing sigma, per-contract windows, prices-to-returns, chronological splits
- BSM-residual mode, 1-epoch training runs on tiny files, checkpoint metadata, `predict.py`

---

## Out of scope

- No market-data connectors (Bloomberg, Refinitiv, broker APIs): export to CSV/xlsx and use `--data`.
- No backtester or transaction-cost model; the BL "Sharpe" printed during training is a quick
  per-batch proxy, not a backtest.
- No hyperparameter search; defaults are the ones listed below.

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

**BL model — more assets:** Provide a returns file with more `asset_*` columns — the model is sized from the file automatically.

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
