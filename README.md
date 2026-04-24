# Systematic Trading — Research & Live Strategies

![Python](https://img.shields.io/badge/python-3.11-blue) ![Status](https://img.shields.io/badge/status-live%20%2F%20research-green) ![Asset-class](https://img.shields.io/badge/asset--class-crypto%20perps-orange) ![Venue](https://img.shields.io/badge/venue-Hyperliquid-black)

This repository showcases three systematic trading strategies I have designed, backtested and, for two of them, deployed in a live or semi-live environment on crypto perpetuals. Signal construction, thresholds and parameter sets are intentionally omitted; this document focuses on architecture, methodology and out-of-sample results.

---

## Strategies Overview

| # | Strategy | Asset class | Universe | Timeframe | Style | Status |
|---|---|---|---|---|---|---|
| 1 | **Cross-Sectional Lead-Lag (Daily)** | Crypto perpetuals | ~40+ Hyperliquid perps | 1D | Cross-sectional L/S, multi-asset relative-value | Live |
| 2 | **Pairwise Vol-Spread (Intraday)** | Crypto perpetuals | Scheduled leader/lagger pairs | 1m | Pairwise statistical arbitrage / trend continuation | Research (walk-forward validated) |
| 3 | **Session-Filtered Directional (Intraday)** | Crypto perpetuals | Scheduled leader/follower pairs | 15m | Directional, session-conditioned | Semi-live |

---

### 1. Cross-Sectional Lead-Lag — Daily

A multi-asset long/short strategy operating on the full Hyperliquid perpetuals universe. Positions are rebalanced daily from a cross-sectional ranking extracted from the asset return matrix over a rolling lookback. Capital is allocated across a basket of followers with position weights driven by the signal strength.

- **Universe:** all liquid Hyperliquid perpetuals passing a liquidity / data-availability filter
- **Approach:** cross-sectional relative-value, rebuilt every cycle
- **Risk:** spread-vol targeting, rolling signal-reliability score, drawdown circuit breaker, intraday stop-loss
- **Live:** deployed through the Hyperliquid REST API, scheduled execution, market/ALO routing

**Backtest — Jan 2025 → Apr 2026 (full walk-forward, fees & spread modelled):**
| Metric | Value |
|---|---|
| Total return (start $10k) | **+85 – 100 %** |
| Max drawdown | ~ -45 % |
| α (excess over market-neutral benchmark) | 48 % |
| β (raw directional hit-rate) | 52.5 % |
| α ∩ β | 31.6 % |

---

### 2. Pairwise Vol-Spread — 1-Minute

High-frequency intraday strategy trading pre-scheduled leader/lagger pairs. Each bar, the engine looks up the currently-active pair from a conjurator schedule, evaluates a volatility-based entry condition on the leader, and takes a directional position on the lagger with a fixed-risk exit rule.

- **Universe:** dynamically rotating leader/lagger pairs (pair schedule rebuilt offline)
- **Approach:** pairwise statistical arbitrage with directional continuation
- **Execution:** entries and exits aligned to bar close; no pyramiding; hard parachute stop
- **Optimisation:** hyperparameter search via Optuna with pruning, objective evaluated on a held-out window

**Walk-forward backtest (fees 0.09 % round-trip):**
| Window | Trades | Return | Max DD |
|---|---|---|---|
| In-sample | 28 | +2.48 % | -2.95 % |
| Out-of-sample (short) | 24 | **+14.54 %** | -0.81 % |
| Out-of-sample (long) | 398 | -27.7 % | -35.4 % |

The long-OOS performance degradation was a deliberate stress test: it confirmed the pair schedule needs periodic re-fitting, which is now part of the production protocol.

---

### 3. Session-Filtered Directional — 15-Minute

A directional strategy on crypto perps restricted to a narrow intraday session. The model is fitted by pooled OLS with HC3 robust standard errors on a pre-built regression dataset; the backtest executes one trade per active bar with a fixed-time exit.

- **Universe:** scheduled leader/follower pairs resampled to 15-minute bars
- **Approach:** directional, conditioned on session and signal magnitude
- **Validation:** Monte-Carlo bootstrap on trade-level PnL (10 000 × n bootstrap, null-PF distribution)

**In-sample — Jan 2026 → Mar 2026 (fees 0.09 % round-trip):**
| Metric | Value |
|---|---|
| Trades | 58 |
| Win rate | ~50 % |
| **Profit factor** | **1.89** |
| Total return | **+9.40 %** |
| Sharpe (annualised) | **2.51** |
| Max drawdown | ~ -3 % |
| Bootstrap p-value (PF under H₀) | **3.49 %** |

The short out-of-sample window (9 trades) came back flat-to-negative, which is consistent with the small sample and reinforces the need for larger OOS horizons before scaling size.

---

## Tech Stack

**Language:** Python 3.11
**Core libs:** `pandas`, `numpy`, `scipy`, `networkx`, `statsmodels`, `matplotlib`
**Optimisation:** `optuna` (TPE sampler, MedianPruner)
**Exchange / data:** Hyperliquid REST & WS APIs (`hyperliquid-python-sdk`, `eth_account`)
**Infra:** scheduled execution (`schedule`), CSV-based data lake with incremental OHLC ingestion, structured logging, modular config via `dataclasses`
**Dev:** local backtests on M-series MacBook; live bot runs on a dedicated Ubuntu VM

---

## Methodology

The three projects share a common research protocol:

1. **Data pipeline.** Raw OHLC pulled from the exchange, cleaned, aligned on a unified time grid, and filtered on data availability before any signal computation.
2. **Walk-forward validation.** Rolling lookback windows — no future leakage. Parameters calibrated on in-sample, evaluated on untouched out-of-sample windows.
3. **Transaction-cost modelling.** Exchange taker/maker fees, half-spread and, where relevant, market-impact slippage are modelled explicitly — net PnL is the only metric reported.
4. **Risk management.** Separate module exposing volatility targeting, rolling signal-reliability scoring, and a drawdown circuit breaker that can flatten all positions. Per-trade intraday stop-loss on the daily strategy.
5. **Statistical significance.** Profit factors are stress-tested with bootstrap resampling; regression-based signals use HC3 robust standard errors.
6. **Execution parity.** Backtest engine and live bot share the same signal-generation code path to minimise IS/live divergence; deviations are tracked via a dedicated comparison notebook.

---

## Repository Layout

```
.
├── strategy_1_daily/          # cross-sectional L/S, 1D
├── strategy_2_vol_spread/     # pairwise vol-spread, 1m
├── strategy_3_session/        # session-filtered directional, 15m
```

Each strategy folder contains its own `backtest/`, `live_bot/` (where applicable), `data/`, `results/` and `trading_signal/` modules. Source code for the proprietary signal engines and parameter sets is **not** included in this public repository.

---

## About Me

Quant / systematic trader with a focus on crypto perpetuals market microstructure, cross-sectional signals and live execution infrastructure. I build strategies end-to-end: research, backtest, risk framework, live deployment.

- **Portfolio / contact:** _(replace with your link — e.g. LinkedIn, personal site, or Twitter/X)_
- **Email:** available on request

---

*This repository is a showcase. It does not constitute investment advice, nor a solicitation, nor a guarantee of future returns. Past backtest and live performance is not indicative of future results.*
