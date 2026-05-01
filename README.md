# Systematic Trading — Research & Live Strategies

---
> [!WARNING]
> **This repository is not actively updated.** Source code, notebooks and results files may be several months old. Only this README reflects the current state of the research. Strategy implementations, signal engines and parameters are intentionally withheld to preserve the integrity of live trading operations.
---

![Python](https://img.shields.io/badge/python-3.11-blue) ![Status](https://img.shields.io/badge/status-live%20%2F%20research-green) ![Asset-class](https://img.shields.io/badge/asset--class-crypto%20perps-orange) ![Venue](https://img.shields.io/badge/venue-Hyperliquid-black)

This repository showcases three systematic trading strategies I have designed, backtested and, for one of them, deployed live on crypto perpetuals. Signal construction, thresholds and parameter sets are intentionally omitted; this document focuses on architecture, methodology and out-of-sample results.

The strategies are grounded in academic research on lead-lag dynamics and rough path theory. The primary theoretical foundation is [Bennett et al. (2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4599565); additional reference papers are available [here](https://drive.google.com/drive/folders/1pJbmYzLIpDhWyt0yeBOKrEfnpawo5FCX?usp=drive_link).

---

## Strategies Overview

| # | Strategy | Asset class | Universe | Timeframe | Style | Status |
|---|---|---|---|---|---|---|
| 1 | **Cross-Sectional Lead-Lag (Daily)** | Crypto perpetuals | ~220 Hyperliquid perps | 1D | Cross-sectional L/S, multi-asset relative-value | Live |
| 2 | **Session-Filtered Directional (Intraday)** | Crypto perpetuals | ~220 Hyperliquid perps | 15m | Directional, session-conditioned | Research |
| 3 | **Pairwise Vol-Spread (Intraday)** | Crypto perpetuals | ~220 Hyperliquid perps | 1m | Pairwise statistical arbitrage / trend continuation | Research |

---

### 1. Cross-Sectional Lead-Lag — Daily

A multi-asset long/short strategy operating on the full Hyperliquid perpetuals universe. Positions are rebalanced daily from a cross-sectional ranking extracted from the asset return matrix over a rolling lookback. Capital is allocated across a basket of followers with position weights driven by the signal strength.

- **Universe:** ~220 Hyperliquid perpetuals, filtered on data availability each cycle
- **Approach:** cross-sectional relative-value, portfolio rebuilt every cycle
- **Risk:** spread-vol targeting, rolling signal-reliability score, drawdown circuit breaker, intraday stop-loss
- **Live:** deployed through the Hyperliquid REST API, scheduled execution, market/ALO routing

**Backtest — Feb 2025 → Apr 2026 (14 months, full walk-forward, fees & spread modelled):**
| Metric | Value |
|---|---|
| Total return ($10 000 start) | **+172 %** ($10k → $27k) |
| Sharpe — 1-year window (annualised, daily) | **1.69** |
| Max drawdown | -41 % |
| Cycle win rate | 49.6 % (191 / 385 cycles) |
| Profit factor (cycle-level) | 1.26 |
| Total fees paid | $8 448 |
| Distinct assets traded | 123 |
| Trading cycles | 385 |

![Backtest equity curve](https://raw.githubusercontent.com/mateofrqt/Crypto-LeadLag-Strategy/main/results/backtest_default.png)

---

### 2. Session-Filtered Directional — 15-Minute

A directional strategy on crypto perps restricted to a narrow intraday session. The model is fitted by pooled OLS with HC3 robust standard errors on a pre-built regression dataset; the backtest executes one trade per active bar with a fixed-time exit.

- **Universe:** scheduled leader/follower pairs drawn from the ~220 Hyperliquid perps, resampled to 15-minute bars
- **Approach:** directional, conditioned on session and signal magnitude
- **Validation:** Monte-Carlo bootstrap on trade-level PnL (10 000 × n bootstrap, null-PF distribution)

**Signal validation — OLS regression (HC3 robust SE, n = 436):** the lead-lag coefficient is statistically significant at the 5 % level (p = 0.014), confirming a reliable predictive relationship between leader and follower returns within the session window. R² is intentionally low — the edge is narrow and the strategy is sized accordingly.

**Backtest — Jan 2026 → Mar 2026 (fees 0.09 % round-trip):**
| Metric | Value |
|---|---|
| Trades | 58 |
| Win rate | ~50 % |
| **Profit factor** | **1.89** |
| Total return | **+9.40 %** |
| Sharpe (annualised) | **2.51** |
| Max drawdown | ~ -3 % |
| Bootstrap p-value (PF under H₀) | **3.49 %** |

The out-of-sample window (9 trades) returned flat-to-negative, consistent with the small sample size — larger OOS horizons are required before any live deployment.

---

### 3. Pairwise Vol-Spread — 1-Minute

Intraday strategy trading pre-scheduled leader/lagger pairs at 1-minute resolution. Both entry and exit are dynamic and volatility-driven.

- **Universe:** dynamically rotating leader/lagger pairs drawn from the ~220 Hyperliquid perps (pair schedule rebuilt offline)
- **Approach:** pairwise statistical arbitrage with directional continuation
- **Optimisation:** Optuna hyperparameter search, objective evaluated on a held-out window

**Backtest — IS / OOS split (fees 0.09 % round-trip):**
| Window | Trades | Return | Max DD |
|---|---|---|---|
| In-sample | 24 | **+14.54 %** | -0.81 % |
| Out-of-sample | 31 | **+6.78 %** | -1.99 % |

Further robustness testing is ongoing before any live deployment.

---

## Tech Stack

**Language:** Python 3.11
**Core libs:** `pandas`, `numpy`, `scipy`, `networkx`, `statsmodels`, `matplotlib`
**Optimisation:** `optuna` (TPE sampler, MedianPruner)
**Exchange / data:** Hyperliquid REST & WS APIs (`hyperliquid-python-sdk`, `eth_account`)

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

---

## About Me

Quant / systematic trader with a focus on crypto perpetuals market microstructure, cross-sectional signals and live execution infrastructure. I build strategies end-to-end: research, backtest, risk framework, live deployment.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/mateofrqt/)

---

*This repository is a showcase. It does not constitute investment advice, nor a solicitation, nor a guarantee of future returns. Past backtest and live performance is not indicative of future results.*
