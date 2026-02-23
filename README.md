# Crypto Lead-Lag Strategy

A cryptocurrency trading bot that detects **lead-lag relationships** between assets using Lévy-area signatures and Hermitian matrix clustering, then generates systematic trading signals on the [Hyperliquid](https://hyperliquid.xyz) perpetual futures exchange.

## How It Works

Some crypto assets consistently move before others — a phenomenon known as the lead-lag effect. This project captures that signal mathematically and turns it into a portfolio strategy.

**Pipeline overview:**

```
Data Collection → Lévy Matrix → Hermitian Clustering → Signal Generation → Portfolio → Execution
(importHL)        (levy.py)     (hermitian.py)         (trading_signal)   (portfolio)  (bot.py)
```

1. **Data Collection** — Multi-timeframe OHLCV candles are scraped from Hyperliquid's API (REST + WebSocket) for all listed perpetual contracts.
2. **Lévy-Area Computation** — Pairwise lead-lag relationships are quantified using the Lévy-area signature between asset return paths. This produces an antisymmetric matrix where positive entries indicate asset *i* leads asset *j*.
3. **Hermitian Clustering** — The Lévy matrix is embedded into a Hermitian matrix and spectrally decomposed. Eigenvector clustering groups assets into leaders and laggers.
4. **Signal Generation** — Leader/lagger roles are translated into trading signals: long laggers expected to catch up, short leaders expected to revert.
5. **Portfolio Construction** — Signals are combined into a weighted portfolio using GCP (Generalized Cluster Portfolio) methodology with risk constraints.
6. **Execution** — The bot places limit orders on Hyperliquid with automated position management, drawdown monitoring, and reconnection logic.

## Project Structure

```
├── importHL_multi.py      # Multi-timeframe data scraper (REST, async)
├── csv_transformer.py     # Data preprocessing and format conversion
├── levy.py                # Lévy-area matrix computation
├── hermitian.py           # Hermitian matrix construction and spectral clustering
├── trading_signal.py      # Signal generation from cluster assignments
├── portfolio.py           # Portfolio construction (CP, GCP, GP strategies)
├── main.py                # Backtesting pipeline — orchestrates the full flow
└── bot.py                 # Live trading bot for Hyperliquid
```

## Mathematical Foundations

### Lévy Area

For two asset return paths *X* and *Y*, the Lévy area is defined as:

```
L(X, Y) = ½ ∫₀ᵀ (Xₜ dYₜ - Yₜ dXₜ)
```

If *X* leads *Y*, the integral is systematically positive. This is computed pairwise across all assets to build the lead-lag matrix.

### Hermitian Clustering

The antisymmetric Lévy matrix **A** is converted into a Hermitian matrix **H = iA** (where *i* is the imaginary unit). Spectral decomposition of **H** yields eigenvectors whose phases encode the lead-lag ordering. K-means clustering on these phases separates assets into leader and lagger groups.

### Portfolio Strategies

- **CP** (Cluster Portfolio) — Equal weight within leader/lagger clusters
- **GCP** (Generalized Cluster Portfolio) — Weights proportional to eigenvector loadings
- **GP** (Generalized Portfolio) — Full spectral weighting across all eigenvectors

## Quick Start

### Requirements

```
Python 3.10+
```

```bash
pip install pandas numpy scipy scikit-learn aiohttp hyperliquid-python-sdk
```

### 1. Collect Data

```bash
# Scrape all coins, all timeframes
python importHL_multi.py

# Or target specific timeframes / coins
python importHL_multi.py -t 1h 1d -c BTC ETH SOL
```

### 2. Run Backtest

```bash
python main.py
```

This runs the full pipeline (Lévy → Hermitian → Signals → Portfolio) on historical data and outputs performance metrics.

### 3. Live Trading

```bash
python bot.py
```

The bot will:
- Generate fresh signals from the latest data
- Place limit orders on Hyperliquid
- Monitor positions and enforce drawdown limits

## Configuration

Key parameters (in `bot.py` / `main.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| Timeframe | `1d` | Signal generation timeframe |
| Strategy | `GCP` | Portfolio construction method |
| Max Drawdown | `20%` | Kill switch threshold |
| Capital | `< $1,000` | Initial test capital |

## Data Pipeline

The scraper supports 4 timeframes with incremental collection:

| Timeframe | Lookback | Use Case |
|-----------|----------|----------|
| `5m` | 30 days | Microstructure research |
| `15m` | 60 days | Intraday signals |
| `1h` | 180 days | Medium-frequency trading |
| `1d` | 365 days | Portfolio rebalancing (primary) |

Data is stored as CSV files in `~/Desktop/data/data/`.

## Research References

This project builds on recent academic work in lead-lag detection:

- **Lévy-area approach** for lead-lag detection in financial time series using rough path signatures
- **Hermitian matrix clustering** framework for directed network analysis of asset relationships
- **DeltaLag** — deep learning approaches for lead-lag detection in cryptocurrency markets
- Foundation code inspired by [ARahimiQuant/lead-lag-portfolios](https://github.com/ARahimiQuant/lead-lag-portfolios)

## Roadmap

- [x] Data collection pipeline (multi-timeframe)
- [x] Lévy matrix computation
- [x] Hermitian spectral clustering
- [x] Backtesting framework (CP, GCP, GP)
- [x] Live trading bot (Hyperliquid)
- [x] WebSocket real-time data mode
- [ ] Multi-timeframe signal confirmation
- [ ] Higher frequency execution (1h, 15m)


## Disclaimer

This software is for **educational and research purposes only**. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## License

MIT
