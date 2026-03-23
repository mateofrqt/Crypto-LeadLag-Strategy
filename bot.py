from __future__ import annotations

import os
import time
import logging
import schedule
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from math import log10, floor
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from eth_account import Account
from hyperliquid.api import API
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info

def _build_spot_meta(base_url: str) -> dict:
    """Workaround SDK bug: spot_meta tokens list is sparse but SDK indexes it directly."""
    spot_meta = API(base_url).post("/info", {"type": "spotMeta"})
    tokens = spot_meta["tokens"]
    if tokens:
        max_idx = max(t.get("index", i) for i, t in enumerate(tokens))
        dense = [{"name": "", "szDecimals": 0, "index": i} for i in range(max_idx + 1)]
        for token in tokens:
            idx = token.get("index")
            if idx is not None:
                dense[idx] = token
        spot_meta["tokens"] = dense
    return spot_meta


from risk import LeadLagRiskManager, RiskConfig
from importHL_multi import MultiTimeframeScraper
from trading_signal import generate_live_signals


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Bot configuration."""
    private_key: str
    account_address: str
    testnet: bool = False
    leverage: int = 1
    budget_usd: float = 0
    timeframe: str = "1h"
    rebalance_threshold: float = 0.30
    min_order_usd: float = 10.0  # Hyperliquid minimum
    min_leader_move: float = 0.0005  # Min leader return to act (0.05%)
    market_mode: bool = True  # If True, all orders use market (IOC) instead of ALO limit
    data_folder: Path = Path.home() / "data" / "data"
    log_folder: Path = Path.home() / "data" / "logs"
    risk_config: Optional[RiskConfig] = None

    def __post_init__(self):
        if not isinstance(self.leverage, int) or self.leverage < 1:
            raise ValueError(f"leverage must be an integer >= 1, got: {self.leverage!r}")

    @property
    def csv_path(self) -> Path:
        return self.data_folder / f"Hyperliquid_ALL_COINS_{self.timeframe}.csv"

    @property
    def base_url(self) -> str:
        return "https://api.hyperliquid-testnet.xyz" if self.testnet else "https://api.hyperliquid.xyz"


# =============================================================================
# TRADING BOT
# =============================================================================

class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        spot_meta = _build_spot_meta(config.base_url)
        self.info = Info(config.base_url, skip_ws=True, spot_meta=spot_meta)
        main_account = Account.from_key(config.private_key)
        self.exchange = Exchange(main_account, config.base_url, spot_meta=spot_meta)
        self._meta_cache: Optional[dict] = None
        self._leverage_cache: dict[str, int] = {}
        self.risk = LeadLagRiskManager(config.risk_config) if config.risk_config else None
        self._prev_signal_meta: Optional[dict] = None
        self._account_equity: float = 0.0  # total account value for circuit breaker (not free margin)

    # --- Price utilities ---

    def _load_meta(self) -> None:
        """Load metadata for all assets."""
        if self._meta_cache is None:
            meta = self.info.meta()
            self._meta_cache = {}
            for asset in meta["universe"]:
                name = asset["name"]
                self._meta_cache[name] = {
                    "szDecimals": asset.get("szDecimals", 0),
                    "pxDecimals": asset.get("pxDecimals"),
                }

    def _get_sz_decimals(self, coin: str) -> int:
        self._load_meta()
        return self._meta_cache.get(coin, {}).get("szDecimals", 0)

    def _round_price(self, coin: str, price: float, mid: float = None) -> float:
        """Round a price to exchange precision: pxDecimals if available, else same decimal count as mid price.
        Always caps at 5 significant figures (Hyperliquid tick rule)."""
        if price == 0:
            return 0.0
        self._load_meta()
        px_dec = self._meta_cache.get(coin, {}).get("pxDecimals")
        if px_dec is not None:
            rounded = round(price, px_dec)
        else:
            # Fallback: count decimal places from mid price (same precision as exchange tick)
            ref = mid if mid is not None else price
            s = str(ref)
            if '.' in s and 'e' not in s.lower():
                decimals = len(s.split('.')[1])
            else:
                decimals = 0
            rounded = round(price, decimals)
        # Hyperliquid requires at most 5 significant figures
        return self._round_to_significant_figures(rounded, 5)

    def _round_size(self, coin: str, size: float) -> float:
        return round(size, self._get_sz_decimals(coin))

    def _fetch_l2_prices(self, coins: list[str]) -> dict[str, tuple[Optional[float], Optional[float]]]:
        """Fetch best bid/ask from L2 orderbook for multiple coins in parallel."""
        def fetch_one(coin):
            try:
                snap = self.info.l2_snapshot(coin)
                levels = snap.get("levels", [[], []])
                best_bid = float(levels[0][0]["px"]) if levels[0] else None
                best_ask = float(levels[1][0]["px"]) if levels[1] else None
                return coin, (best_bid, best_ask)
            except Exception as e:
                logging.warning(f"L2 error {coin}: {e}")
                return coin, (None, None)

        with ThreadPoolExecutor(max_workers=min(len(coins), 10)) as ex:
            return dict(ex.map(fetch_one, coins))

    def _round_to_significant_figures(self, value: float, sig_figs: int = 5) -> float:
        """Round a value to N significant figures (Hyperliquid rule)."""
        if value == 0:
            return 0.0
        magnitude = floor(log10(abs(value)))
        decimals = sig_figs - 1 - magnitude
        if decimals < 0:
            factor = 10 ** (-decimals)
            return round(value / factor) * factor
        else:
            return round(value, decimals)

    def _one_tick(self, coin: str, price: float) -> float:
        """Return smallest price increment (1 tick) for this coin."""
        self._load_meta()
        px_dec = self._meta_cache.get(coin, {}).get("pxDecimals")
        if px_dec is not None:
            return 10 ** (-px_dec)
        magnitude = floor(log10(abs(price)))
        return 10 ** (magnitude - 4)  # last sig fig at 5 sig figs

    # --- API wrappers ---

    def get_mid_price(self, coin: str) -> Optional[float]:
        try:
            price = float(self.info.all_mids().get(coin, 0))
            return price if price > 0 else None
        except Exception as e:
            logging.error(f"Price error {coin}: {e}")
            return None

    def get_spot_balance(self) -> float:
        """Get USDC spot balance."""
        try:
            spot_state = self.info.spot_user_state(self.config.account_address)
            balances = spot_state.get("balances", [])
            for bal in balances:
                if bal.get("coin") == "USDC":
                    return float(bal.get("total", 0))
            return 0.0
        except Exception as e:
            logging.debug(f"Spot balance error: {e}")
            return 0.0

    def get_available_balance(self) -> float:
        """Get available trading balance (Perps + Spot in Unified mode)."""
        for attempt in range(3):
            try:
                state = self.info.user_state(self.config.account_address)
                margin = state.get("marginSummary", {})

                perps_value = float(margin.get("accountValue", 0))
                total_margin_used = float(margin.get("totalMarginUsed", 0))
                withdrawable = float(state.get("withdrawable", 0))

                spot_balance = self.get_spot_balance()

                unrealized_pnl = perps_value - total_margin_used
                self._account_equity = spot_balance + unrealized_pnl
                self._total_margin_used = total_margin_used

                total_available = max(spot_balance - total_margin_used, 0.0)
                logging.info(f"Account: spot=${spot_balance:.2f}, perps=${perps_value:.2f}, marginUsed=${total_margin_used:.2f}, withdrawable=${withdrawable:.2f}, freeMargin=${total_available:.2f}")

                return total_available
            except Exception as e:
                if attempt < 2 and "429" in str(e):
                    logging.warning(f"Balance fetch rate limited, retrying in 5s...")
                    time.sleep(5)
                else:
                    logging.error(f"Balance fetch error: {e}")
                    return 0.0
        return 0.0

    def get_positions(self) -> dict[str, float]:
        positions = {}
        for pos in self.info.user_state(self.config.account_address).get("assetPositions", []):
            data = pos.get("position", {})
            coin, size = data.get("coin"), float(data.get("szi", 0))
            if coin and size != 0:
                positions[coin] = size
        return positions

    def get_open_orders(self) -> list[dict]:
        try:
            return self.info.open_orders(self.config.account_address)
        except Exception:
            return []

    # --- Trading ---

    def cancel_all_orders(self) -> None:
        for order in self.get_open_orders():
            try:
                self.exchange.cancel(order["coin"], order["oid"])
                logging.info(f"Order cancelled: {order['coin']}")
            except Exception as e:
                logging.warning(f"Cancel error: {e}")

    def close_position(self, coin: str) -> bool:
        try:
            result = self.exchange.market_close(coin)
            if isinstance(result, dict):
                status = result.get("status")
                if status == "ok":
                    logging.info(f"Position closed: {coin}")
                    return True
                else:
                    error = result.get("response", {}).get("data", {}).get("error", result)
                    logging.error(f"Close failed {coin}: {error}")
                    return False
            logging.warning(f"Unexpected close result {coin}: {result}")
            return False
        except Exception as e:
            logging.error(f"Close error {coin}: {e}")
            return False

    def open_position(self, coin: str, target_usd: float, is_long: bool, price: Optional[float] = None) -> bool:
        if price is None:
            price = self.get_mid_price(coin)
        if not price:
            logging.warning(f"Invalid price for {coin}")
            return False

        size = self._round_size(coin, target_usd / price)
        if size == 0:
            min_usd = price
            if min_usd <= target_usd * 2:
                size = 1
                logging.info(f"Size adjusted for {coin}: 1 unit")
            else:
                logging.warning(f"Size too small for {coin}: ${target_usd:.2f} / ${price:.2f} (min would be ${min_usd:.2f})")
                return False

        # Leverage (skipped if already set)
        if self._leverage_cache.get(coin) != self.config.leverage:
            try:
                self.exchange.update_leverage(self.config.leverage, coin, is_cross=False)
                self._leverage_cache[coin] = self.config.leverage
                logging.debug(f"Leverage {coin}: {self.config.leverage}x set")
            except Exception as e:
                logging.warning(f"Leverage error {coin}: {e}")

        offset = 0.0005
        limit_price = price * (1 - offset) if is_long else price * (1 + offset)  # maker: passive
        limit_price = self._round_price(coin, limit_price)

        # Check rounded price is not too far from mid (micro-price assets issue)
        deviation = abs(limit_price / price - 1)
        if deviation > 0.01:
            logging.warning(f"{coin}: limit price {limit_price} too far from mid {price} ({deviation*100:.1f}%), order skipped")
            return False

        logging.info(f"Order {coin}: mid={price}, limit={limit_price}, size={size}")

        try:
            result = self.exchange.order(coin, is_long, size, limit_price, {"limit": {"tif": "Gtc"}})
            return self._log_order_result(result, coin, is_long, size, limit_price, target_usd)
        except Exception as e:
            logging.error(f"Order error {coin}: {e}")
            return False

    def adjust_position(self, coin: str, current_size: float, target_size: float, price: Optional[float] = None) -> bool:
        if price is None:
            price = self.get_mid_price(coin)
        if not price:
            return False

        delta = target_size - current_size
        size = self._round_size(coin, abs(delta))
        if size == 0:
            return True

        is_buy = delta > 0
        # reduceOnly=True when reducing position: exchange requires no additional margin
        reduce_only = (is_buy and current_size < 0) or (not is_buy and current_size > 0)
        offset = 0.0005
        limit_price = price * (1 - offset) if is_buy else price * (1 + offset)  # maker: passive
        limit_price = self._round_price(coin, limit_price)

        try:
            result = self.exchange.order(coin, is_buy, size, limit_price, {"limit": {"tif": "Gtc"}}, reduce_only=reduce_only)
            action = "Increase" if delta > 0 else "Decrease"
            if self._log_order_result(result, coin, is_buy, size, limit_price, abs(delta) * price):
                logging.info(f"{action} {coin}: {current_size} -> {target_size}")
                return True
            return False
        except Exception as e:
            logging.error(f"Adjustment error {coin}: {e}")
            return False

    def _log_order_result(self, result: dict, coin: str, is_buy: bool, size: float, price: float, usd: float) -> bool:
        direction = "LONG" if is_buy else "SHORT"
        status = result.get("status") if isinstance(result, dict) else None
        response = result.get("response", {}) if isinstance(result, dict) else {}
        data = response.get("data", {}) if isinstance(response, dict) else {}

        if status != "ok":
            logging.error(f"Order rejected {coin}: {data.get('error', result)}")
            return False

        statuses = data.get("statuses", [{}])
        first = statuses[0] if statuses else {}

        if "error" in first:
            logging.error(f"Order rejected {coin}: {first['error']}")
            return False
        if "filled" in first:
            logging.info(f"Order filled: {coin} | {direction} | ${usd:.2f} | size: {size}")
            return True
        if "resting" in first:
            logging.info(f"Order resting: {coin} | {direction} | ${usd:.2f} | size: {size} | price: {price:.4f}")
            return True

        logging.warning(f"Unknown status {coin}: {statuses}")
        return True

    def _build_order_spec(self, coin: str, target_usd: float, is_long: bool, price: float,
                          best_bid: Optional[float] = None, best_ask: Optional[float] = None) -> Optional[dict]:
        """Build an ALO order spec without API call, using best bid/ask when available."""
        size = self._round_size(coin, target_usd / price)
        if size == 0:
            if price <= target_usd * 2:
                size = 1
                logging.info(f"Size adjusted for {coin}: 1 unit")
            else:
                logging.warning(f"Size too small for {coin}: ${target_usd:.2f} / ${price:.2f}")
                return None

        # L2 prices are already at exchange tick precision — use directly
        # Fallback: round mid ± 0.05% using exchange-declared pxDecimals (or 5 sig figs)
        if is_long and best_bid:
            limit_price = best_bid
        elif not is_long and best_ask:
            limit_price = best_ask
        else:
            limit_price = self._round_price(coin, price * (1 - 0.0005 if is_long else 1 + 0.0005))

        if abs(limit_price / price - 1) > 0.01:
            logging.warning(f"{coin}: limit price {limit_price} too far from mid {price}, skipped")
            return None

        logging.info(f"Order {coin}: mid={price}, limit={limit_price} ({'bid' if is_long else 'ask'}), size={size}")
        return {"coin": coin, "is_buy": is_long, "sz": size, "limit_px": limit_price,
                "order_type": {"limit": {"tif": "Alo"}}, "reduce_only": False}

    def _build_adjust_spec(self, coin: str, current_size: float, target_size: float, price: float,
                           best_bid: Optional[float] = None, best_ask: Optional[float] = None) -> Optional[dict]:
        """Build an ALO adjustment spec without API call, using best bid/ask when available."""
        delta = target_size - current_size
        size = self._round_size(coin, abs(delta))
        if size == 0:
            return None

        is_buy = delta > 0
        reduce_only = (is_buy and current_size < 0) or (not is_buy and current_size > 0)

        # L2 prices are already at exchange tick precision — use directly
        if is_buy and best_bid:
            limit_price = best_bid
        elif not is_buy and best_ask:
            limit_price = best_ask
        else:
            limit_price = self._round_price(coin, price * (1 - 0.0005 if is_buy else 1 + 0.0005))

        logging.info(f"Adjustment {coin}: {current_size} -> {target_size}")
        return {"coin": coin, "is_buy": is_buy, "sz": size, "limit_px": limit_price,
                "order_type": {"limit": {"tif": "Alo"}}, "reduce_only": reduce_only}

    def _log_bulk_result(self, result: dict, order_meta: list) -> None:
        """Log bulk_orders results."""
        if not isinstance(result, dict) or result.get("status") != "ok":
            logging.error(f"bulk_orders rejected: {result}")
            return
        statuses = result.get("response", {}).get("data", {}).get("statuses", [])
        for i, (coin, is_buy, usd) in enumerate(order_meta):
            direction = "LONG" if is_buy else "SHORT"
            s = statuses[i] if i < len(statuses) else {}
            if "error" in s:
                logging.error(f"Order rejected {coin}: {s['error']}")
            elif "filled" in s:
                logging.info(f"Order filled [TAKER]: {coin} | {direction} | ${usd:.2f}")
            elif "resting" in s:
                logging.info(f"Order resting [MAKER]: {coin} | {direction} | ${usd:.2f}")
            else:
                logging.warning(f"Unknown status {coin}: {s}")

    def _bulk_close(self, coins: list[str], positions: dict, mids_cache: dict, flip: bool = False) -> None:
        """Close multiple positions using ALO limit orders.
        flip=True: use mid price so the open (placed 1 tick away) fills after."""
        l2_prices = self._fetch_l2_prices(coins)

        close_specs, skipped = [], []
        for coin in coins:
            size = positions.get(coin, 0)
            price = mids_cache.get(coin)
            if not size or not price:
                skipped.append(coin)
                continue
            is_buy = size < 0  # closing short = buy, closing long = sell
            abs_size = self._round_size(coin, abs(size))
            bid, ask = l2_prices.get(coin, (None, None))
            if flip:
                # Flip close at mid so the open order (1 tick away) fills after
                limit_price = self._round_price(coin, price)
            else:
                # Standard maker close: buy at bid, sell at ask
                if is_buy and bid:
                    limit_price = bid
                elif not is_buy and ask:
                    limit_price = ask
                else:
                    limit_price = self._round_price(coin, price * (1.0005 if is_buy else 0.9995))
            close_specs.append({"coin": coin, "is_buy": is_buy, "sz": abs_size,
                                 "limit_px": limit_price, "order_type": {"limit": {"tif": "Alo"}},
                                 "reduce_only": True})
        for coin in skipped:  # sequential fallback if price missing
            self.close_position(coin)
        if not close_specs:
            return
        try:
            result = self.exchange.bulk_orders(close_specs)
            if isinstance(result, dict) and result.get("status") == "ok":
                statuses = result.get("response", {}).get("data", {}).get("statuses", [])
                for i, spec in enumerate(close_specs):
                    s = statuses[i] if i < len(statuses) else {}
                    if "filled" in s:
                        logging.info(f"Position closed [ALO filled]: {spec['coin']}")
                    elif "resting" in s:
                        logging.info(f"Close order resting [ALO]: {spec['coin']}")
                    elif "error" in s:
                        logging.error(f"Close failed {spec['coin']}: {s['error']}")
                        self.close_position(spec["coin"])
                    else:
                        logging.warning(f"Close status unknown {spec['coin']}: {s}, fallback to market close")
                        self.close_position(spec["coin"])
            else:
                logging.error(f"bulk_close rejected: {result}, sequential fallback")
                for coin in coins:
                    self.close_position(coin)
        except Exception as e:
            logging.error(f"bulk_close error: {e}, sequential fallback")
            for coin in coins:
                self.close_position(coin)

    def _wait_for_positions_closed(self, coins: list[str], timeout: int = 60) -> list[str]:
        """Poll positions until specified coins are closed or timeout. Returns confirmed-closed coins."""
        if not coins:
            return []
        deadline = time.time() + timeout
        pending = set(coins)
        closed = []
        while pending and time.time() < deadline:
            time.sleep(2)
            try:
                current_positions = self.get_positions()
                newly_closed = [c for c in pending if c not in current_positions]
                if newly_closed:
                    closed.extend(newly_closed)
                    pending -= set(newly_closed)
                    logging.info(f"Closes confirmed: {newly_closed}")
            except Exception as e:
                logging.warning(f"Position check error: {e}")
        if pending:
            logging.warning(f"Closes not confirmed within {timeout}s: {list(pending)}, open skipped")
        return closed

    # --- Pipeline ---

    def _get_timeframe_minutes(self) -> int:
        """Return candle duration in minutes for the given timeframe."""
        tf_minutes = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        return tf_minutes.get(self.config.timeframe, 60)

    def is_data_fresh(self, max_candles_age: int = 48) -> bool:
        """
        Check whether data is fresh.

        Args:
            max_candles_age: Max number of candles of lag allowed to consider
                            a coin active (default: 48 candles)
        """
        if not self.config.csv_path.exists():
            return False
        try:
            df = pd.read_csv(self.config.csv_path)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

            now = datetime.now(timezone.utc)
            candle_minutes = self._get_timeframe_minutes()
            cutoff = now - timedelta(minutes=candle_minutes * max_candles_age)

            # Filter active coins (recent data) before computing the min
            latest_per_coin = df.groupby("coin")["datetime"].max()
            active_coins = latest_per_coin[latest_per_coin > cutoff]

            if active_coins.empty:
                logging.warning("No active coin found")
                return False

            # Use 10th percentile to ignore a few lagging coins
            p10 = active_coins.quantile(0.10)
            age_candles = (now - p10.to_pydatetime()).total_seconds() / 60 / candle_minutes
            logging.info(f"Data: {len(active_coins)} active coins, p10 at {age_candles:.1f} candles")
            return age_candles < 2
        except Exception as e:
            logging.warning(f"Data error: {e}")
            return False

    def run_cycle(self) -> None:
        logging.info("--- Trading cycle ---")

        try:
            # 1. Data
            if not self.is_data_fresh():
                logging.info("Updating data...")
                MultiTimeframeScraper(logging.getLogger("Scraper")).run(timeframes=[self.config.timeframe])
            else:
                logging.info("Data OK")

            # 2. Signals
            signals, signal_meta = generate_live_signals(str(self.config.csv_path))
            if signals.empty:
                logging.warning("No signal")
                return

            # 3. Leader move filter
            if signal_meta is not None and "leader_ret" in signal_meta:
                leader_ret = signal_meta["leader_ret"]
                if abs(leader_ret) < self.config.min_leader_move:
                    logging.info(f"Leader move too small ({leader_ret:.4%} < {self.config.min_leader_move:.4%}), skipping cycle")
                    self._prev_signal_meta = signal_meta
                    return

            # 4. Feedback signal reliability (cycle T+1)
            if self.risk is not None and self._prev_signal_meta is not None and signal_meta is not None:
                prev = self._prev_signal_meta
                current_returns = signal_meta["last_returns"]
                available = [f for f in prev["followers"] if f in current_returns.index]
                if available:
                    follower_ret = float(current_returns[available].mean())
                    market_ret = float(current_returns.mean())
                    self.risk.record_signal_outcome(prev["leader_ret"], follower_ret, market_ret)

            self._prev_signal_meta = signal_meta

            # 5. Execution
            self._execute_trades(signals)

            # 6. Reliability score
            if self.risk is not None:
                logging.info(
                    f"Reliability: alpha={self.risk.signal_reliability:.2%} "
                    f"beta={self.risk.signal_beta:.2%} "
                    f"(scalar={self.risk.reliability_scalar:.3f}, "
                    f"n={self.risk._reliability.n_obs})"
                )

        except Exception as e:
            logging.error(f"Cycle error: {e}")

    def _filter_signals_by_budget(self, signals: pd.DataFrame, budget: float, positions: dict) -> pd.DataFrame:
        """Filter signals to keep only those tradeable within budget."""
        if signals.empty:
            return signals

        min_order = self.config.min_order_usd
        leverage = self.config.leverage

        # Count how many new positions we can open
        # Existing positions in targets don't count toward required budget
        target_coins = set(signals["ticker"])
        existing_in_target = sum(1 for coin in positions if coin in target_coins)
        new_positions_needed = len(target_coins) - existing_in_target

        if new_positions_needed == 0:
            return signals  # All positions already held

        # Budget per new position
        budget_per_new = budget / new_positions_needed if new_positions_needed > 0 else 0

        if budget_per_new < min_order / leverage:
            # Not enough budget for all positions, filter
            max_new_positions = int(budget * leverage / min_order)
            if max_new_positions == 0:
                logging.warning(f"Insufficient budget: ${budget:.2f} < ${min_order:.2f} minimum")
                return pd.DataFrame()

            logging.info(f"Limited budget: {max_new_positions} positions max (instead of {new_positions_needed})")

            # Keep existing positions + best new signals
            existing_signals = signals[signals["ticker"].isin(positions.keys())]
            new_signals = signals[~signals["ticker"].isin(positions.keys())]

            # Sort by descending weight and take the best
            new_signals = new_signals.sort_values("weight", ascending=False).head(max_new_positions)
            signals = pd.concat([existing_signals, new_signals], ignore_index=True)

        return signals

    def _execute_trades_market(self, signals: pd.DataFrame) -> None:
        """Market mode: all orders use IOC (taker). No ALO, no flip/spread issues."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_positions = executor.submit(self.get_positions)
            f_balance   = executor.submit(self.get_available_balance)
            f_mids      = executor.submit(self.info.all_mids)
            positions     = f_positions.result()
            available_raw = f_balance.result()
            raw_mids      = f_mids.result()

        mids_cache = {k: float(v) for k, v in raw_mids.items() if float(v) > 0}
        self.cancel_all_orders()

        free_margin = available_raw * 0.98  # cash available for new positions
        # Budget for sizing is total equity (not just free margin)
        equity = self._account_equity
        budget = min(equity * 0.98, self.config.budget_usd) if self.config.budget_usd > 0 else equity * 0.98
        remaining_budget = free_margin
        logging.info(f"Budget (sizing): ${budget:.2f} | Free margin: ${free_margin:.2f}")

        vol_scalar = 1.0
        if self.risk is not None:
            vol_scalar, circuit_breaker = self.risk.update(self._account_equity)
            logging.info(f"Risk: scalar={vol_scalar:.3f}, circuit_breaker={circuit_breaker}")
            if circuit_breaker:
                logging.warning("Circuit breaker — closing all positions")
                for coin in list(positions.keys()):
                    self.close_position(coin)
                return

        target_coins = set(signals["ticker"])
        logging.info(f"Positions: {list(positions.keys())}")
        logging.info(f"Targets: {list(target_coins)}")
        total_weight = signals["weight"].sum()
        if total_weight > 1.0:
            signals = signals.copy()
            signals["weight"] = signals["weight"] / total_weight

        # Close stale positions
        stale = [c for c in positions if c not in target_coins]
        if stale:
            logging.info(f"Stale closes: {stale}")
        for coin in stale:
            price = mids_cache.get(coin, 0)
            self.close_position(coin)
            remaining_budget += abs(positions.get(coin, 0)) * price * 0.98

        # Open / flip positions
        for _, row in signals.iterrows():
            coin = row["ticker"]
            price = mids_cache.get(coin)
            if not price:
                logging.warning(f"{coin} not tradable")
                continue
            is_long = row["direction"] == "LONG"
            target_usd = budget * row["weight"] * self.config.leverage * vol_scalar
            current = positions.get(coin, 0)
            current_usd = abs(current) * price if current else 0
            logging.info(f"{coin}: target=${target_usd:.2f} current=${current_usd:.2f} ({'LONG' if is_long else 'SHORT'})")
            if target_usd < self.config.min_order_usd:
                logging.warning(f"{coin}: order too small ${target_usd:.2f} < min ${self.config.min_order_usd:.0f}, skipped")
                continue

            # Close opposite position first
            if current != 0 and (current > 0) != is_long:
                logging.info(f"Flip {coin}: closing ${current_usd:.2f} {'LONG' if current > 0 else 'SHORT'}")
                self.close_position(coin)
                remaining_budget += current_usd * 0.98
            elif current != 0:
                diff = abs(current_usd - target_usd) / target_usd if target_usd else 0
                if diff <= self.config.rebalance_threshold:
                    logging.info(f"Position {coin} OK (${current_usd:.2f} vs target ${target_usd:.2f}, {diff*100:.1f}% diff)")
                    continue
                if current_usd > target_usd:
                    # Oversized: partial reduce only — no margin needed, one trade instead of two
                    delta_usd = current_usd - target_usd
                    reduce_size = self._round_size(coin, delta_usd / price)
                    if reduce_size > 0:
                        is_buy_to_reduce = current < 0  # SHORT: buy to reduce; LONG: sell to reduce
                        slippage = 1.03 if is_buy_to_reduce else 0.97
                        lp = self._round_price(coin, price * slippage, price)
                        logging.info(f"Reducing {coin}: ${current_usd:.2f} -> ${target_usd:.2f} (partial ${delta_usd:.2f})")
                        try:
                            result = self.exchange.order(coin, is_buy_to_reduce, reduce_size, lp, {"limit": {"tif": "Ioc"}}, reduce_only=True)
                            self._log_order_result(result, coin, is_buy_to_reduce, reduce_size, lp, delta_usd)
                            remaining_budget += delta_usd * 0.98  # freed margin available for new positions
                        except Exception as e:
                            logging.error(f"Reduce error {coin}: {e}")
                    continue  # position already adjusted, skip reopen

            # Set leverage
            if self._leverage_cache.get(coin) != self.config.leverage:
                try:
                    self.exchange.update_leverage(self.config.leverage, coin, is_cross=False)
                    self._leverage_cache[coin] = self.config.leverage
                except Exception as e:
                    logging.warning(f"Leverage error {coin}: {e}")

            # Market open: use IOC with 3% slippage to guarantee fill
            if remaining_budget <= 0:
                logging.warning(f"{coin}: no free margin left, skipped")
                continue
            # For new positions: full target_usd. For increases: only the delta to avoid doubling up.
            same_dir = current != 0 and (current > 0) == is_long
            add_usd = (target_usd - current_usd) if same_dir else target_usd
            order_usd = min(add_usd, remaining_budget)
            size = self._round_size(coin, order_usd / price)
            if size == 0:
                continue
            slippage = 1.03 if is_long else 0.97
            limit_price = self._round_price(coin, price * slippage, price)
            logging.info(f"Order {coin}: mid={price}, limit={limit_price}, size={size}")
            try:
                result = self.exchange.order(coin, is_long, size, limit_price, {"limit": {"tif": "Ioc"}})
                self._log_order_result(result, coin, is_long, size, limit_price, order_usd)
                remaining_budget -= order_usd
            except Exception as e:
                logging.error(f"[MARKET] Order error {coin}: {e}")

    def _execute_trades(self, signals: pd.DataFrame) -> None:
        if self.config.market_mode:
            self._execute_trades_market(signals)
            return

        # 1. Preload prices + positions + balance + orders in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            f_orders    = executor.submit(self.get_open_orders)
            f_positions = executor.submit(self.get_positions)
            f_balance   = executor.submit(self.get_available_balance)
            f_mids      = executor.submit(self.info.all_mids)
            orders        = f_orders.result()
            positions     = f_positions.result()
            available_raw = f_balance.result()
            raw_mids      = f_mids.result()

        mids_cache = {k: float(v) for k, v in raw_mids.items() if float(v) > 0}

        # 2. Cancel entry orders only — keep reduce_only close orders alive (they free margin, not consume it)
        entry_orders = [o for o in orders if not o.get("reduceOnly", False)]
        if entry_orders:
            logging.info(f"Cancelling entry orders: {[o['coin'] for o in entry_orders]}")
            for o in entry_orders:
                try:
                    self.exchange.cancel(o["coin"], o["oid"])
                    logging.info(f"Order cancelled: {o['coin']}")
                except Exception as e:
                    logging.warning(f"Cancel error {o['coin']}: {e}")
            # Refresh balance: margin reserved by cancelled orders is now freed
            available_raw = self.get_available_balance()

        available = available_raw * 0.98
        budget = min(available, self.config.budget_usd) if self.config.budget_usd > 0 else available
        logging.info(f"Budget used: ${budget:.2f} (98% of available: ${available:.2f})")

        # Risk management — use total account equity, not free margin (free margin = 0 with open positions)
        vol_scalar = 1.0
        if self.risk is not None:
            vol_scalar, circuit_breaker = self.risk.update(self._account_equity)
            logging.info(f"Risk: scalar={vol_scalar:.3f}, circuit_breaker={circuit_breaker}")
            if circuit_breaker:
                logging.warning("Circuit breaker active — closing all positions")
                for coin in list(positions.keys()):
                    self.close_position(coin)
                return

        # Filter signals by available budget
        signals = self._filter_signals_by_budget(signals, budget, positions)
        if signals.empty:
            logging.warning("No signal after budget filtering")
            return

        target_coins = set(signals["ticker"])

        logging.info(f"Positions: {list(positions.keys())}")
        logging.info(f"Targets: {list(target_coins)}")

        # 3. Close stale positions (skip coins that already have a resting close order)
        pending_close_coins = {o["coin"] for o in orders if o.get("reduceOnly", False)}
        to_close = [coin for coin in positions if coin not in target_coins and coin not in pending_close_coins]
        if to_close:
            logging.info(f"Stale closes: {to_close}")
            self._bulk_close(to_close, positions, mids_cache)
        if pending_close_coins:
            logging.info(f"Close orders already resting: {list(pending_close_coins)}")

        if budget <= 0:
            logging.warning("No budget available")
            return

        # Normalize weights if sum exceeds 1 (prevents over-allocation)
        total_weight = signals["weight"].sum()
        if total_weight > 1.0:
            logging.warning(f"Total weights = {total_weight:.3f} > 1.0, normalization applied")
            signals = signals.copy()
            signals["weight"] = signals["weight"] / total_weight

        # 4. Prepare actions then execute in parallel
        actions = []
        remaining_budget = budget

        for _, row in signals.iterrows():
            coin = row["ticker"]
            price = mids_cache.get(coin)
            if not price:
                logging.warning(f"{coin} not tradable")
                continue

            is_long = row["direction"] == "LONG"
            target_usd = budget * row["weight"] * self.config.leverage * vol_scalar
            current = positions.get(coin, 0)

            if current == 0:
                if target_usd < self.config.min_order_usd:
                    logging.warning(f"{coin}: order too small ${target_usd:.2f} < min ${self.config.min_order_usd:.0f}, skipped")
                    continue
                if target_usd > remaining_budget + 0.01:
                    logging.warning(f"Remaining budget insufficient for {coin}: ${target_usd:.2f} > ${remaining_budget:.2f}")
                    continue
                actions.append(("open", coin, target_usd, is_long, price))
                remaining_budget -= target_usd / self.config.leverage
            elif (current > 0) != is_long:
                logging.info(f"Flip {coin}")
                can_open = target_usd <= remaining_budget + 0.01 and target_usd >= self.config.min_order_usd
                actions.append(("flip", coin, target_usd, is_long, price, can_open))
                if can_open:
                    remaining_budget -= target_usd / self.config.leverage
                else:
                    logging.warning(f"Flip {coin}: close only (${target_usd:.2f} < min or insufficient budget)")
            else:
                current_usd = abs(current) * price
                diff = abs(current_usd - target_usd) / target_usd if target_usd else 0
                if diff > self.config.rebalance_threshold:
                    target_size = (target_usd / price) if is_long else -(target_usd / price)
                    logging.info(f"Adjustment {coin}: ${current_usd:.2f} -> ${target_usd:.2f}")
                    actions.append(("adjust", coin, current, target_size, price))
                else:
                    logging.info(f"Position {coin} OK ({diff*100:.1f}%)")

        # Phase 1: bulk flip closes at mid — open will be placed at mid ± 1 tick (fills after)
        flip_coins = [action[1] for action in actions if action[0] == "flip"]
        if flip_coins:
            self._bulk_close(flip_coins, positions, mids_cache, flip=True)

        if not actions:
            return

        # Phase 2: leverage updates (slow API, cached — done before L2 fetch to minimize staleness)
        for action in actions:
            kind = action[0]
            if kind == "open":
                coin = action[1]
            elif kind == "flip":
                coin, can_open = action[1], action[5]
                if not can_open:
                    continue
            else:
                continue  # adjust: leverage already set
            if self._leverage_cache.get(coin) != self.config.leverage:
                try:
                    self.exchange.update_leverage(self.config.leverage, coin, is_cross=False)
                    self._leverage_cache[coin] = self.config.leverage
                except Exception as e:
                    logging.warning(f"Leverage error {coin}: {e}")

        # Phase 3: fetch L2 right before building specs (minimizes staleness at submission)
        coins_to_trade = list({action[1] for action in actions})
        l2_prices = self._fetch_l2_prices(coins_to_trade)

        # Phase 4: build order specs then submit immediately
        order_specs: list[dict] = []
        order_meta: list[tuple] = []  # (coin, is_buy, usd)

        for action in actions:
            kind = action[0]
            spec = None
            if kind == "open":
                _, coin, target_usd, is_long, price = action
                bid, ask = l2_prices.get(coin, (None, None))
                spec = self._build_order_spec(coin, target_usd, is_long, price, bid, ask)
                if spec:
                    order_meta.append((coin, is_long, target_usd))
                else:
                    logging.warning(f"{coin}: open order skipped (spec rejected)")
            elif kind == "flip":
                _, coin, target_usd, is_long, price, can_open = action
                if can_open:
                    # Open 1 tick away from mid so close (at mid) fills first
                    tick = self._one_tick(coin, price)
                    raw = price - tick if is_long else price + tick
                    flip_price = self._round_price(coin, raw)
                    spec = self._build_order_spec(coin, target_usd, is_long, flip_price, None, None)
                    if spec:
                        order_meta.append((coin, is_long, target_usd))
                    else:
                        logging.warning(f"{coin}: flip order skipped (spec rejected)")
            elif kind == "adjust":
                _, coin, current, target_size, price = action
                bid, ask = l2_prices.get(coin, (None, None))
                spec = self._build_adjust_spec(coin, current, target_size, price, bid, ask)
                if spec:
                    order_meta.append((coin, spec["is_buy"], abs(target_size - current) * price))
            if spec:
                order_specs.append(spec)

        if not order_specs:
            return

        # Phase 5: submit all orders in 1 single API call (retry once on 429)
        for attempt in range(2):
            try:
                result = self.exchange.bulk_orders(order_specs)
                self._log_bulk_result(result, order_meta)
                break
            except Exception as e:
                if attempt == 0 and "429" in str(e):
                    logging.warning(f"bulk_orders rate limited (429), retrying in 5s...")
                    time.sleep(5)
                else:
                    logging.error(f"bulk_orders error: {e}")
                    return

        # Phase 6: verify orders entered, retry rejected ones after 60s (margin may be freed by then)
        time.sleep(3)
        try:
            final_positions = self.get_positions()
            open_orders = self.get_open_orders()
            resting_coins = {o["coin"] for o in open_orders if not o.get("reduceOnly")}
            rejected = []
            for coin, is_buy, usd in order_meta:
                direction = "LONG" if is_buy else "SHORT"
                if coin in final_positions:
                    logging.info(f"Position confirmed: {coin} | {direction} | size={final_positions[coin]}")
                elif coin in resting_coins:
                    logging.info(f"Order resting (waiting to fill): {coin} | {direction}")
                else:
                    logging.warning(f"Order not found for {coin}: retrying in 60s")
                    rejected.append((coin, is_buy, usd))

            if rejected:
                time.sleep(60)
                mids = {k: float(v) for k, v in self.info.all_mids().items() if float(v) > 0}
                retry_specs, retry_meta = [], []
                for coin, is_long, target_usd in rejected:
                    price = mids.get(coin)
                    if not price:
                        continue
                    size = self._round_size(coin, target_usd / price)
                    if size == 0:
                        continue
                    bid, ask = None, None
                    try:
                        snap = self.info.l2_snapshot(coin)
                        levels = snap.get("levels", [[], []])
                        bid = float(levels[0][0]["px"]) if levels[0] else None
                        ask = float(levels[1][0]["px"]) if levels[1] else None
                    except Exception:
                        pass
                    spec = self._build_order_spec(coin, target_usd, is_long, price, bid, ask)
                    if spec:
                        retry_specs.append(spec)
                        retry_meta.append((coin, is_long, target_usd))
                if retry_specs:
                    logging.info(f"Retrying {[m[0] for m in retry_meta]}...")
                    try:
                        result = self.exchange.bulk_orders(retry_specs)
                        self._log_bulk_result(result, retry_meta)
                    except Exception as e:
                        logging.error(f"Retry bulk_orders error: {e}")
        except Exception as e:
            logging.warning(f"Verification error: {e}")


# =============================================================================
# MAIN
# =============================================================================

def setup_logging(log_folder: Path) -> None:
    log_folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_folder / "bot.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )


def main() -> None:
    private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY", "XXX")
    account_address: str = Account.from_key(private_key).address

    config = Config(
        private_key=private_key,
        account_address=account_address,
        testnet=False,
        risk_config=RiskConfig(timeframe="1h"),
    )

    setup_logging(config.log_folder)

    logging.info("=" * 50)
    logging.info(f"BOT HYPERLIQUID - {'TESTNET' if config.testnet else 'MAINNET'}")
    budget_str = f"${config.budget_usd}" if config.budget_usd > 0 else "auto (account balance)"
    logging.info(f"Timeframe: {config.timeframe} | Budget: {budget_str} | Leverage: {config.leverage}x")
    logging.info(f"Address: {config.account_address}")
    logging.info("=" * 50)

    bot = TradingBot(config)
    bot.run_cycle()

    for m in ["00:32"]:
        schedule.every().hour.at(m).do(bot.run_cycle)

    logging.info("Scheduler running...")

    while True:
        schedule.run_pending()
        time.sleep(5)

if __name__ == "__main__":
    main()
