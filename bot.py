from __future__ import annotations

import os
import time
import logging
import schedule
import pandas as pd
from math import log10, floor
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info

from importHL_multi import MultiTimeframeScraper
from trading_signal import generate_live_signals


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Configuration du bot."""
    private_key: str
    account_address: str
    testnet: bool = True
    leverage: int = 1
    budget_usd: float = 0
    timeframe: str = "1h"
    rebalance_threshold: float = 0.20
    min_order_usd: float = 10.0  # Minimum Hyperliquid
    data_folder: Path = Path.home() / "Desktop" / "data" / "data"
    log_folder: Path = Path.home() / "Desktop" / "data" / "logs"

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
        self.info = Info(config.base_url, skip_ws=True)
        self.exchange = Exchange(Account.from_key(config.private_key), config.base_url)
        self._meta_cache: Optional[dict] = None

    # --- Utilitaires prix ---

    def _load_meta(self) -> None:
        """Charge les métadonnées de tous les assets."""
        if self._meta_cache is None:
            meta = self.info.meta()
            self._meta_cache = {}
            for asset in meta["universe"]:
                name = asset["name"]
                self._meta_cache[name] = {
                    "szDecimals": asset.get("szDecimals", 0),
                }

    def _get_sz_decimals(self, coin: str) -> int:
        self._load_meta()
        return self._meta_cache.get(coin, {}).get("szDecimals", 0)

    def _round_size(self, coin: str, size: float) -> float:
        return round(size, self._get_sz_decimals(coin))

    def _round_to_significant_figures(self, value: float, sig_figs: int = 5, max_decimals: int = 4) -> float:
        """Arrondit une valeur à N chiffres significatifs (règle Hyperliquid), max 4 décimales."""
        if value == 0:
            return 0.0
        magnitude = floor(log10(abs(value)))
        decimals = sig_figs - 1 - magnitude
        decimals = min(decimals, max_decimals)
        if decimals < 0:
            factor = 10 ** (-decimals)
            return round(value / factor) * factor
        else:
            return round(value, decimals)

    # --- API wrappers ---

    def get_mid_price(self, coin: str) -> Optional[float]:
        try:
            price = float(self.info.all_mids().get(coin, 0))
            return price if price > 0 else None
        except Exception as e:
            logging.error(f"Erreur prix {coin}: {e}")
            return None

    def get_spot_balance(self) -> float:
        """Récupère le solde Spot USDC."""
        try:
            spot_state = self.info.spot_user_state(self.config.account_address)
            balances = spot_state.get("balances", [])
            for bal in balances:
                if bal.get("coin") == "USDC":
                    return float(bal.get("total", 0))
            return 0.0
        except Exception as e:
            logging.debug(f"Erreur solde spot: {e}")
            return 0.0

    def get_available_balance(self) -> float:
        """Récupère le solde disponible pour trader (Perps + Spot en Unified)."""
        try:
            state = self.info.user_state(self.config.account_address)
            margin = state.get("marginSummary", {})

            perps_value = float(margin.get("accountValue", 0))
            total_margin_used = float(margin.get("totalMarginUsed", 0))
            withdrawable = float(state.get("withdrawable", 0))

            # Récupérer aussi le solde Spot (disponible en mode Unified)
            spot_balance = self.get_spot_balance()

            # withdrawable est déjà la valeur nette disponible côté perps
            # On ajoute le spot uniquement s'il n'est pas déjà inclus dans accountValue
            total_available = withdrawable + spot_balance

            logging.info(f"Compte: spot=${spot_balance:.2f}, perps=${perps_value:.2f}, marginUsed=${total_margin_used:.2f}, withdrawable=${withdrawable:.2f}, available=${total_available:.2f}")

            return total_available
        except Exception as e:
            logging.error(f"Erreur récupération solde: {e}")
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
                logging.info(f"Ordre annulé: {order['coin']}")
            except Exception as e:
                logging.warning(f"Erreur annulation: {e}")

    def close_position(self, coin: str) -> bool:
        try:
            result = self.exchange.market_close(coin)
            # Vérifier le résultat
            if isinstance(result, dict):
                status = result.get("status")
                if status == "ok":
                    logging.info(f"Position fermée: {coin}")
                    return True
                else:
                    error = result.get("response", {}).get("data", {}).get("error", result)
                    logging.error(f"Échec fermeture {coin}: {error}")
                    return False
            logging.warning(f"Résultat fermeture {coin} inattendu: {result}")
            return False
        except Exception as e:
            logging.error(f"Erreur fermeture {coin}: {e}")
            return False

    def open_position(self, coin: str, target_usd: float, is_long: bool) -> bool:
        price = self.get_mid_price(coin)
        if not price:
            logging.warning(f"Prix invalide pour {coin}")
            return False

        size = self._round_size(coin, target_usd / price)
        if size == 0:
            min_usd = price
            if min_usd <= target_usd * 2:
                size = 1
                logging.info(f"Taille ajustée pour {coin}: 1 unité")
            else:
                logging.warning(f"Taille trop petite pour {coin}: ${target_usd:.2f} / ${price:.2f} (min serait ${min_usd:.2f})")
                return False

        # Levier
        try:
            lev_result = self.exchange.update_leverage(self.config.leverage, coin, is_cross=False)
            logging.debug(f"Levier {coin}: {self.config.leverage}x -> {lev_result}")
        except Exception as e:
            logging.warning(f"Erreur levier {coin}: {e}")

        # Prix limit (arrondi à 5 chiffres significatifs - règle Hyperliquid, max 4 décimales)
        offset = 0.0005
        limit_price = price * (1 + offset) if is_long else price * (1 - offset)
        limit_price = self._round_to_significant_figures(limit_price, 5)

        # Vérifier que le prix arrondi n'est pas trop éloigné du mid (problème assets micro-prix)
        deviation = abs(limit_price / price - 1)
        if deviation > 0.01:
            logging.warning(f"{coin}: prix limit {limit_price} trop éloigné du mid {price} ({deviation*100:.1f}%), ordre ignoré")
            return False

        logging.info(f"Ordre {coin}: mid={price}, limit={limit_price}, size={size}")

        try:
            result = self.exchange.order(coin, is_long, size, limit_price, {"limit": {"tif": "Gtc"}})
            return self._log_order_result(result, coin, is_long, size, limit_price, target_usd)
        except Exception as e:
            logging.error(f"Erreur ordre {coin}: {e}")
            return False

    def adjust_position(self, coin: str, current_size: float, target_size: float) -> bool:
        price = self.get_mid_price(coin)
        if not price:
            return False

        delta = target_size - current_size
        size = self._round_size(coin, abs(delta))
        if size == 0:
            return True

        is_buy = delta > 0
        offset = 0.0005
        limit_price = price * (1 + offset) if is_buy else price * (1 - offset)
        limit_price = self._round_to_significant_figures(limit_price, 5)

        try:
            result = self.exchange.order(coin, is_buy, size, limit_price, {"limit": {"tif": "Gtc"}})
            action = "Augmentation" if delta > 0 else "Réduction"
            if self._log_order_result(result, coin, is_buy, size, limit_price, abs(delta) * price):
                logging.info(f"{action} {coin}: {current_size} -> {target_size}")
                return True
            return False
        except Exception as e:
            logging.error(f"Erreur ajustement {coin}: {e}")
            return False

    def _log_order_result(self, result: dict, coin: str, is_buy: bool, size: float, price: float, usd: float) -> bool:
        direction = "LONG" if is_buy else "SHORT"
        status = result.get("status") if isinstance(result, dict) else None
        data = result.get("response", {}).get("data", {}) if isinstance(result, dict) else {}

        if status != "ok":
            logging.error(f"Ordre rejeté {coin}: {data.get('error', result)}")
            return False

        statuses = data.get("statuses", [{}])
        first = statuses[0] if statuses else {}

        if "error" in first:
            logging.error(f"Ordre rejeté {coin}: {first['error']}")
            return False
        if "filled" in first:
            logging.info(f"Ordre rempli: {coin} | {direction} | ${usd:.2f} | size: {size}")
            return True
        if "resting" in first:
            logging.info(f"Ordre ouvert: {coin} | {direction} | ${usd:.2f} | size: {size} | prix: {price:.4f}")
            return True

        logging.warning(f"Statut inconnu {coin}: {statuses}")
        return True

    # --- Pipeline ---

    def _get_timeframe_minutes(self) -> int:
        """Retourne la durée d'une bougie en minutes selon la timeframe."""
        tf_minutes = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        return tf_minutes.get(self.config.timeframe, 60)

    def is_data_fresh(self, max_candles_age: int = 48) -> bool:
        """
        Vérifie si les données sont fraîches.

        Args:
            max_candles_age: Nombre max de bougies de retard autorisé pour considérer
                            un coin comme actif (défaut: 48 bougies)
        """
        if not self.config.csv_path.exists():
            return False
        try:
            df = pd.read_csv(self.config.csv_path)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

            now = datetime.now(timezone.utc)
            candle_minutes = self._get_timeframe_minutes()
            cutoff = now - timedelta(minutes=candle_minutes * max_candles_age)

            # Filtrer les coins actifs (données récentes) avant de calculer le min
            latest_per_coin = df.groupby("coin")["datetime"].max()
            active_coins = latest_per_coin[latest_per_coin > cutoff]

            if active_coins.empty:
                logging.warning("Aucun coin actif trouvé")
                return False

            # Utiliser le 10e percentile pour ignorer les quelques coins en retard
            p10 = active_coins.quantile(0.10)
            age_candles = (now - p10.to_pydatetime()).total_seconds() / 60 / candle_minutes
            logging.info(f"Données: {len(active_coins)} coins actifs, p10 à {age_candles:.1f} bougies")
            return age_candles < 1
        except Exception as e:
            logging.warning(f"Erreur données: {e}")
            return False

    def run_cycle(self) -> None:
        logging.info("--- Cycle de trading ---")

        try:
            # 1. Données
            if not self.is_data_fresh():
                logging.info("Mise à jour données...")
                MultiTimeframeScraper(logging.getLogger("Scraper")).run(timeframes=[self.config.timeframe])
            else:
                logging.info("Données OK")

            # 2. Signaux
            signals = generate_live_signals(str(self.config.csv_path))
            if signals.empty:
                logging.warning("Aucun signal")
                return

            # 3. Exécution
            self._execute_trades(signals)

        except Exception as e:
            logging.error(f"Erreur cycle: {e}")

    def _filter_signals_by_budget(self, signals: pd.DataFrame, budget: float, positions: dict) -> pd.DataFrame:
        """Filtre les signaux pour ne garder que ceux qu'on peut trader avec le budget."""
        if signals.empty:
            return signals

        min_order = self.config.min_order_usd
        leverage = self.config.leverage

        # Calculer combien de nouvelles positions on peut ouvrir
        # Les positions existantes dans les cibles ne comptent pas dans le budget nécessaire
        target_coins = set(signals["ticker"])
        existing_in_target = sum(1 for coin in positions if coin in target_coins)
        new_positions_needed = len(target_coins) - existing_in_target

        if new_positions_needed == 0:
            return signals  # On a déjà toutes les positions

        # Budget par nouvelle position
        budget_per_new = budget / new_positions_needed if new_positions_needed > 0 else 0

        if budget_per_new < min_order / leverage:
            # Pas assez de budget pour toutes les positions, on filtre
            max_new_positions = int(budget * leverage / min_order)
            if max_new_positions == 0:
                logging.warning(f"Budget insuffisant: ${budget:.2f} < ${min_order:.2f} minimum")
                return pd.DataFrame()

            logging.info(f"Budget limité: {max_new_positions} positions max (au lieu de {new_positions_needed})")

            # Garder les positions existantes + les meilleurs nouveaux signaux
            existing_signals = signals[signals["ticker"].isin(positions.keys())]
            new_signals = signals[~signals["ticker"].isin(positions.keys())]

            # Trier par poids décroissant et prendre les meilleurs
            new_signals = new_signals.sort_values("weight", ascending=False).head(max_new_positions)
            signals = pd.concat([existing_signals, new_signals], ignore_index=True)

        return signals

    def _execute_trades(self, signals: pd.DataFrame) -> None:
        # Annuler ordres en attente
        if orders := self.get_open_orders():
            logging.info(f"Annulation ordres: {[o['coin'] for o in orders]}")
            self.cancel_all_orders()
            time.sleep(0.5)

        positions = self.get_positions()

        # Récupérer le solde disponible (avec buffer conservateur de 90%)
        time.sleep(0.5)
        available = self.get_available_balance() * 0.95
        budget = min(available, self.config.budget_usd) if self.config.budget_usd > 0 else available
        logging.info(f"Budget utilisé: ${budget:.2f} (95% du disponible: ${available:.2f})")

        # Filtrer les signaux selon le budget disponible
        signals = self._filter_signals_by_budget(signals, budget, positions)
        if signals.empty:
            logging.warning("Aucun signal après filtrage budget")
            return

        target_coins = set(signals["ticker"])

        logging.info(f"Positions: {list(positions.keys())}")
        logging.info(f"Cibles: {list(target_coins)}")

        # Fermer positions obsolètes
        for coin in positions:
            if coin not in target_coins:
                logging.info(f"Fermeture obsolète: {coin}")
                self.close_position(coin)

        if budget <= 0:
            logging.warning("Aucun budget disponible")
            return

        # Normaliser les poids si leur somme dépasse 1 (évite sur-allocation)
        total_weight = signals["weight"].sum()
        if total_weight > 1.0:
            logging.warning(f"Poids totaux = {total_weight:.3f} > 1.0, normalisation appliquée")
            signals = signals.copy()
            signals["weight"] = signals["weight"] / total_weight

        # Ouvrir/ajuster
        remaining_budget = budget
        for _, row in signals.iterrows():
            coin = row["ticker"]
            price = self.get_mid_price(coin)
            if not price:
                logging.warning(f"{coin} non tradable")
                continue

            is_long = row["direction"] == "LONG"
            target_usd = budget * row["weight"] * self.config.leverage
            current = positions.get(coin, 0)

            if current == 0:
                if target_usd > remaining_budget + 0.01:
                    logging.warning(f"Budget restant insuffisant pour {coin}: ${target_usd:.2f} > ${remaining_budget:.2f}")
                    continue
                if self.open_position(coin, target_usd, is_long):
                    remaining_budget -= target_usd / self.config.leverage
            elif (current > 0) != is_long:
                logging.info(f"Flip {coin}")
                self.close_position(coin)
                time.sleep(0.5)
                if target_usd <= remaining_budget + 0.01:
                    if self.open_position(coin, target_usd, is_long):
                        remaining_budget -= target_usd / self.config.leverage
                else:
                    logging.warning(f"Budget restant insuffisant pour flip {coin}: ${target_usd:.2f} > ${remaining_budget:.2f}")
            else:
                current_usd = abs(current) * price
                diff = abs(current_usd - target_usd) / target_usd if target_usd else 0
                if diff > self.config.rebalance_threshold:
                    target_size = (target_usd / price) if is_long else -(target_usd / price)
                    logging.info(f"Ajustement {coin}: ${current_usd:.2f} -> ${target_usd:.2f}")
                    self.adjust_position(coin, current, target_size)
                else:
                    logging.info(f"Position {coin} OK ({diff*100:.1f}%)")


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
    # Dériver l'adresse automatiquement de la clé privée
    account_address = Account.from_key(private_key).address

    config = Config(
        private_key=private_key,
        account_address=account_address,
        testnet=True
    )

    setup_logging(config.log_folder)

    logging.info("=" * 50)
    logging.info(f"BOT HYPERLIQUID - {'TESTNET' if config.testnet else 'MAINNET'}")
    budget_str = f"${config.budget_usd}" if config.budget_usd > 0 else "auto (solde compte)"
    logging.info(f"Timeframe: {config.timeframe} | Budget: {budget_str} | Levier: {config.leverage}x")
    logging.info(f"Adresse: {config.account_address}")
    logging.info("=" * 50)

    bot = TradingBot(config)
    bot.run_cycle()

    schedule.every().hour.at(":02").do(bot.run_cycle)
    logging.info("Scheduler actif...")

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()