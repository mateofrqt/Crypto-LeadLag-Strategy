#!/usr/bin/env python3
"""
=============================================================================
HYPERLIQUID MULTI-TIMEFRAME DATA SCRAPER V2.1 (REST + WEBSOCKET)
=============================================================================
Correctifs V2.1 (données non fraîches en REST horaire) :

  PROBLÈME : en lançant le cron à XX:00 avec end_ms = now - 120s,
  la dernière bougie (ex: 13:00) n'est souvent pas encore indexée
  par le L1 → 0 données. À XX+1:00, on récupère 2h d'un coup.

  FIX : alignement de end_ms sur la FRONTIÈRE DE BOUGIE :
    1. safe_now = now - offset (2 min, inchangé)
    2. end_ms   = floor(safe_now, candle_interval) - 1ms
    → On ne demande JAMAIS la bougie la plus récente non scellée
    → Chaque run récupère exactement les bougies fermées depuis le dernier run

  RECOMMANDATION CRON : lancer à XX:03 ou XX:05 (pas XX:00)
  pour laisser le temps à l'indexeur de sceller.

Améliorations héritées de V2 :

1. FINALIZATION OFFSET (REST)
   - Paramétrable via CANDLE_FINALIZATION_OFFSET_SECONDS (défaut : 120s).

2. MODE WEBSOCKET (--websocket)
   - Abonnement au stream {"type":"candle",...} pour chaque coin/timeframe.
   - Détecte la clôture via changement de timestamp open (t) + champ T (close ms).
   - Backfill REST initial automatique avant de passer en mode live.
   - Reconnexion automatique sur coupure WS.
   - Flush CSV périodique.

Architecture :
  - Mode REST (défaut) : async + alignement frontière de bougie
  - Mode WS            : backfill REST + WebSocket live avec buffer CSV

Usage :
    python importHL_multi_v2.py                          # REST, tous TF
    python importHL_multi_v2.py -t 1h                    # REST, 1h seulement
    python importHL_multi_v2.py --websocket -t 1h        # WS live, 1h
    python importHL_multi_v2.py --websocket -t 1h -c BTC ETH  # WS ciblé
    python importHL_multi_v2.py --offset 300             # offset 5 min
=============================================================================
"""

import asyncio
import aiohttp
import functools
import random
import json
import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd
from hyperliquid.info import Info


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration centralisée du scraper"""

    # Dossiers
    USER_HOME = Path.home()
    BASE_FOLDER = USER_HOME / "Desktop" / "data"
    DATA_FOLDER = BASE_FOLDER / "data"
    LOG_FOLDER = BASE_FOLDER / "logs"

    # Timeframes supportés avec leurs paramètres
    TIMEFRAMES = {
        "5m": {
            "interval": "5m",
            "minutes_per_candle": 5,
            "default_lookback_days": 30,
            "chunk_candles": 5000,
            "filename": "Hyperliquid_ALL_COINS_5m.csv"
        },
        "15m": {
            "interval": "15m",
            "minutes_per_candle": 15,
            "default_lookback_days": 60,
            "chunk_candles": 5000,
            "filename": "Hyperliquid_ALL_COINS_15m.csv"
        },
        "1h": {
            "interval": "1h",
            "minutes_per_candle": 60,
            "default_lookback_days": 180,
            "chunk_candles": 5000,
            "filename": "Hyperliquid_ALL_COINS_1h.csv"
        },
        "1d": {
            "interval": "1d",
            "minutes_per_candle": 1440,
            "default_lookback_days": 365,
            "chunk_candles": 5000,
            "filename": "Hyperliquid_ALL_COINS_1d.csv"
        }
    }

    # API endpoints
    INFO_ENDPOINT = "https://api.hyperliquid.xyz/info"
    WS_ENDPOINT   = "wss://api.hyperliquid.xyz/ws"

    # --- Finalization offset (REST) ---
    # L'indexeur L1 peut prendre ~2-3 min pour sceller une bougie après sa
    # clôture théorique (batch indexing).
    #
    # IMPORTANT : l'offset seul ne suffit pas ! Le vrai problème est que
    # end_ms doit être ALIGNÉ sur la frontière de la dernière bougie
    # COMPLÈTEMENT FERMÉE, pas simplement "now - offset".
    #
    # Exemple avec 1h et offset=120s lancé à 14:00:00 UTC :
    #   Ancien : end_ms = 14:00:00 - 2min = 13:58:00 → la bougie 13:00
    #            n'est pas encore indexée → 0 données
    #   Nouveau: end_ms = floor(14:00:00 - 3min, 1h) = 13:00:00 - 1ms
    #            = borne de la bougie 12:00 (dernière complète) → OK
    #            Puis si la bougie 13:00 est déjà indexée, on la prend aussi
    #
    # En pratique on :
    #   1. Calcule now - offset pour être sûr de ne pas demander l'ouverte
    #   2. Floor au timeframe pour aligner sur une frontière de bougie
    #   3. Ajoute le candle_ms pour inclure la dernière bougie complète
    #
    # RECOMMANDATION CRON : lancer à XX:03 (ou XX:05) au lieu de XX:00
    # pour laisser ~3 min à l'indexeur L1 de sceller la bougie.
    CANDLE_FINALIZATION_OFFSET_SECONDS = 120  # 2 minutes

    # Coins inactifs : ignorer si pas de données depuis N bougies
    MAX_INACTIVE_CANDLES = 48

    # --- Rate limiting async (REST) ---
    SEMAPHORE_LIMIT   = 3
    REQUEST_SPACING   = 0.5    # délai minimal tenu dans le slot semaphore
    TIMEFRAME_STAGGER = 3.0    # écart entre le démarrage de chaque timeframe
    MAX_RETRIES       = 4
    RETRY_DELAY       = 2.0
    RETRY_DELAY_429   = 8.0
    REQUEST_TIMEOUT   = 30

    # --- WebSocket ---
    WS_RECONNECT_DELAY       = 5.0   # secondes avant reconnexion
    WS_PING_INTERVAL         = 20    # keepalive ping toutes les N secondes
    WS_BATCH_WRITE_INTERVAL  = 30    # flush buffer CSV toutes les N secondes


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_folder: Path) -> logging.Logger:
    """Configure le système de logging"""
    log_folder.mkdir(parents=True, exist_ok=True)

    log_file = log_folder / f"scraper_{datetime.now().strftime('%Y%m%d')}.log"

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger("HyperliquidScraper")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def dt_to_ms(dt: datetime) -> int:
    """Convertit un datetime en timestamp milliseconds"""
    return int(dt.timestamp() * 1000)


def ms_to_dt(ms: int) -> datetime:
    """Convertit un timestamp ms en datetime UTC"""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def get_all_coins() -> List[str]:
    """Récupère la liste de tous les coins disponibles sur Hyperliquid"""
    try:
        info = Info()
        asset_ctxs = info.meta()["universe"]
        coins = [ctx["name"] for ctx in asset_ctxs if "name" in ctx]
        return sorted(coins)
    except Exception as e:
        logging.error(f"Erreur récupération liste coins: {e}")
        return []


def process_candles(candles: List[dict], coin: str) -> pd.DataFrame:
    """Traite les données brutes REST en DataFrame propre"""
    df = pd.DataFrame(candles)

    df['timestamp_ms'] = df['t'].astype(int)
    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)

    for col in ['o', 'h', 'l', 'c', 'v']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.rename(columns={
        'o': 'open', 'h': 'high', 'l': 'low',
        'c': 'close', 'v': 'volume', 'n': 'trades'
    }, inplace=True)

    df['coin'] = coin

    cols = ['coin', 'datetime', 'timestamp_ms', 'open', 'high', 'low', 'close', 'volume', 'trades']
    available_cols = [c for c in cols if c in df.columns]

    return df[available_cols].drop_duplicates(subset=['timestamp_ms'])


def candle_ws_to_row(data: dict, coin: str) -> dict:
    """
    Convertit un message candle WebSocket en ligne DataFrame.

    Format WS Hyperliquid :
      { "t": <open_ms>, "o": "...", "h": "...", "l": "...",
        "c": "...", "v": "...", "n": <trades> }
    """
    ts_ms = int(data['t'])
    return {
        'coin':         coin,
        'datetime':     ms_to_dt(ts_ms),
        'timestamp_ms': ts_ms,
        'open':         float(data.get('o', 0)),
        'high':         float(data.get('h', 0)),
        'low':          float(data.get('l', 0)),
        'close':        float(data.get('c', 0)),
        'volume':       float(data.get('v', 0)),
        'trades':       int(data.get('n', 0)),
    }


# =============================================================================
# ASYNC REST FETCHING
# =============================================================================

class RateLimitError(Exception):
    """Levée sur HTTP 429 — permet de relâcher le semaphore avant d'attendre"""
    def __init__(self, wait: float):
        self.wait = wait


async def fetch_chunk(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    logger: logging.Logger
) -> List[dict]:
    """Récupère un chunk REST avec semaphore et retry automatique"""
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms
        }
    }
    timeout = aiohttp.ClientTimeout(total=Config.REQUEST_TIMEOUT)

    rate_limited = False
    for attempt in range(Config.MAX_RETRIES):
        try:
            async with semaphore:
                async with session.post(
                    Config.INFO_ENDPOINT, json=payload, timeout=timeout
                ) as resp:
                    if resp.status == 429:
                        retry_after = float(resp.headers.get('Retry-After', 0))
                        base_wait = retry_after if retry_after > 0 else Config.RETRY_DELAY_429 * (attempt + 1)
                        jitter = random.uniform(0, base_wait * 0.5)
                        wait = base_wait + jitter
                        rate_limited = True
                        raise RateLimitError(wait)
                    rate_limited = False
                    resp.raise_for_status()
                    data = await resp.json()
                    await asyncio.sleep(Config.REQUEST_SPACING)
                    return data if isinstance(data, list) else []
        except RateLimitError as e:
            logger.debug(f"429 {coin} (tentative {attempt+1}/{Config.MAX_RETRIES}), attente {e.wait:.0f}s...")
            await asyncio.sleep(e.wait)
        except Exception as e:
            if attempt < Config.MAX_RETRIES - 1:
                await asyncio.sleep(Config.RETRY_DELAY * (attempt + 1))
            else:
                logger.warning(f"Échec fetch {coin} après {Config.MAX_RETRIES} tentatives: {e}")

    if rate_limited:
        logger.warning(f"429 {coin} — {Config.MAX_RETRIES} tentatives épuisées, données perdues")
    return []


async def fetch_coin_candles(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    chunk_candles: int,
    minutes_per_candle: int,
    logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """Récupère toutes les bougies d'un coin (chunks parallèles via semaphore)"""
    chunk_size_ms = chunk_candles * minutes_per_candle * 60 * 1000

    chunk_tasks = []
    current_start = start_ms
    while current_start < end_ms:
        chunk_end = min(current_start + chunk_size_ms, end_ms)
        chunk_tasks.append(
            fetch_chunk(session, semaphore, coin, interval, current_start, chunk_end, logger)
        )
        current_start = chunk_end

    if not chunk_tasks:
        return None

    chunk_results = await asyncio.gather(*chunk_tasks)

    all_candles = []
    for result in chunk_results:
        if result:
            all_candles.extend(result)

    if not all_candles:
        return None

    return process_candles(all_candles, coin)


# =============================================================================
# REST SCRAPER (avec finalization offset)
# =============================================================================

class MultiTimeframeScraper:
    """
    Scraper REST — identique à v1 mais avec finalization offset.

    L'offset est soustrait de now_ms pour ne jamais demander la bougie
    en cours de finalisation (typiquement la dernière bougie de l'heure).
    """

    def __init__(self, logger: logging.Logger, offset_seconds: int = Config.CANDLE_FINALIZATION_OFFSET_SECONDS):
        self.logger = logger
        self.offset_ms = offset_seconds * 1000
        Config.DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        timeframes: Optional[List[str]] = None,
        coins: Optional[List[str]] = None
    ) -> Dict[str, dict]:
        """Point d'entrée synchrone"""
        if timeframes is None:
            timeframes = list(Config.TIMEFRAMES.keys())
        else:
            invalid = [tf for tf in timeframes if tf not in Config.TIMEFRAMES]
            if invalid:
                raise ValueError(f"Timeframes invalides: {invalid}")

        if coins is None:
            coins = get_all_coins()
            if not coins:
                raise RuntimeError("Impossible de récupérer la liste des coins")

        self.logger.info(f"{'='*60}")
        self.logger.info("DÉMARRAGE SCRAPING MULTI-TIMEFRAME (REST + OFFSET)")
        self.logger.info(f"Timeframes : {timeframes}")
        self.logger.info(f"Coins      : {len(coins)}")
        self.logger.info(f"Concurrence: {Config.SEMAPHORE_LIMIT} req simultanées")
        self.logger.info(f"Offset     : {self.offset_ms // 1000}s (finalization lag)")
        self.logger.info(f"{'='*60}")

        return asyncio.run(self._run_async(timeframes, coins))

    async def _run_async(
        self,
        timeframes: List[str],
        coins: List[str]
    ) -> Dict[str, dict]:
        semaphore = asyncio.Semaphore(Config.SEMAPHORE_LIMIT)
        connector = aiohttp.TCPConnector(limit=Config.SEMAPHORE_LIMIT + 10)

        async with aiohttp.ClientSession(
            connector=connector,
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'}
        ) as session:
            async def _delayed_tf(delay: float, tf: str):
                if delay > 0:
                    await asyncio.sleep(delay)
                return await self._process_timeframe(session, semaphore, tf, coins)

            tf_tasks = [
                _delayed_tf(i * Config.TIMEFRAME_STAGGER, tf)
                for i, tf in enumerate(timeframes)
            ]
            tf_results = await asyncio.gather(*tf_tasks, return_exceptions=True)

        results = {}
        for tf, result in zip(timeframes, tf_results):
            if isinstance(result, Exception):
                self.logger.error(f"Erreur timeframe {tf}: {result}")
                results[tf] = {"error": str(result)}
            else:
                results[tf] = result
        return results

    async def _process_timeframe(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        timeframe: str,
        coins: List[str]
    ) -> dict:
        tf_config = Config.TIMEFRAMES[timeframe]
        filepath = Config.DATA_FOLDER / tf_config["filename"]

        loop = asyncio.get_running_loop()
        df_master, last_times = await loop.run_in_executor(
            None, self._load_existing_data, filepath
        )

        now_ms = dt_to_ms(datetime.now(timezone.utc))
        candle_ms   = tf_config["minutes_per_candle"] * 60 * 1000

        # --- FIX V2.1 : alignement sur frontière de bougie ---
        # IMPORTANT sur la convention Hyperliquid :
        #   L'API retourne t = open_ms. Mais pour les bougies 1h, le timestamp
        #   'T' (close) de la doc WS montre que t=13:00 signifie la bougie
        #   qui COUVRE 12:00-13:00 (close=13:00), PAS 13:00-14:00.
        #   Autrement dit, t = close time, pas open time, pour les snapshots REST.
        #
        # Vérification avec les données BSV : à 13:36 UTC, BSV a t=13:00
        #   → c'est la bougie 12:00-13:00 (fermée à 13:00). Normal.
        #   BTC n'a que t=12:00 → il lui manque la bougie 12:00-13:00 (t=13:00).
        #
        # Donc candle_boundary doit être : floor(now - offset, candle) + candle
        # pour inclure la bougie qui vient de se fermer.
        # Ex: 13:34 UTC → floor = 13:00, boundary = 14:00
        #     → on garde open_t < 14:00 → inclut 13:00 ✓
        candle_boundary_ms = ((now_ms - self.offset_ms) // candle_ms) * candle_ms + candle_ms
        safe_end_ms = now_ms  # on laisse l'API retourner tout

        cutoff_ms   = now_ms - candle_ms * Config.MAX_INACTIVE_CANDLES

        updates = []
        for coin in coins:
            if coin in last_times:
                last_ms = int(last_times[coin])
                if last_ms < cutoff_ms:
                    continue

                # --- FIX V2.1 : protection contre les bougies incomplètes ---
                # Si la dernière bougie en base tombe DANS ou APRÈS la bougie
                # en cours (= pas encore scellée au moment du fetch précédent),
                # on la re-fetch pour avoir les données complètes.
                # Ex: last_ms=12:00, boundary=13:00 → start=12:00 (re-fetch 12:00)
                # Ex: last_ms=11:00, boundary=13:00 → start=12:00 (normal)
                if last_ms >= candle_boundary_ms - candle_ms:
                    # La dernière bougie pourrait être incomplète → re-fetch
                    start_ms = last_ms
                else:
                    start_ms = last_ms + candle_ms
            else:
                start_dt = datetime.now(timezone.utc) - timedelta(days=tf_config["default_lookback_days"])
                start_ms = dt_to_ms(start_dt)

            if start_ms < candle_boundary_ms:
                updates.append((coin, start_ms))

        if not updates:
            self.logger.info(f"✅ {timeframe}: Déjà à jour")
            return {"status": "up_to_date", "coins_updated": 0, "candles_added": 0}

        self.logger.info(f"📥 {timeframe}: {len(updates)} coins à mettre à jour (end < {ms_to_dt(candle_boundary_ms).strftime('%H:%M')} UTC)...")

        coin_tasks = [
            fetch_coin_candles(
                session, semaphore,
                coin, tf_config["interval"],
                start_ms, safe_end_ms,
                tf_config["chunk_candles"], tf_config["minutes_per_candle"],
                self.logger
            )
            for coin, start_ms in updates
        ]
        coin_results = await asyncio.gather(*coin_tasks, return_exceptions=True)

        new_dfs = []
        failed_updates = []

        for (coin, start_ms), result in zip(updates, coin_results):
            if isinstance(result, Exception):
                self.logger.debug(f"  ✗ {coin}: {result}")
                failed_updates.append((coin, start_ms))
            elif result is not None and len(result) > 0:
                # --- FILTRE CLIENT : exclure la bougie en cours (pas fermée) ---
                result = result[result['timestamp_ms'] < candle_boundary_ms]
                if len(result) > 0:
                    new_dfs.append(result)
                    self.logger.debug(f"  ✓ {coin}: +{len(result)} bougies")
                else:
                    self.logger.debug(f"  ? {coin}: aucune bougie fermée")
            else:
                self.logger.debug(f"  ? {coin}: aucune donnée")

        # Retry pass
        if failed_updates:
            self.logger.info(f"  ↩  {timeframe}: {len(failed_updates)} coins échoués, retry dans 3s...")
            await asyncio.sleep(3.0)

            retry_tasks = [
                fetch_coin_candles(
                    session, semaphore,
                    coin, tf_config["interval"],
                    start_ms, safe_end_ms,
                    tf_config["chunk_candles"], tf_config["minutes_per_candle"],
                    self.logger
                )
                for coin, start_ms in failed_updates
            ]
            retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)

            retry_ok = errors = 0
            for (coin, _), result in zip(failed_updates, retry_results):
                if isinstance(result, Exception):
                    errors += 1
                    self.logger.warning(f"  ✗ {coin}: échec définitif: {result}")
                elif result is not None and len(result) > 0:
                    # --- FILTRE CLIENT : exclure la bougie en cours ---
                    result = result[result['timestamp_ms'] < candle_boundary_ms]
                    if len(result) > 0:
                        new_dfs.append(result)
                        retry_ok += 1
                    else:
                        errors += 1
                else:
                    errors += 1
                    self.logger.warning(f"  ✗ {coin}: aucune donnée après retry")

            if retry_ok:
                self.logger.info(f"  ↩  {timeframe}: {retry_ok}/{len(failed_updates)} récupérés au retry")
        else:
            errors = 0

        if not new_dfs:
            self.logger.info(f"⚠️  {timeframe}: Aucune nouvelle donnée ({errors} erreurs)")
            return {"status": "no_new_data", "coins_updated": 0, "candles_added": 0, "errors": errors}

        df_updates = pd.concat(new_dfs, ignore_index=True)
        candles_added = len(df_updates)

        df_final = pd.concat([df_master, df_updates], ignore_index=True)
        df_final.drop_duplicates(subset=['coin', 'timestamp_ms'], keep='last', inplace=True)
        df_final['datetime'] = pd.to_datetime(df_final['datetime'], utc=True)
        df_final.sort_values(by=['coin', 'datetime'], inplace=True)

        save_fn = functools.partial(df_final.to_csv, filepath, index=False)
        await loop.run_in_executor(None, save_fn)

        err_str = f" | {errors} erreurs" if errors else ""
        self.logger.info(f"✅ {timeframe}: +{candles_added} bougies | {len(new_dfs)} coins | Total: {len(df_final)}{err_str}")

        return {
            "status": "updated",
            "coins_updated": len(new_dfs),
            "candles_added": candles_added,
            "total_candles": len(df_final),
            "errors": errors
        }

    def _load_existing_data(self, filepath: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
        if not filepath.exists():
            self.logger.info(f"  📄 Nouveau fichier: {filepath.name}")
            return pd.DataFrame(), {}

        try:
            df = pd.read_csv(filepath)
            if 'timestamp_ms' not in df.columns and 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['timestamp_ms'] = df['datetime'].astype('int64') // 10**6
            last_times = df.groupby('coin')['timestamp_ms'].max().to_dict()
            self.logger.info(f"  📂 {filepath.name}: {len(df)} lignes, {len(last_times)} coins")
            return df, last_times
        except Exception as e:
            self.logger.error(f"Erreur lecture {filepath}: {e}")
            return pd.DataFrame(), {}


# =============================================================================
# WEBSOCKET SCRAPER (backfill REST + live candle stream)
# =============================================================================

class WebSocketScraper:
    """
    Scraper WebSocket — plus réactif que le polling REST.

    Fonctionnement :
      1. Backfill REST : rattrapage historique via fetch_coin_candles
      2. Abonnement WS : subscribe {"type":"candle",...} pour chaque coin
      3. Détection de clôture : quand un nouveau timestamp open (t) arrive,
         la bougie précédente est définitivement fermée
      4. Buffer + flush CSV périodique (WS_BATCH_WRITE_INTERVAL secondes)
      5. Reconnexion automatique sur coupure WS

    Limitation : une connexion WS par appel → pour un grand nombre de coins,
    les abonnements sont envoyés séquentiellement au démarrage.
    """

    def __init__(self, logger: logging.Logger, offset_seconds: int = Config.CANDLE_FINALIZATION_OFFSET_SECONDS):
        self.logger = logger
        self.offset_ms = offset_seconds * 1000
        Config.DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        timeframes: Optional[List[str]] = None,
        coins: Optional[List[str]] = None
    ):
        if timeframes is None:
            timeframes = ["1h"]  # défaut WS : 1h (le plus critique)
        else:
            invalid = [tf for tf in timeframes if tf not in Config.TIMEFRAMES]
            if invalid:
                raise ValueError(f"Timeframes invalides: {invalid}")

        if coins is None:
            coins = get_all_coins()
            if not coins:
                raise RuntimeError("Impossible de récupérer la liste des coins")

        self.logger.info(f"{'='*60}")
        self.logger.info("DÉMARRAGE SCRAPING WEBSOCKET LIVE")
        self.logger.info(f"Timeframes : {timeframes}")
        self.logger.info(f"Coins      : {len(coins)}")
        self.logger.info(f"Offset     : {self.offset_ms // 1000}s")
        self.logger.info(f"Flush CSV  : toutes les {Config.WS_BATCH_WRITE_INTERVAL}s")
        self.logger.info(f"{'='*60}")

        asyncio.run(self._run_async(timeframes, coins))

    async def _run_async(self, timeframes: List[str], coins: List[str]):
        # Backfill REST pour chaque timeframe avant de passer en WS
        self.logger.info("📥 Phase 1 : backfill REST...")
        rest_scraper = MultiTimeframeScraper(self.logger, self.offset_ms // 1000)
        # On réutilise la boucle asyncio courante via _run_async directement
        semaphore = asyncio.Semaphore(Config.SEMAPHORE_LIMIT)
        connector = aiohttp.TCPConnector(limit=Config.SEMAPHORE_LIMIT + 10)

        async with aiohttp.ClientSession(
            connector=connector,
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'}
        ) as rest_session:
            async def _delayed_tf(delay: float, tf: str):
                if delay > 0:
                    await asyncio.sleep(delay)
                return await rest_scraper._process_timeframe(rest_session, semaphore, tf, coins)

            tf_tasks = [_delayed_tf(i * Config.TIMEFRAME_STAGGER, tf) for i, tf in enumerate(timeframes)]
            await asyncio.gather(*tf_tasks, return_exceptions=True)

        self.logger.info("✅ Backfill REST terminé — passage en mode WebSocket live")
        self.logger.info("   (Ctrl+C pour arrêter)")

        # Lancer un watcher WS par timeframe en parallèle
        ws_tasks = [
            self._watch_timeframe(timeframe, coins)
            for timeframe in timeframes
        ]
        await asyncio.gather(*ws_tasks)

    async def _watch_timeframe(self, timeframe: str, coins: List[str]):
        """Boucle principale WS avec reconnexion automatique"""
        tf_config = Config.TIMEFRAMES[timeframe]
        filepath = Config.DATA_FOLDER / tf_config["filename"]

        while True:
            try:
                await self._ws_session(timeframe, tf_config, coins, filepath)
            except asyncio.CancelledError:
                self.logger.info(f"[WS {timeframe}] Arrêt demandé")
                return
            except Exception as e:
                self.logger.warning(
                    f"[WS {timeframe}] Connexion perdue ({e}), "
                    f"reconnexion dans {Config.WS_RECONNECT_DELAY}s..."
                )
                await asyncio.sleep(Config.WS_RECONNECT_DELAY)

    async def _ws_session(
        self,
        timeframe: str,
        tf_config: dict,
        coins: List[str],
        filepath: Path
    ):
        """
        Session WebSocket unique pour un timeframe.

        - Souscrit à tous les coins
        - Maintient un buffer des bougies en cours (pending_candles)
        - Détecte les clôtures via changement de timestamp open (t)
        - Flush vers CSV via un timer périodique
        """
        interval = tf_config["interval"]

        # buffer : dernière donnée WS reçue par coin
        pending_candles: Dict[str, dict] = {}
        # buffer des bougies FERMÉES prêtes à être écrites
        closed_buffer: List[dict] = []
        last_flush = asyncio.get_event_loop().time()

        connector = aiohttp.TCPConnector()
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.ws_connect(
                Config.WS_ENDPOINT,
                heartbeat=Config.WS_PING_INTERVAL,
                receive_timeout=60.0
            ) as ws:
                self.logger.info(f"[WS {timeframe}] Connexion établie — abonnement de {len(coins)} coins...")

                # Envoyer toutes les subscriptions
                for coin in coins:
                    await ws.send_json({
                        "method": "subscribe",
                        "subscription": {
                            "type": "candle",
                            "coin": coin,
                            "interval": interval
                        }
                    })
                    # Petit espacage pour ne pas flooder le handshake
                    await asyncio.sleep(0.05)

                self.logger.info(f"[WS {timeframe}] Abonnements envoyés — écoute active...")

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            envelope = json.loads(msg.data)
                        except json.JSONDecodeError:
                            continue

                        channel = envelope.get("channel", "")
                        if channel != "candle":
                            continue

                        data = envelope.get("data", {})
                        # Le champ 's' (symbol) est présent dans les messages candle WS
                        coin = data.get("s")
                        if not coin or 't' not in data:
                            continue

                        new_t = int(data['t'])

                        # --- FIX V2.1 : utiliser le champ 'T' (close millis) ---
                        # D'après la doc WS, Candle a un champ T = close millis.
                        # Si T < now, la bougie est fermée. Combiné avec le
                        # changement de timestamp open, on a une double vérification.

                        if coin in pending_candles:
                            old_t = int(pending_candles[coin]['t'])
                            if new_t != old_t:
                                # Nouveau timestamp open → l'ancienne bougie est fermée
                                closed_row = candle_ws_to_row(pending_candles[coin], coin)
                                closed_buffer.append(closed_row)
                                self.logger.debug(
                                    f"[WS {timeframe}] {coin} bougie fermée "
                                    f"@ {ms_to_dt(old_t).strftime('%Y-%m-%d %H:%M')} UTC"
                                )
                            else:
                                # Même bougie, mise à jour des valeurs OHLCV en cours
                                pass

                        pending_candles[coin] = data

                        # Flush périodique
                        now = asyncio.get_event_loop().time()
                        if closed_buffer and (now - last_flush) >= Config.WS_BATCH_WRITE_INTERVAL:
                            await self._flush_buffer(closed_buffer, filepath)
                            self.logger.info(
                                f"[WS {timeframe}] Flush : {len(closed_buffer)} bougies écrites"
                            )
                            closed_buffer.clear()
                            last_flush = now

                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        self.logger.warning(f"[WS {timeframe}] Message WS inattendu: {msg.type}")
                        break

        # Flush final si session terminée proprement
        if closed_buffer:
            await self._flush_buffer(closed_buffer, filepath)
            self.logger.info(f"[WS {timeframe}] Flush final : {len(closed_buffer)} bougies")

    async def _flush_buffer(self, rows: List[dict], filepath: Path):
        """Écrit les bougies fermées dans le CSV (merge + dedup)"""
        loop = asyncio.get_running_loop()
        df_new = pd.DataFrame(rows)
        df_new['datetime'] = pd.to_datetime(df_new['datetime'], utc=True)

        def _write():
            if filepath.exists():
                df_existing = pd.read_csv(filepath)
                if 'timestamp_ms' not in df_existing.columns and 'datetime' in df_existing.columns:
                    df_existing['datetime'] = pd.to_datetime(df_existing['datetime'])
                    df_existing['timestamp_ms'] = df_existing['datetime'].astype('int64') // 10**6
                df_merged = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_merged = df_new.copy()

            df_merged.drop_duplicates(subset=['coin', 'timestamp_ms'], keep='last', inplace=True)
            df_merged['datetime'] = pd.to_datetime(df_merged['datetime'], utc=True)
            df_merged.sort_values(by=['coin', 'datetime'], inplace=True)
            df_merged.to_csv(filepath, index=False)

        await loop.run_in_executor(None, _write)


# =============================================================================
# CLI & ENTRY POINT
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Hyperliquid Multi-Timeframe Data Scraper V2 (REST + WebSocket)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python importHL_multi_v2.py                           # REST, tous TF
  python importHL_multi_v2.py -t 5m 1h                  # REST, 5m + 1h
  python importHL_multi_v2.py -c BTC ETH SOL            # REST, coins ciblés
  python importHL_multi_v2.py --websocket -t 1h         # WS live, 1h
  python importHL_multi_v2.py --websocket -t 1h -c BTC  # WS live, BTC 1h
  python importHL_multi_v2.py --offset 300              # offset 5 min
        """
    )

    parser.add_argument(
        '-t', '--timeframes',
        nargs='+',
        choices=['5m', '15m', '1h', '1d'],
        help='Timeframes à scraper (défaut: tous en REST, 1h en WS)'
    )

    parser.add_argument(
        '-c', '--coins',
        nargs='+',
        help='Coins spécifiques (défaut: tous)'
    )

    parser.add_argument(
        '--websocket', '--ws',
        action='store_true',
        dest='websocket',
        help='Mode WebSocket live (backfill REST + stream temps réel)'
    )

    parser.add_argument(
        '--offset',
        type=int,
        default=Config.CANDLE_FINALIZATION_OFFSET_SECONDS,
        metavar='SECONDES',
        help=f'Offset de finalisation en secondes (défaut: {Config.CANDLE_FINALIZATION_OFFSET_SECONDS})'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Mode silencieux (logs fichier uniquement)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=Config.SEMAPHORE_LIMIT,
        help=f'Requêtes REST simultanées (défaut: {Config.SEMAPHORE_LIMIT})'
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.workers != Config.SEMAPHORE_LIMIT:
        Config.SEMAPHORE_LIMIT = args.workers

    logger = setup_logging(Config.LOG_FOLDER)

    if args.quiet:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                logger.removeHandler(handler)

    start_time = datetime.now()
    logger.info(f"\n{'='*60}")
    logger.info(f"HYPERLIQUID SCRAPER V2 - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode : {'WebSocket live' if args.websocket else 'REST'}")
    logger.info(f"{'='*60}")

    try:
        if args.websocket:
            scraper = WebSocketScraper(logger, offset_seconds=args.offset)
            scraper.run(timeframes=args.timeframes, coins=args.coins)
        else:
            scraper = MultiTimeframeScraper(logger, offset_seconds=args.offset)
            results = scraper.run(timeframes=args.timeframes, coins=args.coins)

            end_time = datetime.now()
            duration = end_time - start_time

            logger.info(f"\n{'='*60}")
            logger.info("RÉSUMÉ FINAL")
            logger.info(f"{'='*60}")

            total_candles = 0
            for tf, stats in results.items():
                if "candles_added" in stats:
                    total_candles += stats.get("candles_added", 0)
                    logger.info(f"  {tf}: +{stats['candles_added']} bougies ({stats.get('coins_updated', 0)} coins)")
                elif "error" in stats:
                    logger.info(f"  {tf}: ERREUR {stats['error']}")

            logger.info(f"{'─'*40}")
            logger.info(f"  Total : +{total_candles} nouvelles bougies")
            logger.info(f"  Durée : {str(duration).split('.')[0]}")
            logger.info(f"{'='*60}\n")

            stats_file = Config.LOG_FOLDER / "last_run_stats.json"
            with open(stats_file, 'w') as f:
                json.dump({
                    "timestamp": start_time.isoformat(),
                    "duration_seconds": duration.total_seconds(),
                    "offset_seconds": args.offset,
                    "mode": "rest",
                    "results": results
                }, f, indent=2, default=str)

        return 0

    except KeyboardInterrupt:
        logger.info("\nArrêt demandé (Ctrl+C)")
        return 0
    except Exception as e:
        logger.exception(f"ERREUR FATALE: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())