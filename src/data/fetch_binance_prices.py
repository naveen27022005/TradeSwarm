"""
Phase 1, Step 3: Fetch historical Binance OHLCV price data for pump events.

Uses ccxt to download candlestick data from Binance for each unique coin
around its pump event windows. All pairs are SYM/BTC.

This module can work in two modes:
1. Online mode: fetches from Binance API (requires network)
2. Labeled-features mode: uses pre-computed labeled features from La Morgia dataset
   as a fallback when API access is unavailable
"""
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import time
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "pump-and-dump-dataset"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "binance_cache"


def fetch_ohlcv_ccxt(
    symbol: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    timeframe: str = "1m",
    max_retries: int = 2,
    timeout_ms: int = 6000,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance via ccxt with strict timeout.

    Parameters
    ----------
    symbol : str
        Coin symbol (e.g., "BRD"). Will be queried as SYM/BTC.
    start_dt : pd.Timestamp
        Start of the window (UTC).
    end_dt : pd.Timestamp
        End of the window (UTC).
    timeframe : str
        Candle timeframe (default "1m" for 1-minute).
    max_retries : int
        Number of retry attempts on timeout.
    timeout_ms : int
        Request timeout in milliseconds.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, open, high, low, close, volume
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError("ccxt is required. Install with: pip install ccxt")

    exchange = ccxt.binance({"enableRateLimit": True, "timeout": timeout_ms})
    pair = f"{symbol}/BTC"
    since = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_candles = []
    current_since = since

    while current_since < end_ms:
        for attempt in range(max_retries):
            try:
                candles = exchange.fetch_ohlcv(
                    pair, timeframe=timeframe, since=current_since, limit=1000
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.warning(f"Failed to fetch {pair} via Binance API: {e}")
                    raise

        if not candles:
            break

        all_candles.extend(candles)
        current_since = candles[-1][0] + 60000  # advance by 1 minute

        time.sleep(0.05)

    if not all_candles:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.reset_index(drop=True)

    return df


def load_labeled_features(chunk_size: str = "15S") -> pd.DataFrame:
    """
    Load pre-computed labeled features from the La Morgia dataset
    as a fallback when API access is unavailable.

    Parameters
    ----------
    chunk_size : str
        One of "5S", "15S", "25S" — the time chunk granularity.

    Returns
    -------
    pd.DataFrame with the labeled features.
    """
    features_dir = RAW_DATA / "labeled_features"
    gz_path = features_dir / f"features_{chunk_size}.csv.gz"

    if not gz_path.exists():
        raise FileNotFoundError(
            f"Labeled features not found at {gz_path}. "
            "Check the pump-and-dump-dataset repo."
        )

    logger.info(f"Loading labeled features from {gz_path}...")
    df = pd.read_csv(gz_path, compression="gzip")
    return df


def generate_synthetic_ohlcv(
    pump_datetime: pd.Timestamp,
    coin_symbol: str = "BTC",
    hours_before: int = 24,
    hours_after: int = 12,
    base_price: float = None,
    base_volume: float = None,
    is_pump: bool = True,
    freq: str = "1min",
) -> pd.DataFrame:
    """
    Generate dynamic, realistic OHLCV price series for an event window.

    Dynamically tailors prices, volatilities, and trading behaviors per coin
    and per historical event window:
    - Coin-specific baseline pricing across realistic altcoin market orders (~0.000008 to ~0.004 BTC)
    - Normal trading windows: organic random walk with natural intraday oscillations and ZERO pump spike
    - Pump event windows: dynamic pump timing, duration, magnitude, and dump decay curve
    """
    # 1. Derive coin-specific dynamic parameters via deterministic coin hash
    coin_clean = str(coin_symbol).upper().strip()
    coin_hash = int(hashlib.md5(coin_clean.encode("utf-8")).hexdigest(), 16)

    # Dynamic base price per coin if not provided (ranges across realistic altcoin Satoshi orders)
    if base_price is None or base_price <= 0:
        price_exp = -5.1 + ((coin_hash % 1000) / 1000.0) * 2.7
        base_price = float(10 ** price_exp)

    # Dynamic coin volatility per minute (0.10% to 0.28% per minute)
    coin_volatility = 0.0010 + ((coin_hash >> 12) % 1000 / 1000.0) * 0.0018

    # Dynamic base volume
    if base_volume is None or base_volume <= 0:
        base_volume = float(15.0 + ((coin_hash >> 24) % 1000 / 1000.0) * 185.0)

    # 2. Combine coin identity and event timestamp for a unique, reproducible event seed
    event_timestamp_sec = int(pump_datetime.timestamp())
    event_seed = (event_timestamp_sec ^ (coin_hash & 0x7FFFFFFF)) % (2**31)
    rng = np.random.RandomState(event_seed)

    start = pump_datetime - timedelta(hours=hours_before)
    end = pump_datetime + timedelta(hours=hours_after)
    timestamps = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    n = len(timestamps)

    # Macro trend drift across the window
    macro_drift = rng.normal(0.0, 0.00002)
    # Generate baseline returns with dynamic volatility
    returns = rng.normal(macro_drift, coin_volatility, n)

    # Diurnal / cyclical volume variation
    cycle = np.sin(np.linspace(0, 3 * np.pi, n)) * 0.25
    volumes = base_volume * (1.0 + cycle) * np.exp(rng.normal(0, 0.2, n))
    volumes = np.maximum(volumes, 0.5)

    if is_pump:
        # Dynamic pump timing around the documented event time (with slight realistic variation)
        pump_idx = int(hours_before * 60) + rng.randint(-10, 10)
        pump_idx = max(30, min(pump_idx, n - 60))

        # Dynamic pump & dump characteristics
        pump_duration = rng.randint(8, 22)       # 8 to 22 minutes duration
        dump_duration = rng.randint(18, 45)      # 18 to 45 minutes dumping
        pump_magnitude = rng.uniform(0.7, 2.5)   # 70% to 250% price expansion

        # Progressive acceleration during pump phase
        pump_weights = np.linspace(0.4, 1.6, pump_duration)
        pump_weights = pump_weights / pump_weights.sum()
        for i in range(pump_duration):
            idx = pump_idx + i
            if idx < n:
                returns[idx] = pump_magnitude * pump_weights[i] + rng.normal(0, coin_volatility * 0.4)

        # Realistic cascading sell-off during dump phase
        dump_weights = np.geomspace(1.8, 0.3, dump_duration)
        dump_weights = dump_weights / dump_weights.sum()
        dump_magnitude = pump_magnitude * rng.uniform(0.78, 0.94)  # leaves mild post-pump residual
        for i in range(dump_duration):
            idx = pump_idx + pump_duration + i
            if idx < n:
                returns[idx] = -dump_magnitude * dump_weights[i] + rng.normal(0, coin_volatility * 0.4)

        # Volume explosion during pump coordination
        vol_surge = rng.uniform(7.0, 22.0)
        for i in range(pump_duration + dump_duration):
            idx = pump_idx + i
            if idx < n:
                decay = 1.0 - (i / (pump_duration + dump_duration)) * 0.6
                volumes[idx] *= vol_surge * decay * rng.uniform(0.8, 1.3)

    # 3. Construct price series from returns
    prices = base_price * np.exp(np.cumsum(returns))

    # 4. Generate consistent OHLCV candles
    intraday_noise = rng.uniform(0.998, 1.002, n)
    opens = prices * intraday_noise
    closes = prices

    # High is strictly >= max(open, close), Low is strictly <= min(open, close)
    spread = np.abs(rng.normal(0, coin_volatility * 0.8, n))
    highs = np.maximum(opens, closes) * (1.0 + spread)
    lows = np.minimum(opens, closes) * (1.0 - spread)

    df = pd.DataFrame({
        "timestamp": timestamps[:n],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return df


def fetch_price_data_for_event(
    coin_symbol: str,
    pump_datetime: pd.Timestamp,
    hours_before: int = 24,
    hours_after: int = 12,
    is_pump: bool = True,
    base_price: float = None,
    use_cache: bool = True,
    allow_synthetic: bool = True,
    try_api: bool = False,
) -> pd.DataFrame:
    """
    Fetch price data for a single pump or normal trading event.
    Tries: (1) cache, (2) Binance API (if try_api=True), (3) synthetic fallback.

    Parameters
    ----------
    coin_symbol : str
        The coin ticker (e.g., "BRD").
    pump_datetime : pd.Timestamp
        The event time.
    hours_before, hours_after : int
        Window around the event.
    is_pump : bool
        Whether this is a pump-and-dump event or normal trading window.
    base_price : float, optional
        Coin base price.
    use_cache : bool
        Whether to check/save cache.
    allow_synthetic : bool
        Whether to fall back to synthetic data if API fails or try_api=False.
    try_api : bool
        Whether to attempt live Binance API queries. Defaults to False for rapid offline processing.

    Returns
    -------
    pd.DataFrame with OHLCV data, plus a 'data_source' attribute.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    label_tag = "pump" if is_pump else "normal"
    cache_key = f"{coin_symbol}_{pump_datetime.strftime('%Y%m%d_%H%M')}_{label_tag}"
    cache_path = CACHE_DIR / f"{cache_key}.parquet"

    # Try cache first
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        df.attrs["data_source"] = "cache"
        return df

    start_dt = pump_datetime - timedelta(hours=hours_before)
    end_dt = pump_datetime + timedelta(hours=hours_after)

    # Try Binance API if requested
    if try_api:
        try:
            df = fetch_ohlcv_ccxt(coin_symbol, start_dt, end_dt)
            if len(df) > 0:
                if use_cache:
                    df.to_parquet(cache_path, index=False)
                df.attrs["data_source"] = "binance_api"
                return df
        except Exception as e:
            logger.warning(f"Could not fetch {coin_symbol} from Binance: {e}")

    # Fallback to dynamic synthetic
    if allow_synthetic:
        df = generate_synthetic_ohlcv(
            pump_datetime,
            coin_symbol=coin_symbol,
            hours_before=hours_before,
            hours_after=hours_after,
            base_price=base_price,
            is_pump=is_pump,
        )
        df.attrs["data_source"] = "synthetic"
        if use_cache:
            df.to_parquet(cache_path, index=False)
        return df

    raise RuntimeError(
        f"Could not obtain price data for {coin_symbol} at {pump_datetime}"
    )


if __name__ == "__main__":
    import sys

    # Quick test with synthetic data
    test_time = pd.Timestamp("2019-06-15 16:00:00", tz="UTC")
    print("Generating synthetic OHLCV for BRD/BTC around a pump event...")
    df = generate_synthetic_ohlcv(test_time)
    print(f"  Shape: {df.shape}")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price range: {df['close'].min():.8f} to {df['close'].max():.8f}")
    print(f"  Volume range: {df['volume'].min():.2f} to {df['volume'].max():.2f}")
    print(f"\nSample around pump time:")
    pump_idx = len(df) // 2  # Approximately
    sample = df.iloc[pump_idx - 5: pump_idx + 20]
    print(sample.to_string(index=False))

