"""
Phase 1, Step 4: Reconcile pump events from both datasets with price data.

This is the core data reconciliation module that produces the aligned event
tables described in Section 3.3 of the project specification:

Schema per event:
{
    coin_symbol: str,
    pump_start_time: datetime,
    dump_confirmed_time: datetime,
    price_series: DataFrame[timestamp, open, high, low, close, volume],
    telegram_messages: List[{timestamp, group_id, text}],
    label: "pump_and_dump_event" | "normal_trading"
}

Since the Ardia dataset contains only event metadata (no raw Telegram message
text), we use the event timestamps and channel names as proxy "message" signals.
The text content is simulated from the event metadata for FinBERT scoring.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import logging
import json

from src.data.load_pump_events import load_pump_events
from src.data.load_telegram_text import load_ardia_events
from src.data.fetch_binance_prices import (
    fetch_price_data_for_event,
    generate_synthetic_ohlcv,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def merge_event_catalogs(
    time_tolerance_minutes: int = 10,
) -> pd.DataFrame:
    """
    Merge the La Morgia and Ardia event catalogs into a unified event list.

    Events from both datasets are matched by (coin_symbol, timestamp proximity).
    Matched events are merged; unmatched events from either dataset are kept.

    Parameters
    ----------
    time_tolerance_minutes : int
        Maximum time difference (in minutes) to consider two events as the same.

    Returns
    -------
    pd.DataFrame — unified event catalog with columns:
        event_id, coin_symbol, pump_datetime, source, group_code,
        channel_name, is_successful, exchange
    """
    la_morgia = load_pump_events(filter_binance=True)
    ardia = load_ardia_events()

    events = []
    event_id = 0
    ardia_matched = set()

    # For each La Morgia event, try to find a matching Ardia event
    for _, lm_row in la_morgia.iterrows():
        coin = lm_row["coin_symbol"]
        pump_dt = lm_row["pump_datetime"]

        # Find Ardia events for the same coin within the time window
        ardia_same_coin = ardia[ardia["coin_symbol"] == coin]
        if len(ardia_same_coin) > 0:
            time_diffs = abs(
                (ardia_same_coin["signal_datetime"] - pump_dt).dt.total_seconds()
            )
            closest_idx = time_diffs.idxmin()
            min_diff = time_diffs[closest_idx]

            if min_diff <= time_tolerance_minutes * 60:
                # Matched event — merge both sources
                ar_row = ardia.loc[closest_idx]
                ardia_matched.add(closest_idx)
                events.append({
                    "event_id": f"E{event_id:04d}",
                    "coin_symbol": coin,
                    "pump_datetime": pump_dt,
                    "source": "both",
                    "group_code": lm_row["group_code"],
                    "group_name": lm_row.get("group_name"),
                    "channel_name": ar_row["channel_name"],
                    "is_successful": ar_row["is_successful"],
                    "pump_start_actual": ar_row["pump_start_datetime"],
                    "exchange": "binance",
                })
                event_id += 1
                continue

        # Unmatched La Morgia event
        events.append({
            "event_id": f"E{event_id:04d}",
            "coin_symbol": coin,
            "pump_datetime": pump_dt,
            "source": "la_morgia",
            "group_code": lm_row["group_code"],
            "group_name": lm_row.get("group_name"),
            "channel_name": None,
            "is_successful": 1,  # La Morgia lists confirmed pumps
            "pump_start_actual": pd.NaT,
            "exchange": "binance",
        })
        event_id += 1

    # Add unmatched Ardia events
    for idx, ar_row in ardia.iterrows():
        if idx not in ardia_matched:
            events.append({
                "event_id": f"E{event_id:04d}",
                "coin_symbol": ar_row["coin_symbol"],
                "pump_datetime": ar_row["signal_datetime"],
                "source": "ardia",
                "group_code": None,
                "group_name": None,
                "channel_name": ar_row["channel_name"],
                "is_successful": ar_row["is_successful"],
                "pump_start_actual": ar_row["pump_start_datetime"],
                "exchange": "binance",
            })
            event_id += 1

    result = pd.DataFrame(events)
    result = result.sort_values("pump_datetime").reset_index(drop=True)
    return result


def generate_telegram_messages_for_event(
    coin_symbol: str,
    pump_datetime: pd.Timestamp,
    channel_name: str = None,
    group_code: str = None,
    is_pump: bool = True,
) -> list[dict]:
    """
    Generate synthetic Telegram-style message signals for an event.

    Since the actual Telegram message text is not available in the Ardia dataset
    (only metadata), we construct representative message signals that capture
    the key characteristics a FinBERT model would detect:
    - Timing pattern (message volume spikes before pump)
    - Urgency language patterns typical of P&D coordination
    - Coin symbol mentions

    For non-pump (normal) windows, we generate baseline market chatter.

    Parameters
    ----------
    coin_symbol : str
    pump_datetime : pd.Timestamp
    channel_name : str, optional
    group_code : str, optional
    is_pump : bool
        If True, generate manipulation-style messages; if False, normal chatter.

    Returns
    -------
    list of dicts with keys: timestamp, group_id, text
    """
    rng = np.random.RandomState(
        abs(hash(f"{coin_symbol}_{pump_datetime}")) % (2**31)
    )
    group_id = group_code or channel_name or "unknown_group"
    messages = []

    if is_pump:
        # Pre-pump buildup messages (hours before)
        buildup_templates = [
            f"Big announcement coming for {coin_symbol} today! Stay tuned 🚀",
            f"Get ready everyone. Target will be revealed soon. Be prepared to buy fast!",
            f"Next pump target will be on Binance. BTC pair. Don't miss it!",
            f"Remember the rules: buy fast, hold, and don't dump early!",
            f"Signal coming in the next hour. Get your BTC ready on Binance!",
        ]

        # Messages 2-6 hours before
        for i in range(rng.randint(2, 5)):
            hours_before = rng.uniform(2, 6)
            msg_time = pump_datetime - timedelta(hours=hours_before)
            messages.append({
                "timestamp": msg_time.isoformat(),
                "group_id": group_id,
                "text": rng.choice(buildup_templates),
            })

        # Pump signal messages (around pump time)
        signal_templates = [
            f"🚀🚀🚀 BUY {coin_symbol} NOW! Target: {coin_symbol}/BTC on Binance! GO GO GO! 🚀🚀🚀",
            f"COIN: {coin_symbol}\nEXCHANGE: BINANCE\nPAIR: {coin_symbol}/BTC\nBUY NOW!!!",
            f"📢 PUMP SIGNAL: {coin_symbol} on Binance! Buy immediately! Minimum 200% target!",
            f"🎯 TARGET REVEALED: {coin_symbol}! Quick, buy before it moons! 🌙",
        ]

        # Messages right at pump time (within ±2 minutes)
        for i in range(rng.randint(3, 8)):
            seconds_offset = rng.uniform(-120, 120)
            msg_time = pump_datetime + timedelta(seconds=seconds_offset)
            messages.append({
                "timestamp": msg_time.isoformat(),
                "group_id": group_id,
                "text": rng.choice(signal_templates),
            })

        # Post-pump messages (0-30 minutes after)
        post_templates = [
            f"{coin_symbol} is pumping! Hold hold hold! 📈",
            f"We're up 150% on {coin_symbol}! Keep holding!",
            f"Don't sell yet! {coin_symbol} still going up!",
            f"SELL SELL SELL! Take your profits on {coin_symbol}!",
            f"{coin_symbol} is dumping. Get out now!",
        ]

        for i in range(rng.randint(3, 8)):
            minutes_after = rng.uniform(1, 30)
            msg_time = pump_datetime + timedelta(minutes=minutes_after)
            messages.append({
                "timestamp": msg_time.isoformat(),
                "group_id": group_id,
                "text": rng.choice(post_templates),
            })

    else:
        # Normal market chatter for non-event windows
        normal_templates = [
            f"Anyone watching {coin_symbol} today?",
            f"Market is quiet. Just hodling my portfolio.",
            f"BTC is ranging. Not much action in alts.",
            f"What do you guys think about the current market?",
            f"Volume is low today across the board.",
            f"Just DCA-ing as usual. Nothing exciting.",
            f"Waiting for a breakout on BTC before making any moves.",
        ]

        # Sparse messages throughout the day
        for i in range(rng.randint(3, 8)):
            hours_offset = rng.uniform(-12, 12)
            msg_time = pump_datetime + timedelta(hours=hours_offset)
            messages.append({
                "timestamp": msg_time.isoformat(),
                "group_id": group_id,
                "text": rng.choice(normal_templates),
            })

    # Sort by timestamp
    messages.sort(key=lambda m: m["timestamp"])
    return messages


def create_event_record(
    event_row: pd.Series,
    is_pump: bool = True,
    hours_before: int = 24,
    hours_after: int = 12,
    try_api: bool = False,
) -> dict:
    """
    Create a single complete event record with price data and messages.

    Parameters
    ----------
    event_row : pd.Series
        A row from the unified event catalog.
    is_pump : bool
        Whether this is a pump event or normal trading window.
    hours_before, hours_after : int
        Price window around the event.
    try_api : bool
        Whether to attempt live network calls to Binance.

    Returns
    -------
    dict matching the Section 3.3 schema.
    """
    coin = event_row["coin_symbol"]
    pump_dt = event_row["pump_datetime"]

    # Fetch price data
    try:
        price_df = fetch_price_data_for_event(
            coin, pump_dt,
            hours_before=hours_before,
            hours_after=hours_after,
            is_pump=is_pump,
            allow_synthetic=True,
            try_api=try_api,
        )
        data_source = price_df.attrs.get("data_source", "unknown")
    except Exception as e:
        logger.warning(f"Failed to get price data for {coin}: {e}")
        price_df = generate_synthetic_ohlcv(
            pump_dt,
            coin_symbol=coin,
            hours_before=hours_before,
            hours_after=hours_after,
            is_pump=is_pump,
        )
        data_source = "synthetic"

    # Generate message signals
    messages = generate_telegram_messages_for_event(
        coin, pump_dt,
        channel_name=event_row.get("channel_name"),
        group_code=event_row.get("group_code"),
        is_pump=is_pump,
    )

    # Estimate dump confirmation time (for pump events)
    if is_pump:
        # The dump typically occurs 15-60 minutes after the pump start
        pump_start = event_row.get("pump_start_actual", pd.NaT)
        if pd.isna(pump_start):
            pump_start = pump_dt

        # Find the price peak after pump start in the price data
        post_pump = price_df[price_df["timestamp"] >= pump_start]
        if len(post_pump) > 0:
            peak_idx = post_pump["close"].idxmax()
            dump_confirmed = price_df.loc[peak_idx, "timestamp"] + timedelta(minutes=15)
        else:
            dump_confirmed = pump_start + timedelta(minutes=30)
    else:
        pump_start = pump_dt
        dump_confirmed = pd.NaT

    record = {
        "event_id": event_row["event_id"],
        "coin_symbol": coin,
        "pump_start_time": pump_start.isoformat() if not pd.isna(pump_start) else None,
        "dump_confirmed_time": dump_confirmed.isoformat() if not pd.isna(dump_confirmed) else None,
        "label": "pump_and_dump_event" if is_pump else "normal_trading",
        "source": event_row.get("source", "unknown"),
        "group_code": event_row.get("group_code"),
        "channel_name": event_row.get("channel_name"),
        "is_successful": event_row.get("is_successful", 1) if is_pump else 0,
        "data_source": data_source,
        "price_data_points": len(price_df),
        "message_count": len(messages),
        # Store price series as JSON-serializable dict list
        "price_series": price_df.to_dict(orient="records"),
        "telegram_messages": messages,
    }

    return record


def generate_normal_windows(
    pump_events: pd.DataFrame,
    ratio: float = 1.5,
) -> pd.DataFrame:
    """
    Generate non-event (normal trading) windows as negative examples.

    For each unique coin in the pump events, sample random time windows
    that do NOT overlap with any documented pump event for that coin.

    Parameters
    ----------
    pump_events : pd.DataFrame
        The unified event catalog.
    ratio : float
        Number of normal windows per pump event (>= 1.0 for balance).

    Returns
    -------
    pd.DataFrame with the same schema as pump_events but labeled as normal.
    """
    rng = np.random.RandomState(42)
    normal_windows = []
    event_id_counter = len(pump_events)

    # Get all pump events per coin
    coins = pump_events["coin_symbol"].unique()

    for coin in coins:
        coin_events = pump_events[pump_events["coin_symbol"] == coin]
        n_normal = max(1, int(len(coin_events) * ratio))

        # Get the date range for this coin's events
        min_date = coin_events["pump_datetime"].min()
        max_date = coin_events["pump_datetime"].max()

        # Extend the range to sample from
        range_start = min_date - timedelta(days=30)
        range_end = max_date + timedelta(days=30)
        total_seconds = (range_end - range_start).total_seconds()

        pump_times = coin_events["pump_datetime"].tolist()

        sampled = 0
        attempts = 0
        max_attempts = n_normal * 10

        while sampled < n_normal and attempts < max_attempts:
            attempts += 1
            # Random timestamp in the range
            offset = rng.uniform(0, total_seconds)
            candidate = range_start + timedelta(seconds=offset)

            # Check no overlap with any pump event (±2 hours buffer)
            is_safe = all(
                abs((candidate - pt).total_seconds()) > 2 * 3600
                for pt in pump_times
            )

            if is_safe:
                normal_windows.append({
                    "event_id": f"N{event_id_counter:04d}",
                    "coin_symbol": coin,
                    "pump_datetime": candidate,
                    "source": "synthetic_normal",
                    "group_code": None,
                    "group_name": None,
                    "channel_name": None,
                    "is_successful": 0,
                    "pump_start_actual": pd.NaT,
                    "exchange": "binance",
                })
                event_id_counter += 1
                sampled += 1

    result = pd.DataFrame(normal_windows)
    return result


def reconcile_all(
    max_events: int = None,
    normal_ratio: float = 1.5,
    hours_before: int = 24,
    hours_after: int = 12,
    try_api: bool = False,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run the full reconciliation pipeline.

    Parameters
    ----------
    max_events : int, optional
        Limit total pump events to process (for testing/debugging).
    normal_ratio : float
        Ratio of normal windows to pump events.
    hours_before, hours_after : int
        Price data window.
    try_api : bool
        Whether to attempt live Binance API queries.
    progress_callback : callable, optional
        Called with (current, total) for progress updates.

    Returns
    -------
    metadata_df : pd.DataFrame
        Event metadata (without price/message data).
    records : list[dict]
        Full event records with price series and messages.
    """
    logger.info("Step 1: Merging event catalogs...")
    unified_events = merge_event_catalogs()
    logger.info(f"  Unified catalog: {len(unified_events)} pump events")

    if max_events:
        unified_events = unified_events.head(max_events)
        logger.info(f"  Limited to {max_events} events for processing")

    logger.info("Step 2: Generating normal trading windows...")
    normal_windows = generate_normal_windows(unified_events, ratio=normal_ratio)
    logger.info(f"  Generated {len(normal_windows)} normal windows")

    # Combine
    all_events = pd.concat([unified_events, normal_windows], ignore_index=True)
    all_events = all_events.sort_values("pump_datetime").reset_index(drop=True)

    logger.info(f"Step 3: Building event records ({len(all_events)} total)...")
    records = []
    total = len(all_events)

    for i, (_, row) in enumerate(all_events.iterrows()):
        is_pump = row.get("source") != "synthetic_normal"
        try:
            record = create_event_record(row, is_pump=is_pump,
                                         hours_before=hours_before,
                                         hours_after=hours_after,
                                         try_api=try_api)
            records.append(record)
        except Exception as e:
            logger.error(f"Failed to process event {row['event_id']}: {e}")
            continue

        if progress_callback:
            progress_callback(i + 1, total)
        elif (i + 1) % 50 == 0 or i == total - 1:
            logger.info(f"  Processed {i + 1}/{total} events")

    # Build metadata dataframe (without heavy price/message data)
    metadata = []
    for r in records:
        metadata.append({
            "event_id": r["event_id"],
            "coin_symbol": r["coin_symbol"],
            "pump_start_time": r["pump_start_time"],
            "dump_confirmed_time": r["dump_confirmed_time"],
            "label": r["label"],
            "source": r["source"],
            "group_code": r["group_code"],
            "channel_name": r["channel_name"],
            "is_successful": r["is_successful"],
            "data_source": r["data_source"],
            "price_data_points": r["price_data_points"],
            "message_count": r["message_count"],
        })

    metadata_df = pd.DataFrame(metadata)
    return metadata_df, records


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Run a small test reconciliation
    print("=" * 60)
    print("PHASE 1: Data Reconciliation (Test Run)")
    print("=" * 60)

    # First, just test the catalog merge
    print("\n--- Testing catalog merge ---")
    unified = merge_event_catalogs()
    print(f"Unified events: {len(unified)}")
    source_counts = unified["source"].value_counts()
    print(f"Sources: {source_counts.to_dict()}")
    print(f"Date range: {unified['pump_datetime'].min()} to {unified['pump_datetime'].max()}")
    print(f"Unique coins: {unified['coin_symbol'].nunique()}")

    # Test with a small subset
    print("\n--- Testing full reconciliation (10 events) ---")
    metadata, records = reconcile_all(max_events=10, normal_ratio=1.0)
    print(f"\nResults:")
    print(f"  Total records: {len(records)}")
    label_counts = metadata["label"].value_counts()
    print(f"  Labels: {label_counts.to_dict()}")
    print(f"  Data sources: {metadata['data_source'].value_counts().to_dict()}")
    print(f"\nSample metadata:")
    print(metadata.head(10).to_string(index=False))

