"""
Phase 1, Step 2: Load and parse the Ardia & Bluteau pump-and-dump event dataset.

This dataset contains 1,160 pump-and-dump events from 100+ Telegram groups,
all on Binance exchange, BTC trading pairs. It also includes a success flag
and the actual pump start time for successful events.
"""
import pandas as pd
from pathlib import Path

# Project root — resolve relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "PumpDump"


def load_ardia_events(successful_only: bool = False) -> pd.DataFrame:
    """
    Load the Ardia & Bluteau (2024) pump-and-dump event catalog.

    Parameters
    ----------
    successful_only : bool
        If True, keep only events marked as successful (success=1).

    Returns
    -------
    pd.DataFrame with columns:
        coin_symbol, channel_name, signal_datetime (UTC), exchange,
        is_successful, pump_start_datetime (UTC, NaT if unsuccessful)
    """
    data_path = RAW_DATA / "data" / "list_pd_events.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"list_pd_events.csv not found at {data_path}. "
            "Did you clone the PumpDump repo into data/raw/?"
        )

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # Parse datetimes
    df["signal_datetime"] = pd.to_datetime(df["Date"], utc=True)
    df["pump_start_datetime"] = pd.to_datetime(
        df["pump_date"], utc=True, errors="coerce"
    )

    # Standardize columns
    df = df.rename(columns={
        "Currency": "coin_symbol",
        "Channel": "channel_name",
        "Exchange": "exchange",
        "success": "is_successful",
    })

    # Uppercase coin symbols for consistency
    df["coin_symbol"] = df["coin_symbol"].str.upper().str.strip()

    # Standardize exchange name
    df["exchange"] = df["exchange"].str.lower().str.strip()

    # Filter if requested
    if successful_only:
        df = df[df["is_successful"] == 1].copy()

    # Select and order final columns
    cols = [
        "coin_symbol",
        "channel_name",
        "signal_datetime",
        "exchange",
        "is_successful",
        "pump_start_datetime",
    ]
    df = df[cols].reset_index(drop=True)

    return df


def load_ardia_event_stats() -> dict:
    """Return summary statistics about the Ardia dataset."""
    df = load_ardia_events()
    successful = df[df["is_successful"] == 1]
    return {
        "total_events": len(df),
        "successful_events": len(successful),
        "unsuccessful_events": len(df) - len(successful),
        "unique_coins": df["coin_symbol"].nunique(),
        "unique_channels": df["channel_name"].nunique(),
        "date_range": (
            df["signal_datetime"].min().isoformat(),
            df["signal_datetime"].max().isoformat(),
        ),
        "top_coins": df["coin_symbol"].value_counts().head(10).to_dict(),
    }


if __name__ == "__main__":
    print("Loading Ardia & Bluteau pump-and-dump events...")
    df = load_ardia_events()
    print(f"  Total events: {len(df)}")
    successful = df[df["is_successful"] == 1]
    print(f"  Successful: {len(successful)} | Unsuccessful: {len(df) - len(successful)}")
    print(f"  Unique coins: {df['coin_symbol'].nunique()}")
    print(f"  Unique channels: {df['channel_name'].nunique()}")
    print(f"  Date range: {df['signal_datetime'].min()} to {df['signal_datetime'].max()}")
    print(f"\nSample events:")
    print(df.head(10).to_string(index=False))

