"""
Phase 1, Step 1: Load and parse the La Morgia pump-and-dump event dataset.

Reads pump_telegram.csv and groups.csv, filters to Binance events,
and produces a cleaned DataFrame of labeled pump events.
"""
import pandas as pd
import os
from pathlib import Path

# Project root — resolve relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "pump-and-dump-dataset"


def load_pump_events(filter_binance: bool = True) -> pd.DataFrame:
    """
    Load the La Morgia et al. pump-and-dump event catalog.

    Parameters
    ----------
    filter_binance : bool
        If True (default), keep only Binance exchange events.

    Returns
    -------
    pd.DataFrame with columns:
        coin_symbol, group_code, pump_datetime (UTC), exchange, group_name,
        telegram_link
    """
    pump_path = RAW_DATA / "pump_telegram.csv"
    groups_path = RAW_DATA / "groups.csv"

    if not pump_path.exists():
        raise FileNotFoundError(
            f"pump_telegram.csv not found at {pump_path}. "
            "Did you clone the pump-and-dump-dataset repo into data/raw/?"
        )

    # Load events
    pump_df = pd.read_csv(pump_path)
    pump_df.columns = pump_df.columns.str.strip()

    # Combine date + hour into a single UTC datetime
    pump_df["pump_datetime"] = pd.to_datetime(
        pump_df["date"] + " " + pump_df["hour"],
        format="%Y-%m-%d %H:%M",
        utc=True,
    )

    # Standardize column names
    pump_df = pump_df.rename(columns={
        "symbol": "coin_symbol",
        "group": "group_code",
    })

    # Filter to Binance if requested
    if filter_binance:
        pump_df = pump_df[pump_df["exchange"] == "binance"].copy()

    # Uppercase coin symbols for consistency
    pump_df["coin_symbol"] = pump_df["coin_symbol"].str.upper().str.strip()

    # Load group metadata and merge
    if groups_path.exists():
        groups_df = pd.read_csv(groups_path)
        groups_df.columns = groups_df.columns.str.strip()
        groups_df = groups_df.rename(columns={
            "group_name": "group_name",
            "group_code": "group_code",
            "telegram_link": "telegram_link",
        })
        pump_df = pump_df.merge(
            groups_df[["group_code", "group_name", "telegram_link"]],
            on="group_code",
            how="left",
        )
    else:
        pump_df["group_name"] = None
        pump_df["telegram_link"] = None

    # Select and order final columns
    cols = [
        "coin_symbol",
        "group_code",
        "pump_datetime",
        "exchange",
        "group_name",
        "telegram_link",
    ]
    pump_df = pump_df[cols].reset_index(drop=True)

    return pump_df


def load_pump_event_stats() -> dict:
    """Return summary statistics about the loaded pump events."""
    df = load_pump_events(filter_binance=True)
    return {
        "total_events": len(df),
        "unique_coins": df["coin_symbol"].nunique(),
        "date_range": (
            df["pump_datetime"].min().isoformat(),
            df["pump_datetime"].max().isoformat(),
        ),
        "events_by_group": df["group_code"].value_counts().to_dict(),
        "top_coins": df["coin_symbol"].value_counts().head(10).to_dict(),
    }


if __name__ == "__main__":
    print("Loading La Morgia pump-and-dump events (Binance only)...")
    df = load_pump_events(filter_binance=True)
    print(f"  Total events: {len(df)}")
    print(f"  Unique coins: {df['coin_symbol'].nunique()}")
    print(f"  Date range: {df['pump_datetime'].min()} to {df['pump_datetime'].max()}")
    print(f"\nTop 10 coins by event count:")
    print(df["coin_symbol"].value_counts().head(10).to_string())
    print(f"\nSample events:")
    print(df.head(10).to_string(index=False))

