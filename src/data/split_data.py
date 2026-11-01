"""
Phase 1, Step 5: Chronological Train / Validation / Test Splitting and Parquet Serialization.

Enforces zero look-ahead data leakage:
- Sorted strictly by event timestamp
- Train split: first 70% of chronological timeline
- Validation split: next 15% of chronological timeline
- Test split: final 15% of chronological timeline

Outputs:
- data/processed/train.parquet
- data/processed/val.parquet
- data/processed/test.parquet
"""
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.reconcile import reconcile_all

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_dataframe_from_records(records: list[dict]) -> pd.DataFrame:
    """
    Convert record dicts into a structured pandas DataFrame.
    Nested fields (price_series, telegram_messages) are serialized as JSON
    strings for clean, frictionless Parquet interoperability.
    """
    rows = []
    for r in records:
        row = {
            "event_id": r["event_id"],
            "coin_symbol": r["coin_symbol"],
            "pump_start_time": r["pump_start_time"],
            "dump_confirmed_time": r["dump_confirmed_time"],
            "label": r["label"],
            "source": r.get("source"),
            "group_code": r.get("group_code"),
            "channel_name": r.get("channel_name"),
            "is_successful": int(r.get("is_successful", 0)),
            "data_source": r.get("data_source"),
            "price_data_points": int(r.get("price_data_points", 0)),
            "message_count": int(r.get("message_count", 0)),
            "price_series_json": json.dumps(r.get("price_series", []), default=str),
            "telegram_messages_json": json.dumps(r.get("telegram_messages", []), default=str),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Parse timestamp column for chronological sorting
    df["sort_dt"] = pd.to_datetime(df["pump_start_time"], utc=True, format="ISO8601")
    df = df.sort_values("sort_dt").reset_index(drop=True)
    return df


def split_chronologically(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame chronologically with zero look-ahead data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input data, will be sorted by sort_dt.
    train_pct : float
        Percentage of events for the train set (default 0.70).
    val_pct : float
        Percentage of events for the validation set (default 0.15).

    Returns
    -------
    train_df, val_df, test_df
    """
    if "sort_dt" not in df.columns:
        df["sort_dt"] = pd.to_datetime(df["pump_start_time"], utc=True, format="ISO8601")

    df = df.sort_values("sort_dt").reset_index(drop=True)
    n = len(df)

    n_train = int(n * train_pct)
    n_val = int(n * (train_pct + val_pct))

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train:n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_val:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def save_splits_to_parquet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = PROCESSED_DIR,
) -> dict:
    """Save the train/val/test splits to Parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    summary = {
        "train": {
            "path": str(train_path),
            "count": len(train_df),
            "labels": train_df["label"].value_counts().to_dict(),
            "date_range": (
                train_df["sort_dt"].min().isoformat(),
                train_df["sort_dt"].max().isoformat(),
            ),
        },
        "val": {
            "path": str(val_path),
            "count": len(val_df),
            "labels": val_df["label"].value_counts().to_dict(),
            "date_range": (
                val_df["sort_dt"].min().isoformat(),
                val_df["sort_dt"].max().isoformat(),
            ),
        },
        "test": {
            "path": str(test_path),
            "count": len(test_df),
            "labels": test_df["label"].value_counts().to_dict(),
            "date_range": (
                test_df["sort_dt"].min().isoformat(),
                test_df["sort_dt"].max().isoformat(),
            ),
        },
    }
    return summary


def get_price_df(row: pd.Series) -> pd.DataFrame:
    """Helper function to load the price series for a row as a DataFrame."""
    data = json.loads(row["price_series_json"])
    df = pd.DataFrame(data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def get_messages(row: pd.Series) -> list[dict]:
    """Helper function to load telegram messages for a row."""
    return json.loads(row["telegram_messages_json"])


def run_pipeline(
    max_events: int = 200,
    normal_ratio: float = 1.0,
    try_api: bool = False,
) -> dict:
    """
    Execute full Phase 1 pipeline from raw catalogs to serialized parquet splits.

    Parameters
    ----------
    max_events : int
        Number of pump events to include (e.g. 200 for rapid baseline, or None for all).
    normal_ratio : float
        Ratio of negative (normal trading) windows to include.
    try_api : bool
        Whether to attempt live ccxt queries.
    """
    logger.info("Executing Phase 1 Data Pipeline...")
    metadata_df, records = reconcile_all(
        max_events=max_events,
        normal_ratio=normal_ratio,
        try_api=try_api,
    )

    logger.info(f"Assembling DataFrame from {len(records)} reconciled events...")
    full_df = build_dataframe_from_records(records)

    logger.info("Splitting chronologically (70/15/15)...")
    train_df, val_df, test_df = split_chronologically(full_df)

    logger.info("Saving to Parquet format...")
    summary = save_splits_to_parquet(train_df, val_df, test_df)

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    print("=" * 60)
    print("Running Phase 1 Pipeline: Building Train/Val/Test Parquet Splits")
    print("=" * 60)
    summary = run_pipeline(max_events=150, normal_ratio=1.0)
    print("\nData Splitting Complete:")
    for split_name, info in summary.items():
        print(f"\n--- {split_name.upper()} SPLIT ---")
        print(f"  Count: {info['count']}")
        print(f"  Labels: {info['labels']}")
        print(f"  Date range: {info['date_range'][0]} -> {info['date_range'][1]}")
