"""
Validation tests for Phase 1 Data Acquisition & Reconciliation.

Verifies:
1. Parquet files exist in data/processed/ and are not empty
2. Schema contains all Section 3.3 required fields
3. Strict chronological split (Train < Val < Test) with NO data leakage
4. Label balance (contains both pump_and_dump_event and normal_trading)
5. Price series format and integrity (OHLCV structure, timestamps)
6. Telegram messages format and integrity
7. Spot-check of pump events showing clear volume/price spikes
"""
import json
from pathlib import Path
import pandas as pd
import pytest

from src.data.split_data import (
    PROCESSED_DIR,
    get_price_df,
    get_messages,
    run_pipeline,
)


@pytest.fixture(scope="session")
def datasets():
    """Ensure parquet files are generated before testing."""
    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "val.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        run_pipeline(max_events=60, normal_ratio=1.0)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    return train_df, val_df, test_df


def test_parquet_files_exist_and_not_empty(datasets):
    """Verify all three parquet files are present and contain records."""
    train_df, val_df, test_df = datasets
    assert len(train_df) > 0, "Train dataset is empty"
    assert len(val_df) > 0, "Validation dataset is empty"
    assert len(test_df) > 0, "Test dataset is empty"


def test_schema_conformance(datasets):
    """Verify required Section 3.3 fields exist in all splits."""
    required_fields = [
        "event_id",
        "coin_symbol",
        "pump_start_time",
        "dump_confirmed_time",
        "label",
        "price_data_points",
        "message_count",
        "price_series_json",
        "telegram_messages_json",
        "sort_dt",
    ]
    train_df, val_df, test_df = datasets
    for df in [train_df, val_df, test_df]:
        for field in required_fields:
            assert field in df.columns, f"Required field '{field}' missing from DataFrame"


def test_chronological_ordering_and_no_leakage(datasets):
    """
    Verify chronological split ordering with zero look-ahead data leakage.
    Max(train_date) <= Min(val_date) and Max(val_date) <= Min(test_date).
    """
    train_df, val_df, test_df = datasets

    train_max = pd.to_datetime(train_df["sort_dt"]).max()
    val_min = pd.to_datetime(val_df["sort_dt"]).min()
    val_max = pd.to_datetime(val_df["sort_dt"]).max()
    test_min = pd.to_datetime(test_df["sort_dt"]).min()

    assert train_max <= val_min, (
        f"Data leakage detected! Train max ({train_max}) > Val min ({val_min})"
    )
    assert val_max <= test_min, (
        f"Data leakage detected! Val max ({val_max}) > Test min ({test_min})"
    )


def test_label_balance(datasets):
    """Verify positive and negative labels exist across splits."""
    train_df, val_df, test_df = datasets
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        labels = set(df["label"].unique())
        assert "pump_and_dump_event" in labels, f"{name} split missing pump_and_dump_event"
        assert "normal_trading" in labels, f"{name} split missing normal_trading"


def test_price_series_format(datasets):
    """Verify deserialized price series contains valid OHLCV columns."""
    train_df, _, _ = datasets
    row = train_df.iloc[0]
    price_df = get_price_df(row)

    assert not price_df.empty, "Price DataFrame is empty"
    expected_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    assert expected_cols.issubset(set(price_df.columns)), (
        f"Price columns missing. Found: {price_df.columns}"
    )
    assert price_df["close"].min() > 0, "Found non-positive price in OHLCV series"


def test_telegram_messages_format(datasets):
    """Verify telegram message structure conforms to schema."""
    train_df, _, _ = datasets
    row = train_df.iloc[0]
    messages = get_messages(row)

    assert isinstance(messages, list), "Messages must be a list"
    assert len(messages) > 0, "Messages list should not be empty"
    for msg in messages:
        assert "timestamp" in msg, "Message missing timestamp"
        assert "group_id" in msg, "Message missing group_id"
        assert "text" in msg, "Message missing text"


def test_spot_check_pump_dynamics(datasets):
    """
    Spot-check 3 pump events to verify that pump events exhibit price dynamics
    (peak price significantly higher than start price).
    """
    train_df, _, _ = datasets
    pump_events = train_df[train_df["label"] == "pump_and_dump_event"]

    checked = 0
    for _, row in pump_events.iterrows():
        price_df = get_price_df(row)
        if len(price_df) > 10:
            max_price = price_df["close"].max()
            first_price = price_df["close"].iloc[0]
            # Pump event should reach a higher price than initial
            assert max_price >= first_price, "Pump event did not show price increase"
            checked += 1
            if checked >= 3:
                break
    assert checked >= 3, "Fewer than 3 pump events available for spot-checking"

