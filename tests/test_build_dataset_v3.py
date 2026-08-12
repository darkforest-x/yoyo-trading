"""Dataset V3 build: window resolution and the gates that make it trainable."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tools.build_dataset_v3 import csv_index, resolve_event_windows

HOLDOUT = datetime(2026, 5, 4, tzinfo=timezone.utc)


def write_series(repo: Path, symbol: str, start: str, rows: int = 400) -> None:
    directory = repo / "data/kline_fetched"
    directory.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(start, periods=rows, freq="15min", tz="UTC")
    close = np.linspace(100.0, 120.0, rows)
    pd.DataFrame(
        {
            "open_time": times,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1.0,
        }
    ).to_csv(directory / f"okx_{symbol}_15m_{rows}.csv", index=False)


def event(symbol: str, decision: str, window_len: int = 12, **extra) -> dict:
    return {
        "sample_id": f"{symbol}:{decision}",
        "symbol": symbol,
        "decision_time": decision,
        "window_len": window_len,
        **extra,
    }


def test_csv_index_picks_one_path_per_symbol(tmp_path) -> None:
    write_series(tmp_path, "ETH_USDT_SWAP", "2026-01-01")
    assert set(csv_index(tmp_path)) == {"ETH_USDT_SWAP"}


def test_resolve_maps_decision_time_to_bar_indices(tmp_path) -> None:
    write_series(tmp_path, "ETH_USDT_SWAP", "2026-01-01")
    resolved, dropped = resolve_event_windows(
        [event("ETH_USDT_SWAP", "2026-01-02T00:00:00+00:00", 15)], tmp_path, HOLDOUT
    )
    assert not dropped
    row = resolved[0]
    assert row["win_end"] - row["win_start"] == 14
    assert row["visible_end_bar"] == row["decision_bar"] == row["win_end"]


def test_resolve_drops_events_the_series_cannot_support(tmp_path) -> None:
    write_series(tmp_path, "ETH_USDT_SWAP", "2026-01-01")
    _, dropped = resolve_event_windows(
        [
            event("ETH_USDT_SWAP", "2026-02-01T00:00:00+00:00"),
            event("ETH_USDT_SWAP", "2026-01-01T00:00:00+00:00", 12),
            event("SOL_USDT_SWAP", "2026-01-02T00:00:00+00:00"),
        ],
        tmp_path,
        HOLDOUT,
    )
    reasons = {row["drop_reason"] for row in dropped}
    assert reasons == {
        "decision bar not in source series",
        "window starts before series",
        "no source csv",
    }


def test_resolve_refuses_to_touch_holdout(tmp_path) -> None:
    write_series(tmp_path, "ETH_USDT_SWAP", "2026-05-03T00:00:00+00:00", rows=200)
    resolved, dropped = resolve_event_windows(
        [event("ETH_USDT_SWAP", "2026-05-04T06:00:00+00:00")], tmp_path, HOLDOUT
    )
    assert not resolved
    assert dropped[0]["drop_reason"] == "decision at or after holdout"
