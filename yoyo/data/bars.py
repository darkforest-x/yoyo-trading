"""Shared candle-bar utilities for multi-timeframe research paths."""
from __future__ import annotations

from typing import Final

import pandas as pd

# Mainline trading bar is 15m; 1m/2m/3m/5m/30m/1H are research-only paths.
BAR_CHOICES: Final = ("1m", "2m", "3m", "5m", "15m", "30m", "1H")
_BAR_MINUTES: Final = {
    "1m": 1,
    "2m": 2,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1H": 60,
}


def normalize_bar(bar: str) -> str:
    if bar not in _BAR_MINUTES:
        raise ValueError(f"unsupported bar {bar!r}; expected one of {BAR_CHOICES}")
    return bar


def bar_to_timedelta(bar: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=_BAR_MINUTES[normalize_bar(bar)])


def purge_window(horizon_bars: int, bar: str) -> pd.Timedelta:
    if horizon_bars < 0:
        raise ValueError("horizon_bars must be non-negative")
    return (horizon_bars + 1) * bar_to_timedelta(bar)
