"""Owner-facing timestamps in Beijing (UTC+8).

Storage / computation stay UTC everywhere (pandas utc=True, ISO-Z in logs).
Only human-readable strings (Telegram, dashboard) go through this module.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

BEIJING = timezone(timedelta(hours=8))


def to_beijing_ts(value: Any) -> pd.Timestamp | None:
    """Parse anything → timezone-aware Beijing Timestamp, or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    ts = pd.Timestamp(value)
    if ts is pd.NaT or pd.isna(ts):
        return None
    if ts.tzinfo is None:
        # Naive times in this project are always UTC (OKX bars, forward_log).
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.tz_convert(BEIJING)


def format_beijing(
    value: Any,
    *,
    with_seconds: bool = False,
    with_label: bool = True,
    fallback: str = "—",
) -> str:
    """e.g. '2026-07-18 23:15' or '2026-07-18 23:15:00 北京'."""
    ts = to_beijing_ts(value)
    if ts is None:
        return fallback
    fmt = "%Y-%m-%d %H:%M:%S" if with_seconds else "%Y-%m-%d %H:%M"
    s = ts.strftime(fmt)
    return f"{s} 北京" if with_label else s
