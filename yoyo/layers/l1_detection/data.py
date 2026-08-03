"""Minimal 15m OHLCV loading + MA computation for the detection layer.

Standalone on purpose: the judgment layer (src/judgment/, src/data/) owns its
own loaders; keeping a local copy avoids cross-task coupling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Six moving averages drawn on charts and used for density judgment.
MA_PERIODS = (20, 60, 120)
SMA_COLS = tuple(f"sma{p}" for p in MA_PERIODS)
EMA_COLS = tuple(f"ema{p}" for p in MA_PERIODS)
ALL_MA_COLS = SMA_COLS + EMA_COLS
# Fast subset (20/60) mirrors the old project's fast_spread over its fast EMA bundle.
FAST_MA_COLS = ("sma20", "ema20", "sma60", "ema60")

# Warmup bars before MA values are trustworthy (longest SMA window).
WARMUP_BARS = max(MA_PERIODS)

OLD_CACHE_DIR = Path(
    "/Users/zhangzc/Documents/Codex/2026-06-17/yolo-yolo-okx-20-k"
    "/outputs/yolo_ma_cluster_trader/runs/cache"
)


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load one cached OKX candle CSV (ts, open, high, low, close, volume, ...)."""
    df = pd.read_csv(path)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def add_mas(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA/EMA 20/60/120 plus fast/full bundle spreads (relative to close)."""
    out = df.copy()
    close = out["close"]
    for p in MA_PERIODS:
        out[f"sma{p}"] = close.rolling(p).mean()
        out[f"ema{p}"] = close.ewm(span=p, adjust=False).mean()
    mas = out[list(ALL_MA_COLS)]
    fast = out[list(FAST_MA_COLS)]
    safe_close = close.replace(0, pd.NA)
    out["fast_spread"] = (fast.max(axis=1) - fast.min(axis=1)) / safe_close
    out["full_spread"] = (mas.max(axis=1) - mas.min(axis=1)) / safe_close
    return out


def list_cache_files(cache_dir: str | Path = OLD_CACHE_DIR, min_rows: int = 10000) -> list[Path]:
    """Return one 15m cache CSV per symbol (the longest), with >= min_rows candles."""
    cache_dir = Path(cache_dir)
    best: dict[str, tuple[int, Path]] = {}
    for path in sorted(cache_dir.glob("*_15m_*.csv")):
        # gate_* caches use a different schema and unreliable row counts; skip them
        if path.name.startswith("gate_"):
            continue
        try:
            declared = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if declared < min_rows:
            continue
        symbol = path.stem.rsplit("_", 2)[0]
        if symbol not in best or declared > best[symbol][0]:
            best[symbol] = (declared, path)
    return [path for _, path in sorted(best.values(), key=lambda t: t[1].name)]
