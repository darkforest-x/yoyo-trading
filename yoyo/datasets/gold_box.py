"""START/END bar range → full-wick + six-MA axis-aligned box."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from yoyo.layers.l1_detection.data import ALL_MA_COLS
from yoyo.layers.l1_detection.render import MARGIN, make_chart_transform

DEFAULT_PAD_FRAC = 0.04
EXTREME_WICK_RATIO = 8.0


class UnstableWickError(ValueError):
    """Extreme wick would make a silent crop; caller should IGNORE."""


def snap_x_to_bar(x_px: float, n_bars: int, width: int, margin: int = MARGIN) -> int:
    """Map a pixel x on the local image to the nearest bar index 0..n-1."""
    if n_bars < 1:
        raise ValueError("n_bars")
    plot_w = max(width - 2 * margin, 1)
    if n_bars == 1:
        return 0
    rel = (float(x_px) - margin) / plot_w
    index = int(round(rel * (n_bars - 1)))
    return max(0, min(n_bars - 1, index))


def bar_center_x(index: int, n_bars: int, width: int, margin: int = MARGIN) -> float:
    plot_w = width - 2 * margin
    if n_bars <= 1:
        return float(margin)
    return margin + (index / (n_bars - 1)) * plot_w


def core_price_bounds(window: pd.DataFrame, start: int, end: int) -> tuple[float, float]:
    sl = window.iloc[int(start) : int(end) + 1]
    highs = [sl["high"].max()]
    lows = [sl["low"].min()]
    for col in ALL_MA_COLS:
        if col in sl.columns:
            highs.append(sl[col].max())
            lows.append(sl[col].min())
    box_high = float(np.nanmax(np.asarray(highs, dtype=float)))
    box_low = float(np.nanmin(np.asarray(lows, dtype=float)))
    if not np.isfinite(box_high) or not np.isfinite(box_low) or box_high <= box_low:
        raise ValueError("cannot form a finite price box")
    return box_high, box_low


def extreme_wick(window: pd.DataFrame, start: int, end: int) -> bool:
    sl = window.iloc[int(start) : int(end) + 1]
    ranges = (sl["high"] - sl["low"]).astype(float)
    typical = float(ranges.median())
    if not np.isfinite(typical) or typical <= 0:
        return False
    return float(ranges.max()) / typical >= EXTREME_WICK_RATIO


def yolo_xywh(
    window: pd.DataFrame,
    start: int,
    end: int,
    *,
    pad_frac: float = DEFAULT_PAD_FRAC,
    allow_extreme_wick: bool = False,
) -> dict[str, Any]:
    """Return YOLO-normalized box plus geometry. Never silently clips wicks."""
    if extreme_wick(window, start, end) and not allow_extreme_wick:
        raise UnstableWickError("extreme wick; mark IGNORE instead of clipping")
    high, low = core_price_bounds(window, start, end)
    pad = (high - low) * float(pad_frac)
    high += pad
    low -= pad
    transform = make_chart_transform(window)
    left = transform.x_at(int(start)) - transform.candle_half_w
    right = transform.x_at(int(end)) + transform.candle_half_w
    y1 = transform.y_at(high)
    y2 = transform.y_at(low)
    x1 = float(np.clip(min(left, right), 0, transform.width - 1))
    x2 = float(np.clip(max(left, right), 1, transform.width))
    top, bot = float(min(y1, y2)), float(max(y1, y2))
    xc = ((x1 + x2) / 2) / transform.width
    yc = ((top + bot) / 2) / transform.height
    w = (x2 - x1) / transform.width
    h = (bot - top) / transform.height
    return {
        "xc": xc,
        "yc": yc,
        "w": w,
        "h": h,
        "box_high": high,
        "box_low": low,
        "label": f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n",
    }
