"""Render 15m candlestick charts with SMA/EMA 20/60/120 for YOLO detection.

Style follows the old project's TradingView-like renderer (cv2 drawing,
1280x742) but deliberately drops grids, axes and text so the only pixels on
the canvas are candles and the six moving averages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .data import ALL_MA_COLS

# BGR colors. Green up / red down candles matches the old project's TV style.
CANDLE_GREEN = (129, 153, 8)
CANDLE_RED = (69, 54, 242)
WICK = (118, 118, 118)
BG = (255, 255, 255)
# SMA in cool tones, EMA in warm tones; darker = shorter period.
MA_COLORS = {
    "sma20": (196, 114, 32),   # blue
    "sma60": (176, 168, 92),   # teal
    "sma120": (140, 110, 110), # slate
    "ema20": (36, 96, 240),    # orange-red
    "ema60": (60, 160, 250),   # orange
    "ema120": (150, 70, 200),  # purple-pink
}

IMG_WIDTH = 1280
IMG_HEIGHT = 742
MARGIN = 12  # small uniform margin; no room reserved for axes/text


@dataclass(frozen=True)
class ChartTransform:
    """Maps (bar index, price) to pixel coordinates for one rendered window."""

    n_bars: int
    width: int
    height: int
    left: int
    top: int
    plot_w: int
    plot_h: int
    price_min: float
    price_max: float
    candle_half_w: int

    def x_at(self, index: int) -> int:
        if self.n_bars <= 1:
            return self.left
        return int(self.left + (index / (self.n_bars - 1)) * self.plot_w)

    def y_at(self, price: float) -> int:
        span = max(self.price_max - self.price_min, 1e-12)
        return int(self.top + (self.price_max - float(price)) / span * self.plot_h)


# Minimum vertical span as a fraction of price. Without this floor, flat
# windows get extreme y-zoom and a numerically dense MA bundle (full_spread
# <= 0.55%) can fill most of the image, destroying the visual "pinched lines"
# signature the detector must learn (rule is relative, rendering must be too).
MIN_REL_SPAN = 0.06


def _price_bounds(df: pd.DataFrame, pad: float = 0.06) -> tuple[float, float]:
    series = [df["low"], df["high"]]
    series.extend(df[c] for c in ALL_MA_COLS if c in df.columns)
    values = pd.concat(series).dropna()
    lo, hi = float(values.min()), float(values.max())
    mid = (hi + lo) / 2
    span = max(hi - lo, abs(mid) * MIN_REL_SPAN, 1e-9)
    lo, hi = mid - span / 2, mid + span / 2
    return lo - span * pad, hi + span * pad


def make_chart_transform(
    df: pd.DataFrame,
    *,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
) -> ChartTransform:
    """Build the pixel transform for a window without drawing (relabel path)."""
    df = df.reset_index(drop=True)
    n = len(df)
    left = top = MARGIN
    plot_w, plot_h = width - 2 * MARGIN, height - 2 * MARGIN
    price_min, price_max = _price_bounds(df)
    candle_half_w = max(1, int(plot_w / max(n, 1) * 0.34))
    return ChartTransform(
        n_bars=n, width=width, height=height, left=left, top=top,
        plot_w=plot_w, plot_h=plot_h,
        price_min=price_min, price_max=price_max, candle_half_w=candle_half_w,
    )


def render_chart(
    df: pd.DataFrame,
    *,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
    out_path: str | Path | None = None,
) -> tuple[np.ndarray, ChartTransform]:
    """Render one window of candles + 6 MAs. df rows must already contain MA columns."""
    df = df.reset_index(drop=True)
    n = len(df)
    tf = make_chart_transform(df, width=width, height=height)
    candle_half_w = tf.candle_half_w

    img = np.full((height, width, 3), BG, dtype=np.uint8)
    for i in range(n):
        row = df.iloc[i]
        x = tf.x_at(i)
        yh, yl = tf.y_at(row["high"]), tf.y_at(row["low"])
        yo, yc = tf.y_at(row["open"]), tf.y_at(row["close"])
        color = CANDLE_GREEN if float(row["close"]) >= float(row["open"]) else CANDLE_RED
        cv2.line(img, (x, yh), (x, yl), WICK, 1, cv2.LINE_AA)
        y1, y2 = min(yo, yc), max(yo, yc)
        if y2 - y1 < 2:
            y2 = y1 + 2
        cv2.rectangle(img, (x - candle_half_w, y1), (x + candle_half_w, y2), color, -1, cv2.LINE_AA)

    for col in ALL_MA_COLS:
        if col not in df.columns:
            continue
        pts = [
            (tf.x_at(i), tf.y_at(float(v)))
            for i, v in enumerate(df[col])
            if pd.notna(v)
        ]
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, MA_COLORS[col], 1, cv2.LINE_AA)

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
    return img, tf
