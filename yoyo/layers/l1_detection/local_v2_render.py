"""Render causal Local-Signal V2 charts with explicit right-side blank slots.

The legacy detector renderer in :mod:`yoyo.layers.l1_detection.render` is a
pixel-frozen contract.  Local-Signal V2 needs a different layout contract:
only bars at or before the decision point are visible, while a deterministic
number of empty plot slots may remain on the right.  Blank slots change canvas
geometry without appending future candles or altering price bounds.

Inputs use OHLC plus the six moving-average columns over the supplied causal
window.  The function never reads rows outside that window.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .data import ALL_MA_COLS
from .render import (
    BG,
    CANDLE_GREEN,
    CANDLE_RED,
    IMG_HEIGHT,
    IMG_WIDTH,
    MA_COLORS,
    WICK,
    ChartTransform,
    make_chart_transform,
)


def render_causal_chart(
    df: pd.DataFrame,
    *,
    right_blank_slots: int,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
    out_path: str | Path | None = None,
) -> tuple[np.ndarray, ChartTransform]:
    """Render causal bars into ``len(df) + right_blank_slots`` plot slots.

    ``right_blank_slots`` represents layout only: no synthetic or future bar
    is appended to ``df``.  The returned transform maps a real bar index to
    its canvas position and therefore reports the total slot count through
    ``n_bars``.
    """
    if isinstance(right_blank_slots, bool) or not isinstance(right_blank_slots, int):
        raise TypeError("right_blank_slots must be an integer")
    if right_blank_slots < 0:
        raise ValueError("right_blank_slots must be non-negative")
    frame = df.reset_index(drop=True)
    visible_bars = len(frame)
    if visible_bars == 0:
        raise ValueError("cannot render an empty frame")
    total_slots = visible_bars + right_blank_slots
    base = make_chart_transform(frame, width=width, height=height)
    transform = replace(
        base,
        n_bars=total_slots,
        candle_half_w=max(1, int(base.plot_w / total_slots * 0.34)),
    )

    image = np.full((height, width, 3), BG, dtype=np.uint8)
    for index in range(visible_bars):
        row = frame.iloc[index]
        x = transform.x_at(index)
        y_high = transform.y_at(row["high"])
        y_low = transform.y_at(row["low"])
        y_open = transform.y_at(row["open"])
        y_close = transform.y_at(row["close"])
        color = (
            CANDLE_GREEN
            if float(row["close"]) >= float(row["open"])
            else CANDLE_RED
        )
        cv2.line(image, (x, y_high), (x, y_low), WICK, 1, cv2.LINE_AA)
        y1, y2 = min(y_open, y_close), max(y_open, y_close)
        if y2 - y1 < 2:
            y2 = y1 + 2
        cv2.rectangle(
            image,
            (x - transform.candle_half_w, y1),
            (x + transform.candle_half_w, y2),
            color,
            -1,
            cv2.LINE_AA,
        )

    for column in ALL_MA_COLS:
        if column not in frame.columns:
            continue
        points = [
            (transform.x_at(index), transform.y_at(float(value)))
            for index, value in enumerate(frame[column])
            if pd.notna(value)
        ]
        if len(points) >= 2:
            cv2.polylines(
                image,
                [np.array(points, dtype=np.int32)],
                False,
                MA_COLORS[column],
                1,
                cv2.LINE_AA,
            )

    if out_path is not None:
        destination = Path(out_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), image)
    return image, transform
