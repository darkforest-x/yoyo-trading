"""W10 model-image renderer. Reuses frozen candle/MA colors; new y-scale.

``yoyo.layers.l1_detection.render`` pixels stay untouched. Training images for
this task use a different vertical pad (5%, error on zero span) and never draw
overlays. Review overlays are a separate pass on a copy.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd

from yoyo.layers.l1_detection.data import ALL_MA_COLS
from yoyo.layers.l1_detection.render import (
    BG,
    CANDLE_GREEN,
    CANDLE_RED,
    ChartTransform,
    IMG_HEIGHT,
    IMG_WIDTH,
    MA_COLORS,
    MARGIN,
    WICK,
)

from .canonical import parse_time


class SpanError(ValueError):
    """Window has zero price/MA span; must not emit a blank image."""


def make_w10_transform(
    df: pd.DataFrame,
    *,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
    y_pad_frac: float = 0.05,
) -> ChartTransform:
    df = df.reset_index(drop=True)
    n = len(df)
    series = [df["low"], df["high"]]
    series.extend(df[c] for c in ALL_MA_COLS if c in df.columns)
    values = pd.concat(series).dropna()
    raw_high = float(values.max())
    raw_low = float(values.min())
    span = raw_high - raw_low
    if span == 0:
        raise SpanError("span == 0")
    y_max = raw_high + span * float(y_pad_frac)
    y_min = raw_low - span * float(y_pad_frac)
    left = top = MARGIN
    plot_w, plot_h = width - 2 * MARGIN, height - 2 * MARGIN
    candle_half_w = max(1, int(plot_w / max(n, 1) * 0.34))
    return ChartTransform(
        n_bars=n,
        width=width,
        height=height,
        left=left,
        top=top,
        plot_w=plot_w,
        plot_h=plot_h,
        price_min=y_min,
        price_max=y_max,
        candle_half_w=candle_half_w,
    )


def render_w10(
    window: pd.DataFrame,
    *,
    y_pad_frac: float = 0.05,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
    overlay: bool = False,
) -> tuple[np.ndarray, ChartTransform]:
    if len(window) != 10:
        raise ValueError(f"W10 window must have 10 bars, got {len(window)}")
    df = window.reset_index(drop=True)
    tf = make_w10_transform(df, width=width, height=height, y_pad_frac=y_pad_frac)
    img = np.full((height, width, 3), BG, dtype=np.uint8)
    for i in range(10):
        row = df.iloc[i]
        x = tf.x_at(i)
        yh, yl = tf.y_at(row["high"]), tf.y_at(row["low"])
        yo, yc = tf.y_at(row["open"]), tf.y_at(row["close"])
        color = CANDLE_GREEN if float(row["close"]) >= float(row["open"]) else CANDLE_RED
        cv2.line(img, (x, yh), (x, yl), WICK, 1, cv2.LINE_AA)
        y1, y2 = min(yo, yc), max(yo, yc)
        if y2 - y1 < 2:
            y2 = y1 + 2
        cv2.rectangle(
            img,
            (x - tf.candle_half_w, y1),
            (x + tf.candle_half_w, y2),
            color,
            -1,
            cv2.LINE_AA,
        )
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
    if overlay:
        img = draw_review_overlay(img, tf)
    return img, tf


def draw_review_overlay(img: np.ndarray, tf: ChartTransform) -> np.ndarray:
    out = img.copy()
    x0 = tf.x_at(5) - tf.candle_half_w
    x1 = tf.x_at(8) + tf.candle_half_w
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, tf.top), (x1, tf.top + tf.plot_h), (0, 196, 255), -1)
    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
    cv2.line(out, (tf.x_at(5), tf.top), (tf.x_at(5), tf.top + tf.plot_h), (27, 127, 58), 2)
    cv2.line(out, (tf.x_at(9), tf.top), (tf.x_at(9), tf.top + tf.plot_h), (40, 40, 198), 2)
    return out


def post_bars_allowed(decision_time: str, holdout, want: int) -> int:
    dt = parse_time(decision_time)
    hold = holdout if hasattr(holdout, "to_pydatetime") else pd.Timestamp(holdout)
    avail = int((hold - pd.Timestamp(dt)) / pd.Timedelta(minutes=15)) - 1
    return max(0, min(int(want), avail))


def assert_model_window_causal(window: pd.DataFrame, decision_time, holdout) -> None:
    times = pd.to_datetime(window["open_time"], utc=True)
    if (times >= holdout).any():
        raise ValueError("holdout bars in model window")
    last = times.iloc[-1]
    if last.to_pydatetime() > parse_time(str(decision_time)):
        raise ValueError("model window extends past decision")
