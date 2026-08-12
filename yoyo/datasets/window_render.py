"""Turn one causal bar window into the exact bytes the detector was trained on.

This is the narrow waist of Dataset V3: every image and every label must come
through here, so that "the dataset changed" can only ever mean the sample set or
the labels changed -- never the pixels. The renderer itself
(``yoyo.layers.l1_detection.render``) is frozen; this module only decides which
rows go into it and how a bar range becomes a YOLO box.

The three rules that make a rebuild reproducible, all inherited from the
fable-trading builders this replaces:

* load the CSV **prefix** up to the window's last bar and no further, so a
  redraw of an old crop cannot materialise post-boundary rows;
* compute MAs on that whole prefix, never on the crop, so MA values do not
  depend on where the window starts;
* derive the box from the same ``ChartTransform`` that drew the image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yoyo.data.loader import OHLCV_COLUMNS
from yoyo.layers.l1_detection.data import add_mas
from yoyo.layers.l1_detection.render import render_chart

MIN_BOX_PIXELS = 4


class SourceError(RuntimeError):
    """The source series cannot support this window; never silently skipped."""


def load_prefix(path: Path, required_end: int) -> pd.DataFrame:
    """Load the CSV prefix through global index ``required_end`` inclusive."""
    raw = pd.read_csv(path, nrows=required_end + 1, encoding_errors="replace")
    if not set(OHLCV_COLUMNS).issubset(raw.columns):
        raise SourceError(f"bad source schema: {path}")
    if "confirm" in raw.columns:
        raw = raw[raw["confirm"] != 0]
    frame = raw[OHLCV_COLUMNS].copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["open_time", "open", "high", "low", "close"])
        .drop_duplicates("open_time", keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    if len(frame) <= required_end:
        raise SourceError(
            f"prefix index mismatch: {path} has {len(frame)} usable rows, need > {required_end}"
        )
    return frame


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the six MAs on the full prefix -- never on the cropped window."""
    return add_mas(frame)


def window_slice(enriched: pd.DataFrame, win_start: int, win_end: int) -> pd.DataFrame:
    if win_start < 0 or win_end < win_start:
        raise SourceError(f"bad window: [{win_start}, {win_end}]")
    if win_end >= len(enriched):
        raise SourceError(f"window end {win_end} beyond series length {len(enriched)}")
    return enriched.iloc[win_start : win_end + 1].reset_index(drop=True)


def yolo_box_from_bars(transform, window: pd.DataFrame, b0: int, b1: int):
    """Axis-aligned YOLO box covering local bars [b0, b1] by their high/low.

    Ported unchanged from ``scripts/build_w20_midbox_dataset.py``: the detector's
    labels are defined by this arithmetic, so a "cleaner" rewrite would silently
    move every box in the dataset.
    """
    n = len(window)
    b0 = int(max(0, b0))
    b1 = int(min(n - 1, b1))
    if b1 < b0:
        return None
    segment = window.iloc[b0 : b1 + 1]
    high = float(segment["high"].max())
    low = float(segment["low"].min())
    if not np.isfinite(high) or not np.isfinite(low) or high <= low:
        return None
    x1 = transform.x_at(b0) - transform.candle_half_w
    x2 = transform.x_at(b1) + transform.candle_half_w
    y1 = transform.y_at(high)
    y2 = transform.y_at(low)
    x1 = float(np.clip(x1, 0, transform.width - 1))
    x2 = float(np.clip(x2, 1, transform.width))
    y1 = float(np.clip(y1, 0, transform.height - 1))
    y2 = float(np.clip(y2, 1, transform.height))
    if x2 - x1 < MIN_BOX_PIXELS or abs(y2 - y1) < MIN_BOX_PIXELS:
        return None
    return (
        (x1 + x2) / 2 / transform.width,
        (y1 + y2) / 2 / transform.height,
        (x2 - x1) / transform.width,
        abs(y2 - y1) / transform.height,
    )


def label_text(box: tuple[float, float, float, float] | None) -> str:
    """YOLO label body. A negative sample is an empty file, not a missing one."""
    if box is None:
        return ""
    return f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"


def render_sample(
    enriched: pd.DataFrame,
    win_start: int,
    win_end: int,
    core_global: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Render one window; ``core_global`` present means a positive sample."""
    window = window_slice(enriched, win_start, win_end)
    image, transform = render_chart(window, out_path=None)
    box = None
    if core_global is not None:
        core_start, core_end = (int(value) for value in core_global)
        box = yolo_box_from_bars(
            transform, window, core_start - win_start, core_end - win_start
        )
        if box is None:
            raise SourceError(f"empty positive box for window [{win_start}, {win_end}]")
    return {
        "image": image,
        "box": box,
        "label": label_text(box),
        "bars": len(window),
        "visible_end_bar": win_end,
    }
