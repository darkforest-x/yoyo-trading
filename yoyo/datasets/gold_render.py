"""Dual-image render: context (future allowed) vs local causal W30."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yoyo.datasets.window_render import SourceError, enrich, load_prefix
from yoyo.layers.l1_detection.data import ALL_MA_COLS
from yoyo.layers.l1_detection.render import IMG_HEIGHT, IMG_WIDTH, MARGIN, render_chart

HOLD_DEFAULT = pd.Timestamp("2026-05-04T00:00:00+00:00")
FOOTER_H = 56


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_causal_prefix(path: Path, last_bar: int) -> pd.DataFrame:
    """Bars through last_bar inclusive. Never used to pull holdout by itself."""
    return enrich(load_prefix(path, last_bar))


def clip_holdout(frame: pd.DataFrame, holdout: pd.Timestamp) -> pd.DataFrame:
    times = pd.to_datetime(frame["open_time"], utc=True)
    return frame.loc[times < holdout].reset_index(drop=True)


def assert_no_holdout(frame: pd.DataFrame, holdout: pd.Timestamp) -> None:
    times = pd.to_datetime(frame["open_time"], utc=True)
    if (times >= holdout).any():
        raise SourceError("attempted to materialize holdout bars")


def render_local(
    frame: pd.DataFrame,
    decision_bar: int,
    local_len: int = 30,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Causal W local window ending at decision. Zero future bars."""
    start = int(decision_bar) - int(local_len) + 1
    if start < 0:
        raise SourceError("not enough history for local window")
    window = frame.iloc[start : int(decision_bar) + 1].reset_index(drop=True)
    if len(window) != local_len:
        raise SourceError("local window length mismatch")
    image, _tf = render_chart(window, out_path=None)
    footer = _bar_footer(local_len, image.shape[1])
    stacked = np.vstack([image, footer])
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), stacked)
    return {
        "image": stacked,
        "local_start_bar": start,
        "local_end_bar": int(decision_bar),
        "local_window_length": local_len,
        "future_bars": 0,
        "height": stacked.shape[0],
        "width": stacked.shape[1],
    }


def _bar_footer(n: int, width: int) -> np.ndarray:
    """START/END click strip + 0..n-1 indices. Same x mapping as the chart."""
    footer = np.full((FOOTER_H, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(footer, (MARGIN, 4), (width - MARGIN, 28), (220, 220, 210), 1)
    cv2.putText(
        footer,
        "START/END click strip   model input only — decision and earlier",
        (MARGIN, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    plot_w = width - 2 * MARGIN
    for i in range(n):
        x = int(MARGIN + (i / max(n - 1, 1)) * plot_w)
        cv2.line(footer, (x, 30), (x, 36), (80, 80, 80), 1)
        if i % 5 == 0 or i == n - 1:
            cv2.putText(
                footer,
                str(i),
                (x - 6, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    return footer


def render_context(
    frame: pd.DataFrame,
    decision_bar: int,
    core_start: int | None,
    core_end: int | None,
    *,
    pre_bars: int = 50,
    post_bars: int = 120,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Reference chart. Future bars after decision are allowed; holdout is not."""
    anchor = int(core_start) if core_start is not None else int(decision_bar) - 6
    lo = max(0, anchor - pre_bars)
    hi = min(len(frame) - 1, int(core_end or decision_bar) + post_bars)
    segment = frame.iloc[lo : hi + 1].reset_index(drop=True)
    times = pd.to_datetime(segment["open_time"], utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    x = mdates.date2num(times)
    o, h, low, c = (segment[col].to_numpy(float) for col in ("open", "high", "low", "close"))
    width = (x[1] - x[0]) * 0.70 if len(x) > 1 else 0.01
    up = c >= o
    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=110)
    ax.vlines(x, low, h, color="#888888", lw=0.8, zorder=2)
    ax.bar(x[up], (c - o)[up], width, bottom=o[up], color="#26a69a", zorder=3)
    ax.bar(x[~up], (o - c)[~up], width, bottom=c[~up], color="#ef5350", zorder=3)
    colors = {
        "sma20": "#303f9f",
        "ema20": "#ef6c00",
        "sma60": "#039be5",
        "ema60": "#7cb342",
        "sma120": "#8e24aa",
        "ema120": "#d81b60",
    }
    for col in ALL_MA_COLS:
        ax.plot(x, segment[col], color=colors[col], lw=1.0, alpha=0.9)
    local_decision = int(decision_bar) - lo
    if 0 <= local_decision < len(x):
        ax.axvline(x[local_decision], color="#00838f", lw=1.6, ls="--")
        if local_decision + 1 < len(x):
            ax.axvspan(x[local_decision] + width / 2, x[-1] + width / 2, color="#7e57c2", alpha=0.07)
    if core_start is not None and core_end is not None:
        cs, ce = int(core_start) - lo, int(core_end) - lo
        if 0 <= cs <= ce < len(x):
            ax.axvspan(x[cs] - width / 2, x[ce] + width / 2, color="#90a4ae", alpha=0.18)
    ax.set_title(
        "context reference only — not model input   cyan/dash=decision   purple=after decision",
        loc="left",
        fontsize=10,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.grid(alpha=0.15)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
    plt.close(fig)
    return {
        "context_start_bar": lo,
        "context_end_bar": hi,
        "pre_bars": int(anchor - lo),
        "post_bars": int(hi - (core_end or decision_bar)),
        "n_bars": len(segment),
    }
