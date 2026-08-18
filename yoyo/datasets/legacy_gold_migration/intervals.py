"""Half-open core intervals and the locked W10 local mapping.

New contract: core is ``[start, end_exclusive)`` with length 4. Adapters must
convert whatever the source originally meant (inclusive end, pixel box, …)
before calling these helpers. This module never guesses source semantics.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from yoyo.layers.l1_detection.data import ALL_MA_COLS
from yoyo.layers.l1_detection.render import MARGIN


def inclusive_to_half_open(start: int, end_inclusive: int) -> tuple[int, int]:
    start_i = int(start)
    end_i = int(end_inclusive)
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    return start_i, end_i + 1


def center_four_bar(inclusive_lo: int, inclusive_hi: int) -> tuple[int, int]:
    """Center-crop an inclusive core onto a 4-bar half-open interval.

    Owner-short cores are already 4–7 bars taken from the box centre.
    Tonight's one-click path used max-spread suggestion; this helper does
    not look at prices.
    """
    lo, hi = int(inclusive_lo), int(inclusive_hi)
    if hi < lo:
        lo, hi = hi, lo
    length = hi - lo + 1
    if length < 4:
        raise ValueError(f"core shorter than 4 bars: {length}")
    start = lo + (length - 4) // 2
    return start, start + 4


def core_length(start: int, end_exclusive: int) -> int:
    return int(end_exclusive) - int(start)


def map_w10(core_start: int, core_end_exclusive: int) -> dict[str, int]:
    """Map a 4-bar half-open core onto the locked 10-bar window."""
    s = int(core_start)
    e = int(core_end_exclusive)
    if e - s != 4:
        raise ValueError(f"core length must be 4, got {e - s}")
    confirmation = e
    window_start = s - 5
    window_end_exclusive = e + 1
    return {
        "core_start_bar": s,
        "core_end_exclusive_bar": e,
        "core_length": 4,
        "confirmation_bar": confirmation,
        "decision_bar": confirmation,
        "window_start_bar": window_start,
        "window_end_exclusive_bar": window_end_exclusive,
        "window_length": window_end_exclusive - window_start,
        "local_core_start": 5,
        "local_core_end_exclusive": 9,
        "local_confirmation_position": 9,
    }


def candle_centers(n_bars: int, width: int, margin: int = MARGIN) -> list[float]:
    if n_bars <= 1:
        return [float(margin)]
    plot_w = width - 2 * margin
    return [margin + (i / (n_bars - 1)) * plot_w for i in range(n_bars)]


def yolo_box_to_bars(
    xc: float,
    box_w: float,
    n_bars: int,
    width: int,
    margin: int = MARGIN,
    *,
    eps: float = 1e-9,
) -> dict[str, Any]:
    """Map a normalized YOLO x-center/width onto bars whose centers fall inside."""
    x_min = (float(xc) - float(box_w) / 2.0) * width
    x_max = (float(xc) + float(box_w) / 2.0) * width
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    centers = candle_centers(n_bars, width, margin)
    inside: list[int] = []
    on_boundary: list[int] = []
    for i, center in enumerate(centers):
        if x_min - eps <= center <= x_max + eps:
            inside.append(i)
        if abs(center - x_min) <= eps or abs(center - x_max) <= eps:
            on_boundary.append(i)
    if not inside:
        return {
            "local_start": None,
            "local_end_exclusive": None,
            "ambiguous_boundary": bool(on_boundary),
            "x_min": x_min,
            "x_max": x_max,
            "inside": [],
        }
    return {
        "local_start": inside[0],
        "local_end_exclusive": inside[-1] + 1,
        "ambiguous_boundary": bool(on_boundary),
        "x_min": x_min,
        "x_max": x_max,
        "inside": inside,
    }


def _window_spread(frame: pd.DataFrame, start: int, end_exclusive: int) -> float:
    sl = frame.iloc[int(start) : int(end_exclusive)]
    if sl.empty:
        return float("inf")
    if "full_spread" in sl.columns and sl["full_spread"].notna().any():
        return float(np.nanmean(sl["full_spread"].to_numpy(dtype=float)))
    cols = [c for c in ALL_MA_COLS if c in sl.columns]
    if not cols:
        return float("inf")
    mas = sl[cols].to_numpy(dtype=float)
    close = sl["close"].replace(0, np.nan).to_numpy(dtype=float)
    spread = (np.nanmax(mas, axis=1) - np.nanmin(mas, axis=1)) / close
    return float(np.nanmean(spread))


def suggest_four_bar(
    frame: pd.DataFrame,
    core_start: int,
    core_end_exclusive: int,
) -> dict[str, Any] | None:
    """Deterministic 4-bar suggestion. Never auto-confirms."""
    lo = int(core_start) - 1
    hi = int(core_end_exclusive) + 1
    old_center = (int(core_start) + int(core_end_exclusive) - 1) / 2.0
    n = len(frame)
    best: dict[str, Any] | None = None
    for start in range(lo, hi - 3):
        end = start + 4
        if start < 0 or end > n:
            continue
        spread = _window_spread(frame, start, end)
        center = (start + end - 1) / 2.0
        dist = abs(center - old_center)
        key = (spread, dist, -start)
        cand = {
            "core_start_bar": start,
            "core_end_exclusive_bar": end,
            "spread": spread,
            "center_distance": dist,
        }
        if best is None or key < best["_key"]:
            cand["_key"] = key
            best = cand
    if best is None:
        return None
    best.pop("_key", None)
    return best


def cores_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return int(a0) < int(b1) and int(b0) < int(a1)


def window_overlaps_any(ws: int, we: int, protected: Sequence[tuple[int, int]]) -> bool:
    for plo, phi in protected:
        if we > plo and ws < phi:
            return True
    return False


def nearest_empty_confirmation(
    *,
    decision: int,
    n: int,
    warmup: int,
    protected: Sequence[tuple[int, int]],
    radius: int,
) -> int | None:
    """Closest confirmation bar whose W10 misses every protected interval."""
    for dist in range(1, int(radius) + 1):
        for cand in (int(decision) - dist, int(decision) + dist):
            we = cand + 1
            ws = cand - 9
            if ws < warmup or we > n or cand == decision:
                continue
            if window_overlaps_any(ws, we, protected):
                continue
            return cand
    return None


def cluster_event_groups(
    rows: Sequence[dict[str, Any]],
    *,
    center_max_bars: int = 6,
) -> list[list[int]]:
    """Union-find groups: same symbol+tf and (overlap or center distance <= N)."""
    n = len(rows)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    def span(row: dict[str, Any]) -> tuple[int, int] | None:
        s = row.get("core_start_bar")
        e = row.get("core_end_exclusive_bar")
        if s is None or e is None:
            slot_s = row.get("slot_start_bar")
            slot_e = row.get("slot_end_exclusive_bar")
            if slot_s is None or slot_e is None:
                return None
            return int(slot_s), int(slot_e)
        return int(s), int(e)

    by_sym: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(rows):
        key = (str(row.get("symbol")), str(row.get("timeframe") or "15m"))
        by_sym.setdefault(key, []).append(i)
    for idxs in by_sym.values():
        for a in range(len(idxs)):
            ia = idxs[a]
            sa = span(rows[ia])
            if sa is None:
                continue
            ca = (sa[0] + sa[1] - 1) / 2.0
            for b in range(a + 1, len(idxs)):
                ib = idxs[b]
                sb = span(rows[ib])
                if sb is None:
                    continue
                cb = (sb[0] + sb[1] - 1) / 2.0
                if cores_overlap(*sa, *sb) or abs(ca - cb) <= center_max_bars:
                    union(ia, ib)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
