"""Phase 1 of a pulse: series in, candidate bar indices out. Nothing else.

Extracted from forward_scan.py, where discovery and scoring shared one 725-line
function and therefore one namespace. That file is where the 2026-08-03 fault
lived: L2's question (which coordinate system was this model trained in) was being
answered with L1's fact (is this trade long or short), and nothing objected
because both were local variables.

The thread pool comes along rather than being redesigned. Discovery is where the
wall clock goes -- indicators, rendering, YOLO inference -- and the pulse has a
fifteen-minute budget that is an iron rule, not a preference: overrunning it does
not make the system slow, it structurally blocks tip signals. So the concurrency
that was measured and tuned stays exactly as it was, and only the boundary moves.
discover_wall is still printed here, from the same two clock reads.

What changed is the interface. `_discover` used to close over six values from an
enclosing function; they are now parameters, which is what makes this callable
without dragging L2 in.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from yoyo.data.indicators import MIN_GAP_BARS, WARMUP_BARS, add_indicators, strict_mask
from yoyo.layers.l1_detection.candidates import (
    enforce_global_tip_age,
    get_global_tip_age_rejected,
    get_tip_edge_rejected,
    reset_global_tip_age_rejected,
    reset_tip_edge_rejected,
    resolve_tip_conf,
    scan_series_with_yolo,
)


@dataclass(frozen=True)
class DiscoveredSeries:
    """One series after discovery. Carries the frames so phase 2 need not reload."""

    __slots__ = ("source", "symbol", "frame", "enriched", "signal_indices", "detected_at")

    source: str
    symbol: str
    frame: pd.DataFrame
    enriched: pd.DataFrame
    signal_indices: list[int]
    # Per series, not per batch: a scan across 344 symbols finishes them minutes
    # apart, and one shared timestamp claims detections that had not happened yet.
    detected_at: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rule_candidate_indices(enriched: pd.DataFrame) -> list[int]:
    """Research-only rules path. Production is YOLO (iron rule 12)."""
    if len(enriched) < WARMUP_BARS + 2:
        return []
    mask = strict_mask(enriched, mode="expanded").fillna(False)
    idx = np.flatnonzero(mask.to_numpy())
    # live fallback path: the tip bar is a valid signal (entry backfills next pulse)
    idx = idx[(idx >= WARMUP_BARS) & (idx < len(enriched))]
    if len(idx) == 0:
        return []
    scores = enriched["shape_score"].to_numpy()
    selected: list[int] = []
    for signal_i in sorted(idx, key=lambda item: scores[item], reverse=True):
        if all(abs(signal_i - previous) >= MIN_GAP_BARS for previous in selected):
            selected.append(int(signal_i))
    return sorted(selected)


def candidate_indices(
    enriched: pd.DataFrame,
    *,
    candidate_source: str,
    frame: pd.DataFrame | None = None,
    yolo_model=None,
    start_time: pd.Timestamp | None = None,
    yolo_mode: str = "live",
    max_tip_age_bars: int = 2,
) -> list[int]:
    """Candidate bars for one series.

    candidate_source is passed in rather than read from the environment here: the
    validation of production-vs-research provenance belongs to whoever assembles
    the pulse, and a detection module that silently consults os.environ is one
    more thing that can disagree with the rest of the run.
    """
    if candidate_source == "rules":
        return rule_candidate_indices(enriched)

    raw = frame if frame is not None else enriched
    start_from_i = None
    if start_time is not None and "open_time" in raw.columns:
        times = pd.to_datetime(raw["open_time"], utc=True)
        st = pd.Timestamp(start_time)
        st = st.tz_localize("UTC") if st.tzinfo is None else st.tz_convert("UTC")
        hits = np.flatnonzero(times >= st)
        if len(hits) == 0:
            # FORWARD_START often sits *inside* the still-open 15m bar (e.g. start
            # 16:30 while the last closed open_time is 16:15). Returning [] here
            # blanked the whole live gate after the 2026-07-19 retest clock reset
            # (candidates_seen=0 across 344 series). Still scan the tip; the score
            # stage drops signal_time < start_time for new rows anyway.
            start_from_i = max(0, len(raw) - 10)
        else:
            start_from_i = max(0, int(hits[0]) - 5)

    mode = yolo_mode if yolo_mode in ("live", "tip", "full") else "live"
    indices = scan_series_with_yolo(
        raw, yolo_model, start_from_i=start_from_i, mode=mode, tip_conf=resolve_tip_conf()
    )
    if mode in ("live", "tip"):
        return enforce_global_tip_age(
            indices, latest_closed_i=len(raw) - 1, max_age_bars=max_tip_age_bars
        )
    return indices


def discover(
    jobs: list[tuple[str, str, pd.DataFrame]],
    *,
    candidate_source: str,
    yolo_model,
    start_time: pd.Timestamp | None,
    yolo_mode: str,
    max_tip_age_bars: int,
    tracked_keys: set,
    workers: int,
    verbose: bool = True,
) -> list[DiscoveredSeries]:
    """Run discovery across all series. Concurrency and its clock live here.

    tracked_keys re-admits bars that already have an open row, so a signal keeps
    resolving after the detector stops proposing it.
    """
    reset_tip_edge_rejected()
    reset_global_tip_age_rejected()

    def _one(job: tuple[str, str, pd.DataFrame]) -> DiscoveredSeries:
        source, symbol, frame = job
        enriched = add_indicators(frame)
        if candidate_source == "yolo" and yolo_model is None:
            # detector=none idle mode (iron rule 12): no discovery, but rows that
            # are already open still resolve below.
            found: set[int] = set()
        else:
            found = set(
                candidate_indices(
                    enriched,
                    candidate_source=candidate_source,
                    frame=frame,
                    yolo_model=yolo_model,
                    start_time=start_time,
                    yolo_mode=yolo_mode,
                    max_tip_age_bars=max_tip_age_bars,
                )
            )
        tracked_times = {k[2] for k in tracked_keys if k[0] == source and k[1] == symbol}
        if tracked_times:
            signal_times = enriched["open_time"].astype(str)
            found.update(int(i) for i in signal_times[signal_times.isin(tracked_times)].index)
        return DiscoveredSeries(source, symbol, frame, enriched, sorted(found), _utc_now_iso())

    t0 = time.monotonic()
    if workers <= 1:
        out = [_one(job) for job in jobs]
    else:
        out = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one, job) for job in jobs]
            for future in as_completed(futures):
                out.append(future.result())
    if verbose:
        print(
            f"forward_scan: discover_wall={time.monotonic() - t0:.0f}s "
            f"(indicators+render+predict, {workers} workers) "
            f"tip_edge_rejected={get_tip_edge_rejected()} "
            f"global_tip_age_rejected={get_global_tip_age_rejected()}",
            flush=True,
        )
    return out
