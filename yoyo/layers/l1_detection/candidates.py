"""YOLO detector as judgment-layer candidate source (mainline after 2026-07-15).

Replaces rule `scan_candidates` / `forward_candidate_indices` for the critical
path. Downstream labeling, features, LightGBM freeze, and TP5/SL2 exits are
unchanged — only *which bars* are proposed as signals differs.

Requires ultralytics/torch (use `.venv/bin/python` for any path that calls
`scan_series_with_yolo`).
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yoyo.layers.l1_detection.data import add_mas
from yoyo.layers.l1_detection.render import render_chart
from yoyo.data.indicators import MIN_GAP_BARS, WARMUP_BARS
from yoyo.contracts.outcomes import HORIZON_BARS
from yoyo.contracts.paths import data_path, data_root  # noqa: F401

PROJECT_DIR = data_root()
WINDOW = 200
STRIDE = 50
DEFAULT_CONF = 0.30
DEFAULT_WEIGHTS = PROJECT_DIR / "models" / "owner_best.pt"
# Owner 2026-07-31: L1 interim discovery weight when owner_best pointer is
# absent. Not a tip-smoke gold promote — paper + live scan share short_star_v10.
V10_PAPER_WEIGHTS = PROJECT_DIR / "models" / "owner_short_star_v10.pt"
V16_TIPUNI_WEIGHTS = PROJECT_DIR / "models" / "owner_v16_tipuni_cold.pt"


def resolve_default_weights() -> Path:
    """Mainline detector weights: env → owner_best.pt → short_star_v10 → v16 cold.

    Owner directed 2026-07-31: do not idle at detector=none while v10 paper
    weights exist. Formal tip-gold graduation still requires tip-smoke + owner
    promote; this only restores a usable discovery weight for scan/probe/paper.
    """
    env = (os.environ.get("FABLE_YOLO_WEIGHTS") or "").strip()
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = PROJECT_DIR / p
        if p.is_file():
            return p
        raise FileNotFoundError(f"FABLE_YOLO_WEIGHTS missing: {p}")
    for cand in (DEFAULT_WEIGHTS, V10_PAPER_WEIGHTS, V16_TIPUNI_WEIGHTS):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "YOLO weights missing: tried owner_best.pt, "
        "owner_short_star_v10.pt, owner_v16_tipuni_cold.pt "
        "(or set FABLE_YOLO_WEIGHTS)"
    )


def default_weights_label(path: Path | None = None) -> str:
    """Short label for logs / dashboard (v10 / owner_best / basename)."""
    try:
        p = path if path is not None else resolve_default_weights()
    except FileNotFoundError:
        return "none"
    try:
        resolved = p.resolve().name
    except OSError:
        resolved = p.name
    if "short_star_v10" in p.name or "short_star_v10" in resolved:
        return "owner_short_star_v10"
    if "v16_tipuni" in p.name or "v16_tipuni" in resolved:
        return "owner_v16_tipuni_cold"
    if p.name == "owner_best.pt":
        return "owner_best"
    return p.name
# A′ tip-edge gate (owner-approved 2026-07-21): only accept boxes whose
# mapped signal bar sits in the last N bars of the scan window.
# Source: analysis/p_box_to_bar_lag.md — KORU right_norm≈97.5% still mapped
# 3 bars back of tip, so the gate is bar-offset based, not pixel %.
# N=2 → bar_in_win >= window-2 (tip or tip-1; offset 0..1). Does NOT invent
# tip fires when the model draws 0 boxes on tip/tip-1.
TIP_EDGE_BARS = 2
# Optional tip-window conf floor (env TIP_CONF). When set, the tip window
# (rightmost start) may use a lower floor than other live windows; predict
# runs at min(tip_conf, conf) then non-tip windows post-filter to `conf`.
# Unset → every window uses the same `conf` (default 0.30).
# Optional right-bias (env FABLE_YOLO_RIGHT_BIAS=1): within min_gap, keep the
# rightmost signal instead of the leftmost (live multi-window only).
# Base temp dir; each predict call uses a unique filename (thread-safe live scan).
_TMP_DIR = PROJECT_DIR / "data"


def resolve_yolo_mode(default: str = "live") -> str:
    """Mainline scan mode from FABLE_YOLO_MODE (tip|live|full). Default live."""
    raw = os.environ.get("FABLE_YOLO_MODE", "").strip().lower()
    if raw in ("tip", "live", "full"):
        return raw
    return default if default in ("tip", "live", "full") else "live"


def resolve_tip_conf(fallback: float | None = None) -> float | None:
    """Optional tip-window conf from TIP_CONF. None = use the shared conf."""
    raw = os.environ.get("TIP_CONF", "").strip()
    if not raw:
        return fallback
    try:
        v = float(raw)
    except ValueError:
        return fallback
    if not (0.0 < v < 1.0):
        return fallback
    return v


def resolve_right_bias(default: bool = False) -> bool:
    """Prefer rightmost signal within min_gap when FABLE_YOLO_RIGHT_BIAS=1."""
    raw = os.environ.get("FABLE_YOLO_RIGHT_BIAS", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default

_model_cache: dict[str, Any] = {}
_predict_lock = threading.Lock()
_predict_device: str | None = None
_tip_edge_lock = threading.Lock()
_tip_edge_rejected_total = 0
_global_tip_age_rejected_total = 0


def reset_tip_edge_rejected() -> None:
    """Zero the process-wide tip_edge_rejected counter (call before a pulse)."""
    global _tip_edge_rejected_total
    with _tip_edge_lock:
        _tip_edge_rejected_total = 0


def reset_global_tip_age_rejected() -> None:
    """Zero final whole-series age rejects before one forward pulse."""
    global _global_tip_age_rejected_total
    with _tip_edge_lock:
        _global_tip_age_rejected_total = 0


def get_tip_edge_rejected() -> int:
    """Boxes dropped by the A′ tip-edge gate since last reset."""
    with _tip_edge_lock:
        return _tip_edge_rejected_total


def get_global_tip_age_rejected() -> int:
    """Candidates locally valid but too old relative to the global closed tip."""
    with _tip_edge_lock:
        return _global_tip_age_rejected_total


def _bump_tip_edge_rejected(n: int = 1) -> None:
    global _tip_edge_rejected_total
    if n <= 0:
        return
    with _tip_edge_lock:
        _tip_edge_rejected_total += n


def enforce_global_tip_age(
    indices: list[int], *, latest_closed_i: int, max_age_bars: int
) -> list[int]:
    """Final live assertion after every local box-to-bar mapping and dedupe.

    A box on local bar 198 can map to global tip-3 when its scan window starts
    two bars behind the tip. The local A-prime edge gate therefore cannot prove
    global freshness. This function is the final authority for live/tip output.
    """
    global _global_tip_age_rejected_total
    latest = int(latest_closed_i)
    cap = int(max_age_bars)
    if latest < 0:
        return []
    if cap < 0:
        raise ValueError("max_age_bars must be non-negative")
    accepted = sorted({int(i) for i in indices if 0 <= int(i) <= latest})
    kept = [i for i in accepted if latest - i <= cap]
    rejected = len(accepted) - len(kept)
    if rejected:
        with _tip_edge_lock:
            _global_tip_age_rejected_total += rejected
    return kept


def _resolve_predict_device() -> str:
    """Prefer CUDA on VPS; fall back to CPU (MPS has hung multi-series scans)."""
    global _predict_device
    if _predict_device is not None:
        return _predict_device
    forced = os.environ.get("FABLE_YOLO_DEVICE", "").strip()
    if forced:
        _predict_device = forced
        return _predict_device
    try:
        import torch

        if torch.cuda.is_available():
            _predict_device = "0"
            return _predict_device
    except Exception:  # noqa: BLE001
        pass
    _predict_device = "cpu"
    return _predict_device


def right_edge_to_bar(cx: float, w: float, tf, *, n_bars: int) -> int:
    """Normalized box right edge -> bar index within the window."""
    right_px = (cx + w / 2) * tf.width
    if tf.plot_w <= 0:
        return n_bars - 1
    idx = round((right_px - tf.left) / tf.plot_w * (tf.n_bars - 1))
    return int(min(max(idx, 0), tf.n_bars - 1))


@dataclass(frozen=True)
class BoxSignalMapping:
    """Pure, auditable result of mapping one detector box to a series bar."""

    accepted: bool
    rejection_reason: str
    window_start_i: int
    window_end_i: int
    bar_in_window: int
    mapped_signal_i: int
    latest_closed_i: int
    global_tip_age_bars: int


def map_box_to_signal(
    *,
    cx: float,
    w: float,
    tf: Any,
    window_start_i: int,
    n_bars: int,
    frame_length: int,
    latest_closed_i: int,
    tip_edge_bars: int = TIP_EDGE_BARS,
    apply_tip_edge: bool = True,
    max_global_tip_age_bars: int | None = None,
    allow_pending_entry: bool = True,
    warmup_bars: int = WARMUP_BARS,
    start_from_i: int | None = None,
    signal_i_lo: int | None = None,
    signal_i_hi: int | None = None,
) -> BoxSignalMapping:
    """Map a normalized box right edge to one global signal index.

    This is the single box→bar authority for both live discovery and P1
    offline replay.  It uses only detector geometry plus explicit positional
    bounds; no candle after ``mapped_signal_i`` is inspected.  ``latest_closed_i``
    is the whole-series tip for the pulse, not the local window end.
    """
    start = int(window_start_i)
    count = int(n_bars)
    latest = int(latest_closed_i)
    window_end = start + count - 1
    bar_in_window = right_edge_to_bar(float(cx), float(w), tf, n_bars=count)
    signal_i = start + bar_in_window
    age = latest - signal_i

    reason = ""
    edge = max(0, int(tip_edge_bars))
    if apply_tip_edge and edge > 0 and bar_in_window < count - edge:
        reason = "local_tip_edge"
    elif signal_i < int(warmup_bars) or signal_i >= int(frame_length):
        reason = "series_bounds"
    elif not allow_pending_entry and signal_i + 1 >= int(frame_length):
        reason = "missing_entry_bar"
    elif start_from_i is not None and signal_i < int(start_from_i):
        reason = "before_start"
    elif signal_i_lo is not None and signal_i < int(signal_i_lo):
        reason = "before_signal_lo"
    elif signal_i_hi is not None and signal_i > int(signal_i_hi):
        reason = "after_signal_hi"
    elif age < 0:
        reason = "after_latest_closed"
    elif max_global_tip_age_bars is not None:
        cap = int(max_global_tip_age_bars)
        if cap < 0:
            raise ValueError("max_global_tip_age_bars must be non-negative")
        if age > cap:
            reason = "global_tip_age"

    return BoxSignalMapping(
        accepted=not reason,
        rejection_reason=reason,
        window_start_i=start,
        window_end_i=window_end,
        bar_in_window=bar_in_window,
        mapped_signal_i=signal_i,
        latest_closed_i=latest,
        global_tip_age_bars=age,
    )


def load_yolo_model(weights: str | Path | None = None):
    """Lazy-load and cache YOLO weights (heavy import kept local)."""
    if weights is not None:
        path = str(Path(weights))
    else:
        path = str(resolve_default_weights())
    if path not in _model_cache:
        from ultralytics import YOLO

        if not Path(path).exists():
            raise FileNotFoundError(f"YOLO weights missing: {path}")
        _model_cache[path] = YOLO(path)
    return _model_cache[path]


def scan_series_with_yolo(
    frame: pd.DataFrame,
    model=None,
    *,
    conf: float = DEFAULT_CONF,
    window: int = WINDOW,
    stride: int = STRIDE,
    min_gap: int = MIN_GAP_BARS,
    tmp_png: Path | None = None,
    start_from_i: int | None = None,
    mode: str = "full",
    tip_edge_bars: int = TIP_EDGE_BARS,
    tip_conf: float | None = None,
    right_bias: bool | None = None,
    signal_time_lo: pd.Timestamp | None = None,
    signal_time_hi: pd.Timestamp | None = None,
    max_global_tip_age_bars: int | None = None,
) -> list[int]:
    """Return sorted signal bar indices for one OHLCV frame (causal at each bar).

    mode:
      - "full": offline dataset build (stride over history); no tip-edge gate
      - "live": forward/mainline — only windows near the right edge (and
        covering start_from_i..end). Avoids multi-hour full-history scans.
      - "tip": single rightmost window only (H-TIP / FABLE_YOLO_MODE=tip;
        ~1 predict/series vs live's ≤6). Allows tip-bar pending entry like live.

    tip_edge_bars (live/tip only): keep boxes with
    ``bar_in_win >= window - tip_edge_bars`` (default TIP_EDGE_BARS=2).
    See analysis/p_box_to_bar_lag.md option A′. Rejected boxes increment
    ``tip_edge_rejected`` (get_tip_edge_rejected). Set 0 to disable.

    tip_conf: optional lower floor for the tip window only (else env TIP_CONF).
    right_bias: within min_gap keep rightmost signal (else env FABLE_YOLO_RIGHT_BIAS).

    signal_time_lo / signal_time_hi (optional, hi exclusive): only emit signal
    bars whose open_time is in ``[lo, hi)``. Warmup / rendering still use bars
    before lo (full frame kept); only the slid-window schedule is clipped so
    windows that cannot produce an in-range signal are skipped.
    """
    if model is None:
        model = load_yolo_model()
    if len(frame) < WARMUP_BARS + window + 2:
        return []
    if tip_conf is None:
        tip_conf = resolve_tip_conf()
    if right_bias is None:
        right_bias = resolve_right_bias(False)
    # Optional signal-time gate → bar index bounds (inclusive lo, inclusive hi).
    i_lo: int | None = None
    i_hi: int | None = None
    if signal_time_lo is not None or signal_time_hi is not None:
        if "open_time" not in frame.columns:
            return []
        times = pd.to_datetime(frame["open_time"], utc=True)
        mask = pd.Series(True, index=frame.index)
        if signal_time_lo is not None:
            lo = pd.Timestamp(signal_time_lo)
            if lo.tzinfo is None:
                lo = lo.tz_localize("UTC")
            else:
                lo = lo.tz_convert("UTC")
            mask &= times >= lo
        if signal_time_hi is not None:
            hi = pd.Timestamp(signal_time_hi)
            if hi.tzinfo is None:
                hi = hi.tz_localize("UTC")
            else:
                hi = hi.tz_convert("UTC")
            mask &= times < hi
        idxs = np.flatnonzero(mask.to_numpy())
        if len(idxs) == 0:
            return []
        i_lo = int(idxs[0])
        i_hi = int(idxs[-1])
    enriched_ma = add_mas(frame)
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    # Unique path per call (thread id + pid) so parallel live scans never clobber.
    if tmp_png is None:
        tmp_png = _TMP_DIR / f"_yolo_cand_tmp_{os.getpid()}_{threading.get_ident()}.png"
    last_start = len(frame) - window
    tip_last_start = last_start  # true series tip (for tip-window conf / tip mode)
    first_start = WARMUP_BARS
    if start_from_i is not None:
        first_start = max(first_start, int(start_from_i) - window + 1)
    if i_lo is not None and i_hi is not None:
        # Windows that can map a box onto [i_lo, i_hi]; pre-lo bars still render.
        first_start = max(first_start, i_lo - window + 1)
        last_start = min(last_start, i_hi)
        if last_start < first_start:
            return []
    if mode == "tip":
        # Tip-only: single rightmost window (right edge = last closed bar).
        starts = [tip_last_start] if tip_last_start >= first_start else []
    elif mode == "live":
        # Live schedule (2026-07-20): pin the tip and two bars back, then
        # coarse stride for context — at most 6 windows. The 14-window
        # "tip-dense" schedule (backs 0..21 + half-stride walk) rested on a
        # false premise: a box's right edge maps to ANY bar inside the window
        # (right_edge_to_bar), so recent-but-not-tip bars are already
        # discoverable from the tip window itself (EDEN 2026-07-19: the tip
        # window's mid-window box mapped 35 bars back). Its real effect was
        # 14/6 x predict cost: pulses went 6->25 min wall, the 15-min cadence
        # degraded to 25 min, and rows landed older than the 30-min freshness
        # gate — the dense schedule destroyed the very tip-latency it chased.
        # Owner doctrine 2026-07-23: LIVE detection means the newest bars ONLY.
        # The old stride walk-back windows (tip-50/-100/-150) could discover
        # nothing but 12h+ old bars — hindsight rows by construction, every one
        # of them freshness-rejected downstream. A path that can only produce
        # after-the-fact signals must not exist. tip / tip-1 / tip-2 cover
        # pulse-miss overlap; everything older is not a signal, it's history.
        starts_set: set[int] = set()
        for back in (0, 1, 2):
            s = tip_last_start - back
            if s >= first_start:
                starts_set.add(s)
        starts = sorted(starts_set, reverse=True)
    else:
        starts = list(range(first_start, last_start + 1, stride))

    chosen: list[int] = []
    device = _resolve_predict_device()
    n_fail = 0
    last_err: str | None = None
    # Chunked render→predict→unlink: live/tip stay ≤6 windows (one chunk);
    # full offline can be hundreds of windows — holding all PNGs + one giant
    # batch predict OOM-killed Mac scans (2026-07-24 short tip_v1b pool build).
    # Chunk size keeps CPU batching benefit without multi-GB RSS per series.
    predict_conf = conf
    if tip_conf is not None and tip_conf < conf:
        predict_conf = tip_conf
    tip_edge_rejected = 0
    apply_tip_edge = mode in ("live", "tip") and tip_edge_bars > 0
    allow_pending_entry = mode in ("live", "tip")
    # Live/tip: one chunk. Full offline: small chunks — 16 still Jetsam'd
    # 16GB Macs mid-series (2026-07-24 short tip_v1b pool; residual 16 PNGs/pid).
    # Override with FABLE_YOLO_FULL_CHUNK (int >=1) if needed.
    if mode in ("live", "tip"):
        chunk_size = len(starts)
    else:
        raw_chunk = os.environ.get("FABLE_YOLO_FULL_CHUNK", "4").strip()
        try:
            chunk_size = int(raw_chunk)
        except ValueError:
            chunk_size = 4
    chunk_size = max(1, int(chunk_size))
    for chunk_i in range(0, len(starts), chunk_size):
        chunk_starts = starts[chunk_i : chunk_i + chunk_size]
        rendered: list[tuple[int, object, Path]] = []
        for k, start in enumerate(chunk_starts):
            sub = enriched_ma.iloc[start : start + window]
            win_png = tmp_png.with_name(f"{tmp_png.stem}_{chunk_i + k}.png")
            try:
                _, tf = render_chart(sub, out_path=win_png)
            except Exception as exc:  # noqa: BLE001 — keep series alive; count failures
                n_fail += 1
                last_err = f"{type(exc).__name__}: {exc}"
                continue
            rendered.append((start, tf, win_png))
        results = []
        if rendered:
            try:
                # Serialize predict: ultralytics is not reliably thread-safe.
                with _predict_lock:
                    results = model.predict(
                        [str(p) for _, _, p in rendered],
                        conf=predict_conf,
                        verbose=False,
                        device=device,
                    )
            except Exception as exc:  # noqa: BLE001
                n_fail += len(rendered)
                last_err = f"{type(exc).__name__}: {exc}"
                results = []
        for (start, tf, win_png), res in zip(rendered, results):
            try:
                boxes = res.boxes
                if boxes is None:
                    continue
                is_tip_window = start == tip_last_start
                floor = tip_conf if (is_tip_window and tip_conf is not None) else conf
                xywhn = boxes.xywhn.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                for bi, b in enumerate(xywhn):
                    if confs is not None and float(confs[bi]) < floor:
                        continue
                    cx, _, w, _ = map(float, b[:4])
                    mapping = map_box_to_signal(
                        cx=cx,
                        w=w,
                        tf=tf,
                        window_start_i=start,
                        n_bars=window,
                        frame_length=len(frame),
                        latest_closed_i=len(frame) - 1,
                        tip_edge_bars=tip_edge_bars,
                        apply_tip_edge=apply_tip_edge,
                        max_global_tip_age_bars=max_global_tip_age_bars,
                        allow_pending_entry=allow_pending_entry,
                        start_from_i=start_from_i,
                        signal_i_lo=i_lo,
                        signal_i_hi=i_hi,
                    )
                    if mapping.rejection_reason == "local_tip_edge":
                        tip_edge_rejected += 1
                    if not mapping.accepted:
                        continue
                    chosen.append(mapping.mapped_signal_i)
            finally:
                try:
                    win_png.unlink(missing_ok=True)
                except OSError:
                    pass
    if tip_edge_rejected:
        _bump_tip_edge_rejected(tip_edge_rejected)
    if n_fail and n_fail >= len(starts):
        # Only noisy when the whole series failed (data/render/device issue).
        print(f"yolo_live: all {n_fail} windows failed last={last_err}", flush=True)
    if not chosen:
        return []
    if right_bias:
        # Prefer rightmost within min_gap (position bias for multi-window live).
        chosen_desc = sorted(set(chosen), reverse=True)
        deduped: list[int] = []
        for si in chosen_desc:
            if not deduped or deduped[-1] - si >= min_gap:
                deduped.append(si)
        return sorted(deduped)
    chosen = sorted(set(chosen))
    deduped = []
    for si in chosen:
        if not deduped or si - deduped[-1] >= min_gap:
            deduped.append(si)
    return deduped


def dedupe_indices(indices: list[int], min_gap: int = MIN_GAP_BARS) -> list[int]:
    out: list[int] = []
    for si in sorted(indices):
        if not out or si - out[-1] >= min_gap:
            out.append(int(si))
    return out
