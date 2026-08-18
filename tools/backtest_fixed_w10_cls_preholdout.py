#!/usr/bin/env python3
"""Run the frozen W10 SHORT classifier only on materialized pre-holdout data.

This tool reuses the pure classification, gap-deduplication, and barrier-outcome
semantics of ``fable-trading/scripts/backtest_fixed_w10_cls_holdout3d.py``.  It
does not import that script, know its snapshot path, or permit its holdout date
range.  The separation is deliberate: this repository must not consume the
holdout while stop-loss examples are being investigated.

Every CSV must be named in a sidecar JSON manifest.  Before pandas is allowed to
materialize OHLC data, the file is streamed once and its SHA256, data-row count,
and maximum ``open_time`` are checked against the sidecar.  Any row at or after
the 2026-05-04 UTC cutoff fails closed.  The causal model image uses only the ten
bars ending at the decision bar; future bars are passed only to the canonical
outcome resolver.  A stop loss is an L2 outcome negative, never an automatic L1
``NO_SIGNAL`` label.  L1 review exports are explicitly ineligible for training.
Optimizer exposure rows are written to physically separate inner and sealed
files; their future paths are label-only data and never enter the causal image.

Manifest schema::

    {"schema_version": 1, "files": [
      {"path": "BTC_USDT_SWAP.csv", "sha256": "...", "rows": 1234,
       "max_materialized_time": "2026-05-03T23:45:00Z"}
    ]}
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yoyo.contracts.costs import SWAP_MAKER, SWAP_TAKER  # noqa: E402
from yoyo.contracts.outcomes import (  # noqa: E402
    ATR_ELIGIBILITY_PROTOCOL,
    ATR_PCT_MIN,
    HORIZON_BARS,
    OutcomeContractError,
    resolve_barrier_outcome,
)
from yoyo.data.indicators import MIN_GAP_BARS, add_indicators  # noqa: E402
from yoyo.datasets.legacy_gold_migration.renderer import SpanError, render_w10  # noqa: E402
from yoyo.layers.l1_detection.data import ALL_MA_COLS, add_mas  # noqa: E402

PROTOCOL = "fixed_w10_core4_confirm1_v1_cls_preholdout"
HOLDOUT_START = pd.Timestamp("2026-05-04T00:00:00Z")
BAR_DURATION = pd.Timedelta(minutes=15)
WINDOW_BARS = 10
WARMUP_BARS = 240
Y_PAD_FRAC = 0.05
IMGSZ = 960
DEFAULT_THRESHOLD = 0.50
TP_ATR = 5.0
SL_ATR = 2.0
SAME_BAR_POLICY = "conservative_sl"
GAP_POLICY = "barrier_price"
RETURN_CONVENTION = "linear_short"
DEFAULT_EXPORT_MAX_HORIZON = 144
DEFAULT_INNER_END = pd.Timestamp("2026-03-31T07:30:00Z")
SEALED_START = pd.Timestamp("2026-04-02T00:00:00Z")
OUTPUT_NAMES = (
    "predictions.jsonl",
    "signals_raw.jsonl",
    "signals_dedup.jsonl",
    "stop_losses.jsonl",
    "l1_review_queue.jsonl",
    "l2_outcome_negatives.jsonl",
    "optimization_inner.jsonl",
    "optimization_sealed.jsonl",
    "summary.json",
)
MAX_RENDER_WORKERS = 8
MAX_RENDER_PREFETCH_BATCHES = 4


@dataclass(frozen=True)
class OutcomeProtocol:
    """One explicit candidate for reusable pre-holdout outcome evaluation."""

    threshold: float = DEFAULT_THRESHOLD
    tp_atr: float = TP_ATR
    sl_atr: float = SL_ATR
    horizon_bars: int = HORIZON_BARS

    def validated(self) -> "OutcomeProtocol":
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if not np.isfinite(self.tp_atr) or float(self.tp_atr) <= 0:
            raise ValueError("tp_atr must be finite and positive")
        if not np.isfinite(self.sl_atr) or float(self.sl_atr) <= 0:
            raise ValueError("sl_atr must be finite and positive")
        if int(self.horizon_bars) <= 0:
            raise ValueError("horizon_bars must be positive")
        return self


BASELINE_OUTCOME_PROTOCOL = OutcomeProtocol()


@dataclass(frozen=True)
class OptimizationInterval:
    """Half-open decision interval whose labels must also end inside it."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_values(cls, name: str, start: Any, end: Any) -> "OptimizationInterval":
        interval = cls(
            name=str(name),
            start=_utc_timestamp(start, field=f"{name}.start"),
            end=_utc_timestamp(end, field=f"{name}.end"),
        )
        if not interval.start < interval.end:
            raise ValueError(f"{name}: optimization interval is empty")
        if interval.start >= HOLDOUT_START or interval.end > HOLDOUT_START:
            raise ValueError(f"{name}: optimization interval reaches holdout")
        return interval


class WhiteLetterbox:
    """Resize the full frame into a 960-square white canvas, without cropping."""

    def __init__(self, size: int = IMGSZ, fill: int = 255):
        self.size = int(size)
        self.fill = int(fill)

    def __call__(self, image):
        from PIL import Image

        width, height = image.size
        scale = self.size / max(width, height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.BILINEAR,
        )
        if resized.mode != "RGB":
            resized = resized.convert("RGB")
        canvas = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
        canvas.paste(
            resized,
            ((self.size - resized.width) // 2, (self.size - resized.height) // 2),
        )
        return canvas


@dataclass(frozen=True)
class RenderedDecision:
    """Prepared causal tensor for one decision, or a safe renderer skip."""

    decision_i: int
    tensor: Any | None
    skipped_span: bool = False


@dataclass(frozen=True)
class RenderedBatch:
    """One original inference batch after CPU render/tensor preparation."""

    decisions: tuple[RenderedDecision, ...]

    @property
    def skipped_span(self) -> int:
        return sum(1 for decision in self.decisions if decision.skipped_span)


def _utc_timestamp(value: Any, *, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if pd.isna(stamp):
        raise ValueError(f"invalid {field}: {value!r}")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp


def safe_last_decision_time(
    cutoff: Any = HOLDOUT_START,
    *,
    horizon_bars: int = HORIZON_BARS,
    bar_duration: pd.Timedelta = BAR_DURATION,
) -> pd.Timestamp:
    """Return the last aligned decision whose full next-open path is pre-cutoff.

    Entry is one bar after the decision and a 72-bar path ends at decision+72.
    Since materialized bar opens must be strictly before ``cutoff``, one further
    bar is subtracted to obtain the last permissible aligned decision.
    """
    if int(horizon_bars) <= 0 or bar_duration <= pd.Timedelta(0):
        raise ValueError("horizon_bars and bar_duration must be positive")
    return _utc_timestamp(cutoff, field="cutoff") - (int(horizon_bars) + 1) * bar_duration


def validate_date_range(start: Any, end: Any, *, complete_outcomes: bool = True) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate an inclusive pre-holdout decision range."""
    start_ts = _utc_timestamp(start, field="start")
    end_ts = _utc_timestamp(end, field="end")
    if start_ts > end_ts:
        raise ValueError("start must be <= end")
    if start_ts >= HOLDOUT_START or end_ts >= HOLDOUT_START:
        raise ValueError("start/end must be strictly before the holdout cutoff")
    if complete_outcomes and end_ts > safe_last_decision_time():
        raise ValueError(
            "end is too late for a complete next-open + 72-bar outcome without holdout data; "
            f"latest safe decision is {safe_last_decision_time().isoformat()}"
        )
    return start_ts, end_ts


def validate_candidate_date_range(
    start: Any,
    end: Any,
    candidate: OutcomeProtocol,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate dates using the candidate horizon rather than the baseline."""
    candidate.validated()
    start_ts, end_ts = validate_date_range(start, end, complete_outcomes=False)
    safe_last = safe_last_decision_time(horizon_bars=candidate.horizon_bars)
    if end_ts > safe_last:
        raise ValueError(
            "end is too late for the candidate's complete next-open outcome without holdout data; "
            f"latest safe decision is {safe_last.isoformat()}"
        )
    return start_ts, end_ts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_materialization_manifest(snapshot_dir: Path, manifest_path: Path) -> list[dict[str, Any]]:
    """Validate the sidecar itself and return entries in deterministic order."""
    snapshot_dir = snapshot_dir.resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("files"), list):
        raise ValueError("manifest requires schema_version=1 and a files list")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw["files"]:
        if not isinstance(item, dict):
            raise ValueError("manifest file entries must be objects")
        relative = Path(str(item.get("path", "")))
        if not relative.name or relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError(f"manifest path must be a direct child CSV: {relative}")
        if relative.suffix.lower() != ".csv" or relative.name in seen:
            raise ValueError(f"invalid or duplicate manifest CSV: {relative.name}")
        seen.add(relative.name)
        path = (snapshot_dir / relative).resolve()
        if path.parent != snapshot_dir or not path.is_file():
            raise ValueError(f"manifest CSV is missing or escapes snapshot-dir: {relative}")
        digest = str(item.get("sha256", "")).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid sha256 for {relative.name}")
        try:
            rows = int(item["rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid rows for {relative.name}") from exc
        if rows <= 0:
            raise ValueError(f"rows must be positive for {relative.name}")
        maximum = _utc_timestamp(item.get("max_materialized_time"), field="max_materialized_time")
        if maximum >= HOLDOUT_START:
            raise ValueError(f"manifest declares holdout materialization for {relative.name}")
        entries.append({**item, "path": relative.name, "sha256": digest, "rows": rows,
                        "max_materialized_time": maximum.isoformat()})
    if not entries:
        raise ValueError("manifest contains no CSV files")
    actual = {path.name for path in snapshot_dir.glob("*.csv")}
    if actual != seen:
        raise ValueError(f"manifest/CSV set mismatch: undeclared={sorted(actual-seen)}, missing={sorted(seen-actual)}")
    return sorted(entries, key=lambda item: str(item["path"]))


def load_preholdout_csv(path: Path, declaration: dict[str, Any]) -> pd.DataFrame:
    """Stream-gate one CSV before materializing its OHLC values with pandas."""
    expected_sha = str(declaration["sha256"]).lower()
    expected_rows = int(declaration["rows"])
    expected_max = _utc_timestamp(declaration["max_materialized_time"], field="max_materialized_time")
    digest = hashlib.sha256()
    with path.open("rb") as binary:
        for chunk in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(chunk)
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"SHA256 mismatch for {path.name}")
        binary.seek(0)
        text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text)
        if reader.fieldnames is None or "open_time" not in reader.fieldnames:
            raise ValueError(f"{path.name} has no open_time column")
        row_count = 0
        actual_max: pd.Timestamp | None = None
        for row_count, row in enumerate(reader, start=1):
            stamp = _utc_timestamp(row.get("open_time"), field=f"{path.name}:open_time")
            if stamp >= HOLDOUT_START:
                raise ValueError(f"{path.name} contains a row at/after holdout cutoff")
            actual_max = stamp if actual_max is None else max(actual_max, stamp)
        if row_count != expected_rows:
            raise ValueError(f"row-count mismatch for {path.name}: manifest={expected_rows}, actual={row_count}")
        if actual_max is None or actual_max != expected_max:
            raise ValueError(
                f"max_materialized_time mismatch for {path.name}: "
                f"manifest={expected_max.isoformat()}, actual={actual_max}"
            )
        text.seek(0)
        raw = pd.read_csv(text)

    required = {"open_time", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    frame = raw.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{path.name} contains invalid required values")
    if frame["open_time"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate open_time rows")
    frame = frame.sort_values("open_time").reset_index(drop=True)
    return add_indicators(add_mas(frame))


def validate_sources(snapshot_dir: Path, manifest_path: Path) -> list[dict[str, Any]]:
    """Run all source gates without retaining any OHLC frames."""
    declarations = load_materialization_manifest(snapshot_dir, manifest_path)
    checked: list[dict[str, Any]] = []
    for item in declarations:
        frame = load_preholdout_csv(snapshot_dir / str(item["path"]), item)
        checked.append({"path": item["path"], "rows": len(frame),
                        "max_materialized_time": pd.Timestamp(frame["open_time"].max()).isoformat()})
    return checked


def sample_declarations(
    declarations: Sequence[dict[str, Any]],
    sample_symbols: int,
    sample_seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select symbols by a seed hash, without inspecting OHLC or outcomes."""
    count = int(sample_symbols)
    if count < 0:
        raise ValueError("sample_symbols must be non-negative")
    enriched: list[tuple[str, str, dict[str, Any]]] = []
    symbols: set[str] = set()
    for declaration in declarations:
        symbol = str(declaration.get("symbol") or Path(str(declaration["path"])).stem)
        if symbol in symbols:
            raise ValueError(f"duplicate symbol in materialization manifest: {symbol}")
        symbols.add(symbol)
        rank = hashlib.sha256(f"{sample_seed}\0{symbol}".encode("utf-8")).hexdigest()
        enriched.append((rank, symbol, dict(declaration)))
    ordered = sorted(enriched, key=lambda item: (item[0], item[1]))
    selected = ordered if count == 0 else ordered[:count]
    if count > len(ordered):
        raise ValueError(f"sample_symbols={count} exceeds available symbols={len(ordered)}")
    metadata = {
        "population_symbols": len(ordered),
        "sampled_symbols": len(selected),
        "requested_n": count,
        "seed": str(sample_seed),
        "algorithm": "sort_sha256(seed+'\\0'+symbol)_ascending",
        "selected_symbols": [symbol for _, symbol, _ in selected],
    }
    return [declaration for _, _, declaration in selected], metadata


def load_gold_identities(path: Path | None) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    """Load only safe SIGNAL identities from a classification JSONL manifest."""
    if path is None:
        return set(), {"enabled": False, "source": None, "sha256": None, "rows": 0}
    identities: set[tuple[str, str]] = set()
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid gold JSONL at line {line_number}") from exc
            if row.get("label") != "SIGNAL":
                continue
            if row.get("holdout_read") is not False:
                raise ValueError(f"gold SIGNAL line {line_number} must declare holdout_read=false")
            if row.get("future_used_in_model_input") is not False:
                raise ValueError(
                    f"gold SIGNAL line {line_number} must declare future_used_in_model_input=false"
                )
            symbol = str(row.get("symbol", ""))
            decision = _utc_timestamp(row.get("decision_time"), field="gold decision_time")
            if not symbol or decision >= HOLDOUT_START:
                raise ValueError(f"gold SIGNAL line {line_number} is missing identity or reaches holdout")
            identity = (symbol, decision.isoformat())
            if identity in identities:
                raise ValueError(f"duplicate gold identity: {identity}")
            identities.add(identity)
    return identities, {
        "enabled": True, "source": str(path.resolve()), "sha256": sha256_file(path),
        "rows": rows, "signal_identities": len(identities),
    }


def protocol_record(*, start: pd.Timestamp, end: pd.Timestamp, candidate: OutcomeProtocol) -> dict[str, Any]:
    candidate.validated()
    return {
        "protocol": PROTOCOL,
        "evaluation_scope": "preholdout",
        "holdout_cutoff": HOLDOUT_START.isoformat(),
        "holdout_read": False,
        "future_data_in_causal_input": False,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "safe_last_decision_time": safe_last_decision_time(horizon_bars=candidate.horizon_bars).isoformat(),
        "window_bars": WINDOW_BARS,
        "renderer": "yoyo.datasets.legacy_gold_migration.renderer.render_w10",
        "renderer_overlay": False,
        "transform": "white_letterbox_full_frame",
        "imgsz": IMGSZ,
        "parameter_status": "preholdout_candidate_not_promoted",
        "baseline_parameters": asdict(BASELINE_OUTCOME_PROTOCOL),
        "candidate_parameters": asdict(candidate),
        "threshold": float(candidate.threshold),
        "side": "short",
        "entry": "next_open",
        "tp_atr": float(candidate.tp_atr),
        "sl_atr": float(candidate.sl_atr),
        "horizon_bars": int(candidate.horizon_bars),
        "same_bar_policy": SAME_BAR_POLICY,
        "gap_policy": GAP_POLICY,
        "return_convention": RETURN_CONVENTION,
        "atr_eligibility_protocol": ATR_ELIGIBILITY_PROTOCOL,
        "atr_pct_min": ATR_PCT_MIN,
        "min_gap_bars": MIN_GAP_BARS,
        "costs": {"net_maker": SWAP_MAKER, "net_taker": SWAP_TAKER},
    }


def _event_id(symbol: str, decision_time: Any) -> str:
    """Return an identity derived only from protocol, symbol, and decision time."""
    stamp = _utc_timestamp(decision_time, field="decision_time").isoformat()
    material = f"{PROTOCOL}\0{symbol}\0{stamp}".encode("utf-8")
    return "w10pre_" + hashlib.sha256(material).hexdigest()[:24]


def add_safety_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "holdout_read": False, "future_data_in_causal_input": False}


def label_end_time(decision_time: Any, horizon_bars: int) -> pd.Timestamp:
    """Return the exclusive label end for a next-open short outcome."""
    return _utc_timestamp(decision_time, field="decision_time") + (int(horizon_bars) + 1) * BAR_DURATION


def prediction_has_inner_outcome(row: dict[str, Any], inner_end: pd.Timestamp, horizon_bars: int) -> bool:
    """True only when decision and its full label path are inside inner."""
    decision = _utc_timestamp(row["decision_time"], field="decision_time")
    return decision < inner_end and label_end_time(decision, horizon_bars) <= inner_end


def filter_inner_outcome_predictions(
    predictions: Sequence[dict[str, Any]],
    inner_end: pd.Timestamp,
    horizon_bars: int,
) -> list[dict[str, Any]]:
    """Keep only rows eligible for baseline/candidate outcome reporting."""
    inner = [
        dict(row) for row in predictions
        if prediction_has_inner_outcome(row, inner_end, horizon_bars)
    ]
    validate_unique_records(inner)
    return inner


def validate_unique_records(rows: Sequence[dict[str, Any]]) -> None:
    """Reject duplicate event IDs or duplicate symbol/decision identities."""
    identities: set[tuple[str, str]] = set()
    event_ids: set[str] = set()
    for row in rows:
        identity = (str(row["symbol"]), _utc_timestamp(row["decision_time"], field="decision_time").isoformat())
        event_id = str(row["event_id"])
        if identity in identities:
            raise ValueError(f"duplicate symbol+decision_time: {identity}")
        if event_id in event_ids:
            raise ValueError(f"duplicate event_id: {event_id}")
        identities.add(identity)
        event_ids.add(event_id)


def parse_optimization_thresholds(value: str) -> tuple[float, ...]:
    """Parse one explicit, unique optimization threshold grid."""
    parts = [part.strip() for part in str(value).split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("optimization-thresholds must be a non-empty comma-separated list")
    try:
        thresholds = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("optimization-thresholds contains a non-number") from exc
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("optimization-thresholds must be finite and in [0, 1]")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("optimization-thresholds must be unique")
    return tuple(sorted(thresholds))


def build_optimization_rows(
    predictions: Sequence[dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    intervals: Sequence[OptimizationInterval],
    max_horizon: int,
    gold_identities: set[tuple[str, str]] | None = None,
    min_score: float = 0.0,
    thresholds: Sequence[float] = (0.0,),
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Build a lossless union for explicit threshold/gap optimizer arms.

    Each emitted path begins at the next bar open and contains exactly
    ``max_horizon`` bars.  Interval label-end cropping happens first.  For each
    explicit threshold, decision-bar ATR eligibility is applied before events
    are thresholded, ordered by
    symbol/decision/event ID, and de-duplicated at ``MIN_GAP_BARS`` exactly as
    the optimizer does.  Only the union of those arms plus Gold recall rows gets
    a future path, preserving every preregistered trading result at lower cost.
    """
    horizon = int(max_horizon)
    if horizon <= 0:
        raise ValueError("max_horizon must be positive")
    minimum = float(min_score)
    if not np.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise ValueError("optimization min_score must be finite and in [0, 1]")
    grid = tuple(sorted(float(value) for value in thresholds))
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("optimization thresholds must be non-empty and unique")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in grid):
        raise ValueError("optimization thresholds must be finite and in [0, 1]")
    if grid[0] != minimum:
        raise ValueError("optimization min_score must equal the minimum optimization threshold")
    normalized = sorted(intervals, key=lambda interval: interval.start)
    if not normalized:
        raise ValueError("at least one optimization interval is required")
    for interval in normalized:
        if interval.start >= HOLDOUT_START or interval.end > HOLDOUT_START:
            raise ValueError(f"{interval.name}: optimization interval reaches holdout")
    for previous, current in zip(normalized, normalized[1:]):
        if current.start < previous.end:
            raise ValueError("optimization intervals must not overlap")

    output = {interval.name: [] for interval in normalized}
    stats = {"outside_intervals": 0, "interval_end_cropped": 0,
             "atr_floor_excluded": 0,
             "invalid_atr_or_entry": 0, "incomplete_or_gapped_path": 0,
             "below_min_score": 0, "not_in_threshold_union": 0,
             "gold_force_included": 0, "population_predictions": len(predictions),
             "threshold_union_selected": 0,
             "selected_by_threshold": {str(value): 0 for value in grid}}
    validate_unique_records(predictions)
    gold_set = gold_identities or set()
    gold_matches = 0
    candidates: dict[str, list[dict[str, Any]]] = {interval.name: [] for interval in normalized}
    for prediction in predictions:
        decision = _utc_timestamp(prediction["decision_time"], field="decision_time")
        interval = next(
            (candidate for candidate in normalized if candidate.start <= decision < candidate.end),
            None,
        )
        if interval is None:
            stats["outside_intervals"] += 1
            continue
        # Path opens are decision+1 through decision+horizon; its label end is
        # one bar after the final open and must remain inside the interval.
        if decision + (horizon + 1) * BAR_DURATION > interval.end:
            stats["interval_end_cropped"] += 1
            continue
        symbol = str(prediction["symbol"])
        frame = frames.get(symbol)
        if frame is None:
            raise ValueError(f"prediction references unknown symbol: {symbol}")
        decision_i = int(prediction["decision_i"])
        try:
            atr_pct = float(frame["atr_pct"].iloc[decision_i])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"{prediction['event_id']}: decision atr_pct is unavailable") from exc
        if not np.isfinite(atr_pct) or atr_pct < ATR_PCT_MIN:
            stats["atr_floor_excluded"] += 1
            continue
        is_gold = (symbol, decision.isoformat()) in gold_set
        score = float(prediction["p_signal"])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{prediction['event_id']}: p_signal must be finite and in [0, 1]")
        if score < minimum and not is_gold:
            stats["below_min_score"] += 1
            continue
        entry_i = decision_i + 1
        path = frame.iloc[entry_i:entry_i + horizon]
        if len(path) != horizon:
            stats["incomplete_or_gapped_path"] += 1
            continue
        times = pd.to_datetime(path["open_time"], utc=True)
        expected = pd.date_range(decision + BAR_DURATION, periods=horizon, freq=BAR_DURATION)
        if not times.reset_index(drop=True).equals(pd.Series(expected)) or (times >= HOLDOUT_START).any():
            stats["incomplete_or_gapped_path"] += 1
            continue
        atr = float(frame["atr14"].iloc[decision_i])
        entry_price = float(path["open"].iloc[0])
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(entry_price) or entry_price <= 0:
            stats["invalid_atr_or_entry"] += 1
            continue
        candidates[interval.name].append({
            "prediction": prediction, "decision": decision, "symbol": symbol,
            "score": score, "is_gold": is_gold, "atr": atr,
            "atr_pct": atr_pct, "entry_price": entry_price, "path": path,
            "event_id": _event_id(symbol, decision),
        })

    selected_keys: set[str] = set()
    selected_at: dict[str, list[float]] = {}
    for interval in normalized:
        interval_candidates = candidates[interval.name]
        for threshold in grid:
            eligible = [item for item in interval_candidates if item["score"] >= threshold]
            eligible.sort(key=lambda item: (item["symbol"], item["decision"], item["event_id"]))
            last: dict[str, pd.Timestamp] = {}
            selected: list[dict[str, Any]] = []
            for item in eligible:
                symbol = item["symbol"]
                if symbol not in last or item["decision"] - last[symbol] >= MIN_GAP_BARS * BAR_DURATION:
                    selected.append(item)
                    last[symbol] = item["decision"]
            stats["selected_by_threshold"][str(threshold)] = (
                stats["selected_by_threshold"].get(str(threshold), 0) + len(selected)
            )
            for item in selected:
                selected_keys.add(item["event_id"])
                selected_at.setdefault(item["event_id"], []).append(threshold)

    stats["threshold_union_selected"] = len(selected_keys)
    for interval in normalized:
        for item in candidates[interval.name]:
            force_gold = item["is_gold"] and item["event_id"] not in selected_keys
            if item["event_id"] not in selected_keys and not force_gold:
                stats["not_in_threshold_union"] += 1
                continue
            if force_gold:
                stats["gold_force_included"] += 1
            prediction = item["prediction"]
            decision = item["decision"]
            symbol = item["symbol"]
            score = item["score"]
            atr = item["atr"]
            atr_pct = item["atr_pct"]
            entry_price = item["entry_price"]
            path = item["path"]
            future_path = [
                {
                    "time": pd.Timestamp(row.open_time).isoformat(),
                    "open": float(row.open), "high": float(row.high),
                    "low": float(row.low), "close": float(row.close),
                }
                for row in path[["open_time", "open", "high", "low", "close"]].itertuples(index=False)
            ]
            gold_matches += int(item["is_gold"])
            output[interval.name].append(add_safety_fields({
                "protocol": PROTOCOL,
                "event_id": _event_id(symbol, decision),
                "symbol": symbol,
                "decision_time": decision.isoformat(),
                "p_signal": score,
                "gold": item["is_gold"],
                "selected_at_thresholds": selected_at.get(item["event_id"], []),
                "entry_price": entry_price,
                "atr14": atr,
                "atr_pct": atr_pct,
                "atr_pct_min": ATR_PCT_MIN,
                "atr_eligibility_protocol": ATR_ELIGIBILITY_PROTOCOL,
                "future_review_only": True,
                "future_used_in_training_image": False,
                "causal_metadata": {
                    "visible_end_time": decision.isoformat(),
                    "window_start_time": prediction.get("causal_window_start_time"),
                    "window_bars": WINDOW_BARS,
                    "renderer_overlay": False,
                },
                "future_path": future_path,
            }))
    for rows in output.values():
        validate_unique_records(rows)
    combined = [row for rows in output.values() for row in rows]
    validate_unique_records(combined)
    stats["exported"] = len(combined)
    stats["gold_matches"] = gold_matches
    return output, stats


def decision_indices(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    horizon_bars: int,
) -> list[int]:
    """Select decisions with causal W10/MA history and a complete future path."""
    times = pd.to_datetime(frame["open_time"], utc=True)
    selected: list[int] = []
    for index in range(WARMUP_BARS, len(frame)):
        stamp = times.iloc[index]
        if stamp < start or stamp > end:
            continue
        if index + int(horizon_bars) >= len(frame):
            continue
        window = frame.iloc[index - WINDOW_BARS + 1:index + 1]
        if len(window) == WINDOW_BARS and not window[list(ALL_MA_COLS)].isna().any().any():
            selected.append(index)
    return selected


def gap_dedup(rows: Sequence[dict[str, Any]], gap_bars: int = MIN_GAP_BARS) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (str(row["symbol"]), int(row["decision_i"])))
    kept: list[dict[str, Any]] = []
    last: dict[str, int] = {}
    for row in ordered:
        symbol, index = str(row["symbol"]), int(row["decision_i"])
        if symbol not in last or index - last[symbol] >= int(gap_bars):
            kept.append(dict(row))
            last[symbol] = index
    return kept


def _full_frame_transform():
    import torchvision.transforms as transforms

    return transforms.Compose([WhiteLetterbox(IMGSZ), transforms.ToTensor()])


def _load_classifier(weights: Path, device: str):
    import __main__ as main_module
    from ultralytics import YOLO

    main_module.WhiteLetterbox = WhiteLetterbox
    model = YOLO(str(weights), task="classify")
    names = {int(key): str(value) for key, value in dict(model.names).items()}
    signal_ids = [key for key, name in names.items() if name == "SIGNAL"]
    if len(signal_ids) != 1:
        raise ValueError(f"expected exactly one SIGNAL class, got {names}")
    model.model.to(device).eval()
    return model, names, signal_ids[0]


def _pick_device(explicit: str | None) -> str:
    import torch

    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    raise ValueError("no CUDA/MPS available; refusing an accidental CPU backtest")


def _render_decision_tensor(transform, frame: pd.DataFrame, decision_i: int) -> RenderedDecision:
    import cv2
    from PIL import Image

    window = frame.iloc[decision_i - WINDOW_BARS + 1:decision_i + 1]
    try:
        image, _ = render_w10(window, y_pad_frac=Y_PAD_FRAC, overlay=False)
    except (SpanError, ValueError):
        return RenderedDecision(decision_i=int(decision_i), tensor=None, skipped_span=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return RenderedDecision(decision_i=int(decision_i), tensor=transform(Image.fromarray(rgb)))


def _render_batch(transform, frame: pd.DataFrame, indices: Sequence[int]) -> RenderedBatch:
    return RenderedBatch(tuple(_render_decision_tensor(transform, frame, int(decision_i)) for decision_i in indices))


def _batched_indices(indices: Sequence[int], batch_size: int) -> list[tuple[int, ...]]:
    return [
        tuple(int(decision_i) for decision_i in indices[offset:offset + batch_size])
        for offset in range(0, len(indices), batch_size)
    ]


def _iter_rendered_batches(
    transform,
    frame: pd.DataFrame,
    indices: Sequence[int],
    *,
    batch_size: int,
    render_workers: int,
) -> Iterable[RenderedBatch]:
    batches = _batched_indices(indices, batch_size)
    workers = int(render_workers)
    if workers <= 0 or len(batches) <= 1:
        for batch in batches:
            yield _render_batch(transform, frame, batch)
        return

    max_pending = min(len(batches), max(1, min(MAX_RENDER_PREFETCH_BATCHES, workers * 2)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="w10-render") as executor:
        pending: deque[Future[RenderedBatch]] = deque()
        iterator = iter(batches)
        for _ in range(max_pending):
            try:
                pending.append(executor.submit(_render_batch, transform, frame, next(iterator)))
            except StopIteration:
                break
        while pending:
            future = pending.popleft()
            try:
                next_batch = next(iterator)
            except StopIteration:
                next_batch = None
            if next_batch is not None:
                pending.append(executor.submit(_render_batch, transform, frame, next_batch))
            yield future.result()


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    return f"{int(minutes)}m{rem:04.1f}s"


def classify_frame(model, transform, frame: pd.DataFrame, symbol: str, indices: Sequence[int], *,
                   device: str, signal_idx: int, threshold: float, batch_size: int,
                   render_workers: int = 0) -> tuple[list[dict[str, Any]], int]:
    import torch

    rows: list[dict[str, Any]] = []
    skipped_span = 0
    completed = 0
    total = len(indices)
    progress_started = time.monotonic()
    last_progress = progress_started
    if total == 0:
        _maybe_print_symbol_progress(symbol, 0, 0, progress_started, last_progress)
    for rendered in _iter_rendered_batches(
        transform, frame, indices, batch_size=batch_size, render_workers=render_workers,
    ):
        skipped_span += rendered.skipped_span
        kept_decisions = [decision for decision in rendered.decisions if decision.tensor is not None]
        tensors = [decision.tensor for decision in kept_decisions]
        if not tensors:
            completed += len(rendered.decisions)
            last_progress = _maybe_print_symbol_progress(
                symbol, completed, total, progress_started, last_progress,
            )
            continue
        with torch.no_grad():
            logits = model.model(torch.stack(tensors).to(device))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probabilities = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()
        for decision, probability in zip(kept_decisions, probabilities):
            decision_i = decision.decision_i
            decision_time = pd.Timestamp(frame["open_time"].iloc[decision_i]).isoformat()
            p_signal = float(probability[signal_idx])
            rows.append(add_safety_fields({
                "protocol": PROTOCOL,
                "event_id": _event_id(symbol, decision_time),
                "symbol": symbol,
                "decision_i": int(decision_i),
                "decision_time": decision_time,
                "window_start_i": int(decision_i - WINDOW_BARS + 1),
                "window_end_i": int(decision_i),
                "causal_window_start_time": pd.Timestamp(frame["open_time"].iloc[decision_i-WINDOW_BARS+1]).isoformat(),
                "causal_window_end_time": decision_time,
                "p_signal": p_signal,
                "pred": "SIGNAL" if p_signal >= threshold else "NO_SIGNAL",
            }))
        completed += len(rendered.decisions)
        last_progress = _maybe_print_symbol_progress(
            symbol, completed, total, progress_started, last_progress,
        )
    validate_unique_records(rows)
    return rows, skipped_span


def _maybe_print_symbol_progress(
    symbol: str,
    completed: int,
    total: int,
    started: float,
    last_printed: float,
    *,
    min_interval_seconds: float = 10.0,
) -> float:
    now = time.monotonic()
    if total > 0 and completed < total and now - last_printed < min_interval_seconds:
        return last_printed
    elapsed = _format_elapsed(now - started)
    print(
        f"[{symbol}] predictions {completed}/{total} elapsed {elapsed}",
        file=sys.stderr,
        flush=True,
    )
    return now


def outcome_for(frame: pd.DataFrame, decision_i: int, candidate: OutcomeProtocol) -> dict[str, Any]:
    candidate.validated()
    entry_i = int(decision_i) + 1
    atr = float(frame["atr14"].iloc[decision_i])
    atr_pct = float(frame["atr_pct"].iloc[decision_i])
    if not np.isfinite(atr) or atr <= 0:
        return {"status": "skip", "outcome": "bad_atr", "entry_i": entry_i}
    if not np.isfinite(atr_pct) or atr_pct < ATR_PCT_MIN:
        return {"status": "skip", "outcome": "atr_floor", "entry_i": entry_i, "atr_pct": atr_pct}
    entry = float(frame["open"].iloc[entry_i])
    try:
        resolved = resolve_barrier_outcome(
            frame, side="short", entry_i=entry_i, entry_price=entry, atr=atr,
            tp_atr_mult=candidate.tp_atr, sl_atr_mult=candidate.sl_atr,
            horizon_bars=candidate.horizon_bars,
            same_bar_policy=SAME_BAR_POLICY, gap_policy=GAP_POLICY,
            return_convention=RETURN_CONVENTION, allow_partial=False, bar_duration=BAR_DURATION,
        )
    except OutcomeContractError as exc:
        return {"status": "skip", "outcome": f"contract:{exc}", "entry_i": entry_i}
    gross = resolved.gross_ret
    return {
        "status": resolved.status, "outcome": resolved.outcome, "entry_i": entry_i,
        "entry_time": pd.Timestamp(frame["open_time"].iloc[entry_i]).isoformat(),
        "entry_price": entry, "atr14": atr, "atr_pct": atr_pct,
        "exit_offset": resolved.exit_offset, "exit_time": resolved.exit_time,
        "exit_price": resolved.exit_price, "gross_ret": gross,
        "net_taker": None if gross is None else gross - SWAP_TAKER,
        "net_maker": None if gross is None else gross - SWAP_MAKER,
    }


def attach_outcomes(
    rows: Sequence[dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    candidate: OutcomeProtocol,
) -> list[dict[str, Any]]:
    output = [add_safety_fields({
        **row,
        **outcome_for(frames[str(row["symbol"])], int(row["decision_i"]), candidate),
    }) for row in rows]
    validate_unique_records(output)
    return output


def filter_atr_eligible_predictions(
    rows: Sequence[dict[str, Any]], frames: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], int]:
    """Apply the frozen decision-bar ATR floor before threshold and gap gates."""
    eligible: list[dict[str, Any]] = []
    excluded = 0
    for row in rows:
        symbol = str(row["symbol"])
        frame = frames.get(symbol)
        if frame is None:
            raise ValueError(f"prediction references unknown symbol: {symbol}")
        try:
            atr_pct = float(frame["atr_pct"].iloc[int(row["decision_i"])])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"{row['event_id']}: decision atr_pct is unavailable") from exc
        if not np.isfinite(atr_pct) or atr_pct < ATR_PCT_MIN:
            excluded += 1
        else:
            eligible.append(dict(row))
    validate_unique_records(eligible)
    return eligible, excluded


def build_inner_backtest_outputs(
    predictions: Sequence[dict[str, Any]],
    frames: dict[str, pd.DataFrame],
    candidate: OutcomeProtocol,
    inner_end: pd.Timestamp,
) -> dict[str, Any]:
    """Build every outcome-bearing artifact from inner-only predictions."""
    inner_predictions = filter_inner_outcome_predictions(
        predictions, inner_end, candidate.horizon_bars,
    )
    eligible_predictions, atr_floor_excluded = filter_atr_eligible_predictions(
        inner_predictions, frames,
    )
    raw_predictions = [row for row in eligible_predictions if row["pred"] == "SIGNAL"]
    dedup_predictions = gap_dedup(raw_predictions)
    raw_signals = attach_outcomes(raw_predictions, frames, candidate)
    dedup_signals = attach_outcomes(dedup_predictions, frames, candidate)
    stop_losses, l1_review, l2_negative = route_stop_losses(dedup_signals)
    return {
        "predictions": inner_predictions,
        "signals_raw": raw_signals,
        "signals_dedup": dedup_signals,
        "stop_losses": stop_losses,
        "l1_review_queue": l1_review,
        "l2_outcome_negatives": l2_negative,
        "atr_floor_excluded": atr_floor_excluded,
    }


def route_stop_losses(trades: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Route SL outcomes to L2 negatives and manual-only L1 review records."""
    validate_unique_records(trades)
    stops, l1_review, l2_negative = [], [], []
    for trade in trades:
        if trade.get("outcome") not in {"sl", "sl_ambiguous"}:
            continue
        stop = add_safety_fields({**trade, "l2_outcome_negative": True,
                                  "automatic_l1_label": None})
        stops.append(stop)
        l2_negative.append(add_safety_fields({
            "protocol": PROTOCOL, "event_id": trade["event_id"], "symbol": trade["symbol"],
            "decision_time": trade["decision_time"], "outcome": trade["outcome"],
            "l2_outcome_negative": True, "l2_target": 0,
        }))
        l1_review.append(add_safety_fields({
            "protocol": PROTOCOL, "event_id": trade["event_id"], "symbol": trade["symbol"],
            "decision_time": trade["decision_time"],
            "training_eligible": False, "suggested_label": "REVIEW_REQUIRED",
            "automatic_l1_label": None, "future_review_only": True,
            "future_used_in_training_image": False,
            "causal_render": {
                "renderer": "yoyo.datasets.legacy_gold_migration.renderer.render_w10",
                "window_bars": WINDOW_BARS, "overlay": False, "y_pad_frac": Y_PAD_FRAC,
                "letterbox": "white", "imgsz": IMGSZ,
                "window_start_time": trade.get("causal_window_start_time"),
                "window_end_time": trade.get("causal_window_end_time", trade["decision_time"]),
            },
        }))
    return stops, l1_review, l2_negative


def summarize_trades(trades: Sequence[dict[str, Any]]) -> dict[str, Any]:
    closed = [row for row in trades if row.get("status") == "closed" and row.get("net_maker") is not None]
    result: dict[str, Any] = {
        "n_total": len(trades), "n_closed": len(closed),
        "n_open": sum(row.get("status") == "open" for row in trades),
        "n_skip": sum(row.get("status") == "skip" for row in trades),
    }
    if closed:
        gross = np.asarray([row["gross_ret"] for row in closed], dtype=float)
        maker = np.asarray([row["net_maker"] for row in closed], dtype=float)
        taker = np.asarray([row["net_taker"] for row in closed], dtype=float)
        outcomes: dict[str, int] = {}
        for row in closed:
            outcomes[str(row["outcome"])] = outcomes.get(str(row["outcome"]), 0) + 1
        result.update({
            "mean_gross_bp": float(gross.mean() * 1e4),
            "mean_net_maker_bp": float(maker.mean() * 1e4),
            "mean_net_taker_bp": float(taker.mean() * 1e4),
            "total_net_maker": float(maker.sum()), "total_net_taker": float(taker.sum()),
            "win_rate_net_maker": float((maker > 0).mean()), "outcomes": outcomes,
        })
    return result


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def public_optimization_exports_summary(
    optimization_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Expose inner counts while keeping sealed score-derived data opaque."""
    return {
        "inner": {
            "file": "optimization_inner.jsonl",
            "rows": len(optimization_rows.get("inner", [])),
            "atr_eligibility_protocol": ATR_ELIGIBILITY_PROTOCOL,
            "atr_pct_min": ATR_PCT_MIN,
        },
        "sealed": {
            "file": "optimization_sealed.jsonl",
            "future_review_only": True,
            "opaque_until_explicit_sealed_eval": True,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="default: SNAPSHOT_DIR/materialization_manifest.json")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--weights-sha256")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tp-atr", type=float, default=TP_ATR)
    parser.add_argument("--sl-atr", type=float, default=SL_ATR)
    parser.add_argument("--horizon-bars", type=int, default=HORIZON_BARS)
    parser.add_argument("--export-max-horizon", type=int, default=DEFAULT_EXPORT_MAX_HORIZON)
    parser.add_argument("--optimization-min-score", type=float, default=0.0,
                        help="preregistered grid minimum threshold; exposure export only")
    parser.add_argument("--optimization-thresholds", required=True,
                        help="required comma-separated preregistered threshold grid")
    parser.add_argument("--inner-end", default=DEFAULT_INNER_END.isoformat(),
                        help="exclusive inner-export end; the interval to sealed start is purged")
    parser.add_argument("--sample-symbols", type=int, default=0,
                        help="0 uses all symbols; otherwise deterministic seed-hash sample")
    parser.add_argument("--sample-seed", default="fixed-w10-preholdout-v1")
    parser.add_argument("--gold-manifest", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--render-workers", type=int, default=0,
                        help=(
                            "bounded CPU workers for W10 render/white-letterbox tensor prefetch; "
                            "0 keeps the original synchronous path"
                        ))
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    candidate = OutcomeProtocol(
        threshold=args.threshold,
        tp_atr=args.tp_atr,
        sl_atr=args.sl_atr,
        horizon_bars=args.horizon_bars,
    ).validated()
    if int(args.export_max_horizon) <= 0:
        raise ValueError("export-max-horizon must be positive")
    if not np.isfinite(args.optimization_min_score) or not 0.0 <= args.optimization_min_score <= 1.0:
        raise ValueError("optimization-min-score must be finite and in [0, 1]")
    optimization_thresholds = parse_optimization_thresholds(args.optimization_thresholds)
    if optimization_thresholds[0] != float(args.optimization_min_score):
        raise ValueError(
            "minimum --optimization-thresholds value must equal --optimization-min-score"
        )
    if args.batch <= 0:
        raise ValueError("batch must be positive")
    if args.render_workers < 0:
        raise ValueError("render-workers must be non-negative")
    if args.render_workers > MAX_RENDER_WORKERS:
        raise ValueError(f"render-workers must be <= {MAX_RENDER_WORKERS}")
    inner_end = _utc_timestamp(args.inner_end, field="inner_end")
    if inner_end > DEFAULT_INNER_END:
        raise ValueError(
            f"inner_end may not exceed the registered inner boundary {DEFAULT_INNER_END.isoformat()}"
        )
    if inner_end >= SEALED_START:
        raise ValueError("inner_end must precede sealed start")
    start, end = validate_candidate_date_range(args.start, args.end, candidate)
    snapshot_dir = args.snapshot_dir.resolve()
    manifest = (args.manifest or snapshot_dir / "materialization_manifest.json").resolve()
    all_declarations = load_materialization_manifest(snapshot_dir, manifest)
    declarations, sampling = sample_declarations(
        all_declarations, args.sample_symbols, args.sample_seed,
    )
    gold_identities, gold_metadata = load_gold_identities(args.gold_manifest)
    frames: dict[str, pd.DataFrame] = {}
    checked = []
    for item in declarations:
        path = snapshot_dir / str(item["path"])
        frame = load_preholdout_csv(path, item)
        symbol = str(item.get("symbol") or path.stem)
        if symbol in frames:
            raise ValueError(f"duplicate symbol in manifest: {symbol}")
        frames[symbol] = frame
        checked.append({"path": path.name, "symbol": symbol, "rows": len(frame),
                        "max_materialized_time": pd.Timestamp(frame["open_time"].max()).isoformat()})
    plan = add_safety_fields({**protocol_record(start=start, end=end, candidate=candidate),
                              "plan_only": bool(args.plan_only), "snapshot_dir": str(snapshot_dir),
                              "manifest": str(manifest), "sources": checked,
                              "symbol_sampling": sampling,
                              "gold_manifest": gold_metadata,
                              "optimization_export": {
                                  "max_horizon": int(args.export_max_horizon),
                                  "min_score": float(args.optimization_min_score),
                                  "thresholds": list(optimization_thresholds),
                                  "atr_eligibility_protocol": ATR_ELIGIBILITY_PROTOCOL,
                                  "atr_pct_min": ATR_PCT_MIN,
                                  "min_score_semantics": (
                                      "preregistered parameter-grid minimum threshold; "
                                      "does not affect inference or baseline/candidate trade output; Gold is retained"
                                  ),
                                  "selection_algorithm": (
                                      "interval-max-horizon-label-crop, then per-threshold filter, "
                                      "symbol/decision/event sort, MIN_GAP_BARS=18 earliest dedup, "
                                      "threshold union, then force-include Gold exposure"
                                  ),
                                  "safe_last_decision_time": safe_last_decision_time(
                                      horizon_bars=args.export_max_horizon).isoformat(),
                                  "inner_end_exclusive": inner_end.isoformat(),
                                  "sealed_start_inclusive": SEALED_START.isoformat(),
                                  "sealed_end_exclusive": HOLDOUT_START.isoformat(),
                                  "purge_region_exported": False,
                              }})
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if args.out_dir is None or args.weights is None or not args.weights_sha256:
        raise ValueError("--out-dir, --weights, and --weights-sha256 are required unless --plan-only")
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    actual_weights_sha = sha256_file(args.weights)
    if actual_weights_sha != str(args.weights_sha256).lower():
        raise ValueError("weights SHA256 mismatch")

    started = time.monotonic()
    device = _pick_device(args.device)
    model, names, signal_idx = _load_classifier(args.weights, device)
    transform = _full_frame_transform()
    predictions: list[dict[str, Any]] = []
    skipped_span = 0
    for symbol, frame in frames.items():
        indices = decision_indices(frame, start, end, horizon_bars=candidate.horizon_bars)
        rows, skipped = classify_frame(model, transform, frame, symbol, indices, device=device,
                                       signal_idx=signal_idx, threshold=candidate.threshold,
                                       batch_size=args.batch,
                                       render_workers=args.render_workers)
        predictions.extend(rows)
        skipped_span += skipped
    validate_unique_records(predictions)
    optimization_rows, optimization_skips = build_optimization_rows(
        predictions,
        frames,
        (
            OptimizationInterval.from_values("inner", "1900-01-01T00:00:00Z", inner_end),
            OptimizationInterval.from_values("sealed", SEALED_START, HOLDOUT_START),
        ),
        args.export_max_horizon,
        gold_identities,
        args.optimization_min_score,
        optimization_thresholds,
    )
    inner_outputs = build_inner_backtest_outputs(predictions, frames, candidate, inner_end)
    inner_predictions = inner_outputs["predictions"]
    raw_signals = inner_outputs["signals_raw"]
    dedup_signals = inner_outputs["signals_dedup"]
    stop_losses = inner_outputs["stop_losses"]
    l1_review = inner_outputs["l1_review_queue"]
    l2_negative = inner_outputs["l2_outcome_negatives"]
    summary = add_safety_fields({
        **plan, "plan_only": False, "generated_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(args.weights.resolve()), "weights_sha256": actual_weights_sha,
        "device": device, "class_names": names, "n_symbols": len(frames),
        "batch_size": int(args.batch), "render_workers": int(args.render_workers),
        "render_prefetch_max_batches": (
            0 if int(args.render_workers) == 0 else min(MAX_RENDER_PREFETCH_BATCHES, int(args.render_workers) * 2)
        ),
        "n_predictions": len(inner_predictions), "n_signal_raw": len(raw_signals),
        "n_signal_dedup": len(dedup_signals), "n_stop_losses": len(stop_losses),
        "n_l1_review": len(l1_review), "n_l2_outcome_negatives": len(l2_negative),
        "atr_floor_excluded": inner_outputs["atr_floor_excluded"],
        "optimization_export_stats": optimization_skips,
        "raw": summarize_trades(raw_signals),
        "deduped": summarize_trades(dedup_signals),
        "outcome_scope": {
            "name": "optimization_inner",
            "decision_before": inner_end.isoformat(),
            "label_end_lte": inner_end.isoformat(),
            "sealed_outcomes_resolved": False,
            "sealed_predictions_exported_only_to": "optimization_sealed.jsonl",
        },
        "optimization_exports": public_optimization_exports_summary(optimization_rows),
        "wall_seconds": round(time.monotonic() - started, 3),
        "orders_placed": False, "promoted": False,
    })
    out_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (
        ("predictions.jsonl", inner_predictions), ("signals_raw.jsonl", raw_signals),
        ("signals_dedup.jsonl", dedup_signals), ("stop_losses.jsonl", stop_losses),
        ("l1_review_queue.jsonl", l1_review), ("l2_outcome_negatives.jsonl", l2_negative),
        ("optimization_inner.jsonl", optimization_rows["inner"]),
        ("optimization_sealed.jsonl", optimization_rows["sealed"]),
    ):
        _write_jsonl(out_dir / name, rows)
    for export in summary["optimization_exports"].values():
        if isinstance(export, dict) and "file" in export:
            export["sha256"] = sha256_file(out_dir / export["file"])
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
