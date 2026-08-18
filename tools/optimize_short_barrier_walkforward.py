#!/usr/bin/env python3
"""Select SHORT threshold/barriers on inner, purged walk-forward folds.

This tool is the parameter-selection companion to the fixed W10 pre-holdout
backtester.  The decisions here follow the owner-approved experiment protocol:
the model score and causal metadata are already frozen by the backtester, entry
is the next bar open, outcomes use the canonical barrier resolver, and costs are
read from :mod:`yoyo.contracts.costs`.  Same-bar handling, gap fills, the
18-bar event gap, entry policy, and route costs are deliberately not CLI knobs.

The ordinary mode accepts an *inner-only* JSONL and writes ``prereg.json``,
``fold_results.json``, and ``config_selection.json``.  The optional April final
evaluation must live in a different JSONL and is read only with
``--evaluate-sealed``.  It writes ``sealed_eval.json`` exactly once and records
the source digest and sealed interval in a caller-supplied global registry.  No
mode may materialize a decision or future bar at/after 2026-05-04 UTC.

Each JSONL row has this schema (extra provenance fields are preserved)::

    {"event_id":"...", "symbol":"BTC-USDT-SWAP",
     "decision_time":"2026-01-10T00:00:00Z", "p_signal":0.71,
     "entry_price":100.0, "atr14":1.0, "atr_pct":0.01,
     "atr_pct_min":0.0015,
     "atr_eligibility_protocol":"decision_atr_pct_floor_v1", "gold":true,
     "future_review_only":true,
     "causal_metadata":{"visible_end_time":"2026-01-10T00:00:00Z"},
     "future_path":[
       {"time":"2026-01-10T00:15:00Z", "open":100, "high":101,
        "low":99, "close":100}, ... up to max(horizons) bars ...]}

``future_path`` is label/evaluation data only.  It is never a model input.  A
fold JSON is ``{"folds":[...], "sealed_eval":...}``; every inner fold requires
``name``, ``calibration_start``, ``calibration_end``, ``validation_start``, and
``validation_end`` (half-open UTC intervals).  At least 162 15-minute bars must
separate calibration and validation.  The built-in folds validate January,
February, and March 2026; the sealed interval is 2026-04-02 through 2026-05-03.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yoyo.contracts.costs import (  # noqa: E402
    SWAP_MAKER,
    SWAP_TAKER,
    deduct_round_trip_cost_once,
)
from yoyo.contracts.outcomes import (  # noqa: E402
    ATR_ELIGIBILITY_PROTOCOL,
    ATR_PCT_MIN,
    resolve_barrier_outcome,
)
from yoyo.data.indicators import MIN_GAP_BARS  # noqa: E402

HOLDOUT_START = pd.Timestamp("2026-05-04T00:00:00Z")
BAR_DURATION = pd.Timedelta(minutes=15)
PURGE_BARS = 162
BASELINE = (0.50, 5.0, 2.0, 72)
SAME_BAR_POLICY = "conservative_sl"
GAP_POLICY = "barrier_price"
ENTRY_POLICY = "next_open"
RETURN_CONVENTION = "linear_short"
SCHEMA_VERSION = 1


@dataclass(frozen=True, order=True)
class SearchConfig:
    threshold: float
    tp_atr: float
    sl_atr: float
    horizon: int

    @property
    def key(self) -> str:
        return f"threshold={self.threshold:g},tp={self.tp_atr:g},sl={self.sl_atr:g},horizon={self.horizon}"


@dataclass(frozen=True)
class Fold:
    name: str
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass(frozen=True, init=False)
class Guards:
    min_trades: int
    max_gold_recall_drop_pp: float
    max_trigger_density_per_symbol_day: float
    min_positive_folds: int = 0
    min_worst_fold_maker_bp: float = -math.inf

    def __init__(
        self,
        min_trades: int,
        max_gold_recall_drop_pp: float,
        max_trigger_density_per_symbol_day: float | None = None,
        min_positive_folds: int = 0,
        min_worst_fold_maker_bp: float = -math.inf,
        *,
        max_trigger_density_per_day: float | None = None,
    ) -> None:
        if max_trigger_density_per_symbol_day is None:
            if max_trigger_density_per_day is None:
                raise TypeError("max_trigger_density_per_symbol_day is required")
            max_trigger_density_per_symbol_day = max_trigger_density_per_day
        elif max_trigger_density_per_day is not None and not math.isclose(
            float(max_trigger_density_per_symbol_day),
            float(max_trigger_density_per_day),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("density guard aliases disagree")
        object.__setattr__(self, "min_trades", min_trades)
        object.__setattr__(self, "max_gold_recall_drop_pp", max_gold_recall_drop_pp)
        object.__setattr__(
            self,
            "max_trigger_density_per_symbol_day",
            float(max_trigger_density_per_symbol_day),
        )
        object.__setattr__(self, "min_positive_folds", min_positive_folds)
        object.__setattr__(self, "min_worst_fold_maker_bp", min_worst_fold_maker_bp)

    @property
    def max_trigger_density_per_day(self) -> float:
        """Backward-compatible alias; the unit is exposed symbol-day."""
        return self.max_trigger_density_per_symbol_day


DEFAULT_FOLD_DOCUMENT: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "purge_bars": PURGE_BARS,
    "folds": [
        {"name": "2026-01", "calibration_start": "2025-09-01T00:00:00Z",
         "calibration_end": "2025-12-29T06:00:00Z",
         "validation_start": "2026-01-01T00:00:00Z", "validation_end": "2026-02-01T00:00:00Z"},
        {"name": "2026-02", "calibration_start": "2025-09-01T00:00:00Z",
         "calibration_end": "2026-01-29T06:00:00Z",
         "validation_start": "2026-02-01T00:00:00Z", "validation_end": "2026-03-01T00:00:00Z"},
        {"name": "2026-03", "calibration_start": "2025-09-01T00:00:00Z",
         "calibration_end": "2026-02-26T06:00:00Z",
         "validation_start": "2026-03-01T00:00:00Z", "validation_end": "2026-03-31T07:30:00Z"},
    ],
    "sealed_eval": {"name": "2026-04-final", "start": "2026-04-02T00:00:00Z",
                    "end": "2026-05-04T00:00:00Z"},
}


def utc(value: Any, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if pd.isna(stamp):
        raise ValueError(f"invalid {field}: {value!r}")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def parse_fold_document(raw: Mapping[str, Any]) -> tuple[list[Fold], dict[str, Any]]:
    """Parse and enforce expanding, disjoint, 162-bar-purged inner folds."""
    if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("fold config requires schema_version=1")
    if int(raw.get("purge_bars", PURGE_BARS)) != PURGE_BARS:
        raise ValueError(f"purge_bars is frozen at {PURGE_BARS}")
    folds: list[Fold] = []
    for item in raw.get("folds", []):
        fold = Fold(
            name=str(item["name"]),
            calibration_start=utc(item["calibration_start"], "calibration_start"),
            calibration_end=utc(item["calibration_end"], "calibration_end"),
            validation_start=utc(item["validation_start"], "validation_start"),
            validation_end=utc(item["validation_end"], "validation_end"),
        )
        if not fold.calibration_start < fold.calibration_end:
            raise ValueError(f"{fold.name}: calibration interval is empty")
        if fold.calibration_end + PURGE_BARS * BAR_DURATION > fold.validation_start:
            raise ValueError(f"{fold.name}: calibration/validation purge is under {PURGE_BARS} bars")
        if not fold.validation_start < fold.validation_end:
            raise ValueError(f"{fold.name}: validation interval is empty")
        if fold.validation_end > HOLDOUT_START:
            raise ValueError(f"{fold.name}: validation reaches the holdout")
        folds.append(fold)
    if not folds:
        raise ValueError("fold config has no inner folds")
    if len({fold.name for fold in folds}) != len(folds):
        raise ValueError("fold names must be unique")
    for previous, current in zip(folds, folds[1:]):
        if current.calibration_start != previous.calibration_start:
            raise ValueError("calibration must be expanding from one fixed start")
        if current.calibration_end <= previous.calibration_end:
            raise ValueError("calibration ends must strictly increase")
        if current.validation_start < previous.validation_end:
            raise ValueError("validation folds overlap or move backward")

    sealed = dict(raw.get("sealed_eval") or {})
    if not sealed:
        raise ValueError("fold config requires sealed_eval")
    sealed["name"] = str(sealed["name"])
    sealed["start"] = utc(sealed["start"], "sealed.start")
    sealed["end"] = utc(sealed["end"], "sealed.end")
    if not sealed["start"] < sealed["end"]:
        raise ValueError("sealed interval must follow inner validation folds")
    if folds[-1].validation_end + PURGE_BARS * BAR_DURATION > sealed["start"]:
        raise ValueError(f"last inner fold/sealed interval purge is under {PURGE_BARS} bars")
    if sealed["end"] > HOLDOUT_START:
        raise ValueError("sealed interval reaches the holdout")
    return folds, sealed


def build_grid(thresholds: Sequence[float], tp_atrs: Sequence[float],
               sl_atrs: Sequence[float], horizons: Sequence[int]) -> list[SearchConfig]:
    """Build a deterministic explicit grid and require the frozen baseline arm."""
    if not thresholds or not tp_atrs or not sl_atrs or not horizons:
        raise ValueError("all four parameter lists must be non-empty")
    if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in thresholds):
        raise ValueError("thresholds must be finite and in [0, 1]")
    if any(not math.isfinite(float(x)) or float(x) <= 0 for x in (*tp_atrs, *sl_atrs)):
        raise ValueError("TP/SL ATR multipliers must be finite and positive")
    if any(int(x) <= 0 for x in horizons):
        raise ValueError("horizons must be positive")
    grid = sorted({SearchConfig(float(t), float(tp), float(sl), int(h))
                   for t in thresholds for tp in tp_atrs for sl in sl_atrs for h in horizons})
    if SearchConfig(*BASELINE) not in grid:
        raise ValueError("explicit grid must contain baseline threshold=0.5,tp=5,sl=2,horizon=72")
    return grid


def load_single_variable_grid(raw: Mapping[str, Any]) -> tuple[list[SearchConfig], dict[str, Any]]:
    """Validate a preregistered one-variable stage plus baseline control arm."""
    stage = str(raw.get("stage", "")).strip()
    variable = str(raw.get("variable_under_test", ""))
    allowed = {"threshold", "tp_atr", "sl_atr", "horizon"}
    if not stage:
        raise ValueError("grid JSON requires a non-empty stage")
    if variable not in allowed:
        raise ValueError(f"variable_under_test must be one of {sorted(allowed)}")
    try:
        locked = SearchConfig(**raw["locked_config"])
        grid = sorted({SearchConfig(**item) for item in raw["configs"]})
    except (KeyError, TypeError) as exc:
        raise ValueError("grid JSON requires locked_config and configs") from exc
    # Reuse all numeric and mandatory-control validation without creating a
    # cartesian expansion.
    for config in grid:
        build_grid([config.threshold, BASELINE[0]], [config.tp_atr, BASELINE[1]],
                   [config.sl_atr, BASELINE[2]], [config.horizon, BASELINE[3]])
    baseline = SearchConfig(*BASELINE)
    if locked not in grid:
        raise ValueError("locked_config must be an explicit grid arm")
    if baseline not in grid:
        raise ValueError("single-variable grid must contain the baseline control arm")
    fields = ("threshold", "tp_atr", "sl_atr", "horizon")
    for config in grid:
        if config == baseline:
            continue
        differences = {field for field in fields if getattr(config, field) != getattr(locked, field)}
        if differences - {variable}:
            raise ValueError(
                f"{config.key}: changes {sorted(differences)} outside variable_under_test={variable}"
            )
    discipline = {
        "experimental_discipline": "single_variable",
        "stage": stage,
        "variable_under_test": variable,
        "locked_config": asdict(locked),
    }
    return grid, discipline


def _path_frame(row: Mapping[str, Any]) -> pd.DataFrame:
    path = row.get("future_path")
    if not isinstance(path, list) or not path:
        raise ValueError(f"{row.get('event_id')}: future_path must be a non-empty list")
    frame = pd.DataFrame(path).rename(columns={"time": "open_time"})
    required = {"open_time", "open", "high", "low", "close"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"{row.get('event_id')}: future_path missing {missing}")
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    if frame["open_time"].isna().any() or frame["open_time"].duplicated().any():
        raise ValueError(f"{row.get('event_id')}: invalid/duplicate future timestamps")
    if not frame["open_time"].is_monotonic_increasing:
        raise ValueError(f"{row.get('event_id')}: future_path is not ordered")
    if (frame["open_time"] >= HOLDOUT_START).any():
        raise ValueError(f"{row.get('event_id')}: future_path reaches holdout")
    return frame


def validate_candidates(rows: Sequence[Mapping[str, Any]], *, max_horizon: int,
                        allowed_start: pd.Timestamp | None = None,
                        allowed_end: pd.Timestamp | None = None) -> list[dict[str, Any]]:
    """Fail closed on provenance, identities, next-open alignment, and holdout."""
    output: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    identities: set[tuple[str, pd.Timestamp]] = set()
    for raw in rows:
        row = dict(raw)
        missing = [key for key in ("event_id", "symbol", "decision_time", "p_signal",
                                   "entry_price", "atr14", "atr_pct", "atr_pct_min",
                                   "atr_eligibility_protocol", "causal_metadata", "future_path")
                   if key not in row]
        if missing:
            raise ValueError(f"candidate missing fields: {missing}")
        event_id, symbol = str(row["event_id"]), str(row["symbol"])
        if not event_id or not symbol:
            raise ValueError("candidate event_id and symbol must be non-empty")
        decision = utc(row["decision_time"], "decision_time")
        if decision >= HOLDOUT_START:
            raise ValueError(f"{event_id}: decision reaches holdout")
        if allowed_start is not None and decision < allowed_start:
            raise ValueError(f"{event_id}: decision precedes allowed source interval")
        if allowed_end is not None and decision >= allowed_end:
            raise ValueError(f"{event_id}: decision exceeds allowed source interval")
        if event_id in event_ids or (symbol, decision) in identities:
            raise ValueError(f"duplicate event identity: {event_id} {symbol} {decision.isoformat()}")
        event_ids.add(event_id)
        identities.add((symbol, decision))
        score, entry, atr = float(row["p_signal"]), float(row["entry_price"]), float(row["atr14"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{event_id}: invalid p_signal")
        if not math.isfinite(entry) or entry <= 0 or not math.isfinite(atr) or atr <= 0:
            raise ValueError(f"{event_id}: entry_price/atr14 must be finite and positive")
        try:
            atr_pct, declared_floor = float(row["atr_pct"]), float(row["atr_pct_min"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{event_id}: atr_pct/atr_pct_min must be finite numbers") from exc
        if not math.isfinite(atr_pct) or atr_pct < ATR_PCT_MIN:
            raise ValueError(f"{event_id}: atr_pct must be finite and >= frozen floor {ATR_PCT_MIN}")
        if not math.isfinite(declared_floor) or declared_floor != ATR_PCT_MIN:
            raise ValueError(f"{event_id}: atr_pct_min does not match frozen floor {ATR_PCT_MIN}")
        if row["atr_eligibility_protocol"] != ATR_ELIGIBILITY_PROTOCOL:
            raise ValueError(f"{event_id}: atr eligibility protocol mismatch")
        causal = row["causal_metadata"]
        if not isinstance(causal, Mapping) or "visible_end_time" not in causal:
            raise ValueError(f"{event_id}: causal_metadata.visible_end_time is required")
        if utc(causal["visible_end_time"], "visible_end_time") > decision:
            raise ValueError(f"{event_id}: causal metadata looks beyond decision")
        if row.get("future_review_only") is not True:
            raise ValueError(f"{event_id}: future_path must be marked future_review_only=true")
        frame = _path_frame(row)
        if len(frame) < int(max_horizon):
            raise ValueError(f"{event_id}: future_path shorter than max horizon {max_horizon}")
        expected_entry_time = decision + BAR_DURATION
        if frame["open_time"].iloc[0] != expected_entry_time:
            raise ValueError(f"{event_id}: future_path must begin at next open")
        if not math.isclose(float(frame["open"].iloc[0]), entry, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{event_id}: entry_price differs from next open")
        row["decision_time"] = decision.isoformat()
        row["_decision_ts"] = decision
        row["_path_frame"] = frame
        output.append(row)
    return output


def gap_deduplicate(rows: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Threshold first, then keep chronologically first events 18 bars apart."""
    eligible = [dict(row) for row in rows if float(row["p_signal"]) >= float(threshold)]
    eligible.sort(key=lambda row: (str(row["symbol"]), utc(row["decision_time"], "decision_time"),
                                   str(row["event_id"])))
    selected: list[dict[str, Any]] = []
    last: dict[str, pd.Timestamp] = {}
    gap = MIN_GAP_BARS * BAR_DURATION
    for row in eligible:
        symbol = str(row["symbol"])
        stamp = utc(row["decision_time"], "decision_time")
        if symbol not in last or stamp - last[symbol] >= gap:
            selected.append(row)
            last[symbol] = stamp
    return selected


def _resolve(row: Mapping[str, Any], config: SearchConfig) -> dict[str, Any]:
    frame = row.get("_path_frame")
    if frame is None:
        frame = _path_frame(row)
    resolved = resolve_barrier_outcome(
        frame, side="short", entry_i=0, entry_price=float(row["entry_price"]),
        atr=float(row["atr14"]), tp_atr_mult=config.tp_atr, sl_atr_mult=config.sl_atr,
        horizon_bars=config.horizon, same_bar_policy=SAME_BAR_POLICY,
        gap_policy=GAP_POLICY, return_convention=RETURN_CONVENTION,
        allow_partial=False, bar_duration=BAR_DURATION,
    )
    gross = float(resolved.gross_ret)
    return {
        "gross_ret": gross,
        "net_maker": deduct_round_trip_cost_once(
            gross, input_semantics="gross", target_semantics="net_maker"),
        "net_taker": deduct_round_trip_cost_once(
            gross, input_semantics="gross", target_semantics="net_taker"),
        "outcome": resolved.outcome,
        "exit_time": resolved.exit_time,
    }


def evaluate_fold(rows: Sequence[Mapping[str, Any]], fold: Fold, config: SearchConfig,
                  guards: Guards) -> dict[str, Any]:
    fold_candidates = [row for row in rows
                       if fold.validation_start <= utc(row["decision_time"], "decision_time")
                       < fold.validation_end]
    # Label availability is an exposure property for this arm.  Crop before
    # thresholding and gap selection so an unusable month-end row cannot block
    # a nearby usable event, enter the Gold denominator, or reach the resolver.
    candidates: list[Mapping[str, Any]] = []
    n_label_end_cropped = 0
    for row in fold_candidates:
        frame = row.get("_path_frame")
        if frame is None:
            frame = _path_frame(row)
        label_end = pd.Timestamp(frame["open_time"].iloc[config.horizon - 1]) + BAR_DURATION
        if label_end > fold.validation_end:
            n_label_end_cropped += 1
        else:
            candidates.append(row)
    chosen = gap_deduplicate(candidates, config.threshold)
    chosen.sort(key=lambda row: (utc(row["decision_time"], "decision_time"),
                                 str(row["symbol"]), str(row["event_id"])))
    results = [_resolve(row, config) for row in chosen]
    n = len(results)
    maker = [float(result["net_maker"]) for result in results]
    duration_days = (fold.validation_end - fold.validation_start).total_seconds() / 86400.0
    exposure_symbols = {str(row["symbol"]) for row in fold_candidates if str(row["symbol"])}
    exposure_symbol_days = len(exposure_symbols) * duration_days
    density = n / exposure_symbol_days if exposure_symbol_days > 0 else None

    outcomes: dict[str, int] = {}
    for result in results:
        outcome = str(result["outcome"])
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    gains = sum(value for value in maker if value > 0)
    losses = -sum(value for value in maker if value < 0)
    profit_factor = gains / losses if losses > 0 else None
    cumulative = peak = max_drawdown = 0.0
    for value in maker:
        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)

    gold_all = [row for row in candidates if row.get("gold") is True]
    gold_selected = sum(row.get("gold") is True for row in chosen)
    gold_recall = None if not gold_all else gold_selected / len(gold_all)
    baseline_chosen = gap_deduplicate(candidates, BASELINE[0])
    baseline_gold_selected = sum(row.get("gold") is True for row in baseline_chosen)
    baseline_gold_recall = None if not gold_all else baseline_gold_selected / len(gold_all)
    recall_drop_pp = None if gold_recall is None else (baseline_gold_recall - gold_recall) * 100.0

    violations: list[str] = []
    if exposure_symbol_days <= 0:
        violations.append("missing_symbol_exposure")
    if n < guards.min_trades:
        violations.append("min_trades")
    if (density is not None
            and density > guards.max_trigger_density_per_symbol_day):
        violations.append("max_trigger_density")
    if recall_drop_pp is not None and recall_drop_pp > guards.max_gold_recall_drop_pp + 1e-12:
        violations.append("gold_recall_drop")
    return {
        "fold": fold.name, "config": asdict(config), "config_key": config.key,
        "n_candidates": len(fold_candidates), "n_usable_candidates": len(candidates),
        "n_label_end_cropped": n_label_end_cropped, "n_trades": n,
        "n_exposure_symbols": len(exposure_symbols),
        "candidate_exposure_symbol_days": exposure_symbol_days,
        "trigger_density_per_symbol_day": density,
        "mean_net_maker": None if not maker else sum(maker) / n,
        "total_net_maker": sum(maker),
        "mean_net_maker_bp": None if not maker else sum(maker) / n * 1e4,
        "outcomes": outcomes, "profit_factor_maker": profit_factor,
        "win_rate_maker": None if not maker else sum(value > 0 for value in maker) / n,
        "worst_trade_maker": None if not maker else min(maker),
        "max_drawdown_maker": max_drawdown,
        "gold_count": len(gold_all), "gold_recall": gold_recall,
        "baseline_gold_recall": baseline_gold_recall, "gold_recall_drop_pp": recall_drop_pp,
        "guard_violations": violations, "eligible": not violations,
    }


def _distance_from_baseline(config: SearchConfig) -> tuple[int, float]:
    values = (config.threshold, config.tp_atr, config.sl_atr, config.horizon)
    changed = sum(left != right for left, right in zip(values, BASELINE))
    distance = (abs(config.threshold - BASELINE[0]) / 0.05
                + abs(config.tp_atr - BASELINE[1]) + abs(config.sl_atr - BASELINE[2])
                + abs(config.horizon - BASELINE[3]) / 12.0)
    return changed, distance


def select_config(rows: Sequence[Mapping[str, Any]], folds: Sequence[Fold],
                  grid: Sequence[SearchConfig], guards: Guards,
                  discipline: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate every arm; select by event mean then fold stability tie-breaks."""
    fold_results = [evaluate_fold(rows, fold, config, guards) for config in grid for fold in folds]
    rankings: list[dict[str, Any]] = []
    for config in grid:
        items = [item for item in fold_results if item["config_key"] == config.key]
        fold_eligible = all(item["eligible"] for item in items)
        means = [float(item["mean_net_maker"]) for item in items if item["mean_net_maker"] is not None]
        total_n = sum(int(item["n_trades"]) for item in items)
        pooled = (sum(float(item["total_net_maker"]) for item in items) / total_n
                  if total_n else None)
        worst = min(means) if len(means) == len(items) else None
        mean_of_folds = sum(means) / len(means) if means else None
        std = (sum((value - mean_of_folds) ** 2 for value in means) / len(means)) ** 0.5 if means else None
        changed, distance = _distance_from_baseline(config)
        positive_folds = sum(value > 0 for value in means)
        aggregate_violations: list[str] = []
        if positive_folds < guards.min_positive_folds:
            aggregate_violations.append("min_positive_folds")
        if worst is None or worst * 1e4 < guards.min_worst_fold_maker_bp:
            aggregate_violations.append("min_worst_fold_maker_bp")
        eligible = fold_eligible and not aggregate_violations
        rankings.append({
            "config": asdict(config), "config_key": config.key, "eligible": eligible,
            "total_trades": total_n, "pooled_mean_net_maker": pooled,
            "positive_folds": positive_folds, "worst_fold_mean": worst,
            "worst_fold_mean_bp": None if worst is None else worst * 1e4,
            "fold_mean_std": std, "changed_dimensions": changed,
            "baseline_distance": distance,
            "guard_violations": sorted(
                {violation for item in items for violation in item["guard_violations"]}
                | set(aggregate_violations)
            ),
        })
    eligible_rankings = [item for item in rankings if item["eligible"] and item["pooled_mean_net_maker"] is not None]
    if not eligible_rankings:
        raise ValueError("no parameter configuration passes all hard guards")
    eligible_rankings.sort(key=lambda item: (
        -float(item["pooled_mean_net_maker"]), -int(item["positive_folds"]),
        -float(item["worst_fold_mean"]), float(item["fold_mean_std"]),
        int(item["changed_dimensions"]), float(item["baseline_distance"]), item["config_key"],
    ))
    selected = eligible_rankings[0]
    selection = {
        "selected_config": selected["config"], "selected_config_key": selected["config_key"],
        "selection_order": ["pooled_mean_net_maker", "positive_folds", "worst_fold_mean",
                            "fold_mean_std", "changed_dimensions", "baseline_distance"],
        "auto_promote": False, "rankings": rankings,
    }
    selection.update(discipline or {"experimental_discipline": "multi_factor_exploratory"})
    return selection, fold_results


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{number}: row must be an object")
                rows.append(item)
    if not rows:
        raise ValueError(f"{path}: no candidate rows")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _require_schema_version(raw: Mapping[str, Any], artifact: str) -> None:
    version = raw.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(f"{artifact} requires schema_version=1")


def _require_unsealed_artifact(raw: Mapping[str, Any], artifact: str) -> None:
    _require_schema_version(raw, artifact)
    if raw.get("sealed_read") is not False:
        raise ValueError(f"{artifact} must have sealed_read=false")


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _exclusive_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    acquired = False
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        descriptor = None
        acquired = True
        yield
    except FileExistsError as exc:
        raise ValueError(f"registry is locked by another transaction: {lock_path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if acquired:
            lock_path.unlink(missing_ok=True)


def consume_sealed_once(registry_path: Path, *, source_path: Path, sealed: Mapping[str, Any],
                        result_path: Path, payload: Mapping[str, Any]) -> str:
    """Fail closed while claiming a source+interval key in a global ledger."""
    source_digest = _sha256(source_path)
    # Identity is content + interval, not pathname: copying an already consumed
    # sealed file must not manufacture a second evaluation opportunity.
    material = f"{source_digest}\0{sealed['start']}\0{sealed['end']}".encode()
    key = hashlib.sha256(material).hexdigest()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(registry_path.with_name(registry_path.name + ".lock")):
        if registry_path.exists():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        else:
            registry = {"schema_version": SCHEMA_VERSION, "sealed_consumptions": {}}
        if (registry.get("schema_version") != SCHEMA_VERSION
                or not isinstance(registry.get("sealed_consumptions"), dict)):
            raise ValueError("invalid global sealed registry")
        if key in registry["sealed_consumptions"]:
            raise ValueError("this sealed source and interval have already been consumed")
        claim = {
            "status": "pending",
            "source": str(source_path.resolve()), "source_sha256": source_digest,
            "sealed_name": sealed["name"], "start": sealed["start"], "end": sealed["end"],
            "result": str(result_path.resolve()),
        }
        registry["sealed_consumptions"][key] = claim
        _write_json_atomic(registry_path, registry)
        write_new_json(result_path, payload)
        registry["sealed_consumptions"][key] = {
            **claim,
            "status": "complete",
            "result_sha256": _sha256(result_path),
        }
        _write_json_atomic(registry_path, registry)
    return key


def _validate_selected_config_artifact(
    selection_path: Path, *, grid: Sequence[SearchConfig], guards: Guards,
    fold_doc: Mapping[str, Any], discipline: Mapping[str, Any], folds: Sequence[Fold],
    sealed: Mapping[str, Any], sealed_source_path: Path,
) -> SearchConfig:
    """Prove sealed mode is locked to the sibling unsealed inner-search output."""
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise ValueError("selected config artifact must be a JSON object")
    _require_unsealed_artifact(selection, "selected config artifact")
    try:
        selected = SearchConfig(**selection["selected_config"])
    except (KeyError, TypeError) as exc:
        raise ValueError("selected config artifact requires selected_config") from exc
    if selection.get("selected_config_key") != selected.key:
        raise ValueError("selected_config_key does not match selected_config")
    if selected not in grid:
        raise ValueError("selected config is absent from the preregistered explicit grid")

    prereg_path = selection_path.parent / "prereg.json"
    if not prereg_path.exists():
        raise ValueError("selected config artifact requires sibling prereg.json")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if not isinstance(prereg, dict):
        raise ValueError("sibling prereg.json must be a JSON object")
    _require_unsealed_artifact(prereg, "sibling prereg.json")
    expected_prereg = _jsonable(_protocol(grid, guards, fold_doc, discipline))
    for key, expected_value in expected_prereg.items():
        if prereg.get(key) != expected_value:
            raise ValueError(f"sibling prereg.json does not match current protocol field {key}")

    expected_prereg_sha = _sha256(prereg_path)
    if selection.get("prereg_sha256") != expected_prereg_sha:
        raise ValueError("selected config artifact prereg_sha256 does not match sibling prereg.json")
    candidate_source = prereg.get("candidate_source")
    candidate_source_sha = prereg.get("candidate_source_sha256")
    if not isinstance(candidate_source, str) or not candidate_source:
        raise ValueError("sibling prereg.json requires candidate_source")
    if not isinstance(candidate_source_sha, str) or len(candidate_source_sha) != 64:
        raise ValueError("sibling prereg.json requires candidate_source_sha256")
    candidate_path = Path(candidate_source)
    if not candidate_path.exists() or not candidate_path.is_file():
        raise ValueError("sibling prereg.json candidate_source is not a readable file")
    if candidate_path.resolve() == sealed_source_path.resolve():
        raise ValueError("inner candidate_source must be physically separate from sealed source")
    if _sha256(candidate_path) != candidate_source_sha:
        raise ValueError("sibling prereg.json candidate_source_sha256 does not match candidate_source")

    rows = validate_candidates(read_jsonl(candidate_path),
                               max_horizon=max(config.horizon for config in grid),
                               allowed_end=sealed["start"])
    recomputed_selection, _ = select_config(rows, folds, grid, guards, discipline)
    recomputed = _jsonable(recomputed_selection)
    for key in ("selected_config", "selected_config_key", "selection_order", "auto_promote",
                "rankings", "experimental_discipline", "stage", "variable_under_test",
                "locked_config"):
        if key in selection or key in recomputed:
            if selection.get(key) != recomputed.get(key):
                raise ValueError(f"selected config artifact does not match recomputed inner search field {key}")
    return selected


def _csv_numbers(value: str, cast: type) -> list[Any]:
    try:
        return [cast(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid comma-separated values: {value}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, help="physical inner-fold-only JSONL")
    parser.add_argument("--fold-config", type=Path, help="default: built-in Jan/Feb/Mar + sealed April")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grid-json", type=Path,
                        help="single-variable preregistration; mutually exclusive with comma grids")
    parser.add_argument("--thresholds")
    parser.add_argument("--tp-atrs")
    parser.add_argument("--sl-atrs")
    parser.add_argument("--horizons")
    parser.add_argument("--min-trades", type=int, required=True)
    parser.add_argument("--max-gold-recall-drop-pp", type=float, required=True)
    parser.add_argument("--max-trigger-density-per-symbol-day", "--max-trigger-density-per-day",
                        dest="max_trigger_density_per_symbol_day", type=float, required=True,
                        help=("maximum trades per exposed symbol-day; "
                              "--max-trigger-density-per-day is a backward-compatible alias"))
    parser.add_argument("--min-positive-folds", type=int, required=True)
    parser.add_argument("--min-worst-fold-maker-bp", type=float, required=True)
    parser.add_argument("--evaluate-sealed", type=Path, metavar="SEALED_JSONL")
    parser.add_argument("--selected-config", type=Path,
                        help="config_selection.json from an already completed inner search")
    parser.add_argument("--global-registry", type=Path)
    return parser


def _protocol(grid: Sequence[SearchConfig], guards: Guards, fold_doc: Mapping[str, Any],
              discipline: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION, "side": "short", "entry_policy": ENTRY_POLICY,
        "same_bar_policy": SAME_BAR_POLICY, "gap_policy": GAP_POLICY,
        "return_convention": RETURN_CONVENTION, "min_gap_bars": MIN_GAP_BARS,
        "atr_eligibility_protocol": ATR_ELIGIBILITY_PROTOCOL,
        "atr_pct_min": ATR_PCT_MIN,
        "purge_bars": PURGE_BARS, "holdout_start": HOLDOUT_START,
        "costs": {"swap_maker": SWAP_MAKER, "swap_taker": SWAP_TAKER},
        "baseline": {"threshold": BASELINE[0], "tp_atr": BASELINE[1],
                     "sl_atr": BASELINE[2], "horizon": BASELINE[3]},
        "grid": [asdict(config) for config in grid], "guards": asdict(guards),
        "density_guard_semantics": {
            "metric": "trigger_density_per_symbol_day",
            "guard": "max_trigger_density_per_symbol_day",
            "denominator": "n_exposure_symbols * fold_duration_days",
            "n_exposure_symbols_source": (
                "distinct symbols in fold_candidates before label cropping, score thresholding, "
                "same-symbol gap deduplication, and Gold forced-row effects"
            ),
            "threshold_union_limitation": (
                "candidate JSONL cannot prove symbols with zero candidate rows were exposed; "
                "zero-row symbols are not counted"
            ),
        },
        "fold_document": fold_doc, "future_review_only": True, "auto_promote": False,
    }
    payload.update(discipline)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    fold_doc = (json.loads(args.fold_config.read_text(encoding="utf-8"))
                if args.fold_config else DEFAULT_FOLD_DOCUMENT)
    folds, sealed = parse_fold_document(fold_doc)
    comma_values = (args.thresholds, args.tp_atrs, args.sl_atrs, args.horizons)
    if args.grid_json is not None:
        if any(value is not None for value in comma_values):
            raise ValueError("--grid-json is mutually exclusive with comma-grid arguments")
        grid, discipline = load_single_variable_grid(
            json.loads(args.grid_json.read_text(encoding="utf-8"))
        )
    else:
        if any(value is None for value in comma_values):
            raise ValueError("provide --grid-json or all four comma-grid arguments")
        grid = build_grid(_csv_numbers(args.thresholds, float), _csv_numbers(args.tp_atrs, float),
                          _csv_numbers(args.sl_atrs, float), _csv_numbers(args.horizons, int))
        discipline = {"experimental_discipline": "multi_factor_exploratory"}
    guards = Guards(args.min_trades, args.max_gold_recall_drop_pp,
                    args.max_trigger_density_per_symbol_day, args.min_positive_folds,
                    args.min_worst_fold_maker_bp)
    if (guards.min_trades <= 0 or guards.max_gold_recall_drop_pp < 0
            or guards.max_trigger_density_per_symbol_day <= 0
            or not 0 <= guards.min_positive_folds <= len(folds)
            or not math.isfinite(guards.min_worst_fold_maker_bp)):
        raise ValueError("invalid hard guards")
    if args.out_dir.exists() and (not args.out_dir.is_dir() or any(args.out_dir.iterdir())):
        raise ValueError("out-dir already exists and is non-empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate_sealed:
        if args.candidates is not None:
            raise ValueError("sealed mode refuses --candidates; sources must be physically separate")
        if args.selected_config is None or args.global_registry is None:
            raise ValueError("sealed mode requires --selected-config and --global-registry")
        selected = _validate_selected_config_artifact(
            args.selected_config, grid=grid, guards=guards, fold_doc=fold_doc,
            discipline=discipline, folds=folds, sealed=sealed,
            sealed_source_path=args.evaluate_sealed,
        )
        raw_rows = read_jsonl(args.evaluate_sealed)
        rows = validate_candidates(raw_rows, max_horizon=max(config.horizon for config in grid),
                                   allowed_start=sealed["start"], allowed_end=sealed["end"])
        sealed_fold = Fold(str(sealed["name"]), folds[-1].calibration_start,
                           folds[-1].validation_end, sealed["start"], sealed["end"])
        baseline = SearchConfig(*BASELINE)
        selected_result = evaluate_fold(rows, sealed_fold, selected, guards)
        baseline_result = evaluate_fold(rows, sealed_fold, baseline, guards)

        def delta(field: str) -> float | int | None:
            selected_value = selected_result[field]
            baseline_value = baseline_result[field]
            if selected_value is None or baseline_value is None:
                return None
            return selected_value - baseline_value

        payload = {"schema_version": SCHEMA_VERSION, "sealed": True, "config_locked": asdict(selected),
                   "selected": selected_result, "baseline": baseline_result,
                   "delta_selected_minus_baseline": {
                       "mean_net_maker": delta("mean_net_maker"),
                       "total_net_maker": delta("total_net_maker"),
                       "n_trades": delta("n_trades"),
                       "gold_recall": delta("gold_recall"),
                   },
                   "holdout_read": False, "auto_promote": False}
        consume_sealed_once(args.global_registry, source_path=args.evaluate_sealed, sealed=sealed,
                            result_path=args.out_dir / "sealed_eval.json", payload=payload)
        return 0

    if args.candidates is None:
        raise ValueError("inner mode requires --candidates")
    if args.selected_config is not None or args.global_registry is not None:
        raise ValueError("inner mode does not accept sealed-only arguments")
    # The inner source must end before the physical sealed boundary.  Keeping the
    # April source in another file is what makes accidental final evaluation
    # impossible in the normal optimizer invocation.
    rows = validate_candidates(read_jsonl(args.candidates),
                               max_horizon=max(config.horizon for config in grid),
                               allowed_end=sealed["start"])
    selection, fold_results = select_config(rows, folds, grid, guards, discipline)
    prereg = _protocol(grid, guards, fold_doc, discipline)
    prereg.update({"candidate_source": str(args.candidates.resolve()),
                   "candidate_source_sha256": _sha256(args.candidates), "sealed_read": False})
    write_new_json(args.out_dir / "prereg.json", prereg)
    write_new_json(args.out_dir / "fold_results.json",
                   {"schema_version": SCHEMA_VERSION, "results": fold_results})
    selection["prereg_sha256"] = _sha256(args.out_dir / "prereg.json")
    write_new_json(args.out_dir / "config_selection.json",
                   {"schema_version": SCHEMA_VERSION, **selection, "sealed_read": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
