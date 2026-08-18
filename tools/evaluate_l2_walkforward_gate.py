#!/usr/bin/env python3
"""Train, calibrate, and OOS-evaluate a non-deployable SHORT L2 gate.

This tool is the integrated self-evolution gate after
``build_l2_feedback_dataset.py``.  It deliberately ignores that builder's
historical train/validation split and re-slices by time: train before the
calibration purge boundary, calibrate before the test purge boundary, and test
only after the selected threshold is locked.  Candidate future paths are used
only for SHORT barrier labels/economics; the 28 LightGBM inputs are recomputed
from a SHA-gated OHLCV snapshot sliced at each decision bar.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_l2_feedback_dataset import (  # noqa: E402
    DEFAULT_PURGE_BARS,
    FEATURE_SEMANTICS,
    HOLDOUT_START,
    INCLUDED_OUTCOMES,
    INNER_CUTOFF,
    _feature_vector,
    _load_frames,
    _utc,
)
import tools.optimize_short_barrier_walkforward as optimizer  # noqa: E402
from tools.optimize_short_barrier_walkforward import (  # noqa: E402
    BAR_DURATION,
    SearchConfig,
    _resolve,
    gap_deduplicate,
    read_jsonl,
    validate_candidates,
)
from tools.train_l2_feedback_model import (  # noqa: E402
    DATASET_COLUMNS,
    LGB_PARAMS,
    _false_only,
    _json_object,
    _sha256,
    _true_only,
    _utc_series,
)
from yoyo.layers.l2_judgment.features import FEATURE_COLUMNS  # noqa: E402

SCHEMA_VERSION = 1
PROTOCOL = "l2_walkforward_gate_v1"
DEFAULT_CALIBRATION_START = "2026-02-01T00:00:00Z"
DEFAULT_TEST_START = "2026-03-01T00:00:00Z"
DEFAULT_TEST_END = "2026-03-31T07:30:00Z"
FIXED_TRAIN_START = pd.Timestamp("2026-01-01T00:00:00Z")
FIXED_CALIBRATION_START = pd.Timestamp(DEFAULT_CALIBRATION_START)
FIXED_TEST_START = pd.Timestamp(DEFAULT_TEST_START)
FIXED_TEST_END = pd.Timestamp(DEFAULT_TEST_END)
OUTPUT_FILES = (
    "model.txt",
    "prereg.json",
    "calibration_results.json",
    "selection.json",
    "test_eval.json",
    "feature_importance.csv",
    "lineage.json",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _sha_map(directory: Path) -> dict[str, str]:
    return {name: _sha256(directory / name) for name in OUTPUT_FILES}


def _parse_threshold_grid(raw: str) -> list[float]:
    if not raw or not raw.strip():
        raise ValueError("--l2-thresholds is required")
    values: list[float] = []
    for part in raw.split(","):
        if not part.strip():
            continue
        value = float(part)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("all L2 thresholds must be finite and in [0, 1]")
        values.append(value)
    grid = sorted(set(values))
    if not grid:
        raise ValueError("--l2-thresholds produced an empty grid")
    return grid


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(label): int((frame["label"] == label).sum()) for label in (0, 1)}


def _load_training_rows(
    dataset_dir: Path,
    *,
    expected_manifest_sha256: str,
    calibration_start: pd.Timestamp,
    purge_bars: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.json"
    dataset_path = dataset_dir / "dataset.csv"
    expected = expected_manifest_sha256.strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("--training-manifest-sha256 must be 64 lowercase hex")
    if not manifest_path.is_file() or not dataset_path.is_file():
        raise ValueError("--training-dataset-dir must contain manifest.json and dataset.csv")
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != expected:
        raise ValueError("training manifest SHA256 mismatch")
    manifest = _json_object(manifest_path)
    dataset_decl = manifest.get("dataset")
    if not isinstance(dataset_decl, Mapping) or dataset_decl.get("path") != "dataset.csv":
        raise ValueError("training manifest dataset contract is invalid")
    dataset_sha = _sha256(dataset_path)
    if dataset_decl.get("sha256") != dataset_sha:
        raise ValueError("training dataset.csv SHA256 mismatch")
    if type(dataset_decl.get("rows")) is not int or dataset_decl["rows"] <= 0:
        raise ValueError("training manifest dataset.rows must be positive")
    contract = manifest.get("protocol_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("training manifest protocol_contract is missing")
    if contract.get("side") != "short":
        raise ValueError("training manifest must declare side=short")
    if contract.get("feature_semantics") != FEATURE_SEMANTICS:
        raise ValueError(f"training manifest must declare feature_semantics={FEATURE_SEMANTICS}")
    if contract.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("training manifest feature_columns must exactly equal FEATURE_COLUMNS")
    if contract.get("included_outcomes") != INCLUDED_OUTCOMES:
        raise ValueError("training manifest outcome-label contract mismatch")
    if manifest.get("holdout_rows_read") != 0:
        raise ValueError("training manifest must prove holdout_rows_read=0")
    if manifest.get("future_rows_in_features") != 0:
        raise ValueError("training manifest must prove future_rows_in_features=0")
    if manifest.get("auto_promote") is not False:
        raise ValueError("training manifest must prove auto_promote=false")
    if pd.Timestamp(manifest.get("inner_cutoff")).tz_convert("UTC") != INNER_CUTOFF:
        raise ValueError("training manifest inner_cutoff mismatch")
    if pd.Timestamp(manifest.get("holdout_start")).tz_convert("UTC") != HOLDOUT_START:
        raise ValueError("training manifest holdout_start mismatch")

    data = pd.read_csv(dataset_path)
    if len(data) != dataset_decl["rows"]:
        raise ValueError("training dataset row count mismatch")
    if list(data.columns) != DATASET_COLUMNS:
        raise ValueError("training dataset schema/order mismatch")
    data["signal_time"] = _utc_series(data["signal_time"], "signal_time")
    data["feature_as_of"] = _utc_series(data["feature_as_of"], "feature_as_of")
    data["label_end_time"] = _utc_series(data["label_end_time"], "label_end_time")
    data["snapshot_max_materialized_time"] = _utc_series(
        data["snapshot_max_materialized_time"], "snapshot_max_materialized_time"
    )
    if not data["signal_time"].is_monotonic_increasing:
        raise ValueError("training dataset rows must be ordered by signal_time")
    if data["event_id"].astype(str).duplicated().any():
        raise ValueError("training event_id must be unique")
    if data[["symbol", "signal_time"]].duplicated().any():
        raise ValueError("training symbol/signal_time identities must be unique")
    if set(data["side"].astype(str)) != {"short"}:
        raise ValueError("training rows must all be side=short")
    if set(data["feature_semantics"].astype(str)) != {FEATURE_SEMANTICS}:
        raise ValueError("training feature_semantics mismatch")
    labels = pd.to_numeric(data["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("training labels must be binary")
    data["label"] = labels.astype(int)
    expected_labels = data["outcome"].map(INCLUDED_OUTCOMES)
    if expected_labels.isna().any() or not expected_labels.astype(int).eq(data["label"]).all():
        raise ValueError("training labels must map tp=1 and sl/sl_ambiguous=0")
    features = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("training features must be finite")
    data.loc[:, FEATURE_COLUMNS] = features
    for column in ("net_barrier_maker", "net_barrier_taker", "realized_ret"):
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"training {column} must be finite")
        data[column] = values
    _false_only(data["holdout_read"], "holdout_read")
    _false_only(data["auto_promote"], "auto_promote")
    _true_only(data["future_review_only"], "future_review_only")
    future_counts = pd.to_numeric(data["future_rows_in_features"], errors="coerce")
    if future_counts.isna().any() or not future_counts.eq(0).all():
        raise ValueError("training future_rows_in_features must be zero")
    if not data["feature_as_of"].eq(data["signal_time"]).all():
        raise ValueError("training features must be causal at signal_time")
    if (data["signal_time"] >= INNER_CUTOFF).any() or (data["label_end_time"] >= INNER_CUTOFF).any():
        raise ValueError("training dataset reaches sealed interval")
    if (data["signal_time"] >= HOLDOUT_START).any():
        raise ValueError("training dataset contains holdout rows")

    train_boundary = calibration_start - purge_bars * BAR_DURATION
    train = data[
        (data["signal_time"] >= FIXED_TRAIN_START)
        & (data["signal_time"] < calibration_start)
        & (data["label_end_time"] < train_boundary)
    ].copy().reset_index(drop=True)
    if len(train) < 2:
        raise ValueError("independent Jan training partition is too small")
    if set(train["label"]) != {0, 1}:
        raise ValueError("independent Jan training partition must contain both classes")
    if not ((train["signal_time"] >= FIXED_TRAIN_START) & (train["signal_time"] < FIXED_CALIBRATION_START)).all():
        raise ValueError("training partition must be Jan-only")
    return train, manifest, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha,
    }


def _validate_raw_candidate_window(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: int,
    source: Path,
) -> None:
    for raw in raw_rows:
        event_id = str(raw.get("event_id", "<unknown>"))
        if (raw.get("sealed") is True or raw.get("sealed_read") is True
                or raw.get("holdout_read") is True):
            raise ValueError(f"{source}: {event_id} is sealed/holdout-derived")
        if any(raw.get(marker) is True for marker in ("purge", "purged", "is_purge")):
            raise ValueError(f"{source}: {event_id} is a purge row")
        partition = raw.get("source_partition", raw.get("partition"))
        if partition is not None and str(partition).strip().lower() not in {"inner", "mining"}:
            raise ValueError(f"{source}: {event_id} is not an INNER candidate")
        decision = _utc(raw.get("decision_time"), "decision_time")
        label_end = decision + (int(max_horizon) + 1) * BAR_DURATION
        if decision >= FIXED_TEST_END or label_end > FIXED_TEST_END:
            raise ValueError(
                f"{source}: {event_id} exceeds fixed Mar test source window "
                f"(decision={decision.isoformat()}, label_end={label_end.isoformat()})"
            )


def _bounded_future_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: int,
) -> list[dict[str, Any]]:
    bounded: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = dict(raw)
        path = row.get("future_path")
        if isinstance(path, list):
            row["future_path"] = path[:max_horizon]
        bounded.append(row)
    return bounded


def _validate_candidate_source(
    candidates_path: Path,
    selection_path: Path,
) -> tuple[SearchConfig, list[dict[str, Any]], dict[str, Any]]:
    candidates_path = candidates_path.resolve()
    selection_path = selection_path.resolve()
    if not candidates_path.is_file() or not selection_path.is_file():
        raise ValueError("candidate and selection paths must be readable files")
    selection = _json_object(selection_path)
    if selection.get("schema_version") != 1 or selection.get("sealed_read") is not False:
        raise ValueError("config_selection.json must be an unsealed schema_version=1 artifact")
    prereg_path = selection_path.with_name("prereg.json")
    if not prereg_path.is_file():
        raise ValueError("config_selection.json requires sibling prereg.json")
    prereg = _json_object(prereg_path)
    if prereg.get("schema_version") != 1 or prereg.get("sealed_read") is not False:
        raise ValueError("sibling prereg.json must be an unsealed schema_version=1 artifact")
    if Path(str(prereg.get("candidate_source", ""))).resolve() != candidates_path:
        raise ValueError("sibling prereg.json candidate_source must match --candidates")
    candidate_sha = _sha256(candidates_path)
    if prereg.get("candidate_source_sha256") != candidate_sha:
        raise ValueError("sibling prereg.json candidate_source_sha256 mismatch")
    try:
        grid = [SearchConfig(**item) for item in prereg["grid"]]
        guards = optimizer.Guards(**prereg["guards"])
        fold_doc = prereg["fold_document"]
    except (KeyError, TypeError) as exc:
        raise ValueError("sibling prereg.json must contain optimizer grid/guards/fold_document") from exc
    folds, sealed = optimizer.parse_fold_document(fold_doc)
    discipline = {"experimental_discipline": prereg.get("experimental_discipline")}
    if discipline["experimental_discipline"] == "single_variable":
        for key in ("stage", "variable_under_test", "locked_config"):
            discipline[key] = prereg.get(key)
    expected_prereg = optimizer._jsonable(optimizer._protocol(grid, guards, fold_doc, discipline))
    for key, expected in expected_prereg.items():
        if prereg.get(key) != expected:
            raise ValueError(f"sibling prereg.json does not match optimizer protocol field {key}")
    prereg_sha = _sha256(prereg_path)
    if selection.get("prereg_sha256") != prereg_sha:
        raise ValueError("config_selection.json prereg_sha256 mismatch")

    max_horizon = max(config.horizon for config in grid)
    raw_rows = read_jsonl(candidates_path)
    _validate_raw_candidate_window(raw_rows, max_horizon=max_horizon, source=candidates_path)
    bounded_rows = _bounded_future_rows(raw_rows, max_horizon=max_horizon)
    rows = validate_candidates(
        bounded_rows,
        max_horizon=max_horizon,
        allowed_end=sealed["start"],
    )
    recomputed_selection, _ = optimizer.select_config(rows, folds, grid, guards, discipline)
    expected_selection = optimizer._jsonable({
        "schema_version": 1,
        **recomputed_selection,
        "prereg_sha256": prereg_sha,
        "sealed_read": False,
    })
    if selection != expected_selection:
        raise ValueError("config_selection.json does not match recomputed optimizer selection")
    config = SearchConfig(**selection["selected_config"])

    seen_events: set[str] = set()
    seen_identities: set[tuple[str, pd.Timestamp]] = set()
    for row in rows:
        event_id = str(row["event_id"])
        identity = (str(row["symbol"]), _utc(row["decision_time"], "decision_time"))
        if event_id in seen_events or identity in seen_identities:
            raise ValueError(f"duplicate candidate identity: {event_id} {identity}")
        seen_events.add(event_id)
        seen_identities.add(identity)
        label_end = pd.Timestamp(row["_path_frame"]["open_time"].iloc[config.horizon - 1]) + BAR_DURATION
        if label_end >= HOLDOUT_START:
            raise ValueError(f"{event_id}: label_end reaches holdout")
        row["_label_end"] = label_end
    return config, rows, {
        "candidates_path": str(candidates_path),
        "candidates_sha256": candidate_sha,
        "selection_path": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "prereg_path": str(prereg_path.resolve()),
        "prereg_sha256": prereg_sha,
    }


def _materialize_candidate_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: SearchConfig,
    snapshot_dir: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = gap_deduplicate(rows, config.threshold)
    selected.sort(key=lambda row: (
        _utc(row["decision_time"], "decision_time"), str(row["symbol"]), str(row["event_id"])
    ))
    frames, snapshot_lineage, declarations = _load_frames(
        snapshot_dir.resolve(), manifest_path.resolve(), {str(row["symbol"]) for row in selected}
    )
    manifest_sha = _sha256(manifest_path.resolve())
    output: list[dict[str, Any]] = []
    outcome_counts: dict[str, int] = {}
    for row in selected:
        outcome = _resolve(row, config)
        canonical = str(outcome["outcome"]).lower()
        outcome_counts[canonical] = outcome_counts.get(canonical, 0) + 1
        if canonical not in {"tp", "sl", "sl_ambiguous", "timeout"}:
            raise ValueError(f"unsupported canonical outcome: {canonical}")
        decision = _utc(row["decision_time"], "decision_time")
        label_end = _utc(row["_label_end"], "label_end")
        symbol = str(row["symbol"])
        features = _feature_vector(frames[symbol], symbol, decision)
        output.append({
            "event_id": str(row["event_id"]),
            "symbol": symbol,
            "signal_time": decision,
            "side": "short",
            "label": INCLUDED_OUTCOMES.get(canonical),
            "gross_ret": float(outcome["gross_ret"]),
            "net_barrier_maker": float(outcome["net_maker"]),
            "net_barrier_taker": float(outcome["net_taker"]),
            "outcome": canonical,
            "feature_semantics": FEATURE_SEMANTICS,
            **features,
            "feature_as_of": decision,
            "label_end_time": label_end,
            "selected_config_key": config.key,
            "snapshot_manifest_sha256": manifest_sha,
            **snapshot_lineage[symbol],
            "holdout_read": False,
            "future_rows_in_features": 0,
            "future_review_only": True,
            "auto_promote": False,
        })
    frame = pd.DataFrame(output)
    if frame.empty:
        raise ValueError("selected L1 candidate event set is empty")
    frame = frame.sort_values(["signal_time", "symbol", "event_id"], kind="mergesort").reset_index(drop=True)
    return frame, {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "files": declarations,
        "outcomes": outcome_counts,
        "selected_events": len(frame),
    }


def _train_model(train: pd.DataFrame):
    import lightgbm as lgb

    dtrain = lgb.Dataset(train[FEATURE_COLUMNS], label=train["label"], free_raw_data=False)
    return lgb.train(dict(LGB_PARAMS), dtrain, num_boost_round=300)


def _best_iteration(model: Any) -> int | None:
    value = int(getattr(model, "best_iteration", 0) or 0)
    if value > 0:
        return value
    current = getattr(model, "current_iteration", None)
    if callable(current):
        value = int(current())
        if value > 0:
            return value
    return None


def _predict(model: Any, frame: pd.DataFrame) -> np.ndarray:
    kwargs: dict[str, Any] = {}
    iteration = _best_iteration(model)
    if iteration is not None:
        kwargs["num_iteration"] = iteration
    scores = np.asarray(model.predict(frame[FEATURE_COLUMNS], **kwargs), dtype=float)
    if scores.shape != (len(frame),) or not np.isfinite(scores).all():
        raise ValueError("model produced invalid probability scores")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("model scores must be probabilities in [0, 1]")
    return scores


def _partition_events(
    events: pd.DataFrame,
    *,
    calibration_start: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    purge_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    calibration_label_boundary = test_start - purge_bars * BAR_DURATION
    before = len(events)
    calibration = events[
        (events["signal_time"] >= calibration_start)
        & (events["signal_time"] < test_start)
        & (events["label_end_time"] < calibration_label_boundary)
    ].copy()
    test = events[
        (events["signal_time"] >= test_start)
        & (events["signal_time"] < test_end)
        & (events["label_end_time"] <= test_end)
    ].copy()
    purged = before - len(calibration) - len(test) - int((events["signal_time"] < calibration_start).sum())
    return calibration.reset_index(drop=True), test.reset_index(drop=True), max(0, int(purged))


def _profit_factor(values: Sequence[float]) -> float | None:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    if losses > 0:
        return gains / losses
    return None


def _max_drawdown(values: Sequence[float]) -> float:
    cumulative = peak = drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


def _density(frame: pd.DataFrame, selected: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    duration_days = (end - start).total_seconds() / 86400.0
    symbols = {str(symbol) for symbol in frame["symbol"].tolist() if str(symbol)}
    exposure = len(symbols) * duration_days
    return None if exposure <= 0 else len(selected) / exposure


def _economic_metrics(
    frame: pd.DataFrame,
    *,
    threshold: float | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_trades: int | None = None,
    max_density_per_symbol_day: float | None = None,
) -> dict[str, Any]:
    selected = frame if threshold is None else frame[frame["l2_score"] >= threshold]
    selected = selected.sort_values(["signal_time", "symbol", "event_id"], kind="mergesort").reset_index(drop=True)
    maker = selected["net_barrier_maker"].to_numpy(dtype=float).tolist()
    taker = selected["net_barrier_taker"].to_numpy(dtype=float).tolist()
    density = _density(frame, selected, start, end)
    outcomes = {str(key): int(value) for key, value in selected["outcome"].value_counts().sort_index().items()}
    wins = sum(value > 0 for value in maker)
    eligible = True
    violations: list[str] = []
    mean_maker = None if not maker else float(sum(maker) / len(maker))
    if min_trades is not None and len(maker) < min_trades:
        eligible = False
        violations.append("min_trades")
    if max_density_per_symbol_day is not None and density is not None and density > max_density_per_symbol_day:
        eligible = False
        violations.append("max_density_per_symbol_day")
    if mean_maker is not None and mean_maker <= 0:
        eligible = False
        violations.append("mean_net_maker_positive")
    return {
        "threshold": threshold,
        "trades": len(maker),
        "density_per_symbol_day": density,
        "mean_net_maker": mean_maker,
        "total_net_maker": float(sum(maker)),
        "mean_net_taker": None if not taker else float(sum(taker) / len(taker)),
        "total_net_taker": float(sum(taker)),
        "outcomes": outcomes,
        "win_rate_maker": None if not maker else wins / len(maker),
        "profit_factor_maker": _profit_factor(maker),
        "max_drawdown_maker": _max_drawdown(maker),
        "eligible": eligible if min_trades is not None else None,
        "guard_violations": violations,
    }


def _select_threshold(calibration_results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [item for item in calibration_results if item.get("eligible") is True]
    if not eligible:
        raise ValueError("no L2 threshold passes calibration guards")
    eligible.sort(key=lambda item: (
        -float(item["mean_net_maker"]),
        -(float(item["profit_factor_maker"]) if item["profit_factor_maker"] is not None else -math.inf),
        float(item["density_per_symbol_day"] if item["density_per_symbol_day"] is not None else math.inf),
        abs(float(item["threshold"]) - 0.5),
        float(item["threshold"]),
    ))
    return eligible[0]


def _feature_importance(model: Any) -> list[dict[str, Any]]:
    try:
        gain = np.asarray(model.feature_importance(importance_type="gain"), dtype=float)
        split = np.asarray(model.feature_importance(importance_type="split"), dtype=float)
    except Exception:
        gain = np.zeros(len(FEATURE_COLUMNS), dtype=float)
        split = np.zeros(len(FEATURE_COLUMNS), dtype=float)
    if gain.shape != (len(FEATURE_COLUMNS),) or split.shape != (len(FEATURE_COLUMNS),):
        raise ValueError("model feature importance shape mismatch")
    rows = [
        {"feature": name, "gain": float(g), "split": float(s)}
        for name, g, s in zip(FEATURE_COLUMNS, gain, split)
    ]
    rows.sort(key=lambda row: (-float(row["gain"]), str(row["feature"])))
    return rows


def evaluate_l2_walkforward_gate(
    *,
    training_dataset_dir: Path,
    training_manifest_sha256: str,
    candidates: Path,
    selection: Path,
    snapshot_dir: Path,
    manifest: Path,
    experiment_store: Path,
    out_dir: Path,
    calibration_start: Any = DEFAULT_CALIBRATION_START,
    test_start: Any = DEFAULT_TEST_START,
    test_end: Any = DEFAULT_TEST_END,
    purge_bars: int = DEFAULT_PURGE_BARS,
    l2_thresholds: Sequence[float],
    max_density_per_symbol_day: float,
    min_calibration_trades: int,
    min_test_trades: int,
) -> dict[str, Any]:
    """Run the full Jan/Fed/Mar gate and publish atomic artifacts."""
    if type(purge_bars) is not int or purge_bars != DEFAULT_PURGE_BARS:
        raise ValueError(f"--purge-bars is frozen at {DEFAULT_PURGE_BARS}")
    cal_start = _utc(calibration_start, "calibration_start")
    tst_start = _utc(test_start, "test_start")
    tst_end = _utc(test_end, "test_end")
    if (cal_start, tst_start, tst_end) != (FIXED_CALIBRATION_START, FIXED_TEST_START, FIXED_TEST_END):
        raise ValueError(
            "fixed Feb/Mar boundaries required: calibration_start=2026-02-01T00:00:00Z, "
            "test_start=2026-03-01T00:00:00Z, test_end=2026-03-31T07:30:00Z"
        )
    if min_calibration_trades <= 0 or min_test_trades <= 0:
        raise ValueError("minimum trade guards must be positive")
    if max_density_per_symbol_day <= 0:
        raise ValueError("--max-density-per-symbol-day must be positive")

    store = experiment_store.resolve()
    out = out_dir.resolve()
    if not store.is_dir():
        raise ValueError("--experiment-store must be an existing directory")
    try:
        out.relative_to(store)
    except ValueError as exc:
        raise ValueError("--out-dir must be under --experiment-store") from exc
    if out == store:
        raise ValueError("--out-dir must name a fresh child under --experiment-store")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out}")

    selected_config, candidate_rows, candidate_lineage = _validate_candidate_source(candidates, selection)
    train, training_manifest, training_lineage = _load_training_rows(
        training_dataset_dir,
        expected_manifest_sha256=training_manifest_sha256,
        calibration_start=cal_start,
        purge_bars=purge_bars,
    )
    selected_config_from_training = training_manifest.get("selected_config")
    if isinstance(selected_config_from_training, Mapping):
        training_config = SearchConfig(**selected_config_from_training)
        if training_config != selected_config:
            raise ValueError("training dataset selected_config does not match selection input")
    events, snapshot_lineage = _materialize_candidate_events(
        candidate_rows,
        config=selected_config,
        snapshot_dir=snapshot_dir,
        manifest_path=manifest,
    )
    calibration, test, purged_events = _partition_events(
        events,
        calibration_start=cal_start,
        test_start=tst_start,
        test_end=tst_end,
        purge_bars=purge_bars,
    )
    if calibration.empty or test.empty:
        raise ValueError("calibration and test partitions must both be non-empty")
    if len(calibration) < min_calibration_trades:
        raise ValueError("calibration partition has fewer rows than --min-calibration-trades")
    if len(test) < min_test_trades:
        raise ValueError("test partition has fewer rows than --min-test-trades")

    model = _train_model(train)
    train_scores = _predict(model, train)
    calibration["l2_score"] = _predict(model, calibration)
    locked_thresholds = list(l2_thresholds)
    calibration_results = [
        _economic_metrics(
            calibration,
            threshold=threshold,
            start=cal_start,
            end=tst_start,
            min_trades=min_calibration_trades,
            max_density_per_symbol_day=max_density_per_symbol_day,
        )
        for threshold in locked_thresholds
    ]
    selected_threshold = _select_threshold(calibration_results)
    selection_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "selected_threshold": selected_threshold["threshold"],
        "selected_calibration_metrics": selected_threshold,
        "selection_order": [
            "highest_pooled_mean_net_maker",
            "higher_profit_factor",
            "lower_density_per_symbol_day",
            "threshold_closest_0.5",
        ],
        "test_metrics_available_at_selection": False,
        "auto_promote": False,
    }

    # Test scoring and metrics intentionally happen after the threshold payload
    # has been computed.  No test threshold grid is emitted.
    test["l2_score"] = _predict(model, test)
    threshold = float(selection_payload["selected_threshold"])
    test_gate = _economic_metrics(test, threshold=threshold, start=tst_start, end=tst_end)
    baseline_calibration = _economic_metrics(calibration, threshold=None, start=cal_start, end=tst_start)
    baseline_test = _economic_metrics(test, threshold=None, start=tst_start, end=tst_end)
    test_gate["min_test_trades_passed"] = bool(test_gate["trades"] >= min_test_trades)
    test_eval = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "selected_threshold": threshold,
        "l2_gate": test_gate,
        "baseline_all_l1_selected": baseline_test,
        "calibration_baseline_all_l1_selected": baseline_calibration,
        "test_threshold_grid_output": False,
        "holdout_rows_read": 0,
        "sealed_read": False,
        "future_rows_in_features": 0,
        "auto_promote": False,
        "orders": False,
    }

    prereg = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "training_manifest_sha256": training_lineage["manifest_sha256"],
        "candidate_source_sha256": candidate_lineage["candidates_sha256"],
        "selection_sha256": candidate_lineage["selection_sha256"],
        "snapshot_manifest_sha256": snapshot_lineage["manifest_sha256"],
        "selected_config": {
            "threshold": selected_config.threshold,
            "tp_atr": selected_config.tp_atr,
            "sl_atr": selected_config.sl_atr,
            "horizon": selected_config.horizon,
        },
        "selected_config_key": selected_config.key,
        "l2_threshold_grid": locked_thresholds,
        "calibration_start": cal_start.isoformat(),
        "test_start": tst_start.isoformat(),
        "test_end_exclusive": tst_end.isoformat(),
        "purge_bars": purge_bars,
        "max_density_per_symbol_day": max_density_per_symbol_day,
        "min_calibration_trades": min_calibration_trades,
        "min_test_trades": min_test_trades,
        "threshold_policy": "calibration_only",
        "test_inspection_before_selection": False,
        "sealed_read": False,
        "holdout_rows_read": 0,
        "future_rows_in_features": 0,
        "auto_promote": False,
        "orders": False,
    }
    lineage = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "training_input": training_lineage,
        "candidate_input": candidate_lineage,
        "snapshot": snapshot_lineage,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_semantics": FEATURE_SEMANTICS,
        "side": "short",
        "lightgbm_params": dict(LGB_PARAMS),
        "best_iteration": _best_iteration(model),
        "partitions": {
            "train": {
                "rows": len(train),
                "class_counts": _class_counts(train),
                "time_range": [train["signal_time"].min().isoformat(), train["signal_time"].max().isoformat()],
                "max_label_end": train["label_end_time"].max().isoformat(),
            },
            "calibration": {
                "rows": len(calibration),
                "time_range": [calibration["signal_time"].min().isoformat(), calibration["signal_time"].max().isoformat()],
                "max_label_end": calibration["label_end_time"].max().isoformat(),
            },
            "test": {
                "rows": len(test),
                "time_range": [test["signal_time"].min().isoformat(), test["signal_time"].max().isoformat()],
                "max_label_end": test["label_end_time"].max().isoformat(),
            },
            "purged_candidate_events": purged_events,
        },
        "model_train_score_summary": {
            "min": float(train_scores.min()),
            "max": float(train_scores.max()),
            "mean": float(train_scores.mean()),
        },
        "safety": {
            "holdout_rows_read": 0,
            "sealed_read": False,
            "future_rows_in_features": 0,
            "auto_promote": False,
            "orders": False,
            "model_deployable": False,
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(f".{out.name}.tmp-{uuid.uuid4().hex}")
    try:
        temp.mkdir()
        iteration = _best_iteration(model)
        save_kwargs = {} if iteration is None else {"num_iteration": iteration}
        model.save_model(str(temp / "model.txt"), **save_kwargs)
        _write_json(temp / "prereg.json", prereg)
        _write_json(temp / "calibration_results.json", {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "threshold_grid": locked_thresholds,
            "results": calibration_results,
            "baseline_all_l1_selected": baseline_calibration,
        })
        _write_json(temp / "selection.json", selection_payload)
        _write_json(temp / "test_eval.json", test_eval)
        _write_csv(temp / "feature_importance.csv", _feature_importance(model), ["feature", "gain", "split"])
        _write_json(temp / "lineage.json", lineage)
        artifact_sha256 = _sha_map(temp)
        artifact_sha256.pop("lineage.json")
        lineage["artifacts"] = artifact_sha256
        (temp / "lineage.json").unlink()
        _write_json(temp / "lineage.json", lineage)
        os.replace(temp, out)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return {"out_dir": str(out), "selection": selection_payload, "test_eval": test_eval}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dataset-dir", type=Path, required=True)
    parser.add_argument("--training-manifest-sha256", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--experiment-store", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-start", default=DEFAULT_CALIBRATION_START)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", default=DEFAULT_TEST_END)
    parser.add_argument("--purge-bars", type=int, default=DEFAULT_PURGE_BARS)
    parser.add_argument("--l2-thresholds", required=True, help="explicit comma grid, e.g. 0.4,0.5,0.6")
    parser.add_argument("--max-density-per-symbol-day", type=float, default=4.0)
    parser.add_argument("--min-calibration-trades", type=int, default=10)
    parser.add_argument("--min-test-trades", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_l2_walkforward_gate(
        training_dataset_dir=args.training_dataset_dir,
        training_manifest_sha256=args.training_manifest_sha256,
        candidates=args.candidates,
        selection=args.selection,
        snapshot_dir=args.snapshot_dir,
        manifest=args.manifest,
        experiment_store=args.experiment_store,
        out_dir=args.out_dir,
        calibration_start=args.calibration_start,
        test_start=args.test_start,
        test_end=args.test_end,
        purge_bars=args.purge_bars,
        l2_thresholds=_parse_threshold_grid(args.l2_thresholds),
        max_density_per_symbol_day=args.max_density_per_symbol_day,
        min_calibration_trades=args.min_calibration_trades,
        min_test_trades=args.min_test_trades,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
