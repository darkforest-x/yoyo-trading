#!/usr/bin/env python3
"""Retrain one non-deployable SHORT L2 gate after INNER data expansion.

This is the deliberately narrow successor to ``evaluate_l2_walkforward_gate``:
the only experimental variable is the set of native backtester sources.  The
parent gate supplies the frozen L1 threshold/barriers, L1 byte/render contract,
LightGBM parameters, and L2 threshold grid.  Each new source remains a native
``optimization_inner.jsonl``/``summary.json`` pair and is independently
SHA-checked before sources are combined.

Candidate future paths are consumed only by the canonical SHORT barrier
resolver.  All 28 inputs are rebuilt from a SHA-gated snapshot physically
sliced at the decision bar.  Jan TP/SL rows train the model; timeout is not a
supervised class but remains in Feb calibration and locked-threshold Mar
economics.  April, sealed, and holdout partitions are never evaluated.
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
    FEATURE_SEMANTICS,
    HOLDOUT_START,
    INCLUDED_OUTCOMES,
    INNER_CUTOFF,
    _feature_vector,
    _load_frames,
    _utc,
)
import tools.evaluate_frozen_l2_crosssection as crosssection_gate  # noqa: E402
from tools.evaluate_l2_walkforward_gate import (  # noqa: E402
    DEFAULT_CALIBRATION_START,
    DEFAULT_PURGE_BARS,
    DEFAULT_TEST_END,
    DEFAULT_TEST_START,
    FIXED_CALIBRATION_START,
    FIXED_TEST_END,
    FIXED_TEST_START,
    FIXED_TRAIN_START,
    _best_iteration,
    _economic_metrics,
    _feature_importance,
    _predict,
    _select_threshold,
    _train_model,
)
from tools.optimize_short_barrier_walkforward import (  # noqa: E402
    BAR_DURATION,
    SearchConfig,
    _resolve,
    read_jsonl,
)
from tools.train_l2_feedback_model import DATASET_COLUMNS, LGB_PARAMS, _sha256  # noqa: E402
from yoyo.layers.l2_judgment.features import FEATURE_COLUMNS  # noqa: E402

SCHEMA_VERSION = 1
PROTOCOL = "l2_data_expansion_round_v1"
EXPERIMENTAL_CHANGE = "data_expansion_only"
MAX_DENSITY_PER_SYMBOL_DAY = 4.0
MIN_CALIBRATION_TRADES = 100
OUTPUT_FILES = (
    "model.txt",
    "prereg.json",
    "calibration_results.json",
    "selection.json",
    "test_eval.json",
    "feature_importance.csv",
    "dataset.csv",
    "dataset_manifest.json",
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


def _config_dict(config: SearchConfig) -> dict[str, Any]:
    return {"threshold": config.threshold, "tp_atr": config.tp_atr,
            "sl_atr": config.sl_atr, "horizon": config.horizon}


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty:
        return {}
    return {str(key): int(value) for key, value in
            frame[column].value_counts(dropna=False).sort_index().items()}


def _normalize_thresholds(values: Sequence[float], field: str) -> list[float]:
    output: list[float] = []
    for raw in values:
        value = float(raw)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{field} must contain finite probabilities in [0, 1]")
        output.append(value)
    if not output or output != sorted(set(output)):
        raise ValueError(f"{field} must be a non-empty sorted unique grid")
    return output


def _load_walkforward_parent_gate(
    gate_dir: Path, expected_lineage_sha256: str,
) -> tuple[Any, SearchConfig, float, dict[str, Any]]:
    """Resolve the cross-section module's walk-forward loader at call time.

    The cross-section evaluator grew a multi-protocol loader after this tool
    was introduced.  Importing one private helper by name made expansion fail
    during that refactor window.  Prefer the explicit walk-forward entry and
    retain the cross-section module's compatibility wrapper as a fallback.
    """
    for name in ("_load_walkforward_gate", "_load_frozen_gate"):
        loader = getattr(crosssection_gate, name, None)
        if callable(loader):
            return loader(gate_dir, expected_lineage_sha256)
    raise RuntimeError("cross-section module has no walk-forward parent gate loader")


def _load_supported_parent_gate(
    gate_dir: Path, expected_lineage_sha256: str,
) -> tuple[Any, SearchConfig, float, dict[str, Any]]:
    """Load either strict parent protocol through the recursive verifier."""
    loader = getattr(crosssection_gate, "_load_supported_frozen_gate", None)
    if not callable(loader):
        raise RuntimeError("cross-section module has no multi-protocol frozen gate loader")
    return loader(gate_dir, expected_lineage_sha256)


def _validated_parent_candidate_input(
    raw_lineage: Mapping[str, Any], gate_prereg: Mapping[str, Any],
) -> dict[str, str]:
    """Preserve the optimizer lineage omitted by the compact frozen-gate view.

    ``_load_frozen_gate`` has already replayed the optimizer and checked this
    physical chain.  This second pass binds the exact fields copied into the
    expansion dataset, so a future compact return-shape change cannot silently
    replace them with unverified values.
    """
    raw = raw_lineage.get("candidate_input")
    if not isinstance(raw, Mapping):
        raise ValueError("parent gate lineage candidate_input is missing")
    output: dict[str, str] = {}
    for path_field, sha_field, label in (
        ("candidates_path", "candidates_sha256", "parent candidate source"),
        ("selection_path", "selection_sha256", "parent optimizer selection"),
        ("prereg_path", "prereg_sha256", "parent optimizer prereg"),
    ):
        value = raw.get(path_field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} path is missing")
        path = Path(value).resolve()
        if not path.is_file():
            raise ValueError(f"{label} physical path is not readable")
        digest = crosssection_gate._checked_sha(path, raw.get(sha_field), f"{label} SHA256")
        output[path_field], output[sha_field] = str(path), digest

    if gate_prereg.get("candidate_source_sha256") != output["candidates_sha256"]:
        raise ValueError("parent prereg/candidate_input candidate SHA256 mismatch")
    if gate_prereg.get("selection_sha256") != output["selection_sha256"]:
        raise ValueError("parent prereg/candidate_input selection SHA256 mismatch")

    optimizer_selection = crosssection_gate._json(
        Path(output["selection_path"]), "parent optimizer selection"
    )
    if optimizer_selection.get("prereg_sha256") != output["prereg_sha256"]:
        raise ValueError("parent optimizer selection/prereg SHA256 mismatch")
    optimizer_prereg = crosssection_gate._json(
        Path(output["prereg_path"]), "parent optimizer prereg"
    )
    if optimizer_prereg.get("candidate_source_sha256") != output["candidates_sha256"]:
        raise ValueError("parent optimizer prereg/candidate SHA256 mismatch")
    declared = crosssection_gate._resolve_declared_path(
        optimizer_prereg.get("candidate_source"),
        parent=Path(output["prereg_path"]).parent,
        field="parent optimizer prereg candidate_source",
    )
    if declared != Path(output["candidates_path"]):
        raise ValueError("parent optimizer prereg candidate_source path mismatch")
    return output


def _inherited_candidate_input(lineage: Mapping[str, Any]) -> dict[str, str]:
    """Recover the root optimizer lineage from an already validated parent chain."""
    current: Mapping[str, Any] = lineage
    visited: set[str] = set()
    while True:
        directory = str(current.get("directory", ""))
        if directory in visited:
            raise ValueError("validated parent lineage contains a cycle")
        visited.add(directory)
        candidate = current.get("candidate_input")
        if isinstance(candidate, Mapping):
            required = {
                "candidates_path", "candidates_sha256", "selection_path",
                "selection_sha256", "prereg_path", "prereg_sha256",
            }
            if not required <= set(candidate):
                raise ValueError("validated parent candidate_input is incomplete")
            return {field: str(candidate[field]) for field in sorted(required)}
        inherited = current.get("parent_gate")
        if not isinstance(inherited, Mapping):
            raise ValueError("expansion parent lacks inherited optimizer lineage")
        current = inherited


def _load_parent(
    parent_gate_dir: Path, parent_lineage_sha256: str,
) -> tuple[SearchConfig, list[float], dict[str, Any]]:
    """Validate either complete parent chain without weakening walk-forward gates."""
    _unused_model, config, _unused_threshold, lineage = _load_supported_parent_gate(
        parent_gate_dir, parent_lineage_sha256
    )
    gate = parent_gate_dir.resolve()
    raw_lineage = crosssection_gate._json(gate / "lineage.json", "parent gate lineage")
    prereg = crosssection_gate._json(gate / "prereg.json", "parent gate prereg")
    if raw_lineage.get("lightgbm_params") != dict(LGB_PARAMS):
        raise ValueError("parent gate LightGBM parameters differ from exact LGB_PARAMS")
    parent_grid = prereg.get("l2_threshold_grid")
    if not isinstance(parent_grid, list):
        raise ValueError("parent gate prereg lacks explicit l2_threshold_grid")
    grid = _normalize_thresholds(parent_grid, "parent l2_threshold_grid")
    protocol = raw_lineage.get("protocol")
    if protocol == crosssection_gate.FROZEN_GATE_PROTOCOL:
        candidate_input = _validated_parent_candidate_input(raw_lineage, prereg)
    elif protocol == PROTOCOL:
        if (raw_lineage.get("experimental_change") != EXPERIMENTAL_CHANGE
                or prereg.get("experimental_change") != EXPERIMENTAL_CHANGE):
            raise ValueError("expansion parent must prove data_expansion_only")
        candidate_input = _inherited_candidate_input(lineage)
        inherited = lineage.get("parent_gate")
        if not isinstance(inherited, Mapping):
            raise ValueError("expansion parent validated parent_gate chain is missing")
        inherited_dir = Path(str(inherited.get("directory", ""))).resolve()
        inherited_prereg = crosssection_gate._json(
            inherited_dir / "prereg.json", "validated inherited parent prereg"
        )
        inherited_grid = inherited_prereg.get("l2_threshold_grid")
        if not isinstance(inherited_grid, list) or _normalize_thresholds(
            inherited_grid, "inherited parent l2_threshold_grid"
        ) != grid:
            raise ValueError("data expansion parent changed the frozen L2 threshold grid")
    else:
        raise ValueError(f"unsupported parent gate protocol: {protocol!r}")
    safety = raw_lineage.get("safety")
    if not isinstance(safety, Mapping) or safety.get("model_deployable") is not False:
        raise ValueError("parent gate model must be explicitly non-deployable")
    if (safety.get("holdout_rows_read") != 0 or safety.get("sealed_read") is not False
            or safety.get("future_rows_in_features") != 0
            or safety.get("auto_promote") is not False or safety.get("orders") is not False):
        raise ValueError("parent gate safety contract is incomplete")
    if protocol == PROTOCOL and (
        safety.get("april_rows_used") != 0 or safety.get("parent_model_deployable") is not False
    ):
        raise ValueError("expansion parent may not use April or a deployable parent")
    frozen_seen = lineage.get("frozen_seen_symbols")
    if (not isinstance(frozen_seen, list) or not frozen_seen
            or any(not isinstance(symbol, str) or not symbol for symbol in frozen_seen)
            or len(frozen_seen) != len(set(frozen_seen))):
        raise ValueError("parent frozen_seen_symbols lineage is missing or invalid")
    return config, grid, {**lineage, "candidate_input": candidate_input,
                          "lightgbm_params": dict(LGB_PARAMS),
                          "l2_threshold_grid": grid,
                          "frozen_seen_symbols": sorted(frozen_seen)}


def _source_pairs(
    source_pairs: Sequence[tuple[Path, Path]] | None,
    source_candidates: Sequence[Path] | None,
    source_summaries: Sequence[Path] | None,
) -> list[tuple[Path, Path]]:
    pairs = list(source_pairs or [])
    candidates, summaries = list(source_candidates or []), list(source_summaries or [])
    if candidates or summaries:
        if len(candidates) != len(summaries):
            raise ValueError("--source-candidates and --source-summary counts must match")
        pairs.extend(zip(candidates, summaries))
    if not pairs:
        raise ValueError("at least one native INNER source pair is required")
    normalized = [(Path(candidate).resolve(), Path(summary).resolve())
                  for candidate, summary in pairs]
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate source pair")
    return normalized


def _cumulative_source_pairs(
    parent: Mapping[str, Any], requested: Sequence[tuple[Path, Path]],
) -> tuple[list[tuple[Path, Path]], int]:
    """Prepend every native source when expanding an expansion parent.

    A data-expansion child is cumulative.  Replacing the parent's source pool
    with only the newly supplied CLI pairs would make the experimental change
    a data substitution rather than a coverage expansion.
    """
    protocol = parent.get("gate_protocol")
    if protocol == crosssection_gate.FROZEN_GATE_PROTOCOL:
        return list(requested), 0
    if protocol != PROTOCOL:
        raise ValueError(f"unsupported validated parent protocol: {protocol!r}")
    raw_sources = parent.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("expansion parent has no inherited native source lineage")
    inherited: list[tuple[Path, Path]] = []
    inherited_symbols: set[str] = set()
    for index, source in enumerate(raw_sources):
        if not isinstance(source, Mapping) or source.get("source_index") != index:
            raise ValueError("expansion parent source lineage order/index is invalid")
        candidate = Path(str(source.get("path", ""))).resolve()
        summary = Path(str(source.get("summary_path", ""))).resolve()
        if not candidate.is_file() or not summary.is_file():
            raise ValueError("expansion parent inherited source path is unreadable")
        crosssection_gate._checked_sha(
            candidate, source.get("sha256"), f"inherited source {index} candidate SHA256"
        )
        crosssection_gate._checked_sha(
            summary, source.get("summary_sha256"), f"inherited source {index} summary SHA256"
        )
        sampled = source.get("sampled_symbols")
        if (not isinstance(sampled, list) or not sampled
                or any(not isinstance(symbol, str) or not symbol for symbol in sampled)):
            raise ValueError("expansion parent inherited source symbols are invalid")
        overlap = inherited_symbols & set(sampled)
        if overlap:
            raise ValueError(f"expansion parent inherited source symbols overlap: {sorted(overlap)}")
        inherited_symbols.update(sampled)
        inherited.append((candidate, summary))
    if len(inherited) != len(set(inherited)):
        raise ValueError("expansion parent inherited source pairs contain duplicates")
    declared_symbols = parent.get("source_symbols")
    if not isinstance(declared_symbols, list) or inherited_symbols != set(declared_symbols):
        raise ValueError("expansion parent inherited source set is incomplete")
    repeated = set(inherited) & set(requested)
    if repeated:
        raise ValueError("CLI source duplicates an inherited expansion source pair")
    return inherited + list(requested), len(inherited)


def _validate_round_snapshot_manifest(path: Path) -> None:
    """Reject a snapshot that could make the process read April/sealed bytes."""
    raw = crosssection_gate._json(path, "expanded snapshot manifest")
    if (raw.get("sealed") is True or raw.get("sealed_read") is True
            or raw.get("holdout_read") is True
            or any(raw.get(field) is True for field in ("purge", "purged", "is_purge"))):
        raise ValueError("expanded snapshot manifest is sealed/purge/holdout-derived")
    files = raw.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("expanded snapshot manifest files declaration is missing")
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("expanded snapshot manifest contains an invalid file declaration")
        maximum = _utc(item.get("max_materialized_time"), "snapshot max_materialized_time")
        if maximum >= FIXED_TEST_END:
            raise ValueError("expanded snapshot may not materialize April or post-test bars")


def _load_sources(
    pairs: Sequence[tuple[Path, Path]], *, config: SearchConfig,
    start: pd.Timestamp, end: pd.Timestamp, frozen_l1: Mapping[str, Any],
    snapshot_dir: Path, manifest: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    combined: list[dict[str, Any]] = []
    lineages: list[dict[str, Any]] = []
    all_symbols: set[str] = set()
    seen_events: set[str] = set()
    seen_identities: set[tuple[str, pd.Timestamp]] = set()
    for index, (candidates, summary) in enumerate(pairs):
        summary_sha = _sha256(summary) if summary.is_file() else ""
        rows, lineage, candidate_symbols = crosssection_gate._load_independent_candidates(
            candidates, summary, summary_sha, config=config, start=start, end=end
        )
        sampled = set(lineage["sampled_symbols"])
        overlap = all_symbols & sampled
        if overlap:
            raise ValueError(f"native source symbols physically overlap: {sorted(overlap)}")
        all_symbols.update(sampled)
        for field, expected in frozen_l1.items():
            if lineage.get(field) != expected:
                raise ValueError(f"source {index} L1 {field} differs from parent frozen lineage")

        summary_raw = crosssection_gate._json(summary, f"source {index} summary")
        scope = summary_raw.get("outcome_scope")
        if not isinstance(scope, Mapping):
            raise ValueError(f"source {index} outcome_scope is missing")
        if (_utc(scope.get("decision_before"), "outcome_scope.decision_before") != end
                or _utc(scope.get("label_end_lte"), "outcome_scope.label_end_lte") != end):
            raise ValueError(
                f"source {index} outcome_scope must end exactly at the fixed Mar boundary"
            )
        declared_manifest = crosssection_gate._resolve_declared_path(
            summary_raw.get("manifest"), parent=summary.parent, field=f"source {index} summary.manifest"
        )
        declared_snapshot = crosssection_gate._resolve_declared_path(
            summary_raw.get("snapshot_dir"), parent=summary.parent,
            field=f"source {index} summary.snapshot_dir",
        )
        if declared_manifest != manifest.resolve() or declared_snapshot != snapshot_dir.resolve():
            raise ValueError(f"source {index} snapshot/manifest physical lineage mismatch")

        # Identity checks cover the complete physical export, not merely rows
        # surviving the frozen threshold and requested date crop.
        for raw in read_jsonl(candidates):
            event = str(raw.get("event_id", "")).strip()
            if not event:
                raise ValueError("candidate event_id must be non-empty")
            identity = (str(raw.get("symbol", "")), _utc(raw.get("decision_time"), "decision_time"))
            if identity[1] >= end:
                raise ValueError("native source contains post-test/April candidate rows")
            for bar in raw.get("future_path", []):
                if _utc(bar.get("time", bar.get("open_time")), "future_path time") >= end:
                    raise ValueError("native source future_path reaches post-test/April data")
            if event in seen_events:
                raise ValueError(f"duplicate global candidate event_id: {event}")
            if identity in seen_identities:
                raise ValueError(f"duplicate global candidate identity: {identity}")
            seen_events.add(event)
            seen_identities.add(identity)
        for row in rows:
            copied = dict(row)
            copied["_source_index"] = index
            combined.append(copied)
        lineages.append({"source_index": index, **lineage})
    return combined, lineages, all_symbols


def _split_name(signal: pd.Timestamp, label_end: pd.Timestamp) -> str:
    train_boundary = FIXED_CALIBRATION_START - DEFAULT_PURGE_BARS * BAR_DURATION
    calibration_boundary = FIXED_TEST_START - DEFAULT_PURGE_BARS * BAR_DURATION
    if FIXED_TRAIN_START <= signal < FIXED_CALIBRATION_START and label_end < train_boundary:
        return "train"
    if FIXED_CALIBRATION_START <= signal < FIXED_TEST_START and label_end < calibration_boundary:
        return "calibration"
    if FIXED_TEST_START <= signal < FIXED_TEST_END and label_end <= FIXED_TEST_END:
        return "test"
    return "purged"


def _materialize_events(
    rows: Sequence[Mapping[str, Any]], *, config: SearchConfig,
    source_lineages: Sequence[Mapping[str, Any]], snapshot_dir: Path, manifest: Path,
    parent_lineage: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = sorted(rows, key=lambda row: (
        _utc(row["decision_time"], "decision_time"), str(row["symbol"]), str(row["event_id"])
    ))
    if not selected:
        raise ValueError("frozen L1 threshold/gap selected no expanded events")
    symbols = {str(row["symbol"]) for row in selected}
    frames, snapshot_files, declarations = _load_frames(
        snapshot_dir.resolve(), manifest.resolve(), symbols
    )
    snapshot_sha = _sha256(manifest.resolve())
    candidate_input = parent_lineage["candidate_input"]
    output: list[dict[str, Any]] = []
    for row in selected:
        outcome = _resolve(row, config)
        canonical = str(outcome["outcome"]).lower()
        if canonical not in {"tp", "sl", "sl_ambiguous", "timeout"}:
            raise ValueError(f"unsupported canonical SHORT outcome: {canonical}")
        signal = _utc(row["decision_time"], "decision_time")
        label_end = _utc(row["_label_end"], "label_end_time")
        if signal >= FIXED_TEST_END or label_end > FIXED_TEST_END:
            raise ValueError("April or post-test event reached materialization")
        symbol = str(row["symbol"])
        features = _feature_vector(frames[symbol], symbol, signal)
        source = source_lineages[int(row["_source_index"])]
        snapshot = snapshot_files[symbol]
        output.append({
            "event_id": str(row["event_id"]), "symbol": symbol,
            "signal_time": signal, "side": "short",
            "split": _split_name(signal, label_end),
            "label": INCLUDED_OUTCOMES.get(canonical),
            "realized_ret": float(outcome["gross_ret"]),
            "net_barrier_maker": float(outcome["net_maker"]),
            "net_barrier_taker": float(outcome["net_taker"]),
            "outcome": canonical, "feature_semantics": FEATURE_SEMANTICS,
            **features, "feature_as_of": signal, "label_end_time": label_end,
            "selected_config_key": config.key,
            "candidate_source_sha256": source["sha256"],
            "selected_config_sha256": candidate_input["selection_sha256"],
            "prereg_sha256": candidate_input["prereg_sha256"],
            "snapshot_manifest_sha256": snapshot_sha,
            **snapshot, "holdout_read": False, "future_rows_in_features": 0,
            "future_review_only": True, "auto_promote": False,
        })
    frame = pd.DataFrame(output).sort_values(
        ["signal_time", "symbol", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    feature_values = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    if feature_values.shape[1] != 28 or not np.isfinite(feature_values).all():
        raise ValueError("rebuilt 28-feature matrix is invalid")
    if not frame["feature_as_of"].eq(frame["signal_time"]).all():
        raise ValueError("feature causality boundary mismatch")
    return frame, {"directory": str(snapshot_dir.resolve()),
                   "manifest_path": str(manifest.resolve()), "manifest_sha256": snapshot_sha,
                   "files": declarations, "selected_symbol_files": snapshot_files}


def _round_metrics(
    frame: pd.DataFrame, *, threshold: float | None, start: pd.Timestamp,
    end: pd.Timestamp, exposure_symbols: int, guarded: bool = False,
) -> dict[str, Any]:
    """Use every declared source symbol in the density exposure denominator."""
    metrics = _economic_metrics(
        frame, threshold=threshold, start=start, end=end,
        min_trades=MIN_CALIBRATION_TRADES if guarded else None,
        max_density_per_symbol_day=MAX_DENSITY_PER_SYMBOL_DAY if guarded else None,
    )
    days = (end - start).total_seconds() / 86400.0
    if exposure_symbols <= 0 or days <= 0:
        raise ValueError("expanded source symbol exposure must be positive")
    density = metrics["trades"] / (exposure_symbols * days)
    metrics["exposure_symbols"] = exposure_symbols
    metrics["exposure_symbol_days"] = exposure_symbols * days
    metrics["density_per_symbol_day"] = density
    if guarded:
        violations = [item for item in metrics["guard_violations"]
                      if item != "max_density_per_symbol_day"]
        if density > MAX_DENSITY_PER_SYMBOL_DAY:
            violations.append("max_density_per_symbol_day")
        metrics["guard_violations"] = violations
        metrics["eligible"] = not violations
    return metrics


def evaluate_l2_data_expansion_round(
    *, parent_gate_dir: Path, parent_lineage_sha256: str,
    snapshot_dir: Path, manifest: Path, snapshot_manifest_sha256: str,
    experiment_store: Path, out_dir: Path,
    source_pairs: Sequence[tuple[Path, Path]] | None = None,
    source_candidates: Sequence[Path] | None = None,
    source_summaries: Sequence[Path] | None = None,
    l2_thresholds: Sequence[float] | None = None,
    calibration_start: Any = DEFAULT_CALIBRATION_START,
    test_start: Any = DEFAULT_TEST_START, test_end: Any = DEFAULT_TEST_END,
    purge_bars: int = DEFAULT_PURGE_BARS,
) -> dict[str, Any]:
    """Run the frozen-protocol data-expansion round and publish atomically."""
    cal_start, tst_start, tst_end = (_utc(calibration_start, "calibration_start"),
                                     _utc(test_start, "test_start"), _utc(test_end, "test_end"))
    if (cal_start, tst_start, tst_end, purge_bars) != (
        FIXED_CALIBRATION_START, FIXED_TEST_START, FIXED_TEST_END, DEFAULT_PURGE_BARS
    ):
        raise ValueError("fixed Feb/Mar boundaries and 162-bar purge are required")
    store, out = experiment_store.resolve(), out_dir.resolve()
    if not store.is_dir():
        raise ValueError("--experiment-store must be an existing directory")
    try:
        out.relative_to(store)
    except ValueError as exc:
        raise ValueError("--out-dir must be under --experiment-store") from exc
    if out == store:
        raise ValueError("--out-dir must be a fresh child under --experiment-store")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out}")

    requested_pairs = _source_pairs(source_pairs, source_candidates, source_summaries)
    config, parent_grid, parent = _load_parent(parent_gate_dir, parent_lineage_sha256)
    pairs, inherited_source_count = _cumulative_source_pairs(parent, requested_pairs)
    if l2_thresholds is not None and _normalize_thresholds(
        l2_thresholds, "explicit l2_thresholds"
    ) != parent_grid:
        raise ValueError("explicit L2 threshold grid must exactly equal parent prereg grid")
    grid = parent_grid
    manifest_path = manifest.resolve()
    manifest_sha = crosssection_gate._checked_sha(
        manifest_path, snapshot_manifest_sha256, "snapshot manifest SHA256"
    )
    _validate_round_snapshot_manifest(manifest_path)
    rows, sources, sampled_symbols = _load_sources(
        pairs, config=config, start=FIXED_TRAIN_START, end=tst_end,
        frozen_l1=parent["frozen_l1_protocol"], snapshot_dir=snapshot_dir,
        manifest=manifest_path,
    )
    parent_seen_symbols = set(parent["frozen_seen_symbols"])
    inherited_source_symbols = {
        symbol for source in sources[:inherited_source_count]
        for symbol in source["sampled_symbols"]
    }
    new_source_symbols = {
        symbol for source in sources[inherited_source_count:]
        for symbol in source["sampled_symbols"]
    }
    if inherited_source_count:
        if sources[:inherited_source_count] != parent.get("sources"):
            raise ValueError("rebuilt inherited source lineage differs from expansion parent")
        parent_source_symbols = set(parent.get("source_symbols", []))
        if inherited_source_symbols != parent_source_symbols:
            raise ValueError("inherited source symbols do not reproduce expansion parent source set")
        if not parent_source_symbols <= parent_seen_symbols:
            raise ValueError("expansion parent source symbols are missing from frozen_seen_symbols")
    overlap_with_parent = new_source_symbols & parent_seen_symbols
    if overlap_with_parent:
        raise ValueError(
            f"new source symbols overlap parent frozen_seen_symbols: {sorted(overlap_with_parent)}"
        )
    if set(sampled_symbols) != inherited_source_symbols | new_source_symbols:
        raise ValueError("cumulative source symbol accounting mismatch")
    events, snapshot = _materialize_events(
        rows, config=config, source_lineages=sources, snapshot_dir=snapshot_dir,
        manifest=manifest_path, parent_lineage=parent,
    )
    if snapshot["manifest_sha256"] != manifest_sha:
        raise ValueError("snapshot manifest changed during evaluation")

    train = events[(events["split"] == "train") & events["outcome"].isin(INCLUDED_OUTCOMES)].copy()
    calibration = events[events["split"] == "calibration"].copy()
    test = events[events["split"] == "test"].copy()
    if set(train["label"].astype(int)) != {0, 1}:
        raise ValueError("Jan-only supervised train partition must contain both classes")
    if len(calibration) < MIN_CALIBRATION_TRADES:
        raise ValueError(f"Feb calibration requires at least {MIN_CALIBRATION_TRADES} events")
    if test.empty:
        raise ValueError("Mar test partition must be non-empty")
    if (train["outcome"] == "timeout").any():
        raise ValueError("timeout may not enter supervised training")

    model = _train_model(train)
    calibration["l2_score"] = _predict(model, calibration)
    calibration_results = [
        _round_metrics(
            calibration, threshold=value, start=cal_start, end=tst_start,
            exposure_symbols=len(sampled_symbols), guarded=True,
        ) for value in grid
    ]
    winner = _select_threshold(calibration_results)
    selection = {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "selected_threshold": winner["threshold"],
        "selected_calibration_metrics": winner,
        "selection_order": ["highest_pooled_mean_net_maker", "higher_profit_factor",
                            "lower_density_per_symbol_day", "threshold_closest_0.5"],
        "test_metrics_available_at_selection": False,
        "experimental_change": EXPERIMENTAL_CHANGE,
        "auto_promote": False, "model_deployable": False,
    }

    # The test partition is first scored only after the calibration winner is
    # locked above.  No Mar threshold sweep is computed or serialized.
    test["l2_score"] = _predict(model, test)
    locked = float(selection["selected_threshold"])
    test_gate = _round_metrics(
        test, threshold=locked, start=tst_start, end=tst_end,
        exposure_symbols=len(sampled_symbols)
    )
    test_eval = {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "selected_threshold": locked, "l2_gate": test_gate,
        "baseline_all_l1_selected": _round_metrics(
            test, threshold=None, start=tst_start, end=tst_end,
            exposure_symbols=len(sampled_symbols)
        ),
        "test_threshold_grid_output": False, "holdout_rows_read": 0,
        "sealed_read": False, "april_rows_used": 0, "future_rows_in_features": 0,
        "auto_promote": False, "orders": False, "model_deployable": False,
    }

    supervised = events[events["outcome"].isin(INCLUDED_OUTCOMES)].copy()
    supervised["label"] = supervised["label"].astype(int)
    dataset_rows = supervised[DATASET_COLUMNS].to_dict(orient="records")
    dataset_counts = {
        "selected_events_all_outcomes": len(events),
        "supervised_rows": len(supervised),
        "timeouts_excluded_from_supervised_dataset": int((events["outcome"] == "timeout").sum()),
        "outcomes_all_events": _counts(events, "outcome"),
        "outcomes_supervised": _counts(supervised, "outcome"),
        "splits_all_events": _counts(events, "split"),
        "splits_supervised": _counts(supervised, "split"),
        "train_class_counts": {str(label): int((train["label"] == label).sum()) for label in (0, 1)},
    }
    prereg = {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "experimental_change": EXPERIMENTAL_CHANGE,
        "parent_gate_lineage_sha256": parent["lineage_sha256"],
        "source_summary_sha256": [source["summary_sha256"] for source in sources],
        "source_candidate_sha256": [source["sha256"] for source in sources],
        "snapshot_manifest_sha256": manifest_sha,
        "selected_config": _config_dict(config), "selected_config_key": config.key,
        "l2_threshold_grid": grid, "calibration_start": cal_start.isoformat(),
        "test_start": tst_start.isoformat(), "test_end_exclusive": tst_end.isoformat(),
        "purge_bars": purge_bars,
        "calibration_guards": {"max_density_per_symbol_day": MAX_DENSITY_PER_SYMBOL_DAY,
                               "min_trades": MIN_CALIBRATION_TRADES,
                               "mean_net_maker_positive": True},
        "threshold_policy": "feb_calibration_only", "test_inspection_before_selection": False,
        "holdout_rows_read": 0, "sealed_read": False, "april_rows_used": 0,
        "future_rows_in_features": 0, "auto_promote": False, "orders": False,
        "parent_model_deployable": False, "model_deployable": False,
    }
    calibration_payload = {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "threshold_grid": grid, "results": calibration_results,
        "baseline_all_l1_selected": _round_metrics(
            calibration, threshold=None, start=cal_start, end=tst_start,
            exposure_symbols=len(sampled_symbols)
        ),
    }
    lineage = {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "experimental_change": EXPERIMENTAL_CHANGE,
        "parent_gate": parent, "sources": sources, "snapshot": snapshot,
        "source_symbols": sorted(sampled_symbols), "source_symbol_count": len(sampled_symbols),
        "inherited_parent_seen_symbols": sorted(parent_seen_symbols),
        "frozen_seen_symbols": sorted(parent_seen_symbols | set(sampled_symbols)),
        "source_expansion": {
            "mode": "cumulative_parent_sources_plus_new",
            "inherited_source_count": inherited_source_count,
            "new_source_count": len(sources) - inherited_source_count,
            "inherited_source_symbols": sorted(inherited_source_symbols),
            "new_source_symbols": sorted(new_source_symbols),
            "parent_source_set_preserved": inherited_source_count == 0
            or inherited_source_symbols == set(parent.get("source_symbols", [])),
        },
        "feature_columns": list(FEATURE_COLUMNS), "feature_semantics": FEATURE_SEMANTICS,
        "side": "short", "lightgbm_params": dict(LGB_PARAMS),
        "best_iteration": _best_iteration(model), "dataset_counts": dataset_counts,
        "partitions": {
            "train": {"rows": len(train), "outcomes": _counts(train, "outcome")},
            "calibration": {"rows": len(calibration), "outcomes": _counts(calibration, "outcome")},
            "test": {"rows": len(test), "outcomes": _counts(test, "outcome")},
            "purged": {"rows": int((events["split"] == "purged").sum())},
        },
        "safety": {"holdout_rows_read": 0, "sealed_read": False, "april_rows_used": 0,
                   "future_rows_in_features": 0, "auto_promote": False, "orders": False,
                   "parent_model_deployable": False, "model_deployable": False},
        "artifacts": {},
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(f".{out.name}.tmp-{uuid.uuid4().hex}")
    try:
        temp.mkdir()
        iteration = _best_iteration(model)
        model.save_model(str(temp / "model.txt"), **({} if iteration is None else {"num_iteration": iteration}))
        _write_json(temp / "prereg.json", prereg)
        _write_json(temp / "calibration_results.json", calibration_payload)
        _write_json(temp / "selection.json", selection)
        _write_json(temp / "test_eval.json", test_eval)
        _write_csv(temp / "feature_importance.csv", _feature_importance(model),
                   ["feature", "gain", "split"])
        _write_csv(temp / "dataset.csv", dataset_rows, DATASET_COLUMNS)
        dataset_manifest = {
            "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
            "experimental_change": EXPERIMENTAL_CHANGE,
            "dataset": {"path": "dataset.csv", "rows": len(dataset_rows),
                        "sha256": _sha256(temp / "dataset.csv")},
            "selected_config": _config_dict(config), "selected_config_key": config.key,
            "protocol_contract": {"side": "short", "feature_semantics": FEATURE_SEMANTICS,
                                  "feature_columns": list(FEATURE_COLUMNS),
                                  "included_outcomes": INCLUDED_OUTCOMES,
                                  "timeout_supervised": False,
                                  "timeout_in_calibration_test_economics": True},
            "counts": dataset_counts, "inner_cutoff": INNER_CUTOFF.isoformat(),
            "holdout_start": HOLDOUT_START.isoformat(), "holdout_rows_read": 0,
            "sealed_read": False, "april_rows_used": 0, "future_rows_in_features": 0,
            "auto_promote": False, "orders": False, "model_deployable": False,
        }
        _write_json(temp / "dataset_manifest.json", dataset_manifest)
        lineage["artifacts"] = {
            name: _sha256(temp / name) for name in OUTPUT_FILES if name != "lineage.json"
        }
        _write_json(temp / "lineage.json", lineage)
        os.replace(temp, out)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return {"out_dir": str(out), "selection": selection, "test_eval": test_eval,
            "dataset_counts": dataset_counts}


def _parse_grid(raw: str | None) -> list[float] | None:
    return None if raw is None else _normalize_thresholds(
        [float(part) for part in raw.split(",") if part.strip()], "--l2-thresholds"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-gate-dir", type=Path, required=True)
    parser.add_argument("--parent-lineage-sha256", required=True)
    parser.add_argument("--source-candidates", type=Path, action="append", required=True)
    parser.add_argument("--source-summary", type=Path, action="append", required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot-manifest-sha256", required=True)
    parser.add_argument("--experiment-store", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--l2-thresholds", help="optional explicit grid; must exactly equal parent prereg")
    parser.add_argument("--calibration-start", default=DEFAULT_CALIBRATION_START)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--test-end", default=DEFAULT_TEST_END)
    parser.add_argument("--purge-bars", type=int, default=DEFAULT_PURGE_BARS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_l2_data_expansion_round(
        parent_gate_dir=args.parent_gate_dir,
        parent_lineage_sha256=args.parent_lineage_sha256,
        source_candidates=args.source_candidates, source_summaries=args.source_summary,
        snapshot_dir=args.snapshot_dir, manifest=args.manifest,
        snapshot_manifest_sha256=args.snapshot_manifest_sha256,
        experiment_store=args.experiment_store, out_dir=args.out_dir,
        l2_thresholds=_parse_grid(args.l2_thresholds),
        calibration_start=args.calibration_start, test_start=args.test_start,
        test_end=args.test_end, purge_bars=args.purge_bars,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
