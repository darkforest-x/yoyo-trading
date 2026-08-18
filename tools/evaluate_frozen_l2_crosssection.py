#!/usr/bin/env python3
"""Confirm one frozen SHORT L2 gate on physically disjoint INNER symbols.

The model, L1 threshold/barriers, and L2 threshold come only from a completed
``evaluate_l2_walkforward_gate`` directory.  This tool never trains a model or
selects a threshold.  The independent input is the native backtester
``optimization_inner.jsonl`` plus its sibling ``summary.json``; an optimizer
selection on the confirmation symbols is deliberately neither accepted nor
read.

Future paths are label-only.  The frozen 28 inputs are rebuilt from an
SHA-gated OHLCV snapshot after physically slicing each symbol at its decision
bar.  Thus features use OHLCV/indicators through the decision bar only, while
TP/SL/timeout resolution alone may consume later bars.  All date bounds are
half-open for decisions; every included label must end at or before the same
exclusive evaluation end.
"""
from __future__ import annotations

import argparse
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
    _load_json_object,
    _utc,
)
from tools.evaluate_l2_walkforward_gate import (  # noqa: E402
    DEFAULT_PURGE_BARS,
    PROTOCOL as FROZEN_GATE_PROTOCOL,
    _predict,
    _validate_candidate_source as _validate_gate_candidate_source,
)
from tools.optimize_short_barrier_walkforward import (  # noqa: E402
    BAR_DURATION,
    SearchConfig,
    _resolve,
    gap_deduplicate,
    read_jsonl,
    validate_candidates,
)
from tools.train_l2_feedback_model import DATASET_COLUMNS, LGB_PARAMS, _sha256  # noqa: E402
from yoyo.layers.l2_judgment.features import FEATURE_COLUMNS  # noqa: E402

SCHEMA_VERSION = 1
PROTOCOL = "frozen_l2_crosssection_v1"
DEFAULT_START = "2026-01-01T00:00:00Z"
DEFAULT_END = "2026-03-31T07:30:00Z"
OUTPUT_FILES = ("prereg.json", "crosssection_eval.json", "lineage.json")
EXPANSION_GATE_PROTOCOL = "l2_data_expansion_round_v1"
EXPANSION_CHANGE = "data_expansion_only"


def _require_sha(value: Any, field: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field} must be 64 lowercase hex")
    return digest


def _checked_sha(path: Path, expected: Any, field: str) -> str:
    expected_sha = _require_sha(expected, field)
    actual = _sha256(path)
    if actual != expected_sha:
        raise ValueError(f"{field} mismatch")
    return actual


def _json(path: Path, artifact: str) -> dict[str, Any]:
    return _load_json_object(path, artifact)


def _safe_false(raw: Mapping[str, Any], field: str, artifact: str) -> None:
    if raw.get(field) is not False:
        raise ValueError(f"{artifact} must prove {field}=false")


def _resolve_declared_path(raw: Any, *, parent: Path, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(raw)
    return (path if path.is_absolute() else parent / path).resolve()


def _frozen_l1_protocol(candidates_path: Path) -> dict[str, Any]:
    """Recover the L1 bytes/render contract bound to the gate's candidates."""
    if candidates_path.name == "optimization_inner.jsonl":
        source = _json(candidates_path.with_name("summary.json"), "gate candidate summary")
        declaration = source.get("optimization_exports", {}).get("inner")
        protocol: Mapping[str, Any] = source
    else:
        source = _json(candidates_path.with_name("manifest.json"), "gate candidate manifest")
        declaration = source.get("combined")
        raw_protocol = source.get("protocol")
        if not isinstance(raw_protocol, Mapping):
            raise ValueError("gate candidate manifest protocol is missing")
        protocol = raw_protocol
    if not isinstance(declaration, Mapping):
        raise ValueError("gate candidate source declaration is missing")
    if Path(str(declaration.get("file", ""))).name != candidates_path.name:
        raise ValueError("gate candidate source declaration path mismatch")
    _checked_sha(candidates_path, declaration.get("sha256"),
                 "gate candidate source declaration SHA256")
    if int(declaration.get("rows", -1)) != len(read_jsonl(candidates_path)):
        raise ValueError("gate candidate source declaration row count mismatch")
    weights = _require_sha(protocol.get("weights_sha256"), "frozen L1 weights_sha256")
    renderer, transform = protocol.get("renderer"), protocol.get("transform")
    try:
        imgsz = int(protocol.get("imgsz"))
    except (TypeError, ValueError) as exc:
        raise ValueError("frozen L1 imgsz is invalid") from exc
    if not isinstance(renderer, str) or not renderer or not isinstance(transform, str) or not transform:
        raise ValueError("frozen L1 renderer/transform is missing")
    if imgsz <= 0:
        raise ValueError("frozen L1 imgsz is invalid")
    return {"weights_sha256": weights, "renderer": renderer,
            "transform": transform, "imgsz": imgsz}


def _load_walkforward_gate(
    gate_dir: Path, expected_lineage_sha256: str,
) -> tuple[Any, SearchConfig, float, dict[str, Any]]:
    """Validate one original walk-forward gate and its optimizer chain."""
    gate = gate_dir.resolve()
    required = {
        "model.txt", "prereg.json", "selection.json", "lineage.json", "test_eval.json"
    }
    if not gate.is_dir() or any(not (gate / name).is_file() for name in required):
        raise ValueError("frozen gate directory is incomplete")
    lineage_path = gate / "lineage.json"
    lineage_sha = _checked_sha(lineage_path, expected_lineage_sha256, "gate lineage SHA256")
    lineage = _json(lineage_path, "gate lineage")
    prereg = _json(gate / "prereg.json", "gate prereg")
    selection = _json(gate / "selection.json", "gate selection")
    test_eval = _json(gate / "test_eval.json", "gate test_eval")
    for name, raw in (("lineage", lineage), ("prereg", prereg),
                      ("selection", selection), ("test_eval", test_eval)):
        if raw.get("schema_version") != 1 or raw.get("protocol") != FROZEN_GATE_PROTOCOL:
            raise ValueError(f"gate {name} protocol/schema mismatch")

    artifacts = lineage.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("gate lineage.artifacts is missing")
    for name in required - {"lineage.json"}:
        if name not in artifacts:
            raise ValueError(f"gate lineage does not bind {name}")
    for name, digest in artifacts.items():
        relative = Path(str(name))
        if relative.name != str(name) or relative.is_absolute():
            raise ValueError("gate lineage artifact path must be a direct child")
        artifact_path = (gate / relative).resolve()
        if artifact_path.parent != gate or not artifact_path.is_file():
            raise ValueError(f"gate lineage artifact is missing: {name}")
        _checked_sha(artifact_path, digest, f"gate artifact {name} SHA256")

    safety = lineage.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError("gate lineage safety is missing")
    if safety.get("holdout_rows_read") != 0 or safety.get("sealed_read") is not False:
        raise ValueError("frozen gate is sealed/holdout-derived")
    if safety.get("future_rows_in_features") != 0:
        raise ValueError("frozen gate does not prove causal features")
    for field in ("auto_promote", "orders", "model_deployable"):
        _safe_false(safety, field, "gate lineage safety")
    if prereg.get("purge_bars") != DEFAULT_PURGE_BARS:
        raise ValueError("frozen gate purge contract mismatch")
    if prereg.get("holdout_rows_read") != 0 or prereg.get("sealed_read") is not False:
        raise ValueError("gate prereg is sealed/holdout-derived")
    for field in ("auto_promote", "orders", "test_inspection_before_selection"):
        _safe_false(prereg, field, "gate prereg")
    _safe_false(selection, "auto_promote", "gate selection")
    if selection.get("test_metrics_available_at_selection") is not False:
        raise ValueError("gate selection must precede test metrics")
    if test_eval.get("holdout_rows_read") != 0 or test_eval.get("sealed_read") is not False:
        raise ValueError("gate test_eval is sealed/holdout-derived")
    for field in ("auto_promote", "orders", "test_threshold_grid_output"):
        _safe_false(test_eval, field, "gate test_eval")

    try:
        config = SearchConfig(**prereg["selected_config"])
    except (KeyError, TypeError) as exc:
        raise ValueError("gate prereg selected_config is invalid") from exc
    if prereg.get("selected_config_key") != config.key:
        raise ValueError("gate prereg selected_config_key mismatch")
    threshold = float(selection.get("selected_threshold", math.nan))
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("gate selected L2 threshold is invalid")
    if float(test_eval.get("selected_threshold", math.nan)) != threshold:
        raise ValueError("gate selection/test_eval L2 threshold mismatch")

    candidate_input = lineage.get("candidate_input")
    snapshot = lineage.get("snapshot")
    training_input = lineage.get("training_input")
    if not all(isinstance(item, Mapping) for item in (candidate_input, snapshot, training_input)):
        raise ValueError("gate lineage input records are incomplete")
    bindings = (
        (prereg.get("candidate_source_sha256"), candidate_input.get("candidates_sha256"),
         "gate candidate source"),
        (prereg.get("selection_sha256"), candidate_input.get("selection_sha256"),
         "gate optimizer selection"),
        (prereg.get("snapshot_manifest_sha256"), snapshot.get("manifest_sha256"),
         "gate snapshot manifest"),
        (prereg.get("training_manifest_sha256"), training_input.get("manifest_sha256"),
         "gate training manifest"),
    )
    for left, right, label in bindings:
        if _require_sha(left, label) != _require_sha(right, label):
            raise ValueError(f"{label} lineage mismatch")

    for path_field, sha_field, label in (
        ("candidates_path", "candidates_sha256", "gate candidate source"),
        ("selection_path", "selection_sha256", "gate optimizer selection"),
        ("prereg_path", "prereg_sha256", "gate optimizer prereg"),
    ):
        source_path = Path(str(candidate_input.get(path_field, ""))).resolve()
        if not source_path.is_file():
            raise ValueError(f"{label} physical lineage path is not readable")
        _checked_sha(source_path, candidate_input.get(sha_field), f"{label} SHA256")
    optimizer_config, gate_candidate_rows, optimizer_lineage = _validate_gate_candidate_source(
        Path(str(candidate_input["candidates_path"])),
        Path(str(candidate_input["selection_path"])),
    )
    if optimizer_config != config:
        raise ValueError("gate prereg L1 config differs from recomputed optimizer selection")
    for field in ("candidates_sha256", "selection_sha256", "prereg_sha256"):
        if optimizer_lineage[field] != candidate_input.get(field):
            raise ValueError(f"gate optimizer {field} physical lineage mismatch")
    gate_candidate_symbols = {str(row["symbol"]) for row in gate_candidate_rows}
    if not gate_candidate_symbols:
        raise ValueError("gate candidate lineage has no symbols")
    frozen_l1 = _frozen_l1_protocol(
        Path(str(candidate_input["candidates_path"])).resolve()
    )
    gate_snapshot_manifest = Path(str(snapshot.get("manifest_path", ""))).resolve()
    if not gate_snapshot_manifest.is_file():
        raise ValueError("gate snapshot manifest physical lineage path is not readable")
    _checked_sha(
        gate_snapshot_manifest, snapshot.get("manifest_sha256"),
        "gate snapshot manifest SHA256",
    )

    # Follow the model-training bytes as a separate integrity check.  Symbol
    # isolation intentionally uses the gate's original candidate JSONL above,
    # not this dataset and not the snapshot's full symbol inventory.
    manifest_path = Path(str(training_input.get("manifest_path", ""))).resolve()
    dataset_path = Path(str(training_input.get("dataset_path", ""))).resolve()
    if not manifest_path.is_file() or not dataset_path.is_file():
        raise ValueError("gate training lineage paths are not readable")
    _checked_sha(manifest_path, training_input["manifest_sha256"], "training manifest SHA256")
    _checked_sha(dataset_path, training_input["dataset_sha256"], "training dataset SHA256")
    training_manifest = _json(manifest_path, "training manifest")
    dataset_decl = training_manifest.get("dataset")
    if not isinstance(dataset_decl, Mapping):
        raise ValueError("training manifest dataset declaration is missing")
    if _require_sha(dataset_decl.get("sha256"), "training dataset declaration SHA256") != _sha256(dataset_path):
        raise ValueError("training manifest dataset SHA256 mismatch")
    if training_manifest.get("selected_config_key") != config.key:
        raise ValueError("frozen gate L1 config disagrees with training lineage")
    training = pd.read_csv(dataset_path, usecols=["symbol"])
    if int(dataset_decl.get("rows", -1)) != len(training):
        raise ValueError("training dataset row count mismatch")
    dataset_symbols = {str(value) for value in training["symbol"] if str(value)}
    if not dataset_symbols:
        raise ValueError("training lineage has no symbols")

    try:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(gate / "model.txt"))
    except Exception as exc:
        raise ValueError("cannot load frozen LightGBM model") from exc
    names = list(model.feature_name())
    if names != list(FEATURE_COLUMNS):
        raise ValueError("frozen model feature schema/order mismatch")
    return model, config, threshold, {
        "gate_protocol": FROZEN_GATE_PROTOCOL,
        "directory": str(gate), "lineage_sha256": lineage_sha,
        "artifacts": {str(k): str(v) for k, v in artifacts.items()},
        "training_manifest_path": str(manifest_path),
        "training_manifest_sha256": _sha256(manifest_path),
        "training_dataset_path": str(dataset_path),
        "training_dataset_sha256": _sha256(dataset_path),
        "gate_candidate_symbols": sorted(gate_candidate_symbols),
        "training_dataset_symbols": sorted(dataset_symbols),
        "frozen_seen_symbols": sorted(gate_candidate_symbols | dataset_symbols),
        "frozen_l1_protocol": frozen_l1,
        "candidate_input": {str(key): value for key, value in candidate_input.items()},
    }


def _expansion_safety(raw: Mapping[str, Any], artifact: str) -> None:
    if raw.get("holdout_rows_read") != 0 or raw.get("april_rows_used") != 0:
        raise ValueError(f"{artifact} must prove holdout_rows_read=0 and april_rows_used=0")
    if raw.get("future_rows_in_features") != 0:
        raise ValueError(f"{artifact} must prove future_rows_in_features=0")
    for field in ("sealed_read", "auto_promote", "orders", "model_deployable"):
        _safe_false(raw, field, artifact)


def _load_expansion_dataset(
    gate: Path, lineage: Mapping[str, Any], prereg: Mapping[str, Any], config: SearchConfig,
    *, source_candidate_shas: Sequence[str], parent_candidate_input: Mapping[str, Any],
) -> tuple[set[str], dict[str, Any]]:
    manifest_path, dataset_path = gate / "dataset_manifest.json", gate / "dataset.csv"
    manifest = _json(manifest_path, "expansion dataset manifest")
    if (manifest.get("schema_version"), manifest.get("protocol"),
            manifest.get("experimental_change")) != (
                SCHEMA_VERSION, EXPANSION_GATE_PROTOCOL, EXPANSION_CHANGE
            ):
        raise ValueError("expansion dataset manifest protocol/change mismatch")
    _expansion_safety(manifest, "expansion dataset manifest")
    if _utc(manifest.get("inner_cutoff"), "dataset inner_cutoff") != INNER_CUTOFF:
        raise ValueError("expansion dataset inner_cutoff mismatch")
    if _utc(manifest.get("holdout_start"), "dataset holdout_start") != HOLDOUT_START:
        raise ValueError("expansion dataset holdout_start mismatch")
    declaration = manifest.get("dataset")
    if not isinstance(declaration, Mapping) or declaration.get("path") != dataset_path.name:
        raise ValueError("expansion dataset declaration is invalid")
    dataset_sha = _checked_sha(
        dataset_path, declaration.get("sha256"), "expansion dataset declaration SHA256"
    )
    if lineage.get("artifacts", {}).get("dataset.csv") != dataset_sha:
        raise ValueError("expansion lineage/dataset declaration SHA256 mismatch")
    contract = manifest.get("protocol_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("expansion dataset protocol_contract is missing")
    if (contract.get("side") != "short"
            or contract.get("feature_semantics") != FEATURE_SEMANTICS
            or contract.get("feature_columns") != list(FEATURE_COLUMNS)
            or contract.get("included_outcomes") != INCLUDED_OUTCOMES
            or contract.get("timeout_supervised") is not False
            or contract.get("timeout_in_calibration_test_economics") is not True):
        raise ValueError("expansion dataset protocol contract mismatch")
    if manifest.get("selected_config_key") != config.key:
        raise ValueError("expansion dataset selected_config_key mismatch")
    if manifest.get("counts") != lineage.get("dataset_counts"):
        raise ValueError("expansion dataset manifest/lineage counts mismatch")
    try:
        manifest_config = SearchConfig(**manifest["selected_config"])
    except (KeyError, TypeError) as exc:
        raise ValueError("expansion dataset selected_config is invalid") from exc
    if manifest_config != config:
        raise ValueError("expansion dataset selected_config mismatch")

    data = pd.read_csv(dataset_path)
    if list(data.columns) != DATASET_COLUMNS:
        raise ValueError("expansion dataset schema/order mismatch")
    if type(declaration.get("rows")) is not int or declaration["rows"] != len(data) or data.empty:
        raise ValueError("expansion dataset row count mismatch")
    for column in ("signal_time", "feature_as_of", "label_end_time"):
        data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"expansion dataset {column} contains invalid UTC values")
    if (data["signal_time"] >= pd.Timestamp(DEFAULT_END)).any() or (
        data["label_end_time"] > pd.Timestamp(DEFAULT_END)
    ).any():
        raise ValueError("expansion dataset reaches April/post-test data")
    if not data["feature_as_of"].eq(data["signal_time"]).all():
        raise ValueError("expansion dataset feature causality mismatch")
    if set(data["side"].astype(str)) != {"short"}:
        raise ValueError("expansion dataset must be SHORT-only")
    if set(data["feature_semantics"].astype(str)) != {FEATURE_SEMANTICS}:
        raise ValueError("expansion dataset feature semantics mismatch")
    if not set(data["split"].astype(str)) <= {"train", "calibration", "test", "purged"}:
        raise ValueError("expansion dataset split values are invalid")
    labels = pd.to_numeric(data["label"], errors="coerce")
    expected_labels = data["outcome"].map(INCLUDED_OUTCOMES)
    if labels.isna().any() or expected_labels.isna().any() or not labels.eq(expected_labels).all():
        raise ValueError("expansion dataset supervised outcome labels are invalid")
    feature_values = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if feature_values.shape[1] != 28 or not np.isfinite(feature_values).all():
        raise ValueError("expansion dataset 28-feature matrix is invalid")
    if data["holdout_read"].astype(str).str.lower().ne("false").any():
        raise ValueError("expansion dataset contains holdout-read rows")
    if pd.to_numeric(data["future_rows_in_features"], errors="coerce").ne(0).any():
        raise ValueError("expansion dataset contains future feature rows")
    if data["future_review_only"].astype(str).str.lower().ne("true").any():
        raise ValueError("expansion dataset future labels are not review-only")
    if data["auto_promote"].astype(str).str.lower().ne("false").any():
        raise ValueError("expansion dataset contains auto-promote rows")
    if set(data["selected_config_key"].astype(str)) != {config.key}:
        raise ValueError("expansion dataset row config mismatch")
    if data["event_id"].astype(str).duplicated().any() or data[
        ["symbol", "signal_time"]
    ].duplicated().any():
        raise ValueError("expansion dataset contains duplicate event identities")
    if not data.sort_values(
        ["signal_time", "symbol", "event_id"], kind="mergesort"
    ).index.equals(data.index):
        raise ValueError("expansion dataset rows are not deterministically ordered")
    if set(data["candidate_source_sha256"].astype(str)) != set(source_candidate_shas):
        raise ValueError("expansion dataset source candidate SHA lineage mismatch")
    if set(data["selected_config_sha256"].astype(str)) != {
        str(parent_candidate_input.get("selection_sha256"))
    }:
        raise ValueError("expansion dataset parent selection SHA lineage mismatch")
    if set(data["prereg_sha256"].astype(str)) != {
        str(parent_candidate_input.get("prereg_sha256"))
    }:
        raise ValueError("expansion dataset parent prereg SHA lineage mismatch")
    if set(data["snapshot_manifest_sha256"].astype(str)) != {
        str(prereg.get("snapshot_manifest_sha256"))
    }:
        raise ValueError("expansion dataset snapshot lineage mismatch")
    symbols = {str(value) for value in data["symbol"] if str(value)}
    if not symbols:
        raise ValueError("expansion dataset has no symbols")
    return symbols, {"path": str(dataset_path), "sha256": dataset_sha,
                     "manifest_path": str(manifest_path),
                     "manifest_sha256": _sha256(manifest_path), "rows": len(data)}


def _load_expansion_gate(
    gate_dir: Path, expected_lineage_sha256: str, seen: set[Path],
) -> tuple[Any, SearchConfig, float, dict[str, Any]]:
    """Validate one data-expansion gate, all native sources, and its parent."""
    gate = gate_dir.resolve()
    required = {
        "model.txt", "prereg.json", "calibration_results.json", "selection.json",
        "test_eval.json", "feature_importance.csv", "dataset.csv", "dataset_manifest.json",
    }
    lineage_path = gate / "lineage.json"
    lineage_sha = _checked_sha(lineage_path, expected_lineage_sha256, "gate lineage SHA256")
    lineage = _json(lineage_path, "expansion lineage")
    prereg = _json(gate / "prereg.json", "expansion prereg")
    selection = _json(gate / "selection.json", "expansion selection")
    test_eval = _json(gate / "test_eval.json", "expansion test_eval")
    calibration = _json(gate / "calibration_results.json", "expansion calibration results")
    dataset_manifest = _json(gate / "dataset_manifest.json", "expansion dataset manifest")
    for name, raw in (("lineage", lineage), ("prereg", prereg), ("selection", selection),
                      ("test_eval", test_eval), ("calibration_results", calibration),
                      ("dataset_manifest", dataset_manifest)):
        if raw.get("schema_version") != SCHEMA_VERSION or raw.get("protocol") != EXPANSION_GATE_PROTOCOL:
            raise ValueError(f"expansion {name} protocol/schema mismatch")
    for name, raw in (("lineage", lineage), ("prereg", prereg), ("selection", selection),
                      ("dataset_manifest", dataset_manifest)):
        if raw.get("experimental_change") != EXPANSION_CHANGE:
            raise ValueError(f"expansion {name} must prove experimental_change=data_expansion_only")

    artifacts = lineage.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise ValueError("expansion lineage artifacts set is incomplete or unexpected")
    for name, digest in artifacts.items():
        relative = Path(str(name))
        if relative.name != str(name) or relative.is_absolute():
            raise ValueError("expansion artifact path must be a direct child")
        path = (gate / relative).resolve()
        if path.parent != gate or not path.is_file():
            raise ValueError(f"expansion artifact is missing: {name}")
        _checked_sha(path, digest, f"expansion artifact {name} SHA256")

    safety = lineage.get("safety")
    if not isinstance(safety, Mapping):
        raise ValueError("expansion lineage safety is missing")
    _expansion_safety(safety, "expansion lineage safety")
    _expansion_safety(prereg, "expansion prereg")
    _expansion_safety(test_eval, "expansion test_eval")
    for raw, artifact in ((safety, "expansion lineage safety"),
                          (prereg, "expansion prereg")):
        _safe_false(raw, "parent_model_deployable", artifact)
    if prereg.get("purge_bars") != DEFAULT_PURGE_BARS:
        raise ValueError("expansion purge contract mismatch")
    _safe_false(prereg, "test_inspection_before_selection", "expansion prereg")
    for field in ("auto_promote", "model_deployable"):
        _safe_false(selection, field, "expansion selection")
    if selection.get("test_metrics_available_at_selection") is not False:
        raise ValueError("expansion selection must precede test metrics")
    if test_eval.get("test_threshold_grid_output") is not False:
        raise ValueError("expansion test_eval exposes a threshold grid")

    try:
        config = SearchConfig(**prereg["selected_config"])
    except (KeyError, TypeError) as exc:
        raise ValueError("expansion selected_config is invalid") from exc
    if prereg.get("selected_config_key") != config.key:
        raise ValueError("expansion selected_config_key mismatch")
    threshold = float(selection.get("selected_threshold", math.nan))
    grid = prereg.get("l2_threshold_grid")
    try:
        normalized_grid = [float(value) for value in grid] if isinstance(grid, list) else []
    except (TypeError, ValueError) as exc:
        raise ValueError("expansion L2 threshold grid is invalid") from exc
    if (not math.isfinite(threshold) or not 0 <= threshold <= 1
            or not normalized_grid or normalized_grid != sorted(set(normalized_grid))
            or threshold not in normalized_grid):
        raise ValueError("expansion selected L2 threshold/grid is invalid")
    if float(test_eval.get("selected_threshold", math.nan)) != threshold:
        raise ValueError("expansion selection/test_eval threshold mismatch")
    if calibration.get("threshold_grid") != normalized_grid:
        raise ValueError("expansion calibration/prereg threshold grid mismatch")
    calibration_results = calibration.get("results")
    if not isinstance(calibration_results, list):
        raise ValueError("expansion calibration results are missing")
    selected_results = [item for item in calibration_results if isinstance(item, Mapping)
                        and float(item.get("threshold", math.nan)) == threshold]
    if len(selected_results) != 1 or selection.get("selected_calibration_metrics") != selected_results[0]:
        raise ValueError("expansion selection is not bound to calibration winner metrics")
    l2_gate = test_eval.get("l2_gate")
    baseline = test_eval.get("baseline_all_l1_selected")
    if (not isinstance(l2_gate, Mapping) or float(l2_gate.get("threshold", math.nan)) != threshold
            or not isinstance(baseline, Mapping) or baseline.get("threshold") is not None):
        raise ValueError("expansion test_eval threshold contract mismatch")
    if lineage.get("feature_columns") != list(FEATURE_COLUMNS) or lineage.get(
        "feature_semantics"
    ) != FEATURE_SEMANTICS or lineage.get("side") != "short":
        raise ValueError("expansion lineage feature/side contract mismatch")
    if lineage.get("lightgbm_params") != dict(LGB_PARAMS):
        raise ValueError("expansion LightGBM parameters differ from frozen contract")

    parent_record = lineage.get("parent_gate")
    if not isinstance(parent_record, Mapping):
        raise ValueError("expansion parent_gate lineage is missing")
    parent_dir = _resolve_declared_path(
        parent_record.get("directory"), parent=gate, field="expansion parent gate directory"
    )
    parent_sha = _require_sha(parent_record.get("lineage_sha256"), "parent gate lineage SHA256")
    if prereg.get("parent_gate_lineage_sha256") != parent_sha:
        raise ValueError("expansion prereg/parent gate lineage SHA256 mismatch")
    _parent_model, parent_config, _parent_threshold, parent = _load_supported_frozen_gate(
        parent_dir, parent_sha, _seen=seen
    )
    if parent_config != config:
        raise ValueError("expansion changed the frozen parent L1 config")
    for field in ("artifacts", "frozen_l1_protocol", "gate_candidate_symbols",
                  "training_dataset_symbols"):
        if parent_record.get(field) != parent.get(field):
            raise ValueError(f"expansion copied parent {field} does not match validated parent")
    for field in ("training_manifest_path", "training_manifest_sha256",
                  "training_dataset_path", "training_dataset_sha256"):
        if parent_record.get(field) != parent.get(field):
            raise ValueError(f"expansion copied parent {field} lineage mismatch")
    copied_candidate = parent_record.get("candidate_input")
    if not isinstance(copied_candidate, Mapping):
        raise ValueError("expansion copied parent candidate_input is missing")
    for field, value in parent["candidate_input"].items():
        if copied_candidate.get(field) != value:
            raise ValueError(f"expansion copied parent candidate_input.{field} mismatch")

    snapshot = lineage.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("expansion snapshot lineage is missing")
    snapshot_manifest = Path(str(snapshot.get("manifest_path", ""))).resolve()
    snapshot_dir = Path(str(snapshot.get("directory", ""))).resolve()
    snapshot_sha = _checked_sha(
        snapshot_manifest, snapshot.get("manifest_sha256"), "expansion snapshot manifest SHA256"
    )
    if prereg.get("snapshot_manifest_sha256") != snapshot_sha:
        raise ValueError("expansion prereg/snapshot manifest SHA256 mismatch")
    raw_snapshot = _json(snapshot_manifest, "expansion snapshot manifest")
    declarations = raw_snapshot.get("files")
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("expansion snapshot manifest files are missing")
    for item in declarations:
        if not isinstance(item, Mapping) or _utc(
            item.get("max_materialized_time"), "snapshot max_materialized_time"
        ) >= pd.Timestamp(DEFAULT_END):
            raise ValueError("expansion snapshot reaches April/post-test materialization")

    sources = lineage.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("expansion sources lineage is missing")
    source_symbols: set[str] = set()
    source_candidate_shas: list[str] = []
    source_summary_shas: list[str] = []
    seen_events: set[str] = set()
    seen_identities: set[tuple[str, str]] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping) or source.get("source_index") != index:
            raise ValueError("expansion source index/order is invalid")
        candidates = Path(str(source.get("path", ""))).resolve()
        summary = Path(str(source.get("summary_path", ""))).resolve()
        summary_sha = _require_sha(source.get("summary_sha256"), f"source {index} summary SHA256")
        _, validated, symbols = _load_independent_candidates(
            candidates, summary, summary_sha, config=config,
            start=pd.Timestamp(DEFAULT_START), end=pd.Timestamp(DEFAULT_END),
        )
        for field in ("sha256", "rows", "weights_sha256", "renderer", "transform", "imgsz",
                      "optimization_export", "sampled_symbols", "candidate_symbols",
                      "l1_selected_interval_rows", "label_end_cropped"):
            if source.get(field) != validated.get(field):
                raise ValueError(f"expansion source {index} {field} lineage mismatch")
        if source.get("weights_path") != validated.get("weights_path"):
            raise ValueError(f"expansion source {index} weights path mismatch")
        if validated["summary_sha256"] != summary_sha:
            raise ValueError(f"expansion source {index} summary SHA256 mismatch")
        for field, expected in parent["frozen_l1_protocol"].items():
            if validated.get(field) != expected:
                raise ValueError(f"expansion source {index} L1 {field} differs from parent")
        summary_raw = _json(summary, f"expansion source {index} summary")
        scope = summary_raw.get("outcome_scope")
        if (not isinstance(scope, Mapping)
                or _utc(scope.get("decision_before"), "source decision_before") != pd.Timestamp(DEFAULT_END)
                or _utc(scope.get("label_end_lte"), "source label_end_lte") != pd.Timestamp(DEFAULT_END)):
            raise ValueError(f"expansion source {index} outcome scope is not fixed Jan-Mar INNER")
        if (_resolve_declared_path(summary_raw.get("manifest"), parent=summary.parent,
                                   field=f"source {index} manifest") != snapshot_manifest
                or _resolve_declared_path(summary_raw.get("snapshot_dir"), parent=summary.parent,
                                          field=f"source {index} snapshot") != snapshot_dir):
            raise ValueError(f"expansion source {index} snapshot physical lineage mismatch")
        if _sha256(snapshot_manifest) != snapshot_sha:
            raise ValueError("expansion snapshot manifest changed during source validation")
        overlap = source_symbols & symbols
        if overlap:
            raise ValueError(f"expansion source symbols overlap: {sorted(overlap)}")
        source_symbols.update(symbols)
        source_candidate_shas.append(validated["sha256"])
        source_summary_shas.append(validated["summary_sha256"])
        for row in read_jsonl(candidates):
            event = str(row.get("event_id", ""))
            decision = _utc(row.get("decision_time"), "source decision_time")
            identity = (str(row.get("symbol", "")), decision.isoformat())
            if decision >= pd.Timestamp(DEFAULT_END):
                raise ValueError("expansion source contains April/post-test decisions")
            for bar in row.get("future_path", []):
                if _utc(bar.get("time", bar.get("open_time")), "source future_path time") >= pd.Timestamp(
                    DEFAULT_END
                ):
                    raise ValueError("expansion source future_path reaches April/post-test data")
            if event in seen_events or identity in seen_identities:
                raise ValueError("duplicate event identity across expansion sources")
            seen_events.add(event)
            seen_identities.add(identity)
    declared_source_symbols = lineage.get("source_symbols")
    if (declared_source_symbols != sorted(source_symbols)
            or lineage.get("source_symbol_count") != len(source_symbols)):
        raise ValueError("expansion source symbol lineage mismatch")
    if prereg.get("source_candidate_sha256") != source_candidate_shas:
        raise ValueError("expansion prereg source candidate SHA list mismatch")
    if prereg.get("source_summary_sha256") != source_summary_shas:
        raise ValueError("expansion prereg source summary SHA list mismatch")

    dataset_symbols, dataset_lineage = _load_expansion_dataset(
        gate, lineage, prereg, config, source_candidate_shas=source_candidate_shas,
        parent_candidate_input=parent["candidate_input"],
    )
    if dataset_symbols != source_symbols:
        raise ValueError("expansion source_symbols and dataset.csv symbols differ")
    selected_files = snapshot.get("selected_symbol_files")
    if not isinstance(selected_files, Mapping) or set(selected_files) != source_symbols:
        raise ValueError("expansion selected snapshot symbol files mismatch")

    try:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(gate / "model.txt"))
    except Exception as exc:
        raise ValueError("cannot load frozen expansion LightGBM model") from exc
    if list(model.feature_name()) != list(FEATURE_COLUMNS):
        raise ValueError("frozen expansion model feature schema/order mismatch")
    inherited = set(parent["frozen_seen_symbols"])
    if not inherited < source_symbols:
        raise ValueError("data_expansion_only requires a strict superset of parent seen symbols")
    frozen_seen = inherited | source_symbols | dataset_symbols
    return model, config, threshold, {
        "gate_protocol": EXPANSION_GATE_PROTOCOL,
        "directory": str(gate), "lineage_sha256": lineage_sha,
        "artifacts": {str(key): str(value) for key, value in artifacts.items()},
        "training_manifest_path": dataset_lineage["manifest_path"],
        "training_manifest_sha256": dataset_lineage["manifest_sha256"],
        "training_dataset_path": dataset_lineage["path"],
        "training_dataset_sha256": dataset_lineage["sha256"],
        "gate_candidate_symbols": sorted(source_symbols),
        "training_dataset_symbols": sorted(dataset_symbols),
        "source_symbols": sorted(source_symbols),
        "inherited_parent_seen_symbols": sorted(inherited),
        "frozen_seen_symbols": sorted(frozen_seen),
        "frozen_l1_protocol": dict(parent["frozen_l1_protocol"]),
        "parent_gate": parent,
        "sources": [dict(source) for source in sources],
    }


def _load_supported_frozen_gate(
    gate_dir: Path, expected_lineage_sha256: str, *, _seen: set[Path] | None = None,
) -> tuple[Any, SearchConfig, float, dict[str, Any]]:
    """Dispatch to either supported frozen parent protocol, fail-closed."""
    gate = gate_dir.resolve()
    seen = set() if _seen is None else set(_seen)
    if gate in seen:
        raise ValueError("frozen gate parent lineage contains a cycle")
    seen.add(gate)
    lineage_path = gate / "lineage.json"
    _checked_sha(lineage_path, expected_lineage_sha256, "gate lineage SHA256")
    protocol = _json(lineage_path, "gate lineage").get("protocol")
    if protocol == FROZEN_GATE_PROTOCOL:
        return _load_walkforward_gate(gate, expected_lineage_sha256)
    if protocol == EXPANSION_GATE_PROTOCOL:
        return _load_expansion_gate(gate, expected_lineage_sha256, seen)
    raise ValueError(f"unsupported frozen gate protocol: {protocol!r}")


def _load_frozen_gate(
    gate_dir: Path, expected_lineage_sha256: str,
) -> tuple[Any, SearchConfig, float, dict[str, Any]]:
    """Backward-compatible original walk-forward gate loader.

    ``evaluate_l2_data_expansion_round`` imports this private helper to verify
    its walk-forward parent.  Keep that meaning stable; new callers that accept
    either parent protocol must use :func:`_load_supported_frozen_gate`.
    """
    return _load_walkforward_gate(gate_dir, expected_lineage_sha256)


def _load_independent_candidates(
    candidates_path: Path,
    summary_path: Path,
    expected_summary_sha256: str,
    *,
    config: SearchConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    candidates = candidates_path.resolve()
    summary_file = summary_path.resolve()
    if candidates.name != "optimization_inner.jsonl" or candidates.parent != summary_file.parent:
        raise ValueError("candidates must be physical optimization_inner.jsonl with sibling summary.json")
    if summary_file.name != "summary.json" or not candidates.is_file() or not summary_file.is_file():
        raise ValueError("independent candidates/summary files are missing")
    summary_sha = _checked_sha(summary_file, expected_summary_sha256, "candidate summary SHA256")
    summary = _json(summary_file, "candidate sibling summary")
    if summary.get("holdout_read") is not False or summary.get("future_data_in_causal_input") is not False:
        raise ValueError("candidate summary is future/holdout-derived")
    for field in ("orders_placed", "promoted"):
        _safe_false(summary, field, "candidate summary")
    if any(summary.get(field) is True for field in ("sealed", "sealed_read", "purge", "purged", "is_purge")):
        raise ValueError("candidate summary is sealed/purge-derived")
    scope = summary.get("outcome_scope")
    if not isinstance(scope, Mapping) or scope.get("name") != "optimization_inner":
        raise ValueError("candidate summary outcome_scope must be optimization_inner")
    if scope.get("sealed_outcomes_resolved") is not False:
        raise ValueError("candidate summary resolved sealed outcomes")
    decision_before = _utc(scope.get("decision_before"), "outcome_scope.decision_before")
    label_lte = _utc(scope.get("label_end_lte"), "outcome_scope.label_end_lte")
    if decision_before < end or label_lte < end or decision_before >= INNER_CUTOFF or label_lte >= INNER_CUTOFF:
        raise ValueError("candidate summary does not cover the requested INNER interval safely")

    exports = summary.get("optimization_exports")
    inner = exports.get("inner") if isinstance(exports, Mapping) else None
    if not isinstance(inner, Mapping) or inner.get("file") != candidates.name:
        raise ValueError("candidate summary does not declare optimization_inner.jsonl")
    candidate_sha = _checked_sha(candidates, inner.get("sha256"), "candidate export SHA256")
    raw_rows = read_jsonl(candidates)
    if type(inner.get("rows")) is not int or inner["rows"] != len(raw_rows):
        raise ValueError("candidate export row count mismatch")

    weights_sha = _require_sha(summary.get("weights_sha256"), "candidate weights_sha256")
    weights_path = _resolve_declared_path(
        summary.get("weights"), parent=summary_file.parent, field="candidate summary weights"
    )
    if not weights_path.is_file():
        raise ValueError("candidate summary weights path is not readable")
    _checked_sha(weights_path, weights_sha, "candidate weights SHA256")
    if summary.get("transform") != "white_letterbox_full_frame" or summary.get("imgsz") != 960:
        raise ValueError("candidate summary transform/imgsz contract mismatch")
    if not isinstance(summary.get("renderer"), str) or not summary["renderer"]:
        raise ValueError("candidate summary renderer is missing")
    export_protocol = summary.get("optimization_export")
    if not isinstance(export_protocol, Mapping):
        raise ValueError("candidate summary optimization_export is missing")
    try:
        max_horizon = int(export_protocol["max_horizon"])
        thresholds = [float(value) for value in export_protocol["thresholds"]]
        min_score = float(export_protocol["min_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("candidate optimization export protocol is invalid") from exc
    if thresholds != sorted(set(thresholds)) or not thresholds or thresholds[0] != min_score:
        raise ValueError("candidate optimization threshold union is invalid")
    if config.threshold not in thresholds:
        raise ValueError("frozen L1 threshold is absent from candidate export union")
    if max_horizon < config.horizon:
        raise ValueError("candidate export horizon is shorter than frozen horizon")

    sampling = summary.get("symbol_sampling")
    sampled = sampling.get("selected_symbols") if isinstance(sampling, Mapping) else None
    if not isinstance(sampled, list) or not sampled or any(not isinstance(x, str) or not x for x in sampled):
        raise ValueError("candidate summary selected_symbols is invalid")
    if len(sampled) != len(set(sampled)):
        raise ValueError("candidate summary selected_symbols contains duplicates")

    for row in raw_rows:
        event = str(row.get("event_id", "<unknown>"))
        if row.get("holdout_read") is not False or row.get("future_data_in_causal_input") is not False:
            raise ValueError(f"{event}: candidate is future/holdout-derived")
        if row.get("future_review_only") is not True:
            raise ValueError(f"{event}: future path is not label-only")
        if any(row.get(field) is True for field in ("sealed", "sealed_read", "purge", "purged", "is_purge")):
            raise ValueError(f"{event}: candidate is sealed/purge-derived")
        partition = row.get("source_partition", row.get("partition"))
        if partition is not None and str(partition).lower() not in {"inner", "mining"}:
            raise ValueError(f"{event}: candidate is not INNER")
        selected_at = row.get("selected_at_thresholds")
        if not isinstance(selected_at, list):
            raise ValueError(f"{event}: selected_at_thresholds must be a list")
        try:
            normalized = [float(value) for value in selected_at]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{event}: invalid selected_at_thresholds") from exc
        if normalized != sorted(set(normalized)) or any(value not in thresholds for value in normalized):
            raise ValueError(f"{event}: selected_at_thresholds is outside export grid")
        score = float(row.get("p_signal", math.nan))
        if any(score < value for value in normalized):
            raise ValueError(f"{event}: selected_at_thresholds exceeds p_signal")
        if not isinstance(row.get("gold"), bool):
            raise ValueError(f"{event}: gold must be boolean")
        if not row["gold"] and not normalized:
            raise ValueError(f"{event}: non-Gold candidate has no selected threshold")
        for bar in row.get("future_path", []):
            if not isinstance(bar, Mapping):
                raise ValueError(f"{event}: invalid future_path bar")
            if _utc(bar.get("time", bar.get("open_time")), "future_path time") >= INNER_CUTOFF:
                raise ValueError(f"{event}: future_path reaches sealed interval")
    validated = validate_candidates(raw_rows, max_horizon=max_horizon, allowed_end=INNER_CUTOFF)
    source_symbols = {str(row["symbol"]) for row in validated}
    if not source_symbols or not source_symbols <= set(sampled):
        raise ValueError("candidate symbols disagree with summary sampling lineage")
    # Recompute the frozen threshold+gap on the complete physical INNER export,
    # then compare it to the backtester's recorded threshold-union membership.
    # Cropping first would reset gap state at a caller-chosen start boundary.
    selected = gap_deduplicate(validated, config.threshold)
    declared_selected = {
        str(row["event_id"]) for row in validated
        if config.threshold in [float(value) for value in row["selected_at_thresholds"]]
    }
    recomputed_selected = {str(row["event_id"]) for row in selected}
    if recomputed_selected != declared_selected:
        raise ValueError("frozen L1 threshold/gap disagrees with backtester export lineage")

    interval: list[dict[str, Any]] = []
    label_cropped = 0
    for row in selected:
        decision = _utc(row["decision_time"], "decision_time")
        label_end = pd.Timestamp(row["_path_frame"]["open_time"].iloc[config.horizon - 1]) + BAR_DURATION
        if start <= decision < end:
            if label_end > end:
                label_cropped += 1
                continue
            row["_label_end"] = label_end
            interval.append(row)
    if not interval:
        raise ValueError("requested cross-section interval has no usable candidates")
    return interval, {
        "path": str(candidates), "sha256": candidate_sha, "rows": len(raw_rows),
        "summary_path": str(summary_file), "summary_sha256": summary_sha,
        "weights_path": str(weights_path), "weights_sha256": weights_sha,
        "renderer": summary["renderer"],
        "transform": summary["transform"], "imgsz": summary["imgsz"],
        "optimization_export": dict(export_protocol),
        "sampled_symbols": sorted(sampled), "candidate_symbols": sorted(source_symbols),
        "l1_selected_interval_rows": len(interval), "label_end_cropped": label_cropped,
    }, source_symbols


def _profit_factor(values: Sequence[float]) -> float | None:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return None if losses <= 0 else gains / losses


def _max_drawdown(values: Sequence[float]) -> float:
    cumulative = peak = maximum = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        maximum = max(maximum, peak - cumulative)
    return maximum


def _metrics(frame: pd.DataFrame, *, threshold: float | None,
             start: pd.Timestamp, end: pd.Timestamp, exposure_symbols: int) -> dict[str, Any]:
    selected = frame if threshold is None else frame[frame["l2_score"] >= threshold]
    selected = selected.sort_values(["signal_time", "symbol", "event_id"], kind="mergesort")
    maker = selected["net_barrier_maker"].to_numpy(dtype=float).tolist()
    taker = selected["net_barrier_taker"].to_numpy(dtype=float).tolist()
    days = (end - start).total_seconds() / 86400.0
    canonical = {name: int((selected["outcome"] == name).sum())
                 for name in ("tp", "sl", "sl_ambiguous", "timeout")}
    grouped = {
        "tp": canonical["tp"], "sl": canonical["sl"] + canonical["sl_ambiguous"],
        "timeout": canonical["timeout"],
    }
    return {
        "threshold": threshold, "trades": len(selected),
        "exposure_symbols": exposure_symbols, "exposure_symbol_days": exposure_symbols * days,
        "density_per_symbol_day": len(selected) / (exposure_symbols * days),
        "mean_net_maker": None if not maker else float(np.mean(maker)),
        "total_net_maker": float(sum(maker)),
        "mean_net_taker": None if not taker else float(np.mean(taker)),
        "total_net_taker": float(sum(taker)),
        "win_rate_maker": None if not maker else sum(value > 0 for value in maker) / len(maker),
        "profit_factor_maker": _profit_factor(maker),
        "max_drawdown_maker": _max_drawdown(maker),
        "outcomes": grouped, "canonical_outcomes": canonical,
    }


def _periods(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    output: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    cursor = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    while cursor < end:
        following = cursor + pd.offsets.MonthBegin(1)
        left, right = max(start, cursor), min(end, following)
        if left < right:
            output.append((cursor.strftime("%Y-%m"), left, right))
        cursor = following
    return output


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def evaluate_frozen_l2_crosssection(
    *, frozen_gate_dir: Path, frozen_lineage_sha256: str,
    candidates: Path, summary: Path, candidate_summary_sha256: str,
    snapshot_dir: Path, manifest: Path, snapshot_manifest_sha256: str,
    out_dir: Path, start: Any = DEFAULT_START, end: Any = DEFAULT_END,
) -> dict[str, Any]:
    """Run a fixed, non-selective cross-symbol confirmation and publish atomically."""
    start_ts, end_ts = _utc(start, "start"), _utc(end, "end")
    if not start_ts < end_ts < INNER_CUTOFF:
        raise ValueError("evaluation interval must be ordered and end before 2026-04-02")
    if start_ts < pd.Timestamp("2026-01-01T00:00:00Z"):
        raise ValueError("cross-section confirmation may not start before 2026-01-01")
    out = out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out}")

    model, config, l2_threshold, gate_lineage = _load_supported_frozen_gate(
        frozen_gate_dir, frozen_lineage_sha256
    )
    rows, candidate_lineage, candidate_symbols = _load_independent_candidates(
        candidates, summary, candidate_summary_sha256,
        config=config, start=start_ts, end=end_ts,
    )
    gate_candidate_symbols = set(gate_lineage["gate_candidate_symbols"])
    training_dataset_symbols = set(gate_lineage["training_dataset_symbols"])
    frozen_seen_symbols = gate_candidate_symbols | training_dataset_symbols
    overlap = frozen_seen_symbols & candidate_symbols
    if overlap:
        raise ValueError(
            "confirmation symbols overlap frozen gate candidates/training dataset: "
            f"{sorted(overlap)}"
        )
    for field, expected in gate_lineage["frozen_l1_protocol"].items():
        if candidate_lineage[field] != expected:
            raise ValueError(f"independent L1 {field} differs from frozen gate lineage")

    manifest_path = manifest.resolve()
    manifest_sha = _checked_sha(
        manifest_path, snapshot_manifest_sha256, "snapshot manifest SHA256"
    )
    summary_raw = _json(summary.resolve(), "candidate sibling summary")
    declared_manifest = _resolve_declared_path(
        summary_raw.get("manifest"), parent=summary.resolve().parent, field="summary.manifest"
    )
    declared_snapshot = _resolve_declared_path(
        summary_raw.get("snapshot_dir"), parent=summary.resolve().parent,
        field="summary.snapshot_dir",
    )
    if declared_manifest != manifest_path or declared_snapshot != snapshot_dir.resolve():
        raise ValueError("snapshot/manifest do not match candidate summary physical lineage")
    if _sha256(declared_manifest) != manifest_sha:
        raise ValueError("candidate summary snapshot manifest SHA256 mismatch")

    selected = list(rows)
    selected.sort(key=lambda row: (
        _utc(row["decision_time"], "decision_time"), str(row["symbol"]), str(row["event_id"])
    ))
    if not selected:
        raise ValueError("frozen L1 threshold/gap selects no cross-section events")
    frames, snapshot_symbols, declarations = _load_frames(
        snapshot_dir.resolve(), manifest_path, {str(row["symbol"]) for row in selected}
    )
    events: list[dict[str, Any]] = []
    for row in selected:
        outcome = _resolve(row, config)
        decision = _utc(row["decision_time"], "decision_time")
        symbol = str(row["symbol"])
        features = _feature_vector(frames[symbol], symbol, decision)
        events.append({
            "event_id": str(row["event_id"]), "symbol": symbol,
            "signal_time": decision, "label_end_time": _utc(row["_label_end"], "label_end"),
            "outcome": str(outcome["outcome"]).lower(),
            "net_barrier_maker": float(outcome["net_maker"]),
            "net_barrier_taker": float(outcome["net_taker"]), **features,
        })
    frame = pd.DataFrame(events).sort_values(
        ["signal_time", "symbol", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    frame["l2_score"] = _predict(model, frame)
    exposure_symbols = len(candidate_lineage["sampled_symbols"])
    overall = {
        "period_start": start_ts.isoformat(), "period_end_exclusive": end_ts.isoformat(),
        "frozen_l2_gate": _metrics(frame, threshold=l2_threshold, start=start_ts, end=end_ts,
                                   exposure_symbols=exposure_symbols),
        "baseline_all_l1_selected": _metrics(
            frame, threshold=None, start=start_ts, end=end_ts, exposure_symbols=exposure_symbols
        ),
    }
    monthly: dict[str, Any] = {}
    for name, left, right in _periods(start_ts, end_ts):
        month = frame[(frame["signal_time"] >= left) & (frame["signal_time"] < right)]
        monthly[name] = {
            "period_start": left.isoformat(), "period_end_exclusive": right.isoformat(),
            "frozen_l2_gate": _metrics(month, threshold=l2_threshold, start=left, end=right,
                                       exposure_symbols=exposure_symbols),
            "baseline_all_l1_selected": _metrics(
                month, threshold=None, start=left, end=right, exposure_symbols=exposure_symbols
            ),
        }

    prereg = {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "frozen_gate_lineage_sha256": gate_lineage["lineage_sha256"],
        "candidate_summary_sha256": candidate_lineage["summary_sha256"],
        "candidate_source_sha256": candidate_lineage["sha256"],
        "snapshot_manifest_sha256": manifest_sha,
        "selected_config": {"threshold": config.threshold, "tp_atr": config.tp_atr,
                            "sl_atr": config.sl_atr, "horizon": config.horizon},
        "selected_config_key": config.key, "frozen_l2_threshold": l2_threshold,
        "start": start_ts.isoformat(), "end_exclusive": end_ts.isoformat(),
        "selection_or_training_performed": False, "threshold_grid_output": False,
        "holdout_rows_read": 0, "sealed_read": False, "purge_rows_read": 0,
        "future_rows_in_features": 0, "auto_promote": False, "orders": False,
    }
    evaluation = {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "selected_config_key": config.key, "frozen_l2_threshold": l2_threshold,
        "l1_selected_rows_in_interval": candidate_lineage["l1_selected_interval_rows"],
        "l1_selected_events": len(frame), "overall": overall, "monthly": monthly,
        "threshold_grid_output": False, "holdout_rows_read": 0, "sealed_read": False,
        "future_rows_in_features": 0, "auto_promote": False, "orders": False,
    }
    lineage = {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
        "frozen_gate": gate_lineage, "candidate_input": candidate_lineage,
        "snapshot": {"directory": str(snapshot_dir.resolve()),
                     "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
                     "files": declarations, "selected_symbol_files": snapshot_symbols},
        "feature_columns": list(FEATURE_COLUMNS), "feature_semantics": FEATURE_SEMANTICS,
        "side": "short", "gate_candidate_symbols": sorted(gate_candidate_symbols),
        "training_dataset_symbols": sorted(training_dataset_symbols),
        "frozen_seen_symbols": sorted(frozen_seen_symbols),
        "confirmation_candidate_symbols": sorted(candidate_symbols),
        "symbol_overlap": [], "artifacts": {},
        "safety": {"holdout_rows_read": 0, "sealed_read": False, "purge_rows_read": 0,
                   "future_rows_in_features": 0, "selection_or_training_performed": False,
                   "auto_promote": False, "orders": False, "model_deployable": False},
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(f".{out.name}.tmp-{uuid.uuid4().hex}")
    try:
        temp.mkdir()
        _write_json(temp / "prereg.json", prereg)
        _write_json(temp / "crosssection_eval.json", evaluation)
        lineage["artifacts"] = {
            "prereg.json": _sha256(temp / "prereg.json"),
            "crosssection_eval.json": _sha256(temp / "crosssection_eval.json"),
        }
        _write_json(temp / "lineage.json", lineage)
        os.replace(temp, out)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return {"out_dir": str(out), "overall": overall, "monthly": monthly}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-gate-dir", type=Path, required=True)
    parser.add_argument("--frozen-lineage-sha256", required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidate-summary-sha256", required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot-manifest-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_frozen_l2_crosssection(
        frozen_gate_dir=args.frozen_gate_dir,
        frozen_lineage_sha256=args.frozen_lineage_sha256,
        candidates=args.candidates, summary=args.summary,
        candidate_summary_sha256=args.candidate_summary_sha256,
        snapshot_dir=args.snapshot_dir, manifest=args.manifest,
        snapshot_manifest_sha256=args.snapshot_manifest_sha256,
        out_dir=args.out_dir, start=args.start, end=args.end,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
