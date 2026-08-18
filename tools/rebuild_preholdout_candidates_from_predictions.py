#!/usr/bin/env python3
"""Rebuild inner-only candidate artifacts from a frozen prediction ledger.

This is the repair path for pre-holdout runs whose image inference is valid but
whose optimizer export predates the canonical ATR floor.  The source run is
used only for ``summary.json`` and ``predictions.jsonl``.  Predictions are
reconciled against a separately SHA-pinned, physically bounded snapshot, then
the current backtester owns both optimizer selection and outcome resolution.

No model or renderer is loaded here.  In particular, source sealed exports are
never opened and no output with a sealed scope is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.backtest_fixed_w10_cls_preholdout import (  # noqa: E402
    ATR_PCT_MIN,
    DEFAULT_INNER_END,
    HOLDOUT_START,
    IMGSZ,
    PROTOCOL,
    WINDOW_BARS,
    OptimizationInterval,
    OutcomeProtocol,
    _event_id,
    _utc_timestamp,
    build_inner_backtest_outputs,
    build_optimization_rows,
    label_end_time,
    load_materialization_manifest,
    load_preholdout_csv,
    sha256_file,
    summarize_trades,
    validate_unique_records,
)


REBUILD_REASON = "atr_floor_contract_fix"
EXPECTED_RENDERER = "yoyo.datasets.legacy_gold_migration.renderer.render_w10"
EXPECTED_TRANSFORM = "white_letterbox_full_frame"
JSONL_OUTPUTS = (
    "predictions.jsonl",
    "signals_raw.jsonl",
    "signals_dedup.jsonl",
    "stop_losses.jsonl",
    "l1_review_queue.jsonl",
    "l2_outcome_negatives.jsonl",
    "optimization_inner.jsonl",
)


class RebuildError(ValueError):
    """The input lineage cannot prove a clean inner-only rebuild."""


def _require_sha(value: Any, name: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RebuildError(f"{name} must be a lowercase SHA256")
    return digest


def _safe_source_run(path: Path) -> Path:
    """Reject result directories whose identity denotes sealed/holdout use.

    ``preholdout`` is allowed because it names the approved source protocol;
    the residual component may not itself name sealed or holdout material.
    """
    for component in path.parts:
        residual = component.lower().replace("preholdout", "")
        if "sealed" in residual or "holdout" in residual:
            raise RebuildError("source-run path must not identify sealed or holdout material")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise RebuildError("source-run must be an existing directory")
    return resolved


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RebuildError(f"invalid {name}") from exc
    if not isinstance(value, dict):
        raise RebuildError(f"{name} must be a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RebuildError(
                        f"invalid predictions.jsonl at line {line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise RebuildError(
                        f"predictions.jsonl line {line_number} must be an object"
                    )
                rows.append(row)
    except OSError as exc:
        raise RebuildError("cannot read predictions.jsonl") from exc
    return rows


def _unsafe_claims(value: Any, *, location: str) -> None:
    """Fail closed on positive safety counters or explicit unsafe booleans."""
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            child_location = f"{location}.{raw_key}"
            if key in {"holdout_read", "sealed_read"} and child is not False:
                raise RebuildError(f"{child_location} must be false")
            if key in {"sealed", "is_sealed", "holdout", "is_holdout"} and child is True:
                raise RebuildError(f"{child_location}=true is forbidden")
            if (
                any(token in key for token in ("holdout_rows", "sealed_rows", "april_rows"))
                and isinstance(child, (int, float))
                and not isinstance(child, bool)
                and child != 0
            ):
                raise RebuildError(f"{child_location} must be zero")
            if key in {
                "future_data_in_causal_input",
                "future_used_in_model_input",
                "future_rows_in_features",
                "future_rows_in_model_input",
            } and child not in (False, 0):
                raise RebuildError(f"{child_location} proves a future-data leak")
            _unsafe_claims(child, location=child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _unsafe_claims(child, location=f"{location}[{index}]")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise RebuildError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RebuildError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise RebuildError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise RebuildError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RebuildError(f"{name} must be an integer") from exc
    if str(value).strip() not in {str(result), f"{result}.0"} and not isinstance(value, int):
        raise RebuildError(f"{name} must be an integer")
    return result


def _same_number(left: Any, right: Any, name: str) -> None:
    if _number(left, name) != _number(right, name):
        raise RebuildError(f"source summary has inconsistent {name}")


def _validate_source_summary(summary: dict[str, Any]) -> tuple[OutcomeProtocol, int, float, tuple[float, ...], pd.Timestamp]:
    _unsafe_claims(summary, location="summary")
    if summary.get("protocol") != PROTOCOL:
        raise RebuildError("source summary protocol mismatch")
    if summary.get("evaluation_scope") != "preholdout":
        raise RebuildError("source summary must declare evaluation_scope=preholdout")
    if summary.get("holdout_read") is not False:
        raise RebuildError("source summary must declare holdout_read=false")
    if summary.get("future_data_in_causal_input") is not False:
        raise RebuildError("source summary must declare future_data_in_causal_input=false")
    if summary.get("renderer") != EXPECTED_RENDERER or summary.get("renderer_overlay") is not False:
        raise RebuildError("source summary renderer contract mismatch")
    if summary.get("transform") != EXPECTED_TRANSFORM:
        raise RebuildError("source summary transform contract mismatch")
    if _integer(summary.get("imgsz"), "summary.imgsz") != IMGSZ:
        raise RebuildError("source summary imgsz contract mismatch")
    if _integer(summary.get("window_bars"), "summary.window_bars") != WINDOW_BARS:
        raise RebuildError("source summary window contract mismatch")
    _require_sha(summary.get("weights_sha256"), "summary.weights_sha256")
    if summary.get("orders_placed") is not False or summary.get("promoted") is not False:
        raise RebuildError("source summary must prove no orders and no promotion")

    raw_candidate = summary.get("candidate_parameters")
    if not isinstance(raw_candidate, Mapping):
        raise RebuildError("source summary lacks candidate_parameters")
    candidate = OutcomeProtocol(
        threshold=_number(raw_candidate.get("threshold"), "candidate threshold"),
        tp_atr=_number(raw_candidate.get("tp_atr"), "candidate tp_atr"),
        sl_atr=_number(raw_candidate.get("sl_atr"), "candidate sl_atr"),
        horizon_bars=_integer(raw_candidate.get("horizon_bars"), "candidate horizon_bars"),
    ).validated()
    for field in ("threshold", "tp_atr", "sl_atr", "horizon_bars"):
        if field not in summary:
            raise RebuildError(f"source summary lacks top-level {field}")
        _same_number(summary[field], getattr(candidate, field), field)

    export = summary.get("optimization_export")
    if not isinstance(export, Mapping):
        raise RebuildError("source summary lacks optimization_export")
    max_horizon = _integer(export.get("max_horizon"), "optimization max_horizon")
    if max_horizon <= 0:
        raise RebuildError("optimization max_horizon must be positive")
    min_score = _number(export.get("min_score"), "optimization min_score")
    raw_thresholds = export.get("thresholds")
    if not isinstance(raw_thresholds, list) or not raw_thresholds:
        raise RebuildError("optimization thresholds must be a non-empty list")
    thresholds = tuple(sorted(_number(value, "optimization threshold") for value in raw_thresholds))
    if len(set(thresholds)) != len(thresholds) or any(not 0 <= value <= 1 for value in thresholds):
        raise RebuildError("optimization thresholds must be unique and in [0, 1]")
    if not 0 <= min_score <= 1 or thresholds[0] != min_score:
        raise RebuildError("optimization min_score must equal the minimum threshold")
    inner_end = _utc_timestamp(export.get("inner_end_exclusive"), field="inner_end_exclusive")
    if inner_end > DEFAULT_INNER_END:
        raise RebuildError(
            f"source inner_end exceeds {DEFAULT_INNER_END.isoformat()}"
        )
    if inner_end >= HOLDOUT_START:
        raise RebuildError("source inner_end reaches holdout")
    scope = summary.get("outcome_scope")
    if not isinstance(scope, Mapping) or scope.get("name") != "optimization_inner":
        raise RebuildError("source outcome_scope must be optimization_inner")
    for field in ("decision_before", "label_end_lte"):
        if _utc_timestamp(scope.get(field), field=f"outcome_scope.{field}") != inner_end:
            raise RebuildError(f"source outcome_scope.{field} disagrees with inner_end")
    return candidate, max_horizon, min_score, thresholds, inner_end


def _load_snapshot(
    snapshot_dir: Path,
    manifest: Path,
    expected_manifest_sha: str,
    inner_end: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], str]:
    snapshot_dir = snapshot_dir.resolve()
    manifest = manifest.resolve()
    if not snapshot_dir.is_dir() or not manifest.is_file():
        raise RebuildError("snapshot-dir and manifest must exist")
    expected = _require_sha(expected_manifest_sha, "snapshot-manifest-sha256")
    actual = sha256_file(manifest)
    if actual != expected:
        raise RebuildError("snapshot manifest SHA256 mismatch")
    raw_manifest = _read_json_object(manifest, "snapshot manifest")
    _unsafe_claims(raw_manifest, location="snapshot_manifest")
    declarations = load_materialization_manifest(snapshot_dir, manifest)
    frames: dict[str, pd.DataFrame] = {}
    checked: list[dict[str, Any]] = []
    for declaration in declarations:
        declared_max = _utc_timestamp(
            declaration.get("max_materialized_time"), field="max_materialized_time"
        )
        if declared_max >= DEFAULT_INNER_END or declared_max >= inner_end:
            raise RebuildError("snapshot max time must be strictly before inner_end")
        path = snapshot_dir / str(declaration["path"])
        frame = load_preholdout_csv(path, declaration)
        times = pd.to_datetime(frame["open_time"], utc=True)
        if len(times) > 1 and not (times.diff().iloc[1:] == pd.Timedelta(minutes=15)).all():
            raise RebuildError(f"snapshot {path.name} is not continuous 15-minute data")
        symbol = str(declaration.get("symbol") or path.stem)
        if not symbol or symbol in frames:
            raise RebuildError(f"duplicate or empty snapshot symbol: {symbol}")
        frames[symbol] = frame
        checked.append({
            "path": path.name,
            "symbol": symbol,
            "sha256": str(declaration["sha256"]),
            "rows": len(frame),
            "max_materialized_time": pd.Timestamp(times.max()).isoformat(),
        })
    return frames, checked, actual


def _validate_predictions(
    rows: list[dict[str, Any]],
    summary: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    candidate: OutcomeProtocol,
    inner_end: pd.Timestamp,
) -> None:
    expected_rows = _integer(summary.get("n_predictions"), "summary.n_predictions")
    if expected_rows != len(rows):
        raise RebuildError(
            f"prediction row-count mismatch: summary={expected_rows}, actual={len(rows)}"
        )
    validate_unique_records(rows)
    for line_number, row in enumerate(rows, 1):
        location = f"predictions line {line_number}"
        _unsafe_claims(row, location=location)
        if any("sealed" in str(key).lower() for key in row):
            raise RebuildError(f"{location} contains a sealed field")
        if "future_path" in row or "outcome" in row:
            raise RebuildError(f"{location} is not a pure causal prediction")
        if row.get("protocol") != PROTOCOL:
            raise RebuildError(f"{location} protocol mismatch")
        if row.get("holdout_read") is not False:
            raise RebuildError(f"{location} must declare holdout_read=false")
        if row.get("future_data_in_causal_input") is not False:
            raise RebuildError(f"{location} must declare future_data_in_causal_input=false")
        symbol = str(row.get("symbol") or "")
        frame = frames.get(symbol)
        if frame is None:
            raise RebuildError(f"{location} references unknown symbol {symbol!r}")
        decision_i = _integer(row.get("decision_i"), f"{location}.decision_i")
        if decision_i < WINDOW_BARS - 1 or decision_i >= len(frame):
            raise RebuildError(f"{location} decision_i is outside the snapshot")
        decision = _utc_timestamp(row.get("decision_time"), field=f"{location}.decision_time")
        physical_decision = pd.Timestamp(frame["open_time"].iloc[decision_i])
        if decision != physical_decision:
            raise RebuildError(f"{location} decision identity does not match snapshot")
        if row.get("event_id") != _event_id(symbol, decision):
            raise RebuildError(f"{location} event_id mismatch")
        start_i = decision_i - WINDOW_BARS + 1
        if _integer(row.get("window_start_i"), f"{location}.window_start_i") != start_i:
            raise RebuildError(f"{location} window_start_i mismatch")
        if _integer(row.get("window_end_i"), f"{location}.window_end_i") != decision_i:
            raise RebuildError(f"{location} window_end_i mismatch")
        if _utc_timestamp(
            row.get("causal_window_start_time"), field=f"{location}.causal_window_start_time"
        ) != pd.Timestamp(frame["open_time"].iloc[start_i]):
            raise RebuildError(f"{location} causal window start mismatch")
        if _utc_timestamp(
            row.get("causal_window_end_time"), field=f"{location}.causal_window_end_time"
        ) != decision:
            raise RebuildError(f"{location} causal window end mismatch")
        score = _number(row.get("p_signal"), f"{location}.p_signal")
        if not 0 <= score <= 1:
            raise RebuildError(f"{location} p_signal must be in [0, 1]")
        expected_pred = "SIGNAL" if score >= candidate.threshold else "NO_SIGNAL"
        if row.get("pred") != expected_pred:
            raise RebuildError(f"{location} pred disagrees with frozen threshold")
        if decision >= inner_end or label_end_time(decision, candidate.horizon_bars) > inner_end:
            raise RebuildError(f"{location} label_end exceeds inner_end")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def rebuild(
    *,
    source_run: Path,
    snapshot_dir: Path,
    manifest: Path,
    snapshot_manifest_sha256: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Validate immutable inputs and atomically write a clean inner rebuild."""
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    source_run = _safe_source_run(source_run)
    source_summary_path = source_run / "summary.json"
    source_predictions_path = source_run / "predictions.jsonl"
    if not source_summary_path.is_file() or not source_predictions_path.is_file():
        raise RebuildError("source-run must contain summary.json and predictions.jsonl")
    source_summary_sha = sha256_file(source_summary_path)
    source_predictions_sha = sha256_file(source_predictions_path)
    source_summary = _read_json_object(source_summary_path, "source summary")
    candidate, max_horizon, min_score, thresholds, inner_end = _validate_source_summary(
        source_summary
    )
    frames, snapshot_sources, snapshot_manifest_sha = _load_snapshot(
        snapshot_dir, manifest, snapshot_manifest_sha256, inner_end
    )
    predictions = _read_jsonl(source_predictions_path)
    _validate_predictions(predictions, source_summary, frames, candidate, inner_end)

    optimization_rows, optimization_stats = build_optimization_rows(
        predictions,
        frames,
        (OptimizationInterval.from_values("inner", "1900-01-01T00:00:00Z", inner_end),),
        max_horizon,
        None,
        min_score,
        thresholds,
    )
    inner_outputs = build_inner_backtest_outputs(predictions, frames, candidate, inner_end)
    if len(inner_outputs["predictions"]) != len(predictions):
        raise RebuildError("current backtester unexpectedly removed a validated inner prediction")
    output_rows = {
        "signals_raw.jsonl": inner_outputs["signals_raw"],
        "signals_dedup.jsonl": inner_outputs["signals_dedup"],
        "stop_losses.jsonl": inner_outputs["stop_losses"],
        "l1_review_queue.jsonl": inner_outputs["l1_review_queue"],
        "l2_outcome_negatives.jsonl": inner_outputs["l2_outcome_negatives"],
        "optimization_inner.jsonl": optimization_rows["inner"],
    }
    for name, rows in output_rows.items():
        validate_unique_records(rows)
        for row in rows:
            if label_end_time(row["decision_time"], (
                max_horizon if name == "optimization_inner.jsonl" else candidate.horizon_bars
            )) > inner_end:
                raise RebuildError(f"{name} contains a label beyond inner_end")

    out_dir = out_dir.resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir.with_name(f".{out_dir.name}.tmp-{uuid.uuid4().hex}")
    try:
        temp_dir.mkdir(exist_ok=False)
        shutil.copyfile(source_predictions_path, temp_dir / "predictions.jsonl")
        if sha256_file(temp_dir / "predictions.jsonl") != source_predictions_sha:
            raise RebuildError("copied prediction ledger SHA256 mismatch")
        for name, rows in output_rows.items():
            _write_jsonl(temp_dir / name, rows)
        artifact_rows = {"predictions.jsonl": len(predictions), **{
            name: len(rows) for name, rows in output_rows.items()
        }}
        artifacts = {
            name: {"rows": artifact_rows[name], "sha256": sha256_file(temp_dir / name)}
            for name in JSONL_OUTPUTS
        }
        summary = {
            "schema_version": 1,
            "artifact": "preholdout_candidate_rebuild_from_predictions",
            "protocol": PROTOCOL,
            "rebuild_reason": REBUILD_REASON,
            "source_run": str(source_run),
            "source_predictions_sha256": source_predictions_sha,
            "source_summary_sha256": source_summary_sha,
            "snapshot_dir": str(snapshot_dir.resolve()),
            "snapshot_manifest": str(manifest.resolve()),
            "snapshot_manifest_sha256": snapshot_manifest_sha,
            "snapshot_sources": snapshot_sources,
            "weights_sha256": str(source_summary["weights_sha256"]).lower(),
            "renderer": EXPECTED_RENDERER,
            "renderer_overlay": False,
            "transform": EXPECTED_TRANSFORM,
            "imgsz": IMGSZ,
            "window_bars": WINDOW_BARS,
            "candidate_parameters": {
                "threshold": float(candidate.threshold),
                "tp_atr": float(candidate.tp_atr),
                "sl_atr": float(candidate.sl_atr),
                "horizon_bars": int(candidate.horizon_bars),
            },
            "threshold": float(candidate.threshold),
            "tp_atr": float(candidate.tp_atr),
            "sl_atr": float(candidate.sl_atr),
            "horizon_bars": int(candidate.horizon_bars),
            "atr_pct_min": float(ATR_PCT_MIN),
            "optimization_export": {
                "file": "optimization_inner.jsonl",
                "max_horizon": max_horizon,
                "min_score": min_score,
                "thresholds": list(thresholds),
                "inner_end_exclusive": inner_end.isoformat(),
                "sealed_exported": False,
            },
            "optimization_stats": optimization_stats,
            "outcome_scope": {
                "name": "optimization_inner",
                "decision_before": inner_end.isoformat(),
                "label_end_lte": inner_end.isoformat(),
            },
            "n_predictions": len(predictions),
            "n_signal_raw": len(inner_outputs["signals_raw"]),
            "n_signal_dedup": len(inner_outputs["signals_dedup"]),
            "n_stop_losses": len(inner_outputs["stop_losses"]),
            "n_l1_review": len(inner_outputs["l1_review_queue"]),
            "n_l2_outcome_negatives": len(inner_outputs["l2_outcome_negatives"]),
            "raw": summarize_trades(inner_outputs["signals_raw"]),
            "deduped": summarize_trades(inner_outputs["signals_dedup"]),
            "artifacts": artifacts,
            "evaluation_scope": "preholdout_inner_only",
            "holdout_read": False,
            "holdout_rows_read": 0,
            "sealed_read": False,
            "sealed_rows_read": 0,
            "april_rows_read": 0,
            "future_data_in_causal_input": False,
            "future_rows_in_features": 0,
            "future_rows_in_model_input": 0,
            "auto_promote": False,
            "promoted": False,
            "orders": False,
            "orders_placed": False,
        }
        (temp_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_dir, out_dir)
        return summary
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--snapshot-manifest-sha256", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = rebuild(
            source_run=args.source_run,
            snapshot_dir=args.snapshot_dir,
            manifest=args.manifest,
            snapshot_manifest_sha256=args.snapshot_manifest_sha256,
            out_dir=args.out_dir,
        )
    except (RebuildError, FileExistsError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "source_predictions_sha256": summary["source_predictions_sha256"],
        "artifacts": summary["artifacts"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
