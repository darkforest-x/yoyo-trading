#!/usr/bin/env python3
"""Materialize a causal binary L2 dataset from completed INNER optimizer runs.

This is a post-selection data builder, not an optimizer or evaluator.  Each
candidate JSONL must be paired with the genuine ``config_selection.json`` from
the run that selected its barrier contract.  The sibling ``prereg.json`` binds
that selection to the candidate bytes.  Multiple runs may be pooled only when
their selected configs are identical.

Future paths are consumed solely by the canonical SHORT barrier resolver.  The
28 model inputs are recomputed from a SHA/row/max-time-gated, inner-only OHLCV
snapshot sliced at the decision bar before causal feature extraction.  TP is a
positive, SL and conservative same-bar ambiguity are negatives, and timeout is
reported but excluded.  A caller-supplied UTC validation boundary creates a
deterministic train/validation split; overlap with the fixed purge window is
excluded rather than randomly assigned.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.backtest_fixed_w10_cls_preholdout import (  # noqa: E402
    HOLDOUT_START,
    load_materialization_manifest,
    load_preholdout_csv,
    sha256_file,
)
from tools.optimize_short_barrier_walkforward import (  # noqa: E402
    BAR_DURATION,
    ENTRY_POLICY,
    GAP_POLICY,
    RETURN_CONVENTION,
    SAME_BAR_POLICY,
    SearchConfig,
    _resolve,
    gap_deduplicate,
    read_jsonl,
    validate_candidates,
)
from yoyo.data.indicators import MIN_GAP_BARS  # noqa: E402
from yoyo.layers.l2_judgment.features import (  # noqa: E402
    FEATURE_COLUMNS,
    add_features,
    extract_feature_rows_for_semantics,
)

SCHEMA_VERSION = 1
PROTOCOL = "l2_feedback_dataset_v1"
INNER_CUTOFF = pd.Timestamp("2026-04-02T00:00:00Z")
DEFAULT_PURGE_BARS = 162
FEATURE_SEMANTICS = "side_aligned_v1"
INCLUDED_OUTCOMES = {"tp": 1, "sl": 0, "sl_ambiguous": 0}


def _utc(value: Any, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if pd.isna(stamp):
        raise ValueError(f"invalid {field}: {value!r}")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _load_json_object(path: Path, artifact: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {artifact}: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{artifact} must be a JSON object")
    return raw


def _require_unsealed_v1(raw: Mapping[str, Any], artifact: str) -> None:
    if type(raw.get("schema_version")) is not int or raw.get("schema_version") != 1:
        raise ValueError(f"{artifact} requires schema_version=1")
    if raw.get("sealed_read") is not False:
        raise ValueError(f"{artifact} must prove sealed_read=false")
    if raw.get("sealed") is True or raw.get("holdout_read") is True:
        raise ValueError(f"{artifact} is sealed/holdout-derived")
    if any(raw.get(marker) is True for marker in ("purge", "purged", "is_purge")):
        raise ValueError(f"{artifact} is a purge artifact")


def _valid_config(raw: Mapping[str, Any]) -> SearchConfig:
    try:
        config = SearchConfig(**raw)
    except TypeError as exc:
        raise ValueError("selected_config does not match optimizer schema") from exc
    if (not math.isfinite(config.threshold) or not 0 <= config.threshold <= 1
            or not math.isfinite(config.tp_atr) or config.tp_atr <= 0
            or not math.isfinite(config.sl_atr) or config.sl_atr <= 0
            or type(config.horizon) is not int or config.horizon <= 0):
        raise ValueError("selected_config contains invalid values")
    return config


def load_source_pair(candidates_path: Path, selection_path: Path) -> tuple[SearchConfig, dict[str, Any]]:
    """Validate selection/prereg/candidate lineage before reading candidate rows."""
    candidates_path = candidates_path.resolve()
    selection_path = selection_path.resolve()
    if not candidates_path.is_file() or not selection_path.is_file():
        raise ValueError("candidate and selection paths must be readable files")

    selection = _load_json_object(selection_path, "config selection")
    _require_unsealed_v1(selection, "config selection")
    if not isinstance(selection.get("selected_config"), Mapping):
        raise ValueError("config selection requires selected_config")
    config = _valid_config(selection["selected_config"])
    if selection.get("selected_config_key") != config.key:
        raise ValueError("selected_config_key does not match selected_config")

    prereg_path = selection_path.with_name("prereg.json")
    if not prereg_path.is_file():
        raise ValueError("config selection requires sibling prereg.json")
    prereg = _load_json_object(prereg_path, "sibling prereg")
    _require_unsealed_v1(prereg, "sibling prereg")
    prereg_sha = sha256_file(prereg_path)
    if selection.get("prereg_sha256") != prereg_sha:
        raise ValueError("config selection prereg_sha256 mismatch")
    candidate_sha = sha256_file(candidates_path)
    if prereg.get("candidate_source_sha256") != candidate_sha:
        raise ValueError("sibling prereg candidate_source_sha256 mismatch")
    declared_source = prereg.get("candidate_source")
    if not isinstance(declared_source, str) or not declared_source:
        raise ValueError("sibling prereg requires candidate_source")
    if prereg.get("side") != "short":
        raise ValueError("sibling prereg must declare side=short")
    if prereg.get("min_gap_bars") != MIN_GAP_BARS:
        raise ValueError(f"sibling prereg must declare min_gap_bars={MIN_GAP_BARS}")
    for key, expected in {
        "entry_policy": ENTRY_POLICY,
        "same_bar_policy": SAME_BAR_POLICY,
        "gap_policy": GAP_POLICY,
        "return_convention": RETURN_CONVENTION,
    }.items():
        if prereg.get(key) != expected:
            raise ValueError(f"sibling prereg has non-canonical {key}")

    return config, {
        "candidates_path": str(candidates_path),
        "candidates_sha256": candidate_sha,
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "prereg_path": str(prereg_path.resolve()),
        "prereg_sha256": prereg_sha,
    }


def _validate_inner_markers(rows: Sequence[Mapping[str, Any]], source: Path) -> None:
    for row in rows:
        event_id = str(row.get("event_id", "<unknown>"))
        if (row.get("sealed") is True or row.get("sealed_read") is True
                or row.get("holdout_read") is True):
            raise ValueError(f"{source}: {event_id} is sealed/holdout-derived")
        if any(row.get(marker) is True for marker in ("purge", "purged", "is_purge")):
            raise ValueError(f"{source}: {event_id} is a purge row")
        partition = row.get("source_partition", row.get("partition"))
        if partition is not None and str(partition).strip().lower() not in {"inner", "mining"}:
            raise ValueError(f"{source}: {event_id} is not an INNER candidate")
        decision = _utc(row.get("decision_time"), "decision_time")
        if decision >= INNER_CUTOFF:
            raise ValueError(f"{source}: {event_id} decision reaches sealed interval")
        path = row.get("future_path")
        if isinstance(path, list):
            for bar in path:
                if isinstance(bar, Mapping):
                    stamp = _utc(bar.get("time", bar.get("open_time")), "future_path time")
                    if stamp >= HOLDOUT_START:
                        raise ValueError(f"{source}: {event_id} future_path reaches holdout")


def _load_frames(
    snapshot_dir: Path,
    manifest_path: Path,
    symbols: set[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Gate an explicitly inner-only snapshot before pandas sees OHLCV values."""
    raw_manifest = _load_json_object(manifest_path, "snapshot manifest")
    if (raw_manifest.get("sealed") is True or raw_manifest.get("sealed_read") is True
            or raw_manifest.get("holdout_read") is True
            or any(raw_manifest.get(marker) is True
                   for marker in ("purge", "purged", "is_purge"))):
        raise ValueError("snapshot manifest is sealed/purge/holdout-derived")
    declarations = load_materialization_manifest(snapshot_dir, manifest_path)
    for item in declarations:
        maximum = _utc(item["max_materialized_time"], "max_materialized_time")
        if maximum >= INNER_CUTOFF:
            raise ValueError("snapshot is not inner-only: max_materialized_time reaches sealed interval")
    frames: dict[str, pd.DataFrame] = {}
    lineage: dict[str, dict[str, Any]] = {}
    for item in declarations:
        path = snapshot_dir / str(item["path"])
        symbol = str(item.get("symbol") or path.stem)
        if symbol not in symbols:
            continue
        if symbol in frames:
            raise ValueError(f"duplicate snapshot symbol: {symbol}")
        frames[symbol] = load_preholdout_csv(path, item)
        lineage[symbol] = {
            "snapshot_file": str(item["path"]),
            "snapshot_file_sha256": str(item["sha256"]),
            "snapshot_file_rows": int(item["rows"]),
            "snapshot_max_materialized_time": str(item["max_materialized_time"]),
        }
    missing = sorted(symbols - set(frames))
    if missing:
        raise ValueError(f"candidate symbols missing from snapshot manifest: {missing}")
    return frames, lineage, declarations


def _feature_vector(frame: pd.DataFrame, symbol: str, decision: pd.Timestamp) -> dict[str, float]:
    times = pd.to_datetime(frame["open_time"], utc=True)
    matches = np.flatnonzero((times == decision).to_numpy())
    if len(matches) != 1:
        raise ValueError(
            f"{symbol} {decision.isoformat()}: expected exactly one decision bar, got {len(matches)}"
        )
    decision_i = int(matches[0])
    # The physical slice makes the causal boundary explicit and testable: no
    # later row exists when feature code executes.
    featured = add_features(frame.iloc[: decision_i + 1].copy())
    vector = extract_feature_rows_for_semantics(
        featured,
        [len(featured) - 1],
        feature_semantics=FEATURE_SEMANTICS,
        side="short",
    )
    if list(vector.columns) != FEATURE_COLUMNS or vector.shape != (1, len(FEATURE_COLUMNS)):
        raise ValueError("feature extraction does not match frozen 28-column schema")
    values = vector.iloc[0].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [name for name in FEATURE_COLUMNS if not np.isfinite(float(vector.iloc[0][name]))]
        raise ValueError(f"{symbol} {decision.isoformat()}: non-finite features: {bad}")
    return {name: float(vector.iloc[0][name]) for name in FEATURE_COLUMNS}


def _csv_sha(path: Path) -> str:
    return sha256_file(path)


def _rows_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _time_range(rows: Sequence[Mapping[str, Any]]) -> list[str | None]:
    if not rows:
        return [None, None]
    times = [_utc(row["signal_time"], "signal_time") for row in rows]
    return [min(times).isoformat(), max(times).isoformat()]


def _class_balance(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {"0": sum(int(row["label"]) == 0 for row in rows),
            "1": sum(int(row["label"]) == 1 for row in rows)}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def build_l2_feedback_dataset(
    *,
    sources: Sequence[tuple[Path, Path]],
    snapshot_dir: Path,
    materialization_manifest: Path,
    out_dir: Path,
    val_start: Any,
    purge_bars: int = DEFAULT_PURGE_BARS,
) -> dict[str, Any]:
    """Build and atomically publish a train/validation L2 feedback dataset."""
    if not sources:
        raise ValueError("at least one --source pair is required")
    if type(purge_bars) is not int or purge_bars != DEFAULT_PURGE_BARS:
        raise ValueError(f"purge-bars is frozen at {DEFAULT_PURGE_BARS}")
    val_start_ts = _utc(val_start, "val_start")
    if val_start_ts >= INNER_CUTOFF:
        raise ValueError("val_start must be before the inner cutoff")
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out_dir}")

    configs: list[SearchConfig] = []
    source_lineages: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    identities: set[tuple[str, pd.Timestamp]] = set()
    for source_index, (candidate_arg, selection_arg) in enumerate(sources):
        candidate_path, selection_path = candidate_arg.resolve(), selection_arg.resolve()
        config, lineage = load_source_pair(candidate_path, selection_path)
        raw_rows = read_jsonl(candidate_path)
        _validate_inner_markers(raw_rows, candidate_path)
        validated = validate_candidates(
            raw_rows, max_horizon=config.horizon, allowed_end=INNER_CUTOFF
        )
        for row in validated:
            event_id = str(row["event_id"])
            identity = (str(row["symbol"]), _utc(row["decision_time"], "decision_time"))
            if event_id in event_ids or identity in identities:
                raise ValueError(f"duplicate event/identity across sources: {event_id} {identity}")
            event_ids.add(event_id)
            identities.add(identity)
            row["_source_index"] = source_index
            label_end = pd.Timestamp(row["_path_frame"]["open_time"].iloc[config.horizon - 1]) + BAR_DURATION
            if label_end >= INNER_CUTOFF:
                raise ValueError(f"{event_id}: label_end reaches sealed interval")
            row["_label_end"] = label_end
        configs.append(config)
        source_lineages.append(lineage)
        all_rows.extend(validated)

    selected_config = configs[0]
    if any(config != selected_config for config in configs[1:]):
        raise ValueError("selected configs across sources must match exactly")

    chosen = gap_deduplicate(all_rows, selected_config.threshold)
    chosen.sort(key=lambda row: (
        _utc(row["decision_time"], "decision_time"), str(row["symbol"]), str(row["event_id"])
    ))
    outcome_counts: dict[str, int] = {}
    excluded_timeouts = 0
    resolved_rows: list[dict[str, Any]] = []
    for row in chosen:
        resolved = _resolve(row, selected_config)
        outcome = str(resolved["outcome"]).lower()
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        if outcome == "timeout":
            excluded_timeouts += 1
            continue
        if outcome not in INCLUDED_OUTCOMES:
            raise ValueError(f"unsupported canonical outcome: {outcome}")
        resolved_rows.append({**row, **resolved, "outcome": outcome})

    snapshot_dir = snapshot_dir.resolve()
    materialization_manifest = materialization_manifest.resolve()
    manifest_sha = sha256_file(materialization_manifest)
    frames, snapshot_lineage, declarations = _load_frames(
        snapshot_dir, materialization_manifest, {str(row["symbol"]) for row in resolved_rows}
    )
    purge_boundary = val_start_ts - purge_bars * BAR_DURATION
    dataset_rows: list[dict[str, Any]] = []
    purged_rows = 0
    for row in resolved_rows:
        decision = _utc(row["decision_time"], "decision_time")
        label_end = _utc(row["_label_end"], "label_end")
        if decision >= val_start_ts:
            split = "val"
        elif label_end < purge_boundary:
            split = "train"
        else:
            purged_rows += 1
            continue
        source_index = int(row["_source_index"])
        lineage = source_lineages[source_index]
        symbol = str(row["symbol"])
        features = _feature_vector(frames[symbol], symbol, decision)
        dataset_rows.append({
            "event_id": str(row["event_id"]),
            "symbol": symbol,
            "signal_time": decision.isoformat(),
            "side": "short",
            "split": split,
            "label": INCLUDED_OUTCOMES[str(row["outcome"])],
            "realized_ret": float(row["gross_ret"]),
            "net_barrier_maker": float(row["net_maker"]),
            "net_barrier_taker": float(row["net_taker"]),
            "outcome": str(row["outcome"]),
            "feature_semantics": FEATURE_SEMANTICS,
            **features,
            "feature_as_of": decision.isoformat(),
            "label_end_time": label_end.isoformat(),
            "selected_config_key": selected_config.key,
            "candidate_source_sha256": lineage["candidates_sha256"],
            "selected_config_sha256": lineage["selection_sha256"],
            "prereg_sha256": lineage["prereg_sha256"],
            "snapshot_manifest_sha256": manifest_sha,
            **snapshot_lineage[symbol],
            "holdout_read": False,
            "future_rows_in_features": 0,
            "future_review_only": True,
            "auto_promote": False,
        })

    dataset_rows.sort(key=lambda row: (row["signal_time"], row["symbol"], row["event_id"]))
    by_split = {name: [row for row in dataset_rows if row["split"] == name]
                for name in ("train", "val")}
    for name, rows in by_split.items():
        if not rows:
            raise ValueError(f"{name} split is empty")
        if {int(row["label"]) for row in rows} != {0, 1}:
            raise ValueError(f"{name} split must contain both classes")

    columns = [
        "event_id", "symbol", "signal_time", "side", "split", "label",
        "realized_ret", "net_barrier_maker", "net_barrier_taker", "outcome",
        "feature_semantics", *FEATURE_COLUMNS, "feature_as_of", "label_end_time",
        "selected_config_key", "candidate_source_sha256", "selected_config_sha256",
        "prereg_sha256", "snapshot_manifest_sha256", "snapshot_file",
        "snapshot_file_sha256", "snapshot_file_rows", "snapshot_max_materialized_time",
        "holdout_read", "future_rows_in_features", "future_review_only", "auto_promote",
    ]

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir.with_name(f".{out_dir.name}.tmp-{uuid.uuid4().hex}")
    try:
        temp_dir.mkdir()
        dataset_path = temp_dir / "dataset.csv"
        _write_csv(dataset_path, dataset_rows, columns)
        dataset_sha = _csv_sha(dataset_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "dataset": {
                "path": "dataset.csv", "rows": len(dataset_rows), "sha256": dataset_sha,
                "rows_sha256": _rows_sha(dataset_rows),
            },
            "selected_config": asdict(selected_config),
            "selected_config_key": selected_config.key,
            "inputs": source_lineages,
            "snapshot": {
                "directory": str(snapshot_dir),
                "manifest_path": str(materialization_manifest),
                "manifest_sha256": manifest_sha,
                "files": declarations,
            },
            "counts": {
                "input_candidates": len(all_rows),
                "threshold_gap_selected": len(chosen),
                "resolved_outcomes": outcome_counts,
                "timeouts_excluded": excluded_timeouts,
                "purge_rows_excluded": purged_rows,
                "dataset_rows": len(dataset_rows),
                "train_rows": len(by_split["train"]),
                "val_rows": len(by_split["val"]),
            },
            "class_balance": {
                "all": _class_balance(dataset_rows),
                "train": _class_balance(by_split["train"]),
                "val": _class_balance(by_split["val"]),
            },
            "time_ranges": {
                "all": _time_range(dataset_rows),
                "train": _time_range(by_split["train"]),
                "val": _time_range(by_split["val"]),
            },
            "split": {
                "method": "explicit_time_boundary",
                "val_start": val_start_ts.isoformat(),
                "purge_bars": purge_bars,
                "purge_duration": str(purge_bars * BAR_DURATION),
                "train_rule": "label_end_time < val_start - purge_duration",
                "val_rule": "signal_time >= val_start",
                "random": False,
            },
            "protocol_contract": {
                "side": "short", "feature_semantics": FEATURE_SEMANTICS,
                "feature_columns": list(FEATURE_COLUMNS), "min_gap_bars": MIN_GAP_BARS,
                "entry_policy": ENTRY_POLICY, "same_bar_policy": SAME_BAR_POLICY,
                "gap_policy": GAP_POLICY, "return_convention": RETURN_CONVENTION,
                "included_outcomes": INCLUDED_OUTCOMES,
            },
            "inner_cutoff": INNER_CUTOFF.isoformat(),
            "holdout_start": HOLDOUT_START.isoformat(),
            "holdout_rows_read": 0,
            "future_rows_in_features": 0,
            "auto_promote": False,
        }
        with (temp_dir / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temp_dir, out_dir)
        return manifest
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _source_arg(value: str) -> tuple[Path, Path]:
    left, separator, right = value.partition(",")
    if not separator or not left.strip() or not right.strip() or "," in right:
        raise argparse.ArgumentTypeError("--source must be CANDIDATES_JSONL,CONFIG_SELECTION_JSON")
    return Path(left.strip()), Path(right.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_source_arg, required=True,
                        metavar="CANDIDATES,SELECTION")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="materialized inner-only snapshot manifest")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-start", required=True, help="UTC validation decision boundary")
    parser.add_argument("--purge-bars", type=int, default=DEFAULT_PURGE_BARS,
                        help=f"frozen at {DEFAULT_PURGE_BARS}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_l2_feedback_dataset(
        sources=args.source,
        snapshot_dir=args.snapshot_dir,
        materialization_manifest=args.manifest,
        out_dir=args.out_dir,
        val_start=args.val_start,
        purge_bars=args.purge_bars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
