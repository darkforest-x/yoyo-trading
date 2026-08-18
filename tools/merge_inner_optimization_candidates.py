#!/usr/bin/env python3
"""Merge independently sampled INNER optimization exports without inference.

The fixed-W10 backtester deliberately emits one physically separate
``optimization_inner.jsonl`` beside an auditable ``summary.json``.  This tool
accepts only those pairs, proves that their model/render/export/snapshot
contracts are compatible and that their sampling groups are disjoint, then
publishes a deterministic candidate file for a later INNER-only optimizer run.
It never resolves outcomes, reads the sealed export, promotes a model, or
places an order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.optimize_short_barrier_walkforward import validate_candidates  # noqa: E402

SEALED_START = pd.Timestamp("2026-04-02T00:00:00Z")
HOLDOUT_START = pd.Timestamp("2026-05-04T00:00:00Z")
SCHEMA_VERSION = 1
SHA_FIELDS = (
    "snapshot_manifest_sha256",
    "materialization_manifest_sha256",
    "manifest_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc(value: Any, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if pd.isna(stamp):
        raise ValueError(f"invalid {field}: {value!r}")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _require_sha(value: Any, field: str) -> str:
    normalized = str(value or "").lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{field} must be a lowercase-compatible SHA256")
    return normalized


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return raw


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"candidate row must be an object at {path}:{line_number}")
                rows.append(row)
    except OSError as exc:
        raise ValueError(f"cannot read source: {path}") from exc
    if not rows:
        raise ValueError(f"source contains no candidate rows: {path}")
    return rows


def _declared_inner(summary: Mapping[str, Any], source: Path) -> dict[str, Any]:
    exports = summary.get("optimization_exports")
    if not isinstance(exports, Mapping) or not isinstance(exports.get("inner"), Mapping):
        raise ValueError("summary requires optimization_exports.inner")
    inner = dict(exports["inner"])
    declared = inner.get("path", inner.get("file"))
    if not isinstance(declared, str) or not declared:
        raise ValueError("summary optimization inner export requires file/path")
    declared_path = Path(declared)
    if not declared_path.is_absolute():
        declared_path = source.parent / declared_path
    if declared_path.resolve() != source.resolve():
        raise ValueError("summary optimization inner file path does not match source")
    expected = _require_sha(inner.get("sha256"), "optimization inner SHA256")
    if sha256_file(source) != expected:
        raise ValueError("optimization inner source SHA256 mismatch")
    if "rows" not in inner:
        raise ValueError("summary optimization inner export requires row count")
    return inner


def _snapshot_lineage(summary: Mapping[str, Any], summary_path: Path) -> tuple[str | None, Any | None]:
    explicit: str | None = None
    for field in SHA_FIELDS:
        if field in summary:
            explicit = _require_sha(summary[field], field)
            break
    snapshot = summary.get("snapshot")
    if explicit is None and isinstance(snapshot, Mapping):
        for field in SHA_FIELDS:
            if field in snapshot:
                explicit = _require_sha(snapshot[field], f"snapshot.{field}")
                break
    manifest = summary.get("manifest")
    if explicit is None and isinstance(manifest, str) and manifest:
        path = Path(manifest)
        if not path.is_absolute():
            path = summary_path.parent / path
        if path.is_file():
            explicit = sha256_file(path.resolve())
    contract = summary.get("snapshot_contract")
    if contract is None and isinstance(snapshot, Mapping):
        contract = snapshot.get("contract")
    if explicit is None and contract is None:
        raise ValueError("summary requires snapshot manifest SHA/path or snapshot_contract")
    return explicit, contract


def _sampling_symbols(summary: Mapping[str, Any]) -> tuple[set[str], Any]:
    sampling = summary.get("symbol_sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("summary requires symbol_sampling")
    selected = sampling.get("selected_symbols")
    if not isinstance(selected, list) or not selected or any(not isinstance(item, str) or not item for item in selected):
        raise ValueError("symbol_sampling.selected_symbols must be a non-empty string list")
    if len(set(selected)) != len(selected):
        raise ValueError("symbol_sampling.selected_symbols contains duplicates")
    group = sampling.get("group", sampling.get("sampling_group", sorted(selected)))
    return set(selected), group


def _protocol(summary: Mapping[str, Any]) -> dict[str, Any]:
    export = summary.get("optimization_export")
    if not isinstance(export, Mapping):
        raise ValueError("summary requires optimization_export")
    try:
        max_horizon = int(export["max_horizon"])
        min_score = float(export["min_score"])
        thresholds = [float(value) for value in export["thresholds"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid optimization export max_horizon/min_score/thresholds") from exc
    if max_horizon <= 0:
        raise ValueError("optimization export max_horizon must be positive")
    if not math.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("optimization export min_score must be in [0, 1]")
    if not thresholds or any(not math.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("optimization export thresholds must be finite values in [0, 1]")
    if thresholds != sorted(set(thresholds)) or thresholds[0] != min_score:
        raise ValueError("optimization export thresholds must be sorted unique and start at min_score")
    required = {
        "weights_sha256": _require_sha(summary.get("weights_sha256"), "weights_sha256"),
        "renderer": summary.get("renderer"),
        "transform": summary.get("transform"),
        "imgsz": summary.get("imgsz"),
        "max_horizon": max_horizon,
        "min_score": min_score,
        "thresholds": thresholds,
    }
    if not isinstance(required["renderer"], str) or not required["renderer"]:
        raise ValueError("summary requires renderer")
    if not isinstance(required["transform"], str) or not required["transform"]:
        raise ValueError("summary requires transform")
    try:
        required["imgsz"] = int(required["imgsz"])
    except (TypeError, ValueError) as exc:
        raise ValueError("summary requires integer imgsz") from exc
    if required["imgsz"] <= 0:
        raise ValueError("imgsz must be positive")
    return required


def _validate_summary(summary: Mapping[str, Any], source: Path, summary_path: Path) -> None:
    if summary.get("holdout_read") is not False:
        raise ValueError("summary must prove holdout_read=false")
    if summary.get("future_data_in_causal_input") is not False:
        raise ValueError("summary must prove future_data_in_causal_input=false")
    scope = summary.get("outcome_scope")
    if not isinstance(scope, Mapping) or scope.get("name") != "optimization_inner":
        raise ValueError("summary outcome_scope.name must be optimization_inner")
    if scope.get("sealed_outcomes_resolved") is not False:
        raise ValueError("summary must prove sealed_outcomes_resolved=false")
    before = _utc(scope.get("decision_before"), "outcome_scope.decision_before")
    if before > SEALED_START:
        raise ValueError("summary outcome scope reaches optimization_sealed")
    if before >= HOLDOUT_START:
        raise ValueError("summary outcome scope reaches holdout")
    _declared_inner(summary, source)
    exports = summary["optimization_exports"]
    sealed = exports.get("sealed") if isinstance(exports, Mapping) else None
    if isinstance(sealed, Mapping):
        declared = sealed.get("path", sealed.get("file"))
        if isinstance(declared, str) and declared:
            sealed_path = Path(declared)
            if not sealed_path.is_absolute():
                sealed_path = source.parent / sealed_path
            if sealed_path.resolve() == source.resolve():
                raise ValueError("source is physically the optimization_sealed export")
    if source.name != "optimization_inner.jsonl" or "optimization_sealed" in str(source).lower():
        raise ValueError("source must be the physical optimization_inner.jsonl export")


def _validate_row(row: Mapping[str, Any], *, thresholds: Sequence[float]) -> None:
    event = str(row.get("event_id", "<missing>"))
    partitions = [row[key] for key in ("partition", "source_partition") if key in row]
    # Physical source identity plus the sibling summary prove INNER for the
    # backtester's native schema, which intentionally has no partition field.
    # If a downstream producer adds one, it becomes an additional fail-closed
    # assertion rather than a requirement imposed on the canonical export.
    if any(value not in {"inner", "mining"} for value in partitions):
        raise ValueError(f"{event}: explicit partition/source_partition must be inner/mining")
    if row.get("sealed") is True or row.get("sealed_read") is True:
        raise ValueError(f"{event}: explicit sealed/sealed_read=true is forbidden")
    if row.get("holdout_read") is not False:
        raise ValueError(f"{event}: holdout_read must be false")
    if row.get("future_data_in_causal_input") is not False:
        raise ValueError(f"{event}: future data appears in causal input")
    for field in ("future_used_in_model_input", "future_used_in_training_image"):
        if field in row and row[field] is not False:
            raise ValueError(f"{event}: {field} must be false")
    if row.get("future_review_only") is not True:
        raise ValueError(f"{event}: future_path must be label-only")
    decision = _utc(row.get("decision_time"), "decision_time")
    if decision >= SEALED_START:
        raise ValueError(f"{event}: decision reaches optimization_sealed")
    if decision >= HOLDOUT_START:
        raise ValueError(f"{event}: decision reaches holdout")
    selected = row.get("selected_at_thresholds")
    if not isinstance(selected, list):
        raise ValueError(f"{event}: selected_at_thresholds must be a list")
    try:
        normalized = [float(value) for value in selected]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{event}: invalid selected_at_thresholds") from exc
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{event}: selected_at_thresholds must be sorted and unique")
    if any(value not in thresholds for value in normalized):
        raise ValueError(f"{event}: selected_at_thresholds is outside export grid")
    if not isinstance(row.get("gold"), bool):
        raise ValueError(f"{event}: gold must be boolean")
    if not row["gold"] and not normalized:
        raise ValueError(f"{event}: non-Gold candidate has no selected threshold")
    score = float(row.get("p_signal", math.nan))
    if any(score < value for value in normalized):
        raise ValueError(f"{event}: selected_at_thresholds exceeds p_signal")
    path = row.get("future_path")
    if not isinstance(path, list) or not path:
        raise ValueError(f"{event}: future_path must be non-empty label data")
    for bar in path:
        if not isinstance(bar, Mapping) or "time" not in bar:
            raise ValueError(f"{event}: invalid future_path bar")
        if _utc(bar["time"], "future_path.time") >= HOLDOUT_START:
            raise ValueError(f"{event}: future_path reaches holdout")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def merge_sources(sources: Sequence[Path], out_dir: Path) -> dict[str, Any]:
    """Validate sources and atomically publish a fresh merged directory."""
    if len(sources) < 2:
        raise ValueError("at least two --source files are required")
    out_dir = out_dir.expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    resolved = [path.expanduser().resolve() for path in sources]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate --source path")

    all_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    expected_protocol: dict[str, Any] | None = None
    snapshot_shas: list[str | None] = []
    snapshot_contracts: list[Any | None] = []
    sampled_seen: set[str] = set()
    sampling_groups_seen: set[str] = set()
    event_seen: set[str] = set()
    identity_seen: set[tuple[str, str]] = set()

    for source in sorted(resolved, key=str):
        if not source.is_file():
            raise ValueError(f"source is not a file: {source}")
        summary_path = source.parent / "summary.json"
        summary = _read_json(summary_path, "sibling summary.json")
        _validate_summary(summary, source, summary_path)
        protocol = _protocol(summary)
        if expected_protocol is None:
            expected_protocol = protocol
        elif protocol != expected_protocol:
            raise ValueError("sources have incompatible weights/render/export grid protocol")
        snapshot_sha, snapshot_contract = _snapshot_lineage(summary, summary_path)
        snapshot_shas.append(snapshot_sha)
        snapshot_contracts.append(snapshot_contract)
        sampled, sampling_group = _sampling_symbols(summary)
        sampling_overlap = sampled_seen & sampled
        sampling_group_key = json.dumps(
            sampling_group, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        group_overlap = sampling_group_key in sampling_groups_seen

        rows = _read_jsonl(source)
        inner = summary["optimization_exports"]["inner"]
        if int(inner["rows"]) != len(rows):
            raise ValueError("optimization inner row count does not match summary")
        source_symbols: set[str] = set()
        for row in rows:
            _validate_row(row, thresholds=protocol["thresholds"])
            event = str(row["event_id"])
            symbol = str(row["symbol"])
            if not event or not symbol:
                raise ValueError("candidate event_id and symbol must be non-empty")
            decision = _utc(row["decision_time"], "decision_time").isoformat()
            if event in event_seen:
                raise ValueError(f"duplicate event_id across sources: {event}")
            if (symbol, decision) in identity_seen:
                raise ValueError(f"duplicate symbol+decision identity across sources: {symbol} {decision}")
            event_seen.add(event)
            identity_seen.add((symbol, decision))
            source_symbols.add(symbol)
        if not source_symbols <= sampled:
            raise ValueError("source candidates contain symbols outside its declared sampling group")
        if sampling_overlap:
            raise ValueError(f"sampling groups/symbols overlap: {sorted(sampling_overlap)}")
        if group_overlap:
            raise ValueError("sampling group identity overlaps another source")
        sampled_seen.update(sampled)
        sampling_groups_seen.add(sampling_group_key)
        # Reuse the optimizer's canonical structural/causal/next-open validation.
        validate_candidates(rows, max_horizon=int(protocol["max_horizon"]), allowed_end=SEALED_START)
        all_rows.extend(rows)
        source_records.append({
            "path": str(source),
            "sha256": sha256_file(source),
            "rows": len(rows),
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": sha256_file(summary_path),
            "symbols": sorted(source_symbols),
            "sampling_group": sampling_group,
            "sampled_symbols": sorted(sampled),
            "snapshot_manifest_sha256": snapshot_sha,
        })

    assert expected_protocol is not None
    non_null_shas = [value for value in snapshot_shas if value is not None]
    same_snapshot_sha = len(non_null_shas) == len(snapshot_shas) and len(set(non_null_shas)) == 1
    encoded_contracts = [json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                         for value in snapshot_contracts if value is not None]
    same_contract = len(encoded_contracts) == len(snapshot_contracts) and len(set(encoded_contracts)) == 1
    if not same_snapshot_sha and not same_contract:
        raise ValueError("sources have incompatible snapshot manifests/contracts")

    ordered = sorted(
        all_rows,
        key=lambda row: (_utc(row["decision_time"], "decision_time"), str(row["symbol"]), str(row["event_id"])),
    )
    candidate_bytes = _jsonl_bytes(ordered)
    identity_bytes = b"".join(
        f"{row['event_id']}\0{row['symbol']}\0{_utc(row['decision_time'], 'decision_time').isoformat()}\n".encode("utf-8")
        for row in ordered
    )
    decisions = [_utc(row["decision_time"], "decision_time") for row in ordered]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "merged_optimization_inner_candidates",
        "sources": source_records,
        "source_count": len(source_records),
        "protocol": expected_protocol,
        "snapshot_compatibility": {
            "same_manifest_sha256": same_snapshot_sha,
            "same_contract": same_contract,
            "manifest_sha256": non_null_shas[0] if same_snapshot_sha else None,
        },
        "combined": {
            "file": "candidates.jsonl",
            "sha256": _sha256_bytes(candidate_bytes),
            "rows": len(ordered),
            "decision_time_min": min(decisions).isoformat(),
            "decision_time_max": max(decisions).isoformat(),
            "event_identity_sha256": _sha256_bytes(identity_bytes),
            "symbols": sorted({str(row["symbol"]) for row in ordered}),
        },
        "combined_sha256": _sha256_bytes(candidate_bytes),
        "combined_rows": len(ordered),
        "decision_time_min": min(decisions).isoformat(),
        "decision_time_max": max(decisions).isoformat(),
        "event_identity_sha256": _sha256_bytes(identity_bytes),
        "holdout_rows_read": 0,
        "future_rows_in_causal_input": 0,
        "sealed_rows": 0,
        "auto_promote": False,
        "safety": {
            "outcome_scope": "optimization_inner",
            "decision_before": SEALED_START.isoformat(),
            "holdout_start": HOLDOUT_START.isoformat(),
            "holdout_rows_read": 0,
            "future_rows_in_causal_input": 0,
            "sealed_rows": 0,
            "sealed_outcomes_resolved": False,
            "auto_promote": False,
            "orders_placed": False,
            "future_path_role": "label_only_unchanged",
        },
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    try:
        (temp_dir / "candidates.jsonl").write_bytes(candidate_bytes)
        (temp_dir / "manifest.json").write_bytes(manifest_bytes)
        if out_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
        os.replace(temp_dir, out_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True,
                        help="repeat for each optimization_inner.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True, help="fresh output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = merge_sources(args.source, args.out_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
