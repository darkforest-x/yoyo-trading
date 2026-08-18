#!/usr/bin/env python3
"""Materialize a bounded, auditable pre-holdout K-line snapshot.

The owner-gold manifest is the only safe index for the historical CSV archive:
it records a known ``win_end`` and its timestamp for every source.  This tool
uses that pair to calculate a requested prefix length, then asks pandas for
*only that prefix*.  It deliberately never scans a CSV to discover the holdout
boundary: those archives can contain later data, and merely reading it would
violate the holdout discipline.  The snapshot is for L2/L3 outcome labels only
after a decision; L1 rendering must still stop at the decision bar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd


BAR = pd.Timedelta(minutes=15)
DEFAULT_START = "2026-01-01T00:00:00Z"
DEFAULT_END = "2026-05-03T23:45:00Z"
DEFAULT_CUTOFF = "2026-05-04T00:00:00Z"
DEFAULT_MANIFEST = Path("/Users/zhangzc/fable-trading/datasets/owner_short_gold_center_v1/positive_manifest.jsonl")
DEFAULT_SOURCE_ROOT = Path("/Users/zhangzc/fable-trading")


class SnapshotError(ValueError):
    """The requested materialization cannot prove its pre-holdout contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _timestamp(value: str | pd.Timestamp, *, name: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise SnapshotError(f"{name} must include a timezone")
    return parsed.tz_convert("UTC")


def _require_bar_aligned(value: pd.Timestamp, *, name: str) -> None:
    if value != value.floor("15min"):
        raise SnapshotError(f"{name} is not aligned to a 15 minute bar: {_iso(value)}")


def _read_manifest_metadata(path: Path) -> list[dict[str, Any]]:
    """Read JSONL metadata only; source CSVs are intentionally untouched here."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                for field in ("symbol", "source_csv", "win_end", "end_time"):
                    if field not in row:
                        raise SnapshotError(f"manifest line {line_no} lacks {field}")
                rows.append(row)
            except json.JSONDecodeError as exc:
                raise SnapshotError(f"invalid JSON on manifest line {line_no}") from exc
    if not rows:
        raise SnapshotError("source manifest contains no rows")
    return rows


def _group_sources(rows: list[dict[str, Any]], max_symbols: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["symbol"])].append(row)
    if len(grouped) > max_symbols:
        raise SnapshotError(f"manifest has {len(grouped)} symbols, exceeds max_symbols={max_symbols}")
    for symbol, references in grouped.items():
        sources = {str(reference["source_csv"]) for reference in references}
        if len(sources) != 1:
            raise SnapshotError(f"symbol {symbol} has multiple source_csv values: {sorted(sources)}")
    return dict(sorted(grouped.items()))


def _index_for_time(reference: dict[str, Any], requested: pd.Timestamp) -> int:
    reference_time = _timestamp(str(reference["end_time"]), name="reference end_time")
    _require_bar_aligned(reference_time, name="reference end_time")
    delta = requested - reference_time
    if delta % BAR:
        raise SnapshotError(f"requested time cannot map to 15m reference for {reference['symbol']}")
    return int(reference["win_end"]) + int(delta // BAR)


def _validate_prefix(frame: pd.DataFrame, *, symbol: str) -> pd.Series:
    if "open_time" not in frame.columns:
        raise SnapshotError(f"{symbol} source has no open_time column")
    times = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    if times.isna().any():
        raise SnapshotError(f"{symbol} prefix has invalid open_time")
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise SnapshotError(f"{symbol} prefix times are not unique and strictly increasing")
    if len(times) > 1 and not (times.diff().iloc[1:] == BAR).all():
        raise SnapshotError(f"{symbol} prefix is not continuous 15m data")
    if any(time != time.floor("15min") for time in times):
        raise SnapshotError(f"{symbol} prefix has a non-15m-aligned timestamp")
    return times


def _source_path(source_root: Path, manifest_value: str) -> Path:
    candidate = Path(manifest_value)
    return candidate if candidate.is_absolute() else source_root / candidate


def materialize_snapshot(
    *,
    manifest_path: Path,
    source_root: Path,
    output_dir: Path,
    start: str | pd.Timestamp = DEFAULT_START,
    end: str | pd.Timestamp = DEFAULT_END,
    cutoff: str | pd.Timestamp = DEFAULT_CUTOFF,
    warmup_bars: int = 288,
    max_horizon: int = 144,
    max_symbols: int = 215,
) -> dict[str, Any]:
    """Write one new snapshot directory, reading no CSV row at/after ``cutoff``."""
    requested_start = _timestamp(start, name="start")
    requested_end = _timestamp(end, name="end")
    holdout_cutoff = _timestamp(cutoff, name="cutoff")
    for name, value in (("start", requested_start), ("end", requested_end), ("cutoff", holdout_cutoff)):
        _require_bar_aligned(value, name=name)
    if requested_start > requested_end:
        raise SnapshotError("start must not be after end")
    if requested_end >= holdout_cutoff:
        raise SnapshotError("end must be strictly before cutoff")
    if warmup_bars < 0 or max_horizon <= 0 or max_symbols <= 0:
        raise SnapshotError("warmup_bars must be nonnegative; horizons and symbol count must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    rows = _read_manifest_metadata(manifest_path)
    grouped = _group_sources(rows, max_symbols)
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    files: list[dict[str, Any]] = []
    skips: list[dict[str, str]] = []
    try:
        (temp_dir / "kline_snapshot").mkdir(parents=True)
        for symbol, references in grouped.items():
            # All references must give the same index for each requested wall time.
            end_indices = {_index_for_time(row, requested_end) for row in references}
            start_indices = {_index_for_time(row, requested_start) for row in references}
            if len(end_indices) != 1 or len(start_indices) != 1:
                raise SnapshotError(f"{symbol} manifest references disagree about source index/time")
            target_end_index, target_start_index = end_indices.pop(), start_indices.pop()
            reference = references[0]
            if target_end_index < int(reference["win_end"]):
                skips.append({"symbol": symbol, "reason": "requested_end_before_source_reference"})
                continue
            prefix_start_index = target_start_index - warmup_bars
            if prefix_start_index < 0:
                skips.append({"symbol": symbol, "reason": "insufficient_warmup_before_requested_start"})
                continue
            source = _source_path(source_root, str(reference["source_csv"]))
            if not source.is_file():
                skips.append({"symbol": symbol, "reason": "source_csv_missing"})
                continue
            # This is the safety boundary: do not remove nrows or replace it with a full read.
            prefix_rows_requested = target_end_index + 1
            frame = pd.read_csv(source, nrows=prefix_rows_requested)
            times = _validate_prefix(frame, symbol=symbol)
            if len(frame) <= int(reference["win_end"]):
                raise SnapshotError(f"{symbol} source ended before its manifest reference")
            ref_time = _timestamp(str(reference["end_time"]), name="reference end_time")
            if times.iloc[int(reference["win_end"])] != ref_time:
                raise SnapshotError(f"{symbol} reference win_end/end_time does not match source prefix")
            if len(frame) <= prefix_start_index:
                skips.append({"symbol": symbol, "reason": "source_ended_before_requested_start_warmup"})
                continue
            # If a calculated target index is unavailable, the already-read final safe row is allowed.
            safe_positions = times[times <= requested_end]
            if safe_positions.empty:
                skips.append({"symbol": symbol, "reason": "no_row_at_or_before_requested_end"})
                continue
            actual_end_index = int(safe_positions.index[-1])
            if actual_end_index < target_start_index:
                skips.append({"symbol": symbol, "reason": "source_ended_before_requested_start"})
                continue
            if times.iloc[actual_end_index] >= holdout_cutoff:
                raise SnapshotError(f"{symbol} would materialize holdout data")
            sliced = frame.iloc[prefix_start_index : actual_end_index + 1].copy()
            sliced_times = times.iloc[prefix_start_index : actual_end_index + 1]
            if sliced_times.iloc[-1] > requested_end:
                raise SnapshotError(f"{symbol} materialized end exceeds requested end")
            destination = temp_dir / "kline_snapshot" / f"{symbol}.csv"
            sliced.to_csv(destination, index=False)
            relative = destination.relative_to(temp_dir).as_posix()
            files.append(
                {
                    "symbol": symbol,
                    "path": relative,
                    "sha256": _sha256_file(destination),
                    "rows": len(sliced),
                    "start": _iso(sliced_times.iloc[0]),
                    "end": _iso(sliced_times.iloc[-1]),
                    "source_path": str(source),
                    "source_prefix_rows_requested": prefix_rows_requested,
                    "source_reference": {
                        "sample_id": reference.get("sample_id"),
                        "win_end": int(reference["win_end"]),
                        "end_time": _iso(ref_time),
                    },
                    "prefix_max_time": _iso(times.iloc[-1]),
                    "holdout_read": False,
                }
            )
        last_safe = requested_end - max_horizon * BAR
        content_shas = {item["path"]: item["sha256"] for item in files}
        snapshot_manifest: dict[str, Any] = {
            "schema_version": 1,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256_file(manifest_path),
            "cutoff": _iso(holdout_cutoff),
            "requested_start": _iso(requested_start),
            "requested_end": _iso(requested_end),
            "warmup_bars": warmup_bars,
            "max_horizon_supported": max_horizon,
            "last_safe_decision_time": _iso(last_safe),
            "future_data_role": "outcome_label_only_after_decision",
            "training_render_input_ends_at": "decision",
            "counts": {"manifest_rows": len(rows), "symbols_seen": len(grouped), "materialized": len(files), "skipped": len(skips)},
            "skips": skips,
            "files": files,
            "content_shas": content_shas,
        }
        (temp_dir / "snapshot_manifest.json").write_text(
            json.dumps(snapshot_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # The L3 preholdout backtester intentionally accepts only files directly
        # below its --snapshot-dir.  Keep this narrow sidecar beside that
        # directory, while snapshot_manifest.json remains the complete audit log.
        backtester_manifest = {
            "schema_version": 1,
            "files": [
                {
                    "path": Path(item["path"]).name,
                    "symbol": item["symbol"],
                    "sha256": item["sha256"],
                    "rows": item["rows"],
                    "max_materialized_time": item["end"],
                }
                for item in files
            ],
        }
        (temp_dir / "materialization_manifest.json").write_text(
            json.dumps(backtester_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temp_dir, output_dir)
        return snapshot_manifest
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--warmup-bars", type=int, default=288)
    parser.add_argument("--max-horizon", type=int, default=144)
    parser.add_argument("--max-symbols", type=int, default=215)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = materialize_snapshot(
            manifest_path=args.manifest,
            source_root=args.source_root,
            output_dir=args.output_dir,
            start=args.start,
            end=args.end,
            cutoff=args.cutoff,
            warmup_bars=args.warmup_bars,
            max_horizon=args.max_horizon,
            max_symbols=args.max_symbols,
        )
    except (SnapshotError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.output_dir), "counts": result["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
