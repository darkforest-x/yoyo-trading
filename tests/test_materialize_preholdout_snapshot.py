"""Safety tests for the bounded pre-holdout CSV materializer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from tools.materialize_preholdout_snapshot import SnapshotError, materialize_snapshot


def _write_source(root: Path, symbol: str = "ETH_USDT_SWAP", rows: int = 12) -> Path:
    path = root / "data" / "kline_fetched" / f"okx_{symbol}_15m_{rows}.csv"
    path.parent.mkdir(parents=True)
    times = pd.date_range("2026-05-03T21:00:00Z", periods=rows, freq="15min")
    # The final row is the explicit post-cutoff sentinel in the normal test.
    pd.DataFrame(
        {"open_time": times, "open": range(rows), "high": range(rows), "low": range(rows), "close": range(rows)}
    ).to_csv(path, index=False)
    return path


def _write_manifest(root: Path, rows: list[dict]) -> Path:
    path = root / "positive_manifest.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _row(symbol: str = "ETH_USDT_SWAP", *, source: str = "data/kline_fetched/okx_ETH_USDT_SWAP_15m_12.csv", end_time: str = "2026-05-03T21:30:00Z", win_end: int = 2) -> dict:
    return {"sample_id": "reference", "symbol": symbol, "source_csv": source, "win_end": win_end, "end_time": end_time}


def test_materializer_never_requests_or_writes_post_cutoff_sentinel(tmp_path, monkeypatch) -> None:
    _write_source(tmp_path)
    manifest_path = _write_manifest(tmp_path, [_row()])
    output = tmp_path / "snapshot"
    calls: list[tuple[Path, int | None]] = []
    actual_read_csv = pd.read_csv

    def tracked_read_csv(path, *args, **kwargs):
        calls.append((Path(path), kwargs.get("nrows")))
        return actual_read_csv(path, *args, **kwargs)

    monkeypatch.setattr("tools.materialize_preholdout_snapshot.pd.read_csv", tracked_read_csv)
    result = materialize_snapshot(
        manifest_path=manifest_path,
        source_root=tmp_path,
        output_dir=output,
        start="2026-05-03T21:30:00Z",
        end="2026-05-03T23:45:00Z",
        cutoff="2026-05-04T00:00:00Z",
        warmup_bars=2,
        max_horizon=4,
    )

    assert calls == [(tmp_path / "data/kline_fetched/okx_ETH_USDT_SWAP_15m_12.csv", 12)]
    written = pd.read_csv(output / "kline_snapshot" / "ETH_USDT_SWAP.csv")
    assert pd.to_datetime(written["open_time"], utc=True).max() < pd.Timestamp("2026-05-04T00:00:00Z")
    item = result["files"][0]
    assert item["source_prefix_rows_requested"] == 12
    assert item["holdout_read"] is False
    compatible = json.loads((output / "materialization_manifest.json").read_text(encoding="utf-8"))
    assert compatible["files"] == [{
        "path": "ETH_USDT_SWAP.csv", "symbol": "ETH_USDT_SWAP", "sha256": item["sha256"],
        "rows": item["rows"], "max_materialized_time": item["end"],
    }]


def test_materializer_uses_bounded_prefix_before_a_real_cutoff_sentinel(tmp_path, monkeypatch) -> None:
    source = _write_source(tmp_path, rows=13)
    manifest_path = _write_manifest(tmp_path, [_row(source=source.relative_to(tmp_path).as_posix())])
    output = tmp_path / "snapshot"
    actual_read_csv = pd.read_csv
    nrows_seen: list[int | None] = []

    def tracked_read_csv(path, *args, **kwargs):
        nrows_seen.append(kwargs.get("nrows"))
        return actual_read_csv(path, *args, **kwargs)

    monkeypatch.setattr("tools.materialize_preholdout_snapshot.pd.read_csv", tracked_read_csv)
    materialize_snapshot(
        manifest_path=manifest_path, source_root=tmp_path, output_dir=output,
        start="2026-05-03T21:30:00Z", end="2026-05-03T23:45:00Z", warmup_bars=2, max_horizon=4,
    )
    # The cutoff sentinel is index 12 (00:00); prefix nrows=12 ends at index 11.
    assert nrows_seen == [12]
    assert pd.Timestamp(pd.read_csv(output / "kline_snapshot/ETH_USDT_SWAP.csv")["open_time"].iloc[-1]) < pd.Timestamp("2026-05-04T00:00:00Z")


def test_materializer_rejects_duplicate_source_for_one_symbol(tmp_path) -> None:
    manifest_path = _write_manifest(tmp_path, [_row(), _row(source="data/kline_fetched/other.csv")])
    with pytest.raises(SnapshotError, match="multiple source_csv"):
        materialize_snapshot(manifest_path=manifest_path, source_root=tmp_path, output_dir=tmp_path / "snapshot")


def test_materializer_rejects_misaligned_reference_before_reading_csv(tmp_path, monkeypatch) -> None:
    manifest_path = _write_manifest(tmp_path, [_row(end_time="2026-05-03T21:31:00Z")])
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("source CSV must not be read")

    monkeypatch.setattr("tools.materialize_preholdout_snapshot.pd.read_csv", forbidden)
    with pytest.raises(SnapshotError, match="not aligned"):
        materialize_snapshot(manifest_path=manifest_path, source_root=tmp_path, output_dir=tmp_path / "snapshot")
    assert not called


def test_target_before_manifest_reference_skips_without_reading_source(tmp_path, monkeypatch) -> None:
    manifest_path = _write_manifest(tmp_path, [_row()])

    def forbidden(*args, **kwargs):
        raise AssertionError("an earlier target must skip before source CSV reading")

    monkeypatch.setattr("tools.materialize_preholdout_snapshot.pd.read_csv", forbidden)
    result = materialize_snapshot(
        manifest_path=manifest_path,
        source_root=tmp_path,
        output_dir=tmp_path / "snapshot",
        start="2026-05-03T20:00:00Z",
        end="2026-05-03T21:00:00Z",
    )
    assert result["counts"]["materialized"] == 0
    assert result["skips"] == [{"symbol": "ETH_USDT_SWAP", "reason": "requested_end_before_source_reference"}]


def test_materializer_refuses_existing_directory(tmp_path) -> None:
    output = tmp_path / "snapshot"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_snapshot(manifest_path=tmp_path / "does-not-matter", source_root=tmp_path, output_dir=output)


def test_manifest_sha_and_last_safe_time_are_recorded(tmp_path) -> None:
    source = _write_source(tmp_path, rows=13)
    manifest_path = _write_manifest(tmp_path, [_row(source=source.relative_to(tmp_path).as_posix())])
    output = tmp_path / "snapshot"
    result = materialize_snapshot(
        manifest_path=manifest_path, source_root=tmp_path, output_dir=output,
        start="2026-05-03T21:30:00Z", end="2026-05-03T23:45:00Z", warmup_bars=2, max_horizon=4,
    )
    persisted = json.loads((output / "snapshot_manifest.json").read_text(encoding="utf-8"))
    assert persisted == result
    assert result["source_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    item = result["files"][0]
    assert result["content_shas"][item["path"]] == item["sha256"]
    assert result["last_safe_decision_time"] == "2026-05-03T22:45:00Z"
    assert result["future_data_role"] == "outcome_label_only_after_decision"
