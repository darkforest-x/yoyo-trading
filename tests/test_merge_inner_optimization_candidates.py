"""Strict safety and determinism tests for INNER candidate merging."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import tools.merge_inner_optimization_candidates as merger
from yoyo.contracts.outcomes import ATR_ELIGIBILITY_PROTOCOL, ATR_PCT_MIN

BAR = pd.Timedelta(minutes=15)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(event: str, symbol: str, decision: str, *, score: float = 0.8) -> dict:
    stamp = pd.Timestamp(decision)
    path = []
    for offset in (1, 2):
        path.append({
            "time": (stamp + offset * BAR).isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
        })
    return {
        "protocol": "fixed_w10_core4_confirm1_v1_cls_preholdout",
        "event_id": event,
        "symbol": symbol,
        "decision_time": stamp.isoformat(),
        "p_signal": score,
        "gold": False,
        "selected_at_thresholds": [0.5],
        "entry_price": 100.0,
        "atr14": 1.0,
        "atr_pct": 0.01,
        "atr_pct_min": ATR_PCT_MIN,
        "atr_eligibility_protocol": ATR_ELIGIBILITY_PROTOCOL,
        "partition": "inner",
        "source_partition": "inner",
        "holdout_read": False,
        "future_data_in_causal_input": False,
        "future_used_in_training_image": False,
        "future_review_only": True,
        "causal_metadata": {"visible_end_time": stamp.isoformat()},
        "future_path": path,
    }


def _native_backtester_row(event: str, symbol: str, decision: str) -> dict:
    """Mirror the real optimization_inner row schema (no partition fields)."""
    row = _row(event, symbol, decision)
    row.pop("partition")
    row.pop("source_partition")
    return row


def _write_source(
    root: Path,
    name: str,
    rows: list[dict],
    symbols: list[str],
    *,
    weights: str = "a" * 64,
    thresholds: list[float] | None = None,
    snapshot_sha: str | None = "b" * 64,
    snapshot_contract: dict | None = None,
    renderer: str = "yoyo.datasets.legacy_gold_migration.renderer.render_w10",
) -> tuple[Path, Path]:
    directory = root / name
    directory.mkdir()
    source = directory / "optimization_inner.jsonl"
    source.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    grid = thresholds or [0.0, 0.5]
    summary = {
        "holdout_read": False,
        "future_data_in_causal_input": False,
        "weights_sha256": weights,
        "renderer": renderer,
        "transform": "white_letterbox_full_frame",
        "imgsz": 960,
        "symbol_sampling": {
            "sampling_group": name,
            "seed": "confirm",
            "selected_symbols": symbols,
        },
        "optimization_export": {
            "max_horizon": 2,
            "min_score": grid[0],
            "thresholds": grid,
        },
        "outcome_scope": {
            "name": "optimization_inner",
            "decision_before": "2026-03-31T07:30:00Z",
            "sealed_outcomes_resolved": False,
        },
        "optimization_exports": {
            "inner": {
                "file": source.name,
                "sha256": _sha(source),
                "rows": len(rows),
            },
            "sealed": {"file": "optimization_sealed.jsonl"},
        },
    }
    if snapshot_sha is not None:
        summary["snapshot_manifest_sha256"] = snapshot_sha
    if snapshot_contract is not None:
        summary["snapshot_contract"] = snapshot_contract
    summary_path = directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return source, summary_path


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    late = _write_source(
        tmp_path,
        "group_b",
        [_row("event-b", "ETH-USDT-SWAP", "2026-02-01T00:00:00Z")],
        ["ETH-USDT-SWAP"],
    )[0]
    early = _write_source(
        tmp_path,
        "group_a",
        [_row("event-a", "BTC-USDT-SWAP", "2026-01-01T00:00:00Z")],
        ["BTC-USDT-SWAP"],
    )[0]
    return late, early


def test_deterministic_success_preserves_future_path_and_cli(tmp_path: Path) -> None:
    late, early = _pair(tmp_path)
    original_paths = {
        row["event_id"]: row["future_path"]
        for source in (late, early)
        for row in [json.loads(source.read_text(encoding="utf-8"))]
    }
    first = tmp_path / "merged-1"
    second = tmp_path / "merged-2"
    assert merger.main(["--source", str(late), "--source", str(early), "--out-dir", str(first)]) == 0
    merger.merge_sources([early, late], second)

    assert (first / "candidates.jsonl").read_bytes() == (second / "candidates.jsonl").read_bytes()
    rows = [json.loads(line) for line in (first / "candidates.jsonl").read_text().splitlines()]
    assert [row["event_id"] for row in rows] == ["event-a", "event-b"]
    assert {row["event_id"]: row["future_path"] for row in rows} == original_paths
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["combined"]["sha256"] == _sha(first / "candidates.jsonl")
    assert manifest["combined"]["rows"] == 2
    assert manifest["safety"] == {
        "auto_promote": False,
        "decision_before": "2026-04-02T00:00:00+00:00",
        "future_path_role": "label_only_unchanged",
        "future_rows_in_causal_input": 0,
        "holdout_rows_read": 0,
        "holdout_start": "2026-05-04T00:00:00+00:00",
        "orders_placed": False,
        "outcome_scope": "optimization_inner",
        "sealed_outcomes_resolved": False,
        "sealed_rows": 0,
    }
    assert all(Path(item["path"]).is_absolute() for item in manifest["sources"])
    assert all(item["summary_sha256"] for item in manifest["sources"])


def test_native_backtester_schema_without_partition_is_accepted_and_explicit_noninner_rejected(
    tmp_path: Path,
) -> None:
    first = _write_source(
        tmp_path, "native_a",
        [_native_backtester_row("native-a", "BTC", "2026-01-01")], ["BTC"],
    )[0]
    second = _write_source(
        tmp_path, "native_b",
        [_native_backtester_row("native-b", "ETH", "2026-02-01")], ["ETH"],
    )[0]
    result = merger.merge_sources([first, second], tmp_path / "accepted")
    assert result["combined_rows"] == 2

    cases = (
        {"partition": "sealed"},
        {"source_partition": "other"},
        {"sealed": True},
        {"sealed_read": True},
    )
    for index, mutation in enumerate(cases):
        unsafe = _native_backtester_row(f"unsafe-{index}", "SOL", "2026-03-01")
        unsafe.update(mutation)
        source = _write_source(
            tmp_path, f"unsafe_{index}", [unsafe], ["SOL"],
        )[0]
        with pytest.raises(ValueError, match="partition|sealed"):
            merger.merge_sources([first, source], tmp_path / f"rejected-{index}")


def test_source_sha_mismatch_fails_before_output(tmp_path: Path) -> None:
    first, second = _pair(tmp_path)
    summary = first.parent / "summary.json"
    payload = json.loads(summary.read_text())
    payload["optimization_exports"]["inner"]["sha256"] = "0" * 64
    summary.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        merger.merge_sources([first, second], out)
    assert not out.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(partition="sealed", source_partition="sealed"), "inner/mining"),
        (lambda row: row.update(holdout_read=True), "holdout_read"),
        (lambda row: row.update(future_data_in_causal_input=True), "future data"),
        (lambda row: row.update(decision_time="2026-04-02T00:00:00Z",
                                causal_metadata={"visible_end_time": "2026-04-02T00:00:00Z"}),
         "optimization_sealed"),
        (lambda row: row["future_path"][-1].update(time="2026-05-04T00:00:00Z"), "future_path reaches holdout"),
        (lambda row: row["causal_metadata"].update(visible_end_time="2026-02-01T00:15:00Z"),
         "beyond decision"),
        (lambda row: row.update(selected_at_thresholds=[0.4]), "outside export grid"),
    ],
)
def test_sealed_holdout_or_future_causal_input_rejected(
    tmp_path: Path, mutation, message: str,
) -> None:
    first, second = _pair(tmp_path)
    row = json.loads(first.read_text())
    mutation(row)
    first.write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary = first.parent / "summary.json"
    payload = json.loads(summary.read_text())
    payload["optimization_exports"]["inner"]["sha256"] = _sha(first)
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        merger.merge_sources([first, second], tmp_path / "out")


@pytest.mark.parametrize("same_event", [True, False])
def test_global_duplicate_event_or_identity_rejected(tmp_path: Path, same_event: bool) -> None:
    first = _write_source(
        tmp_path, "one", [_row("shared" if same_event else "one", "BTC", "2026-01-01")], ["BTC"]
    )[0]
    second_symbol = "ETH" if same_event else "BTC"
    second = _write_source(
        tmp_path, "two", [_row("shared" if same_event else "two", second_symbol, "2026-01-01")],
        [second_symbol],
    )[0]
    message = "duplicate event_id" if same_event else r"duplicate symbol\+decision"
    with pytest.raises(ValueError, match=message):
        merger.merge_sources([first, second], tmp_path / "out")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("weights", "incompatible weights"),
        ("renderer", "incompatible weights"),
        ("grid", "incompatible weights"),
        ("snapshot", "incompatible snapshot"),
        ("symbols", "overlap"),
    ],
)
def test_incompatible_lineage_grid_or_sampling_rejected(
    tmp_path: Path, change: str, message: str,
) -> None:
    first = _write_source(tmp_path, "one", [_row("one", "BTC", "2026-01-01")], ["BTC"])[0]
    kwargs = {}
    symbol = "ETH"
    if change == "weights":
        kwargs["weights"] = "c" * 64
    elif change == "renderer":
        kwargs["renderer"] = "different.renderer"
    elif change == "grid":
        kwargs["thresholds"] = [0.0, 0.4]
    elif change == "snapshot":
        kwargs["snapshot_sha"] = "d" * 64
    elif change == "symbols":
        symbol = "BTC"
    second = _write_source(
        tmp_path, "two", [_row("two", symbol, "2026-02-01")], [symbol], **kwargs
    )[0]
    with pytest.raises(ValueError, match=message):
        merger.merge_sources([first, second], tmp_path / "out")


def test_different_snapshot_sha_allowed_only_with_compatible_contract(tmp_path: Path) -> None:
    contract = {"cutoff": "2026-05-04", "schema": 1}
    first = _write_source(
        tmp_path, "one", [_row("one", "BTC", "2026-01-01")], ["BTC"],
        snapshot_sha="a" * 64, snapshot_contract=contract,
    )[0]
    second = _write_source(
        tmp_path, "two", [_row("two", "ETH", "2026-02-01")], ["ETH"],
        snapshot_sha="b" * 64, snapshot_contract=contract,
    )[0]
    result = merger.merge_sources([first, second], tmp_path / "out")
    assert result["snapshot_compatibility"] == {
        "same_manifest_sha256": False,
        "same_contract": True,
        "manifest_sha256": None,
    }


def test_overwrite_and_failed_publish_leave_no_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _pair(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep"
    marker.write_text("mine", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        merger.merge_sources([first, second], existing)
    assert marker.read_text() == "mine"

    target = tmp_path / "atomic"

    def fail_replace(_source, _target):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(merger.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publish failure"):
        merger.merge_sources([first, second], target)
    assert not target.exists()
    assert not list(tmp_path.glob(".atomic.*"))
