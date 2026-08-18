"""Safety and schema tests for the INNER-only L2 feedback dataset builder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tools.build_l2_feedback_dataset as builder
from yoyo.contracts.outcomes import ATR_ELIGIBILITY_PROTOCOL, ATR_PCT_MIN
from tools.build_l2_feedback_dataset import (
    FEATURE_COLUMNS,
    build_l2_feedback_dataset,
    load_source_pair,
    main,
)

BAR = pd.Timedelta(minutes=15)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _future(decision: str, outcome: str, horizon: int = 4) -> list[dict]:
    start = pd.Timestamp(decision) + BAR
    output = []
    for offset in range(horizon):
        high, low = 100.5, 99.5
        if offset == 0 and outcome == "tp":
            low = 94.9
        elif offset == 0 and outcome == "sl":
            high = 102.1
        elif offset == 0 and outcome == "sl_ambiguous":
            high, low = 102.1, 94.9
        output.append({
            "time": (start + offset * BAR).isoformat(),
            "open": 100.0, "high": high, "low": low, "close": 100.0,
        })
    return output


def _candidate(event_id: str, decision: str, outcome: str, *, symbol: str = "BTC-USDT-SWAP",
               score: float = .8) -> dict:
    return {
        "event_id": event_id,
        "symbol": symbol,
        "decision_time": decision,
        "p_signal": score,
        "entry_price": 100.0,
        "atr14": 1.0,
        "atr_pct": 0.01,
        "atr_pct_min": ATR_PCT_MIN,
        "atr_eligibility_protocol": ATR_ELIGIBILITY_PROTOCOL,
        "source_partition": "inner",
        "future_review_only": True,
        "causal_metadata": {"visible_end_time": decision},
        "future_path": _future(decision, outcome),
    }


def _snapshot(tmp_path: Path, *, name: str = "snapshot", mutate_after: int | None = None,
              max_override: str | None = None) -> tuple[Path, Path, pd.DataFrame]:
    directory = tmp_path / name
    directory.mkdir()
    times = pd.date_range("2026-01-01T00:00:00Z", periods=800, freq=BAR)
    x = np.arange(len(times), dtype=float)
    close = 100 + .012 * x + .7 * np.sin(x / 8)
    if mutate_after is not None:
        close[mutate_after:] += np.linspace(50, 100, len(close) - mutate_after)
    frame = pd.DataFrame({
        "open_time": times,
        "open": close - .05,
        "high": close + .4,
        "low": close - .4,
        "close": close,
        "volume": 20 + (x % 13),
    })
    csv_path = directory / "BTC-USDT-SWAP.csv"
    frame.to_csv(csv_path, index=False)
    manifest = tmp_path / f"{name}_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "files": [{
            "path": csv_path.name,
            "symbol": "BTC-USDT-SWAP",
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "rows": len(frame),
            "max_materialized_time": max_override or times[-1].isoformat(),
        }],
    }), encoding="utf-8")
    return directory, manifest, frame


def _optimizer_pair(directory: Path, rows: list[dict], *, name: str = "optimization_inner.jsonl",
                    config: dict | None = None) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    candidates = directory / name
    _write_jsonl(candidates, rows)
    selected = config or {"threshold": .6, "tp_atr": 5.0, "sl_atr": 2.0, "horizon": 4}
    key = (f"threshold={selected['threshold']:g},tp={selected['tp_atr']:g},"
           f"sl={selected['sl_atr']:g},horizon={selected['horizon']}")
    prereg = {
        "schema_version": 1,
        "sealed_read": False,
        "side": "short",
        "entry_policy": "next_open",
        "same_bar_policy": "conservative_sl",
        "gap_policy": "barrier_price",
        "return_convention": "linear_short",
        "min_gap_bars": 18,
        "candidate_source": str(candidates.resolve()),
        "candidate_source_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
    }
    prereg_path = directory / "prereg.json"
    prereg_path.write_text(json.dumps(prereg, sort_keys=True), encoding="utf-8")
    selection = directory / "config_selection.json"
    selection.write_text(json.dumps({
        "schema_version": 1,
        "sealed_read": False,
        "selected_config": selected,
        "selected_config_key": key,
        "prereg_sha256": hashlib.sha256(prereg_path.read_bytes()).hexdigest(),
    }, sort_keys=True), encoding="utf-8")
    return candidates, selection


def _fixture(tmp_path: Path) -> dict:
    snapshot, manifest, frame = _snapshot(tmp_path)
    decisions = {index: pd.Timestamp(frame.open_time.iloc[index]).isoformat()
                 for index in (300, 330, 350, 550, 580, 610)}
    rows = [
        _candidate("train-tp", decisions[300], "tp"),
        _candidate("train-sl", decisions[330], "sl"),
        _candidate("purged", decisions[350], "tp"),
        _candidate("val-tp", decisions[550], "tp"),
        _candidate("val-sl-ambiguous", decisions[580], "sl_ambiguous"),
        _candidate("timeout", decisions[610], "timeout"),
    ]
    candidates, selection = _optimizer_pair(tmp_path / "optimizer", rows)
    return {
        "snapshot": snapshot, "manifest": manifest, "frame": frame, "rows": rows,
        "candidates": candidates, "selection": selection,
        "val_start": decisions[500] if 500 in decisions else pd.Timestamp(frame.open_time.iloc[500]).isoformat(),
    }


def _build(fixture: dict, out: Path, **kwargs) -> dict:
    return build_l2_feedback_dataset(
        sources=kwargs.pop("sources", [(fixture["candidates"], fixture["selection"])]),
        snapshot_dir=kwargs.pop("snapshot_dir", fixture["snapshot"]),
        materialization_manifest=kwargs.pop("materialization_manifest", fixture["manifest"]),
        out_dir=out,
        val_start=kwargs.pop("val_start", fixture["val_start"]),
        **kwargs,
    )


def test_builds_trainable_flat_schema_labels_and_reports_exclusions(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    out = tmp_path / "out"
    manifest = _build(fixture, out)
    data = pd.read_csv(out / "dataset.csv")

    assert list(data.event_id) == ["train-tp", "train-sl", "val-tp", "val-sl-ambiguous"]
    assert dict(zip(data.event_id, data.label)) == {
        "train-tp": 1, "train-sl": 0, "val-tp": 1, "val-sl-ambiguous": 0,
    }
    assert dict(zip(data.event_id, data.outcome))["val-sl-ambiguous"] == "sl_ambiguous"
    assert set(data.side) == {"short"}
    assert set(data.feature_semantics) == {"side_aligned_v1"}
    assert list(data.columns[11:39]) == FEATURE_COLUMNS
    assert np.isfinite(data[FEATURE_COLUMNS].to_numpy()).all()
    assert "future_path" not in data.columns
    assert manifest["counts"]["timeouts_excluded"] == 1
    assert manifest["counts"]["purge_rows_excluded"] == 1
    assert manifest["counts"]["resolved_outcomes"] == {
        "tp": 3, "sl": 1, "sl_ambiguous": 1, "timeout": 1,
    }
    assert manifest["class_balance"]["train"] == {"0": 1, "1": 1}
    assert manifest["class_balance"]["val"] == {"0": 1, "1": 1}
    assert manifest["holdout_rows_read"] == 0
    assert manifest["future_rows_in_features"] == 0
    assert manifest["auto_promote"] is False
    assert hashlib.sha256((out / "dataset.csv").read_bytes()).hexdigest() == manifest["dataset"]["sha256"]
    assert len(manifest["dataset"]["rows_sha256"]) == 64


def test_features_do_not_change_when_snapshot_future_changes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = tmp_path / "first"
    _build(fixture, first)
    altered_snapshot, altered_manifest, _ = _snapshot(
        tmp_path, name="altered", mutate_after=611
    )
    second = tmp_path / "second"
    _build(fixture, second, snapshot_dir=altered_snapshot,
           materialization_manifest=altered_manifest)
    a = pd.read_csv(first / "dataset.csv").set_index("event_id")
    b = pd.read_csv(second / "dataset.csv").set_index("event_id")
    pd.testing.assert_frame_equal(a[FEATURE_COLUMNS], b[FEATURE_COLUMNS])
    assert (
        pd.to_datetime(a.feature_as_of, utc=True)
        == pd.to_datetime(a.signal_time, utc=True)
    ).all()


@pytest.mark.parametrize("duplicate_kind", ["event", "identity"])
def test_duplicate_event_and_identity_across_sources_are_refused(
    tmp_path: Path, duplicate_kind: str
) -> None:
    fixture = _fixture(tmp_path)
    original = fixture["rows"][0]
    if duplicate_kind == "identity":
        duplicate = dict(original, event_id="other-id")
    else:
        duplicate = _candidate(
            original["event_id"],
            pd.Timestamp(fixture["frame"].open_time.iloc[370]).isoformat(),
            "tp",
        )
    duplicate_rows = [duplicate]
    second = _optimizer_pair(tmp_path / "optimizer2", duplicate_rows)
    with pytest.raises(ValueError, match="duplicate event/identity across sources"):
        _build(fixture, tmp_path / "out", sources=[
            (fixture["candidates"], fixture["selection"]), second,
        ])


@pytest.mark.parametrize("mutation, match", [
    (lambda selection, prereg, candidates: selection.update(sealed_read=True), "sealed_read=false"),
    (lambda selection, prereg, candidates: selection.update(selected_config_key="wrong"),
     "selected_config_key"),
    (lambda selection, prereg, candidates: selection.update(prereg_sha256="0" * 64),
     "prereg_sha256"),
])
def test_selection_lineage_fails_closed(tmp_path: Path, mutation, match: str) -> None:
    fixture = _fixture(tmp_path)
    selection_path = fixture["selection"]
    prereg_path = selection_path.with_name("prereg.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    mutation(selection, prereg, fixture["candidates"])
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_source_pair(fixture["candidates"], selection_path)


def test_candidate_sha_mismatch_fails_before_candidate_jsonl_read(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    fixture["candidates"].write_text(
        fixture["candidates"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    called = False

    def forbidden(_path):
        nonlocal called
        called = True
        raise AssertionError("candidate JSONL was read")

    monkeypatch.setattr(builder, "read_jsonl", forbidden)
    with pytest.raises(ValueError, match="candidate_source_sha256 mismatch"):
        _build(fixture, tmp_path / "out")
    assert called is False


@pytest.mark.parametrize("field,value,match", [
    ("decision", "2026-04-02T00:00:00Z", "sealed interval"),
    ("future", "2026-05-04T00:00:00Z", "future_path reaches holdout"),
    ("sealed", True, "sealed/holdout-derived"),
])
def test_rejects_sealed_or_holdout_candidate_content(tmp_path: Path, field: str, value, match: str) -> None:
    fixture = _fixture(tmp_path)
    row = fixture["rows"][0]
    if field == "decision":
        row["decision_time"] = value
        row["causal_metadata"]["visible_end_time"] = value
        row["future_path"] = _future(value, "tp")
    elif field == "future":
        row["future_path"][0]["time"] = value
    else:
        row["sealed"] = value
    candidates, selection = _optimizer_pair(tmp_path / "bad", [row])
    with pytest.raises(ValueError, match=match):
        _build(fixture, tmp_path / "out", sources=[(candidates, selection)])


def test_rejects_candidate_marked_as_purge(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    row = fixture["rows"][0]
    row["purged"] = True
    candidates, selection = _optimizer_pair(tmp_path / "bad-purge-source", [row])
    with pytest.raises(ValueError, match="purge row"):
        _build(fixture, tmp_path / "out", sources=[(candidates, selection)])


def test_snapshot_rejects_sealed_max_time_before_materialization(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    raw = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    raw["files"][0]["max_materialized_time"] = "2026-04-02T00:00:00Z"
    fixture["manifest"].write_text(json.dumps(raw), encoding="utf-8")
    called = False

    def forbidden(*_args):
        nonlocal called
        called = True
        raise AssertionError("OHLCV was materialized")

    monkeypatch.setattr(builder, "load_preholdout_csv", forbidden)
    with pytest.raises(ValueError, match="snapshot is not inner-only"):
        _build(fixture, tmp_path / "out")
    assert called is False


def test_snapshot_sha_gate_rejects_tampering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    csv_path = fixture["snapshot"] / "BTC-USDT-SWAP.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _build(fixture, tmp_path / "out")


def test_selected_configs_must_match_across_sources(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    later = _candidate(
        "other-source", pd.Timestamp(fixture["frame"].open_time.iloc[390]).isoformat(), "tp"
    )
    second = _optimizer_pair(
        tmp_path / "optimizer2",
        [later],
        config={"threshold": .7, "tp_atr": 5.0, "sl_atr": 2.0, "horizon": 4},
    )
    with pytest.raises(ValueError, match="configs across sources must match exactly"):
        _build(fixture, tmp_path / "out", sources=[
            (fixture["candidates"], fixture["selection"]), second,
        ])


def test_time_split_requires_both_classes_and_frozen_purge(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="purge-bars is frozen"):
        _build(fixture, tmp_path / "bad-purge", purge_bars=10)
    with pytest.raises(ValueError, match="val split is empty"):
        _build(fixture, tmp_path / "bad-val", val_start="2026-03-01T00:00:00Z")


def test_atomic_fresh_output_and_cli_repeated_source(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    out = tmp_path / "out"
    argv = [
        "--source", f"{fixture['candidates']},{fixture['selection']}",
        "--snapshot-dir", str(fixture["snapshot"]),
        "--manifest", str(fixture["manifest"]),
        "--out-dir", str(out),
        "--val-start", fixture["val_start"],
    ]
    assert main(argv) == 0
    before = (out / "dataset.csv").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        main(argv)
    assert (out / "dataset.csv").read_bytes() == before
    assert not list(tmp_path.glob(".out.tmp-*"))


def test_failed_build_leaves_no_partial_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="val split is empty"):
        _build(fixture, out, val_start="2026-03-01T00:00:00Z")
    assert not out.exists()
    assert not list(tmp_path.glob(".out.tmp-*"))
