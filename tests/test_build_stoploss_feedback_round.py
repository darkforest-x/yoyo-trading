"""Contract tests for the inner-only stop-loss feedback materializer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.backtest_fixed_w10_cls_preholdout import load_preholdout_csv
import tools.build_stoploss_feedback_round as feedback
from yoyo.contracts.outcomes import ATR_ELIGIBILITY_PROTOCOL, ATR_PCT_MIN
from tools.build_stoploss_feedback_round import (
    FEATURE_COLUMNS,
    MINING_END_LIMIT,
    _causal_features_and_window,
    blind_id,
    build_feedback_round,
)


BAR = pd.Timedelta(minutes=15)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _path(decision: str, outcome: str, horizon: int = 4) -> list[dict]:
    start = pd.Timestamp(decision) + BAR
    rows = []
    for offset in range(horizon):
        high, low = 100.5, 99.5
        if offset == 0 and outcome == "sl":
            high = 102.1
        elif offset == 0 and outcome == "sl_ambiguous":
            high, low = 102.1, 94.9
        elif offset == 0 and outcome == "tp":
            low = 94.9
        rows.append({
            "time": (start + offset * BAR).isoformat(),
            "open": 100.0, "high": high, "low": low, "close": 100.0,
        })
    return rows


def _candidate(event_id: str, decision: str, score: float, outcome: str, *, symbol: str = "BTC-USDT-SWAP") -> dict:
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
        "future_review_only": True,
        "source_partition": "inner",
        "causal_metadata": {"visible_end_time": decision},
        "future_path": _path(decision, outcome),
    }


def _snapshot(tmp_path: Path, *, tamper: bool = False) -> tuple[Path, Path, pd.DataFrame]:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    times = pd.date_range("2026-01-01T00:00:00Z", periods=380, freq=BAR)
    x = np.arange(len(times), dtype=float)
    close = 100.0 + 0.015 * x + 0.8 * np.sin(x / 7.0)
    frame = pd.DataFrame({
        "open_time": times,
        "open": close - 0.08 * np.cos(x / 3.0),
        "high": close + 0.45,
        "low": close - 0.45,
        "close": close,
        "volume": 10.0 + (x % 11),
    })
    csv_path = snapshot / "BTC-USDT-SWAP.csv"
    frame.to_csv(csv_path, index=False)
    declaration = {
        "path": csv_path.name,
        "symbol": "BTC-USDT-SWAP",
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "rows": len(frame),
        "max_materialized_time": times[-1].isoformat(),
    }
    manifest = tmp_path / "materialization_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "files": [declaration]}), encoding="utf-8")
    if tamper:
        csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    return snapshot, manifest, frame


def _selection(path: Path, candidates: Path, **extra) -> Path:
    payload = {
        "schema_version": 1,
        "sealed_read": False,
        "selected_config": {"threshold": .6, "tp_atr": 5.0, "sl_atr": 2.0, "horizon": 4},
        "selected_config_key": "threshold=0.6,tp=5,sl=2,horizon=4",
        "source_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
        **extra,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, Path | list[dict]]:
    snapshot, manifest, raw_frame = _snapshot(tmp_path)
    # Index 300 is selected.  Index 317 is above threshold but inside the
    # frozen 18-bar gap and must not replace the chronologically earlier event.
    decisions = [pd.Timestamp(raw_frame["open_time"].iloc[i]).isoformat()
                 for i in (280, 300, 317, 318, 336)]
    rows = [
        _candidate("below", decisions[0], .59, "sl"),
        _candidate("stop", decisions[1], .70, "sl"),
        _candidate("gap-blocked", decisions[2], .99, "sl"),
        _candidate("take-profit", decisions[3], .80, "tp"),
        _candidate("ambiguous-stop", decisions[4], .75, "sl_ambiguous"),
    ]
    candidates = tmp_path / "optimization_inner.jsonl"
    _write_jsonl(candidates, rows)
    selection = _selection(tmp_path / "config_selection.json", candidates)
    return {"snapshot": snapshot, "manifest": manifest, "frame": raw_frame,
            "rows": rows, "candidates": candidates, "selection": selection}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _build(data: dict, out: Path, registry: Path | None = None) -> dict:
    return build_feedback_round(
        candidates_path=data["candidates"],
        selected_config_path=data["selection"],
        snapshot_dir=data["snapshot"],
        materialization_manifest=data["manifest"],
        out_dir=out,
        global_event_registry=registry,
    )


def test_stop_loss_is_l2_negative_but_blind_l1_review_required(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    out = tmp_path / "round"
    summary = _build(data, out)

    stops = _read_jsonl(out / "stop_losses.jsonl")
    l2 = _read_jsonl(out / "l2_outcome_negatives.jsonl")
    review = _read_jsonl(out / "l1_blind_review" / "review_manifest.jsonl")
    private = _read_jsonl(out / "lineage_private.jsonl")
    assert [row["event_id"] for row in stops] == ["stop", "ambiguous-stop"]
    assert [row["event_id"] for row in l2] == ["stop", "ambiguous-stop"]
    assert all(row["l2_target"] == 0 and row["training_eligible"] is True for row in l2)
    assert all(row["training_scope"] == "l2_only" for row in l2)
    assert all(row["suggested_label"] == "REVIEW_REQUIRED" for row in review)
    assert all(row["training_eligible"] is False for row in review)
    assert all(row["future_review_only"] is True for row in private)
    assert all(row["automatic_l1_label"] is None for row in stops)
    assert summary["counts"]["threshold_gap_selected"] == 3
    assert summary["counts"]["selected_outcomes"] == {"sl": 1, "tp": 1, "sl_ambiguous": 1}
    assert summary["counts"]["stop_losses_materialized"] == 2
    assert summary["holdout_rows_read"] == 0
    assert summary["future_rows_in_causal_input"] == 0
    assert summary["auto_promote"] is False


def test_review_manifest_and_filename_do_not_leak_private_fields(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    out = tmp_path / "round"
    _build(data, out)
    review = _read_jsonl(out / "l1_blind_review" / "review_manifest.jsonl")[0]
    forbidden = {"event_id", "symbol", "decision_time", "outcome", "p_signal", "model_score",
                 "gross_ret", "net_maker", "net_taker", "future_path"}
    assert forbidden.isdisjoint(review)
    assert set(review) == {
        "blind_id", "image_path", "image_sha256", "class_status", "box_status",
        "training_eligible", "suggested_label", "future_used_in_training_image", "holdout_read",
    }
    assert "BTC" not in review["blind_id"] and "2026" not in review["blind_id"]
    assert review["image_path"] == f"images/{review['blind_id']}.png"
    image = out / "l1_blind_review" / review["image_path"]
    assert hashlib.sha256(image.read_bytes()).hexdigest() == review["image_sha256"]


def test_l2_features_are_frozen_28d_short_aligned_and_causal(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    out = tmp_path / "round"
    _build(data, out)
    l2 = _read_jsonl(out / "l2_outcome_negatives.jsonl")[0]
    assert l2["feature_columns"] == FEATURE_COLUMNS
    assert len(l2["features"]) == len(FEATURE_COLUMNS) == 28
    assert set(l2["features"]) == set(FEATURE_COLUMNS)
    assert all(np.isfinite(value) for value in l2["features"].values())
    assert l2["feature_semantics"] == "side_aligned_v1"
    assert l2["feature_as_of"] == l2["decision_time"]
    assert l2["future_data_in_causal_input"] is False

    declaration = json.loads(data["manifest"].read_text(encoding="utf-8"))["files"][0]
    loaded = load_preholdout_csv(data["snapshot"] / declaration["path"], declaration)
    expected, window = _causal_features_and_window(
        loaded, "BTC-USDT-SWAP", pd.Timestamp(l2["decision_time"])
    )
    assert l2["features"] == pytest.approx(expected)
    assert pd.Timestamp(window["open_time"].max()) == pd.Timestamp(l2["decision_time"])


def test_blind_id_is_stable_and_opaque() -> None:
    first = blind_id("a" * 64, "event/BTC-USDT-SWAP/2026-03-01/sl/0.99")
    assert first == blind_id("a" * 64, "event/BTC-USDT-SWAP/2026-03-01/sl/0.99")
    assert first != blind_id("b" * 64, "event/BTC-USDT-SWAP/2026-03-01/sl/0.99")
    for leaked in ("BTC", "2026", "sl", "0.99", "event"):
        assert leaked not in first


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(decision_time="2026-04-02T00:00:00Z",
                                causal_metadata={"visible_end_time": "2026-04-02T00:00:00Z"},
                                future_path=_path("2026-04-02T00:00:00Z", "sl")), "sealed interval"),
        (lambda row: row.update(decision_time="2026-05-04T00:00:00Z",
                                causal_metadata={"visible_end_time": "2026-05-04T00:00:00Z"},
                                future_path=_path("2026-05-04T00:00:00Z", "sl")), "sealed interval"),
    ],
)
def test_sealed_and_holdout_decisions_are_rejected_before_snapshot(
    tmp_path: Path, mutate, message: str
) -> None:
    decision = "2026-01-04T03:00:00Z"
    row = _candidate("bad", decision, .8, "sl")
    mutate(row)
    candidates = tmp_path / "optimization_inner.jsonl"
    _write_jsonl(candidates, [row])
    selection = _selection(tmp_path / "config_selection.json", candidates)
    with pytest.raises(ValueError, match=message):
        build_feedback_round(
            candidates_path=candidates, selected_config_path=selection,
            snapshot_dir=tmp_path / "unused", materialization_manifest=tmp_path / "unused.json",
            out_dir=tmp_path / "out",
        )


def test_sealed_declaration_and_label_crossing_are_rejected(tmp_path: Path) -> None:
    decision = (MINING_END_LIMIT - 2 * BAR).isoformat()
    row = _candidate("cross", decision, .8, "sl")
    candidates = tmp_path / "optimization_inner.jsonl"
    _write_jsonl(candidates, [row])
    selection = _selection(tmp_path / "config_selection.json", candidates)
    with pytest.raises(ValueError, match="label_end exceeds mining_end"):
        build_feedback_round(
            candidates_path=candidates, selected_config_path=selection,
            snapshot_dir=tmp_path / "unused", materialization_manifest=tmp_path / "unused.json",
            out_dir=tmp_path / "out",
        )
    selection = _selection(tmp_path / "sealed_config.json", candidates, sealed=True)
    with pytest.raises(ValueError, match="unsealed inner-search"):
        build_feedback_round(
            candidates_path=candidates, selected_config_path=selection,
            snapshot_dir=tmp_path / "unused", materialization_manifest=tmp_path / "unused.json",
            out_dir=tmp_path / "out2",
        )


def test_source_and_snapshot_sha_gates_fail_closed(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    data["candidates"].write_text(data["candidates"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate source SHA mismatch"):
        _build(data, tmp_path / "source-bad")

    other = tmp_path / "other"
    other.mkdir()
    snapshot, manifest, frame = _snapshot(other, tamper=True)
    candidates = other / "optimization_inner.jsonl"
    decision = pd.Timestamp(frame["open_time"].iloc[300]).isoformat()
    _write_jsonl(candidates, [_candidate("stop", decision, .8, "sl")])
    selection = _selection(other / "config_selection.json", candidates)
    bad = {"snapshot": snapshot, "manifest": manifest, "candidates": candidates, "selection": selection}
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _build(bad, other / "manifest-bad")


def test_duplicate_overwrite_and_registry_replay_are_audited(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    registry = tmp_path / "event_registry.json"
    first = tmp_path / "round-one"
    _build(data, first, registry)
    with pytest.raises(FileExistsError, match="overwrite"):
        _build(data, first, registry)

    second = tmp_path / "round-two"
    summary = _build(data, second, registry)
    assert summary["counts"]["registry_skipped"] == 2
    assert summary["counts"]["stop_losses_materialized"] == 0
    assert _read_jsonl(second / "stop_losses.jsonl") == []
    assert set(json.loads(registry.read_text(encoding="utf-8"))["events"]) == {
        "stop", "ambiguous-stop"
    }

    duplicate = list(data["rows"])
    duplicate.append({**duplicate[0], "event_id": "duplicate-identity"})
    duplicate_path = tmp_path / "duplicate.jsonl"
    _write_jsonl(duplicate_path, duplicate)
    duplicate_selection = _selection(tmp_path / "duplicate_selection.json", duplicate_path)
    duplicate_data = {**data, "candidates": duplicate_path, "selection": duplicate_selection}
    with pytest.raises(ValueError, match="duplicate event identity"):
        _build(duplicate_data, tmp_path / "duplicate-out")


def test_registry_pending_claim_survives_artifact_failure_and_blocks_replay(
    tmp_path: Path, monkeypatch
) -> None:
    data = _fixture(tmp_path)
    registry = tmp_path / "event_registry.json"

    def fail_summary_write(path: Path, payload: dict) -> None:
        if path.name == "summary.json":
            raise OSError("simulated summary write failure")
        feedback._write_json.__wrapped__(path, payload)  # type: ignore[attr-defined]

    real_write_json = feedback._write_json
    fail_summary_write.__wrapped__ = real_write_json  # type: ignore[attr-defined]
    monkeypatch.setattr(feedback, "_write_json", fail_summary_write)
    with pytest.raises(OSError, match="simulated summary"):
        _build(data, tmp_path / "failed-round", registry)

    assert not (tmp_path / "failed-round").exists()
    events = json.loads(registry.read_text(encoding="utf-8"))["events"]
    assert {event_id: item["status"] for event_id, item in events.items()} == {
        "stop": "pending",
        "ambiguous-stop": "pending",
    }

    monkeypatch.setattr(feedback, "_write_json", real_write_json)
    summary = _build(data, tmp_path / "retry-round", registry)
    assert summary["counts"]["registry_skipped"] == 2
    assert summary["counts"]["stop_losses_materialized"] == 0
    assert _read_jsonl(tmp_path / "retry-round" / "stop_losses.jsonl") == []
    events_after = json.loads(registry.read_text(encoding="utf-8"))["events"]
    assert {event_id: item["status"] for event_id, item in events_after.items()} == {
        "stop": "pending",
        "ambiguous-stop": "pending",
    }


def test_registry_pending_claim_survives_complete_write_failure(
    tmp_path: Path, monkeypatch
) -> None:
    data = _fixture(tmp_path)
    registry = tmp_path / "event_registry.json"
    real_write_atomic = feedback._write_json_atomic
    calls = {"n": 0}

    def fail_complete_write(path: Path, payload: dict) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated registry complete failure")
        real_write_atomic(path, payload)

    monkeypatch.setattr(feedback, "_write_json_atomic", fail_complete_write)
    with pytest.raises(OSError, match="simulated registry complete"):
        _build(data, tmp_path / "published-but-pending", registry)

    assert (tmp_path / "published-but-pending" / "summary.json").is_file()
    events = json.loads(registry.read_text(encoding="utf-8"))["events"]
    assert {event_id: item["status"] for event_id, item in events.items()} == {
        "stop": "pending",
        "ambiguous-stop": "pending",
    }

    monkeypatch.setattr(feedback, "_write_json_atomic", real_write_atomic)
    summary = _build(data, tmp_path / "second-round", registry)
    assert summary["counts"]["registry_skipped"] == 2
    assert summary["counts"]["stop_losses_materialized"] == 0


def test_feedback_registry_lock_is_exclusive_and_writes_no_output(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    registry = tmp_path / "event_registry.json"
    registry.with_name(registry.name + ".lock").write_text("held", encoding="utf-8")

    with pytest.raises(ValueError, match="registry is locked"):
        _build(data, tmp_path / "locked-round", registry)
    assert not (tmp_path / "locked-round").exists()
    assert not registry.exists()
