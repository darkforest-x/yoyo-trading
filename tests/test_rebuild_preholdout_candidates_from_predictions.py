"""Contract tests for rebuilding pre-holdout artifacts without model inference."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tools.backtest_fixed_w10_cls_preholdout import (
    ATR_PCT_MIN,
    PROTOCOL,
    _event_id,
)
from tools.rebuild_preholdout_candidates_from_predictions import RebuildError, rebuild


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(parents=True)
    csv_path = snapshot / "BTC_USDT_SWAP.csv"
    times = pd.date_range("2026-03-30T00:00:00Z", periods=80, freq="15min")
    close = [100.0] * 80
    # First event has tiny causal ATR.  Volatility begins only afterwards so the
    # later event, fewer than 18 bars away, has a valid canonical ATR percentage.
    for index in range(21, 80):
        close[index] = 100.0 + (2.0 if index % 2 else -2.0)
    frame = pd.DataFrame({
        "open_time": times,
        "open": close,
        "high": [value + (0.01 if i <= 20 else 1.0) for i, value in enumerate(close)],
        "low": [value - (0.01 if i <= 20 else 1.0) for i, value in enumerate(close)],
        "close": close,
        "volume": [1.0] * 80,
    })
    frame.to_csv(csv_path, index=False)
    manifest = snapshot / "materialization_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "files": [{
            "path": csv_path.name,
            "symbol": "BTC_USDT_SWAP",
            "sha256": _sha(csv_path),
            "rows": len(frame),
            "max_materialized_time": times[-1].isoformat(),
        }],
    }), encoding="utf-8")

    source = tmp_path / "source_preholdout_run"
    source.mkdir()
    predictions = []
    for decision_i in (20, 30):
        decision = times[decision_i].isoformat()
        predictions.append({
            "protocol": PROTOCOL,
            "event_id": _event_id("BTC_USDT_SWAP", decision),
            "symbol": "BTC_USDT_SWAP",
            "decision_i": decision_i,
            "decision_time": decision,
            "window_start_i": decision_i - 9,
            "window_end_i": decision_i,
            "causal_window_start_time": times[decision_i - 9].isoformat(),
            "causal_window_end_time": decision,
            "p_signal": 0.8,
            "pred": "SIGNAL",
            "holdout_read": False,
            "future_data_in_causal_input": False,
        })
    predictions_path = source / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    summary = {
        "protocol": PROTOCOL,
        "evaluation_scope": "preholdout",
        "holdout_read": False,
        "future_data_in_causal_input": False,
        "renderer": "yoyo.datasets.legacy_gold_migration.renderer.render_w10",
        "renderer_overlay": False,
        "transform": "white_letterbox_full_frame",
        "imgsz": 960,
        "window_bars": 10,
        "weights_sha256": "a" * 64,
        "orders_placed": False,
        "promoted": False,
        "candidate_parameters": {
            "threshold": 0.5, "tp_atr": 5.0, "sl_atr": 2.0, "horizon_bars": 4,
        },
        "threshold": 0.5,
        "tp_atr": 5.0,
        "sl_atr": 2.0,
        "horizon_bars": 4,
        "n_predictions": 2,
        "optimization_export": {
            "max_horizon": 4,
            "min_score": 0.0,
            "thresholds": [0.0, 0.5],
            "inner_end_exclusive": "2026-03-31T07:30:00Z",
        },
        "outcome_scope": {
            "name": "optimization_inner",
            "decision_before": "2026-03-31T07:30:00Z",
            "label_end_lte": "2026-03-31T07:30:00Z",
        },
    }
    summary_path = source / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return {
        "source": source,
        "snapshot": snapshot,
        "manifest": manifest,
        "manifest_sha": _sha(manifest),
        "predictions": predictions_path,
        "summary": summary_path,
    }


def _run(fixture: dict[str, Path | str], out: Path) -> dict:
    return rebuild(
        source_run=fixture["source"],  # type: ignore[arg-type]
        snapshot_dir=fixture["snapshot"],  # type: ignore[arg-type]
        manifest=fixture["manifest"],  # type: ignore[arg-type]
        snapshot_manifest_sha256=str(fixture["manifest_sha"]),
        out_dir=out,
    )


def test_rebuild_applies_atr_floor_before_gap_and_writes_inner_only(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    out = tmp_path / "rebuilt"
    summary = _run(fixture, out)

    optimization = _jsonl(out / "optimization_inner.jsonl")
    source_predictions = _jsonl(fixture["predictions"])  # type: ignore[arg-type]
    assert [row["decision_time"] for row in optimization] == [source_predictions[1]["decision_time"]]
    assert optimization[0]["atr_pct"] >= ATR_PCT_MIN
    assert not (out / "optimization_sealed.jsonl").exists()
    assert set(path.name for path in out.iterdir()) == {
        "predictions.jsonl", "signals_raw.jsonl", "signals_dedup.jsonl",
        "stop_losses.jsonl", "l1_review_queue.jsonl", "l2_outcome_negatives.jsonl",
        "optimization_inner.jsonl", "summary.json",
    }
    assert (out / "predictions.jsonl").read_bytes() == Path(fixture["predictions"]).read_bytes()
    assert summary["atr_pct_min"] == ATR_PCT_MIN
    assert summary["rebuild_reason"] == "atr_floor_contract_fix"
    assert summary["sealed_read"] is False
    assert summary["holdout_rows_read"] == 0
    assert summary["future_rows_in_features"] == 0
    assert summary["future_rows_in_model_input"] == 0
    assert summary["auto_promote"] is False and summary["orders"] is False
    assert summary["source_predictions_sha256"] == _sha(Path(fixture["predictions"]))
    assert summary["source_summary_sha256"] == _sha(Path(fixture["summary"]))
    assert all(
        item["sha256"] == _sha(out / name) and item["rows"] == len(_jsonl(out / name))
        for name, item in summary["artifacts"].items()
    )


@pytest.mark.parametrize("target", ["predictions", "summary"])
def test_tampered_source_fails_closed(tmp_path: Path, target: str) -> None:
    fixture = _fixture(tmp_path)
    path = Path(fixture[target])
    if target == "predictions":
        rows = _jsonl(path)
        rows[0]["event_id"] = "tampered"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        match = "event_id mismatch"
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["imgsz"] = 640
        path.write_text(json.dumps(raw), encoding="utf-8")
        match = "imgsz"
    with pytest.raises(RebuildError, match=match):
        _run(fixture, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_snapshot_manifest_and_physical_tampering_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = Path(fixture["manifest"])
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["files"][0]["rows"] += 1
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RebuildError, match="manifest SHA256 mismatch"):
        _run(fixture, tmp_path / "manifest-tamper")

    fixture = _fixture(tmp_path / "physical")
    csv_path = Path(fixture["snapshot"]) / "BTC_USDT_SWAP.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _run(fixture, tmp_path / "physical-out")


def test_rejects_sealed_source_run_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    source = Path(fixture["source"])
    sealed_source = source.with_name("sealed_run")
    source.rename(sealed_source)
    fixture["source"] = sealed_source
    with pytest.raises(RebuildError, match="path"):
        _run(fixture, tmp_path / "rejected-path-out")


# Kept apart from the path-guard test above: `_safe_source_run` scans every
# component of the source path, including the ancestors pytest derives from the
# test function name.  A name carrying "sealed" or "holdout" would trip the path
# guard first and hide the declaration/interval guards this test is about.
def test_rejects_declared_read_flag_late_inner_and_snapshot(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "field")
    summary_path = Path(fixture["summary"])
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    raw["sealed_read"] = True
    summary_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RebuildError, match="sealed_read"):
        _run(fixture, tmp_path / "declared-field-out")

    fixture = _fixture(tmp_path / "late-inner")
    summary_path = Path(fixture["summary"])
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    for field in ("inner_end_exclusive",):
        raw["optimization_export"][field] = "2026-03-31T07:45:00Z"
    raw["outcome_scope"]["decision_before"] = "2026-03-31T07:45:00Z"
    raw["outcome_scope"]["label_end_lte"] = "2026-03-31T07:45:00Z"
    summary_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RebuildError, match="inner_end exceeds"):
        _run(fixture, tmp_path / "late-inner-out")

    fixture = _fixture(tmp_path / "late-snapshot")
    manifest = Path(fixture["manifest"])
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["files"][0]["max_materialized_time"] = "2026-03-31T07:30:00Z"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    fixture["manifest_sha"] = _sha(manifest)
    with pytest.raises(RebuildError, match="snapshot max time"):
        _run(fixture, tmp_path / "late-snapshot-out")


def test_refuses_overwrite_and_cleans_failed_atomic_directory(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        _run(fixture, out)
    assert list(out.iterdir()) == []


def test_cli_in_fresh_process_does_not_load_a_model(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    out = tmp_path / "fresh-process-output"
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "ultralytics.py").write_text(
        "raise AssertionError('model loading/import is forbidden during rebuild')\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(blocker), str(Path.cwd()), env.get("PYTHONPATH", "")))
    result = subprocess.run(
        [
            sys.executable,
            str(Path.cwd() / "tools/rebuild_preholdout_candidates_from_predictions.py"),
            "--source-run", str(fixture["source"]),
            "--snapshot-dir", str(fixture["snapshot"]),
            "--manifest", str(fixture["manifest"]),
            "--snapshot-manifest-sha256", str(fixture["manifest_sha"]),
            "--out-dir", str(out),
        ],
        cwd=Path.cwd(), env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_dir()
    assert json.loads(result.stdout)["out_dir"] == str(out)
