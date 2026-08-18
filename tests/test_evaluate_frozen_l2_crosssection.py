"""Safety contracts for frozen L2 cross-section confirmation."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tools.evaluate_frozen_l2_crosssection as cross
from tools.optimize_short_barrier_walkforward import SearchConfig
from yoyo.contracts.outcomes import ATR_ELIGIBILITY_PROTOCOL, ATR_PCT_MIN
from yoyo.layers.l2_judgment.features import FEATURE_COLUMNS

BAR = pd.Timedelta(minutes=15)
REAL_EXPANSION_GATE = Path(
    "/Users/zhangzc/fable-trading/experiments/"
    "fixed_w10_evolution_r5_l2_data_expansion_inner12_v1"
)
REAL_EXPANSION_LINEAGE_SHA256 = "5de01132ce9b3bdf80da6c17f8742c4e68ee043bf37baed2a2292ec5980f2e76"


class FakeBooster:
    def __init__(self, **_kwargs) -> None:
        pass

    def feature_name(self):
        return list(FEATURE_COLUMNS)

    def predict(self, frame, **_kwargs):
        return np.asarray([.9, .1, .8][:len(frame)], dtype=float)

    def current_iteration(self):
        return 7


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _future(decision: str, outcome: str) -> list[dict]:
    start = pd.Timestamp(decision) + BAR
    output = []
    for index in range(4):
        high, low = 100.5, 99.5
        if index == 0 and outcome == "tp":
            low = 94.5
        elif index == 0 and outcome == "sl":
            high = 102.5
        output.append({"time": (start + index * BAR).isoformat(), "open": 100.0,
                       "high": high, "low": low, "close": 100.0})
    return output


def _candidate(event: str, symbol: str, decision: str, outcome: str) -> dict:
    return {
        "event_id": event, "symbol": symbol, "decision_time": decision,
        "p_signal": .9, "entry_price": 100.0, "atr14": 1.0, "atr_pct": 0.01,
        "atr_pct_min": ATR_PCT_MIN,
        "atr_eligibility_protocol": ATR_ELIGIBILITY_PROTOCOL,
        "causal_metadata": {"visible_end_time": decision},
        "future_path": _future(decision, outcome), "future_review_only": True,
        "holdout_read": False, "future_data_in_causal_input": False,
        "selected_at_thresholds": [.6], "gold": False,
    }


def _fixture(tmp_path: Path, monkeypatch, *, independent_symbol: str = "CROSS_USDT_SWAP") -> dict:
    monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace(Booster=FakeBooster))
    # Physical training lineage followed from the frozen gate.
    training = tmp_path / "training"
    training.mkdir()
    dataset = training / "dataset.csv"
    pd.DataFrame({"symbol": ["TRAIN_USDT_SWAP", "TRAIN_USDT_SWAP"]}).to_csv(dataset, index=False)
    training_manifest = training / "manifest.json"
    _write(training_manifest, {
        "schema_version": 1,
        "dataset": {"path": "dataset.csv", "rows": 2, "sha256": _sha(dataset)},
        "selected_config_key": "threshold=0.6,tp=5,sl=2,horizon=4",
    })

    gate = tmp_path / "gate"
    gate.mkdir()
    (gate / "model.txt").write_text("fake model\n", encoding="utf-8")
    gate_optimizer = tmp_path / "gate-optimizer"
    gate_optimizer.mkdir()
    gate_candidates = gate_optimizer / "optimization_inner.jsonl"
    gate_candidates.write_text("{}\n", encoding="utf-8")
    gate_weights = gate_optimizer / "weights.pt"
    gate_weights.write_bytes(b"independent frozen L1 weights")
    _write(gate_optimizer / "summary.json", {
        "weights_sha256": _sha(gate_weights),
        "renderer": "yoyo.datasets.legacy_gold_migration.renderer.render_w10",
        "transform": "white_letterbox_full_frame", "imgsz": 960,
        "optimization_exports": {"inner": {"file": gate_candidates.name, "rows": 1,
                                                       "sha256": _sha(gate_candidates)}},
    })
    _write(gate_optimizer / "prereg.json", {
        "schema_version": 1, "sealed_read": False, "side": "short",
        "candidate_source": str(gate_candidates),
        "candidate_source_sha256": _sha(gate_candidates),
        "min_gap_bars": 18, "entry_policy": "next_open",
        "same_bar_policy": "conservative_sl", "gap_policy": "barrier_price",
        "return_convention": "linear_short",
    })
    _write(gate_optimizer / "selection.json", {
        "schema_version": 1, "sealed_read": False,
        "selected_config": {"threshold": .6, "tp_atr": 5.0, "sl_atr": 2.0, "horizon": 4},
        "selected_config_key": "threshold=0.6,tp=5,sl=2,horizon=4",
        "prereg_sha256": _sha(gate_optimizer / "prereg.json"),
    })
    monkeypatch.setattr(cross, "_validate_gate_candidate_source", lambda *_args: (
        SearchConfig(.6, 5.0, 2.0, 4), [{"symbol": "GATE_USDT_SWAP"}], {
            "candidates_sha256": _sha(gate_candidates),
            "selection_sha256": _sha(gate_optimizer / "selection.json"),
            "prereg_sha256": _sha(gate_optimizer / "prereg.json"),
        },
    ))
    gate_snapshot = tmp_path / "gate-snapshot-manifest.json"
    _write(gate_snapshot, {"schema_version": 1})
    gate_prereg = {
        "schema_version": 1, "protocol": "l2_walkforward_gate_v1",
        "candidate_source_sha256": _sha(gate_candidates),
        "selection_sha256": _sha(gate_optimizer / "selection.json"),
        "snapshot_manifest_sha256": _sha(gate_snapshot),
        "training_manifest_sha256": _sha(training_manifest),
        "selected_config": {"threshold": .6, "tp_atr": 5.0, "sl_atr": 2.0, "horizon": 4},
        "selected_config_key": "threshold=0.6,tp=5,sl=2,horizon=4",
        "purge_bars": 162, "holdout_rows_read": 0, "sealed_read": False,
        "auto_promote": False, "orders": False,
        "test_inspection_before_selection": False,
    }
    _write(gate / "prereg.json", gate_prereg)
    _write(gate / "selection.json", {
        "schema_version": 1, "protocol": "l2_walkforward_gate_v1",
        "selected_threshold": .5, "auto_promote": False,
        "test_metrics_available_at_selection": False,
    })
    _write(gate / "test_eval.json", {
        "schema_version": 1, "protocol": "l2_walkforward_gate_v1",
        "selected_threshold": .5, "holdout_rows_read": 0, "sealed_read": False,
        "auto_promote": False, "orders": False, "test_threshold_grid_output": False,
    })
    _write(gate / "calibration_results.json", {
        "schema_version": 1, "protocol": "l2_walkforward_gate_v1",
    })
    (gate / "feature_importance.csv").write_text("feature,gain,split\n", encoding="utf-8")
    artifacts = {name: _sha(gate / name) for name in (
        "model.txt", "prereg.json", "selection.json", "test_eval.json",
        "calibration_results.json", "feature_importance.csv",
    )}
    _write(gate / "lineage.json", {
        "schema_version": 1, "protocol": "l2_walkforward_gate_v1",
        "artifacts": artifacts,
        "candidate_input": {
            "candidates_path": str(gate_candidates),
            "candidates_sha256": _sha(gate_candidates),
            "selection_path": str(gate_optimizer / "selection.json"),
            "selection_sha256": _sha(gate_optimizer / "selection.json"),
            "prereg_path": str(gate_optimizer / "prereg.json"),
            "prereg_sha256": _sha(gate_optimizer / "prereg.json"),
        },
        "snapshot": {"manifest_path": str(gate_snapshot), "manifest_sha256": _sha(gate_snapshot)},
        "training_input": {
            "manifest_path": str(training_manifest), "manifest_sha256": _sha(training_manifest),
            "dataset_path": str(dataset), "dataset_sha256": _sha(dataset),
        },
        "safety": {"holdout_rows_read": 0, "sealed_read": False,
                   "future_rows_in_features": 0, "auto_promote": False,
                   "orders": False, "model_deployable": False},
    })

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    times = pd.date_range("2025-12-20T00:00:00Z", "2026-03-31T07:15:00Z", freq=BAR)
    values = np.arange(len(times), dtype=float)
    prices = 100 + values * .001 + np.sin(values / 13) * .1
    csv = snapshot / f"{independent_symbol}.csv"
    pd.DataFrame({"open_time": times, "open": prices, "high": prices + .5,
                  "low": prices - .5, "close": prices, "volume": 100 + values % 9}).to_csv(csv, index=False)
    manifest = snapshot / "materialization_manifest.json"
    _write(manifest, {"schema_version": 1, "sealed_read": False, "files": [{
        "path": csv.name, "symbol": independent_symbol, "sha256": _sha(csv),
        "rows": len(times), "max_materialized_time": times[-1].isoformat(),
    }]})

    backtest = tmp_path / "backtest"
    backtest.mkdir()
    weights = backtest / "weights.pt"
    weights.write_bytes(b"independent frozen L1 weights")
    candidates = backtest / "optimization_inner.jsonl"
    rows = [
        _candidate("jan-tp", independent_symbol, "2026-01-10T00:00:00Z", "tp"),
        _candidate("feb-sl", independent_symbol, "2026-02-10T00:00:00Z", "sl"),
        _candidate("mar-timeout", independent_symbol, "2026-03-10T00:00:00Z", "timeout"),
    ]
    candidates.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                          encoding="utf-8")
    summary = backtest / "summary.json"
    _write(summary, {
        "schema_version": 1, "snapshot_dir": str(snapshot), "manifest": str(manifest),
        "weights": str(weights), "weights_sha256": _sha(weights),
        "renderer": "yoyo.datasets.legacy_gold_migration.renderer.render_w10",
        "transform": "white_letterbox_full_frame", "imgsz": 960,
        "symbol_sampling": {"selected_symbols": [independent_symbol]},
        "optimization_export": {"max_horizon": 4, "min_score": .6, "thresholds": [.6]},
        "optimization_exports": {"inner": {"file": candidates.name, "rows": len(rows),
                                                       "sha256": _sha(candidates)}},
        "outcome_scope": {"name": "optimization_inner",
                          "decision_before": "2026-03-31T07:30:00Z",
                          "label_end_lte": "2026-03-31T07:30:00Z",
                          "sealed_outcomes_resolved": False},
        "holdout_read": False, "future_data_in_causal_input": False,
        "orders_placed": False, "promoted": False,
    })
    return {"gate": gate, "gate_sha": _sha(gate / "lineage.json"),
            "candidates": candidates, "summary": summary, "summary_sha": _sha(summary),
            "snapshot": snapshot, "manifest": manifest, "manifest_sha": _sha(manifest)}


def _run(fixture: dict, out: Path):
    return cross.evaluate_frozen_l2_crosssection(
        frozen_gate_dir=fixture["gate"], frozen_lineage_sha256=fixture["gate_sha"],
        candidates=fixture["candidates"], summary=fixture["summary"],
        candidate_summary_sha256=fixture["summary_sha"], snapshot_dir=fixture["snapshot"],
        manifest=fixture["manifest"], snapshot_manifest_sha256=fixture["manifest_sha"],
        out_dir=out,
    )


def test_frozen_crosssection_outputs_monthly_timeout_and_no_selection(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    result = _run(fixture, tmp_path / "result")
    out = Path(result["out_dir"])
    assert sorted(path.name for path in out.iterdir()) == sorted(cross.OUTPUT_FILES)
    evaluation = json.loads((out / "crosssection_eval.json").read_text(encoding="utf-8"))
    prereg = json.loads((out / "prereg.json").read_text(encoding="utf-8"))
    assert list(evaluation["monthly"]) == ["2026-01", "2026-02", "2026-03"]
    assert evaluation["overall"]["baseline_all_l1_selected"]["outcomes"] == {
        "sl": 1, "timeout": 1, "tp": 1,
    }
    assert evaluation["overall"]["frozen_l2_gate"]["trades"] == 2
    assert evaluation["monthly"]["2026-03"]["frozen_l2_gate"]["outcomes"]["timeout"] == 1
    assert evaluation["overall"]["frozen_l2_gate"]["density_per_symbol_day"] == pytest.approx(2 / 89.3125)
    assert prereg["selected_config_key"] == "threshold=0.6,tp=5,sl=2,horizon=4"
    assert prereg["frozen_l2_threshold"] == .5
    assert prereg["selection_or_training_performed"] is False
    assert prereg["threshold_grid_output"] is False


@pytest.mark.parametrize("target,match", [
    ("gate", "gate artifact model.txt SHA256 mismatch"),
    ("summary", "candidate summary SHA256 mismatch"),
    ("manifest", "snapshot manifest SHA256 mismatch"),
])
def test_frozen_input_sha_mutations_are_rejected(tmp_path: Path, monkeypatch, target: str, match: str) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    if target == "gate":
        (fixture["gate"] / "model.txt").write_text("tampered", encoding="utf-8")
    elif target == "summary":
        fixture["summary"].write_text(fixture["summary"].read_text() + " ", encoding="utf-8")
    else:
        fixture["manifest"].write_text(fixture["manifest"].read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _run(fixture, tmp_path / "bad")


def test_candidate_weights_bytes_are_sha_gated(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    summary = json.loads(fixture["summary"].read_text())
    Path(summary["weights"]).write_bytes(b"tampered weights")
    with pytest.raises(ValueError, match="candidate weights SHA256 mismatch"):
        _run(fixture, tmp_path / "bad")


def test_real_backtester_summary_protocol_fields_are_required(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    summary = json.loads(fixture["summary"].read_text())
    summary.pop("renderer")
    _write(fixture["summary"], summary)
    fixture["summary_sha"] = _sha(fixture["summary"])
    with pytest.raises(ValueError, match="renderer is missing"):
        _run(fixture, tmp_path / "bad")


def test_gate_and_confirmation_candidate_symbols_must_be_disjoint(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch, independent_symbol="GATE_USDT_SWAP")
    with pytest.raises(ValueError, match="overlap frozen gate candidates/training dataset"):
        _run(fixture, tmp_path / "bad")


def test_training_only_symbol_overlap_fails_before_features_or_prediction(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch, independent_symbol="TRAIN_USDT_SWAP")
    monkeypatch.setattr(
        cross, "_load_frames",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("features materialized")),
    )
    monkeypatch.setattr(
        cross, "_predict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("model predicted")),
    )
    with pytest.raises(ValueError, match="overlap frozen gate candidates/training dataset"):
        _run(fixture, tmp_path / "bad")


def test_atomic_nonoverwrite_and_no_independent_optimizer_interface(tmp_path: Path, monkeypatch, capsys) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    out = tmp_path / "result"
    _run(fixture, out)
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(FileExistsError, match="overwrite"):
        _run(fixture, out)
    assert before == {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(SystemExit) as exc:
        cross.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--summary" in help_text
    assert "optimizer-selection" not in help_text


@pytest.mark.skipif(not (REAL_EXPANSION_GATE / "lineage.json").is_file(),
                    reason="real INNER expansion fixture is not present")
def test_real_data_expansion_parent_schema_and_physical_lineage_smoke() -> None:
    model, config, threshold, lineage = cross._load_supported_frozen_gate(
        REAL_EXPANSION_GATE, REAL_EXPANSION_LINEAGE_SHA256
    )
    expected = {
        "AERO_USDT_SWAP", "DASH_USDT_SWAP", "EGLD_USDT_SWAP", "GMT_USDT_SWAP",
        "GPS_USDT_SWAP", "HBAR_USDT_SWAP", "IOTA_USDT_SWAP", "JUP_USDT_SWAP",
        "MORPHO_USDT_SWAP", "ONE_USDT_SWAP", "ZK_USDT_SWAP", "ZRO_USDT_SWAP",
    }
    assert lineage["gate_protocol"] == cross.EXPANSION_GATE_PROTOCOL
    assert lineage["lineage_sha256"] == REAL_EXPANSION_LINEAGE_SHA256
    assert config == SearchConfig(.25, 5.0, 2.0, 72)
    assert threshold == .4
    assert set(lineage["source_symbols"]) == expected
    assert set(lineage["training_dataset_symbols"]) == expected
    assert set(lineage["inherited_parent_seen_symbols"]) < expected
    assert set(lineage["frozen_seen_symbols"]) == expected
    assert lineage["frozen_l1_protocol"] == {
        "weights_sha256": "f953797c58c0551ae6bafec7bf37d302d35ba3b2e0eb694ab0c8278fff265fce",
        "renderer": "yoyo.datasets.legacy_gold_migration.renderer.render_w10",
        "transform": "white_letterbox_full_frame", "imgsz": 960,
    }
    assert list(model.feature_name()) == list(FEATURE_COLUMNS)
