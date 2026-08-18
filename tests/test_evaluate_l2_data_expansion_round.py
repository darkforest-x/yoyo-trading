"""Safety contracts for the INNER-only L2 data-expansion round."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tools.evaluate_l2_data_expansion_round as expansion
from tools.optimize_short_barrier_walkforward import SearchConfig
from yoyo.layers.l2_judgment.features import FEATURE_COLUMNS

REAL_EXPANSION_PARENT = Path(
    "/Users/zhangzc/fable-trading/experiments/"
    "fixed_w10_evolution_r5_l2_data_expansion_inner12_v1"
)
REAL_EXPANSION_PARENT_SHA256 = "5de01132ce9b3bdf80da6c17f8742c4e68ee043bf37baed2a2292ec5980f2e76"


class FakeBooster:
    def predict(self, frame, **_kwargs):
        # Scores are feature-derived, so changing Mar rows cannot alter Feb's
        # selected threshold or model bytes.
        return frame[FEATURE_COLUMNS[0]].to_numpy(dtype=float)

    def feature_importance(self, *, importance_type: str):
        assert importance_type in {"gain", "split"}
        return np.arange(len(FEATURE_COLUMNS), dtype=float)

    def save_model(self, path: str, **_kwargs):
        Path(path).write_text("deterministic expanded model\n", encoding="utf-8")

    def current_iteration(self):
        return 13


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event(index: int, *, split: str, outcome: str, score: float,
           symbol: str = "NEW_USDT_SWAP") -> dict:
    starts = {
        "train": pd.Timestamp("2026-01-02T00:00:00Z"),
        "calibration": pd.Timestamp("2026-02-02T00:00:00Z"),
        "test": pd.Timestamp("2026-03-02T00:00:00Z"),
        "purged": pd.Timestamp("2026-02-28T12:00:00Z"),
    }
    stamp = starts[split] + index * pd.Timedelta(minutes=15)
    maker = .02 if outcome == "tp" else (-.02 if outcome.startswith("sl") else .001)
    return {
        "event_id": f"{split}-{index}", "symbol": symbol,
        "signal_time": stamp, "side": "short", "split": split,
        "label": {"tp": 1, "sl": 0, "sl_ambiguous": 0}.get(outcome),
        "realized_ret": maker, "net_barrier_maker": maker,
        "net_barrier_taker": maker - .001, "outcome": outcome,
        "feature_semantics": "side_aligned_v1",
        **{name: score + offset / 10000 for offset, name in enumerate(FEATURE_COLUMNS)},
        "feature_as_of": stamp, "label_end_time": stamp + pd.Timedelta(hours=1),
        "selected_config_key": "threshold=0.6,tp=5,sl=2,horizon=4",
        "candidate_source_sha256": "1" * 64,
        "selected_config_sha256": "2" * 64, "prereg_sha256": "3" * 64,
        "snapshot_manifest_sha256": "4" * 64, "snapshot_file": "NEW.csv",
        "snapshot_file_sha256": "5" * 64, "snapshot_file_rows": 1000,
        "snapshot_max_materialized_time": "2026-03-31T07:15:00+00:00",
        "holdout_read": False, "future_rows_in_features": 0,
        "future_review_only": True, "auto_promote": False,
    }


def _events(*, mutate_test: bool = False) -> pd.DataFrame:
    rows = []
    for index in range(20):
        rows.append(_event(index, split="train", outcome="tp" if index % 2 else "sl", score=.5))
    # Both thresholds satisfy the frozen 100-trade minimum and density guard;
    # the high threshold retains 100 TP rows and wins calibration economics.
    for index in range(110):
        high = index < 100
        rows.append(_event(index, split="calibration", outcome="tp" if high else "sl",
                           score=.9 if high else .5))
    rows.append(_event(100, split="calibration", outcome="timeout", score=.9))
    for index in range(5):
        outcome = "sl" if mutate_test else ("timeout" if index == 0 else "tp")
        rows.append(_event(index, split="test", outcome=outcome, score=.9))
    rows.append(_event(0, split="purged", outcome="tp", score=.9))
    return pd.DataFrame(rows)


def _parent(tmp_path: Path) -> tuple[Path, str]:
    gate = tmp_path / "parent"
    gate.mkdir(parents=True)
    lineage = gate / "lineage.json"
    lineage.write_text("{}\n", encoding="utf-8")
    return gate, _sha(lineage)


def _parent_schema_fixture(tmp_path: Path) -> tuple[Path, str, dict]:
    """Mirror the real parent lineage shape: compact view omits candidate_input."""
    gate = tmp_path / "real-shape-parent"
    gate.mkdir()
    optimizer = tmp_path / "optimizer"
    optimizer.mkdir()
    candidates = optimizer / "candidates.jsonl"
    candidates.write_text('{"event_id":"inner"}\n', encoding="utf-8")
    optimizer_prereg = optimizer / "prereg.json"
    optimizer_prereg.write_text(json.dumps({
        "candidate_source": str(candidates),
        "candidate_source_sha256": _sha(candidates),
    }) + "\n", encoding="utf-8")
    optimizer_selection = optimizer / "config_selection.json"
    optimizer_selection.write_text(json.dumps({
        "prereg_sha256": _sha(optimizer_prereg),
    }) + "\n", encoding="utf-8")
    candidate_input = {
        "candidates_path": str(candidates), "candidates_sha256": _sha(candidates),
        "selection_path": str(optimizer_selection),
        "selection_sha256": _sha(optimizer_selection),
        "prereg_path": str(optimizer_prereg), "prereg_sha256": _sha(optimizer_prereg),
    }
    raw_lineage = {
        "protocol": expansion.crosssection_gate.FROZEN_GATE_PROTOCOL,
        "candidate_input": candidate_input,
        "lightgbm_params": dict(expansion.LGB_PARAMS),
        "safety": {"holdout_rows_read": 0, "sealed_read": False,
                   "future_rows_in_features": 0, "auto_promote": False,
                   "orders": False, "model_deployable": False},
    }
    lineage = gate / "lineage.json"
    lineage.write_text(json.dumps(raw_lineage, sort_keys=True) + "\n", encoding="utf-8")
    (gate / "prereg.json").write_text(json.dumps({
        "l2_threshold_grid": [.5, .8],
        "candidate_source_sha256": candidate_input["candidates_sha256"],
        "selection_sha256": candidate_input["selection_sha256"],
    }) + "\n", encoding="utf-8")
    return gate, _sha(lineage), candidate_input


def _fixture(tmp_path: Path, monkeypatch, *, mutate_test: bool = False) -> dict:
    gate, gate_sha = _parent(tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest = snapshot / "manifest.json"
    manifest.write_text(json.dumps({"sealed_read": False, "files": [{
        "path": "NEW.csv", "symbol": "NEW_USDT_SWAP", "sha256": "5" * 64,
        "rows": 1000, "max_materialized_time": "2026-03-31T07:15:00Z",
    }]}) + "\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    candidates = source / "optimization_inner.jsonl"
    candidates.write_text("{}\n", encoding="utf-8")
    summary = source / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    store = tmp_path / "experiment-store"
    store.mkdir()
    config = SearchConfig(.6, 5.0, 2.0, 4)
    parent_lineage = {
        "gate_protocol": expansion.crosssection_gate.FROZEN_GATE_PROTOCOL,
        "lineage_sha256": gate_sha,
        "candidate_input": {"selection_sha256": "2" * 64, "prereg_sha256": "3" * 64},
        "frozen_l1_protocol": {"weights_sha256": "a" * 64, "renderer": "render_w10",
                               "transform": "white_letterbox_full_frame", "imgsz": 960},
        "lightgbm_params": dict(expansion.LGB_PARAMS), "l2_threshold_grid": [.5, .8],
        "frozen_seen_symbols": ["PARENT_USDT_SWAP"],
    }
    source_lineage = [{
        "source_index": 0, "sha256": "1" * 64, "summary_sha256": "6" * 64,
        "sampled_symbols": ["NEW_USDT_SWAP"], "candidate_symbols": ["NEW_USDT_SWAP"],
    }]
    event_frame = _events(mutate_test=mutate_test)
    monkeypatch.setattr(expansion, "_load_parent", lambda *_args: (config, [.5, .8], parent_lineage))
    monkeypatch.setattr(
        expansion.crosssection_gate, "_checked_sha",
        lambda path, expected, _field: _sha(path),
    )
    monkeypatch.setattr(expansion, "_load_sources", lambda *_args, **_kwargs: (
        [{"placeholder": True}], source_lineage, {"NEW_USDT_SWAP"}
    ))
    monkeypatch.setattr(expansion, "_materialize_events", lambda *_args, **_kwargs: (
        event_frame.copy(), {"manifest_sha256": _sha(manifest), "files": []}
    ))
    monkeypatch.setattr(expansion, "_train_model", lambda train: (
        FakeBooster() if set(train["outcome"]) == {"tp", "sl"}
        and set(train["split"]) == {"train"} else pytest.fail("invalid supervised train input")
    ))
    return {"gate": gate, "gate_sha": gate_sha, "snapshot": snapshot, "manifest": manifest,
            "source": (candidates, summary), "store": store}


def _run(fixture: dict, out: Path, **kwargs):
    return expansion.evaluate_l2_data_expansion_round(
        parent_gate_dir=fixture["gate"], parent_lineage_sha256=fixture["gate_sha"],
        source_pairs=[fixture["source"]], snapshot_dir=fixture["snapshot"],
        manifest=fixture["manifest"], snapshot_manifest_sha256=_sha(fixture["manifest"]),
        experiment_store=fixture["store"], out_dir=out, **kwargs,
    )


def test_writes_atomic_round_dataset_and_keeps_timeout_in_economics(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    result = _run(fixture, fixture["store"] / "round")
    out = Path(result["out_dir"])
    assert sorted(path.name for path in out.iterdir()) == sorted(expansion.OUTPUT_FILES)
    prereg = json.loads((out / "prereg.json").read_text())
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    calibration = json.loads((out / "calibration_results.json").read_text())
    test_eval = json.loads((out / "test_eval.json").read_text())
    dataset = pd.read_csv(out / "dataset.csv")
    assert prereg["experimental_change"] == "data_expansion_only"
    assert prereg["calibration_guards"] == {
        "max_density_per_symbol_day": 4.0, "mean_net_maker_positive": True,
        "min_trades": 100,
    }
    assert set(dataset["outcome"]) == {"tp", "sl"}
    assert manifest["counts"]["timeouts_excluded_from_supervised_dataset"] == 2
    assert calibration["baseline_all_l1_selected"]["outcomes"]["timeout"] == 1
    assert test_eval["baseline_all_l1_selected"]["outcomes"]["timeout"] == 1
    assert test_eval["test_threshold_grid_output"] is False
    assert prereg["parent_model_deployable"] is False
    assert prereg["model_deployable"] is False
    assert manifest["dataset"]["sha256"] == _sha(out / "dataset.csv")


@pytest.mark.parametrize("field,value", [
    ("calibration_start", "2026-02-01T00:15:00Z"),
    ("test_start", "2026-03-01T00:15:00Z"),
    ("test_end", "2026-03-31T07:45:00Z"),
    ("purge_bars", 161),
])
def test_fixed_boundaries_rejected_before_parent_or_source_reads(
    tmp_path: Path, monkeypatch, field: str, value
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("parent read reached")

    monkeypatch.setattr(expansion, "_load_parent", forbidden)
    with pytest.raises(ValueError, match="fixed Feb/Mar boundaries"):
        _run(fixture, fixture["store"] / "bad", **{field: value})
    assert called is False


def test_fake_parent_lineage_and_explicit_grid_mismatch_are_rejected(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)

    def reject(*_args):
        raise ValueError("gate lineage SHA256 mismatch")

    monkeypatch.setattr(expansion, "_load_parent", reject)
    with pytest.raises(ValueError, match="lineage SHA256"):
        _run(fixture, fixture["store"] / "bad-parent")

    fixture = _fixture(tmp_path / "other", monkeypatch)
    with pytest.raises(ValueError, match="exactly equal parent"):
        _run(fixture, fixture["store"] / "bad-grid", l2_thresholds=[.5, .7])


def test_real_parent_schema_shape_restores_candidate_input_from_raw_lineage(
    tmp_path: Path, monkeypatch
) -> None:
    gate, lineage_sha, expected = _parent_schema_fixture(tmp_path)
    config = SearchConfig(.6, 5.0, 2.0, 4)
    compact = {
        "gate_protocol": expansion.crosssection_gate.FROZEN_GATE_PROTOCOL,
        "lineage_sha256": lineage_sha,
        "frozen_l1_protocol": {"weights_sha256": "a" * 64, "renderer": "render_w10",
                               "transform": "white_letterbox_full_frame", "imgsz": 960},
        "frozen_seen_symbols": ["PARENT_USDT_SWAP"],
    }
    assert "candidate_input" not in compact
    monkeypatch.setattr(expansion, "_load_supported_parent_gate", lambda *_args: (
        object(), config, .5, compact
    ))
    loaded_config, grid, parent = expansion._load_parent(gate, lineage_sha)
    assert loaded_config == config
    assert grid == [.5, .8]
    assert parent["candidate_input"] == expected
    assert parent["frozen_l1_protocol"] == compact["frozen_l1_protocol"]


def test_parent_candidate_input_tamper_fails_without_output_or_temp_residue(
    tmp_path: Path, monkeypatch
) -> None:
    gate, lineage_sha, _expected = _parent_schema_fixture(tmp_path)
    compact = {"gate_protocol": expansion.crosssection_gate.FROZEN_GATE_PROTOCOL,
               "lineage_sha256": lineage_sha, "frozen_l1_protocol": {},
               "frozen_seen_symbols": ["PARENT_USDT_SWAP"]}
    monkeypatch.setattr(expansion, "_load_supported_parent_gate", lambda *_args: (
        object(), SearchConfig(.6, 5, 2, 4), .5, compact
    ))
    raw = json.loads((gate / "lineage.json").read_text())
    raw["candidate_input"]["selection_sha256"] = "0" * 64
    (gate / "lineage.json").write_text(json.dumps(raw), encoding="utf-8")
    store = tmp_path / "store"
    store.mkdir()
    out = store / "failed-round"
    with pytest.raises(ValueError, match="parent optimizer selection SHA256 mismatch"):
        expansion.evaluate_l2_data_expansion_round(
            parent_gate_dir=gate, parent_lineage_sha256=_sha(gate / "lineage.json"),
            source_pairs=[(tmp_path / "unused.jsonl", tmp_path / "unused.json")],
            snapshot_dir=tmp_path / "snapshot", manifest=tmp_path / "manifest.json",
            snapshot_manifest_sha256="0" * 64, experiment_store=store, out_dir=out,
        )
    assert not out.exists()
    assert not list(store.glob(".failed-round.tmp-*"))


def test_expansion_parent_sources_are_cumulative_and_new_sources_are_appended(
    tmp_path: Path,
) -> None:
    pairs = []
    sources = []
    for index, name in enumerate(("old-a", "old-b")):
        directory = tmp_path / name
        directory.mkdir()
        candidate, summary = directory / "optimization_inner.jsonl", directory / "summary.json"
        candidate.write_text(f'{{"source":{index}}}\n', encoding="utf-8")
        summary.write_text(f'{{"source":{index}}}\n', encoding="utf-8")
        pairs.append((candidate.resolve(), summary.resolve()))
        sources.append({
            "source_index": index, "path": str(candidate.resolve()), "sha256": _sha(candidate),
            "summary_path": str(summary.resolve()), "summary_sha256": _sha(summary),
        })
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    new_pair = (new_dir / "optimization_inner.jsonl", new_dir / "summary.json")
    for path in new_pair:
        path.write_text("{}\n", encoding="utf-8")
    parent = {"gate_protocol": expansion.PROTOCOL, "sources": sources}
    parent["source_symbols"] = ["OLD_A", "OLD_B"]
    sources[0]["sampled_symbols"] = ["OLD_A"]
    sources[1]["sampled_symbols"] = ["OLD_B"]
    combined, inherited_count = expansion._cumulative_source_pairs(parent, [new_pair])
    assert combined == [*pairs, (new_pair[0].resolve(), new_pair[1].resolve())]
    assert inherited_count == 2


def test_expansion_parent_inherited_source_tamper_and_duplicate_are_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "old"
    directory.mkdir()
    candidate, summary = directory / "optimization_inner.jsonl", directory / "summary.json"
    candidate.write_text("{}\n", encoding="utf-8")
    summary.write_text("{}\n", encoding="utf-8")
    source = {"source_index": 0, "path": str(candidate), "sha256": _sha(candidate),
              "summary_path": str(summary), "summary_sha256": _sha(summary),
              "sampled_symbols": ["OLD"]}
    parent = {"gate_protocol": expansion.PROTOCOL, "sources": [source],
              "source_symbols": ["OLD"]}
    with pytest.raises(ValueError, match="duplicates an inherited"):
        expansion._cumulative_source_pairs(parent, [(candidate, summary)])
    candidate.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="candidate SHA256 mismatch"):
        expansion._cumulative_source_pairs(parent, [])


def test_new_source_seen_symbol_overlap_fails_before_feature_materialization(
    tmp_path: Path, monkeypatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    config = SearchConfig(.6, 5.0, 2.0, 4)
    parent = {
        "gate_protocol": expansion.crosssection_gate.FROZEN_GATE_PROTOCOL,
        "lineage_sha256": fixture["gate_sha"],
        "candidate_input": {"selection_sha256": "2" * 64, "prereg_sha256": "3" * 64},
        "frozen_l1_protocol": {"weights_sha256": "a" * 64, "renderer": "render_w10",
                               "transform": "white_letterbox_full_frame", "imgsz": 960},
        "lightgbm_params": dict(expansion.LGB_PARAMS), "l2_threshold_grid": [.5, .8],
        "frozen_seen_symbols": ["NEW_USDT_SWAP"],
    }
    monkeypatch.setattr(expansion, "_load_parent", lambda *_args: (config, [.5, .8], parent))
    monkeypatch.setattr(
        expansion, "_materialize_events",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("features materialized")),
    )
    with pytest.raises(ValueError, match="overlap parent frozen_seen_symbols"):
        _run(fixture, fixture["store"] / "bad-seen")


@pytest.mark.skipif(not (REAL_EXPANSION_PARENT / "lineage.json").is_file(),
                    reason="real INNER expansion parent is not present")
def test_real_expansion_parent_recursive_schema_and_seen_symbols() -> None:
    config, grid, parent = expansion._load_parent(
        REAL_EXPANSION_PARENT, REAL_EXPANSION_PARENT_SHA256
    )
    expected = {
        "AERO_USDT_SWAP", "DASH_USDT_SWAP", "EGLD_USDT_SWAP", "GMT_USDT_SWAP",
        "GPS_USDT_SWAP", "HBAR_USDT_SWAP", "IOTA_USDT_SWAP", "JUP_USDT_SWAP",
        "MORPHO_USDT_SWAP", "ONE_USDT_SWAP", "ZK_USDT_SWAP", "ZRO_USDT_SWAP",
    }
    assert config == SearchConfig(.25, 5.0, 2.0, 72)
    assert grid == [.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7]
    assert parent["gate_protocol"] == expansion.PROTOCOL
    assert set(parent["frozen_seen_symbols"]) == expected
    assert set(parent["source_symbols"]) == expected
    assert len(parent["sources"]) == 4
    assert parent["lightgbm_params"] == dict(expansion.LGB_PARAMS)


@pytest.mark.skipif(not (REAL_EXPANSION_PARENT / "lineage.json").is_file(),
                    reason="real INNER expansion parent is not present")
def test_real_expansion_parent_fake_lineage_sha_is_rejected() -> None:
    with pytest.raises(ValueError, match="gate lineage SHA256 mismatch"):
        expansion._load_parent(REAL_EXPANSION_PARENT, "0" * 64)


def test_global_source_symbol_and_identity_overlap_rejected(tmp_path: Path, monkeypatch) -> None:
    candidate_a, summary_a = tmp_path / "a" / "optimization_inner.jsonl", tmp_path / "a" / "summary.json"
    candidate_b, summary_b = tmp_path / "b" / "optimization_inner.jsonl", tmp_path / "b" / "summary.json"
    for path in (candidate_a, summary_a, candidate_b, summary_b):
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    config = SearchConfig(.6, 5.0, 2.0, 4)
    base_lineage = {"sampled_symbols": ["SAME"], "weights_sha256": "a" * 64,
                    "renderer": "r", "transform": "t", "imgsz": 1}
    monkeypatch.setattr(expansion.crosssection_gate, "_load_independent_candidates", lambda *args, **kwargs: (
        [], {**base_lineage, "sha256": "1" * 64, "summary_sha256": "2" * 64}, {"SAME"}
    ))
    monkeypatch.setattr(expansion.crosssection_gate, "_json", lambda path, *_args: {
        "manifest": str(manifest), "snapshot_dir": str(snapshot),
        "outcome_scope": {"decision_before": expansion.FIXED_TEST_END.isoformat(),
                          "label_end_lte": expansion.FIXED_TEST_END.isoformat()},
    })
    monkeypatch.setattr(expansion, "read_jsonl", lambda path: [{
        "event_id": f"event-{path.parent.name}", "symbol": "SAME",
        "decision_time": "2026-01-02T00:00:00Z",
    }])
    snapshot = tmp_path / "snapshot"
    manifest = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="symbols physically overlap"):
        expansion._load_sources(
            [(candidate_a, summary_a), (candidate_b, summary_b)], config=config,
            start=expansion.FIXED_TRAIN_START, end=expansion.FIXED_TEST_END,
            frozen_l1={"weights_sha256": "a" * 64, "renderer": "r", "transform": "t", "imgsz": 1},
            snapshot_dir=snapshot, manifest=manifest,
        )


def test_native_source_contract_failure_propagates_before_materialization(tmp_path: Path, monkeypatch) -> None:
    candidates = tmp_path / "optimization_inner.jsonl"
    summary = tmp_path / "summary.json"
    candidates.write_text("{}\n", encoding="utf-8")
    summary.write_text("{}\n", encoding="utf-8")

    def incompatible(*_args, **_kwargs):
        raise ValueError("candidate optimization export horizon is shorter than frozen horizon")

    monkeypatch.setattr(expansion.crosssection_gate, "_load_independent_candidates", incompatible)
    with pytest.raises(ValueError, match="shorter than frozen horizon"):
        expansion._load_sources(
            [(candidates, summary)], config=SearchConfig(.6, 5, 2, 4),
            start=expansion.FIXED_TRAIN_START, end=expansion.FIXED_TEST_END,
            frozen_l1={}, snapshot_dir=tmp_path, manifest=summary,
        )


@pytest.mark.parametrize("duplicate", ["event", "identity"])
def test_global_event_and_identity_uniqueness_across_disjoint_sources(
    tmp_path: Path, monkeypatch, duplicate: str
) -> None:
    pairs = []
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        candidates, summary = directory / "optimization_inner.jsonl", directory / "summary.json"
        candidates.write_text("{}\n", encoding="utf-8")
        summary.write_text("{}\n", encoding="utf-8")
        pairs.append((candidates, summary))
    calls = 0

    def load(*_args, **_kwargs):
        nonlocal calls
        symbol = f"SYM-{calls}"
        calls += 1
        return [], {"sampled_symbols": [symbol], "sha256": "1" * 64,
                    "summary_sha256": "2" * 64}, {symbol}

    monkeypatch.setattr(expansion.crosssection_gate, "_load_independent_candidates", load)
    monkeypatch.setattr(expansion.crosssection_gate, "_json", lambda *_args: {
        "manifest": str(tmp_path / "manifest.json"), "snapshot_dir": str(tmp_path / "snapshot"),
        "outcome_scope": {"decision_before": expansion.FIXED_TEST_END.isoformat(),
                          "label_end_lte": expansion.FIXED_TEST_END.isoformat()},
    })
    rows = {
        "a": {"event_id": "same", "symbol": "SYM-0",
              "decision_time": "2026-01-02T00:00:00Z", "future_path": []},
        "b": {"event_id": "same" if duplicate == "event" else "different",
              "symbol": "SYM-0" if duplicate == "identity" else "SYM-1",
              "decision_time": "2026-01-02T00:00:00Z", "future_path": []},
    }
    monkeypatch.setattr(expansion, "read_jsonl", lambda path: [rows[path.parent.name]])
    with pytest.raises(ValueError, match=f"duplicate global candidate {duplicate}"):
        expansion._load_sources(
            pairs, config=SearchConfig(.6, 5, 2, 4), start=expansion.FIXED_TRAIN_START,
            end=expansion.FIXED_TEST_END, frozen_l1={},
            snapshot_dir=tmp_path / "snapshot", manifest=tmp_path / "manifest.json",
        )


@pytest.mark.parametrize("field", ["decision", "future"])
def test_post_test_april_or_holdout_candidate_bytes_are_rejected(
    tmp_path: Path, monkeypatch, field: str
) -> None:
    candidates, summary = tmp_path / "optimization_inner.jsonl", tmp_path / "summary.json"
    candidates.write_text("{}\n", encoding="utf-8")
    summary.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(expansion.crosssection_gate, "_load_independent_candidates", lambda *_args, **_kwargs: (
        [], {"sampled_symbols": ["SYM"], "sha256": "1" * 64,
             "summary_sha256": "2" * 64}, {"SYM"}
    ))
    monkeypatch.setattr(expansion.crosssection_gate, "_json", lambda *_args: {
        "manifest": str(tmp_path / "manifest.json"), "snapshot_dir": str(tmp_path / "snapshot"),
        "outcome_scope": {"decision_before": expansion.FIXED_TEST_END.isoformat(),
                          "label_end_lte": expansion.FIXED_TEST_END.isoformat()},
    })
    row = {"event_id": "unsafe", "symbol": "SYM",
           "decision_time": "2026-04-01T00:00:00Z" if field == "decision" else "2026-03-01T00:00:00Z",
           "future_path": []}
    if field == "future":
        row["future_path"] = [{"time": "2026-05-04T00:00:00Z"}]
    monkeypatch.setattr(expansion, "read_jsonl", lambda _path: [row])
    with pytest.raises(ValueError, match="post-test/April"):
        expansion._load_sources(
            [(candidates, summary)], config=SearchConfig(.6, 5, 2, 4),
            start=expansion.FIXED_TRAIN_START, end=expansion.FIXED_TEST_END,
            frozen_l1={}, snapshot_dir=tmp_path / "snapshot",
            manifest=tmp_path / "manifest.json",
        )


def test_snapshot_must_be_inner_only_and_end_before_fixed_test_boundary(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sealed_read": False, "files": [{
        "path": "SYM.csv", "symbol": "SYM", "sha256": "1" * 64, "rows": 1,
        "max_materialized_time": "2026-04-01T00:00:00Z",
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="April or post-test"):
        expansion._validate_round_snapshot_manifest(manifest)
    manifest.write_text(json.dumps({"sealed_read": True, "files": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="sealed/purge/holdout"):
        expansion._validate_round_snapshot_manifest(manifest)


def test_mar_outcomes_never_affect_model_or_feb_selection(tmp_path: Path, monkeypatch) -> None:
    first_fixture = _fixture(tmp_path / "first", monkeypatch, mutate_test=False)
    first = first_fixture["store"] / "round"
    _run(first_fixture, first)
    second_fixture = _fixture(tmp_path / "second", monkeypatch, mutate_test=True)
    second = second_fixture["store"] / "round"
    _run(second_fixture, second)
    assert (first / "model.txt").read_bytes() == (second / "model.txt").read_bytes()
    assert (first / "selection.json").read_bytes() == (second / "selection.json").read_bytes()
    assert (first / "test_eval.json").read_bytes() != (second / "test_eval.json").read_bytes()


def test_atomic_no_overwrite_and_cli_exposes_repeatable_pairs(tmp_path: Path, monkeypatch, capsys) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    out = fixture["store"] / "round"
    _run(fixture, out)
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(FileExistsError, match="overwrite"):
        _run(fixture, out)
    assert before == {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(SystemExit) as exc:
        expansion.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--source-candidates" in help_text and "--source-summary" in help_text
    assert "--parent-lineage-sha256" in help_text


def test_fresh_process_imports_against_real_crosssection_module() -> None:
    """Exercise module resolution without pytest's sys.modules monkeypatches."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    process = subprocess.run(
        [sys.executable, "-c", (
            "import tools.evaluate_frozen_l2_crosssection as c; "
            "import tools.evaluate_l2_data_expansion_round as e; "
            "assert e.crosssection_gate is c; "
            "assert callable(e._load_walkforward_parent_gate); "
            "assert callable(e._load_supported_parent_gate); "
            "print('FRESH_IMPORT_OK')"
        )],
        cwd=Path(__file__).resolve().parents[1], env=env,
        text=True, capture_output=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "FRESH_IMPORT_OK"
