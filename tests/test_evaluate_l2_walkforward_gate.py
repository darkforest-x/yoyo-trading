"""Safety-contract tests for the integrated L2 walk-forward gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tools.evaluate_l2_walkforward_gate as gate
import tools.optimize_short_barrier_walkforward as optimizer
from yoyo.contracts.outcomes import ATR_ELIGIBILITY_PROTOCOL, ATR_PCT_MIN
from tools.build_l2_feedback_dataset import FEATURE_COLUMNS, INCLUDED_OUTCOMES

BAR = pd.Timedelta(minutes=15)


class FakeBooster:
    def __init__(self, offset: float = 0.0) -> None:
        self.offset = offset

    def predict(self, frame, **_kwargs):
        n = len(frame)
        if n == 4:
            return np.array([0.9, 0.9, 0.5, 0.5], dtype=float)
        if n == 3:
            return np.array([0.9, 0.5, 0.5], dtype=float)
        return np.full(n, 0.5 + self.offset, dtype=float)

    def feature_importance(self, *, importance_type: str):
        assert importance_type in {"gain", "split"}
        return np.arange(len(FEATURE_COLUMNS), dtype=float)

    def save_model(self, path: str, **_kwargs):
        Path(path).write_text("fake-lightgbm deterministic\n", encoding="utf-8")

    def current_iteration(self):
        return 11


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _features(score: float, index: int) -> dict[str, float]:
    return {name: (float(score) if name == "ma_spread_pct" else float(index + offset / 1000))
            for offset, name in enumerate(FEATURE_COLUMNS)}


def _training_dataset(tmp_path: Path, *, mutate: dict | None = None) -> tuple[Path, str]:
    directory = tmp_path / "training"
    directory.mkdir()
    rows = []
    times = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="6h")
    for index, stamp in enumerate(times):
        label = index % 2
        outcome = "tp" if label else "sl"
        rows.append({
            "event_id": f"train-{index}",
            "symbol": f"SYM-{index % 2}",
            "signal_time": stamp.isoformat(),
            "side": "short",
            "split": "train" if index < 4 else "val",
            "label": label,
            "realized_ret": .01 if label else -.01,
            "net_barrier_maker": .009 if label else -.011,
            "net_barrier_taker": .008 if label else -.012,
            "outcome": outcome,
            "feature_semantics": "side_aligned_v1",
            **_features(.2 + .1 * label, index),
            "feature_as_of": stamp.isoformat(),
            "label_end_time": (stamp + BAR).isoformat(),
            "selected_config_key": "threshold=0.6,tp=5,sl=2,horizon=4",
            "candidate_source_sha256": "1" * 64,
            "selected_config_sha256": "2" * 64,
            "prereg_sha256": "3" * 64,
            "snapshot_manifest_sha256": "4" * 64,
            "snapshot_file": "BTC-USDT-SWAP.csv",
            "snapshot_file_sha256": "5" * 64,
            "snapshot_file_rows": 1000,
            "snapshot_max_materialized_time": "2026-04-01T23:45:00Z",
            "holdout_read": False,
            "future_rows_in_features": 0,
            "future_review_only": True,
            "auto_promote": False,
        })
    data = pd.DataFrame(rows, columns=gate.DATASET_COLUMNS)
    if mutate and "data" in mutate:
        mutate["data"](data)
    dataset_path = directory / "dataset.csv"
    data.to_csv(dataset_path, index=False)
    manifest = {
        "schema_version": 1,
        "protocol": "l2_feedback_dataset_v1",
        "dataset": {"path": "dataset.csv", "rows": len(data), "sha256": _sha(dataset_path)},
        "selected_config": {"threshold": .6, "tp_atr": 5.0, "sl_atr": 2.0, "horizon": 4},
        "selected_config_key": "threshold=0.6,tp=5,sl=2,horizon=4",
        "protocol_contract": {
            "side": "short",
            "feature_semantics": "side_aligned_v1",
            "feature_columns": list(FEATURE_COLUMNS),
            "included_outcomes": INCLUDED_OUTCOMES,
        },
        "inner_cutoff": "2026-04-02T00:00:00Z",
        "holdout_start": "2026-05-04T00:00:00Z",
        "holdout_rows_read": 0,
        "future_rows_in_features": 0,
        "auto_promote": False,
    }
    if mutate and "manifest" in mutate:
        mutate["manifest"](manifest)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return directory, _sha(manifest_path)


def _snapshot(tmp_path: Path, *, mutate_after: int | None = None,
              max_time: str = "2026-04-01T23:45:00Z") -> tuple[Path, Path]:
    directory = tmp_path / f"snapshot-{len(list(tmp_path.glob('snapshot-*')))}"
    directory.mkdir()
    times = pd.date_range("2026-01-01T00:00:00Z", "2026-04-01T23:45:00Z", freq=BAR)
    x = np.arange(len(times), dtype=float)
    close = 100 + .01 * x + .5 * np.sin(x / 9)
    if mutate_after is not None:
        close[mutate_after:] += 500
    frame = pd.DataFrame({
        "open_time": times,
        "open": close,
        "high": close + .5,
        "low": close - .5,
        "close": close,
        "volume": 100 + x % 7,
    })
    csv_path = directory / "BTC-USDT-SWAP.csv"
    frame.to_csv(csv_path, index=False)
    manifest = tmp_path / f"{directory.name}_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "sealed_read": False,
        "files": [{
            "path": csv_path.name,
            "symbol": "BTC-USDT-SWAP",
            "sha256": _sha(csv_path),
            "rows": len(frame),
            "max_materialized_time": max_time,
        }],
    }, sort_keys=True), encoding="utf-8")
    return directory, manifest


def _future(decision: str, outcome: str, horizon: int = 4) -> list[dict]:
    start = pd.Timestamp(decision) + BAR
    rows = []
    for offset in range(horizon):
        high, low = 100.5, 99.5
        if offset == 0 and outcome == "tp":
            low = 94.5
        elif offset == 0 and outcome == "sl":
            high = 102.5
        elif offset == 0 and outcome == "sl_ambiguous":
            high, low = 102.5, 94.5
        rows.append({
            "time": (start + offset * BAR).isoformat(),
            "open": 100.0,
            "high": high,
            "low": low,
            "close": 100.0,
        })
    return rows


def _candidate(event_id: str, decision: str, outcome: str, score: float = .9) -> dict:
    return {
        "event_id": event_id,
        "symbol": "BTC-USDT-SWAP",
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


def _optimizer_pair(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    directory = tmp_path / "optimizer"
    directory.mkdir(parents=True)
    candidates = directory / "merged_inner_candidates.jsonl"
    _write_jsonl(candidates, rows)
    grid = [optimizer.SearchConfig(.6, 5.0, 2.0, 4)]
    guards = optimizer.Guards(
        min_trades=1,
        max_gold_recall_drop_pp=100,
        max_trigger_density_per_symbol_day=100,
        min_positive_folds=0,
        min_worst_fold_maker_bp=-10000,
    )
    fold_doc = {
        "schema_version": 1,
        "purge_bars": 162,
        "folds": [{
            "name": "2026-02-inner",
            "calibration_start": "2026-01-01T00:00:00Z",
            "calibration_end": "2026-01-30T07:30:00Z",
            "validation_start": "2026-02-01T00:00:00Z",
            "validation_end": "2026-03-01T00:00:00Z",
        }],
        "sealed_eval": {
            "name": "2026-04-final",
            "start": "2026-04-02T00:00:00Z",
            "end": "2026-05-04T00:00:00Z",
        },
    }
    folds, sealed = optimizer.parse_fold_document(fold_doc)
    discipline = {"experimental_discipline": "multi_factor_exploratory"}
    validated = optimizer.validate_candidates(
        [dict(row) for row in rows],
        max_horizon=4,
        allowed_end=sealed["start"],
    )
    selected, _ = optimizer.select_config(validated, folds, grid, guards, discipline)
    prereg = optimizer._jsonable(optimizer._protocol(grid, guards, fold_doc, discipline))
    prereg.update({
        "candidate_source": str(candidates.resolve()),
        "candidate_source_sha256": _sha(candidates),
        "sealed_read": False,
    })
    prereg_path = directory / "prereg.json"
    prereg_path.write_text(json.dumps(prereg, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    selection = directory / "config_selection.json"
    selection.write_text(json.dumps({
        "schema_version": 1,
        **optimizer._jsonable(selected),
        "prereg_sha256": _sha(prereg_path),
        "sealed_read": False,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return candidates, selection


def _fixture(tmp_path: Path) -> dict:
    training, training_sha = _training_dataset(tmp_path)
    snapshot, manifest = _snapshot(tmp_path)
    rows = [
        _candidate("cal-tp-high", "2026-02-01T00:00:00Z", "tp"),
        _candidate("cal-sl-high", "2026-02-04T00:00:00Z", "sl"),
        _candidate("cal-timeout-mid", "2026-02-07T00:00:00Z", "timeout"),
        _candidate("cal-tp-mid", "2026-02-10T00:00:00Z", "tp"),
        _candidate("purged-gap", "2026-02-28T12:00:00Z", "tp"),
        _candidate("test-tp", "2026-03-01T00:00:00Z", "tp"),
        _candidate("test-sl", "2026-03-04T00:00:00Z", "sl"),
        _candidate("test-timeout", "2026-03-07T00:00:00Z", "timeout"),
    ]
    candidates, selection = _optimizer_pair(tmp_path, rows)
    store = tmp_path / "external-store"
    store.mkdir()
    return {
        "training": training,
        "training_sha": training_sha,
        "snapshot": snapshot,
        "manifest": manifest,
        "candidates": candidates,
        "selection": selection,
        "store": store,
    }


def _run(fixture: dict, out: Path, **kwargs):
    return gate.evaluate_l2_walkforward_gate(
        training_dataset_dir=fixture["training"],
        training_manifest_sha256=fixture["training_sha"],
        candidates=fixture["candidates"],
        selection=fixture["selection"],
        snapshot_dir=kwargs.pop("snapshot_dir", fixture["snapshot"]),
        manifest=kwargs.pop("manifest", fixture["manifest"]),
        experiment_store=fixture["store"],
        out_dir=out,
        l2_thresholds=kwargs.pop("l2_thresholds", [.2, .5, .8]),
        max_density_per_symbol_day=kwargs.pop("max_density_per_symbol_day", 4.0),
        min_calibration_trades=kwargs.pop("min_calibration_trades", 1),
        min_test_trades=kwargs.pop("min_test_trades", 1),
        **kwargs,
    )


def _rewrite_selection_prereg_sha(selection_path: Path) -> None:
    raw = json.loads(selection_path.read_text(encoding="utf-8"))
    raw["prereg_sha256"] = _sha(selection_path.with_name("prereg.json"))
    selection_path.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_integrated_gate_writes_atomic_artifacts_and_includes_timeout_economics(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(gate, "_train_model", lambda _train: FakeBooster())
    result = _run(fixture, fixture["store"] / "run")
    out = Path(result["out_dir"])
    assert sorted(path.name for path in out.iterdir()) == sorted(gate.OUTPUT_FILES)
    prereg = json.loads((out / "prereg.json").read_text(encoding="utf-8"))
    calibration = json.loads((out / "calibration_results.json").read_text(encoding="utf-8"))
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    test_eval = json.loads((out / "test_eval.json").read_text(encoding="utf-8"))
    lineage = json.loads((out / "lineage.json").read_text(encoding="utf-8"))
    assert prereg["threshold_policy"] == "calibration_only"
    assert selection["selected_threshold"] in [.2, .5, .8]
    assert test_eval["test_threshold_grid_output"] is False
    assert "timeout" in calibration["baseline_all_l1_selected"]["outcomes"]
    assert "timeout" in test_eval["baseline_all_l1_selected"]["outcomes"]
    assert lineage["partitions"]["train"]["max_label_end"] < "2026-01-30T07:30:00+00:00"
    assert lineage["partitions"]["calibration"]["max_label_end"] < "2026-02-27T07:30:00+00:00"
    assert lineage["partitions"]["purged_candidate_events"] >= 1
    assert lineage["safety"] == {
        "holdout_rows_read": 0,
        "sealed_read": False,
        "future_rows_in_features": 0,
        "auto_promote": False,
        "orders": False,
        "model_deployable": False,
    }
    assert all(len(value) == 64 for value in lineage["artifacts"].values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"calibration_start": "2026-02-02T00:00:00Z"},
        {"test_start": "2026-03-02T00:00:00Z"},
        {"test_end": "2026-03-31T07:45:00Z"},
    ],
)
def test_boundaries_must_exactly_match_fixed_feb_mar_protocol(tmp_path: Path, monkeypatch, kwargs) -> None:
    fixture = _fixture(tmp_path)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("selection validation was reached")

    monkeypatch.setattr(gate, "_validate_candidate_source", forbidden)
    with pytest.raises(ValueError, match="fixed Feb/Mar boundaries"):
        _run(fixture, fixture["store"] / "bad-boundary", **kwargs)
    assert called is False


def test_handwritten_self_consistent_selection_is_rejected_before_training(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    candidates = fixture["candidates"]
    prereg = {
        "schema_version": 1,
        "sealed_read": False,
        "side": "short",
        "candidate_source": str(candidates.resolve()),
        "candidate_source_sha256": _sha(candidates),
    }
    prereg_path = fixture["selection"].with_name("prereg.json")
    prereg_path.write_text(json.dumps(prereg, sort_keys=True), encoding="utf-8")
    fixture["selection"].write_text(json.dumps({
        "schema_version": 1,
        "sealed_read": False,
        "selected_config": {"threshold": .6, "tp_atr": 5.0, "sl_atr": 2.0, "horizon": 4},
        "selected_config_key": "threshold=0.6,tp=5,sl=2,horizon=4",
        "prereg_sha256": _sha(prereg_path),
    }, sort_keys=True), encoding="utf-8")
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("training rows were read")

    monkeypatch.setattr(gate, "_load_training_rows", forbidden)
    with pytest.raises(ValueError, match="grid/guards/fold_document"):
        _run(fixture, fixture["store"] / "bad-handwritten")
    assert called is False


@pytest.mark.parametrize("target", ["rankings", "guards", "folds"])
def test_optimizer_selection_tampering_is_recomputed_and_rejected(
    tmp_path: Path, monkeypatch, target: str
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(gate, "_train_model", lambda _train: FakeBooster())
    prereg_path = fixture["selection"].with_name("prereg.json")
    if target == "rankings":
        raw = json.loads(fixture["selection"].read_text(encoding="utf-8"))
        raw["rankings"] = []
        fixture["selection"].write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        match = "recomputed optimizer selection"
    else:
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        if target == "guards":
            prereg["guards"]["min_trades"] = 99
            match = "no parameter configuration passes"
        else:
            prereg["fold_document"]["folds"][0]["validation_start"] = "2026-02-02T00:00:00Z"
            match = "recomputed optimizer selection"
        prereg_path.write_text(json.dumps(prereg, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        _rewrite_selection_prereg_sha(fixture["selection"])
    with pytest.raises(ValueError, match=match):
        _run(fixture, fixture["store"] / f"bad-{target}")


def test_future_snapshot_mutation_after_decisions_does_not_change_selection_or_model(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(gate, "_train_model", lambda _train: FakeBooster())
    first = fixture["store"] / "first"
    _run(fixture, first)
    mutated_snapshot, mutated_manifest = _snapshot(tmp_path, mutate_after=6500)
    second = fixture["store"] / "second"
    _run(fixture, second, snapshot_dir=mutated_snapshot, manifest=mutated_manifest)
    assert (first / "selection.json").read_bytes() == (second / "selection.json").read_bytes()
    assert (first / "model.txt").read_bytes() == (second / "model.txt").read_bytes()


def test_threshold_selection_is_calibration_only_not_test_mutable(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(gate, "_train_model", lambda _train: FakeBooster())
    first = fixture["store"] / "first"
    _run(fixture, first)
    rows = [json.loads(line) for line in fixture["candidates"].read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["event_id"].startswith("test-"):
            row["future_path"] = _future(row["decision_time"], "tp" if row["event_id"] == "test-sl" else "sl")
    fixture["candidates"].write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    prereg_path = fixture["selection"].with_name("prereg.json")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    prereg["candidate_source_sha256"] = _sha(fixture["candidates"])
    prereg_path.write_text(json.dumps(prereg, sort_keys=True), encoding="utf-8")
    selected = json.loads(fixture["selection"].read_text(encoding="utf-8"))
    selected["prereg_sha256"] = _sha(prereg_path)
    fixture["selection"].write_text(json.dumps(selected, sort_keys=True), encoding="utf-8")
    second = fixture["store"] / "second"
    _run(fixture, second)
    assert (first / "selection.json").read_bytes() == (second / "selection.json").read_bytes()
    assert (first / "model.txt").read_bytes() == (second / "model.txt").read_bytes()
    assert (first / "test_eval.json").read_bytes() != (second / "test_eval.json").read_bytes()


def test_density_uses_exposed_symbol_days(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(gate, "_train_model", lambda _train: FakeBooster())
    out = fixture["store"] / "run"
    _run(fixture, out, l2_thresholds=[.2], max_density_per_symbol_day=4.0)
    calibration = json.loads((out / "calibration_results.json").read_text(encoding="utf-8"))
    result = calibration["results"][0]
    assert result["trades"] == 4
    assert result["density_per_symbol_day"] == pytest.approx(4 / 28)
    assert result["eligible"] is True


def test_apr1_candidate_source_fails_before_features_or_outcome_materialization(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    rows = [json.loads(line) for line in fixture["candidates"].read_text(encoding="utf-8").splitlines()]
    rows.append(_candidate("apr1-purge", "2026-04-01T00:00:00Z", "tp"))
    candidates, selection = _optimizer_pair(tmp_path / "apr1-source", rows)
    fixture["candidates"], fixture["selection"] = candidates, selection
    called = {"frames": False, "resolve": False}

    def forbidden_frames(*_args, **_kwargs):
        called["frames"] = True
        raise AssertionError("_load_frames was reached")

    def forbidden_resolve(*_args, **_kwargs):
        called["resolve"] = True
        raise AssertionError("_resolve was reached")

    monkeypatch.setattr(gate, "_load_frames", forbidden_frames)
    monkeypatch.setattr(gate, "_resolve", forbidden_resolve)
    with pytest.raises(ValueError, match="exceeds fixed Mar test source window"):
        _run(fixture, fixture["store"] / "bad-apr1")
    assert called == {"frames": False, "resolve": False}


def test_candidate_source_label_end_boundary_is_exclusive_horizon_plus_one_bar(tmp_path: Path) -> None:
    allowed_decision = gate.FIXED_TEST_END - (4 + 1) * BAR
    gate._validate_raw_candidate_window(
        [_candidate("last-allowed", allowed_decision.isoformat(), "tp")],
        max_horizon=4,
        source=tmp_path / "source.jsonl",
    )
    with pytest.raises(ValueError, match="exceeds fixed Mar test source window"):
        gate._validate_raw_candidate_window(
            [_candidate("too-late", (allowed_decision + BAR).isoformat(), "tp")],
            max_horizon=4,
            source=tmp_path / "source.jsonl",
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda f: (f["training"] / "manifest.json").write_text("{}", encoding="utf-8"), "manifest SHA256"),
        (lambda f: (f["training"] / "dataset.csv").write_text((f["training"] / "dataset.csv").read_text(encoding="utf-8") + "\n", encoding="utf-8"), "dataset.csv SHA256"),
        (lambda f: json.loads(f["selection"].read_text(encoding="utf-8")).update(sealed_read=True), "noop"),
    ],
)
def test_sha_lineage_failures_are_rejected(tmp_path: Path, monkeypatch, mutation, match: str) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(gate, "_train_model", lambda _train: FakeBooster())
    if match == "noop":
        raw = json.loads(fixture["selection"].read_text(encoding="utf-8"))
        raw["prereg_sha256"] = "0" * 64
        fixture["selection"].write_text(json.dumps(raw), encoding="utf-8")
        expected = "prereg_sha256"
    else:
        mutation(fixture)
        expected = match
    with pytest.raises(ValueError, match=expected):
        _run(fixture, fixture["store"] / "bad")


@pytest.mark.parametrize(
    "target,match",
    [
        ("training_holdout", "holdout_read"),
        ("candidate_sealed", "sealed/holdout-derived"),
        ("snapshot_sealed", "sealed/purge/holdout"),
    ],
)
def test_sealed_holdout_rejection(tmp_path: Path, monkeypatch, target: str, match: str) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(gate, "_train_model", lambda _train: FakeBooster())
    if target == "training_holdout":
        data = pd.read_csv(fixture["training"] / "dataset.csv")
        data.loc[0, "holdout_read"] = True
        data.to_csv(fixture["training"] / "dataset.csv", index=False)
        manifest = json.loads((fixture["training"] / "manifest.json").read_text(encoding="utf-8"))
        manifest["dataset"]["sha256"] = _sha(fixture["training"] / "dataset.csv")
        (fixture["training"] / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        fixture["training_sha"] = _sha(fixture["training"] / "manifest.json")
    elif target == "candidate_sealed":
        row = json.loads(fixture["candidates"].read_text(encoding="utf-8").splitlines()[0])
        row["sealed_read"] = True
        candidates, selection = _optimizer_pair(tmp_path / "bad-sealed", [row])
        fixture["candidates"], fixture["selection"] = candidates, selection
    else:
        raw = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
        raw["sealed_read"] = True
        fixture["manifest"].write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _run(fixture, fixture["store"] / "bad")


def test_duplicate_candidate_identity_rejected(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(gate, "_train_model", lambda _train: FakeBooster())
    row = json.loads(fixture["candidates"].read_text(encoding="utf-8").splitlines()[0])
    duplicate = dict(row, event_id="different-id")
    fixture["candidates"].write_text(
        fixture["candidates"].read_text(encoding="utf-8") + json.dumps(duplicate, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prereg = json.loads(fixture["selection"].with_name("prereg.json").read_text(encoding="utf-8"))
    prereg["candidate_source_sha256"] = _sha(fixture["candidates"])
    fixture["selection"].with_name("prereg.json").write_text(json.dumps(prereg, sort_keys=True), encoding="utf-8")
    selection = json.loads(fixture["selection"].read_text(encoding="utf-8"))
    selection["prereg_sha256"] = _sha(fixture["selection"].with_name("prereg.json"))
    fixture["selection"].write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate event identity"):
        _run(fixture, fixture["store"] / "bad")


def test_atomic_overwrite_refusal_preserves_existing_artifacts(tmp_path: Path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(gate, "_train_model", lambda _train: FakeBooster())
    out = fixture["store"] / "run"
    _run(fixture, out)
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(FileExistsError, match="overwrite"):
        _run(fixture, out)
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before


def test_label_future_causality_gates_and_cli_help(tmp_path: Path, monkeypatch, capsys) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(gate, "_train_model", lambda _train: FakeBooster())
    rows = [json.loads(line) for line in fixture["candidates"].read_text(encoding="utf-8").splitlines()]
    rows[0]["causal_metadata"]["visible_end_time"] = "2026-02-01T00:15:00Z"
    fixture["candidates"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    prereg_path = fixture["selection"].with_name("prereg.json")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    prereg["candidate_source_sha256"] = _sha(fixture["candidates"])
    prereg_path.write_text(json.dumps(prereg, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _rewrite_selection_prereg_sha(fixture["selection"])
    with pytest.raises(ValueError, match="looks beyond decision"):
        _run(fixture, fixture["store"] / "bad")
    with pytest.raises(SystemExit) as exc:
        gate.main(["--help"])
    assert exc.value.code == 0
    assert "--l2-thresholds" in capsys.readouterr().out
