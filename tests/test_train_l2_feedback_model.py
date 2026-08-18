"""Safety-contract tests for the explicit L2 feedback trainer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tools.train_l2_feedback_model as trainer
from tools.build_l2_feedback_dataset import FEATURE_COLUMNS, INCLUDED_OUTCOMES


class FakeBooster:
    best_iteration = 7

    def predict(self, frame, *, num_iteration: int):
        assert list(frame.columns) == FEATURE_COLUMNS
        assert num_iteration == self.best_iteration
        values = frame["ma_spread_pct"].to_numpy(dtype=float)
        lo, hi = float(values.min()), float(values.max())
        return 0.2 + 0.6 * (values - lo) / (hi - lo)

    def feature_importance(self, *, importance_type: str):
        assert importance_type in {"gain", "split"}
        return np.arange(len(FEATURE_COLUMNS), dtype=float)

    def save_model(self, path: str, *, num_iteration: int):
        Path(path).write_text(f"fake-lightgbm best_iteration={num_iteration}\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(tmp_path: Path, *, n_train: int = 20, n_val: int = 20) -> tuple[Path, str]:
    directory = tmp_path / "feedback-dataset"
    directory.mkdir(parents=True)
    val_start = pd.Timestamp("2026-01-10T00:00:00Z")
    train_times = pd.date_range("2026-01-01T00:00:00Z", periods=n_train, freq="1h")
    val_times = pd.date_range(val_start, periods=n_val, freq="1h")
    rows = []
    for index, (split, stamp) in enumerate(
        [("train", stamp) for stamp in train_times] + [("val", stamp) for stamp in val_times]
    ):
        label = index % 2
        outcome = "tp" if label else ("sl_ambiguous" if index % 4 == 0 else "sl")
        features = {name: float(index + offset / 100) for offset, name in enumerate(FEATURE_COLUMNS)}
        rows.append({
            "event_id": f"event-{index}",
            "symbol": f"SYMBOL-{index % 3}",
            "signal_time": stamp.isoformat(),
            "side": "short",
            "split": split,
            "label": label,
            "realized_ret": .01 if label else -.01,
            "net_barrier_maker": .009 if label else -.011,
            "net_barrier_taker": .008 if label else -.012,
            "outcome": outcome,
            "feature_semantics": "side_aligned_v1",
            **features,
            "feature_as_of": stamp.isoformat(),
            "label_end_time": (stamp + pd.Timedelta(hours=1)).isoformat(),
            "selected_config_key": "fixed",
            "candidate_source_sha256": "1" * 64,
            "selected_config_sha256": "2" * 64,
            "prereg_sha256": "3" * 64,
            "snapshot_manifest_sha256": "4" * 64,
            "snapshot_file": "SYMBOL.csv",
            "snapshot_file_sha256": "5" * 64,
            "snapshot_file_rows": 1000,
            "snapshot_max_materialized_time": "2026-03-01T00:00:00Z",
            "holdout_read": False,
            "future_rows_in_features": 0,
            "future_review_only": True,
            "auto_promote": False,
        })
    data = pd.DataFrame(rows, columns=trainer.DATASET_COLUMNS)
    dataset_path = directory / "dataset.csv"
    data.to_csv(dataset_path, index=False)
    counts = lambda frame: {"0": int((frame.label == 0).sum()), "1": int((frame.label == 1).sum())}
    train, val = data[data.split == "train"], data[data.split == "val"]
    manifest = {
        "schema_version": 1,
        "protocol": "l2_feedback_dataset_v1",
        "dataset": {"path": "dataset.csv", "rows": len(data), "sha256": _sha(dataset_path)},
        "counts": {"dataset_rows": len(data), "train_rows": len(train), "val_rows": len(val)},
        "class_balance": {"train": counts(train), "val": counts(val)},
        "split": {
            "method": "explicit_time_boundary",
            "val_start": val_start.isoformat(),
            "purge_bars": 162,
            "purge_duration": "1 days 16:30:00",
            "train_rule": "label_end_time < val_start - purge_duration",
            "val_rule": "signal_time >= val_start",
            "random": False,
        },
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
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return directory, _sha(manifest_path)


def _run(tmp_path: Path, monkeypatch, *, output_name: str = "run"):
    dataset, manifest_sha = _dataset(tmp_path)
    store = tmp_path / "external-experiments"
    store.mkdir()
    monkeypatch.setattr(trainer, "_train_model", lambda _train, _val: FakeBooster())
    result = trainer.train_l2_feedback_model(
        dataset_dir=dataset,
        expected_manifest_sha256=manifest_sha,
        experiment_store=store,
        out_dir=store / output_name,
    )
    return dataset, store, result


def test_trains_binary_feedback_and_writes_deterministic_non_promotable_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    dataset, store, result = _run(tmp_path, monkeypatch, output_name="first")
    out = store / "first"
    assert sorted(path.name for path in out.iterdir()) == [
        "feature_importance.csv", "lineage.json", "metrics.json", "model.txt",
    ]
    lineage = json.loads((out / "lineage.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert lineage["objective"] == "binary"
    assert lineage["best_iteration"] == 7
    assert lineage["feature_columns"] == FEATURE_COLUMNS
    assert lineage["lightgbm_params"]["deterministic"] is True
    assert lineage["lightgbm_params"]["num_threads"] == 1
    assert lineage["input"]["dataset_sha256"] == _sha(dataset / "dataset.csv")
    assert lineage["holdout_rows_read"] == metrics["holdout_rows_read"] == 0
    assert lineage["auto_promote"] is metrics["auto_promote"] is False
    assert metrics["threshold_policy"]["preregistered_reporting_thresholds"] == [.4, .5, .6, .7]
    assert metrics["threshold_policy"]["threshold_tuned_on_val"] is False
    assert metrics["baselines"]["ma_spread_logistic"]["available"] is True
    assert "roc_auc" in metrics["model"] and "pr_auc" in metrics["model"]
    assert metrics["model"]["top_decile"]["returns_are_net"] is True
    assert result["out_dir"] == str(out)

    trainer.train_l2_feedback_model(
        dataset_dir=dataset,
        expected_manifest_sha256=lineage["input"]["manifest_sha256"],
        experiment_store=store,
        out_dir=store / "second",
    )
    for name in ("model.txt", "metrics.json", "feature_importance.csv", "lineage.json"):
        assert (store / "first" / name).read_bytes() == (store / "second" / name).read_bytes()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda data, manifest: data.__setitem__("split", "train"), "split must contain"),
        (lambda data, manifest: data.__setitem__("side", "long"), "side=short"),
        (lambda data, manifest: data.__setitem__("feature_semantics", "legacy_unaligned"), "feature_semantics"),
        (lambda data, manifest: data.__setitem__("holdout_read", True), "holdout_read"),
        (lambda data, manifest: data.__setitem__("future_rows_in_features", 1), "future_rows"),
        (lambda data, manifest: manifest["split"].update(purge_bars=161), "purge_bars"),
    ],
)
def test_dataset_safety_gates_fail_before_training(
    tmp_path: Path, monkeypatch, mutation, match: str
) -> None:
    dataset, _ = _dataset(tmp_path)
    data_path, manifest_path = dataset / "dataset.csv", dataset / "manifest.json"
    data = pd.read_csv(data_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(data, manifest)
    data.to_csv(data_path, index=False)
    manifest["dataset"]["sha256"] = _sha(data_path)
    manifest["dataset"]["rows"] = len(data)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    called = False

    def forbidden(*_args):
        nonlocal called
        called = True
        raise AssertionError("training was reached")

    monkeypatch.setattr(trainer, "_train_model", forbidden)
    with pytest.raises(ValueError, match=match):
        trainer.train_l2_feedback_model(
            dataset_dir=dataset,
            expected_manifest_sha256=_sha(manifest_path),
            experiment_store=tmp_path,
            out_dir=tmp_path / "out",
        )
    assert called is False


def test_manifest_and_dataset_sha_row_and_schema_gates(tmp_path: Path) -> None:
    dataset, manifest_sha = _dataset(tmp_path)
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        trainer.load_feedback_splits(dataset, expected_manifest_sha256="0" * 64)

    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["rows"] += 1
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row count"):
        trainer.load_feedback_splits(dataset, expected_manifest_sha256=_sha(manifest_path))

    dataset, _ = _dataset(tmp_path / "schema")
    data_path, manifest_path = dataset / "dataset.csv", dataset / "manifest.json"
    data = pd.read_csv(data_path).rename(columns={FEATURE_COLUMNS[0]: "wrong_feature"})
    data.to_csv(data_path, index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["sha256"] = _sha(data_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema/order"):
        trainer.load_feedback_splits(dataset, expected_manifest_sha256=_sha(manifest_path))


def test_split_order_purge_partition_size_and_single_class_gates(tmp_path: Path) -> None:
    dataset, _ = _dataset(tmp_path)
    data_path, manifest_path = dataset / "dataset.csv", dataset / "manifest.json"
    data = pd.read_csv(data_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    data.loc[data.split == "train", "label_end_time"] = "2026-01-09T00:00:00Z"
    data.to_csv(data_path, index=False)
    manifest["dataset"]["sha256"] = _sha(data_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="purge gap"):
        trainer.load_feedback_splits(dataset, expected_manifest_sha256=_sha(manifest_path))

    small, small_sha = _dataset(tmp_path / "small", n_train=19)
    with pytest.raises(ValueError, match="train split is too small"):
        trainer.load_feedback_splits(small, expected_manifest_sha256=small_sha)

    one_class, _ = _dataset(tmp_path / "class")
    data_path, manifest_path = one_class / "dataset.csv", one_class / "manifest.json"
    data = pd.read_csv(data_path)
    data.loc[data.split == "val", "label"] = 0
    data.loc[data.split == "val", "outcome"] = "sl"
    data.to_csv(data_path, index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["sha256"] = _sha(data_path)
    manifest["class_balance"]["val"] = {"0": 20, "1": 0}
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="val split must contain both classes"):
        trainer.load_feedback_splits(one_class, expected_manifest_sha256=_sha(manifest_path))


def test_external_store_and_overwrite_refusal_preserve_existing_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    dataset, store, _ = _run(tmp_path, monkeypatch)
    out = store / "run"
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    manifest_sha = _sha(dataset / "manifest.json")
    with pytest.raises(FileExistsError, match="overwrite"):
        trainer.train_l2_feedback_model(
            dataset_dir=dataset, expected_manifest_sha256=manifest_sha,
            experiment_store=store, out_dir=out,
        )
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before

    repo_store = trainer.ROOT / "reports"
    with pytest.raises(ValueError, match="external"):
        trainer.train_l2_feedback_model(
            dataset_dir=dataset, expected_manifest_sha256=manifest_sha,
            experiment_store=repo_store, out_dir=repo_store / "forbidden-run",
        )
