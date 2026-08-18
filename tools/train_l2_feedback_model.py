#!/usr/bin/env python3
"""Train one non-deployable LightGBM model from an INNER feedback dataset.

This entry point intentionally does not use the historical judgment training
CLI: that CLI discovers its own time split and has an optional holdout path.
Here, the dataset builder's explicit train/validation split is treated as an
input contract and is revalidated before any feature values reach LightGBM.
Outputs are written only to a caller-declared experiment store, atomically,
and carry ``auto_promote=false`` and ``holdout_rows_read=0``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_l2_feedback_dataset import (  # noqa: E402
    DEFAULT_PURGE_BARS,
    FEATURE_SEMANTICS,
    HOLDOUT_START,
    INCLUDED_OUTCOMES,
    INNER_CUTOFF,
    PROTOCOL as DATASET_PROTOCOL,
    SCHEMA_VERSION as DATASET_SCHEMA_VERSION,
)
from yoyo.layers.l2_judgment.features import FEATURE_COLUMNS  # noqa: E402
from yoyo.layers.l2_judgment.train import evaluate  # noqa: E402

BAR_DURATION = pd.Timedelta(minutes=15)
MIN_PARTITION_ROWS = 20
THRESHOLDS = (0.4, 0.5, 0.6, 0.7)
OUTPUT_SCHEMA_VERSION = 1
OUTPUT_PROTOCOL = "l2_feedback_lightgbm_v1"

# One-threaded deterministic mode is slower than opportunistic threading, but
# makes this small feedback-round artifact byte/review reproducible.
LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
    "data_random_seed": 42,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 1,
    "verbosity": -1,
}
NUM_BOOST_ROUND = 600
EARLY_STOPPING_ROUNDS = 50

DATASET_COLUMNS = [
    "event_id", "symbol", "signal_time", "side", "split", "label",
    "realized_ret", "net_barrier_maker", "net_barrier_taker", "outcome",
    "feature_semantics", *FEATURE_COLUMNS, "feature_as_of", "label_end_time",
    "selected_config_key", "candidate_source_sha256", "selected_config_sha256",
    "prereg_sha256", "snapshot_manifest_sha256", "snapshot_file",
    "snapshot_file_sha256", "snapshot_file_rows", "snapshot_max_materialized_time",
    "holdout_read", "future_rows_in_features", "future_review_only", "auto_promote",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _utc_series(values: pd.Series, field: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid UTC timestamps in {field}") from exc
    if parsed.isna().any():
        raise ValueError(f"null timestamps in {field}")
    return parsed


def _false_only(values: pd.Series, field: str) -> None:
    normalized = values.map(lambda value: str(value).strip().lower())
    if not normalized.eq("false").all():
        raise ValueError(f"{field} must be false for every row")


def _true_only(values: pd.Series, field: str) -> None:
    normalized = values.map(lambda value: str(value).strip().lower())
    if not normalized.eq("true").all():
        raise ValueError(f"{field} must be true for every row")


def _time_range(frame: pd.DataFrame) -> list[str]:
    return [frame["signal_time"].min().isoformat(), frame["signal_time"].max().isoformat()]


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(label): int((frame["label"] == label).sum()) for label in (0, 1)}


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest {field} must be an object")
    return value


def load_feedback_splits(
    dataset_dir: Path, *, expected_manifest_sha256: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, str]]:
    """Verify builder lineage and return the predeclared INNER-only splits."""
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.json"
    dataset_path = dataset_dir / "dataset.csv"
    expected = expected_manifest_sha256.strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("expected manifest SHA256 must be 64 lowercase hex characters")
    if not manifest_path.is_file() or not dataset_path.is_file():
        raise ValueError("dataset directory must contain flat manifest.json and dataset.csv")
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != expected:
        raise ValueError("input manifest SHA256 mismatch")

    manifest = _json_object(manifest_path)
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"manifest requires schema_version={DATASET_SCHEMA_VERSION}")
    if manifest.get("protocol") != DATASET_PROTOCOL:
        raise ValueError(f"manifest requires protocol={DATASET_PROTOCOL}")
    dataset_decl = _require_mapping(manifest.get("dataset"), "dataset")
    if dataset_decl.get("path") != "dataset.csv":
        raise ValueError("manifest dataset.path must be the flat dataset.csv")
    dataset_sha = _sha256(dataset_path)
    if dataset_decl.get("sha256") != dataset_sha:
        raise ValueError("dataset.csv SHA256 mismatch")
    if type(dataset_decl.get("rows")) is not int or dataset_decl["rows"] <= 0:
        raise ValueError("manifest dataset.rows must be a positive integer")

    contract = _require_mapping(manifest.get("protocol_contract"), "protocol_contract")
    if contract.get("side") != "short":
        raise ValueError("manifest protocol requires side=short")
    if contract.get("feature_semantics") != FEATURE_SEMANTICS:
        raise ValueError(f"manifest requires feature_semantics={FEATURE_SEMANTICS}")
    if contract.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("manifest feature_columns must exactly equal FEATURE_COLUMNS")
    if contract.get("included_outcomes") != INCLUDED_OUTCOMES:
        raise ValueError("manifest outcome-label contract mismatch")
    if manifest.get("holdout_rows_read") != 0:
        raise ValueError("manifest must prove holdout_rows_read=0")
    if manifest.get("future_rows_in_features") != 0:
        raise ValueError("manifest must prove future_rows_in_features=0")
    if manifest.get("auto_promote") is not False:
        raise ValueError("manifest must prove auto_promote=false")

    split_decl = _require_mapping(manifest.get("split"), "split")
    purge_bars = split_decl.get("purge_bars")
    if type(purge_bars) is not int or purge_bars < DEFAULT_PURGE_BARS:
        raise ValueError(f"manifest must prove purge_bars >= {DEFAULT_PURGE_BARS}")
    if split_decl.get("method") != "explicit_time_boundary" or split_decl.get("random") is not False:
        raise ValueError("manifest must prove an explicit non-random time split")
    try:
        val_start = pd.Timestamp(split_decl["val_start"])
        val_start = val_start.tz_localize("UTC") if val_start.tzinfo is None else val_start.tz_convert("UTC")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("manifest split.val_start is invalid") from exc

    try:
        data = pd.read_csv(dataset_path)
    except Exception as exc:
        raise ValueError(f"cannot read dataset.csv: {dataset_path}") from exc
    if len(data) != dataset_decl["rows"]:
        raise ValueError("dataset.csv row count does not match manifest")
    if list(data.columns) != DATASET_COLUMNS:
        raise ValueError("dataset.csv schema/order does not exactly match feedback schema")
    if data.empty:
        raise ValueError("feedback dataset is empty")

    data["signal_time"] = _utc_series(data["signal_time"], "signal_time")
    data["feature_as_of"] = _utc_series(data["feature_as_of"], "feature_as_of")
    data["label_end_time"] = _utc_series(data["label_end_time"], "label_end_time")
    data["snapshot_max_materialized_time"] = _utc_series(
        data["snapshot_max_materialized_time"], "snapshot_max_materialized_time"
    )
    if not data["signal_time"].is_monotonic_increasing:
        raise ValueError("dataset rows must be ordered by signal_time")
    event_ids = data["event_id"].astype("string")
    symbols = data["symbol"].astype("string")
    if (event_ids.isna().any() or event_ids.str.strip().eq("").any()
            or event_ids.duplicated().any()):
        raise ValueError("event_id must be non-null and unique")
    if symbols.isna().any() or symbols.str.strip().eq("").any():
        raise ValueError("symbol must be non-null for identity verification")
    if data[["symbol", "signal_time"]].duplicated().any():
        raise ValueError("symbol/signal_time identity must be unique")
    if set(data["split"].astype(str)) != {"train", "val"}:
        raise ValueError("split must contain train|val and no other values")
    if set(data["side"].astype(str)) != {"short"}:
        raise ValueError("every row must have side=short")
    if set(data["feature_semantics"].astype(str)) != {FEATURE_SEMANTICS}:
        raise ValueError(f"every row must have feature_semantics={FEATURE_SEMANTICS}")

    numeric_labels = pd.to_numeric(data["label"], errors="coerce")
    if numeric_labels.isna().any() or not numeric_labels.isin([0, 1]).all():
        raise ValueError("labels must be binary integers")
    data["label"] = numeric_labels.astype(int)
    expected_labels = data["outcome"].map(INCLUDED_OUTCOMES)
    if expected_labels.isna().any() or not expected_labels.astype(int).eq(data["label"]).all():
        raise ValueError("labels must map tp=1 and sl/sl_ambiguous=0 exactly")

    features = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if features.shape[1] != len(FEATURE_COLUMNS) or not np.isfinite(features).all():
        raise ValueError("all and only FEATURE_COLUMNS must be finite numeric model inputs")
    data.loc[:, FEATURE_COLUMNS] = features
    returns = pd.to_numeric(data["net_barrier_maker"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(returns).all():
        raise ValueError("net_barrier_maker must be finite")
    data["net_barrier_maker"] = returns

    _false_only(data["holdout_read"], "holdout_read")
    _false_only(data["auto_promote"], "auto_promote")
    _true_only(data["future_review_only"], "future_review_only")
    future_counts = pd.to_numeric(data["future_rows_in_features"], errors="coerce")
    if future_counts.isna().any() or not future_counts.eq(0).all():
        raise ValueError("future_rows_in_features must be zero for every row")
    if not data["feature_as_of"].eq(data["signal_time"]).all():
        raise ValueError("feature_as_of must exactly equal causal signal_time")

    try:
        declared_inner = pd.Timestamp(manifest["inner_cutoff"])
        declared_holdout = pd.Timestamp(manifest["holdout_start"])
        declared_inner = declared_inner.tz_localize("UTC") if declared_inner.tzinfo is None else declared_inner.tz_convert("UTC")
        declared_holdout = declared_holdout.tz_localize("UTC") if declared_holdout.tzinfo is None else declared_holdout.tz_convert("UTC")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("manifest inner/holdout boundaries are invalid") from exc
    if declared_inner != INNER_CUTOFF:
        raise ValueError("manifest inner_cutoff does not match the frozen INNER boundary")
    if declared_holdout != HOLDOUT_START:
        raise ValueError("manifest holdout_start does not match the frozen holdout boundary")
    inner_cutoff = INNER_CUTOFF
    holdout_start = HOLDOUT_START
    if (data["signal_time"] >= inner_cutoff).any() or (data["label_end_time"] >= inner_cutoff).any():
        raise ValueError("dataset reaches sealed inner cutoff")
    if (data["snapshot_max_materialized_time"] >= inner_cutoff).any():
        raise ValueError("dataset snapshot lineage reaches sealed inner cutoff")
    if (data["signal_time"] >= holdout_start).any():
        raise ValueError("dataset contains holdout rows")

    train = data[data["split"] == "train"].copy().reset_index(drop=True)
    val = data[data["split"] == "val"].copy().reset_index(drop=True)
    for name, frame in (("train", train), ("val", val)):
        if len(frame) < MIN_PARTITION_ROWS:
            raise ValueError(
                f"{name} split is too small: {len(frame)} rows; require >= {MIN_PARTITION_ROWS}"
            )
        if set(frame["label"]) != {0, 1}:
            raise ValueError(f"{name} split must contain both classes")
    if train["signal_time"].max() >= val["signal_time"].min():
        raise ValueError("all train signal_time values must be strictly earlier than val")
    purge_boundary = val_start - purge_bars * BAR_DURATION
    if val["signal_time"].min() < val_start:
        raise ValueError("validation rows precede manifest val_start")
    if not (train["label_end_time"] < purge_boundary).all():
        raise ValueError("train label windows do not prove the declared purge gap")

    counts = _require_mapping(manifest.get("counts"), "counts")
    if counts.get("dataset_rows") != len(data) or counts.get("train_rows") != len(train) or counts.get("val_rows") != len(val):
        raise ValueError("manifest split counts do not match dataset.csv")
    balance = _require_mapping(manifest.get("class_balance"), "class_balance")
    if balance.get("train") != _class_counts(train) or balance.get("val") != _class_counts(val):
        raise ValueError("manifest class_balance does not match dataset.csv")

    return train, val, manifest, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha,
    }


def _train_model(train: pd.DataFrame, val: pd.DataFrame):
    import lightgbm as lgb

    dtrain = lgb.Dataset(train[FEATURE_COLUMNS], label=train["label"], free_raw_data=False)
    dval = lgb.Dataset(val[FEATURE_COLUMNS], label=val["label"], reference=dtrain, free_raw_data=False)
    return lgb.train(
        dict(LGB_PARAMS), dtrain, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )


def _metric_block(y: np.ndarray, score: np.ndarray, net_maker: np.ndarray) -> dict[str, Any]:
    metrics = evaluate(y, score, net_maker, objective="binary", returns_are_net=True)
    metrics["top_decile"]["mean_net_barrier_maker"] = metrics["top_decile"][
        "mean_realized_ret"
    ]
    return metrics


def _ma_spread_baseline(train: pd.DataFrame, val: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    values = train["ma_spread_pct"].to_numpy(dtype=float)
    if np.unique(values).size < 2:
        return None, "ma_spread_pct is constant in train"
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[["ma_spread_pct"]])
        model = LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000)
        model.fit(x_train, train["label"])
        return model.predict_proba(scaler.transform(val[["ma_spread_pct"]]))[:, 1], None
    except Exception as exc:  # baseline absence must be explicit, not fatal to the primary model
        return None, f"{type(exc).__name__}: {exc}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def train_l2_feedback_model(
    *, dataset_dir: Path, expected_manifest_sha256: str,
    experiment_store: Path, out_dir: Path,
) -> dict[str, Any]:
    """Validate, train, evaluate, and atomically publish a feedback artifact."""
    store = experiment_store.resolve()
    out = out_dir.resolve()
    if not store.is_dir():
        raise ValueError("--experiment-store must be an existing directory")
    try:
        out.relative_to(store)
    except ValueError as exc:
        raise ValueError("--out-dir must be under --experiment-store") from exc
    try:
        store.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("experiment store must be external to the code repository")
    if out == store:
        raise ValueError("--out-dir must name a fresh child of --experiment-store")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out}")

    train, val, manifest, lineage_input = load_feedback_splits(
        dataset_dir, expected_manifest_sha256=expected_manifest_sha256
    )
    model = _train_model(train, val)
    best_iteration = int(getattr(model, "best_iteration", 0) or getattr(model, "current_iteration")())
    if best_iteration <= 0:
        raise ValueError("LightGBM returned no positive best_iteration")
    val_score = np.asarray(
        model.predict(val[FEATURE_COLUMNS], num_iteration=best_iteration), dtype=float
    )
    if val_score.shape != (len(val),) or not np.isfinite(val_score).all():
        raise ValueError("LightGBM produced invalid validation probabilities")

    y_val = val["label"].to_numpy(dtype=int)
    net_val = val["net_barrier_maker"].to_numpy(dtype=float)
    train_rate = float(train["label"].mean())
    constant_score = np.full(len(val), train_rate, dtype=float)
    ma_score, ma_reason = _ma_spread_baseline(train, val)
    metrics: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "protocol": OUTPUT_PROTOCOL,
        "objective": "binary",
        "validation_only": True,
        "threshold_policy": {
            "preregistered_reporting_thresholds": list(THRESHOLDS),
            "threshold_tuned_on_val": False,
            "top_decile": "reporting_only",
        },
        "model": _metric_block(y_val, val_score, net_val),
        "baselines": {
            "constant_train_positive_rate": {
                "train_positive_rate": train_rate,
                "metrics": _metric_block(y_val, constant_score, net_val),
            },
            "ma_spread_logistic": (
                {"available": True, "metrics": _metric_block(y_val, ma_score, net_val)}
                if ma_score is not None else {"available": False, "reason": ma_reason}
            ),
        },
        "economic_return_column": "net_barrier_maker",
        "holdout_rows_read": 0,
        "auto_promote": False,
    }
    lineage = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "protocol": OUTPUT_PROTOCOL,
        "input": lineage_input,
        "input_dataset_protocol": manifest["protocol"],
        "objective": "binary",
        "best_iteration": best_iteration,
        "lightgbm_params": dict(LGB_PARAMS),
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_semantics": FEATURE_SEMANTICS,
        "side": "short",
        "splits": {
            "train": {"rows": len(train), "class_counts": _class_counts(train), "time_range": _time_range(train)},
            "val": {"rows": len(val), "class_counts": _class_counts(val), "time_range": _time_range(val)},
        },
        "purge_bars": int(manifest["split"]["purge_bars"]),
        "holdout_rows_read": 0,
        "auto_promote": False,
    }
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "gain": np.asarray(model.feature_importance(importance_type="gain"), dtype=float),
        "split": np.asarray(model.feature_importance(importance_type="split"), dtype=float),
    }).sort_values(["gain", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    if len(importance) != len(FEATURE_COLUMNS) or not np.isfinite(importance[["gain", "split"]]).to_numpy().all():
        raise ValueError("LightGBM produced invalid feature importance")

    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(f".{out.name}.tmp-{uuid.uuid4().hex}")
    try:
        temp.mkdir()
        model.save_model(str(temp / "model.txt"), num_iteration=best_iteration)
        _write_json(temp / "metrics.json", metrics)
        importance.to_csv(temp / "feature_importance.csv", index=False, lineterminator="\n")
        _write_json(temp / "lineage.json", lineage)
        os.replace(temp, out)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return {"metrics": metrics, "lineage": lineage, "out_dir": str(out)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True,
                        help="expected SHA256 of input manifest.json")
    parser.add_argument("--experiment-store", type=Path, required=True,
                        help="existing experiment root outside this code repository")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="fresh artifact directory under --experiment-store")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = train_l2_feedback_model(
        dataset_dir=args.dataset_dir,
        expected_manifest_sha256=args.manifest_sha256,
        experiment_store=args.experiment_store,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
