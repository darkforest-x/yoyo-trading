#!/usr/bin/env python3
"""Build one causally clean stop-loss feedback round from inner mining data.

This is a post-selection materializer, not another optimizer.  It replays the
selected SHORT barrier contract on ``optimization_inner.jsonl``, routes only
SL outcomes into L2 negatives, and creates blind causal W10 images for separate
human L1 review.  Future paths are labels only and never enter feature or image
construction.  The source snapshot is accepted only through the pre-holdout
sidecar SHA/row/max-time gates used by the fixed-W10 backtester.

The output directory is built beside its destination and atomically renamed.
Existing output is never overwritten.  The optional registry prevents the same
event from entering more than one feedback round; exact repeats are audited as
skips while identity collisions fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import shutil
import sys
import uuid
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.backtest_fixed_w10_cls_preholdout import (  # noqa: E402
    HOLDOUT_START,
    load_materialization_manifest,
    load_preholdout_csv,
    sha256_file,
)
from tools.optimize_short_barrier_walkforward import (  # noqa: E402
    BAR_DURATION,
    ENTRY_POLICY,
    GAP_POLICY,
    RETURN_CONVENTION,
    SAME_BAR_POLICY,
    SearchConfig,
    _resolve,
    _exclusive_lock,
    _write_json_atomic,
    gap_deduplicate,
    read_jsonl,
    validate_candidates,
)
from yoyo.contracts.costs import SWAP_MAKER, SWAP_TAKER  # noqa: E402
from yoyo.data.indicators import MIN_GAP_BARS  # noqa: E402
from yoyo.datasets.legacy_gold_migration.renderer import render_w10  # noqa: E402
from yoyo.layers.l2_judgment.features import (  # noqa: E402
    FEATURE_COLUMNS,
    add_features,
    extract_feature_rows_for_semantics,
)

SCHEMA_VERSION = 1
PROTOCOL = "stoploss_feedback_round_v1"
MINING_END_LIMIT = pd.Timestamp("2026-03-31T07:30:00Z")
SEALED_START = pd.Timestamp("2026-04-02T00:00:00Z")
WINDOW_BARS = 10
Y_PAD_FRAC = 0.05
STOP_OUTCOMES = frozenset({"sl", "sl_ambiguous"})


def _utc(value: Any, field: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if pd.isna(stamp):
        raise ValueError(f"invalid {field}: {value!r}")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _candidate_source_sha(selection: Mapping[str, Any], selected_path: Path) -> str:
    """Read the optimizer's candidate digest from selection or sibling prereg."""
    candidates = [
        selection.get("candidate_source_sha256"),
        selection.get("source_sha256"),
    ]
    prereg_path = selected_path.with_name("prereg.json")
    if prereg_path.is_file():
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        if prereg.get("sealed_read") is not False:
            raise ValueError("optimizer prereg does not prove sealed_read=false")
        candidates.append(prereg.get("candidate_source_sha256"))
    declared = {str(value).lower() for value in candidates if value is not None}
    if not declared:
        raise ValueError("selected config has no candidate source SHA (nor sibling prereg.json)")
    if len(declared) != 1:
        raise ValueError("selected config and prereg disagree on candidate source SHA")
    digest = declared.pop()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("invalid candidate source SHA in selected config lineage")
    return digest


def load_selected_config(path: Path, candidates_path: Path) -> tuple[SearchConfig, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("selected config requires schema_version=1")
    if raw.get("sealed") is True or raw.get("sealed_read") is not False:
        raise ValueError("selected config must be an unsealed inner-search artifact")
    try:
        config = SearchConfig(**raw["selected_config"])
    except (KeyError, TypeError) as exc:
        raise ValueError("selected config is missing selected_config") from exc
    if (not math.isfinite(config.threshold) or not 0 <= config.threshold <= 1
            or not math.isfinite(config.tp_atr) or config.tp_atr <= 0
            or not math.isfinite(config.sl_atr) or config.sl_atr <= 0
            or config.horizon <= 0):
        raise ValueError("selected config has invalid parameters")
    source_sha = _candidate_source_sha(raw, path)
    actual_source_sha = sha256_file(candidates_path)
    if actual_source_sha != source_sha:
        raise ValueError("candidate source SHA mismatch against selected config lineage")
    declared_key = raw.get("selected_config_key")
    if declared_key is not None and str(declared_key) != config.key:
        raise ValueError("selected_config_key does not match selected_config")
    return config, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "candidate_source_sha256": actual_source_sha,
    }


def _validate_inner_rows(rows: Sequence[Mapping[str, Any]], mining_end: pd.Timestamp) -> None:
    for row in rows:
        event_id = str(row.get("event_id", "<unknown>"))
        if row.get("sealed") is True or row.get("sealed_read") is True:
            raise ValueError(f"{event_id}: sealed candidate input is forbidden")
        partition = row.get("source_partition", row.get("partition"))
        if partition is not None and str(partition).lower() not in {"inner", "mining"}:
            raise ValueError(f"{event_id}: candidate is not from inner/mining partition")
        decision = _utc(row.get("decision_time"), "decision_time")
        if decision >= SEALED_START:
            raise ValueError(f"{event_id}: decision reaches sealed interval")
        if decision > mining_end:
            raise ValueError(f"{event_id}: decision exceeds mining_end")


def _registry_state(
    path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    if path is None or not path.exists():
        return {"schema_version": 1, "events": {}}, {}, {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("invalid global event registry")
    stored = raw.get("events", raw.get("event_ids", {}))
    if isinstance(stored, list):
        events = {str(event_id): {} for event_id in stored}
    elif isinstance(stored, dict):
        events = {str(event_id): value for event_id, value in stored.items()}
    else:
        raise ValueError("global event registry events must be an object or list")
    identities: dict[str, str] = {}
    for event_id, value in events.items():
        if isinstance(value, Mapping) and value.get("symbol") and value.get("decision_time"):
            key = f"{value['symbol']}\0{_utc(value['decision_time'], 'registry decision_time').isoformat()}"
            if key in identities and identities[key] != event_id:
                raise ValueError("global event registry contains duplicate identities")
            identities[key] = event_id
    return {"schema_version": 1, "events": events}, events, identities


def blind_id(round_salt: str, event_id: str) -> str:
    """Return a stable opaque identifier without symbol/time/outcome/score."""
    digest = hmac.new(round_salt.encode("ascii"), str(event_id).encode("utf-8"), hashlib.sha256)
    return "blind_" + digest.hexdigest()[:24]


def _round_salt(candidate_sha: str, selection_sha: str) -> str:
    material = f"{PROTOCOL}\0{candidate_sha}\0{selection_sha}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def _load_frames(
    snapshot_dir: Path,
    manifest_path: Path,
    symbols: set[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    declarations = load_materialization_manifest(snapshot_dir, manifest_path)
    frames: dict[str, pd.DataFrame] = {}
    lineage: dict[str, dict[str, Any]] = {}
    for item in declarations:
        path = snapshot_dir / str(item["path"])
        symbol = str(item.get("symbol") or path.stem)
        if symbol not in symbols:
            continue
        if symbol in frames:
            raise ValueError(f"duplicate snapshot symbol: {symbol}")
        frames[symbol] = load_preholdout_csv(path, item)
        lineage[symbol] = {
            "snapshot_file": str(item["path"]),
            "snapshot_file_sha256": str(item["sha256"]),
            "snapshot_rows": int(item["rows"]),
            "snapshot_max_materialized_time": str(item["max_materialized_time"]),
        }
    missing = sorted(symbols - set(frames))
    if missing:
        raise ValueError(f"candidate symbols missing from snapshot manifest: {missing}")
    return frames, lineage


def _decision_position(frame: pd.DataFrame, symbol: str, decision: pd.Timestamp) -> int:
    times = pd.to_datetime(frame["open_time"], utc=True)
    positions = np.flatnonzero((times == decision).to_numpy())
    if len(positions) != 1:
        raise ValueError(
            f"{symbol} {decision.isoformat()}: expected exactly one snapshot decision bar, got {len(positions)}"
        )
    return int(positions[0])


def _causal_features_and_window(
    frame: pd.DataFrame,
    symbol: str,
    decision: pd.Timestamp,
) -> tuple[dict[str, float], pd.DataFrame]:
    decision_i = _decision_position(frame, symbol, decision)
    if decision_i < WINDOW_BARS - 1:
        raise ValueError(f"{symbol} {decision.isoformat()}: fewer than W10 causal bars")
    # Slicing before feature extraction makes the no-future boundary explicit,
    # even though all upstream rolling/EMA calculations are themselves causal.
    causal = frame.iloc[: decision_i + 1].copy()
    featured = add_features(causal)
    vector = extract_feature_rows_for_semantics(
        featured,
        [len(featured) - 1],
        feature_semantics="side_aligned_v1",
        side="short",
    )
    if list(vector.columns) != FEATURE_COLUMNS or vector.shape != (1, len(FEATURE_COLUMNS)):
        raise ValueError("L2 feature schema is not the frozen 28-column contract")
    values = vector.iloc[0]
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        missing = [name for name in FEATURE_COLUMNS if pd.isna(values[name])]
        raise ValueError(f"{symbol} {decision.isoformat()}: missing/non-finite L2 features: {missing}")
    window = causal.iloc[-WINDOW_BARS:].copy()
    if pd.to_datetime(window["open_time"], utc=True).max() != decision:
        raise ValueError("causal W10 does not end at the decision bar")
    return {name: float(values[name]) for name in FEATURE_COLUMNS}, window


def _registry_skip(
    row: Mapping[str, Any], registered_events: Mapping[str, Any], identities: Mapping[str, str]
) -> bool:
    event_id = str(row["event_id"])
    decision = _utc(row["decision_time"], "decision_time").isoformat()
    identity = f"{row['symbol']}\0{decision}"
    existing = identities.get(identity)
    if existing is not None and existing != event_id:
        raise ValueError(f"registry identity collision: {row['symbol']} {decision}")
    if event_id not in registered_events:
        return False
    prior = registered_events[event_id]
    if isinstance(prior, Mapping) and prior.get("symbol") and prior.get("decision_time"):
        prior_identity = f"{prior['symbol']}\0{_utc(prior['decision_time'], 'registry decision_time').isoformat()}"
        if prior_identity != identity:
            raise ValueError(f"registry event_id collision: {event_id}")
    return True


def build_feedback_round(
    *,
    candidates_path: Path,
    selected_config_path: Path,
    snapshot_dir: Path,
    materialization_manifest: Path,
    out_dir: Path,
    global_event_registry: Path | None = None,
    mining_end: Any = MINING_END_LIMIT,
) -> dict[str, Any]:
    """Build and atomically publish one stop-loss feedback round."""
    candidates_path = candidates_path.resolve()
    selected_config_path = selected_config_path.resolve()
    snapshot_dir = snapshot_dir.resolve()
    materialization_manifest = materialization_manifest.resolve()
    out_dir = out_dir.resolve()
    mining_end_ts = _utc(mining_end, "mining_end")
    if mining_end_ts > MINING_END_LIMIT:
        raise ValueError(f"mining_end may not exceed {MINING_END_LIMIT.isoformat()}")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    if not candidates_path.is_file() or not selected_config_path.is_file():
        raise ValueError("candidates and selected config must exist")

    config, selection_lineage = load_selected_config(selected_config_path, candidates_path)
    raw_rows = read_jsonl(candidates_path)
    _validate_inner_rows(raw_rows, mining_end_ts)
    rows = validate_candidates(raw_rows, max_horizon=config.horizon)
    for row in rows:
        label_end = pd.Timestamp(row["_path_frame"]["open_time"].iloc[config.horizon - 1]) + BAR_DURATION
        if label_end > mining_end_ts:
            raise ValueError(f"{row['event_id']}: label_end exceeds mining_end")
        row["_label_end"] = label_end

    chosen = gap_deduplicate(rows, config.threshold)
    chosen.sort(key=lambda row: (_utc(row["decision_time"], "decision_time"),
                                 str(row["symbol"]), str(row["event_id"])))
    resolved_rows: list[dict[str, Any]] = []
    outcome_counts: dict[str, int] = {}
    for row in chosen:
        resolved = _resolve(row, config)
        outcome = str(resolved["outcome"])
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        if outcome in STOP_OUTCOMES:
            resolved_rows.append({**row, **resolved})

    candidate_sha = selection_lineage["candidate_source_sha256"]
    salt = _round_salt(candidate_sha, selection_lineage["sha256"])
    manifest_sha = sha256_file(materialization_manifest)

    temp_dir = out_dir.with_name(f".{out_dir.name}.tmp-{uuid.uuid4().hex}")
    registry_path = global_event_registry.resolve() if global_event_registry is not None else None
    lock_context = (
        nullcontext()
        if registry_path is None
        else _exclusive_lock(registry_path.with_name(registry_path.name + ".lock"))
    )
    try:
        with lock_context:
            registry, registered_events, registered_identities = _registry_state(registry_path)
            skipped_registry: list[str] = []
            accepted: list[dict[str, Any]] = []
            for row in resolved_rows:
                if _registry_skip(row, registered_events, registered_identities):
                    skipped_registry.append(str(row["event_id"]))
                else:
                    accepted.append(row)

            if registry_path is not None:
                registry_path.parent.mkdir(parents=True, exist_ok=True)
                for row in accepted:
                    event_id = str(row["event_id"])
                    registry["events"][event_id] = {
                        "status": "pending",
                        "symbol": str(row["symbol"]),
                        "decision_time": _utc(row["decision_time"], "decision_time").isoformat(),
                        "outcome": str(row["outcome"]),
                        "round": str(out_dir),
                    }
                _write_json_atomic(registry_path, registry)

            frames, source_lineage = _load_frames(
                snapshot_dir, materialization_manifest, {str(row["symbol"]) for row in accepted}
            )
            stop_losses: list[dict[str, Any]] = []
            l2_rows: list[dict[str, Any]] = []
            review_rows: list[dict[str, Any]] = []
            private_rows: list[dict[str, Any]] = []

            images_dir = temp_dir / "l1_blind_review" / "images"
            images_dir.mkdir(parents=True)
            for row in accepted:
                event_id = str(row["event_id"])
                symbol = str(row["symbol"])
                decision = _utc(row["decision_time"], "decision_time")
                features, window = _causal_features_and_window(frames[symbol], symbol, decision)
                opaque_id = blind_id(salt, event_id)
                image, _ = render_w10(window, y_pad_frac=Y_PAD_FRAC, overlay=False)
                image_path = images_dir / f"{opaque_id}.png"
                if not cv2.imwrite(str(image_path), image):
                    raise OSError(f"failed to write review image: {image_path}")
                image_sha = sha256_file(image_path)
                selected_config = asdict(config)
                label_end = pd.Timestamp(row["_label_end"]).isoformat()
                common = {
                    "protocol": PROTOCOL,
                    "event_id": event_id,
                    "symbol": symbol,
                    "decision_time": decision.isoformat(),
                    "outcome": row["outcome"],
                    "label_end_time": label_end,
                    "selected_config": selected_config,
                    "holdout_read": False,
                    "future_data_in_causal_input": False,
                    "future_label_used": True,
                }
                stop_losses.append({
                    **common,
                    "p_signal": float(row["p_signal"]),
                    "entry_price": float(row["entry_price"]),
                    "atr14": float(row["atr14"]),
                    "exit_time": row["exit_time"],
                    "gross_ret": float(row["gross_ret"]),
                    "net_maker": float(row["net_maker"]),
                    "net_taker": float(row["net_taker"]),
                    "l2_outcome_negative": True,
                    "automatic_l1_label": None,
                    "candidate_source_sha256": candidate_sha,
                    "selected_config_sha256": selection_lineage["sha256"],
                    "materialization_manifest_sha256": manifest_sha,
                    **source_lineage[symbol],
                })
                l2_rows.append({
                    **common,
                    "l2_target": 0,
                    "l2_outcome_negative": True,
                    "feature_semantics": "side_aligned_v1",
                    "feature_columns": list(FEATURE_COLUMNS),
                    "features": features,
                    "feature_as_of": decision.isoformat(),
                    "training_eligible": True,
                    "training_scope": "l2_only",
                })
                review_rows.append({
                    "blind_id": opaque_id,
                    "image_path": f"images/{opaque_id}.png",
                    "image_sha256": image_sha,
                    "class_status": "unreviewed",
                    "box_status": "not_applicable",
                    "training_eligible": False,
                    "suggested_label": "REVIEW_REQUIRED",
                    "future_used_in_training_image": False,
                    "holdout_read": False,
                })
                private_rows.append({
                    "blind_id": opaque_id,
                    "event_id": event_id,
                    "symbol": symbol,
                    "decision_time": decision.isoformat(),
                    "outcome": row["outcome"],
                    "p_signal": float(row["p_signal"]),
                    "future_review_only": True,
                    "label_end_time": label_end,
                })

            _write_jsonl(temp_dir / "stop_losses.jsonl", stop_losses)
            _write_jsonl(temp_dir / "l2_outcome_negatives.jsonl", l2_rows)
            _write_jsonl(temp_dir / "l1_blind_review" / "review_manifest.jsonl", review_rows)
            _write_jsonl(temp_dir / "lineage_private.jsonl", private_rows)
            artifacts = {
                "stop_losses.jsonl": len(stop_losses),
                "l2_outcome_negatives.jsonl": len(l2_rows),
                "l1_blind_review/review_manifest.jsonl": len(review_rows),
                "lineage_private.jsonl": len(private_rows),
            }
            artifact_shas = {name: sha256_file(temp_dir / name) for name in artifacts}
            summary = {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "counts": {
                    "input_candidates": len(rows),
                    "threshold_gap_selected": len(chosen),
                    "selected_outcomes": outcome_counts,
                    "stop_losses_before_registry": len(resolved_rows),
                    "registry_skipped": len(skipped_registry),
                    "stop_losses_materialized": len(stop_losses),
                },
                "artifacts": {name: {"rows": count, "sha256": artifact_shas[name]}
                              for name, count in artifacts.items()},
                "candidates": {"path": str(candidates_path), "sha256": candidate_sha},
                "selected_config_artifact": selection_lineage,
                "selected_config": asdict(config),
                "snapshot_dir": str(snapshot_dir),
                "materialization_manifest": {
                    "path": str(materialization_manifest), "sha256": manifest_sha,
                },
                "round_salt_sha256": hashlib.sha256(salt.encode("ascii")).hexdigest(),
                "mining_end": mining_end_ts.isoformat(),
                "protocol_contract": {
                    "side": "short", "entry_policy": ENTRY_POLICY,
                    "same_bar_policy": SAME_BAR_POLICY, "gap_policy": GAP_POLICY,
                    "return_convention": RETURN_CONVENTION, "min_gap_bars": MIN_GAP_BARS,
                    "feature_semantics": "side_aligned_v1", "feature_count": len(FEATURE_COLUMNS),
                    "window_bars": WINDOW_BARS, "renderer_overlay": False,
                    "renderer_y_pad_frac": Y_PAD_FRAC,
                    "costs": {"swap_maker": SWAP_MAKER, "swap_taker": SWAP_TAKER},
                },
                "registry": {
                    "enabled": registry_path is not None,
                    "path": None if registry_path is None else str(registry_path),
                    "skipped_event_ids": skipped_registry,
                    "claimed_event_ids": [str(row["event_id"]) for row in accepted],
                    "claim_state": "claimed_before_publish" if registry_path is not None else "disabled",
                },
                "holdout_start": HOLDOUT_START.isoformat(),
                "holdout_rows_read": 0,
                "future_rows_in_causal_input": 0,
                "auto_promote": False,
            }
            _write_json(temp_dir / "summary.json", summary)

            os.replace(temp_dir, out_dir)
            if registry_path is not None:
                for row in stop_losses:
                    registry["events"][row["event_id"]] = {
                        "status": "complete",
                        "symbol": row["symbol"], "decision_time": row["decision_time"],
                        "outcome": row["outcome"], "round": str(out_dir),
                        "round_summary": str(out_dir / "summary.json"),
                    }
                _write_json_atomic(registry_path, registry)
            return summary
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--global-event-registry", type=Path)
    parser.add_argument("--mining-end", default=MINING_END_LIMIT.isoformat())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = build_feedback_round(
        candidates_path=args.candidates,
        selected_config_path=args.selected_config,
        snapshot_dir=args.snapshot_dir,
        materialization_manifest=args.materialization_manifest,
        out_dir=args.out_dir,
        global_event_registry=args.global_event_registry,
        mining_end=args.mining_end,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
