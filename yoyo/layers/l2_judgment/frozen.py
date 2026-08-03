"""Frozen LightGBM artifacts for forward validation.

The project selected tp5_sl2 on the SWAP universe as the current mainline.
This module centralizes artifact discovery, metadata fingerprints, and
frozen-model scoring so dashboards and forward tracking do not retrain.
"""
from __future__ import annotations

import csv

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Mapping, TypedDict

import lightgbm as lgb
import numpy as np
import pandas as pd

from yoyo.layers.l2_judgment.features import FEATURE_COLUMNS
from yoyo.layers.l2_judgment.train import DEFAULT_HORIZON_BARS, load_splits, train_model

PROJECT_DIR: Final = Path(__file__).resolve().parents[2]
BAR: Final = pd.Timedelta(minutes=15)
DEFAULT_SCORE_QUANTILE: Final = 0.90
# Tiered sizing (owner 2026-07-20, analysis/p_weight_centric_val.md):
# val-score quantile bands [q90,q95) / [q95,q99) / q99+ -> notional multiplier.
# Band edges live in the artifact sidecar ("sizing_tiers"); multipliers are a
# fixed owner decision, not a tunable.
TIER_MULTIPLIERS: Final = {"q90_q95": 1.0, "q95_q99": 1.5, "q99_plus": 2.0}
# Mainline 2026-07-31+: v10 detector candidate pool (owner-directed with L1 v10).
# Previous mainline was v11_chain (2026-07-18). Rollback: yolo_v11_pool_config.
DEFAULT_CONFIG_NAME: Final = "tp5_sl2_swap_yolo_v10_reg"
V10_POOL_CONFIG_NAME: Final = DEFAULT_CONFIG_NAME
V11_POOL_CONFIG_NAME: Final = "tp5_sl2_swap_yolo_v11_reg"
# v12 pool artifact name only — never default until owner promotes.
V12_POOL_CONFIG_NAME: Final = "tp5_sl2_swap_yolo_v12_reg"
V8_POOL_CONFIG_NAME: Final = "tp5_sl2_swap_yolo_v8_reg"
# 2026-07-15 mainline (old pool, pre-lr-fix detector); rollback only.
OLD_POOL_CONFIG_NAME: Final = "tp5_sl2_swap_yolo_reg"
# Previous YOLO binary freeze (shadow / rollback / dashboard compare).
BINARY_YOLO_CONFIG_NAME: Final = "tp5_sl2_swap_yolo"
# Legacy rule-scan freeze (pre-cutover); kept for rollback / comparisons.
LEGACY_RULES_CONFIG_NAME: Final = "tp5_sl2_swap"


class ScoreCacheMetadata(TypedDict, total=False):
    threshold: float
    model_path: str
    dataset_path: str
    dataset_sha256: str


@dataclass(frozen=True)
class FrozenConfig:
    __slots__ = (
        "name", "project_dir", "dataset_path", "models_dir",
        "score_quantile", "horizon_bars", "objective", "side",
    )

    name: str
    project_dir: Path
    dataset_path: Path
    models_dir: Path
    score_quantile: float
    horizon_bars: int
    objective: str  # binary | regression
    # Trade direction for forward exit geometry + feature alignment + ledger side.
    # Mainline v10 short_star pool is short (2026-07-31); legacy long configs stay long.
    side: str


@dataclass(frozen=True)
class SizingTiers:
    """Val-score quantile band edges for tiered notional sizing.

    Owner-approved 2026-07-20 (analysis/p_weight_centric_val.md): bands
    [q90,q95) / [q95,q99) / q99+ map to 1x / 1.5x / 2x. q90 is the existing
    entry threshold (threshold_val_q90); q95/q99 come from the same frozen
    val-score distribution and live in the artifact sidecar.
    """

    __slots__ = ("q95", "q99")

    q95: float
    q99: float

    def tier_for_score(self, score: float, threshold: float) -> tuple[str, float]:
        """Map a frozen score to (tier name, notional multiplier).

        Below-threshold scores never trade (multiplier 0.0) — same semantics
        as the experiment's weight function, so a bad caller can only shrink
        exposure, never inflate it.
        """
        if not (score >= threshold):  # catches NaN too
            return "below_q90", 0.0
        if score >= self.q99:
            return "q99_plus", TIER_MULTIPLIERS["q99_plus"]
        if score >= self.q95:
            return "q95_q99", TIER_MULTIPLIERS["q95_q99"]
        return "q90_q95", TIER_MULTIPLIERS["q90_q95"]


@dataclass(frozen=True)
class FrozenArtifact:
    __slots__ = (
        "config",
        "model_path",
        "metadata_path",
        "dataset_path",
        "relative_model_path",
        "relative_dataset_path",
        "threshold",
        "feature_columns",
        "dataset_sha256",
        "dataset_size_bytes",
        "best_iteration",
        "sizing_tiers",
        "feature_semantics",
    )

    config: FrozenConfig
    model_path: Path
    metadata_path: Path
    dataset_path: Path
    relative_model_path: str
    relative_dataset_path: str
    threshold: float
    feature_columns: tuple[str, ...]
    dataset_sha256: str
    dataset_size_bytes: int
    best_iteration: int
    # None when the sidecar predates tiered sizing → everything trades 1x.
    sizing_tiers: SizingTiers | None
    # Which coordinate system the model was TRAINED in. Distinct from trade side:
    # a short artifact can hold either, and serving the wrong one negates six
    # directional features. Absent → legacy_unaligned, which is the truthful
    # reading of every artifact frozen before the field existed.
    feature_semantics: str


class FrozenArtifactError(RuntimeError):
    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


def default_config(project_dir: Path = PROJECT_DIR) -> FrozenConfig:
    """Mainline: regression on the v10 short_star candidate pool (2026-07-31).

    Dataset is built from judgment_v10_wide (net_barrier_taker as realized_ret).
    Pairs with L1 owner_short_star_v10 interim discovery weight.
    """
    return FrozenConfig(
        name=DEFAULT_CONFIG_NAME,
        project_dir=project_dir,
        dataset_path=project_dir / "data" / "judgment_yolo_swap_v10.csv",
        models_dir=project_dir / "models",
        score_quantile=DEFAULT_SCORE_QUANTILE,
        horizon_bars=DEFAULT_HORIZON_BARS,
        objective="regression",
        side="short",
    )


yolo_v10_pool_config = default_config


def yolo_v11_pool_config(project_dir: Path = PROJECT_DIR) -> FrozenConfig:
    """2026-07-18 mainline (v11_chain pool). Rollback after 2026-07-31 v10 cutover."""
    return FrozenConfig(
        name=V11_POOL_CONFIG_NAME,
        project_dir=project_dir,
        dataset_path=project_dir / "data" / "judgment_yolo_swap_v11.csv",
        models_dir=project_dir / "models",
        score_quantile=DEFAULT_SCORE_QUANTILE,
        horizon_bars=DEFAULT_HORIZON_BARS,
        objective="regression",
        side="long",  # historical long barrier ledger; do not mix with short v10
    )


def yolo_v12_pool_config(project_dir: Path = PROJECT_DIR) -> FrozenConfig:
    """v12 H-TIP candidate pool (2026-07-20). Artifact ready; promote needs owner."""
    return FrozenConfig(
        name=V12_POOL_CONFIG_NAME,
        project_dir=project_dir,
        dataset_path=project_dir / "data" / "judgment_yolo_swap_v12.csv",
        models_dir=project_dir / "models",
        score_quantile=DEFAULT_SCORE_QUANTILE,
        horizon_bars=DEFAULT_HORIZON_BARS,
        objective="regression",
        side="short",
    )


def yolo_v8_pool_config(project_dir: Path = PROJECT_DIR) -> FrozenConfig:
    """2026-07-16 mainline (v8_chain pool). Rollback / SHADOW compare only."""
    return FrozenConfig(
        name=V8_POOL_CONFIG_NAME,
        project_dir=project_dir,
        dataset_path=project_dir / "data" / "judgment_yolo_swap_v8.csv",
        models_dir=project_dir / "models",
        score_quantile=DEFAULT_SCORE_QUANTILE,
        horizon_bars=DEFAULT_HORIZON_BARS,
        objective="regression",
        side="long",
    )


def yolo_old_pool_config(project_dir: Path = PROJECT_DIR) -> FrozenConfig:
    """2026-07-15 mainline (old pool, pre-lr-fix detector). Rollback only."""
    return FrozenConfig(
        name=OLD_POOL_CONFIG_NAME,
        project_dir=project_dir,
        dataset_path=project_dir / "data" / "judgment_yolo_swap.csv",
        models_dir=project_dir / "models",
        score_quantile=DEFAULT_SCORE_QUANTILE,
        horizon_bars=DEFAULT_HORIZON_BARS,
        objective="regression",
        side="long",
    )


def binary_yolo_shadow_config(project_dir: Path = PROJECT_DIR) -> FrozenConfig:
    """Previous YOLO binary freeze — shadow compare / emergency rollback."""
    return FrozenConfig(
        name=BINARY_YOLO_CONFIG_NAME,
        project_dir=project_dir,
        dataset_path=project_dir / "data" / "judgment_yolo_swap.csv",
        models_dir=project_dir / "models",
        score_quantile=DEFAULT_SCORE_QUANTILE,
        horizon_bars=DEFAULT_HORIZON_BARS,
        objective="binary",
        side="long",
    )


def rules_legacy_config(project_dir: Path = PROJECT_DIR) -> FrozenConfig:
    """Pre-cutover rule-scan freeze (rollback only)."""
    return FrozenConfig(
        name=LEGACY_RULES_CONFIG_NAME,
        project_dir=project_dir,
        dataset_path=project_dir / "data" / "swap_replication" / "swap_tp5_sl2.csv",
        models_dir=project_dir / "models",
        score_quantile=DEFAULT_SCORE_QUANTILE,
        horizon_bars=DEFAULT_HORIZON_BARS,
        objective="binary",
        side="long",
    )


DEFAULT_FROZEN_CONFIG: Final = default_config()


def latest_artifact(config: FrozenConfig = DEFAULT_FROZEN_CONFIG) -> FrozenArtifact | None:
    # date-suffix only: frozen_{name}_YYYYMMDD.json -- a greedy * here once
    # matched a different config (…_ma206_…) and crashed the dashboard
    pattern = re.compile(rf"^frozen_{re.escape(config.name)}_\d{{8}}\.json$")
    metadata_paths = sorted(
        p for p in config.models_dir.glob(f"frozen_{config.name}_*.json")
        if pattern.match(p.name))
    for path in reversed(metadata_paths):  # newest valid wins; corrupt ones skip
        try:
            return load_artifact(config, path)
        except FrozenArtifactError as exc:
            print(f"frozen: skipping {path.name}: {exc}")
    return None


def config_for_name(config_name: str, project_dir: Path = PROJECT_DIR) -> FrozenConfig:
    """Map freeze meta ``config`` field / artifact stem to a FrozenConfig."""
    name = str(config_name or "").strip()
    table = {
        DEFAULT_CONFIG_NAME: default_config,
        V10_POOL_CONFIG_NAME: default_config,
        V11_POOL_CONFIG_NAME: yolo_v11_pool_config,
        V12_POOL_CONFIG_NAME: yolo_v12_pool_config,
        V8_POOL_CONFIG_NAME: yolo_v8_pool_config,
        OLD_POOL_CONFIG_NAME: yolo_old_pool_config,
        BINARY_YOLO_CONFIG_NAME: binary_yolo_shadow_config,
        LEGACY_RULES_CONFIG_NAME: rules_legacy_config,
    }
    factory = table.get(name)
    if factory is None:
        # Best-effort: treat unknown as default mainline (short v10).
        print(f"frozen: unknown config name {name!r}; using default_config()")
        return default_config(project_dir)
    return factory(project_dir)


def load_runtime_artifact(project_dir: Path = PROJECT_DIR) -> FrozenArtifact | None:
    """Load the owner-facing ACTIVE freeze, else latest default_config artifact.

    ``models/ACTIVE`` may contain a relative path to ``.txt`` or ``.json``
    (e.g. ``models/frozen_…_20260731.txt``). Runtime **must** honor this pointer
    so dashboard ACTIVE and forward pulse cannot diverge silently.
    """
    active_path = project_dir / "models" / "ACTIVE"
    if active_path.is_file():
        raw = active_path.read_text(encoding="utf-8").strip()
        if raw:
            rel = Path(raw)
            candidate = rel if rel.is_absolute() else (project_dir / rel)
            if candidate.suffix == ".txt":
                meta_path = candidate.with_suffix(".json")
            elif candidate.suffix == ".json":
                meta_path = candidate
            else:
                stem = candidate.name
                if stem.endswith(".txt") or stem.endswith(".json"):
                    stem = Path(stem).stem
                meta_path = project_dir / "models" / f"{stem}.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    cfg_name = str(meta.get("config") or DEFAULT_CONFIG_NAME)
                    config = config_for_name(cfg_name, project_dir)
                    art = load_artifact(config, meta_path)
                    print(
                        f"frozen: runtime ACTIVE → {meta_path.name} "
                        f"(config={cfg_name} side={config.side})",
                        flush=True,
                    )
                    return art
                except (OSError, json.JSONDecodeError, FrozenArtifactError) as exc:
                    print(f"frozen: ACTIVE pointer unusable ({exc}); falling back")
            else:
                print(f"frozen: ACTIVE points to missing {meta_path}; falling back")
    # Fallback: newest valid default_config freeze
    art = latest_artifact(default_config(project_dir))
    if art is not None:
        print(
            f"frozen: runtime fallback latest default → {art.metadata_path.name} "
            f"(side={art.config.side})",
            flush=True,
        )
    return art


def load_artifact(config: FrozenConfig, metadata_path: Path) -> FrozenArtifact:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = tuple(metadata["feature_columns"])
    if feature_columns != tuple(FEATURE_COLUMNS):
        raise FrozenArtifactError(metadata_path, "feature list does not match current FEATURE_COLUMNS")
    model_path = _project_path(config, metadata["model_path"])
    if not model_path.exists():
        raise FrozenArtifactError(metadata_path, "model file is missing")
    dataset_path = _project_path(config, metadata["dataset_path"])
    sizing_tiers = _load_sizing_tiers(metadata_path, metadata)
    return FrozenArtifact(
        config=config,
        model_path=model_path,
        metadata_path=metadata_path,
        dataset_path=dataset_path,
        relative_model_path=str(metadata["model_path"]),
        relative_dataset_path=str(metadata["dataset_path"]),
        threshold=float(metadata["threshold_val_q90"]),
        feature_columns=feature_columns,
        dataset_sha256=str(metadata["dataset_sha256"]),
        dataset_size_bytes=int(metadata["dataset_size_bytes"]),
        best_iteration=int(metadata["best_iteration"]),
        sizing_tiers=sizing_tiers,
        feature_semantics=_load_feature_semantics(metadata_path, metadata),
    )


FEATURE_SEMANTICS = ("legacy_unaligned", "side_aligned_v1")


def _load_feature_semantics(metadata_path: Path, metadata: Mapping) -> str:
    """Which extractor produced this model's training features.

    Defaulting to legacy_unaligned is not a guess. Every artifact frozen before
    this field existed was trained on plain extract_feature_rows(), verified on
    2026-08-03 by recomputing the v10 pool both ways: 14 of 14 rows matched the
    plain extractor exactly and the aligned one on none. Defaulting the other way
    would silently feed a short model six negated features.

    An unknown value raises rather than falling back, because the whole point of
    the field is that guessing here is what broke serving.
    """
    raw = metadata.get("feature_semantics")
    if raw is None:
        return "legacy_unaligned"
    value = str(raw).strip()
    if value not in FEATURE_SEMANTICS:
        raise FrozenArtifactError(
            metadata_path,
            f"unknown feature_semantics {value!r}; expected one of {FEATURE_SEMANTICS}",
        )
    return value


def _load_sizing_tiers(metadata_path: Path, metadata: Mapping) -> SizingTiers | None:
    """Optional "sizing_tiers" sidecar block. Missing → None (legacy 1x).

    A malformed block raises: silently trading 1x when the owner enabled
    tiered sizing would misreport live risk, so fail loudly instead.
    """
    raw = metadata.get("sizing_tiers")
    if raw is None:
        return None
    try:
        q95 = float(raw["q95"])
        q99 = float(raw["q99"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FrozenArtifactError(metadata_path, f"bad sizing_tiers block: {exc}")
    threshold = float(metadata["threshold_val_q90"])
    if not (threshold < q95 < q99):
        raise FrozenArtifactError(
            metadata_path, f"sizing_tiers not ordered: q90={threshold} q95={q95} q99={q99}"
        )
    return SizingTiers(q95=q95, q99=q99)


def train_frozen_artifact(config: FrozenConfig, artifact_date: str) -> FrozenArtifact:
    config.models_dir.mkdir(parents=True, exist_ok=True)
    train, val, _ = load_splits(config.dataset_path, horizon_bars=config.horizon_bars)
    model = train_model(train, val, objective=config.objective)
    best_iteration = int(model.best_iteration or model.current_iteration())
    val_scores = model.predict(val[FEATURE_COLUMNS], num_iteration=best_iteration)
    threshold = float(np.quantile(val_scores, config.score_quantile))

    stem = f"frozen_{config.name}_{artifact_date}"
    model_path = config.models_dir / f"{stem}.txt"
    metadata_path = config.models_dir / f"{stem}.json"
    model.save_model(str(model_path), num_iteration=best_iteration)
    metadata = {
        "artifact_version": 1,
        "config": config.name,
        "objective": config.objective,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": _relative_path(config, model_path),
        "dataset_path": _relative_path(config, config.dataset_path),
        "dataset_sha256": file_sha256(config.dataset_path),
        "dataset_size_bytes": config.dataset_path.stat().st_size,
        "threshold_val_q90": threshold,
        "score_quantile": config.score_quantile,
        "feature_columns": list(FEATURE_COLUMNS),
        "best_iteration": best_iteration,
        "splits": {
            "train": _split_summary(train),
            "val": _split_summary(val),
        },
        "holdout_policy": "holdout excluded from training and threshold selection; not evaluated",
        "score_semantics": (
            "predicted_realized_ret" if config.objective == "regression" else "class_probability"
        ),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_artifact(config, metadata_path)


def score_with_artifact(artifact: FrozenArtifact) -> tuple[pd.DataFrame, float]:
    model = lgb.Booster(model_file=str(artifact.model_path))
    full = pd.read_csv(artifact.dataset_path, parse_dates=["signal_time"])
    full["score"] = model.predict(full[list(artifact.feature_columns)], num_iteration=artifact.best_iteration)
    full["entry_time"] = full["signal_time"] + BAR
    full["exit_time"] = full["entry_time"] + full["exit_offset"] * BAR
    return full.sort_values(["entry_time", "score"], ascending=[True, False]), artifact.threshold


def cache_metadata(threshold: float, artifact: FrozenArtifact | None) -> ScoreCacheMetadata:
    metadata: ScoreCacheMetadata = {"threshold": threshold}
    if artifact is not None:
        metadata["model_path"] = artifact.relative_model_path
        metadata["dataset_path"] = artifact.relative_dataset_path
        metadata["dataset_sha256"] = artifact.dataset_sha256
    return metadata


def cache_matches_artifact(
    metadata: Mapping[str, str | float],
    artifact: FrozenArtifact | None,
) -> bool:
    if artifact is None:
        return "model_path" not in metadata
    return (
        metadata.get("model_path") == artifact.relative_model_path
        and metadata.get("dataset_path") == artifact.relative_dataset_path
        and metadata.get("dataset_sha256") == artifact.dataset_sha256
    )



def read_dataset_before(
    path: Path,
    *,
    end_before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read a chronological dataset up to, but never including, a cutoff.

    Used by shadow / q80 tooling so research paths can slice pre-holdout rows
    without loading the full CSV when the tail is holdout-only.
    """
    from yoyo.layers.l2_judgment.train import HOLDOUT_START

    cutoff = HOLDOUT_START if end_before is None else end_before
    safe_rows = 0
    previous_time: pd.Timestamp | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "signal_time" not in (reader.fieldnames or []):
            raise FrozenArtifactError(path, "dataset has no signal_time column")
        for row in reader:
            signal_time = pd.Timestamp(row["signal_time"])
            if previous_time is not None and signal_time < previous_time:
                raise FrozenArtifactError(path, "dataset must be sorted by signal_time")
            previous_time = signal_time
            if signal_time >= cutoff:
                break
            safe_rows += 1
    return pd.read_csv(path, nrows=safe_rows, parse_dates=["signal_time"])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(config: FrozenConfig, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config.project_dir / path


def _relative_path(config: FrozenConfig, path: Path) -> str:
    return path.relative_to(config.project_dir).as_posix()


def _split_summary(frame: pd.DataFrame) -> dict[str, int | list[str]]:
    return {
        "n": int(len(frame)),
        "range": [str(frame["signal_time"].min()), str(frame["signal_time"].max())],
    }
