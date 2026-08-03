"""Causal, read-only bridge from an L1 candidate to one frozen L2 score.

The historical full-window detector attached boxes to bars inside an image and
then treated that box time as if the signal had existed then.  That backdated
the L2 feature row and made an offline result look tradable before the detector
could actually have emitted it.  This bridge makes the opposite contract
executable: ``available_at`` is the candidate's only fact time.  Box timestamps,
when present in a manifest, are descriptive metadata and are never consulted.

Inputs are caller-owned objects only: a candidate mapping, already supplied 15m
bars, an explicit :class:`FrozenArtifact`, and an already constructed booster.
There is deliberately no ACTIVE/default discovery, artifact loader, training
path, market-data loader, or read of ``artifact.dataset_path`` in this module.
The companion schema-v2 adapter is a pure mapping-to-mappings transform: it
does not accept paths or perform any file I/O, and rejects attempts to inject a
backfilled ``signal_time`` into a YOLO detection.

Feature sources are causal.  Only rows whose ``open_time + 15m <= available_at``
are retained before indicators or features are computed.  The selected feature
row is the newest such closed bar.  Its 28 features use only OHLCV columns from
that row and earlier rows via ``add_indicators`` and ``add_features``; see those
functions for each rolling/EMA window.  Appending or changing later rows cannot
change a score at a fixed ``available_at``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, TypedDict

import numpy as np
import pandas as pd

from yoyo.contracts.protocol import file_sha256
from yoyo.data.bars import bar_to_timedelta, normalize_bar
from yoyo.data.indicators import WARMUP_BARS, add_indicators
from yoyo.layers.l2_judgment.features import (
    FEATURE_COLUMNS,
    add_features,
    extract_feature_rows_for_semantics,
)

if TYPE_CHECKING:  # Avoid importing frozen.py's runtime loaders in this bridge.
    from yoyo.layers.l2_judgment.frozen import FrozenArtifact


BAR_15M = pd.Timedelta(minutes=15)
BAR_30M = pd.Timedelta(minutes=30)
REQUIRED_CANDIDATE_FIELDS = (
    "candidate_id",
    "source",
    "symbol",
    "detector_timeframe",
    "available_at",
    "window_end_close_time",
    "weights_sha256",
    "dataset_manifest_sha256",
    "image_sha256",
    "source_sha256",
)
REQUIRED_BAR_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


class CandidateBridgeError(ValueError):
    """The candidate, bar history, artifact, or prediction is not auditable."""


_VERIFIED_BOOSTER_TOKEN = object()


class VerifiedBooster:
    """A booster loaded from one path whose bytes stayed stable during load.

    Ordinary callers cannot legitimately construct this wrapper: its private
    constructor token is held by this module and ``score_candidate`` checks both
    exact type and token identity.  Use :func:`load_verified_booster` with one
    explicit model path.  The wrapped LightGBM object is never exposed.
    """

    __slots__ = ("__booster", "__model_path", "__model_sha256", "__token")

    def __init__(
        self,
        booster: object,
        model_path: Path,
        model_sha256: str,
        *,
        _token: object,
    ) -> None:
        if _token is not _VERIFIED_BOOSTER_TOKEN:
            raise TypeError("VerifiedBooster must be created by load_verified_booster")
        self.__booster = booster
        self.__model_path = model_path
        self.__model_sha256 = model_sha256
        self.__token = _token

    @property
    def model_path(self) -> Path:
        return self.__model_path

    @property
    def model_sha256(self) -> str:
        return self.__model_sha256

    def predict(self, data: pd.DataFrame, *, num_iteration: int) -> Any:
        predictor = getattr(self.__booster, "predict", None)
        if not callable(predictor):  # pragma: no cover - real loader invariant
            raise CandidateBridgeError("verified booster has no callable predict")
        return predictor(data, num_iteration=num_iteration)

    def _is_factory_verified(self) -> bool:
        return self.__token is _VERIFIED_BOOSTER_TOKEN


def load_verified_booster(explicit_model_path: str | Path) -> VerifiedBooster:
    """Load exactly one LightGBM file and reject any load-time byte drift.

    There is no discovery, ACTIVE pointer, default, or fallback.  The resolved
    explicit path is stat'ed and hashed before loading, passed verbatim to
    ``lightgbm.Booster(model_file=...)``, then stat'ed and hashed again.  A
    replacement, rewrite, symlink-target drift, or content change fails closed.
    """
    requested = Path(explicit_model_path)
    try:
        model_path = requested.resolve(strict=True)
    except OSError as exc:
        raise CandidateBridgeError(f"explicit model path is unavailable: {requested}") from exc
    if not model_path.is_file():
        raise CandidateBridgeError(f"explicit model path is not a file: {model_path}")

    stat_before = _model_file_identity(model_path)
    hash_before = file_sha256(model_path)
    if _model_file_identity(model_path) != stat_before:
        raise CandidateBridgeError("model file drifted while hashing before load")

    booster = _load_lightgbm_model(model_path)

    stat_after_load = _model_file_identity(model_path)
    hash_after = file_sha256(model_path)
    stat_after_hash = _model_file_identity(model_path)
    if (
        stat_after_load != stat_before
        or stat_after_hash != stat_before
        or hash_after != hash_before
    ):
        raise CandidateBridgeError("model file drifted during verified booster load")
    return VerifiedBooster(
        booster,
        model_path,
        hash_before,
        _token=_VERIFIED_BOOSTER_TOKEN,
    )


def _load_lightgbm_model(model_path: Path) -> object:
    """Narrow monkeypatch seam; production always loads this explicit path."""
    import lightgbm as lgb

    return lgb.Booster(model_file=str(model_path))


def _model_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


@dataclass(frozen=True)
class Bars15mBatch:
    """Caller-supplied 15m bars plus immutable snapshot identity metadata.

    A naked frame cannot prove whether BTC features were accidentally computed
    from ETH bars.  The bridge therefore requires the source, symbol, declared
    timeframe and snapshot digest alongside the frame, validates them, and
    carries the identity into every score result.  The snapshot digest is an
    assertion supplied by the immutable-data producer; this read-only bridge
    never opens the source snapshot.
    """

    frame: pd.DataFrame
    source: str
    symbol: str
    timeframe: str
    snapshot_sha256: str


class CandidateScore(TypedDict):
    """JSON-ready result of one causal L1 -> frozen L2 research score."""

    candidate_id: str
    source: str
    symbol: str
    detector_timeframe: str
    available_at: str
    feature_bar_open_time: str
    feature_bar_closed_at: str
    feature_source_max_time: str
    score: float
    threshold: float
    passed: bool
    model_sha256: str
    booster_model_sha256: str
    bars_snapshot_sha256: str
    weights_sha256: str
    dataset_manifest_sha256: str
    image_sha256: str
    source_sha256: str
    window_end_close_time: str
    model_config_name: str
    model_side: str
    model_objective: str
    model_label: str | None
    score_semantics: str
    threshold_operator: str
    feature_semantics: str
    best_iteration: int
    next_15m_open: str
    next_30m_open: str
    research_only: bool
    paper_only: bool
    execution_eligible: bool


def flatten_schema_v2_predictions(
    predictions: Mapping[str, object],
) -> list[dict[str, object]]:
    """Turn one in-memory yolo-xx schema-v2 payload into candidate mappings.

    Each detection becomes one candidate.  Its stable id is a SHA-256 over the
    validated item identity, its detection ordinal, and the canonical detection
    object, so repeated conversion of the same manifest is deterministic.  No
    path is accepted and no file is opened by this pure adapter.

    The item's ``available_at`` remains the only fact time.  A detection must
    repeat it byte-for-byte, and any detection carrying ``signal_time`` is
    rejected: accepting such a field would give a third-party postprocessor a
    route to backfill the old box time.  Optional identity fields repeated on a
    detection must also exactly match their item values.
    """
    if not isinstance(predictions, Mapping):
        raise CandidateBridgeError("predictions must be a mapping")
    if predictions.get("schema_version") != 2:
        raise CandidateBridgeError("predictions must use schema_version 2")
    weights_sha256 = _mapping_sha256(predictions, "weights_sha256", "predictions")
    dataset_manifest_sha256 = _mapping_sha256(
        predictions, "dataset_manifest_sha256", "predictions"
    )
    items = predictions.get("items")
    if not isinstance(items, list):
        raise CandidateBridgeError("predictions.items must be a list")

    candidates: list[dict[str, object]] = []
    seen_sample_ids: set[str] = set()
    for item_index, item_raw in enumerate(items):
        location = f"predictions.items[{item_index}]"
        if not isinstance(item_raw, Mapping):
            raise CandidateBridgeError(f"{location} must be a mapping")
        sample_id = _mapping_text(item_raw, "sample_id", location)
        detector_symbol = _mapping_text(item_raw, "symbol", location)
        detector_timeframe = _mapping_text(item_raw, "detector_timeframe", location)
        try:
            normalize_bar(detector_timeframe)
        except ValueError as exc:
            raise CandidateBridgeError(
                f"{location}.detector_timeframe is unsupported: {detector_timeframe!r}"
            ) from exc
        available_raw = _mapping_text(item_raw, "available_at", location)
        available_at = _parse_utc(available_raw, f"{location}.available_at")
        window_end_close_time = _mapping_text(
            item_raw, "window_end_close_time", location
        )
        if window_end_close_time != available_raw:
            raise CandidateBridgeError(
                f"{location}.window_end_close_time must exactly match available_at"
            )
        timeframe_delta = bar_to_timedelta(detector_timeframe)
        if available_at.value % timeframe_delta.value != 0:
            raise CandidateBridgeError(
                f"{location}.available_at must align to {detector_timeframe} boundaries"
            )
        relative_image = _mapping_text(item_raw, "relative_image", location)
        relative_path = Path(relative_image)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise CandidateBridgeError(
                f"{location}.relative_image must be a safe relative path"
            )
        image_sha256 = _mapping_sha256(item_raw, "image_sha256", location)
        source_file = _mapping_text(item_raw, "source_file", location)
        source_sha256 = _mapping_sha256(item_raw, "source_sha256", location)
        window_start_time = _mapping_text(item_raw, "window_start_time", location)
        parsed_window_start = _parse_utc(
            window_start_time, f"{location}.window_start_time"
        )
        if parsed_window_start >= available_at:
            raise CandidateBridgeError(
                f"{location}.window_start_time must be before available_at"
            )
        label_sha256 = _optional_mapping_sha256(item_raw, "label_sha256", location)
        image_path = _optional_mapping_text(item_raw, "image", location)
        source_start_index, source_end_index = _optional_source_indices(
            item_raw, location
        )
        if sample_id in seen_sample_ids:
            raise CandidateBridgeError(f"duplicate predictions sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        source, symbol = _normalize_detector_symbol(detector_symbol, location)

        detections = item_raw.get("detections")
        if not isinstance(detections, list):
            raise CandidateBridgeError(f"{location}.detections must be a list")
        for detection_index, detection_raw in enumerate(detections):
            detection_location = f"{location}.detections[{detection_index}]"
            if not isinstance(detection_raw, Mapping):
                raise CandidateBridgeError(f"{detection_location} must be a mapping")
            _validate_detection_schema(detection_raw, detection_location)
            if "signal_time" in detection_raw:
                raise CandidateBridgeError(
                    f"{detection_location}.signal_time is forbidden; use available_at"
                )
            detection_available = _mapping_text(
                detection_raw, "available_at", detection_location
            )
            _parse_utc(detection_available, f"{detection_location}.available_at")
            if detection_available != available_raw:
                raise CandidateBridgeError(
                    f"{detection_location}.available_at must exactly match item available_at"
                )
            for field, expected in (
                ("sample_id", sample_id),
                ("symbol", detector_symbol),
                ("detector_timeframe", detector_timeframe),
            ):
                if field in detection_raw:
                    repeated = _mapping_text(detection_raw, field, detection_location)
                    if repeated != expected:
                        raise CandidateBridgeError(
                            f"{detection_location}.{field} must exactly match item {field}"
                        )

            detection = dict(detection_raw)
            candidate_id = _stable_candidate_id(
                sample_id=sample_id,
                detector_symbol=detector_symbol,
                detector_timeframe=detector_timeframe,
                available_at=available_raw,
                weights_sha256=weights_sha256,
                dataset_manifest_sha256=dataset_manifest_sha256,
                image_sha256=image_sha256,
                source_sha256=source_sha256,
                detection_index=detection_index,
                detection=detection,
                location=detection_location,
            )
            # Detection values are retained flat for audit, then authoritative
            # identity/time fields overwrite any optional repeats.
            candidate: dict[str, object] = {
                **detection,
                "candidate_id": candidate_id,
                "sample_id": sample_id,
                "source": source,
                "symbol": symbol,
                "detector_symbol": detector_symbol,
                "detector_timeframe": detector_timeframe,
                "available_at": _iso_utc(available_at),
                "window_start_time": _iso_utc(parsed_window_start),
                "window_end_close_time": window_end_close_time,
                "weights_sha256": weights_sha256,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "image_sha256": image_sha256,
                "relative_image": relative_image,
                "image": image_path,
                "source_file": source_file,
                "source_sha256": source_sha256,
                "source_start_index": source_start_index,
                "source_end_index": source_end_index,
                "label_sha256": label_sha256,
                "detection_index": detection_index,
            }
            candidates.append(candidate)
    return candidates


def score_candidate(
    candidate: Mapping[str, object],
    bars_15m: Bars15mBatch,
    *,
    artifact: "FrozenArtifact",
    booster: VerifiedBooster,
) -> CandidateScore:
    """Score one candidate without backdating it or discovering runtime state.

    ``artifact.feature_semantics`` and ``artifact.config.side`` choose the exact
    coordinate system used during training.  They cannot be overridden by the
    candidate or caller.  ``booster`` is injected and never loaded here.

    The returned ``next_15m_open`` and ``next_30m_open`` are the first aligned
    boundaries *strictly later* than ``available_at``.  They are research routing
    fields only; this function has no execution or order side effect.
    """
    _validate_candidate_fields(candidate)
    candidate_id = _required_text(candidate, "candidate_id")
    source = _required_text(candidate, "source")
    symbol = _required_text(candidate, "symbol")
    detector_timeframe = _required_text(candidate, "detector_timeframe")
    available_raw = _required_text(candidate, "available_at")
    available_at = _parse_utc(available_raw, "candidate.available_at")
    try:
        detector_delta = bar_to_timedelta(normalize_bar(detector_timeframe))
    except ValueError as exc:
        raise CandidateBridgeError(
            f"candidate.detector_timeframe is unsupported: {detector_timeframe!r}"
        ) from exc
    if available_at.value % detector_delta.value != 0:
        raise CandidateBridgeError(
            f"candidate.available_at must align to {detector_timeframe} boundaries"
        )
    window_end_close_time = _required_text(candidate, "window_end_close_time")
    if window_end_close_time != available_raw:
        raise CandidateBridgeError(
            "candidate.window_end_close_time must exactly match available_at"
        )
    weights_sha256 = _sha256_value(
        candidate["weights_sha256"], "candidate.weights_sha256"
    )
    dataset_manifest_sha256 = _sha256_value(
        candidate["dataset_manifest_sha256"], "candidate.dataset_manifest_sha256"
    )
    image_sha256 = _sha256_value(candidate["image_sha256"], "candidate.image_sha256")
    source_sha256 = _sha256_value(candidate["source_sha256"], "candidate.source_sha256")

    batch = _validate_bars_batch(bars_15m)
    if source != batch.source or symbol != batch.symbol:
        raise CandidateBridgeError(
            "candidate/bars identity mismatch: "
            f"candidate={source}/{symbol} bars={batch.source}/{batch.symbol}"
        )
    normalized = _validate_bar_times(batch.frame)
    closed = normalized.loc[
        normalized["open_time"] + BAR_15M <= available_at
    ].reset_index(drop=True)
    _require_continuous_closed_bars(closed)
    if len(closed) < WARMUP_BARS:
        raise CandidateBridgeError(
            "insufficient causal 15m history at available_at: "
            f"need at least {WARMUP_BARS} closed bars, got {len(closed)}"
        )
    closed = _validate_closed_ohlcv(closed)
    expected_latest_open = _floor_boundary(available_at, BAR_15M) - BAR_15M
    actual_latest_open = pd.Timestamp(closed["open_time"].iloc[-1])
    if actual_latest_open != expected_latest_open:
        raise CandidateBridgeError(
            "latest closed 15m bar is stale at available_at: "
            f"expected open={_iso_utc(expected_latest_open)}, "
            f"got open={_iso_utc(actual_latest_open)}"
        )

    feature_columns = tuple(artifact.feature_columns)
    if feature_columns != tuple(FEATURE_COLUMNS):
        raise CandidateBridgeError(
            "artifact.feature_columns does not match judgment_28_v1 FEATURE_COLUMNS"
        )
    best_iteration = int(artifact.best_iteration)
    if best_iteration <= 0:
        raise CandidateBridgeError("artifact.best_iteration must be positive")
    threshold = float(artifact.threshold)
    if not np.isfinite(threshold):
        raise CandidateBridgeError("artifact.threshold must be finite")
    model_path = Path(artifact.model_path)
    if not model_path.is_file():
        raise CandidateBridgeError(f"artifact model file is missing: {model_path}")
    try:
        resolved_model_path = model_path.resolve(strict=True)
    except OSError as exc:
        raise CandidateBridgeError(f"artifact model path is unavailable: {model_path}") from exc
    if type(booster) is not VerifiedBooster or not booster._is_factory_verified():
        raise CandidateBridgeError(
            "booster must be a factory-verified load_verified_booster wrapper"
        )
    if booster.model_path != resolved_model_path:
        raise CandidateBridgeError(
            "verified booster path does not match artifact.model_path"
        )
    model_sha256 = file_sha256(resolved_model_path)
    booster_model_sha256 = booster.model_sha256
    if booster_model_sha256 != model_sha256:
        raise CandidateBridgeError(
            "booster.model_sha256 does not match artifact.model_path sha256"
        )

    feature_semantics = str(artifact.feature_semantics).strip()
    side = str(artifact.config.side).strip()
    if side not in ("long", "short"):
        raise CandidateBridgeError(
            f"artifact.config.side must be long|short, got {side!r}"
        )
    config_name = _artifact_text(artifact.config, "name", "artifact.config")
    model_objective = _artifact_objective(artifact)
    model_label = _artifact_optional_text(artifact, "label", "artifact")
    # FrozenArtifact's q90 threshold was historically selected and served with
    # score >= threshold.  frozen.py also fixes score meaning from objective in
    # every sidecar it writes.  Derive those two legacy contract fields here;
    # optional explicit attributes may confirm them but may not contradict them.
    score_semantics = (
        "predicted_realized_ret"
        if model_objective == "regression"
        else "class_probability"
    )
    declared_score_semantics = _artifact_optional_text(
        artifact, "score_semantics", "artifact"
    )
    if (
        declared_score_semantics is not None
        and declared_score_semantics != score_semantics
    ):
        raise CandidateBridgeError(
            "artifact.score_semantics contradicts its frozen objective contract"
        )
    threshold_operator = ">="
    declared_operator = _artifact_optional_text(
        artifact, "threshold_operator", "artifact"
    )
    if declared_operator is not None and declared_operator != threshold_operator:
        raise CandidateBridgeError(
            "artifact.threshold_operator contradicts frozen q90 >= contract"
        )
    featured = add_features(add_indicators(closed))
    feature_rows = extract_feature_rows_for_semantics(
        featured,
        [len(featured) - 1],
        feature_semantics=feature_semantics,
        side=side,
    )
    values = feature_rows.loc[:, list(feature_columns)].to_numpy(dtype=float)
    if values.shape != (1, len(FEATURE_COLUMNS)) or not np.isfinite(values).all():
        missing = [
            name
            for name, value in feature_rows.iloc[0].items()
            if not np.isfinite(float(value))
        ]
        raise CandidateBridgeError(
            "latest causal feature row is incomplete or non-finite: "
            + ", ".join(missing)
        )

    raw_prediction = booster.predict(feature_rows, num_iteration=best_iteration)
    prediction = np.asarray(raw_prediction, dtype=float).reshape(-1)
    if prediction.size != 1 or not np.isfinite(prediction[0]):
        raise CandidateBridgeError(
            "booster.predict must return exactly one finite score for one candidate"
        )
    score = float(prediction[0])

    feature_bar_open = pd.Timestamp(closed["open_time"].iloc[-1])
    feature_bar_closed = feature_bar_open + BAR_15M
    if feature_bar_closed > available_at:  # defensive duplicate of the filter
        raise CandidateBridgeError("selected feature bar was not closed at available_at")

    passed = score >= threshold
    return CandidateScore(
        candidate_id=candidate_id,
        source=source,
        symbol=symbol,
        detector_timeframe=detector_timeframe,
        available_at=_iso_utc(available_at),
        feature_bar_open_time=_iso_utc(feature_bar_open),
        feature_bar_closed_at=_iso_utc(feature_bar_closed),
        # The newest source row actually read, expressed in its source key.
        feature_source_max_time=_iso_utc(feature_bar_open),
        score=score,
        threshold=threshold,
        passed=passed,
        model_sha256=model_sha256,
        booster_model_sha256=booster_model_sha256,
        bars_snapshot_sha256=batch.snapshot_sha256,
        weights_sha256=weights_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        image_sha256=image_sha256,
        source_sha256=source_sha256,
        window_end_close_time=window_end_close_time,
        model_config_name=config_name,
        model_side=side,
        model_objective=model_objective,
        model_label=model_label,
        score_semantics=score_semantics,
        threshold_operator=threshold_operator,
        feature_semantics=feature_semantics,
        best_iteration=best_iteration,
        next_15m_open=_iso_utc(_next_boundary(available_at, BAR_15M)),
        next_30m_open=_iso_utc(_next_boundary(available_at, BAR_30M)),
        research_only=True,
        paper_only=True,
        execution_eligible=False,
    )


def _validate_candidate_fields(candidate: Mapping[str, object]) -> None:
    missing = [field for field in REQUIRED_CANDIDATE_FIELDS if field not in candidate]
    if missing:
        raise CandidateBridgeError(
            "candidate missing required field(s): " + ", ".join(missing)
        )


def _required_text(candidate: Mapping[str, object], field: str) -> str:
    if candidate[field] is None:
        raise CandidateBridgeError(f"candidate.{field} must be non-empty")
    value = str(candidate[field]).strip()
    if not value:
        raise CandidateBridgeError(f"candidate.{field} must be non-empty")
    return value


def _mapping_text(record: Mapping[str, object], field: str, location: str) -> str:
    if field not in record or not isinstance(record[field], str):
        raise CandidateBridgeError(f"{location}.{field} must be non-empty")
    value = record[field].strip()
    if not value:
        raise CandidateBridgeError(f"{location}.{field} must be non-empty")
    return value


def _optional_mapping_text(
    record: Mapping[str, object], field: str, location: str
) -> str | None:
    if field not in record or record[field] is None:
        return None
    return _mapping_text(record, field, location)


def _sha256_value(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise CandidateBridgeError(f"{field} must be a lowercase sha256")
    digest = value.strip()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CandidateBridgeError(f"{field} must be a lowercase sha256")
    return digest


def _mapping_sha256(record: Mapping[str, object], field: str, location: str) -> str:
    if field not in record:
        raise CandidateBridgeError(f"{location}.{field} must be a lowercase sha256")
    return _sha256_value(record[field], f"{location}.{field}")


def _optional_mapping_sha256(
    record: Mapping[str, object], field: str, location: str
) -> str | None:
    if field not in record or record[field] is None:
        return None
    return _sha256_value(record[field], f"{location}.{field}")


def _optional_source_indices(
    record: Mapping[str, object], location: str
) -> tuple[int | None, int | None]:
    start_present = "source_start_index" in record and record["source_start_index"] is not None
    end_present = "source_end_index" in record and record["source_end_index"] is not None
    if start_present != end_present:
        raise CandidateBridgeError(
            f"{location}.source_start_index/source_end_index must be declared together"
        )
    if not start_present:
        return None, None
    start_raw = record["source_start_index"]
    end_raw = record["source_end_index"]
    if (
        isinstance(start_raw, bool)
        or isinstance(end_raw, bool)
        or not isinstance(start_raw, int)
        or not isinstance(end_raw, int)
        or start_raw < 0
        or end_raw < start_raw
    ):
        raise CandidateBridgeError(f"{location} has invalid source index identity")
    return start_raw, end_raw


def _validate_detection_schema(
    detection: Mapping[str, object], location: str
) -> None:
    class_id = detection.get("class_id")
    if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id != 0:
        raise CandidateBridgeError(f"{location}.class_id must be exactly 0")
    class_name = _mapping_text(detection, "class_name", location)
    if class_name != "dense_cluster":
        raise CandidateBridgeError(
            f"{location}.class_name must be exactly 'dense_cluster'"
        )
    confidence = detection.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise CandidateBridgeError(f"{location}.confidence must be numeric")
    confidence_value = float(confidence)
    if not np.isfinite(confidence_value) or not 0 <= confidence_value <= 1:
        raise CandidateBridgeError(f"{location}.confidence must be finite in [0, 1]")
    xywhn = detection.get("xywhn")
    if not isinstance(xywhn, (list, tuple)) or len(xywhn) != 4:
        raise CandidateBridgeError(f"{location}.xywhn must contain xc,yc,width,height")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in xywhn):
        raise CandidateBridgeError(f"{location}.xywhn values must be numeric")
    box = [float(value) for value in xywhn]
    if not all(np.isfinite(value) for value in box):
        raise CandidateBridgeError(f"{location}.xywhn values must be finite")
    xc, yc, width, height = box
    if not (
        0 <= xc <= 1
        and 0 <= yc <= 1
        and 0 < width <= 1
        and 0 < height <= 1
    ):
        raise CandidateBridgeError(f"{location}.xywhn is outside normalized bounds")


def _normalize_detector_symbol(detector_symbol: str, location: str) -> tuple[str, str]:
    if detector_symbol.startswith("okx_"):
        symbol = detector_symbol[len("okx_") :]
        if not symbol:
            raise CandidateBridgeError(f"{location}.symbol has an empty okx symbol")
        return "okx", symbol
    return "local", detector_symbol


def _stable_candidate_id(
    *,
    sample_id: str,
    detector_symbol: str,
    detector_timeframe: str,
    available_at: str,
    weights_sha256: str,
    dataset_manifest_sha256: str,
    image_sha256: str,
    source_sha256: str,
    detection_index: int,
    detection: Mapping[str, object],
    location: str,
) -> str:
    identity = {
        "bridge_schema_version": 1,
        "sample_id": sample_id,
        "detector_symbol": detector_symbol,
        "detector_timeframe": detector_timeframe,
        "available_at": available_at,
        "weights_sha256": weights_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "image_sha256": image_sha256,
        "source_sha256": source_sha256,
        "detection_index": detection_index,
        "detection": dict(detection),
    }
    try:
        canonical = json.dumps(
            identity,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CandidateBridgeError(
            f"{location} must contain finite JSON values: {exc}"
        ) from exc
    return hashlib.sha256(canonical).hexdigest()


def _artifact_text(owner: object, field: str, location: str) -> str:
    value = getattr(owner, field, None)
    if not isinstance(value, str) or not value.strip():
        raise CandidateBridgeError(f"{location}.{field} must be non-empty")
    return value.strip()


def _artifact_optional_text(owner: object, field: str, location: str) -> str | None:
    value = getattr(owner, field, None)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CandidateBridgeError(f"{location}.{field} must be non-empty when declared")
    text = value.strip()
    if text.lower() == "flat":
        raise CandidateBridgeError(f"{location}.{field}=flat is not a directional contract")
    return text


def _artifact_objective(artifact: object) -> str:
    config = getattr(artifact, "config", None)
    config_objective = getattr(config, "objective", None)
    direct_objective = getattr(artifact, "objective", None)
    declared = [value for value in (config_objective, direct_objective) if value is not None]
    if not declared:
        raise CandidateBridgeError("artifact objective is not declared")
    normalized: list[str] = []
    for value in declared:
        if not isinstance(value, str):
            raise CandidateBridgeError("artifact objective must be regression|binary")
        objective = value.strip()
        if objective not in ("regression", "binary"):
            raise CandidateBridgeError("artifact objective must be regression|binary")
        normalized.append(objective)
    if len(set(normalized)) != 1:
        raise CandidateBridgeError("artifact objective fields disagree")
    return normalized[0]


def _validate_bars_batch(batch: Bars15mBatch) -> Bars15mBatch:
    if not isinstance(batch, Bars15mBatch):
        raise CandidateBridgeError(
            "bars_15m must be Bars15mBatch with source/symbol/timeframe/snapshot_sha256"
        )
    for field in ("source", "symbol"):
        value = getattr(batch, field)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise CandidateBridgeError(f"bars_15m.{field} must be a canonical non-empty string")
    if batch.timeframe != "15m":
        raise CandidateBridgeError("bars_15m.timeframe must be exactly '15m'")
    _sha256_value(batch.snapshot_sha256, "bars_15m.snapshot_sha256")
    return batch


def _parse_utc(value: object, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise CandidateBridgeError(f"{field} is not a timestamp: {value!r}") from exc
    if pd.isna(timestamp):
        raise CandidateBridgeError(f"{field} is not a timestamp: {value!r}")
    if timestamp.tzinfo is None:
        raise CandidateBridgeError(f"{field} must be timezone-aware UTC")
    if timestamp.utcoffset() != pd.Timedelta(0):
        raise CandidateBridgeError(f"{field} must be UTC, got {timestamp.tzinfo}")
    return timestamp.tz_convert("UTC")


def _validate_bar_times(bars_15m: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(bars_15m, pd.DataFrame):
        raise CandidateBridgeError("bars_15m must be a pandas DataFrame")
    missing = [name for name in REQUIRED_BAR_COLUMNS if name not in bars_15m.columns]
    if missing:
        raise CandidateBridgeError(
            "bars_15m missing required column(s): " + ", ".join(missing)
        )
    if bars_15m.empty:
        raise CandidateBridgeError("bars_15m is empty")

    out = bars_15m.loc[:, list(REQUIRED_BAR_COLUMNS)].copy()
    out["open_time"] = [
        _parse_utc(value, f"bars_15m.open_time[{index}]")
        for index, value in enumerate(out["open_time"].tolist())
    ]
    if out["open_time"].duplicated().any():
        raise CandidateBridgeError("bars_15m.open_time contains duplicates")
    if not out["open_time"].is_monotonic_increasing:
        raise CandidateBridgeError("bars_15m.open_time must be strictly increasing")

    nanos = out["open_time"].astype("int64").to_numpy()
    if np.any(nanos % BAR_15M.value != 0):
        raise CandidateBridgeError("bars_15m.open_time must align to UTC 15m boundaries")

    return out


def _validate_closed_ohlcv(closed: pd.DataFrame) -> pd.DataFrame:
    """Validate only causally available values; later OHLCV stays unobserved."""
    out = closed.copy()
    for column in REQUIRED_BAR_COLUMNS[1:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    numeric = out.loc[:, list(REQUIRED_BAR_COLUMNS[1:])].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise CandidateBridgeError("bars_15m OHLCV columns must be finite numeric values")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise CandidateBridgeError("bars_15m OHLC prices must be strictly positive")
    if (out["volume"] < 0).any():
        raise CandidateBridgeError("bars_15m volume must be non-negative")
    body_high = out[["open", "close"]].max(axis=1)
    body_low = out[["open", "close"]].min(axis=1)
    if (
        (out["high"] < body_high).any()
        or (out["low"] > body_low).any()
        or (out["high"] < out["low"]).any()
    ):
        raise CandidateBridgeError(
            "bars_15m high/low must contain open and close with high >= low"
        )
    return out


def _require_continuous_closed_bars(closed: pd.DataFrame) -> None:
    """Fail when the causal feature history skips or stretches a 15m bar."""
    diffs = closed["open_time"].diff().iloc[1:]
    if not diffs.eq(BAR_15M).all():
        bad = diffs.loc[~diffs.eq(BAR_15M)].iloc[0]
        raise CandidateBridgeError(
            "closed causal bars must be strictly continuous at 15m cadence; "
            f"found diff={bad}"
        )


def _next_boundary(timestamp: pd.Timestamp, duration: pd.Timedelta) -> pd.Timestamp:
    nanos = timestamp.value
    next_nanos = (nanos // duration.value + 1) * duration.value
    boundary = pd.Timestamp(next_nanos, tz="UTC")
    if boundary <= timestamp:  # pragma: no cover - arithmetic invariant
        raise CandidateBridgeError("next routing boundary is not strictly later")
    return boundary


def _floor_boundary(timestamp: pd.Timestamp, duration: pd.Timedelta) -> pd.Timestamp:
    floor_nanos = timestamp.value // duration.value * duration.value
    return pd.Timestamp(floor_nanos, tz="UTC")


def _iso_utc(timestamp: pd.Timestamp) -> str:
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")
