"""Causal contract tests for the small-timeframe candidate -> frozen L2 bridge."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import yoyo.layers.l2_judgment.candidate_bridge as candidate_bridge_module
from yoyo.layers.l2_judgment.candidate_bridge import (
    Bars15mBatch,
    CandidateBridgeError,
    flatten_schema_v2_predictions,
    load_verified_booster,
    score_candidate,
)
from yoyo.layers.l2_judgment.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class _Config:
    name: str = "fixture_regression"
    objective: str = "regression"
    side: str = "long"


@dataclass(frozen=True)
class _Artifact:
    model_path: Path
    dataset_path: Path
    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)
    threshold: float = 0.5
    best_iteration: int = 7
    feature_semantics: str = "legacy_unaligned"
    score_semantics: str | None = "predicted_realized_ret"
    threshold_operator: str | None = ">="
    label: str | None = "net_taker_return"
    config: _Config = _Config()


FIXTURE_MODEL_BYTES = b"synthetic frozen booster\n"
FIXTURE_MODEL_SHA256 = hashlib.sha256(FIXTURE_MODEL_BYTES).hexdigest()


class _Booster:
    instances: list["_Booster"] = []

    def __init__(self) -> None:
        self.seen: list[pd.DataFrame] = []
        self.iterations: list[int] = []
        self.__class__.instances.append(self)

    def predict(self, data: pd.DataFrame, *, num_iteration: int) -> np.ndarray:
        self.seen.append(data.copy())
        self.iterations.append(num_iteration)
        return np.array([float(data["ret_4"].iloc[0]) + 1.0])


@pytest.fixture(autouse=True)
def _use_fake_explicit_model_loader(monkeypatch) -> None:
    _Booster.instances.clear()
    monkeypatch.setattr(
        candidate_bridge_module,
        "_load_lightgbm_model",
        lambda model_path: _Booster(),
    )


def _bars(*, periods: int = 360, future_periods: int = 0) -> pd.DataFrame:
    end = pd.Timestamp("2026-04-01T10:00:00Z") + pd.Timedelta(
        minutes=15 * future_periods
    )
    times = pd.date_range(end=end, periods=periods + future_periods, freq="15min")
    phase = np.arange(len(times), dtype=float)
    close = 100.0 + phase * 0.02 + np.sin(phase / 11.0) * 0.4
    return pd.DataFrame(
        {
            "open_time": times,
            "open": close - 0.05,
            "high": close + 0.35,
            "low": close - 0.30,
            "close": close,
            "volume": 1000.0 + (phase % 31.0) * 13.0,
        }
    )


@pytest.fixture
def artifact(tmp_path: Path) -> _Artifact:
    model = tmp_path / "frozen.txt"
    model.write_bytes(FIXTURE_MODEL_BYTES)
    # This path must not exist/read; the bridge has no dataset I/O.
    return _Artifact(model_path=model, dataset_path=tmp_path / "DO_NOT_READ.csv")


def _candidate(available_at: str, **extra: object) -> dict[str, object]:
    return {
        "candidate_id": "candidate-1",
        "source": "okx",
        "symbol": "BTC_USDT_SWAP",
        "detector_timeframe": "1m",
        "available_at": available_at,
        "window_end_close_time": available_at,
        "weights_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "image_sha256": "d" * 64,
        "source_sha256": "f" * 64,
        **extra,
    }


def _batch(
    frame: pd.DataFrame,
    *,
    source: str = "okx",
    symbol: str = "BTC_USDT_SWAP",
    timeframe: str = "15m",
    snapshot_sha256: str = "c" * 64,
) -> Bars15mBatch:
    return Bars15mBatch(
        frame=frame,
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        snapshot_sha256=snapshot_sha256,
    )


def _verified_booster(artifact: _Artifact):
    return load_verified_booster(artifact.model_path)


def test_1007_uses_the_0945_bar_and_routes_strictly_forward(
    artifact: _Artifact,
) -> None:
    booster = _verified_booster(artifact)
    result = score_candidate(
        _candidate("2026-04-01T10:07:00Z"),
        _batch(_bars()),
        artifact=artifact,
        booster=booster,
    )

    assert result["feature_bar_open_time"] == "2026-04-01T09:45:00Z"
    assert result["feature_bar_closed_at"] == "2026-04-01T10:00:00Z"
    assert result["feature_source_max_time"] == "2026-04-01T09:45:00Z"
    assert result["next_15m_open"] == "2026-04-01T10:15:00Z"
    assert result["next_30m_open"] == "2026-04-01T10:30:00Z"
    assert result["best_iteration"] == 7
    assert result["source"] == "okx"
    assert result["bars_snapshot_sha256"] == "c" * 64
    assert result["booster_model_sha256"] == FIXTURE_MODEL_SHA256
    assert result["weights_sha256"] == "a" * 64
    assert result["dataset_manifest_sha256"] == "b" * 64
    assert result["image_sha256"] == "d" * 64
    assert result["source_sha256"] == "f" * 64
    assert result["window_end_close_time"] == "2026-04-01T10:07:00Z"
    assert result["model_config_name"] == "fixture_regression"
    assert result["model_side"] == "long"
    assert result["model_objective"] == "regression"
    assert result["model_label"] == "net_taker_return"
    assert result["score_semantics"] == "predicted_realized_ret"
    assert result["threshold_operator"] == ">="
    assert result["research_only"] is True
    assert result["paper_only"] is True
    assert result["execution_eligible"] is False
    raw_booster = _Booster.instances[-1]
    assert raw_booster.iterations == [7]
    assert list(raw_booster.seen[0].columns) == FEATURE_COLUMNS
    assert result["model_sha256"] == hashlib.sha256(
        artifact.model_path.read_bytes()
    ).hexdigest()
    assert not artifact.dataset_path.exists()


def test_1015_can_use_the_1000_bar_and_exact_boundaries_move_forward(
    artifact: _Artifact,
) -> None:
    result = score_candidate(
        _candidate("2026-04-01T10:15:00Z"),
        _batch(_bars()),
        artifact=artifact,
        booster=_verified_booster(artifact),
    )

    assert result["feature_bar_open_time"] == "2026-04-01T10:00:00Z"
    assert result["feature_bar_closed_at"] == "2026-04-01T10:15:00Z"
    assert result["next_15m_open"] == "2026-04-01T10:30:00Z"
    assert result["next_30m_open"] == "2026-04-01T10:30:00Z"


def test_future_rows_do_not_change_features_or_score(artifact: _Artifact) -> None:
    base = _bars()
    with_future = _bars(future_periods=12)
    with_future.loc[
        with_future["open_time"] >= pd.Timestamp("2026-04-01T10:00:00Z"),
        ["open", "high", "low", "close", "volume"],
    ] = np.nan
    first = _verified_booster(artifact)
    first_raw = _Booster.instances[-1]
    second = _verified_booster(artifact)
    second_raw = _Booster.instances[-1]

    before = score_candidate(
        _candidate("2026-04-01T10:07:00Z"),
        _batch(base),
        artifact=artifact,
        booster=first,
    )
    after = score_candidate(
        _candidate("2026-04-01T10:07:00Z"),
        _batch(with_future),
        artifact=artifact,
        booster=second,
    )

    assert before == after
    pd.testing.assert_frame_equal(first_raw.seen[0], second_raw.seen[0])


def test_artifact_dataset_path_is_never_accessed(
    artifact: _Artifact,
) -> None:
    class _PoisonDatasetArtifact:
        model_path = artifact.model_path
        feature_columns = tuple(FEATURE_COLUMNS)
        threshold = 0.5
        best_iteration = 7
        feature_semantics = "legacy_unaligned"
        score_semantics = "predicted_realized_ret"
        threshold_operator = ">="
        label = "net_taker_return"
        config = _Config()

        @property
        def dataset_path(self):
            raise AssertionError("candidate bridge must not read artifact.dataset_path")

    result = score_candidate(
        _candidate("2026-04-01T10:07:00Z"),
        _batch(_bars()),
        artifact=_PoisonDatasetArtifact(),  # type: ignore[arg-type]
        booster=_verified_booster(artifact),
    )
    assert result["candidate_id"] == "candidate-1"


def test_box_end_time_is_descriptive_and_cannot_backdate_score(
    artifact: _Artifact,
) -> None:
    old_box = score_candidate(
        _candidate("2026-04-01T10:07:00Z", box_end_time="2026-03-31T00:00:00Z"),
        _batch(_bars()),
        artifact=artifact,
        booster=_verified_booster(artifact),
    )
    future_box = score_candidate(
        _candidate("2026-04-01T10:07:00Z", box_end_time="2099-01-01T00:00:00Z"),
        _batch(_bars()),
        artifact=artifact,
        booster=_verified_booster(artifact),
    )

    assert old_box == future_box
    assert old_box["available_at"] == "2026-04-01T10:07:00Z"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.assign(open_time=frame["open_time"].dt.tz_localize(None)), "UTC"),
        (
            lambda frame: frame.assign(
                open_time=frame["open_time"].where(
                    frame.index != 100, frame["open_time"] + pd.Timedelta(minutes=1)
                )
            ),
            "15m boundaries",
        ),
        (lambda frame: frame.iloc[::-1].reset_index(drop=True), "strictly increasing"),
        (
            lambda frame: frame.assign(
                open_time=frame["open_time"].where(
                    frame.index != 101, frame["open_time"].iloc[100]
                )
            ),
            "duplicates",
        ),
    ],
)
def test_bar_time_contract_fails_closed(
    artifact: _Artifact,
    mutate,
    message: str,
) -> None:
    with pytest.raises(CandidateBridgeError, match=message):
        score_candidate(
            _candidate("2026-04-01T10:07:00Z"),
            _batch(mutate(_bars())),
            artifact=artifact,
            booster=_verified_booster(artifact),
        )


def test_requires_enough_closed_causal_history(artifact: _Artifact) -> None:
    with pytest.raises(CandidateBridgeError, match="insufficient causal"):
        score_candidate(
            _candidate("2026-04-01T10:07:00Z"),
            _batch(_bars(periods=200)),
            artifact=artifact,
            booster=_verified_booster(artifact),
        )


def test_closed_causal_history_rejects_a_15m_gap(artifact: _Artifact) -> None:
    bars = _bars().drop(index=100).reset_index(drop=True)
    with pytest.raises(CandidateBridgeError, match="strictly continuous"):
        score_candidate(
            _candidate("2026-04-01T10:07:00Z"),
            _batch(bars),
            artifact=artifact,
            booster=_verified_booster(artifact),
        )


def test_latest_closed_bar_must_touch_available_at(artifact: _Artifact) -> None:
    stale = _bars().iloc[:-3].reset_index(drop=True)
    with pytest.raises(CandidateBridgeError, match="latest closed 15m bar is stale"):
        score_candidate(
            _candidate("2026-04-01T10:07:00Z"),
            _batch(stale),
            artifact=artifact,
            booster=_verified_booster(artifact),
        )


@pytest.mark.parametrize(
    "batch",
    [
        _batch(_bars(), symbol="ETH_USDT_SWAP"),
        _batch(_bars(), source="local"),
    ],
)
def test_candidate_and_bars_identity_must_match(
    artifact: _Artifact, batch: Bars15mBatch
) -> None:
    with pytest.raises(CandidateBridgeError, match="identity mismatch"):
        score_candidate(
            _candidate("2026-04-01T10:07:00Z"),
            batch,
            artifact=artifact,
            booster=_verified_booster(artifact),
        )


def test_naked_dataframe_is_not_an_auditable_bar_batch(artifact: _Artifact) -> None:
    with pytest.raises(CandidateBridgeError, match="Bars15mBatch"):
        score_candidate(
            _candidate("2026-04-01T10:07:00Z"),
            _bars(),  # type: ignore[arg-type]
            artifact=artifact,
            booster=_verified_booster(artifact),
        )


def test_score_rejects_a_naked_fake_booster(artifact: _Artifact) -> None:
    naked = _Booster()
    with pytest.raises(CandidateBridgeError, match="factory-verified"):
        score_candidate(
            _candidate("2026-04-01T10:07:00Z"),
            _batch(_bars()),
            artifact=artifact,
            booster=naked,  # type: ignore[arg-type]
        )
    assert naked.seen == []


def test_verified_wrapper_rejects_another_artifact_path(
    artifact: _Artifact, tmp_path: Path
) -> None:
    wrapper = _verified_booster(artifact)
    another_path = tmp_path / "same-bytes-another-path.txt"
    another_path.write_bytes(FIXTURE_MODEL_BYTES)
    another_artifact = replace(artifact, model_path=another_path)
    with pytest.raises(CandidateBridgeError, match="path does not match"):
        score_candidate(
            _candidate("2026-04-01T10:07:00Z"),
            _batch(_bars()),
            artifact=another_artifact,
            booster=wrapper,
        )


def test_verified_load_rejects_model_file_drift(
    artifact: _Artifact, monkeypatch
) -> None:
    def drift_while_loading(model_path: Path) -> _Booster:
        model_path.write_bytes(b"changed during load\n")
        return _Booster()

    monkeypatch.setattr(
        candidate_bridge_module, "_load_lightgbm_model", drift_while_loading
    )
    with pytest.raises(CandidateBridgeError, match="drifted during"):
        load_verified_booster(artifact.model_path)


def test_artifact_side_rejects_flat(artifact: _Artifact) -> None:
    invalid = replace(artifact, config=replace(artifact.config, side="flat"))
    with pytest.raises(CandidateBridgeError, match=r"must be long\|short"):
        score_candidate(
            _candidate("2026-04-01T10:07:00Z"),
            _batch(_bars()),
            artifact=invalid,
            booster=_verified_booster(artifact),
        )


def test_legacy_artifact_derives_frozen_score_contract(artifact: _Artifact) -> None:
    legacy = replace(artifact, threshold_operator=None, score_semantics=None, label=None)
    result = score_candidate(
        _candidate("2026-04-01T10:07:00Z"),
        _batch(_bars()),
        artifact=legacy,
        booster=_verified_booster(artifact),
    )
    assert result["threshold_operator"] == ">="
    assert result["score_semantics"] == "predicted_realized_ret"
    assert result["model_label"] is None
    assert result["passed"] is (result["score"] >= result["threshold"])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("close", 0.0, "strictly positive"),
        ("volume", -1.0, "non-negative"),
        ("high", 1.0, "high/low must contain"),
        ("low", 1000.0, "high/low must contain"),
    ],
)
def test_closed_ohlcv_geometry_fails_closed(
    artifact: _Artifact, column: str, value: float, message: str
) -> None:
    bars = _bars()
    bars.loc[100, column] = value
    with pytest.raises(CandidateBridgeError, match=message):
        score_candidate(
            _candidate("2026-04-01T10:07:00Z"),
            _batch(bars),
            artifact=artifact,
            booster=_verified_booster(artifact),
        )


def test_available_at_must_be_utc(artifact: _Artifact) -> None:
    with pytest.raises(CandidateBridgeError, match="must be UTC"):
        score_candidate(
            _candidate("2026-04-01T18:07:00+08:00"),
            _batch(_bars()),
            artifact=artifact,
            booster=_verified_booster(artifact),
        )


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (
            _candidate("2026-04-01T10:07:00Z", detector_timeframe="7m"),
            "detector_timeframe is unsupported",
        ),
        (
            _candidate("2026-04-01T10:07:00Z", detector_timeframe="5m"),
            "must align to 5m boundaries",
        ),
        (
            _candidate(
                "2026-04-01T10:07:00Z",
                window_end_close_time="2026-04-01T10:06:00Z",
            ),
            "window_end_close_time must exactly match",
        ),
    ],
)
def test_direct_score_candidate_enforces_detector_time_contract(
    artifact: _Artifact, candidate: dict[str, object], message: str
) -> None:
    with pytest.raises(CandidateBridgeError, match=message):
        score_candidate(
            candidate,
            _batch(_bars()),
            artifact=artifact,
            booster=_verified_booster(artifact),
        )


def test_artifact_semantics_and_side_are_the_only_extractor_controls(
    artifact: _Artifact,
) -> None:
    aligned = _Artifact(
        model_path=artifact.model_path,
        dataset_path=artifact.dataset_path,
        feature_semantics="side_aligned_v1",
        config=_Config(side="short"),
    )
    candidate = _candidate(
        "2026-04-01T10:07:00Z",
        feature_semantics="legacy_unaligned",
        side="long",
    )
    booster = _verified_booster(artifact)
    result = score_candidate(candidate, _batch(_bars()), artifact=aligned, booster=booster)

    assert result["feature_semantics"] == "side_aligned_v1"
    # Fake booster returns 1 + ret_4. Short alignment negates ret_4.
    raw_booster = _Booster.instances[-1]
    expected = 1.0 + float(raw_booster.seen[0]["ret_4"].iloc[0])
    assert result["score"] == pytest.approx(expected)
    assert float(raw_booster.seen[0]["ret_4"].iloc[0]) < 0


def test_passed_uses_the_frozen_threshold_without_tuning(artifact: _Artifact) -> None:
    result = score_candidate(
        _candidate("2026-04-01T10:07:00Z"),
        _batch(_bars()),
        artifact=artifact,
        booster=_verified_booster(artifact),
    )
    assert result["threshold"] == artifact.threshold
    assert result["passed"] is (result["score"] >= artifact.threshold)


def _predictions() -> dict[str, object]:
    return {
        "schema_version": 2,
        "weights_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "items": [
            {
                "sample_id": "sample-okx",
                "symbol": "okx_BTC_USDT_SWAP",
                "detector_timeframe": "5m",
                "relative_image": "images/val/sample-okx.png",
                "image": "/immutable/images/val/sample-okx.png",
                "image_sha256": "d" * 64,
                "label_sha256": "e" * 64,
                "source_file": "/immutable/okx_BTC_USDT_SWAP_5m.csv",
                "source_sha256": "f" * 64,
                "source_start_index": 100,
                "source_end_index": 699,
                "window_start_time": "2026-03-30T08:00:00Z",
                "window_end_close_time": "2026-04-01T10:00:00Z",
                "available_at": "2026-04-01T10:00:00Z",
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "dense_cluster",
                        "confidence": 0.91,
                        "xywhn": [0.5, 0.4, 0.2, 0.1],
                        "available_at": "2026-04-01T10:00:00Z",
                    },
                    {
                        "class_id": 0,
                        "class_name": "dense_cluster",
                        "confidence": 0.82,
                        "xywhn": [0.6, 0.4, 0.15, 0.1],
                        "available_at": "2026-04-01T10:00:00Z",
                    },
                ],
            },
            {
                "sample_id": "sample-local",
                "symbol": "fixture_ETH_USDT_SWAP",
                "detector_timeframe": "3m",
                "relative_image": "images/val/sample-local.png",
                "image_sha256": "1" * 64,
                "source_file": "/immutable/fixture_ETH_USDT_SWAP_3m.csv",
                "source_sha256": "2" * 64,
                "window_start_time": "2026-03-31T04:03:00Z",
                "window_end_close_time": "2026-04-01T10:03:00Z",
                "available_at": "2026-04-01T10:03:00Z",
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "dense_cluster",
                        "confidence": 0.77,
                        "xywhn": [0.5, 0.5, 0.1, 0.1],
                        "available_at": "2026-04-01T10:03:00Z",
                    }
                ],
            },
        ],
    }


def test_schema_v2_predictions_flatten_one_candidate_per_detection() -> None:
    first = flatten_schema_v2_predictions(_predictions())
    second = flatten_schema_v2_predictions(_predictions())

    assert first == second
    assert len(first) == 3
    assert len({row["candidate_id"] for row in first}) == 3
    assert first[0]["sample_id"] == "sample-okx"
    assert first[0]["source"] == "okx"
    assert first[0]["symbol"] == "BTC_USDT_SWAP"
    assert first[0]["detector_symbol"] == "okx_BTC_USDT_SWAP"
    assert first[0]["detector_timeframe"] == "5m"
    assert first[0]["available_at"] == "2026-04-01T10:00:00Z"
    assert first[0]["window_end_close_time"] == "2026-04-01T10:00:00Z"
    assert first[0]["weights_sha256"] == "a" * 64
    assert first[0]["dataset_manifest_sha256"] == "b" * 64
    assert first[0]["image_sha256"] == "d" * 64
    assert first[0]["source_file"] == "/immutable/okx_BTC_USDT_SWAP_5m.csv"
    assert first[0]["source_sha256"] == "f" * 64
    assert first[0]["relative_image"] == "images/val/sample-okx.png"
    assert first[0]["source_start_index"] == 100
    assert first[0]["source_end_index"] == 699
    assert first[0]["detection_index"] == 0
    assert first[0]["confidence"] == 0.91
    assert first[2]["source"] == "local"
    assert first[2]["symbol"] == "fixture_ETH_USDT_SWAP"
    assert first[2]["detector_symbol"] == "fixture_ETH_USDT_SWAP"


def test_schema_v2_detection_availability_must_exactly_match_item() -> None:
    payload = _predictions()
    payload["items"][0]["detections"][0]["available_at"] = (
        "2026-04-01T10:00:00+00:00"
    )
    with pytest.raises(CandidateBridgeError, match="exactly match"):
        flatten_schema_v2_predictions(payload)


def test_schema_v2_detection_forbids_signal_time() -> None:
    payload = _predictions()
    payload["items"][0]["detections"][0]["signal_time"] = (
        "2026-03-31T20:00:00Z"
    )
    with pytest.raises(CandidateBridgeError, match="signal_time is forbidden"):
        flatten_schema_v2_predictions(payload)


def test_schema_v2_repeated_detection_identity_must_match_item() -> None:
    payload = _predictions()
    payload["items"][0]["detections"][0]["symbol"] = "okx_ETH_USDT_SWAP"
    with pytest.raises(CandidateBridgeError, match="symbol must exactly match"):
        flatten_schema_v2_predictions(payload)


def test_schema_v2_window_close_must_exactly_match_available_at() -> None:
    payload = _predictions()
    payload["items"][0]["window_end_close_time"] = "2026-04-01T09:55:00Z"
    with pytest.raises(CandidateBridgeError, match="window_end_close_time must exactly"):
        flatten_schema_v2_predictions(payload)


def test_schema_v2_availability_must_align_to_detector_timeframe() -> None:
    payload = _predictions()
    item = payload["items"][0]
    item["available_at"] = "2026-04-01T10:02:00Z"
    item["window_end_close_time"] = item["available_at"]
    for detection in item["detections"]:
        detection["available_at"] = item["available_at"]
    with pytest.raises(CandidateBridgeError, match="align to 5m boundaries"):
        flatten_schema_v2_predictions(payload)


@pytest.mark.parametrize("field", ["weights_sha256", "dataset_manifest_sha256"])
def test_schema_v2_requires_root_provenance_hashes(field: str) -> None:
    payload = _predictions()
    payload.pop(field)
    with pytest.raises(CandidateBridgeError, match=field):
        flatten_schema_v2_predictions(payload)


@pytest.mark.parametrize(
    "field",
    ["image_sha256", "source_file", "source_sha256", "relative_image"],
)
def test_schema_v2_requires_item_source_identity(field: str) -> None:
    payload = _predictions()
    payload["items"][0].pop(field)
    with pytest.raises(CandidateBridgeError, match=field):
        flatten_schema_v2_predictions(payload)


def test_candidate_id_binds_detector_dataset_and_image_hashes() -> None:
    original = flatten_schema_v2_predictions(_predictions())[0]["candidate_id"]
    for owner, field, replacement_sha in (
        ("root", "weights_sha256", "3" * 64),
        ("root", "dataset_manifest_sha256", "4" * 64),
        ("item", "image_sha256", "5" * 64),
    ):
        payload = _predictions()
        if owner == "root":
            payload[field] = replacement_sha
        else:
            payload["items"][0][field] = replacement_sha
        changed = flatten_schema_v2_predictions(payload)[0]["candidate_id"]
        assert changed != original


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("class_id", -1, "class_id"),
        ("class_name", "", "class_name"),
        ("confidence", 1.1, "confidence"),
        ("xywhn", [1.2, 0.5, 0.1, 0.1], "xywhn"),
    ],
)
def test_schema_v2_validates_detection_geometry(
    field: str, value: object, message: str
) -> None:
    payload = _predictions()
    payload["items"][0]["detections"][0][field] = value
    with pytest.raises(CandidateBridgeError, match=message):
        flatten_schema_v2_predictions(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version=1), "schema_version 2"),
        (
            lambda payload: payload["items"][0].pop("sample_id"),
            "sample_id must be non-empty",
        ),
        (
            lambda payload: payload["items"][0]["detections"][0].pop("available_at"),
            "available_at must be non-empty",
        ),
        (
            lambda payload: payload["items"][0].update(detector_timeframe="7m"),
            "detector_timeframe is unsupported",
        ),
        (
            lambda payload: payload["items"][0].pop("window_end_close_time"),
            "window_end_close_time must be non-empty",
        ),
    ],
)
def test_schema_v2_prediction_contract_fails_closed(mutation, message: str) -> None:
    payload = _predictions()
    mutation(payload)
    with pytest.raises(CandidateBridgeError, match=message):
        flatten_schema_v2_predictions(payload)
