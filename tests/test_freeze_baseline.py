"""The freeze must fail loudly rather than freeze a hole or a mixed contract."""
from __future__ import annotations

import json

import pytest

from tools.freeze_baseline import freeze, scalar_yaml, static_metrics

ARGS_YAML = """task: detect
model: C:/fable/models/yolo11s_w20.pt
epochs: 40
patience: 10
batch: 8
imgsz: 960
seed: 0
optimizer: AdamW
rect: true
deterministic: true
augment: false
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0
degrees: 0.0
translate: 0.02
scale: 0.1
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.0
mosaic: 0.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
erasing: 0.0
"""

RECIPE_KEYS = [
    "epochs", "patience", "batch", "imgsz", "seed", "optimizer", "rect", "deterministic",
    "augment", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "mosaic", "mixup", "cutmix", "copy_paste", "erasing",
]

CONTRACT_KEYS = ["protocol", "hours", "window_lengths", "confidence", "nms_iou", "event_gap_bars"]


def scan_summary(weights_sha: str, events: int, **overrides) -> dict:
    payload = {
        "protocol": "canary_v1",
        "weights_sha256": weights_sha,
        "scanned_symbols": ["ETH_USDT_SWAP", "SOL_USDT_SWAP"],
        "hours": 12.0,
        "window_lengths": [12, 13],
        "confidence": 0.25,
        "nms_iou": 0.7,
        "event_gap_bars": 5,
        "raw_detections": events * 20,
        "deduplicated_events": events,
    }
    payload.update(overrides)
    return payload


def build_source(tmp_path, canary_overrides=None):
    repo = tmp_path / "fable"
    (repo / "weights").mkdir(parents=True)
    (repo / "weights/r1.pt").write_bytes(b"r1")
    (repo / "run").mkdir()
    (repo / "run/args.yaml").write_text(ARGS_YAML, encoding="utf-8")
    (repo / "run/val.json").write_text(
        json.dumps(
            {
                "weights_sha256": "abc",
                "metrics": {"precision": 0.86, "recall": 0.78, "map50": 0.9, "map50_95": 0.74},
            }
        ),
        encoding="utf-8",
    )
    dataset = repo / "datasets/ds"
    dataset.mkdir(parents=True)
    (dataset / "data.yaml").write_text("names:\n  0: ma_dense_core\n", encoding="utf-8")
    (dataset / "positive_manifest.jsonl").write_text(
        json.dumps({"split": "train", "win_len": 12}) + "\n", encoding="utf-8"
    )
    (repo / "gold.csv").write_text("box_id,owner_side\nA,short\n", encoding="utf-8")
    (repo / "canary").mkdir()
    left = scan_summary("sha_r1", 331)
    right = scan_summary("sha_r2", 195, **(canary_overrides or {}))
    (repo / "canary/r1.json").write_text(json.dumps(left), encoding="utf-8")
    (repo / "canary/r2.json").write_text(json.dumps(right), encoding="utf-8")

    source_config = tmp_path / "source.json"
    source_config.write_text(
        json.dumps({"source_repo": str(repo), "holdout_start": "2026-05-04T00:00:00+00:00"}),
        encoding="utf-8",
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "protocol": "test_freeze_v1",
                "detection_class": "ma_dense_core",
                "direction_scope": "short_pipeline",
                "window_protocol": {"window_lengths": [12, 13]},
                "weights": {"r1": "weights/r1.pt"},
                "training_recipe_source": "run/args.yaml",
                "recipe_keys": RECIPE_KEYS,
                "static_metrics": {"r1": "run/val.json"},
                "datasets": {"ds": "datasets/ds"},
                "gold_geometry_source": "gold.csv",
                "continuous_canary": {"am": {"r1": "canary/r1.json", "r2": "canary/r2.json"}},
                "canary_contract_keys": CONTRACT_KEYS,
                "owner_review_packs": {"missing_pack": "nope/summary.json"},
            }
        ),
        encoding="utf-8",
    )
    return source_config, spec


def test_scalar_yaml_reads_flat_scalars(tmp_path) -> None:
    path = tmp_path / "args.yaml"
    path.write_text(ARGS_YAML, encoding="utf-8")
    values = scalar_yaml(path, ["epochs", "rect", "fliplr", "optimizer"])
    assert values == {"epochs": 40, "rect": True, "fliplr": 0.0, "optimizer": "AdamW"}


def test_scalar_yaml_rejects_missing_recipe_key(tmp_path) -> None:
    path = tmp_path / "args.yaml"
    path.write_text("epochs: 40\n", encoding="utf-8")
    with pytest.raises(KeyError, match="missing recipe keys"):
        scalar_yaml(path, ["epochs", "seed"])


def test_static_metrics_accepts_wrapped_and_flat_shapes(tmp_path) -> None:
    flat = {"precision": 0.85, "recall": 0.90, "map50": 0.92, "map50_95": 0.73}
    assert static_metrics(flat, tmp_path / "flat.json") == flat
    assert static_metrics({"metrics": flat, "weights_sha256": "x"}, tmp_path / "w.json") == flat


def test_static_metrics_rejects_a_file_with_no_headline_numbers(tmp_path) -> None:
    with pytest.raises(KeyError, match="map50"):
        static_metrics({"precision": 0.85, "recall": 0.9}, tmp_path / "half.json")


def test_freeze_records_shas_and_flags_augmentation_off(tmp_path) -> None:
    source_config, spec = build_source(tmp_path)
    frozen = freeze(source_config, spec, tmp_path / "out/frozen.json")
    assert frozen["weights"]["r1"]["sha256"]
    assert frozen["training_recipe"]["augmentation_fully_disabled"] is True
    assert frozen["continuous_canary"]["am"]["r1"]["deduplicated_events"] == 331
    assert frozen["datasets"]["ds"]["manifests"]["positive_manifest"]["rows"] == 1
    assert frozen["owner_review_packs"]["missing_pack"]["present"] is False
    assert frozen["holdout_read"] is False
    assert json.loads((tmp_path / "out/frozen.json").read_text(encoding="utf-8"))["protocol"] == "test_freeze_v1"


def test_freeze_rejects_mismatched_canary_contracts(tmp_path) -> None:
    source_config, spec = build_source(tmp_path, canary_overrides={"confidence": 0.4})
    with pytest.raises(ValueError, match="different contracts"):
        freeze(source_config, spec, tmp_path / "out/frozen.json")


def test_freeze_rejects_missing_weights(tmp_path) -> None:
    source_config, spec = build_source(tmp_path)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    payload["weights"]["ghost"] = "weights/ghost.pt"
    spec.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="ghost weights"):
        freeze(source_config, spec, tmp_path / "out/frozen.json")
