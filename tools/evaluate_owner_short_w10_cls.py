#!/usr/bin/env python3
"""Evaluate a fixed-W10 classifier on a manifest-backed pre-holdout split.

This evaluator intentionally bypasses Ultralytics' classification validator.
Training used a white, aspect-preserving letterbox followed by ``ToTensor``;
using the stock square crop here would hide the right-edge confirmation bar
and make the reported metrics incomparable.  Labels and image paths must come
from the dataset manifest, and every requested row is rechecked for causal
input, a pre-2026-05-04 decision time, physical-path agreement, and image SHA.

Only already-rendered images are read.  No K-line source or holdout data is
opened by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image


HOLDOUT_START = datetime(2026, 5, 4, tzinfo=timezone.utc)
CLASS_NAMES = ("NO_SIGNAL", "SIGNAL")


class WhiteLetterbox:
    """Pad with renderer white while preserving the entire chart frame."""

    def __init__(self, size: int, fill: int = 255):
        self.size = int(size)
        self.fill = int(fill)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        image = image.resize((new_width, new_height), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
        if image.mode != "RGB":
            image = image.convert("RGB")
        canvas.paste(
            image,
            ((self.size - new_width) // 2, (self.size - new_height) // 2),
        )
        return canvas


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for an artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc(value: object) -> datetime:
    """Parse an ISO timestamp and require an explicit timezone."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    stamp = datetime.fromisoformat(text)
    if stamp.tzinfo is None:
        raise ValueError(f"naive decision_time is forbidden: {value}")
    return stamp.astimezone(timezone.utc)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"manifest row {line_number} is not an object")
        rows.append(value)
    return rows


def validate_split(data: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate lineage and return rows in deterministic manifest order."""
    data = data.resolve()
    dataset_manifest_path = data / "dataset_manifest.json"
    rows_path = data / "manifest.jsonl"
    if not dataset_manifest_path.is_file() or not rows_path.is_file():
        raise ValueError("dataset_manifest.json and manifest.jsonl are both required")

    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("holdout_read") is not False:
        raise ValueError("dataset manifest does not explicitly declare holdout_read=false")
    gates = dataset_manifest.get("gates") or {}
    for gate in ("holdout_rows", "future_input_rows"):
        if gates.get(gate) != 0:
            raise ValueError(f"unsafe dataset gate {gate}={gates.get(gate)!r}")
    expected_manifest_sha = dataset_manifest.get("manifest_jsonl_sha256")
    if not expected_manifest_sha or sha256_file(rows_path) != expected_manifest_sha:
        raise ValueError("manifest.jsonl SHA-256 does not match dataset_manifest.json")

    all_rows = _read_jsonl(rows_path)
    requested_rows = [row for row in all_rows if row.get("split") == split]
    if not requested_rows:
        raise ValueError(f"manifest has no rows for split={split!r}")

    rows: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    seen_sample_ids: set[str] = set()
    seen_image_sha: set[str] = set()
    counts: Counter[str] = Counter()
    for row in requested_rows:
        sample_id = str(row.get("sample_id") or "")
        label = str(row.get("label") or "")
        if not sample_id or sample_id in seen_sample_ids:
            raise ValueError(f"missing or duplicate sample_id in {split}: {sample_id!r}")
        if label not in CLASS_NAMES:
            raise ValueError(f"unexpected label for {sample_id}: {label!r}")
        if row.get("holdout_read") is not False:
            raise ValueError(f"row does not declare holdout_read=false: {sample_id}")
        if row.get("future_used_in_model_input") is not False:
            raise ValueError(f"future input is not explicitly false: {sample_id}")
        if int(row["visible_end_bar"]) > int(row["decision_bar"]):
            raise ValueError(f"visible_end_bar exceeds decision_bar: {sample_id}")
        if parse_utc(row["decision_time"]) >= HOLDOUT_START:
            raise ValueError(f"requested split reaches holdout: {sample_id}")

        declared_path = Path(str(row.get("image_path") or ""))
        expected_path = (data / split / label / declared_path.name).resolve()
        declared_parts = declared_path.parts
        if len(declared_parts) < 3 or tuple(declared_parts[-3:-1]) != (split, label):
            raise ValueError(f"manifest image_path has wrong split/label tail: {sample_id}")
        if expected_path.parent != (data / split / label).resolve():
            raise ValueError(f"image path escapes requested class directory: {sample_id}")
        if not expected_path.is_file():
            raise ValueError(f"missing image for {sample_id}: {expected_path}")
        image_sha = str(row.get("image_sha256") or "")
        if len(image_sha) != 64 or sha256_file(expected_path) != image_sha:
            raise ValueError(f"image SHA-256 mismatch: {sample_id}")
        if expected_path in expected_paths or image_sha in seen_image_sha:
            raise ValueError(f"duplicate image in requested split: {sample_id}")

        item = dict(row)
        item["resolved_image_path"] = str(expected_path)
        rows.append(item)
        expected_paths.add(expected_path)
        seen_sample_ids.add(sample_id)
        seen_image_sha.add(image_sha)
        counts[label] += 1

    physical_paths = {
        path.resolve()
        for label in CLASS_NAMES
        for path in (data / split / label).glob("*.png")
    }
    if physical_paths != expected_paths:
        missing = len(expected_paths - physical_paths)
        extra = len(physical_paths - expected_paths)
        raise ValueError(f"physical/manifest image mismatch: missing={missing} extra={extra}")
    for label in CLASS_NAMES:
        expected_count = dataset_manifest.get("counts", {}).get(f"{split}/{label}")
        if expected_count != counts[label]:
            raise ValueError(
                f"dataset count mismatch for {split}/{label}: "
                f"manifest={expected_count!r} rows={counts[label]}"
            )
    return rows, dataset_manifest


def full_frame_transforms(size: int):
    """Return the exact white-letterbox and tensor transform used for training."""
    import torchvision.transforms as transforms

    return transforms.Compose([WhiteLetterbox(size), transforms.ToTensor()])


def choose_device(explicit: str | None) -> str:
    """Prefer CUDA, then MPS, while permitting an explicit CPU audit."""
    import torch

    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_classifier(weights: Path, device: str):
    """Load weights, including checkpoints pickled from the trainer's __main__."""
    import __main__ as main_module

    main_module.WhiteLetterbox = WhiteLetterbox
    from ultralytics import YOLO

    model = YOLO(str(weights), task="classify")
    names_value = model.names
    if isinstance(names_value, dict):
        names = {int(key): str(value) for key, value in names_value.items()}
    else:
        names = {index: str(value) for index, value in enumerate(names_value)}
    if set(names.values()) != set(CLASS_NAMES) or len(names) != 2:
        raise ValueError(f"expected exactly {CLASS_NAMES}, got {names}")
    signal_ids = [index for index, name in names.items() if name == "SIGNAL"]
    model.model.to(device).eval()
    return model, names, signal_ids[0]


def _prediction_tensor(output: Any, batch_size: int):
    """Extract the two-class batch tensor from an Ultralytics model output."""
    import torch

    candidates: Iterable[Any]
    candidates = output if isinstance(output, (tuple, list)) else (output,)
    for candidate in candidates:
        if torch.is_tensor(candidate) and candidate.ndim == 2 and candidate.shape == (batch_size, 2):
            return candidate
    raise ValueError(f"unexpected classifier output type/shape: {type(output).__name__}")


def _as_probabilities(scores):
    """Accept either classifier logits or an already-softmaxed Ultralytics output."""
    import torch

    detached = scores.float()
    sums = detached.sum(dim=1)
    already_probabilities = bool(
        torch.all(detached >= 0).item()
        and torch.all(detached <= 1).item()
        and torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)
    )
    return detached if already_probabilities else torch.softmax(detached, dim=1)


def predict_signal_probabilities(
    model: Any,
    rows: Sequence[dict[str, Any]],
    *,
    device: str,
    signal_id: int,
    image_size: int,
    batch_size: int,
) -> list[float]:
    """Run deterministic full-frame inference in manifest order."""
    import torch

    transform = full_frame_transforms(image_size)
    probabilities: list[float] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        tensors = []
        for row in chunk:
            with Image.open(row["resolved_image_path"]) as image:
                tensors.append(transform(image.convert("RGB")))
        inputs = torch.stack(tensors).to(device)
        with torch.inference_mode():
            output = model.model(inputs)
            scores = _prediction_tensor(output, len(chunk))
            probs = _as_probabilities(scores).detach().cpu()
        probabilities.extend(float(value) for value in probs[:, signal_id].tolist())
    return probabilities


def classification_metrics(actual: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    """Compute confusion and class-balanced metrics without sklearn."""
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must be non-empty and equal length")
    if set(actual) - set(CLASS_NAMES) or set(predicted) - set(CLASS_NAMES):
        raise ValueError("unexpected class label")

    confusion = {
        truth: {guess: 0 for guess in CLASS_NAMES}
        for truth in CLASS_NAMES
    }
    for truth, guess in zip(actual, predicted):
        confusion[truth][guess] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in CLASS_NAMES:
        other = CLASS_NAMES[1] if label == CLASS_NAMES[0] else CLASS_NAMES[0]
        true_positive = confusion[label][label]
        false_negative = confusion[label][other]
        false_positive = confusion[other][label]
        support = true_positive + false_negative
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    accuracy = sum(confusion[label][label] for label in CLASS_NAMES) / len(actual)
    balanced_accuracy = sum(float(per_class[label]["recall"]) for label in CLASS_NAMES) / 2
    macro_f1 = sum(float(per_class[label]["f1"]) for label in CLASS_NAMES) / 2
    return {
        "confusion_matrix": {
            "actual_labels": list(CLASS_NAMES),
            "predicted_labels": list(CLASS_NAMES),
            "counts": [[confusion[truth][guess] for guess in CLASS_NAMES] for truth in CLASS_NAMES],
        },
        "per_class": per_class,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
    }


def probability_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize SIGNAL probabilities with linearly interpolated quantiles."""
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    mean = sum(ordered) / len(ordered)
    variance = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p05": quantile(0.05),
        "p25": quantile(0.25),
        "median": quantile(0.50),
        "p75": quantile(0.75),
        "p95": quantile(0.95),
        "max": ordered[-1],
        "mean": mean,
        "std_population": math.sqrt(variance),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device")
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"missing model: {args.model}")
    if not args.data.is_dir():
        raise SystemExit(f"missing dataset: {args.data}")
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.imgsz < 1 or args.batch < 1:
        raise SystemExit("--imgsz and --batch must be positive")

    try:
        rows, dataset_manifest = validate_split(args.data, args.split)
        device = choose_device(args.device)
        model, names, signal_id = load_classifier(args.model, device)
        probabilities = predict_signal_probabilities(
            model,
            rows,
            device=device,
            signal_id=signal_id,
            image_size=args.imgsz,
            batch_size=args.batch,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"evaluation refused: {exc}") from exc

    actual = [str(row["label"]) for row in rows]
    predicted = ["SIGNAL" if value >= args.threshold else "NO_SIGNAL" for value in probabilities]
    metrics = classification_metrics(actual, predicted)
    probability_by_actual = {
        label: probability_summary(
            [value for value, truth in zip(probabilities, actual) if truth == label]
        )
        for label in CLASS_NAMES
    }
    report: dict[str, Any] = {
        "protocol": "fixed_w10_manifest_split_evaluation_v1",
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "data": str(args.data.resolve()),
        "dataset_protocol": dataset_manifest.get("protocol"),
        "manifest_jsonl_sha256": dataset_manifest.get("manifest_jsonl_sha256"),
        "split": args.split,
        "holdout_read": False,
        "holdout_start": HOLDOUT_START.isoformat(),
        "n_images": len(rows),
        "threshold": args.threshold,
        "image_size": args.imgsz,
        "transform": "white_letterbox_full_frame_then_ToTensor",
        "device": device,
        "model_class_ids": names,
        **metrics,
        "signal_probability": {
            "overall": probability_summary(probabilities),
            "by_actual_class": probability_by_actual,
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
