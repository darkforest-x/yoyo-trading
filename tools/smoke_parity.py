#!/usr/bin/env python3
"""Prove the yoyo build chain reproduces the frozen dataset byte for byte.

Before Dataset V3 can claim to reuse R1's samples, this repo has to show it can
regenerate them: same prefix loading, same MAs, same renderer, same box
arithmetic. The check is a SHA comparison against the image and label hashes
recorded in fable-trading's manifests -- not a visual inspection, and not a
"looks the same" tolerance. A single differing byte fails the sample.

Samples are drawn deterministically (seeded, spread across window lengths) so
the same command reproduces the same sample set.

Read-only with respect to fable-trading: images are re-rendered into a
temporary directory and deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from yoyo.datasets.window_render import SourceError, enrich, load_prefix, render_sample

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CONFIG = ROOT / "configs/source_repo.json"
DEFAULT_DATASET = "datasets/owner_short_gold_center_hardneg_r1"
DEFAULT_OUT = ROOT / "manifests/smoke_parity_r1_dataset.json"
MANIFESTS = ("positive_manifest", "negative_manifest", "hard_negative_manifest")
SEED = 20260812


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def pick(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    """Spread the sample across window lengths instead of clustering on one."""
    buckets: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row.get("win_len"), []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    picked: list[dict[str, Any]] = []
    order = sorted(buckets, key=lambda key: (key is None, key))
    while len(picked) < count and any(buckets[key] for key in order):
        for key in order:
            if buckets[key] and len(picked) < count:
                picked.append(buckets[key].pop())
    return picked


def check_row(row: dict[str, Any], repo: Path) -> dict[str, Any]:
    source = repo / str(row["source_csv"])
    win_start, win_end = int(row["win_start"]), int(row["win_end"])
    core = row.get("core_global")
    result: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "class": row.get("class"),
        "symbol": row["symbol"],
        "win_len": row.get("win_len"),
        "split": row.get("split"),
        "image_sha256_expected": row["image_sha256"],
        "label_sha256_expected": row["label_sha256"],
    }
    try:
        enriched = enrich(load_prefix(source, win_end))
        rendered = render_sample(enriched, win_start, win_end, tuple(core) if core else None)
    except (SourceError, OSError, FileNotFoundError) as error:
        result.update({"image_match": False, "label_match": False, "error": str(error)})
        return result

    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "sample.png"
        if not cv2.imwrite(str(path), rendered["image"]):
            raise OSError(path)
        image_sha = sha256_bytes(path.read_bytes())
    label_sha = sha256_bytes(rendered["label"].encode("utf-8"))
    result.update(
        {
            "image_sha256_rebuilt": image_sha,
            "label_sha256_rebuilt": label_sha,
            "image_match": image_sha == row["image_sha256"],
            "label_match": label_sha == row["label_sha256"],
            "bars": rendered["bars"],
            "error": None,
        }
    )
    if core and row.get("yolo_box"):
        expected = [round(float(value), 6) for value in row["yolo_box"]]
        rebuilt = [round(float(value), 6) for value in rendered["box"]]
        result["box_match"] = expected == rebuilt
    return result


def run(source_config: Path, dataset_rel: str, per_class: int, out_path: Path) -> dict[str, Any]:
    source = json.loads(source_config.read_text(encoding="utf-8"))
    repo = Path(source["source_repo"])
    dataset = repo / dataset_rel
    if not dataset.exists():
        raise FileNotFoundError(f"dataset not found: {dataset}")
    rng = random.Random(SEED)

    checked: list[dict[str, Any]] = []
    for name in MANIFESTS:
        path = dataset / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = [row for row in read_jsonl(path) if row.get("source_csv")]
        for row in pick(rows, per_class, rng):
            checked.append(check_row(row, repo))

    failures = [row for row in checked if not (row["image_match"] and row["label_match"])]
    summary = {
        "protocol": "yoyo_l1_smoke_parity_v1_20260812",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(repo),
        "dataset": dataset_rel,
        "seed": SEED,
        "per_class": per_class,
        "checked": len(checked),
        "by_class": dict(Counter(row["class"] for row in checked)),
        "image_and_label_match": len(checked) - len(failures),
        "failures": len(failures),
        "box_mismatches": sum(1 for row in checked if row.get("box_match") is False),
        "parity": not failures,
        "samples": checked,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--per-class", type=int, default=15)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = run(args.source_config, args.dataset, args.per_class, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, ensure_ascii=False, indent=2))
    return 0 if summary["parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
