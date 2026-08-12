#!/usr/bin/env python3
"""Build Dataset V3.1: same Gold Core labels, causal W re-picked for L/M/R.

One variable vs V3: window length in 12–19, still ending at decision.
SMA+EMA 20/60/120 renderer is frozen. Holdout unread. Negatives copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from yoyo.datasets.window_render import SourceError, enrich, load_prefix, render_sample

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / "datasets/dataset_v3_gold_core_v1"
DEFAULT_ASSIGN = ROOT / "manifests/dataset_v3_1_position_spread_assignments.jsonl"
DEFAULT_OUT = ROOT / "datasets/dataset_v3_1_position_spread_v1"
DEFAULT_MANIFEST = ROOT / "manifests/dataset_v3_1_position_spread_v1.json"
CLASS_NAME = "ma_dense_core"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build(parent: Path, assign_path: Path, out_dir: Path, manifest_path: Path, builder_commit: str) -> dict:
    parent_rows = read_jsonl(parent / "manifest.jsonl")
    assign = {row["sample_id"]: row for row in read_jsonl(assign_path)}
    source = json.loads((ROOT / "configs/source_repo.json").read_text(encoding="utf-8"))
    repo = Path(source["source_repo"])

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    written: list[dict[str, Any]] = []
    positives = [r for r in parent_rows if r.get("class_name")]
    negatives = [r for r in parent_rows if not r.get("class_name")]

    # copy negatives byte-for-byte
    for row in negatives:
        src_img = ROOT / row["image_path"]
        dst_img = out_dir / "images" / row["split"] / f"{row['sample_id']}.png"
        dst_lbl = out_dir / "labels" / row["split"] / f"{row['sample_id']}.txt"
        shutil.copy2(src_img, dst_img)
        dst_lbl.write_text("", encoding="utf-8")
        new = dict(row)
        new["image_path"] = str(dst_img.relative_to(ROOT))
        new["builder_commit"] = builder_commit
        new["parent_sample_id"] = row["sample_id"]
        new["spread_bin"] = None
        written.append(new)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        groups[str(row["symbol"])].append(row)

    for symbol, items in sorted(groups.items()):
        required_end = max(int(assign[i["sample_id"]]["chosen_win_end"]) for i in items)
        source_csv = repo / items[0]["source_path"]
        enriched = enrich(load_prefix(source_csv, required_end))
        for row in items:
            plan = assign[row["sample_id"]]
            if plan["spread_status"] != "ok":
                raise SourceError(f"{row['sample_id']} has no legal window")
            rendered = render_sample(
                enriched,
                int(plan["chosen_win_start"]),
                int(plan["chosen_win_end"]),
                (int(row["box_start_bar"]), int(row["box_end_bar"])),
            )
            dst_img = out_dir / "images" / row["split"] / f"{row['sample_id']}.png"
            dst_lbl = out_dir / "labels" / row["split"] / f"{row['sample_id']}.txt"
            if not cv2.imwrite(str(dst_img), rendered["image"]):
                raise OSError(dst_img)
            dst_lbl.write_text(rendered["label"], encoding="utf-8")
            new = dict(row)
            new.update(
                {
                    "window_start_bar": int(plan["chosen_win_start"]),
                    "window_end_bar": int(plan["chosen_win_end"]),
                    "window_length": int(plan["chosen_window_len"]),
                    "visible_end_bar": int(plan["chosen_win_end"]),
                    "decision_bar": int(row["decision_bar"]),
                    "image_path": str(dst_img.relative_to(ROOT)),
                    "image_sha256": sha256_bytes(dst_img.read_bytes()),
                    "label_sha256": sha256_bytes(rendered["label"].encode("utf-8")),
                    "builder_commit": builder_commit,
                    "parent_sample_id": row["sample_id"],
                    "spread_bin": plan["chosen_bin"],
                    "spread_rel": plan["chosen_rel"],
                    "old_window_length": row["window_length"],
                    "parity": None,
                }
            )
            window = enriched.iloc[int(plan["chosen_win_start"]) : int(plan["chosen_win_end"]) + 1]
            new["window_start_time"] = window["open_time"].iloc[0].isoformat()
            new["decision_time"] = window["open_time"].iloc[-1].isoformat()
            written.append(new)

    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: {CLASS_NAME}\n",
        encoding="utf-8",
    )
    rows_path = out_dir / "manifest.jsonl"
    rows_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in written), encoding="utf-8"
    )
    pos = [r for r in written if r.get("class_name")]
    summary = {
        "protocol": "yoyo_dataset_v3_1_position_spread_v1_20260812",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parent_dataset": "dataset_v3_gold_core_v1",
        "class_name": CLASS_NAME,
        "direction_scope": "short_pipeline",
        "window_protocol": "owner_w12_19_causal_end_at_decision_position_spread",
        "feature_stack": "SMA20/60/120 + EMA20/60/120",
        "single_variable": "positive_crop_W_in_12_19",
        "dataset_dir": str(out_dir),
        "builder_commit": builder_commit,
        "source_repo": str(repo),
        "counts": {
            "total": len(written),
            "positive": len(pos),
            "negative": len(written) - len(pos),
            "train_positive": sum(1 for r in pos if r["split"] == "train"),
            "val_positive": sum(1 for r in pos if r["split"] == "val"),
        },
        "positive_core_position": dict(sorted(Counter(r["spread_bin"] for r in pos).items())),
        "window_lengths": dict(sorted(Counter(r["window_length"] for r in pos).items())),
        "holdout_start": source["holdout_start"],
        "holdout_read": False,
        "future_used_for_l1_label": False,
        "manifest_path": str(rows_path),
        "manifest_sha256": sha256_bytes(rows_path.read_bytes()),
        "training_eligible": True,
        "note": (
            "Left is capped by geometry: only cores far enough before decision "
            "can sit in the left third at W>=12. This is the maximum causal spread."
        ),
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGN)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--builder-commit", default="unrecorded")
    args = p.parse_args()
    summary = build(args.parent, args.assignments, args.out_dir, args.manifest, args.builder_commit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
