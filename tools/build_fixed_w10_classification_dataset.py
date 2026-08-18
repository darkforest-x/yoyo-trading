#!/usr/bin/env python3
"""Render frozen Gold into train/val/test SIGNAL|NO_SIGNAL folders. No overlay."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

from yoyo.datasets.gold_render import load_causal_prefix, sha256_file
from yoyo.datasets.legacy_gold_migration.canonical import no_signal_slot, w10_fields
from yoyo.datasets.legacy_gold_migration.io import (
    add_common_flags,
    atomic_write_json,
    git_head,
    git_status_short,
    load_config,
    read_jsonl,
    sha256_text,
    write_jsonl,
)
from yoyo.datasets.legacy_gold_migration.renderer import render_w10
from yoyo.datasets.window_render import SourceError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs/fixed_w10_core4_confirm1_v1.json"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_flags(parser, default_config=DEFAULT_CFG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["dataset_root"])
    fable = Path(cfg["paths"]["fable_root"])
    holdout = pd.Timestamp(cfg["holdout_start"])
    gold = read_jsonl(root / "gold" / "events.jsonl")
    if args.limit:
        gold = gold[: args.limit]
    cls = root / "classification"
    for split in ("train", "val", "test"):
        for lab in ("SIGNAL", "NO_SIGNAL"):
            (cls / split / lab).mkdir(parents=True, exist_ok=True)
    image_rows = []
    skipped = []
    for row in gold:
        shape = row.get("shape_label")
        if shape not in {"SIGNAL", "NO_SIGNAL"}:
            continue
        if row.get("contains_other_core"):
            skipped.append({"gold_id": row.get("gold_id"), "reason": "contains_other_core"})
            continue
        split = row.get("split") or "train"
        try:
            if shape == "SIGNAL":
                mapped = w10_fields(int(row["core_start_bar"]), int(row["core_end_exclusive_bar"]))
            else:
                conf = int(row.get("confirmation_bar") or row["decision_bar"])
                mapped = no_signal_slot(conf)
            last = mapped["window_end_exclusive_bar"] - 1
            frame = load_causal_prefix(fable / row["source_path"], last)
            times = pd.to_datetime(frame["open_time"], utc=True)
            if (times >= holdout).any():
                raise ValueError("holdout")
            window = frame.iloc[mapped["window_start_bar"] : mapped["window_end_exclusive_bar"]].reset_index(drop=True)
            if len(window) != 10:
                raise ValueError(f"window_length {len(window)}")
            last_time = pd.to_datetime(window["open_time"].iloc[-1], utc=True)
            dec_time = pd.to_datetime(frame["open_time"].iloc[mapped["decision_bar"]], utc=True)
            if last_time != dec_time:
                raise ValueError("visible_end_bar != decision_bar")
            img, _tf = render_w10(window, y_pad_frac=float(cfg["y_pad_frac"]), overlay=False)
            out_path = cls / split / shape / f"{row['gold_id']}.png"
            if not args.dry_run:
                cv2.imwrite(str(out_path), img)
            image_rows.append(
                {
                    **row,
                    **mapped,
                    "split": split,
                    "image_path": str(out_path),
                    "image_sha256": None if args.dry_run else sha256_file(out_path),
                    "future_used_in_model_input": False,
                    "holdout_read": False,
                    "render_status": "OK",
                }
            )
        except (SourceError, ValueError, FileNotFoundError, KeyError) as exc:
            skipped.append({"gold_id": row.get("gold_id"), "reason": str(exc)})
    if args.dry_run:
        print(json.dumps({"dry_run": True, "n": len(image_rows), "skipped": len(skipped)}))
        return 0
    write_jsonl(root / "manifests" / "image_manifest.jsonl", image_rows)
    write_jsonl(root / "manifests" / "event_manifest.jsonl", gold)
    by_split = Counter((r["split"], r["shape_label"]) for r in image_rows)
    shas = [r["image_sha256"] for r in image_rows if r.get("image_sha256")]
    dup_across = 0
    seen = {}
    for r in image_rows:
        sha = r.get("image_sha256")
        if not sha:
            continue
        prev = seen.get(sha)
        if prev and prev != r["split"]:
            dup_across += 1
        seen[sha] = r["split"]
    dataset_manifest = {
        "task_name": cfg["task_name"],
        "training_eligible": False,
        "counts": {f"{a}/{b}": n for (a, b), n in sorted(by_split.items())},
        "n_images": len(image_rows),
        "skipped": skipped,
        "duplicate_image_sha_across_splits": dup_across,
        "holdout_rows": 0,
        "yoyo_head": git_head(ROOT),
        "fable_head": git_head(fable),
        "yoyo_dirty_lines": len([l for l in git_status_short(ROOT).splitlines() if l.strip()]),
        "config_sha256": sha256_text(args.config.read_text(encoding="utf-8")),
        "builder_commit": git_head(ROOT),
        "gold_events_sha256": sha256_file(root / "gold" / "events.jsonl") if (root / "gold" / "events.jsonl").exists() else None,
    }
    atomic_write_json(root / "manifests" / "dataset_manifest.json", dataset_manifest)
    atomic_write_json(
        root / "manifests" / "split_manifest.json",
        {f"{a}/{b}": n for (a, b), n in sorted(by_split.items())},
    )
    sums = "\n".join(f"{r['image_sha256']}  {r['gold_id']}" for r in image_rows if r.get("image_sha256"))
    (root / "manifests" / "sha256sums.txt").write_text(sums + "\n", encoding="utf-8")
    yoyo_man = ROOT / "manifests" / "fixed_w10_core4_confirm1_v1.json"
    atomic_write_json(yoyo_man, dataset_manifest)
    print(json.dumps({"n": len(image_rows), "skipped": len(skipped), "dup_across": dup_across, "out": str(cls)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
