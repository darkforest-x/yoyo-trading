#!/usr/bin/env python3
"""Static val recall by left/middle/right crop bin (V5 §8.5).

Uses the V3.1 manifest spread_bin field. Not a promotion criterion.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


def load_rows(manifest: Path, split: str) -> list[dict]:
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("split") != split:
            continue
        rows.append(row)
    return rows


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = ua + ub - inter
    return inter / den if den else 0.0


def label_xyxy(label_path: Path, width: int, height: int) -> list[list[float]]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = map(float, parts[:5])
        boxes.append(
            [
                (xc - w / 2) * width,
                (yc - h / 2) * height,
                (xc + w / 2) * width,
                (yc + h / 2) * height,
            ]
        )
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/dataset_v3_1_position_spread_v1/manifest.jsonl"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.manifest, "val")
    model = YOLO(str(args.weights))
    hits = defaultdict(lambda: {"pos": 0, "hit": 0})
    fp = 0
    tn = 0
    for row in rows:
        image = args.root / row["image_path"]
        pred = model.predict(source=str(image), conf=args.conf, verbose=False, device=args.device)[0]
        pred_boxes = [list(map(float, box.xyxy[0].tolist())) for box in pred.boxes]
        label = args.root / row["image_path"].replace("/images/", "/labels/").replace(".png", ".txt")
        gt = label_xyxy(label, int(pred.orig_shape[1]), int(pred.orig_shape[0]))
        bin_name = row.get("spread_bin") or ("negative" if not row.get("class_name") else "unbinned")
        if gt:
            hits[bin_name]["pos"] += 1
            ok = any(iou_xyxy(g, p) >= args.iou for g in gt for p in pred_boxes)
            if ok:
                hits[bin_name]["hit"] += 1
            if not pred_boxes:
                hits[bin_name].setdefault("miss", 0)
                hits[bin_name]["miss"] = hits[bin_name].get("miss", 0) + 1
        else:
            if pred_boxes:
                fp += 1
            else:
                tn += 1

    by_bin = {}
    for name, rec in sorted(hits.items()):
        pos = rec["pos"]
        by_bin[name] = {
            "positives": pos,
            "hits": rec["hit"],
            "recall": (rec["hit"] / pos) if pos else None,
        }
    payload = {
        "schema": "yoyo_static_position_bins_v1",
        "per_v5": True,
        "note": "mAP/bin-recall is not a promotion criterion",
        "weights": str(args.weights),
        "conf": args.conf,
        "match_iou": args.iou,
        "val_rows": len(rows),
        "negative_fp_images": fp,
        "negative_tn_images": tn,
        "by_bin": by_bin,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
