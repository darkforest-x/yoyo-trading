#!/usr/bin/env python3
"""Contact sheet of frozen W10 images for Owner skim. Overlay-free sources only."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from yoyo.datasets.legacy_gold_migration.io import add_common_flags, load_config, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs/fixed_w10_core4_confirm1_v1.json"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_flags(parser, default_config=DEFAULT_CFG)
    parser.add_argument("--n", type=int, default=24)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["dataset_root"])
    rows = read_jsonl(root / "manifests" / "image_manifest.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[: args.n]
    thumbs = []
    for row in rows:
        path = Path(row["image_path"])
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        thumbs.append(cv2.resize(img, (320, 186)))
    if args.dry_run:
        print(json.dumps({"dry_run": True, "n": len(thumbs)}))
        return 0
    out = root / "reports" / "contact_sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not thumbs:
        out.write_bytes(b"")
        print(json.dumps({"n": 0, "out": str(out)}))
        return 0
    cols = 4
    rows_n = (len(thumbs) + cols - 1) // cols
    h, w = thumbs[0].shape[:2]
    canvas = np.full((rows_n * h, cols * w, 3), 240, dtype=np.uint8)
    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = im
    cv2.imwrite(str(out), canvas)
    print(json.dumps({"n": len(thumbs), "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
