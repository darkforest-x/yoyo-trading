#!/usr/bin/env python3
"""Audit Gold JSONL + optional YOLO dir: leak, split, context mix, SHA."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from yoyo.datasets.gold_schema import parse_jsonl

ROOT = Path(__file__).resolve().parents[1]
HOLD = pd.Timestamp("2026-05-04T00:00:00+00:00")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--yolo-dir", type=Path, default=None)
    args = parser.parse_args()
    rows = parse_jsonl(args.gold.read_text(encoding="utf-8"))
    events = defaultdict(set)
    holdout = 0
    for row in rows:
        t = pd.Timestamp(row["decision_time"])
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        if t >= HOLD or row.get("holdout_read"):
            holdout += 1
        events[row["gold_id"]].add(row.get("split") or "gold")
    cross = {k: list(v) for k, v in events.items() if len(v) > 1}
    context_in_yolo = []
    if args.yolo_dir:
        yolo_imgs = {p.name for p in (args.yolo_dir / "images").rglob("*.png")}
        ctx = args.yolo_dir.parent / "gold_labelstudio_v1" / "images" / "context"
        if ctx.exists():
            context_in_yolo = sorted(yolo_imgs & {p.name for p in ctx.glob("*.png")})
    report = {
        "n": len(rows),
        "by_shape": dict(Counter(r["shape_label"] for r in rows)),
        "holdout_rows": holdout,
        "duplicate_gold_ids_across_split": cross,
        "context_images_in_yolo": context_in_yolo,
        "ok": holdout == 0 and not cross and not context_in_yolo,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
