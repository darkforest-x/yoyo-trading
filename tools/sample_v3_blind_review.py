#!/usr/bin/env python3
"""Frozen 10% stratified sample for Owner V5 §6.3 blind re-review.

Does not open images or holdout. Writes sample_id + stratum only.
Owner reviews blindly; this file is the draw, not the answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "datasets/dataset_v3_gold_core_v1/manifest.jsonl"
DEFAULT_OUT = ROOT / "manifests/v3_blind_review_sample_v1.jsonl"
SEED = 0
FRACTION = 0.10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-manifest", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fraction", type=float, default=FRACTION)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.in_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        kind = "positive" if row.get("class_name") else "negative"
        buckets[(str(row.get("split")), kind)].append(row)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    for (split, kind), items in sorted(buckets.items()):
        items = sorted(items, key=lambda r: r["sample_id"])
        rng.shuffle(items)
        n = max(1, round(len(items) * args.fraction))
        for row in items[:n]:
            picked.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "kind": kind,
                    "symbol": row.get("symbol"),
                    "decision_time": row.get("decision_time"),
                    "image_path": row.get("image_path"),
                    "draw_seed": args.seed,
                    "draw_fraction": args.fraction,
                    "stratum": f"{split}:{kind}",
                }
            )
    picked.sort(key=lambda r: r["sample_id"])
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked)
    args.out.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(
        json.dumps(
            {
                "n": len(picked),
                "by_stratum": {
                    f"{split}:{kind}": sum(1 for r in picked if r["stratum"] == f"{split}:{kind}")
                    for split, kind in sorted(buckets)
                },
                "out": str(args.out),
                "sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
