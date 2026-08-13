#!/usr/bin/env python3
"""Build a small candidate pool. Hidden labels. No future / conf prefilter."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/gold_annotation_v1.json").read_text())
HOLD = pd.Timestamp(CFG["holdout_start"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-manifest",
        type=Path,
        default=ROOT / "datasets/dataset_v3_2_reviewed_core_v1/manifest.jsonl",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "datasets/gold_candidates_v1.jsonl")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.from_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    usable = []
    for row in rows:
        decision_t = pd.Timestamp(row["decision_time"])
        if decision_t.tzinfo is None:
            decision_t = decision_t.tz_localize("UTC")
        if decision_t >= HOLD:
            continue
        if int(row["decision_bar"]) < CFG["local_window_length"] - 1:
            continue
        usable.append(row)
    rng = random.Random(args.seed)
    usable.sort(key=lambda r: r["sample_id"])
    rng.shuffle(usable)
    picked = usable[: args.n]
    out_rows = []
    for row in picked:
        out_rows.append(
            {
                "sample_id": row["sample_id"],
                "symbol": row["symbol"],
                "timeframe": row.get("timeframe", "15m"),
                "source_repo": "fable-trading",
                "source_path": row["source_path"],
                "decision_bar": int(row["decision_bar"]),
                "decision_time": row["decision_time"],
                "candidate_core_start": row.get("box_start_bar"),
                "candidate_core_end": row.get("box_end_bar"),
                "candidate_source": "reviewed_core_hidden",
                "holdout_read": False,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows),
        encoding="utf-8",
    )
    print(json.dumps({"n": len(out_rows), "out": str(args.out), "pool": len(usable)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
