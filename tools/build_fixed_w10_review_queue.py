#!/usr/bin/env python3
"""Build the Owner review queue. DIRECT is sampled at 15%, not fully re-labeled."""

from __future__ import annotations

import json
import random
from math import ceil
from pathlib import Path

from yoyo.datasets.legacy_gold_migration.io import add_common_flags, load_config, read_jsonl, write_jsonl
from yoyo.datasets.legacy_gold_migration.review_queue import build_queue

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs/fixed_w10_core4_confirm1_v1.json"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_flags(parser, default_config=DEFAULT_CFG)
    parser.add_argument("--in-jsonl", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["dataset_root"])
    src = args.in_jsonl or root / "migration" / "migrated_candidates.jsonl"
    events = read_jsonl(src)
    if args.limit:
        events = events[: args.limit]
    rng = random.Random(args.seed)
    direct = [r for r in events if r.get("migration_status") == "DIRECT"]
    n_spot = ceil(len(direct) * float(cfg["direct_spot_check_frac"])) if direct else 0
    spot = list(direct)
    rng.shuffle(spot)
    spot = spot[:n_spot]
    for row in spot:
        row["spot_check"] = True
    queue = build_queue(events, spot)
    out = args.out or root / "review" / "queue.jsonl"
    if args.dry_run:
        print(json.dumps({"dry_run": True, "queue": len(queue), "direct": len(direct), "spot": len(spot)}))
        return 0
    write_jsonl(out, queue)
    write_jsonl(root / "migration" / "direct_spot_check.jsonl", spot)
    print(json.dumps({"out": str(out), "queue": len(queue), "direct": len(direct), "spot": len(spot)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
