#!/usr/bin/env python3
"""Convert every adapted legacy source into event-level W10 candidates."""

from __future__ import annotations

import json
from pathlib import Path

from yoyo.datasets.legacy_gold_migration.io import add_common_flags, load_config
from yoyo.datasets.legacy_gold_migration.migration import migrate, write_migration

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs/fixed_w10_core4_confirm1_v1.json"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_flags(parser, default_config=DEFAULT_CFG)
    parser.add_argument("--yoyo-root", type=Path, default=ROOT)
    parser.add_argument("--fable-root", type=Path, default=Path("/Users/zhangzc/fable-trading"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = args.out or Path(cfg["paths"]["dataset_root"]) / "migration"
    result = migrate(
        args.yoyo_root,
        args.fable_root,
        cfg,
        limit=args.limit,
        seed=args.seed,
    )
    summary = {k: v for k, v in result.items() if k not in {"events", "conflicts", "source_records"}}
    if args.dry_run:
        print(json.dumps({"dry_run": True, **summary}, indent=2, ensure_ascii=False))
        return 0
    write_migration(result, out)
    print(json.dumps({"out": str(out), **summary}, ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
