#!/usr/bin/env python3
"""List every legacy annotation source and write the inventory files."""

from __future__ import annotations

import json
from pathlib import Path

from yoyo.datasets.legacy_gold_migration.audit import source_inventory
from yoyo.datasets.legacy_gold_migration.io import add_common_flags, atomic_write_json, load_config
from yoyo.datasets.legacy_gold_migration.sources import load_all_sources
import pandas as pd

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
    inv = source_inventory(args.yoyo_root, args.fable_root)
    holdout = pd.Timestamp(cfg["holdout_start"])
    _recs, counts = load_all_sources(args.yoyo_root, args.fable_root, holdout)
    inv["loaded_records"] = counts
    inv["loaded_total"] = sum(counts.values())
    out = args.out or Path(cfg["paths"]["dataset_root"]) / "migration" / "source_inventory.json"
    if args.dry_run:
        print(json.dumps({"dry_run": True, "loaded_records": counts}, indent=2, ensure_ascii=False))
        return 0
    atomic_write_json(out, inv)
    report_dir = Path(cfg["paths"]["dataset_root"]) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Source inventory\n"]
    md.append(f"- yoyo HEAD `{inv['yoyo_head']}`")
    md.append(f"- fable HEAD `{inv['fable_head']}`")
    md.append("")
    md.append("| source | records |")
    md.append("|---|---:|")
    for k, v in sorted(counts.items()):
        md.append(f"| `{k}` | {v} |")
    md.append("")
    md.append("## Interval semantics")
    for k, v in inv["interval_semantics"].items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("## 8768 annotator")
    for k, v in inv["annotator"].items():
        md.append(f"- **{k}**: `{v}`")
    (report_dir / "source_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    yoyo_rep = args.yoyo_root / "reports" / "fixed_w10_core4_confirm1_v1"
    yoyo_rep.mkdir(parents=True, exist_ok=True)
    (yoyo_rep / "source_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "loaded_records": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
