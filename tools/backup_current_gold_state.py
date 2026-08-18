#!/usr/bin/env python3
"""Copy current Gold / LS / git state before any migration rewrite."""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from yoyo.datasets.legacy_gold_migration.io import (
    add_common_flags,
    atomic_write_json,
    git_head,
    git_status_short,
    sha256_file,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs/fixed_w10_core4_confirm1_v1.json"


def _copy_if(src: Path, dest: Path) -> dict | None:
    if not src.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dest)
    info = {"path": str(src), "backup": str(dest)}
    if src.is_file():
        info["sha256"] = sha256_file(src)
        info["bytes"] = src.stat().st_size
    return info


def dump_ls_yoyo_gold(fable: Path, dest: Path) -> dict | None:
    db = fable / "label_studio_data/label_studio.sqlite3"
    if not db.exists():
        return None
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        proj = con.execute("SELECT id FROM project WHERE title=?", ("yoyo_gold_v1",)).fetchone()
        if not proj:
            return {"project": "yoyo_gold_v1", "n": 0}
        rows = []
        q = """
        SELECT t.data, tc.result, tc.updated_at
        FROM task_completion tc JOIN task t ON t.id=tc.task_id
        WHERE t.project_id=?
        """
        for data, result, updated in con.execute(q, (proj[0],)):
            rows.append(
                {
                    "data": json.loads(data),
                    "result": json.loads(result) if result else [],
                    "updated_at": updated,
                }
            )
        write_jsonl(dest, rows)
        return {"project": "yoyo_gold_v1", "n": len(rows), "path": str(dest)}
    finally:
        con.close()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_flags(parser, default_config=DEFAULT_CFG)
    parser.add_argument("--yoyo-root", type=Path, default=ROOT)
    parser.add_argument("--fable-root", type=Path, default=Path("/Users/zhangzc/fable-trading"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = Path(cfg["paths"]["dataset_root"]) / "backups" / f"gold_migration_{stamp}"
    if args.dry_run:
        print(json.dumps({"dry_run": True, "dest": str(dest)}))
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    yoyo, fable = args.yoyo_root, args.fable_root
    copied = []
    for rel in (
        "datasets/gold_v1.jsonl",
        "datasets/gold_v1_demo.jsonl",
        "datasets/gold_candidates_v1.jsonl",
        "datasets/gold_labelstudio_v1/tasks.json",
        "configs/gold_annotation_v1.json",
        "configs/labelstudio_gold_v1.xml",
        "tools/gold_annotate.html",
        "tools/serve_gold_annotate.py",
    ):
        info = _copy_if(yoyo / rel, dest / rel)
        if info:
            copied.append(info)
    ls = dump_ls_yoyo_gold(fable, dest / "labelstudio_yoyo_gold_v1.jsonl")
    meta = {
        "timestamp": stamp,
        "yoyo_head": git_head(yoyo),
        "fable_head": git_head(fable),
        "yoyo_status": git_status_short(yoyo)[:4000],
        "fable_status": git_status_short(fable)[:2000],
        "copied": copied,
        "labelstudio_yoyo_gold_v1": ls,
        "notes": "PNG task images not copied (large); tasks.json + gold_v1.jsonl are the save files. 8768 stores JSONL, not localStorage.",
    }
    atomic_write_json(dest / "MANIFEST.json", meta)
    lines = []
    for item in copied:
        if "sha256" in item:
            lines.append(f"{item['sha256']}  {item['path']}")
    (dest / "sha256sums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"dest": str(dest), "n_copied": len(copied), "ls": ls}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
