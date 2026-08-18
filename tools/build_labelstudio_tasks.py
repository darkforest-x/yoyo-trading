#!/usr/bin/env python3
"""Render dual images and write a Label Studio task JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from yoyo.datasets.gold_render import load_causal_prefix, render_context, render_local, sha256_file
from yoyo.datasets.gold_schema import gold_id
from yoyo.datasets.window_render import SourceError

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/gold_annotation_v1.json").read_text())
HOLD = pd.Timestamp(CFG["holdout_start"])
FABLE = Path(json.loads((ROOT / "configs/source_repo.json").read_text())["source_repo"])


def post_bars(decision_time: str, want: int) -> int:
    dt = pd.Timestamp(decision_time)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    avail = int((HOLD - dt) / pd.Timedelta(minutes=15)) - 1
    return max(0, min(want, avail))


def task_config_sha() -> str:
    xml = (ROOT / "configs/labelstudio_gold_v1.xml").read_bytes()
    cfg = (ROOT / "configs/gold_annotation_v1.json").read_bytes()
    return hashlib.sha256(xml + cfg).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "datasets/gold_candidates_v1.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "datasets/gold_labelstudio_v1",
    )
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.candidates.read_text(encoding="utf-8").splitlines()
        if line
    ]
    img_c = args.out_dir / "images" / "context"
    img_l = args.out_dir / "images" / "local"
    img_c.mkdir(parents=True, exist_ok=True)
    img_l.mkdir(parents=True, exist_ok=True)
    cfg_sha = task_config_sha()
    tasks = []
    skipped = []
    for row in rows:
        gid = gold_id(row["symbol"], row.get("timeframe", "15m"), row["decision_time"])
        post = post_bars(row["decision_time"], CFG["context_post_bars"])
        last = int(row["decision_bar"]) + post
        try:
            frame = load_causal_prefix(FABLE / row["source_path"], last)
        except (SourceError, FileNotFoundError) as exc:
            skipped.append({"gold_id": gid, "error": str(exc)})
            continue
        times = pd.to_datetime(frame["open_time"], utc=True)
        if (times >= HOLD).any():
            skipped.append({"gold_id": gid, "error": "holdout"})
            continue
        local_path = img_l / f"{gid}.png"
        ctx_path = img_c / f"{gid}.png"
        print(f"render {len(tasks)+1}/{len(rows)} {gid}", flush=True)
        local = render_local(frame, int(row["decision_bar"]), CFG["local_window_length"], local_path)
        ctx = render_context(
            frame,
            int(row["decision_bar"]),
            row.get("candidate_core_start"),
            row.get("candidate_core_end"),
            pre_bars=CFG["context_pre_bars"],
            post_bars=post,
            out_path=ctx_path,
        )
        tasks.append(
            {
                "data": {
                    "gold_id": gid,
                    "symbol": row["symbol"],
                    "timeframe": row.get("timeframe", "15m"),
                    "source_repo": row.get("source_repo", "fable-trading"),
                    "source_path": row["source_path"],
                    "decision_bar": int(row["decision_bar"]),
                    "decision_time": row["decision_time"],
                    "local_start_bar": local["local_start_bar"],
                    "local_end_bar": local["local_end_bar"],
                    "local_window_length": local["local_window_length"],
                    "context_start_bar": ctx["context_start_bar"],
                    "context_end_bar": ctx["context_end_bar"],
                    "context_image": str(ctx_path.resolve()),
                    "local_image": str(local_path.resolve()),
                    "context_image_sha256": sha256_file(ctx_path),
                    "local_image_sha256": sha256_file(local_path),
                    "task_config_sha256": cfg_sha,
                    "local_image_width": local["width"],
                }
            }
        )
    (args.out_dir / "tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"tasks": len(tasks), "skipped": skipped, "out": str(args.out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
