#!/usr/bin/env python3
"""Gold JSONL → causal YOLO dataset. Context images never enter images/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from yoyo.datasets.gold_box import UnstableWickError, yolo_xywh
from yoyo.datasets.gold_render import load_causal_prefix, render_local
from yoyo.datasets.gold_schema import parse_jsonl
from yoyo.datasets.window_render import SourceError

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/gold_annotation_v1.json").read_text())
HOLD = pd.Timestamp(CFG["holdout_start"])
FABLE = Path(json.loads((ROOT / "configs/source_repo.json").read_text())["source_repo"])


def split_for(decision_time: str) -> str:
    t = pd.Timestamp(decision_time)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return "val" if t >= pd.Timestamp("2026-03-15T00:00:00+00:00") else "train"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = parse_jsonl(args.gold.read_text(encoding="utf-8"))
    for split in ("train", "val"):
        (args.out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    written = []
    for row in rows:
        if row["shape_label"] == "IGNORE":
            continue
        if pd.Timestamp(row["decision_time"]) >= HOLD:
            raise SystemExit(f"holdout gold: {row['gold_id']}")
        split = split_for(row["decision_time"])
        frame = load_causal_prefix(FABLE / row["source_path"], int(row["decision_bar"]))
        times = pd.to_datetime(frame["open_time"], utc=True)
        if (times >= HOLD).any():
            raise SystemExit("holdout leaked into local render")
        img_path = args.out_dir / "images" / split / f"{row['gold_id']}.png"
        local = render_local(frame, int(row["decision_bar"]), int(row["local_window_length"]), img_path)
        # rewrite without footer for detector? Spec: same renderer. Footer is
        # annotation aid. Training image must be causal renderer only.
        from yoyo.layers.l1_detection.render import render_chart

        window = frame.iloc[local["local_start_bar"] : local["local_end_bar"] + 1].reset_index(drop=True)
        render_chart(window, out_path=img_path)
        lbl_path = args.out_dir / "labels" / split / f"{row['gold_id']}.txt"
        if row["shape_label"] == "POSITIVE":
            start = int(row["core_start_bar"]) - local["local_start_bar"]
            end = int(row["core_end_bar"]) - local["local_start_bar"]
            try:
                box = yolo_xywh(window, start, end, pad_frac=CFG["box_pad_frac"])
            except (UnstableWickError, ValueError, SourceError):
                continue
            lbl_path.write_text(box["label"], encoding="utf-8")
        else:
            lbl_path.write_text("", encoding="utf-8")
        written.append({**row, "split": split, "image_path": str(img_path.resolve().relative_to(ROOT))})
    (args.out_dir / "data.yaml").write_text(
        f"path: {args.out_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: ma_dense_core\n",
        encoding="utf-8",
    )
    (args.out_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in written), encoding="utf-8"
    )
    print(json.dumps({"n": len(written), "out": str(args.out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
