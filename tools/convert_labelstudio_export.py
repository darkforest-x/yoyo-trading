#!/usr/bin/env python3
"""Label Studio export JSON → Gold JSONL. Never silently edits Owner verdicts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from yoyo.datasets.gold_box import snap_x_to_bar
from yoyo.datasets.gold_schema import parse_jsonl, validate_gold
from yoyo.layers.l1_detection.render import IMG_WIDTH, MARGIN

ROOT = Path(__file__).resolve().parents[1]


def _choice(results: list[dict]) -> str | None:
    for item in results:
        if item.get("from_name") == "shape":
            choices = item.get("value", {}).get("choices") or []
            if choices:
                return str(choices[0])
    return None


def _points(results: list[dict], width: int, n_bars: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in results:
        if item.get("from_name") != "core":
            continue
        labels = item.get("value", {}).get("keypointlabels") or []
        if not labels:
            continue
        x_pct = float(item["value"]["x"])
        x_px = x_pct / 100.0 * width
        out[str(labels[0])] = snap_x_to_bar(x_px, n_bars, width, MARGIN)
    return out


def _notes(results: list[dict]) -> str | None:
    for item in results:
        if item.get("from_name") == "notes":
            text = item.get("value", {}).get("text")
            if isinstance(text, list) and text:
                return str(text[0])
            if isinstance(text, str) and text:
                return text
    return None


def convert_task(task: dict, existing: dict[str, dict]) -> dict | None:
    data = task.get("data") or {}
    gold_id = data.get("gold_id")
    if not gold_id:
        raise ValueError("task missing gold_id")
    if gold_id in existing:
        return None
    anns = task.get("annotations") or []
    if not anns:
        return None
    results = anns[-1].get("result") or []
    shape = _choice(results)
    if shape is None:
        return None
    n_bars = int(data.get("local_window_length") or 30)
    width = int(data.get("local_image_width") or IMG_WIDTH)
    pts = _points(results, width, n_bars)
    local_start = int(data["local_start_bar"])
    row = {
        "gold_id": gold_id,
        "symbol": data["symbol"],
        "timeframe": data.get("timeframe", "15m"),
        "source_repo": data.get("source_repo", "fable-trading"),
        "source_commit": data.get("source_commit"),
        "source_path": data["source_path"],
        "candidate_source": data.get("candidate_source", "unknown"),
        "decision_bar": int(data["decision_bar"]),
        "decision_time": data["decision_time"],
        "context_start_bar": data.get("context_start_bar"),
        "context_end_bar": data.get("context_end_bar"),
        "local_start_bar": local_start,
        "local_end_bar": int(data["local_end_bar"]),
        "local_window_length": n_bars,
        "shape_label": shape,
        "core_start_bar": None,
        "core_end_bar": None,
        "box_rule": "full_wicks_plus_six_ma",
        "box_status": "none",
        "reviewer": "owner",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "notes": _notes(results),
        "context_image_sha256": data.get("context_image_sha256"),
        "local_image_sha256": data.get("local_image_sha256"),
        "task_config_sha256": data.get("task_config_sha256"),
        "holdout_read": False,
    }
    if shape == "POSITIVE":
        if "START" not in pts or "END" not in pts:
            raise ValueError(f"{gold_id}: POSITIVE missing START/END")
        start_i, end_i = pts["START"], pts["END"]
        if end_i < start_i:
            start_i, end_i = end_i, start_i
        row["core_start_bar"] = local_start + start_i
        row["core_end_bar"] = local_start + end_i
        row["box_status"] = "owner_bar_range"
    validate_gold(row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--existing", type=Path, default=None)
    args = parser.parse_args()
    existing: dict[str, dict] = {}
    if args.existing and args.existing.exists():
        for row in parse_jsonl(args.existing.read_text(encoding="utf-8")):
            existing[row["gold_id"]] = row
    payload = json.loads(args.export.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        tasks = payload.get("tasks") or payload.get("annotations") or [payload]
    else:
        tasks = payload
    written = list(existing.values())
    seen = set(existing)
    for task in tasks:
        row = convert_task(task, existing)
        if row is None or row["gold_id"] in seen:
            continue
        written.append(row)
        seen.add(row["gold_id"])
        existing[row["gold_id"]] = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in written),
        encoding="utf-8",
    )
    print(json.dumps({"n": len(written), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
