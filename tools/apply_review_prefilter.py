#!/usr/bin/env python3
"""Owner pre-filter on the frozen V3 review pack.

Rule: decision close > mean(close) of the purple-band bars (after decision,
same 24-bar window as the context chart, holdout-clipped).

This uses future. It is a review convenience / short-side screen, not an L1
label rule. Results are written beside the pack and onto public items.json
so the gallery can hide failing cards.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from yoyo.datasets.window_render import load_prefix

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "reviews/v3_blind_r1"
SOURCE = json.loads((ROOT / "configs/source_repo.json").read_text())
FABLE = Path(SOURCE["source_repo"])
HOLDOUT = pd.Timestamp(SOURCE["holdout_start"])
FUTURE_BARS = 24
MANIFEST = {
    json.loads(line)["sample_id"]: json.loads(line)
    for line in (ROOT / "datasets/dataset_v3_gold_core_v1/manifest.jsonl").read_text().splitlines()
    if line
}


def measure(key: dict) -> dict:
    sample = MANIFEST[key["sample_id"]]
    decision = int(sample["decision_bar"])
    frame = load_prefix(FABLE / sample["source_path"], decision + FUTURE_BARS)
    times = pd.to_datetime(frame["open_time"], utc=True)
    frame = frame.loc[times < HOLDOUT].reset_index(drop=True)
    if decision >= len(frame):
        return {
            "review_id": key["review_id"],
            "sample_id": key["sample_id"],
            "ok": False,
            "reason": "decision_clipped",
        }
    future = frame.iloc[decision + 1 : decision + 1 + FUTURE_BARS]
    if len(future) == 0:
        return {
            "review_id": key["review_id"],
            "sample_id": key["sample_id"],
            "ok": False,
            "reason": "no_future_bars",
        }
    decision_close = float(frame.iloc[decision]["close"])
    future_mean = float(future["close"].mean())
    return {
        "review_id": key["review_id"],
        "sample_id": key["sample_id"],
        "ok": True,
        "decision_close": decision_close,
        "purple_mean_close": future_mean,
        "delta": decision_close - future_mean,
        "rel_delta": (decision_close - future_mean) / decision_close if decision_close else None,
        "future_bars": int(len(future)),
        "pass": decision_close > future_mean,
        "rule": "decision_close > mean(purple_close)",
    }


def main() -> int:
    keys = [
        json.loads(line)
        for line in (PACK / "answer_key.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = [measure(key) for key in keys]
    passed = [row for row in rows if row.get("pass")]
    failed = [row for row in rows if row.get("ok") and not row.get("pass")]
    broken = [row for row in rows if not row.get("ok")]
    receipt = {
        "protocol": "yoyo_v3_review_prefilter_decision_gt_purple_mean_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rule": "decision_close > mean(close of next 24 bars, holdout-clipped)",
        "uses_future": True,
        "l1_label_rule": False,
        "n": len(rows),
        "pass": len(passed),
        "fail": len(failed),
        "broken": len(broken),
        "pass_rate": len(passed) / len(rows) if rows else None,
    }
    (PACK / "prefilter_decision_gt_purple_mean.json").write_text(
        json.dumps({"summary": receipt, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    items_path = PACK / "public" / "items.json"
    items = json.loads(items_path.read_text(encoding="utf-8"))
    by_id = {row["review_id"]: row for row in rows}
    for item in items["items"]:
        row = by_id[item["review_id"]]
        item["prefilter_pass"] = bool(row.get("pass"))
        item["decision_close"] = row.get("decision_close")
        item["purple_mean_close"] = row.get("purple_mean_close")
        item["prefilter_delta"] = row.get("delta")
    items["prefilter"] = receipt
    items_path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
