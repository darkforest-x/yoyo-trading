#!/usr/bin/env python3
"""Write the V3.1 crop-assignment 清单 without rendering images.

Single variable vs V3 Gold Core: the causal window length in 12–19, still
ending at decision. No future K. Same boxes and splits.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from yoyo.datasets.position_spread import assign_all, placements

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "datasets/dataset_v3_gold_core_v1/manifest.jsonl"
DEFAULT_OUT = ROOT / "manifests/dataset_v3_1_position_spread_plan.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-manifest", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.in_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    positives = [r for r in rows if r.get("class_name")]
    negatives = [r for r in rows if not r.get("class_name")]
    assigned = assign_all(positives)

    achievable = Counter()
    for row in positives:
        bins = {
            o["bin"]
            for o in placements(int(row["box_start_bar"]), int(row["box_end_bar"]), int(row["decision_bar"]))
        }
        for b in bins:
            achievable[f"can_{b}"] += 1
        if not bins:
            achievable["can_none"] += 1

    chosen = Counter(r.get("chosen_bin") for r in assigned if r.get("spread_status") == "ok")
    unchanged = sum(
        1
        for r in assigned
        if r.get("spread_status") == "ok" and r.get("chosen_window_len") == r.get("window_length")
    )

    plan = {
        "protocol": "yoyo_dataset_v3_1_position_spread_plan_20260812",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parent_dataset": "dataset_v3_gold_core_v1",
        "single_variable": "causal_window_length_in_w12_19_ending_at_decision",
        "feature_stack": "SMA20/60/120 + EMA20/60/120",
        "holdout_read": False,
        "counts": {
            "parent_rows": len(rows),
            "positives": len(positives),
            "negatives": len(negatives),
            "assigned_ok": sum(1 for r in assigned if r["spread_status"] == "ok"),
            "no_legal_window": sum(1 for r in assigned if r["spread_status"] != "ok"),
            "window_len_changed": sum(
                1
                for r in assigned
                if r.get("spread_status") == "ok" and r.get("chosen_window_len") != r.get("window_length")
            ),
            "window_len_unchanged": unchanged,
        },
        "achievable_if_we_could_pick_freely": dict(sorted(achievable.items())),
        "greedy_assignment": dict(sorted(chosen.items())),
        "geometry_limit": (
            "A core only a few bars before decision cannot enter the left third "
            "at W>=12 if the window must end at decision. Left count is therefore "
            "bounded by how many gold cores sit far enough before decision."
        ),
        "assignments_path": "manifests/dataset_v3_1_position_spread_assignments.jsonl",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    assign_path = ROOT / "manifests/dataset_v3_1_position_spread_assignments.jsonl"
    slim = []
    for r in assigned:
        slim.append(
            {
                "sample_id": r["sample_id"],
                "symbol": r["symbol"],
                "split": r["split"],
                "decision_bar": r["decision_bar"],
                "box_start_bar": r["box_start_bar"],
                "box_end_bar": r["box_end_bar"],
                "old_window_length": r["window_length"],
                "old_window_start_bar": r["window_start_bar"],
                "spread_status": r["spread_status"],
                "chosen_window_len": r.get("chosen_window_len"),
                "chosen_win_start": r.get("chosen_win_start"),
                "chosen_win_end": r.get("chosen_win_end"),
                "chosen_rel": r.get("chosen_rel"),
                "chosen_bin": r.get("chosen_bin"),
                "achievable_bins": r.get("achievable_bins"),
            }
        )
    assign_path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in slim), encoding="utf-8"
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {assign_path} ({len(slim)} positives)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
