#!/usr/bin/env python3
"""Same-contract canary table for V5 quick screen. Not a promotion verdict.

Reads scan_summary.json (+ events.csv counts) from each model dir. Refuses to
compare if protocol / window / conf / NMS / exposures / scope differ.
Does not open klines or holdout.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

CONTRACT = (
    "protocol",
    "latest_bar",
    "replay_start_exclusive",
    "hours",
    "window_lengths",
    "confidence",
    "nms_iou",
    "event_gap_bars",
    "bar_endpoints",
    "window_exposures",
    "evaluation_scope",
    "holdout_use_number",
    "symbols",
)


def load_scan(path: Path) -> dict:
    summary = json.loads((path / "scan_summary.json").read_text(encoding="utf-8"))
    events = path / "events.csv"
    n_events = None
    if events.exists():
        n_events = max(sum(1 for _ in events.open()) - 1, 0)
    return summary, n_events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "DIR"),
        required=True,
        help="repeatable: --model R1 dir --model R3A dir",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ref", default="R1", help="denominator for ratios")
    args = parser.parse_args()

    rows = []
    contract = None
    for name, raw in args.model:
        directory = Path(raw)
        summary, n_events = load_scan(directory)
        got = {k: summary.get(k) for k in CONTRACT}
        if contract is None:
            contract = got
        elif got != contract:
            raise SystemExit(f"contract mismatch for {name}: {got} vs {contract}")
        if summary.get("evaluation_scope") != "preholdout_postval_canary":
            raise SystemExit(f"{name} is not a pre-holdout canary")
        if int(summary.get("holdout_use_number", -1)) != 0:
            raise SystemExit(f"{name} reports a holdout read")
        raw_n = int(summary["raw_detections"])
        ev = int(summary["deduplicated_events"])
        if n_events is not None and n_events != ev:
            raise SystemExit(f"{name} events.csv={n_events} summary={ev}")
        exposures = int(summary["window_exposures"])
        hours = float(summary["hours"])
        rows.append(
            {
                "model": name,
                "dir": str(directory),
                "weights_sha256": summary.get("weights_sha256"),
                "raw_detections": raw_n,
                "deduplicated_events": ev,
                "events_per_day": ev / hours * 24,
                "raw_per_1k_exp": raw_n / exposures * 1000,
                "events_per_1k_exp": ev / exposures * 1000,
            }
        )

    ref = next((r for r in rows if r["model"] == args.ref), rows[0])
    for row in rows:
        if row["model"] == ref["model"]:
            row["events_ratio_vs_ref"] = 1.0
            continue
        row["events_ratio_vs_ref"] = row["deduplicated_events"] / ref["deduplicated_events"]
        row["raw_ratio_vs_ref"] = row["raw_detections"] / ref["raw_detections"]

    payload = {
        "schema": "yoyo_canary_compare_v1",
        "per_v5": True,
        "protocol": contract["protocol"],
        "block": "am_20260503_00_12UTC",
        "evaluation_scope": "preholdout_postval_canary",
        "holdout_use_number": 0,
        "promoted": False,
        "matched_contract": contract,
        "feature_stack": "SMA+EMA 20/60/120",
        "models": rows,
        "ref": args.ref,
        "note": (
            "Density alone is NOT success. V5 wants lower FP at similar recall. "
            "Shape-YES audit and Gold Core event recall still required."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
