#!/usr/bin/env python3
"""Build the Dataset V3 migration table from every legacy label we hold.

Covers the four legacy populations the V5 brief names: gold-derived positives,
rule-sampled easy negatives, hard negatives (Owner-long mirrors and model-mined
backgrounds), and the six Owner review packs. Each row gets one migration
verdict plus separate class and box status, so "the Owner said yes" can never
be read as "the box is gold".

Also answers the brief's shape-vs-outcome question the only way the artifacts
allow: every pack showed a future panel next to the causal chart, so no verdict
is purely causal. What can be measured is whether a verdict *agrees* with what
the market then did -- agreement leaves the motive ambiguous, contradiction is
evidence the Owner judged the shape.

Holdout discipline: bars at or after the holdout boundary are dropped on read,
so the forward-return diagnostic can never see them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from yoyo.datasets.legacy_audit import (
    classify_easy_negative,
    classify_hard_negative,
    classify_positive,
    classify_review_event,
    intersects_owner_gold_box,
    outcome_alignment,
    owner_box_spans,
    parse_time,
    tally,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CONFIG = ROOT / "configs/source_repo.json"
DEFAULT_OUT = ROOT / "manifests/legacy_label_migration_v3.json"
DEFAULT_ROWS_OUT = ROOT / "manifests/legacy_label_migration_v3.jsonl"

DATASET = "datasets/owner_short_gold_center_hardneg_r1"
GOLD_SHEET = "analysis/output/owner_side_review/review_sheet.csv"
BAR = timedelta(minutes=15)
FORWARD_BARS = 16

# review_id prefix -> (path, verdict field, verdict mapping)
PACKS: dict[str, dict[str, Any]] = {
    "hardneg200_v1": {
        "path": "analysis/output/owner_short_train_hardneg_review200_v1/owner_review_labeled_manifest.jsonl",
        "verdict_field": "owner_decision",
        "map": {"target": "YES", "hard_negative": "NO", "rebox": "REBOX"},
    },
    "positive100_v1": {
        "path": "analysis/output/owner_short_train_positive_retrieval100_v1/owner_review_labeled_manifest.jsonl",
        "verdict_field": "owner_decision",
        "map": {"target": "YES", "hard_negative": "NO", "rebox": "REBOX"},
    },
    "expansion200_v2": {
        "path": "analysis/output/owner_short_train_hardneg_expansion200_v2/owner_review_labeled_manifest.jsonl",
        "verdict_field": "owner_decision",
        "map": {"target": "YES", "hard_negative": "NO", "rebox": "REBOX"},
    },
    "newblocks200_v3": {
        "path": "analysis/output/owner_short_train_hardneg_newblocks200_v3/owner_review_labeled_manifest.jsonl",
        "verdict_field": "owner_decision",
        "map": {"target": "YES", "hard_negative": "NO", "rebox": "REBOX"},
    },
    "semantic200_v2": {
        "path": "analysis/output/local_signal_v2_positive_semantic_review200_v2/owner_review_joined.jsonl",
        "verdict_field": "owner_verdict",
        "map": {"YES": "YES", "NO": "NO", "SKIP": "SKIP"},
    },
    "early300_v1": {
        "path": "analysis/output/local_signal_v2_early_frontier_review300_v1/owner_review_joined.jsonl",
        "verdict_field": "owner_verdict",
        "map": {"YES": "YES", "NO": "NO", "SKIP": "SKIP"},
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_bounds(positives: list[dict[str, Any]]) -> tuple[datetime, datetime]:
    train_end = max(parse_time(row["end_time"]) for row in positives if row["split"] == "train")
    val_start = min(parse_time(row["start_time"]) for row in positives if row["split"] == "val")
    return train_end, val_start


def normalise_pack(name: str, spec: dict[str, Any], repo: Path) -> list[dict[str, Any]]:
    path = repo / spec["path"]
    if not path.exists():
        return []
    rows = []
    for row in read_jsonl(path):
        raw = str(row.get(spec["verdict_field"]))
        rows.append(
            {
                "population": "owner_review_event",
                "pack": name,
                "sample_id": f"{name}:{row.get('review_id')}",
                "event_id": row.get("event_id"),
                "symbol": row["symbol"],
                "decision_time": row["decision_time"],
                "verdict": spec["map"].get(raw, raw),
                "raw_verdict": raw,
                "future_assisted": True,
                "touches_owner_box_guard": bool(row.get("touches_owner_box_guard")),
                "confidence": row.get("conf", row.get("model_confidence")),
                "window_len": row.get("window_len", row.get("window_length")),
                "core_start_time": row.get("core_start_time"),
                "core_end_time": row.get("core_end_time"),
            }
        )
    return rows


def forward_returns(
    events: list[dict[str, Any]], repo: Path, holdout_start: datetime
) -> dict[str, float | None]:
    """Close-to-close return over FORWARD_BARS after each decision, in bp."""
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_symbol[event["symbol"]].append(event)
    csv_by_symbol: dict[str, Path] = {}
    for path in sorted((repo / "data/kline_fetched").glob("okx_*_15m_*.csv")):
        symbol = path.name.split("_15m_")[0][len("okx_") :]
        csv_by_symbol.setdefault(symbol, path)

    out: dict[str, float | None] = {}
    for symbol, rows in by_symbol.items():
        path = csv_by_symbol.get(symbol)
        if path is None:
            for row in rows:
                out[row["sample_id"]] = None
            continue
        frame = pd.read_csv(path, usecols=["open_time", "close"], encoding_errors="replace")
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["open_time"])
        frame = frame[frame["open_time"] < holdout_start]
        closes = frame.set_index("open_time")["close"].astype(float)
        for row in rows:
            decision = parse_time(row["decision_time"])
            if decision not in closes.index:
                out[row["sample_id"]] = None
                continue
            target = decision + BAR * FORWARD_BARS
            window = closes[(closes.index > decision) & (closes.index <= target)]
            if window.empty:
                out[row["sample_id"]] = None
                continue
            start = float(closes.loc[decision])
            out[row["sample_id"]] = (float(window.iloc[-1]) / start - 1.0) * 10000 if start else None
    return out


def audit(source_config: Path, out_path: Path, rows_path: Path) -> dict[str, Any]:
    source = json.loads(source_config.read_text(encoding="utf-8"))
    repo = Path(source["source_repo"])
    holdout_start = parse_time(source["holdout_start"])
    dataset = repo / DATASET

    positives = read_jsonl(dataset / "positive_manifest.jsonl")
    negatives = read_jsonl(dataset / "negative_manifest.jsonl")
    hard = read_jsonl(dataset / "hard_negative_manifest.jsonl")
    easy = [row for row in negatives if str(row.get("class")) == "easy_negative"]
    train_end, val_start = split_bounds(positives)

    events: list[dict[str, Any]] = []
    for name, spec in PACKS.items():
        events.extend(normalise_pack(name, spec, repo))
    owner_no_events = {
        str(event["event_id"]) for event in events if event["verdict"] == "NO" and event["event_id"]
    }

    returns = forward_returns(events, repo, holdout_start)

    rows: list[dict[str, Any]] = []
    for row in positives:
        rows.append({**{"population": "legacy_positive", "sample_id": row["sample_id"],
                        "symbol": row["symbol"], "split": row["split"],
                        "win_len": row.get("win_len")}, **classify_positive(row)})
    for row in easy:
        rows.append({**{"population": "legacy_easy_negative", "sample_id": row["sample_id"],
                        "symbol": row["symbol"], "split": row["split"],
                        "win_len": row.get("win_len")}, **classify_easy_negative(row)})
    for row in hard:
        rows.append({**{"population": "legacy_hard_negative", "sample_id": row["sample_id"],
                        "symbol": row["symbol"], "split": row["split"],
                        "win_len": row.get("win_len"), "candidate_kind": row.get("candidate_kind")},
                     **classify_hard_negative(row, owner_no_events)})
    spans = owner_box_spans(positives)
    for event in events:
        window_end = parse_time(event["decision_time"])
        window_start = window_end - BAR * (int(event.get("window_len") or 12) - 1)
        event["intersects_owner_gold_box"] = intersects_owner_gold_box(
            str(event["symbol"]), window_start, window_end, spans
        )
        verdict_row = {**event}
        verdict_row.update(classify_review_event(event, train_end, val_start))
        forward = returns.get(event["sample_id"])
        verdict_row["forward_return_bp_16bar"] = forward
        verdict_row["outcome_alignment"] = outcome_alignment(event["verdict"], forward)
        rows.append(verdict_row)

    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )

    review_rows = [row for row in rows if row["population"] == "owner_review_event"]
    duplicate_events = {
        event_id: count
        for event_id, count in Counter(
            str(row["event_id"]) for row in review_rows if row.get("event_id")
        ).items()
        if count > 1
    }
    conflicting = {
        event_id
        for event_id, verdicts in (
            (event_id, {row["verdict"] for row in review_rows if str(row["event_id"]) == event_id})
            for event_id in duplicate_events
        )
        if len(verdicts) > 1
    }

    alignment: dict[str, dict[str, int]] = {}
    for verdict in ("YES", "NO"):
        subset = [row for row in review_rows if row["verdict"] == verdict]
        alignment[verdict] = tally(subset, "outcome_alignment")

    summary = {
        "protocol": "yoyo_l1_legacy_label_migration_v3_20260812",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(repo),
        "dataset": DATASET,
        "holdout_start": source["holdout_start"],
        "holdout_rows_used": 0,
        "split_bounds": {"train_end": train_end.isoformat(), "val_start": val_start.isoformat()},
        "populations": {
            "legacy_positive": len(positives),
            "legacy_easy_negative": len(easy),
            "legacy_hard_negative": len(hard),
            "owner_review_event": len(events),
        },
        "migration_table": tally(rows, "migration"),
        "by_population": {
            population: tally([row for row in rows if row["population"] == population], "migration")
            for population in ("legacy_positive", "legacy_easy_negative", "legacy_hard_negative", "owner_review_event")
        },
        "class_status": tally(rows, "class_status"),
        "box_status": tally(rows, "box_status"),
        "hard_negative_kinds": tally(
            [row for row in rows if row["population"] == "legacy_hard_negative"], "candidate_kind"
        ),
        "review_verdicts": tally(review_rows, "verdict"),
        "review_by_pack": {
            pack: tally([row for row in review_rows if row["pack"] == pack], "verdict")
            for pack in PACKS
        },
        "usable_owner_shape_no": sum(
            1 for row in review_rows if row["migration"] == "KEEP_NEGATIVE"
        ),
        "owner_yes_awaiting_box_review": sum(
            1 for row in review_rows if row["migration"] == "REBOX_REQUIRED"
        ),
        "events_overlapping_owner_gold_box": sum(
            1 for row in review_rows if row.get("intersects_owner_gold_box")
        ),
        "pack_guard_flag_true": sum(
            1 for row in review_rows if row.get("touches_owner_box_guard")
        ),
        "duplicate_event_ids_across_packs": len(duplicate_events),
        "conflicting_duplicate_verdicts": sorted(conflicting),
        "outcome_alignment": alignment,
        "forward_return_horizon_bars": FORWARD_BARS,
        "all_verdicts_future_assisted": True,
        "gold_sheet_sha256": sha256_file(repo / GOLD_SHEET),
        "rows_path": str(rows_path.relative_to(ROOT)),
        "rows_sha256": sha256_file(rows_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rows-out", type=Path, default=DEFAULT_ROWS_OUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.source_config, args.out, args.rows_out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
