#!/usr/bin/env python3
"""Build W10 Gold from Owner-confirmed SHORT boxes + internal empty windows.

Uses owner_short_gold_center_v1 (1,345 unique Owner-confirmed shorts), not the
1,267 one-click suggested cores from the v1 freeze. Long boxes are not read.
Negatives are the nearest empty W10 on the same series, not far easy background.
Does not train. Does not overwrite v1.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from yoyo.datasets.gold_render import load_causal_prefix
from yoyo.datasets.gold_schema import gold_id
from yoyo.datasets.legacy_gold_migration.canonical import base_event, no_signal_slot, w10_fields
from yoyo.datasets.legacy_gold_migration.intervals import (
    center_four_bar,
    nearest_empty_confirmation,
)
from yoyo.datasets.legacy_gold_migration.io import (
    add_common_flags,
    atomic_write_json,
    git_head,
    load_config,
    read_jsonl,
    write_jsonl,
)
from yoyo.datasets.window_render import SourceError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs/fixed_w10_core4_confirm1_v2_owner_short.json"


def _last_pre_holdout_index(path: Path, holdout: pd.Timestamp) -> int:
    times = pd.to_datetime(pd.read_csv(path, usecols=["open_time"])["open_time"], utc=True, errors="coerce")
    mask = times >= holdout
    if not mask.any():
        return int(len(times) - 1)
    return int(mask.to_numpy().nonzero()[0][0]) - 1


def build(cfg: dict) -> dict:
    fable = Path(cfg["paths"]["fable_root"])
    holdout = pd.Timestamp(cfg["holdout_start"])
    warmup = int(cfg["indicator_warmup_bars"])
    guard = int(cfg["owner_guard_bars"])
    radius = int(cfg["internal_neg_radius"])
    fallback = int(cfg["internal_neg_fallback_radius"])
    pos_path = Path(cfg["paths"]["owner_short_positive_manifest"])
    positives = read_jsonl(pos_path)
    builder = git_head(ROOT)
    source_commit = git_head(fable)

    by_csv: dict[str, list[dict]] = defaultdict(list)
    protected_by_csv: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in positives:
        if not row.get("source_owner_gold_confirmed"):
            continue
        csv = row["source_csv"]
        by_csv[csv].append(row)
        lo, hi = (int(x) for x in row["source_owner_global"])
        protected_by_csv[csv].append((lo - guard, hi + guard + 1))

    events: list[dict] = []
    skipped: list[dict] = []
    unmatched_neg = 0

    for csv, rows in sorted(by_csv.items()):
        path = fable / csv
        try:
            last = _last_pre_holdout_index(path, holdout)
            if last < warmup + 10:
                raise SourceError(f"too short after holdout clip: {last}")
            frame = load_causal_prefix(path, last)
        except (SourceError, FileNotFoundError, ValueError) as exc:
            for row in rows:
                skipped.append({"sample_id": row.get("sample_id"), "reason": str(exc)})
            continue
        n = len(frame)
        times = pd.to_datetime(frame["open_time"], utc=True)
        protected = protected_by_csv[csv]
        for row in rows:
            try:
                core_lo, core_hi = (int(x) for x in row["core_global"])
                start, end = center_four_bar(core_lo, core_hi)
                mapped = w10_fields(start, end)
            except ValueError as exc:
                skipped.append({"sample_id": row.get("sample_id"), "reason": str(exc)})
                continue
            if mapped["window_start_bar"] < warmup or mapped["window_end_exclusive_bar"] > n:
                skipped.append({"sample_id": row.get("sample_id"), "reason": "warmup_or_oob"})
                continue
            dec = mapped["decision_bar"]
            dec_time = times.iloc[dec]
            if dec_time >= holdout:
                skipped.append({"sample_id": row.get("sample_id"), "reason": "holdout"})
                continue
            gid = gold_id(row["symbol"], "15m", dec_time.isoformat())
            split = row.get("split") or "train"
            if split not in {"train", "val", "test"}:
                split = "train"
            signal = base_event(
                gold_id=gid,
                event_id=f"{gid}__sig",
                event_group_id=f"{row['symbol']}_15m_{row['sample_id']}",
                symbol=row["symbol"],
                source_commit=source_commit,
                source_dataset=str(pos_path),
                source_record_id=row["sample_id"],
                source_annotation_type="human_gold_owner_box",
                source_interval_semantics="inclusive",
                source_path=csv,
                shape_label="SIGNAL",
                migration_status="OWNER_SHORT_CENTER4",
                owner_confirmation="OWNER_CONFIRMED_SHORT",
                split=split,
                builder_commit=builder,
                decision_time=dec_time.isoformat(),
                negative_type=None,
                negative_sampling_rule=None,
            )
            signal.update(mapped)
            events.append(signal)

            cand = nearest_empty_confirmation(
                decision=dec,
                n=n,
                warmup=warmup,
                protected=protected,
                radius=radius,
            )
            used_radius = radius
            if cand is None:
                cand = nearest_empty_confirmation(
                    decision=dec,
                    n=n,
                    warmup=warmup,
                    protected=protected,
                    radius=fallback,
                )
                used_radius = fallback
            if cand is None:
                unmatched_neg += 1
                continue
            neg_time = times.iloc[cand]
            if neg_time >= holdout:
                unmatched_neg += 1
                continue
            slot = no_signal_slot(cand)
            ngid = gold_id(row["symbol"], "15m", neg_time.isoformat())
            if ngid == gid:
                ngid = f"{ngid}__empty"
            neg = base_event(
                gold_id=ngid,
                event_id=f"{ngid}__neg",
                event_group_id=signal["event_group_id"],
                symbol=row["symbol"],
                source_commit=source_commit,
                source_dataset=str(pos_path),
                source_record_id=f"neg_internal_{row['sample_id']}",
                source_annotation_type="internal_empty_window",
                source_interval_semantics="source_declared",
                source_path=csv,
                shape_label="NO_SIGNAL",
                migration_status="INTERNAL_EMPTY",
                owner_confirmation="NOT_REQUIRED",
                split=split,
                builder_commit=builder,
                decision_time=neg_time.isoformat(),
                negative_type="internal_empty",
                negative_sampling_rule=f"nearest_empty_r{used_radius}_guard{guard}",
            )
            neg.update(slot)
            events.append(neg)

    signal_cores = [
        (e["symbol"], int(e["core_start_bar"]), int(e["core_end_exclusive_bar"]))
        for e in events
        if e["shape_label"] == "SIGNAL" and e.get("core_start_bar") is not None
    ]
    for event in events:
        ws, we = event.get("window_start_bar"), event.get("window_end_exclusive_bar")
        if ws is None or we is None:
            continue
        own = None
        if event["shape_label"] == "SIGNAL":
            own = (event["symbol"], int(event["core_start_bar"]), int(event["core_end_exclusive_bar"]))
        for sym, cs, ce in signal_cores:
            if sym != event["symbol"]:
                continue
            if own is not None and (sym, cs, ce) == own:
                continue
            if ce <= ws or cs >= we:
                continue
            event["contains_other_core"] = True
            break

    events.sort(key=lambda r: (r.get("gold_id") or "", r.get("event_id") or ""))
    return {
        "n_source_positives": len(positives),
        "n_events": len(events),
        "by_shape": dict(Counter(e["shape_label"] for e in events)),
        "by_split": {f"{a}/{b}": n for (a, b), n in Counter((e["split"], e["shape_label"]) for e in events).items()},
        "unmatched_negatives": unmatched_neg,
        "skipped": skipped,
        "contains_other_core": sum(1 for e in events if e.get("contains_other_core")),
        "holdout_rows": 0,
        "events": events,
        "builder_commit": builder,
        "source_commit": source_commit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_flags(parser, default_config=DEFAULT_CFG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = build(cfg)
    root = Path(cfg["paths"]["dataset_root"])
    gold = root / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    usable = [e for e in result["events"] if not e.get("contains_other_core")]
    if not args.dry_run:
        write_jsonl(gold / "events.jsonl", usable)
        write_jsonl(gold / "all_events.jsonl", result["events"])
        write_jsonl(
            gold / "exclusions.jsonl",
            [{"gold_id": e["gold_id"], "reason": "contains_other_core"} for e in result["events"] if e.get("contains_other_core")],
        )
        summary = {k: v for k, v in result.items() if k != "events"}
        summary["n_usable"] = len(usable)
        summary["n_usable_by_shape"] = dict(Counter(e["shape_label"] for e in usable))
        atomic_write_json(gold / "build_summary.json", summary)
        yoyo_rep = ROOT / "reports" / "fixed_w10_core4_confirm1_v2_owner_short"
        yoyo_rep.mkdir(parents=True, exist_ok=True)
        (yoyo_rep / "gold_build.md").write_text(
            "\n".join(
                [
                    "# Owner-short W10 Gold rebuild",
                    "",
                    "正例 = Owner 确认做空金标中心 4 根，不是今晚一键核。",
                    "负例 = 同币同图最近空窗。未覆盖 v1。未训练。",
                    "",
                    "```json",
                    json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "n_events": result["n_events"],
                "by_shape": result["by_shape"],
                "unmatched_negatives": result["unmatched_negatives"],
                "skipped": len(result["skipped"]),
                "contains_other_core": result["contains_other_core"],
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
