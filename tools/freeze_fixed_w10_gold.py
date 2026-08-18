#!/usr/bin/env python3
"""Freeze Owner-confirmed + DIRECT events into gold/*.jsonl. Does not train.

Known freeze-time pits (do not rewrite review state.jsonl):
- CONFLICT stays out of gold.
- Accepted A with no 4-bar core is not frozen as SIGNAL. The three
  Owner-corrected EASY_BACKGROUND rows stay NO_SIGNAL; other coreless A
  become NO_SIGNAL at freeze (queue label was already NO_SIGNAL).
- Competing cores are recomputed on the frozen 4-bar geometry.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from yoyo.datasets.legacy_gold_migration.canonical import no_signal_slot, w10_fields
from yoyo.datasets.legacy_gold_migration.io import add_common_flags, load_config, read_jsonl, write_jsonl
from yoyo.datasets.legacy_gold_migration.review_queue import load_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs/fixed_w10_core4_confirm1_v1.json"

# Batch-accept corrected these from mistaken A back to N; later they reappeared
# as coreless A. Freeze keeps them as negatives without touching state.
CORRECTED_EASY_BACKGROUND_N = frozenset(
    {
        "2Z_USDT_SWAP_15m_20251017T170000Z",
        "ACH_USDT_SWAP_15m_20251113T133000Z",
        "ADA_USDT_SWAP_15m_20251015T151500Z",
    }
)


def _four_bar(row: dict) -> bool:
    start, end = row.get("core_start_bar"), row.get("core_end_exclusive_bar")
    return start is not None and end is not None and int(end) - int(start) == 4


def _as_no_signal(row: dict) -> dict:
    out = dict(row)
    out["shape_label"] = "NO_SIGNAL"
    out["core_start_bar"] = None
    out["core_end_exclusive_bar"] = None
    out["core_length"] = None
    conf = int(out.get("confirmation_bar") or out["decision_bar"])
    out.update(no_signal_slot(conf))
    return out


def _window_bounds(row: dict) -> tuple[int, int] | None:
    shape = row.get("shape_label")
    if shape == "SIGNAL" and _four_bar(row):
        mapped = w10_fields(int(row["core_start_bar"]), int(row["core_end_exclusive_bar"]))
        return mapped["window_start_bar"], mapped["window_end_exclusive_bar"]
    if shape == "NO_SIGNAL":
        conf = row.get("confirmation_bar") or row.get("decision_bar")
        if conf is None:
            return None
        mapped = no_signal_slot(int(conf))
        return mapped["window_start_bar"], mapped["window_end_exclusive_bar"]
    return None


def _overlaps_other_core(row: dict, cores: list[tuple[str, int, int]]) -> bool:
    bounds = _window_bounds(row)
    if bounds is None:
        return False
    ws, we = bounds
    own = None
    if row.get("shape_label") == "SIGNAL" and _four_bar(row):
        own = (row["symbol"], int(row["core_start_bar"]), int(row["core_end_exclusive_bar"]))
    for sym, cs, ce in cores:
        if sym != row.get("symbol"):
            continue
        if own is not None and (sym, cs, ce) == own:
            continue
        if ce <= ws or cs >= we:
            continue
        return True
    return False


def _finalize(row: dict) -> dict:
    out = dict(row)
    if out.get("shape_label") == "SIGNAL" and out.get("core_start_bar") is not None:
        mapped = w10_fields(int(out["core_start_bar"]), int(out["core_start_bar"]) + 4)
        out.update(mapped)
        out["core_end_exclusive_bar"] = mapped["core_end_exclusive_bar"]
    elif out.get("shape_label") == "NO_SIGNAL":
        conf = out.get("confirmation_bar") or out.get("decision_bar")
        if conf is not None:
            out.update(no_signal_slot(int(conf)))
            out["core_start_bar"] = None
            out["core_end_exclusive_bar"] = None
            out["core_length"] = None
    out["owner_confirmation"] = out.get("owner_confirmation") or "MIGRATED"
    out["contains_other_core"] = bool(out.get("contains_other_core"))
    return out


def freeze_rows(events: list[dict], state: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Return (frozen gold rows, exclusions). Does not write disk."""
    paths_by_symbol: dict[str, set[str]] = defaultdict(set)
    for row in events:
        if row.get("symbol") and row.get("source_path"):
            paths_by_symbol[row["symbol"]].add(row["source_path"])

    frozen: list[dict] = []
    excluded: list[dict] = []

    for row in events:
        gid = row.get("gold_id")
        if gid in state:
            rec = dict(row)
            rec.update(state[gid])
            if row.get("migration_status") == "CONFLICT":
                excluded.append({"gold_id": gid, "reason": "conflict", "shape_label": rec.get("shape_label")})
                continue
            if rec.get("shape_label") == "IGNORE":
                rec["migration_status"] = "IGNORE"
                frozen.append(_finalize(rec))
                continue
            if gid in CORRECTED_EASY_BACKGROUND_N:
                rec = _as_no_signal(rec)
                rec["negative_type"] = rec.get("negative_type") or "EASY_BACKGROUND"
                rec["freeze_note"] = "corrected_easy_background_kept_negative"
            elif rec.get("shape_label") == "SIGNAL" and not _four_bar(rec):
                rec = _as_no_signal(rec)
                rec["freeze_note"] = "coreless_accepted_a_not_signal"
            if rec.get("shape_label") == "SIGNAL" and not rec.get("source_path"):
                paths = paths_by_symbol.get(rec.get("symbol") or "", set())
                if len(paths) == 1:
                    rec["source_path"] = next(iter(paths))
                    rec["freeze_note"] = "source_path_recovered_from_symbol"
            frozen.append(_finalize(rec))
            continue
        if row.get("migration_status") == "DIRECT" and row.get("shape_label") in {"SIGNAL", "NO_SIGNAL"}:
            if row.get("contains_other_core"):
                excluded.append({"gold_id": gid, "reason": "contains_other_core", "shape_label": row.get("shape_label")})
                continue
            if row.get("migration_status") == "CONFLICT":
                continue
            frozen.append(_finalize(row))

    cores = [
        (r["symbol"], int(r["core_start_bar"]), int(r["core_end_exclusive_bar"]))
        for r in frozen
        if r.get("shape_label") == "SIGNAL" and _four_bar(r)
    ]
    kept: list[dict] = []
    for rec in frozen:
        if rec.get("shape_label") not in {"SIGNAL", "NO_SIGNAL"}:
            kept.append(rec)
            continue
        if _overlaps_other_core(rec, cores):
            rec["contains_other_core"] = True
            excluded.append(
                {
                    "gold_id": rec.get("gold_id"),
                    "reason": "contains_other_core",
                    "shape_label": rec.get("shape_label"),
                }
            )
            continue
        rec["contains_other_core"] = False
        kept.append(rec)
    return kept, excluded


def main() -> int:
    import argparse
    import shutil
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_flags(parser, default_config=DEFAULT_CFG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["dataset_root"])
    events = read_jsonl(root / "migration" / "migrated_candidates.jsonl")
    state_path = root / "review" / "state.jsonl"
    state = load_state(state_path)
    frozen, excluded = freeze_rows(events, state)
    if args.limit:
        frozen = frozen[: args.limit]
    gold_dir = root / "gold"
    summary = {
        "n": len(frozen),
        "signal": sum(r.get("shape_label") == "SIGNAL" for r in frozen),
        "no_signal": sum(r.get("shape_label") == "NO_SIGNAL" for r in frozen),
        "ignore": sum(r.get("shape_label") == "IGNORE" for r in frozen),
        "excluded": len(excluded),
        "excluded_by_reason": {},
        "out": str(gold_dir),
    }
    by_reason: dict[str, int] = {}
    for row in excluded:
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
    summary["excluded_by_reason"] = by_reason
    if args.dry_run:
        print(json.dumps({**summary, "dry_run": True}, ensure_ascii=False))
        return 0
    gold_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    shutil.copy2(state_path, gold_dir / f"review_state_at_freeze_{stamp}.jsonl")
    write_jsonl(gold_dir / "events.jsonl", frozen)
    write_jsonl(gold_dir / "signal.jsonl", [r for r in frozen if r.get("shape_label") == "SIGNAL"])
    write_jsonl(gold_dir / "no_signal.jsonl", [r for r in frozen if r.get("shape_label") == "NO_SIGNAL"])
    write_jsonl(gold_dir / "ignored.jsonl", [r for r in frozen if r.get("shape_label") == "IGNORE"])
    write_jsonl(gold_dir / "exclusions.jsonl", excluded)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
