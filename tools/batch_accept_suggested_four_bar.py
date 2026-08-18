#!/usr/bin/env python3
"""One-shot: accept suggest_four_bar cores into review state. Does not freeze."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from yoyo.datasets.legacy_gold_migration.canonical import w10_fields
from yoyo.datasets.legacy_gold_migration.io import (
    add_common_flags,
    atomic_write_text,
    load_config,
    read_jsonl,
    write_jsonl,
)
from yoyo.datasets.legacy_gold_migration.review_queue import apply_action, load_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs/fixed_w10_core4_confirm1_v1.json"

# Mis-accepted EASY_BACKGROUND negatives: Owner A'd them as coreless SIGNAL.
WRONG_SIGNAL_NEGATIVES = frozenset(
    {
        "2Z_USDT_SWAP_15m_20251017T170000Z",
        "ACH_USDT_SWAP_15m_20251113T133000Z",
        "ADA_USDT_SWAP_15m_20251015T151500Z",
    }
)


def _four_bar(start, end) -> bool:
    return start is not None and end is not None and int(end) - int(start) == 4


def _keep_existing_four_bar(existing: dict | None) -> bool:
    if not existing:
        return False
    if existing.get("review_action") != "A":
        return False
    return _four_bar(existing.get("core_start_bar"), existing.get("core_end_exclusive_bar"))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_flags(parser, default_config=DEFAULT_CFG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["dataset_root"])
    state_path = root / "review" / "state.jsonl"
    history_dir = root / "review" / "history"
    queue = read_jsonl(root / "review" / "queue.jsonl")
    state = load_state(state_path)

    n_accept = 0
    n_keep = 0
    n_correct = 0
    n_skip_conflict = 0
    n_skip_not_signal = 0
    n_skip_no_suggestion = 0
    skipped_conflicts: list[str] = []
    changed: list[dict] = []

    for row in queue:
        gid = row.get("gold_id")
        if not gid:
            continue

        if gid in WRONG_SIGNAL_NEGATIVES:
            saved = apply_action(row, "N", None)
            if saved.get("shape_label") == "SIGNAL":
                raise RuntimeError(f"{gid} correction still SIGNAL")
            state[gid] = saved
            changed.append(saved)
            n_correct += 1
            continue

        if _keep_existing_four_bar(state.get(gid)):
            n_keep += 1
            continue

        if row.get("migration_status") == "CONFLICT" or row.get("queue_status") == "CONFLICT":
            n_skip_conflict += 1
            skipped_conflicts.append(gid)
            continue

        if row.get("shape_label") != "SIGNAL":
            n_skip_not_signal += 1
            continue

        sug_s = row.get("suggested_core_start_bar")
        sug_e = row.get("suggested_core_end_exclusive_bar")
        if not _four_bar(sug_s, sug_e):
            n_skip_no_suggestion += 1
            continue

        saved = apply_action(row, "A", int(sug_s))
        saved.update(w10_fields(int(sug_s), int(sug_e)))
        if saved.get("shape_label") != "SIGNAL" or not _four_bar(
            saved.get("core_start_bar"), saved.get("core_end_exclusive_bar")
        ):
            raise RuntimeError(f"{gid} accept did not write a 4-bar core")
        if int(saved["core_length"]) != 4 or int(saved["window_length"]) != 10:
            raise RuntimeError(f"{gid} not on W10/core4 contract")
        if int(saved["confirmation_bar"]) != int(saved["core_end_exclusive_bar"]):
            raise RuntimeError(f"{gid} confirm is not the bar after the core")
        state[gid] = saved
        changed.append(saved)
        n_accept += 1

    summary = {
        "accepted_new": n_accept,
        "kept_existing_four_bar": n_keep,
        "corrected_false_signal": n_correct,
        "corrected_ids": sorted(WRONG_SIGNAL_NEGATIVES),
        "skipped_conflict": n_skip_conflict,
        "skipped_conflict_ids": skipped_conflicts,
        "skipped_not_signal": n_skip_not_signal,
        "skipped_no_suggestion": n_skip_no_suggestion,
        "n_state_after": len(state),
        "n_changed": len(changed),
        "dry_run": bool(args.dry_run),
        "froze": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = state_path.with_name(f"state.jsonl.bak.{stamp}")
    if state_path.exists():
        shutil.copy2(state_path, backup)
        summary["backup"] = str(backup)
    history_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(state_path, [state[k] for k in sorted(state)])
    hist_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for row in changed:
        hist = history_dir / f"{hist_stamp}_{row['gold_id']}.json"
        atomic_write_text(hist, json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"backup": str(backup), "wrote": str(state_path), "history": len(changed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
