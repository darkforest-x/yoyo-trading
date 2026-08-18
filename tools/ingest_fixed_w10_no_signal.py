#!/usr/bin/env python3
"""One-shot: ingest reused NO_SIGNAL rows into review state. Does not freeze.

Owner 2026-08-13 authorized filling the missing negatives so freeze can have
both classes. Do not relabel. Sources, in order:

1. Legacy easy-negative pool (rule-sampled background; gold_eligible was False).
2. Migrated candidates already labeled NO_SIGNAL that Owner confirmed have no
   dense core (ONE_CLICK_REVIEW from review packs / V3.2).
3. Keep the existing N rows.

Does not convert SIGNAL, CONFLICT, or accepted A rows. Does not train.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from yoyo.datasets.legacy_gold_migration.canonical import is_holdout
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


def _empty_core(row: dict) -> bool:
    return row.get("core_start_bar") is None and row.get("core_end_exclusive_bar") is None


def _bucket(row: dict) -> str | None:
    if row.get("source_annotation_type") == "easy_negative_pool":
        return "easy_pool"
    if row.get("migration_status") == "ONE_CLICK_REVIEW":
        return "owner_one_click"
    return None


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_flags(parser, default_config=DEFAULT_CFG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["dataset_root"])
    state_path = root / "review" / "state.jsonl"
    history_dir = root / "review" / "history"
    holdout = pd.Timestamp(cfg["holdout_start"])
    warmup = int(cfg["indicator_warmup_bars"])
    events = read_jsonl(root / "migration" / "migrated_candidates.jsonl")
    state = load_state(state_path)

    a_ids = {gid for gid, row in state.items() if row.get("review_action") == "A"}
    n_ids = {gid for gid, row in state.items() if row.get("review_action") == "N"}
    skip: Counter[str] = Counter()
    added: list[dict] = []
    by_bucket: Counter[str] = Counter()

    for row in events:
        gid = row.get("gold_id")
        if not gid:
            skip["no_gold_id"] += 1
            continue
        if gid in n_ids:
            skip["keep_existing_n"] += 1
            continue
        if gid in a_ids:
            skip["keep_existing_a"] += 1
            continue
        if row.get("migration_status") == "CONFLICT":
            skip["conflict"] += 1
            continue
        if is_holdout(row.get("decision_time"), holdout):
            skip["holdout"] += 1
            continue
        if row.get("shape_label") != "NO_SIGNAL":
            skip["not_no_signal"] += 1
            continue
        if row.get("contains_other_core"):
            skip["contains_other_core"] += 1
            continue
        if not row.get("source_path"):
            skip["no_source_path"] += 1
            continue
        if row.get("decision_bar") is None:
            skip["no_decision_bar"] += 1
            continue
        if int(row.get("window_start_bar") or -1) < warmup:
            skip["warmup"] += 1
            continue
        bucket = _bucket(row)
        if bucket is None:
            skip["not_easy_or_owner_confirmed"] += 1
            continue
        saved = apply_action(row, "N", None)
        if saved.get("shape_label") != "NO_SIGNAL" or saved.get("review_action") != "N":
            raise RuntimeError(f"{gid} ingest did not write NO_SIGNAL")
        if not _empty_core(saved):
            raise RuntimeError(f"{gid} ingest still has a core")
        if saved.get("shape_label") == "SIGNAL":
            raise RuntimeError(f"{gid} ingest wrote SIGNAL")
        state[gid] = saved
        added.append(saved)
        by_bucket[bucket] += 1

    n_state = list(state.values())
    n_a = sum(r.get("review_action") == "A" for r in n_state)
    n_n = sum(r.get("review_action") == "N" for r in n_state)
    summary = {
        "ingested_new": len(added),
        "by_bucket": dict(by_bucket),
        "kept_existing_a": len(a_ids),
        "kept_existing_n": len(n_ids),
        "skipped": dict(skip),
        "n_state_after": len(state),
        "n_a_after": n_a,
        "n_n_after": n_n,
        "dry_run": bool(args.dry_run),
        "froze": False,
        "gold_v1_touched": False,
        "render_py_touched": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if n_a != len(a_ids):
        raise RuntimeError("accepted A count changed")
    if n_n != len(n_ids) + len(added):
        raise RuntimeError("N count mismatch")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = state_path.with_name(f"state.jsonl.bak.{stamp}")
    if state_path.exists():
        shutil.copy2(state_path, backup)
        summary["backup"] = str(backup)
    history_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(state_path, [state[k] for k in sorted(state)])
    hist = history_dir / f"{stamp}_ingest_no_signal_batch.json"
    atomic_write_text(
        hist,
        json.dumps(
            {
                **summary,
                "gold_ids": [r["gold_id"] for r in added],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    print(json.dumps({"backup": str(backup), "wrote": str(state_path), "history": str(hist)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
