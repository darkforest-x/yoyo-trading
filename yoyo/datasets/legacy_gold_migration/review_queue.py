"""Review queue + atomic Owner save for the W10 migration page."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import atomic_write_text, jsonl_dumps, read_jsonl, write_jsonl

REVIEW_STATUSES = frozenset({"ONE_CLICK_REVIEW", "MANUAL_ADJUST", "CONFLICT", "DIRECT_SPOT_CHECK"})


def build_queue(events: list[dict[str, Any]], spot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in list(events) + list(spot):
        gid = row.get("gold_id")
        if not gid or gid in seen:
            continue
        status = row.get("queue_status") or row.get("migration_status")
        if status not in REVIEW_STATUSES and row not in spot:
            if row.get("migration_status") not in {"ONE_CLICK_REVIEW", "MANUAL_ADJUST", "CONFLICT"}:
                continue
        seen.add(gid)
        item = dict(row)
        item["queue_status"] = (
            "DIRECT_SPOT_CHECK" if row.get("spot_check") else row.get("migration_status")
        )
        out.append(item)
    return out


def load_state(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {row["gold_id"]: row for row in rows if row.get("gold_id")}


def apply_action(row: dict[str, Any], action: str, core_start: int | None) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    out = dict(row)
    out["reviewed_at"] = now
    out["review_action"] = action
    out["owner_confirmation"] = "RECONFIRMED"
    if action == "A":
        out["shape_label"] = "SIGNAL"
        out["migration_status"] = "DIRECT"
        if core_start is not None:
            out["core_start_bar"] = int(core_start)
            out["core_end_exclusive_bar"] = int(core_start) + 4
    elif action == "N":
        out["shape_label"] = "NO_SIGNAL"
        out["core_start_bar"] = None
        out["core_end_exclusive_bar"] = None
        out["core_length"] = None
    elif action == "I":
        out["shape_label"] = "IGNORE"
        out["migration_status"] = "IGNORE"
    else:
        raise ValueError(f"unknown action {action}")
    return out


def save_review(
    state_path: Path,
    history_dir: Path,
    row: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    history_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(state_path)
    state[row["gold_id"]] = row
    write_jsonl(state_path, [state[k] for k in sorted(state)])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    hist = history_dir / f"{stamp}_{row['gold_id']}.json"
    atomic_write_text(hist, json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    return state
