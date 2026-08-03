"""Append-only JSONL ledger for demo fills / attempts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_all(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def signal_keys_already_taken(path: Path) -> set[str]:
    """Return signal_key values we already tried to open (success, fail, or partial).

    order_partial MUST be included: entry already filled, retrying would double
    the position (2026-07-16 DOGE-style incident class).
    """
    out: set[str] = set()
    for row in load_all(path):
        if row.get("event") in {
            "order_placed",
            "order_partial",
            "order_skipped",
            "order_failed",
            "skipped",
            "skipped_invalid_barriers",
            "skipped_unsupported_side",
            "skipped_protocol_mismatch",
            "skipped_ineligible_protocol",
            "dry_run",
        }:
            k = row.get("signal_key")
            if k:
                out.add(str(k))
    return out


def consecutive_losses(path: Path) -> int:
    """Count trailing closed losses from the end of the ledger."""
    n = 0
    for row in reversed(load_all(path)):
        if row.get("event") != "closed":
            continue
        net = row.get("net_ret")
        if net is None:
            break
        if float(net) < 0:
            n += 1
        else:
            break
    return n
