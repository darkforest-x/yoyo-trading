"""Gold JSONL schema: Owner verdict is the source of truth, not YOLO txt."""

from __future__ import annotations

import hashlib
import json
from typing import Any

REQUIRED = (
    "gold_id",
    "symbol",
    "timeframe",
    "source_repo",
    "source_path",
    "candidate_source",
    "decision_bar",
    "decision_time",
    "local_start_bar",
    "local_end_bar",
    "local_window_length",
    "shape_label",
    "box_rule",
    "box_status",
    "reviewer",
    "holdout_read",
)
SHAPE = frozenset({"POSITIVE", "NEGATIVE", "IGNORE"})


def gold_id(symbol: str, timeframe: str, decision_time: str) -> str:
    stamp = decision_time.replace("-", "").replace(":", "").replace("+00:00", "Z")
    if stamp.endswith("Z") is False:
        stamp = stamp.replace("+0000", "Z")
    return f"{symbol}_{timeframe}_{stamp}"


def validate_gold(row: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED if key not in row]
    if missing:
        raise ValueError(f"gold missing fields: {missing}")
    if row["shape_label"] not in SHAPE:
        raise ValueError(f"bad shape_label: {row['shape_label']}")
    if int(row["local_end_bar"]) != int(row["decision_bar"]):
        raise ValueError("local_end_bar must equal decision_bar")
    if int(row["local_window_length"]) != int(row["local_end_bar"]) - int(row["local_start_bar"]) + 1:
        raise ValueError("local_window_length mismatch")
    if row.get("holdout_read"):
        raise ValueError("gold row reports a holdout read")
    if row["shape_label"] == "POSITIVE":
        if row.get("core_start_bar") is None or row.get("core_end_bar") is None:
            raise ValueError("POSITIVE requires core_start_bar/core_end_bar")
        if int(row["core_end_bar"]) < int(row["core_start_bar"]):
            raise ValueError("core_end_bar < core_start_bar")
        if int(row["core_end_bar"]) > int(row["decision_bar"]):
            raise ValueError("core uses future bars")
        if int(row["core_start_bar"]) < int(row["local_start_bar"]):
            raise ValueError("core starts before local window")
    else:
        if row.get("core_start_bar") is not None or row.get("core_end_bar") is not None:
            raise ValueError("NEGATIVE/IGNORE must not carry a core box")


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in text.splitlines() if line]
    seen: set[str] = set()
    out = []
    for row in rows:
        validate_gold(row)
        if row["gold_id"] in seen:
            continue
        seen.add(row["gold_id"])
        out.append(row)
    return out


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
