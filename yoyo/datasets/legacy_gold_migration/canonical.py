"""Canonical event-level Gold records for the fixed W10 contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from yoyo.datasets.gold_schema import gold_id as compact_gold_id

from .intervals import map_w10

SHAPE = frozenset({"SIGNAL", "NO_SIGNAL", "IGNORE"})
STATUS = frozenset({"DIRECT", "ONE_CLICK_REVIEW", "MANUAL_ADJUST", "IGNORE", "CONFLICT"})


def gold_id_from_time(symbol: str, timeframe: str, decision_time: str) -> str:
    """Keep the compact id used by 8768 / Label Studio tasks so rows join."""
    return compact_gold_id(symbol, timeframe, decision_time)


def parse_time(value: str):
    text = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_holdout(decision_time: str | None, holdout) -> bool:
    if not decision_time:
        return False
    return parse_time(decision_time) >= holdout


def w10_fields(core_start: int, core_end_exclusive: int) -> dict[str, int]:
    return map_w10(core_start, core_end_exclusive)


def no_signal_slot(confirmation_bar: int) -> dict[str, int]:
    e = int(confirmation_bar)
    s = e - 4
    mapped = map_w10(s, e)
    return {
        "slot_start_bar": s,
        "slot_end_exclusive_bar": e,
        "confirmation_bar": e,
        "decision_bar": e,
        "window_start_bar": mapped["window_start_bar"],
        "window_end_exclusive_bar": mapped["window_end_exclusive_bar"],
        "window_length": 10,
        "local_core_start": 5,
        "local_core_end_exclusive": 9,
        "local_confirmation_position": 9,
    }


def base_event(**kwargs: Any) -> dict[str, Any]:
    row = {
        "gold_id": None,
        "event_id": None,
        "event_group_id": None,
        "symbol": None,
        "timeframe": "15m",
        "source_repo": "fable-trading",
        "source_commit": None,
        "source_dataset": None,
        "source_record_id": None,
        "source_annotation_type": None,
        "source_interval_semantics": None,
        "shape_label": None,
        "core_start_bar": None,
        "core_end_exclusive_bar": None,
        "core_length": None,
        "confirmation_bar": None,
        "decision_bar": None,
        "window_start_bar": None,
        "window_end_exclusive_bar": None,
        "window_length": None,
        "local_core_start": 5,
        "local_core_end_exclusive": 9,
        "local_confirmation_position": 9,
        "slot_start_bar": None,
        "slot_end_exclusive_bar": None,
        "migration_status": None,
        "owner_confirmation": "NOT_REQUIRED",
        "legacy_decision_offset": None,
        "decision_rebased": False,
        "contains_other_core": False,
        "render_status": "PENDING",
        "split": None,
        "holdout_read": False,
        "future_used_in_model_input": False,
        "suggested_core_start_bar": None,
        "suggested_core_end_exclusive_bar": None,
        "negative_type": None,
        "negative_sampling_rule": None,
        "ohlcv_sha256": None,
        "image_sha256": None,
        "config_sha256": None,
        "builder_commit": None,
    }
    row.update(kwargs)
    return row
