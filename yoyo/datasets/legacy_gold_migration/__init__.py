"""Migrate legacy Gold / YOLO / review labels into fixed W10 Core4 Confirm1 events.

Source: Owner 2026-08-13 brief. Old assets already exist (8768 JSONL, Label Studio
``yoyo_gold_v1``, Owner review packs, ``source_owner_global`` boxes, V3
rule-crops). This package converts each source through an adapter that declares
its original interval semantics, then emits half-open event records. It does not
retrain a detector and does not treat model boxes as Gold.
"""

from .canonical import gold_id_from_time, w10_fields
from .intervals import (
    inclusive_to_half_open,
    map_w10,
    suggest_four_bar,
    yolo_box_to_bars,
)

__all__ = [
    "gold_id_from_time",
    "inclusive_to_half_open",
    "map_w10",
    "suggest_four_bar",
    "w10_fields",
    "yolo_box_to_bars",
]
