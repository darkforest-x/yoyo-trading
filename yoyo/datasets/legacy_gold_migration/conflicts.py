"""Resolve overlapping source records. Higher precedence never loses silently."""

from __future__ import annotations

from typing import Any

from .canonical import gold_id_from_time
from .sources import SourceRecord


def _norm_time(value: str | None) -> str | None:
    if not value:
        return None
    from .canonical import parse_time

    return parse_time(value).isoformat()


def _gold_key(rec: SourceRecord) -> tuple:
    return (rec.symbol, rec.timeframe or "15m", rec.decision_time or "", rec.decision_bar)


def cluster_records(records: list[SourceRecord], center_max_bars: int = 6) -> list[list[int]]:
    """Match sources that describe the same decision, not merely nearby cores.

    The 6-bar event-group rule is applied later for split assignment. Using it
    here would let a nearby Owner hard-negative veto an Owner gold box.
    """
    parent = {i: i for i in range(len(records))}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    buckets: dict[tuple, list[int]] = {}
    for i, rec in enumerate(records):
        gid = rec.extra.get("legacy_gold_id")
        if not gid and rec.decision_time:
            gid = gold_id_from_time(rec.symbol, rec.timeframe, rec.decision_time)
        sid = rec.extra.get("sample_id") or rec.source_record_id
        keys = []
        if gid:
            keys.append(("gold_id", str(gid)))
        if sid:
            keys.append(("sid", rec.symbol, str(sid)))
        if rec.symbol and rec.decision_time:
            keys.append(("dt", rec.symbol, rec.timeframe, _norm_time(rec.decision_time)))
        if rec.symbol and rec.decision_bar is not None:
            keys.append(("db", rec.symbol, rec.timeframe, int(rec.decision_bar)))
        for key in keys:
            buckets.setdefault(key, []).append(i)
    for idxs in buckets.values():
        for a, b in zip(idxs, idxs[1:]):
            union(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _sort_key(rec: SourceRecord) -> tuple:
    reviewed = rec.reviewed_at or ""
    return (rec.precedence, 0 if reviewed else 1, reviewed)


def resolve_cluster(records: list[SourceRecord], idxs: list[int]) -> dict[str, Any]:
    ordered = sorted(idxs, key=lambda i: _sort_key(records[i]))
    winner = records[ordered[0]]
    conflicts: list[dict[str, Any]] = []
    for i in ordered[1:]:
        other = records[i]
        same_shape = other.shape_label == winner.shape_label
        same_core = (
            other.core_start_bar == winner.core_start_bar
            and other.core_end_exclusive_bar == winner.core_end_exclusive_bar
        )
        if same_shape and same_core:
            continue
        if other.precedence == winner.precedence and not same_shape:
            resolution = "OWNER_REVIEW_REQUIRED"
        else:
            resolution = "KEEP_HIGHER"
        conflicts.append(
            {
                "higher_source": winner.source_annotation_type,
                "higher_value": {
                    "shape_label": winner.shape_label,
                    "core_start_bar": winner.core_start_bar,
                    "core_end_exclusive_bar": winner.core_end_exclusive_bar,
                    "source_record_id": winner.source_record_id,
                },
                "lower_source": other.source_annotation_type,
                "lower_value": {
                    "shape_label": other.shape_label,
                    "core_start_bar": other.core_start_bar,
                    "core_end_exclusive_bar": other.core_end_exclusive_bar,
                    "source_record_id": other.source_record_id,
                },
                "resolution": resolution,
            }
        )
    geometry = winner
    if winner.core_start_bar is None:
        for i in ordered[1:]:
            if records[i].core_start_bar is not None:
                geometry = records[i]
                break
    needs_owner = any(c["resolution"] == "OWNER_REVIEW_REQUIRED" for c in conflicts)
    return {
        "winner": winner,
        "geometry": geometry,
        "conflicts": conflicts,
        "needs_owner": needs_owner,
        "member_ids": [records[i].source_record_id for i in ordered],
    }
