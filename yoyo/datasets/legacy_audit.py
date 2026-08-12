"""Decide, per legacy sample, what Dataset V3 is allowed to do with it.

The V5 brief forbids two shortcuts at once: treating every old positive as gold,
and treating every old hard negative as a true negative. So each legacy row gets
an explicit migration verdict instead of being carried over by default.

Three facts drive most decisions, and each is recorded rather than assumed:

* **who confirmed the class** -- an Owner box, a sampling rule, or only a model;
* **who confirmed the geometry** -- the Owner's original rectangle, a rule
  derived from it, or a model proposal;
* **when the sample happens** -- anything at or after the train cutoff cannot
  enter train, and the frozen val set is never touched.

Owner-long boxes are the one case worth spelling out: they are real structures
the Owner labelled as the opposite direction. The repo's standing rule is that
an unconfirmed mirror is neither a positive nor a negative, so they move to
ignore instead of silently teaching the detector that a dense core is not a
dense core.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

KEEP_POSITIVE = "KEEP_POSITIVE"
KEEP_NEGATIVE = "KEEP_NEGATIVE"
MOVE_TO_IGNORE = "MOVE_TO_IGNORE"
REBOX_REQUIRED = "REBOX_REQUIRED"
RELABEL_REQUIRED = "RELABEL_REQUIRED"
REJECT = "REJECT"

CLASS_CONFIRMED = "confirmed"
CLASS_REJECTED = "rejected"
CLASS_IGNORE = "ignore"

BOX_HUMAN_GOLD = "human_gold"
BOX_RULE_VERIFIED = "rule_verified"
BOX_MODEL_PROPOSED = "model_proposed"
BOX_NONE = "none"


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def classify_positive(row: dict[str, Any]) -> dict[str, Any]:
    """Gold-derived positives: class confirmed by the Owner, box derived by rule."""
    if not row.get("source_owner_gold_confirmed"):
        return {
            "migration": RELABEL_REQUIRED,
            "class_status": CLASS_IGNORE,
            "box_status": BOX_MODEL_PROPOSED,
            "reason": "positive without a confirmed Owner source box",
        }
    if not row.get("center_crop_protocol_owner_directed"):
        return {
            "migration": REBOX_REQUIRED,
            "class_status": CLASS_CONFIRMED,
            "box_status": BOX_MODEL_PROPOSED,
            "reason": "box not produced by the Owner-directed centre-crop rule",
        }
    return {
        "migration": KEEP_POSITIVE,
        "class_status": CLASS_CONFIRMED,
        "box_status": BOX_RULE_VERIFIED,
        "reason": "Owner short box, centre-half rule, geometry never re-drawn by hand",
    }


def classify_easy_negative(row: dict[str, Any]) -> dict[str, Any]:
    """Rule-sampled background: no Owner eyes on it, but no Owner box under it either."""
    if row.get("later_return_used") or row.get("model_score_used"):
        return {
            "migration": REJECT,
            "class_status": CLASS_IGNORE,
            "box_status": BOX_NONE,
            "reason": "selection used future return or model score",
        }
    return {
        "migration": KEEP_NEGATIVE,
        "class_status": CLASS_REJECTED,
        "box_status": BOX_NONE,
        "reason": "sampled outside every Owner box plus guard",
    }


def classify_hard_negative(row: dict[str, Any], owner_no_events: set[str]) -> dict[str, Any]:
    """Hard negatives must be shape-NO. Model-mined ones have never been looked at."""
    kind = str(row.get("candidate_kind"))
    if kind == "owner_long":
        return {
            "migration": MOVE_TO_IGNORE,
            "class_status": CLASS_IGNORE,
            "box_status": BOX_HUMAN_GOLD,
            "reason": "Owner-long mirror: unconfirmed mirrors are neither positive nor negative",
        }
    if str(row.get("source_sample_id")) in owner_no_events:
        return {
            "migration": KEEP_NEGATIVE,
            "class_status": CLASS_REJECTED,
            "box_status": BOX_NONE,
            "reason": "model-mined background later confirmed shape-NO by the Owner",
        }
    return {
        "migration": RELABEL_REQUIRED,
        "class_status": CLASS_IGNORE,
        "box_status": BOX_NONE,
        "reason": "model-mined hard negative, never reviewed: outcome-NO is not shape-NO",
    }


def classify_review_event(
    row: dict[str, Any], train_end: datetime, val_start: datetime
) -> dict[str, Any]:
    """Owner review verdicts: NO is directly usable, YES is class-only."""
    verdict = str(row.get("verdict"))
    decision = parse_time(row["decision_time"])
    if row.get("touches_owner_box_guard") or row.get("intersects_owner_gold_box"):
        # Recomputed here rather than trusted from the pack: the packs' own guard
        # flag is false for all 1,200 events, yet ten of them do overlap a gold
        # box once the spans are rebuilt from the positive manifest.
        return {
            "migration": MOVE_TO_IGNORE,
            "class_status": CLASS_IGNORE,
            "box_status": BOX_MODEL_PROPOSED,
            "reason": "event window overlaps an Owner gold box plus guard",
        }
    if decision >= val_start:
        return {
            "migration": MOVE_TO_IGNORE,
            "class_status": CLASS_IGNORE,
            "box_status": BOX_MODEL_PROPOSED,
            "reason": "decision falls in the frozen val period; val stays byte-identical",
        }
    if decision > train_end:
        return {
            "migration": MOVE_TO_IGNORE,
            "class_status": CLASS_IGNORE,
            "box_status": BOX_MODEL_PROPOSED,
            "reason": "decision falls inside the purge gap between train and val",
        }
    if verdict == "NO":
        return {
            "migration": KEEP_NEGATIVE,
            "class_status": CLASS_REJECTED,
            "box_status": BOX_NONE,
            "reason": "Owner shape-NO on a real detection: usable hard negative",
        }
    if verdict == "YES":
        return {
            "migration": REBOX_REQUIRED,
            "class_status": CLASS_CONFIRMED,
            "box_status": BOX_MODEL_PROPOSED,
            "reason": "class confirmed by the Owner, geometry still model-proposed",
        }
    return {
        "migration": REBOX_REQUIRED,
        "class_status": CLASS_IGNORE,
        "box_status": BOX_MODEL_PROPOSED,
        "reason": f"Owner asked for a re-box ({verdict})",
    }


def owner_box_spans(
    positives: Iterable[dict[str, Any]], bar_minutes: int = 15, guard_bars: int = 12
) -> dict[str, list[tuple[datetime, datetime]]]:
    """Absolute time span of every original Owner box, widened by the guard.

    Rebuilt from the positive manifest rather than read off a flag, because the
    packs' own ``touches_owner_box_guard`` is false everywhere and still misses
    real overlaps.
    """
    from datetime import timedelta

    bar = timedelta(minutes=bar_minutes)
    spans: dict[str, list[tuple[datetime, datetime]]] = {}
    for row in positives:
        source = row.get("source_owner_global")
        if not source:
            continue
        window_start = int(row["win_start"])
        start_time = parse_time(row["start_time"])
        box_start, box_end = (int(value) for value in source)
        spans.setdefault(str(row["symbol"]), []).append(
            (
                start_time + (box_start - window_start) * bar - guard_bars * bar,
                start_time + (box_end - window_start) * bar + guard_bars * bar,
            )
        )
    return spans


def intersects_owner_gold_box(
    symbol: str,
    window_start: datetime,
    window_end: datetime,
    spans: dict[str, list[tuple[datetime, datetime]]],
) -> bool:
    return any(
        window_start <= box_end and box_start <= window_end
        for box_start, box_end in spans.get(symbol, ())
    )


def outcome_alignment(verdict: str, forward_return_bp: float | None) -> str:
    """Does the verdict agree with what happened next?

    A short-side YES that is followed by a fall, or a NO followed by a rise,
    agrees with the outcome and therefore cannot tell us whether the Owner
    judged the shape or the result. The verdicts that *contradict* the outcome
    are the ones we can be confident were shape judgements.
    """
    if forward_return_bp is None:
        return "unknown"
    fell = forward_return_bp < 0
    if verdict == "YES":
        return "agrees_with_outcome" if fell else "contradicts_outcome"
    if verdict == "NO":
        return "agrees_with_outcome" if not fell else "contradicts_outcome"
    return "unknown"


def tally(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get(key))] = counts.get(str(row.get(key)), 0) + 1
    return dict(sorted(counts.items()))
