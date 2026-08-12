"""Migration verdicts: the rules that decide what Dataset V3 may reuse."""
from __future__ import annotations

from datetime import datetime, timezone

from yoyo.datasets.legacy_audit import (
    classify_easy_negative,
    classify_hard_negative,
    classify_positive,
    classify_review_event,
    outcome_alignment,
)

TRAIN_END = datetime(2026, 3, 13, 23, 45, tzinfo=timezone.utc)
VAL_START = datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc)


def review(decision: str, verdict: str = "NO", **extra) -> dict:
    return {"decision_time": decision, "verdict": verdict, **extra}


def test_gold_derived_positive_is_kept_but_its_box_is_only_rule_verified() -> None:
    result = classify_positive(
        {"source_owner_gold_confirmed": True, "center_crop_protocol_owner_directed": True}
    )
    assert result["migration"] == "KEEP_POSITIVE"
    assert result["class_status"] == "confirmed"
    assert result["box_status"] == "rule_verified"


def test_positive_without_owner_source_box_needs_relabelling() -> None:
    assert classify_positive({"source_owner_gold_confirmed": False})["migration"] == "RELABEL_REQUIRED"


def test_positive_from_a_non_owner_rule_needs_reboxing() -> None:
    result = classify_positive(
        {"source_owner_gold_confirmed": True, "center_crop_protocol_owner_directed": False}
    )
    assert result["migration"] == "REBOX_REQUIRED"
    assert result["class_status"] == "confirmed"


def test_easy_negative_selected_with_future_return_is_rejected() -> None:
    assert classify_easy_negative({"later_return_used": True})["migration"] == "REJECT"
    assert classify_easy_negative({"model_score_used": True})["migration"] == "REJECT"
    assert classify_easy_negative({})["migration"] == "KEEP_NEGATIVE"


def test_owner_long_mirror_goes_to_ignore_not_to_negatives() -> None:
    result = classify_hard_negative({"candidate_kind": "owner_long"}, set())
    assert result["migration"] == "MOVE_TO_IGNORE"
    assert result["class_status"] == "ignore"


def test_model_mined_hard_negative_needs_review_unless_owner_said_no() -> None:
    unreviewed = classify_hard_negative(
        {"candidate_kind": "model_mined_pool", "source_sample_id": "e1"}, set()
    )
    assert unreviewed["migration"] == "RELABEL_REQUIRED"
    confirmed = classify_hard_negative(
        {"candidate_kind": "model_mined_pool", "source_sample_id": "e1"}, {"e1"}
    )
    assert confirmed["migration"] == "KEEP_NEGATIVE"


def test_owner_no_inside_train_becomes_a_usable_hard_negative() -> None:
    result = classify_review_event(review("2026-01-15T00:00:00+00:00", "NO"), TRAIN_END, VAL_START)
    assert result["migration"] == "KEEP_NEGATIVE"
    assert result["box_status"] == "none"


def test_owner_yes_is_class_only_until_the_box_is_reviewed() -> None:
    result = classify_review_event(review("2026-01-15T00:00:00+00:00", "YES"), TRAIN_END, VAL_START)
    assert result["migration"] == "REBOX_REQUIRED"
    assert result["class_status"] == "confirmed"
    assert result["box_status"] == "model_proposed"


def test_val_period_and_purge_gap_events_never_enter_training() -> None:
    in_val = classify_review_event(review("2026-04-01T00:00:00+00:00", "NO"), TRAIN_END, VAL_START)
    assert in_val["migration"] == "MOVE_TO_IGNORE" and "val period" in in_val["reason"]
    in_purge = classify_review_event(review("2026-03-14T06:00:00+00:00", "NO"), TRAIN_END, VAL_START)
    assert in_purge["migration"] == "MOVE_TO_IGNORE" and "purge" in in_purge["reason"]


def test_event_touching_a_gold_box_guard_is_ignored_even_when_owner_said_no() -> None:
    result = classify_review_event(
        review("2026-01-15T00:00:00+00:00", "NO", touches_owner_box_guard=True),
        TRAIN_END,
        VAL_START,
    )
    assert result["migration"] == "MOVE_TO_IGNORE"


def test_outcome_alignment_marks_contradiction_as_the_informative_case() -> None:
    assert outcome_alignment("YES", -120.0) == "agrees_with_outcome"
    assert outcome_alignment("YES", 80.0) == "contradicts_outcome"
    assert outcome_alignment("NO", 80.0) == "agrees_with_outcome"
    assert outcome_alignment("NO", -120.0) == "contradicts_outcome"
    assert outcome_alignment("YES", None) == "unknown"


def test_gold_box_overlap_is_recomputed_not_trusted() -> None:
    """The packs' own guard flag is false everywhere; the recomputed span still bites."""
    from yoyo.datasets.legacy_audit import intersects_owner_gold_box, owner_box_spans

    spans = owner_box_spans(
        [
            {
                "symbol": "ETH_USDT_SWAP",
                "win_start": 100,
                "start_time": "2026-01-01T00:00:00+00:00",
                "source_owner_global": [104, 108],
            }
        ],
        guard_bars=12,
    )
    box_start, box_end = spans["ETH_USDT_SWAP"][0]
    assert intersects_owner_gold_box("ETH_USDT_SWAP", box_start, box_end, spans)
    far_start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    assert not intersects_owner_gold_box(
        "ETH_USDT_SWAP", far_start, far_start, spans
    )
    assert not intersects_owner_gold_box("SOL_USDT_SWAP", box_start, box_end, spans)


def test_recomputed_overlap_sends_an_owner_no_to_ignore() -> None:
    result = classify_review_event(
        review("2026-01-15T00:00:00+00:00", "NO", intersects_owner_gold_box=True),
        TRAIN_END,
        VAL_START,
    )
    assert result["migration"] == "MOVE_TO_IGNORE"
    assert "gold box" in result["reason"]
