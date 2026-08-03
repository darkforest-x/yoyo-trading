"""P0.3: a forward row records which contract produced it, and when it was decided.

Acceptance B-03/B-04 (identity), H-01/H-02/H-03/H-05 (log isolation), F-05
(per-candidate decision time).

The failure being prevented is quiet contamination rather than a crash. The log
already holds rows written under a long resolver, and the repaired short protocol
is supposed to earn a fresh 100 trades. Without provenance in the key, the same
bar under two contracts is one row, and the old one absorbs the new one's outcome
-- no error, just a book that averages two different strategies and calls it
evidence.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from yoyo.contracts.forward_log import (
    LEGACY_SIDE,
    actionable_rows,
    forward_key,
    merge_forward_log,
    normalize_log,
    read_forward_log,
    rows_for_protocol,
    write_forward_log,
)
from yoyo.contracts.forward_log import LEGACY_PROTOCOL, LEGACY_SEMANTICS

T = pd.Timestamp("2026-08-01 03:00:00+00:00")


def _row(**over) -> dict:
    row = {
        "source": "okx", "symbol": "BTC_USDT_SWAP", "signal_time": str(T),
        "detected_at": str(T), "status": "open", "score": 0.1, "threshold": 0.0,
        "model_path": "m", "dataset_sha256": "dataset-sha", "signal_i": 5,
        "entry_time": str(T), "entry_price": 100.0, "maker_filled": True,
        "outcome": "", "label": -1, "exit_offset": 0, "exit_time": "",
        "realized_ret": 0.0, "atr_pct": 0.01, "dense_run_len": 8,
        "tier": "q90_q95", "size_mult": 1.0, "side": "short",
        "protocol_version": "short_v11", "strategy_id": "dense_start_short_15m",
        "feature_semantics": "side_aligned_v1", "decision_at": str(T),
        "execution_eligible": True,
        "model_sha256": "model-sha", "detector_sha256": "detector-sha",
    }
    row.update(over)
    return row


# ── B-03 / B-04 ───────────────────────────────────────────────────────────────
def test_side_changes_the_key() -> None:
    assert forward_key("okx", "BTC", T, "short", "p1") != forward_key("okx", "BTC", T, "long", "p1")


def test_protocol_version_changes_the_key() -> None:
    assert forward_key("okx", "BTC", T, "short", "p1") != forward_key("okx", "BTC", T, "short", "p2")


def test_score_is_not_in_the_key() -> None:
    """Re-scoring the same event must not create a second one (B-01)."""
    assert forward_key("okx", "BTC", T, "short", "p1") == forward_key("okx", "BTC", T, "short", "p1")


def test_absent_provenance_gets_its_own_identity_not_the_live_one() -> None:
    """An old row must not be adopted by whatever protocol is running now."""
    legacy = forward_key("okx", "BTC", T)
    assert legacy == ("okx", "BTC", str(T), LEGACY_SIDE, LEGACY_PROTOCOL)
    assert legacy != forward_key("okx", "BTC", T, "short", "short_v11")


# ── H-01 / H-02 ───────────────────────────────────────────────────────────────
def test_two_protocols_on_the_same_bar_stay_two_rows() -> None:
    """The contamination this whole change exists to stop.

    Same symbol, same signal_time, one legacy long row and one repaired short row.
    Merging them would let the old contract's outcome stand in for the new one's.
    """
    existing = pd.DataFrame([_row(side="long", protocol_version=LEGACY_PROTOCOL)])
    result = merge_forward_log(existing, [_row(side="short", protocol_version="short_v11")])
    assert len(result.frame) == 2
    assert result.new_signals == 1


def test_summary_can_isolate_one_protocol() -> None:
    frame = pd.DataFrame([
        _row(protocol_version=LEGACY_PROTOCOL, symbol="A_USDT_SWAP"),
        _row(protocol_version="short_v11", symbol="B_USDT_SWAP"),
        _row(protocol_version="short_v11", symbol="C_USDT_SWAP"),
    ])
    assert len(rows_for_protocol(frame, "short_v11")) == 2
    assert len(rows_for_protocol(frame, LEGACY_PROTOCOL)) == 1


def test_closed_outcome_does_not_cross_protocols() -> None:
    """A close under one protocol must not close the other protocol's open row."""
    existing = pd.DataFrame([_row(protocol_version="short_v11", status="open")])
    closing = _row(protocol_version=LEGACY_PROTOCOL, status="closed",
                   outcome="TP", label=1, realized_ret=0.05)
    result = merge_forward_log(existing, [closing])
    still_open = result.frame[result.frame["protocol_version"] == "short_v11"]
    assert list(still_open["status"]) == ["open"]
    assert result.closed_updates == 0


# ── H-03 ──────────────────────────────────────────────────────────────────────
def test_only_execution_eligible_rows_are_actionable() -> None:
    frame = pd.DataFrame([
        _row(symbol="A_USDT_SWAP", execution_eligible=True),
        _row(symbol="B_USDT_SWAP", execution_eligible=False),
    ])
    actionable = actionable_rows(frame)
    assert list(actionable["symbol"]) == ["A_USDT_SWAP"]


def test_the_string_False_is_not_eligible(tmp_path: Path) -> None:
    """CSV has no booleans. "False" read back and cast with astype(bool) is True.

    That single coercion would turn every ineligible row into an order candidate,
    so eligibility is decided by a truth test on the text, not by a cast.
    """
    path = tmp_path / "forward_log.csv"
    write_forward_log(path, pd.DataFrame([_row(execution_eligible=False)]))
    assert "False" in path.read_text(encoding="utf-8")
    assert actionable_rows(read_forward_log(path)).empty


def test_eligible_survives_a_csv_round_trip(tmp_path: Path) -> None:
    """The guard must not be so blunt that nothing is ever actionable."""
    path = tmp_path / "forward_log.csv"
    write_forward_log(path, pd.DataFrame([_row(execution_eligible=True)]))
    assert len(actionable_rows(read_forward_log(path))) == 1


# ── H-05 ──────────────────────────────────────────────────────────────────────
def test_a_log_predating_provenance_reads_back_marked_not_crashed(tmp_path: Path) -> None:
    old = _row()
    for column in ("protocol_version", "strategy_id", "feature_semantics",
                   "decision_at", "execution_eligible", "model_sha256", "detector_sha256"):
        old.pop(column)
    path = tmp_path / "forward_log.csv"
    pd.DataFrame([old]).to_csv(path, index=False)

    frame = read_forward_log(path)
    assert len(frame) == 1
    assert frame["protocol_version"].iloc[0] == LEGACY_PROTOCOL
    assert frame["feature_semantics"].iloc[0] == LEGACY_SEMANTICS
    assert bool(frame["execution_eligible"].iloc[0]) is False
    # and being legacy, it is not in anyone's actionable set
    assert actionable_rows(frame).empty


def test_legacy_rows_do_not_enter_a_new_protocols_count(tmp_path: Path) -> None:
    """H-02 stated as the number it protects: the repaired book's 100 trades."""
    old = _row(symbol="OLD_USDT_SWAP")
    for column in ("protocol_version", "strategy_id", "feature_semantics",
                   "decision_at", "execution_eligible", "model_sha256", "detector_sha256"):
        old.pop(column)
    path = tmp_path / "forward_log.csv"
    pd.DataFrame([old, _row(symbol="NEW_USDT_SWAP")]).to_csv(path, index=False)

    frame = read_forward_log(path)
    assert len(rows_for_protocol(frame, "short_v11")) == 1


# ── F-05 ──────────────────────────────────────────────────────────────────────
def test_decision_at_is_a_column_distinct_from_detected_at() -> None:
    """Detection and decision are different moments and must be separately readable.

    A scan covering 344 symbols finishes them minutes apart; one shared timestamp
    would claim decisions that had not happened yet.
    """
    frame = normalize_log(pd.DataFrame([_row()]))
    assert "decision_at" in frame.columns
    assert "detected_at" in frame.columns


def test_outcome_update_preserves_first_decision_and_artifact_identity() -> None:
    existing = pd.DataFrame(
        [
            _row(
                status="open",
                detected_at="first-detected",
                decision_at="first-decision",
                model_sha256="first-model",
                detector_sha256="first-detector",
            )
        ]
    )
    closing = _row(
        status="closed",
        outcome="tp",
        label=1,
        realized_ret=0.05,
        detected_at="later-detected",
        decision_at="later-decision",
        model_sha256="later-model",
        detector_sha256="later-detector",
    )
    row = merge_forward_log(existing, [closing]).frame.iloc[0]
    assert row["detected_at"] == "first-detected"
    assert row["decision_at"] == "first-decision"
    assert row["model_sha256"] == "first-model"
    assert row["detector_sha256"] == "first-detector"


def test_normalize_is_idempotent() -> None:
    """Reading a normalized log again must not re-mark or drop anything."""
    once = normalize_log(pd.DataFrame([_row()]))
    twice = normalize_log(once)
    pd.testing.assert_frame_equal(once, twice)


@pytest.mark.parametrize("blank", ["", "nan", None])
def test_blank_provenance_is_treated_as_legacy_not_as_a_real_protocol(blank) -> None:
    key = forward_key("okx", "BTC", T, "short", blank)
    assert key[4] == LEGACY_PROTOCOL
