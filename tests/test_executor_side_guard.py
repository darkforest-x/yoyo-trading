"""Direction compatibility guard for the current long-only executor."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from unittest.mock import Mock

from yoyo.layers.l4_execution import ledger
from yoyo.layers.l4_execution.config import ExecutorConfig
from yoyo.layers.l4_execution.executor import (
    MISSING_SIDE,
    open_one,
    run_once,
    signal_key,
    signal_trade_side,
)
from yoyo.contracts.forward_log import LEGACY_PROTOCOL


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("LONG", "long"),
        ("short", "short"),
        ("buy", "buy"),
    ],
)
def test_signal_trade_side_preserves_explicit_direction(raw, expected) -> None:
    assert signal_trade_side(pd.Series({"side": raw})) == expected


@pytest.mark.parametrize("raw", [None, float("nan"), "", "   "])
def test_absent_side_is_not_long(raw) -> None:
    """The one default that can turn an unlabelled row into a real buy.

    Until 2026-08-03 these resolved to "long". The mainline is short, so the row
    most likely to arrive without a side is a short one, and defaulting it to the
    only direction this executor can actually trade is the worst available guess.
    Takeover plan P0-02 / acceptance A-03.
    """
    assert signal_trade_side(pd.Series({"side": raw})) == MISSING_SIDE


def _protocol(*, side: str = "short", eligible: bool = True):
    return SimpleNamespace(
        protocol_version="short_v1" if side == "short" else "long_test_v1",
        strategy_id="dense_start_short_15m" if side == "short" else "legacy_long_test",
        feature_semantics="side_aligned_v1",
        threshold=0.5,
        side=side,
        execution_eligible=eligible,
        model_sha256="model-sha",
        detector_sha256="detector-sha",
        dataset_sha256="dataset-sha",
        passes_threshold=lambda score: score >= 0.5,
        accepts_row_side=lambda raw: (
            raw is not None and not pd.isna(raw) and str(raw).strip().lower() == side
        ),
    )


def test_signal_key_is_stable_under_rescoring_and_model_change() -> None:
    base = {
        "source": "okx",
        "symbol": "ETH_USDT_SWAP",
        "signal_time": "2026-05-03T03:00:00+00:00",
        "side": "short",
        "protocol_version": "short_v1",
    }
    first = pd.Series({**base, "score": 0.1, "model_sha256": "aaa"})
    rescored = pd.Series({**base, "score": 0.9, "model_sha256": "bbb"})

    assert signal_key(first) == signal_key(rescored)
    assert signal_key(first).split("|") == [
        "okx", "ETH_USDT_SWAP", "2026-05-03T03:00:00+00:00", "short", "short_v1"
    ]


def test_signal_key_separates_side_and_protocol() -> None:
    base = pd.Series(
        {
            "source": "okx",
            "symbol": "ETH_USDT_SWAP",
            "signal_time": "2026-05-03T03:00:00+00:00",
            "side": "short",
            "protocol_version": "short_v1",
        }
    )

    assert signal_key(base) != signal_key(pd.Series({**base.to_dict(), "side": "long"}))
    assert signal_key(base) != signal_key(
        pd.Series({**base.to_dict(), "protocol_version": "short_v2"})
    )
    legacy = signal_key(
        pd.Series(
            {
                "source": "okx",
                "symbol": "ETH_USDT_SWAP",
                "signal_time": "2026-05-03T03:00:00+00:00",
                "side": "short",
            }
        )
    )
    assert legacy.endswith(f"|short|{LEGACY_PROTOCOL}")


def _write_signal(path: Path, *, side: str, eligible: bool = True) -> None:
    now = pd.Timestamp.now(tz="UTC")
    pd.DataFrame(
        [
            {
                "source": "okx",
                "symbol": "BTC_USDT_SWAP",
                "signal_time": str(now - pd.Timedelta(minutes=5)),
                "detected_at": str(now),
                "status": "open",
                "score": 0.9,
                "threshold": 0.5,
                "entry_price": 100.0,
                "atr_pct": 0.01,
                "side": side,
                "protocol_version": "short_v1",
                "strategy_id": "dense_start_short_15m",
                "feature_semantics": "side_aligned_v1",
                "execution_eligible": eligible,
                "model_sha256": "model-sha",
                "detector_sha256": "detector-sha",
                "dataset_sha256": "dataset-sha",
            }
        ]
    ).to_csv(path, index=False)


@pytest.mark.parametrize(
    ("side", "expected_event"),
    [
        ("short", "skipped_unsupported_side"),
        ("unknown", "skipped_protocol_mismatch"),
        ("", "skipped_protocol_mismatch"),
    ],
)
def test_run_once_rejects_non_long_without_retry_spam(
    tmp_path: Path, side: str, expected_event: str
) -> None:
    forward_log = tmp_path / "forward_log.csv"
    ledger_path = tmp_path / "ledger.jsonl"
    _write_signal(forward_log, side=side)
    cfg = ExecutorConfig(
        forward_log=str(forward_log),
        ledger=str(ledger_path),
        kill_switch_file=str(tmp_path / "KILL"),
        sizing_mode="fixed",
        notional_usdt=10.0,
    )

    first = run_once(cfg, dry_run=True, protocol=_protocol())

    assert first["opened"] == 0
    assert first["skipped"] == 1
    events = ledger.load_all(ledger_path)
    assert len(events) == 1
    assert events[0]["event"] == expected_event
    # normalized, not raw: an empty CSV field reads back as NaN and must land on
    # MISSING_SIDE so the ledger says why it was refused rather than showing blank
    assert events[0]["signal_side"] == (side or MISSING_SIDE)
    assert events[0]["side"] is None

    second = run_once(cfg, dry_run=True, protocol=_protocol())
    assert second["opened"] == 0
    assert second["skipped"] == 0
    assert len(ledger.load_all(ledger_path)) == 1


@pytest.mark.parametrize("side", ["short", "unknown", None, float("nan"), ""])
def test_rejected_side_never_calls_trading_client(side) -> None:
    client = Mock()
    row = pd.Series(
        {
            "source": "okx",
            "symbol": "BTC_USDT_SWAP",
            "signal_time": "2026-05-03T03:00:00+00:00",
            "protocol_version": "short_v1",
            "side": side,
            "score": 0.9,
            "threshold": 0.5,
            "entry_price": 100.0,
            "atr_pct": 0.01,
        }
    )

    event = open_one(client, ExecutorConfig(), row, dry_run=False, protocol=_protocol())

    expected = "skipped_unsupported_side" if side == "short" else "skipped_protocol_mismatch"
    assert event["event"] == expected
    assert event["side"] is None
    assert client.mock_calls == []


def test_long_row_under_short_protocol_cannot_reach_buy_client() -> None:
    client = Mock()
    row = pd.Series(
        {
            "source": "okx",
            "symbol": "BTC_USDT_SWAP",
            "signal_time": "2026-05-03T03:00:00+00:00",
            "protocol_version": "short_v1",
            "side": "long",
            "score": 0.9,
            "threshold": 0.5,
            "entry_price": 100.0,
            "atr_pct": 0.01,
        }
    )

    event = open_one(client, ExecutorConfig(), row, dry_run=False, protocol=_protocol())

    assert event["event"] == "skipped_protocol_mismatch"
    assert event["side"] is None
    assert client.mock_calls == []


def test_execution_ineligible_row_never_enters_order_queue(tmp_path: Path) -> None:
    forward_log = tmp_path / "forward_log.csv"
    _write_signal(forward_log, side="short", eligible=False)
    cfg = ExecutorConfig(forward_log=str(forward_log))

    summary = run_once(cfg, dry_run=True, protocol=_protocol())

    assert summary["opened"] == 0
    assert summary["note"] == "no actionable rows in forward_log"
