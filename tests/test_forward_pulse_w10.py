"""Live-pulse safety tests: tip restriction, freshness, causality, dedup, no orders.

These cover the parts that would fail silently in production rather than loudly
in a backtest -- a stale series scored as current, a bar outside tip/tip-1/tip-2,
a duplicate entry on the same event, or a config whose barrier drifts from the
frozen contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import yoyo.cli.forward_pulse_w10 as pulse
from yoyo.contracts.outcomes import ATR_PCT_MIN

CONFIG = Path(__file__).resolve().parents[1] / "configs/live_w10_pulse_v1.json"


def _frame(bars: int = 320, *, tip: str = "2026-08-14T04:00:00Z") -> pd.DataFrame:
    """A frame built through the real indicator chain, so features are well-defined.

    A hand-written frame with only the MA columns passes the tip/freshness tests
    but not `add_features`, and a test that skips the real chain would not be
    testing the thing that runs live.
    """
    end = pd.Timestamp(tip)
    times = pd.date_range(end - pd.Timedelta(minutes=15 * (bars - 1)), periods=bars, freq="15min")
    rng = np.random.default_rng(20260814)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, bars)))
    spread = np.abs(rng.normal(0.0, 0.003, bars)) * close + 1e-6
    raw = pd.DataFrame({
        "open_time": times, "open": close, "high": close + spread,
        "low": close - spread, "close": close,
        "volume": rng.uniform(5.0, 50.0, bars),
    })
    return pulse.add_indicators(pulse.add_mas(raw))


def _candidate(symbol: str, decision_i: int, *, l2: float = 0.9) -> pulse.Candidate:
    return pulse.Candidate(
        symbol=symbol, decision_i=decision_i,
        decision_time=pd.Timestamp("2026-08-14T04:00:00Z"),
        p_signal=0.9, l2_score=l2, atr=1.0, atr_pct=0.01, entry_projection=100.0,
    )


def test_shipped_config_loads_and_declares_paper_only() -> None:
    config = pulse.load_config(CONFIG)
    assert config["safety"] == pulse.SAFETY
    assert config["barrier"]["atr_pct_min"] == ATR_PCT_MIN
    assert config["side"] == "short"


@pytest.mark.parametrize("mutation", [
    {"safety": {"places_orders": True}},
    {"safety": {"paper_only": False}},
    {"safety": {"writes_forward_log": True}},
    {"safety": {"auto_promote": True}},
    {"side": "long"},
    {"protocol": "something_else"},
    {"barrier": {"atr_pct_min": 0.05}},
])
def test_unsafe_config_is_refused(tmp_path: Path, mutation: dict) -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key, value in mutation.items():
        if isinstance(value, dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(pulse.PulseError):
        pulse.load_config(path)


def test_only_tip_and_two_predecessors_are_scorable() -> None:
    """Iron rule 9: a path that can produce after-the-fact signals must not exist."""
    frame = _frame()
    indices = pulse.live_decision_indices(frame, max_tip_age_bars=2)
    assert indices == [len(frame) - 3, len(frame) - 2, len(frame) - 1]

    assert pulse.live_decision_indices(frame, max_tip_age_bars=0) == [len(frame) - 1]
    with pytest.raises(pulse.PulseError):
        pulse.live_decision_indices(frame, max_tip_age_bars=-1)


def test_bars_without_a_complete_ma_bundle_are_not_scorable() -> None:
    frame = _frame()
    frame.loc[frame.index[-2], list(pulse.ALL_MA_COLS)[0]] = np.nan
    assert len(frame) - 2 not in pulse.live_decision_indices(frame, max_tip_age_bars=2)


def test_a_short_series_never_reaches_back_before_warmup() -> None:
    frame = _frame(bars=pulse.WARMUP_BARS + 1)
    assert pulse.live_decision_indices(frame, max_tip_age_bars=2) == [len(frame) - 1]


def test_stale_series_is_refused_rather_than_scored_as_current() -> None:
    frame = _frame(tip="2026-08-14T03:00:00Z")
    now = pd.Timestamp("2026-08-14T04:00:00Z")
    with pytest.raises(pulse.PulseError, match="min old"):
        pulse.check_freshness(frame, now=now, max_minutes=30, symbol="BTC_USDT_SWAP")
    age = pulse.check_freshness(frame, now=now, max_minutes=90, symbol="BTC_USDT_SWAP")
    assert age == pytest.approx(60.0)


def test_atr_floor_uses_the_frozen_contract_value() -> None:
    frame = _frame()
    ok, _, atr_pct = pulse.atr_eligible(frame, len(frame) - 1)
    assert ok and atr_pct >= ATR_PCT_MIN

    frame.loc[frame.index[-1], "atr_pct"] = ATR_PCT_MIN / 2
    assert pulse.atr_eligible(frame, len(frame) - 1)[0] is False

    frame.loc[frame.index[-1], "atr_pct"] = np.nan
    assert pulse.atr_eligible(frame, len(frame) - 1)[0] is False


def test_event_id_is_derived_from_the_bar_not_the_wall_clock() -> None:
    decision = pd.Timestamp("2026-08-14T04:00:00Z")
    first = pulse.event_id("BTC_USDT_SWAP", decision)
    assert first == pulse.event_id("BTC_USDT_SWAP", decision)
    assert first != pulse.event_id("ETH_USDT_SWAP", decision)
    assert first != pulse.event_id("BTC_USDT_SWAP", decision + pulse.BAR)


def test_gap_filter_honours_bars_already_claimed_in_the_ledger() -> None:
    candidates = [_candidate("BTC_USDT_SWAP", 300), _candidate("BTC_USDT_SWAP", 310)]
    kept = pulse.gap_filter(candidates, gap_bars=18, seen={})
    assert [c.decision_i for c in kept] == [300]

    kept = pulse.gap_filter(candidates, gap_bars=18, seen={"BTC_USDT_SWAP": 295})
    assert kept == []

    kept = pulse.gap_filter(candidates, gap_bars=18, seen={"BTC_USDT_SWAP": 280})
    assert [c.decision_i for c in kept] == [300]


def test_gap_filter_keeps_symbols_independent() -> None:
    candidates = [_candidate("BTC_USDT_SWAP", 300), _candidate("ETH_USDT_SWAP", 301)]
    kept = pulse.gap_filter(candidates, gap_bars=18, seen={})
    assert {c.symbol for c in kept} == {"BTC_USDT_SWAP", "ETH_USDT_SWAP"}


def test_ledger_round_trip_and_open_rows_stay_open_without_a_frame(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    assert pulse.read_ledger(path) == []
    rows = [{"event_id": "a", "symbol": "BTC_USDT_SWAP", "outcome": "open",
             "entry_time": "2026-08-14T04:15:00+00:00", "entry_price": 100.0, "atr": 1.0}]
    pulse.append_ledger(path, rows)
    assert pulse.read_ledger(path) == rows

    config = pulse.load_config(CONFIG)
    updated, filled = pulse.resolve_open_rows(rows, {}, barrier=config["barrier"])
    assert filled == 0 and updated[0]["outcome"] == "open"


def test_projected_entry_is_settled_to_the_real_next_open() -> None:
    """The signal fires before the entry bar exists; the ledger must not keep the guess."""
    config = pulse.load_config(CONFIG)
    frame = _frame()
    entry_i = len(frame) - 5
    entry_time = pd.Timestamp(frame["open_time"].iloc[entry_i]).isoformat()
    real_open = float(frame["open"].iloc[entry_i])
    rows = [{"event_id": "a", "symbol": "BTC_USDT_SWAP", "outcome": "open",
             "entry_time": entry_time, "entry_price": real_open * 1.5,
             "entry_price_is_projection": True, "atr": 1.0}]

    updated, _ = pulse.resolve_open_rows(rows, {"BTC_USDT_SWAP": frame},
                                         barrier=config["barrier"])
    assert updated[0]["entry_price"] == real_open
    assert updated[0]["entry_price_is_projection"] is False
    assert updated[0]["entry_price_projected"] == real_open * 1.5

    # Idempotent: a second pulse must not re-settle against a later bar.
    again, _ = pulse.resolve_open_rows(updated, {"BTC_USDT_SWAP": frame},
                                       barrier=config["barrier"])
    assert again[0]["entry_price"] == real_open


def test_a_closed_barrier_fills_price_time_and_return() -> None:
    """The open-path test above never reaches the fill branch; this one does."""
    config = pulse.load_config(CONFIG)
    frame = _frame()
    entry_i = 260
    entry_time = pd.Timestamp(frame["open_time"].iloc[entry_i]).isoformat()
    entry_price = float(frame["open"].iloc[entry_i])
    # A tiny ATR guarantees the short TP (5x below entry) is reached immediately.
    rows = [{"event_id": "a", "symbol": "BTC_USDT_SWAP", "outcome": "open",
             "entry_time": entry_time, "entry_price": entry_price,
             "entry_price_is_projection": False, "atr": entry_price * 1e-6}]

    updated, filled = pulse.resolve_open_rows(rows, {"BTC_USDT_SWAP": frame},
                                              barrier=config["barrier"])
    assert filled == 1
    row = updated[0]
    assert row["outcome"] in {"tp", "sl", "sl_ambiguous", "timeout"}
    assert isinstance(row["exit_price"], float) and row["exit_price"] > 0
    assert isinstance(row["gross_return"], float)
    assert row["exit_time"] and row["exit_offset"] >= 1


def test_resolved_rows_are_not_re_resolved(tmp_path: Path) -> None:
    config = pulse.load_config(CONFIG)
    rows = [{"event_id": "a", "symbol": "BTC_USDT_SWAP", "outcome": "tp",
             "entry_time": "2026-08-14T04:15:00+00:00", "entry_price": 100.0,
             "atr": 1.0, "exit_price": 95.0}]
    updated, filled = pulse.resolve_open_rows(rows, {"BTC_USDT_SWAP": _frame()},
                                              barrier=config["barrier"])
    assert filled == 0 and updated[0]["exit_price"] == 95.0


def test_module_cannot_reach_the_execution_layer() -> None:
    """No order path may exist in a paper pulse -- checked at the import level."""
    source = Path(pulse.__file__).read_text(encoding="utf-8")
    for forbidden in ("l4_execution", "okx_client", "executor", "forward_log"):
        assert f"import {forbidden}" not in source
        assert f"from yoyo.layers.l4_execution" not in source


def test_l2_scoring_reads_no_bar_after_the_decision() -> None:
    """The physical cut is the causality proof; mutating the future must not move it."""
    lgb = pytest.importorskip("lightgbm")
    frame = _frame()
    decision_i = len(frame) - 40

    class _Booster:
        best_iteration = 0

        @staticmethod
        def predict(rows, **_kwargs):
            return np.asarray([float(rows.iloc[0]["atr_pct"])])

    before = pulse.score_l2(_Booster(), frame, decision_i, feature_semantics="side_aligned_v1")
    poisoned = frame.copy()
    poisoned.loc[poisoned.index[decision_i + 1:], ["open", "high", "low", "close"]] = 1e6
    after = pulse.score_l2(_Booster(), poisoned, decision_i, feature_semantics="side_aligned_v1")
    assert before == after
    assert lgb is not None


def test_unknown_feature_semantics_fails_rather_than_guessing() -> None:
    frame = _frame()

    class _Booster:
        best_iteration = 0

        @staticmethod
        def predict(rows, **_kwargs):
            return np.asarray([0.5])

    with pytest.raises(ValueError, match="unknown feature_semantics"):
        pulse.score_l2(_Booster(), frame, len(frame) - 1, feature_semantics="guess")
