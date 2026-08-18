"""Safety and routing tests for the pre-holdout W10 classification backtest."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.backtest_fixed_w10_cls_preholdout import (
    ATR_ELIGIBILITY_PROTOCOL,
    ATR_PCT_MIN,
    HOLDOUT_START,
    OptimizationInterval,
    OutcomeProtocol,
    build_inner_backtest_outputs,
    build_optimization_rows,
    classify_frame,
    public_optimization_exports_summary,
    load_gold_identities,
    parse_optimization_thresholds,
    load_materialization_manifest,
    load_preholdout_csv,
    protocol_record,
    route_stop_losses,
    safe_last_decision_time,
    sample_declarations,
    summarize_trades,
    validate_candidate_date_range,
    validate_date_range,
    validate_unique_records,
)


def _csv(path: Path, times: list[str]) -> dict:
    rows = ["open_time,open,high,low,close,volume"]
    rows.extend(f"{stamp},100,101,99,100.5,1" for stamp in times)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(times),
        "max_materialized_time": max(pd.Timestamp(value) for value in times).isoformat(),
    }


def _trade(event: str, outcome: str, decision: str = "2026-04-01T00:00:00Z") -> dict:
    return {
        "event_id": event, "symbol": "BTC_USDT_SWAP", "decision_time": decision,
        "decision_i": 300, "outcome": outcome, "status": "closed",
        "causal_window_start_time": "2026-03-31T21:45:00Z",
        "causal_window_end_time": decision,
        "holdout_read": False, "future_data_in_causal_input": False,
    }


def test_time_gate_and_safe_last_decision() -> None:
    assert safe_last_decision_time() == pd.Timestamp("2026-05-03T05:45:00Z")
    assert validate_date_range("2026-05-03T00:00:00Z", "2026-05-03T05:45:00Z")[1] == safe_last_decision_time()
    with pytest.raises(ValueError, match="too late"):
        validate_date_range("2026-05-03", "2026-05-03T06:00:00Z")
    with pytest.raises(ValueError, match="holdout cutoff"):
        validate_date_range(HOLDOUT_START, HOLDOUT_START)
    assert safe_last_decision_time(horizon_bars=10) == pd.Timestamp("2026-05-03T21:15:00Z")
    with pytest.raises(ValueError, match="candidate's complete"):
        validate_candidate_date_range(
            "2026-05-03T00:00:00Z", "2026-05-03T21:30:00Z",
            OutcomeProtocol(horizon_bars=10),
        )


def test_mixed_csv_fails_before_pandas_materialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "BTC_USDT_SWAP.csv"
    declaration = _csv(path, ["2026-05-03T23:45:00Z", "2026-05-04T00:00:00Z"])
    # The sidecar itself lies below cutoff; the streamed row inspection must still catch the file.
    declaration["max_materialized_time"] = "2026-05-03T23:45:00Z"
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("pandas must not materialize a mixed CSV")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    with pytest.raises(ValueError, match="at/after holdout cutoff"):
        load_preholdout_csv(path, declaration)
    assert called is False


def test_manifest_checks_sha_rows_max_and_exact_csv_set(tmp_path: Path) -> None:
    csv_path = tmp_path / "ETH_USDT_SWAP.csv"
    entry = _csv(csv_path, ["2026-04-01T00:00:00Z", "2026-04-01T00:15:00Z"])
    manifest = tmp_path / "materialization_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "files": [entry]}), encoding="utf-8")
    declared = load_materialization_manifest(tmp_path, manifest)
    frame = load_preholdout_csv(csv_path, declared[0])
    assert len(frame) == 2
    assert frame["open_time"].max() < HOLDOUT_START

    bad = dict(entry, rows=3)
    with pytest.raises(ValueError, match="row-count mismatch"):
        load_preholdout_csv(csv_path, bad)


def test_stop_loss_routes_to_l2_and_manual_only_l1_review() -> None:
    trades = [_trade("sl-one", "sl"), _trade("amb-one", "sl_ambiguous", "2026-04-02T00:00:00Z"),
              _trade("tp-one", "tp", "2026-04-03T00:00:00Z")]
    stops, review, l2 = route_stop_losses(trades)
    assert {row["outcome"] for row in stops} == {"sl", "sl_ambiguous"}
    assert all(row["l2_outcome_negative"] is True for row in stops + l2)
    assert all(row["automatic_l1_label"] is None for row in stops)
    assert all(row["training_eligible"] is False for row in review)
    assert all(row["suggested_label"] == "REVIEW_REQUIRED" for row in review)
    assert all(row["future_review_only"] is True for row in review)
    assert all(row["future_used_in_training_image"] is False for row in review)
    assert all(row["causal_render"]["overlay"] is False for row in review)
    for row in stops + review + l2:
        assert row["holdout_read"] is False
        assert row["future_data_in_causal_input"] is False
        assert row.get("suggested_label") != "NO_SIGNAL"


def test_duplicate_identity_or_event_id_is_rejected() -> None:
    first = _trade("event-1", "sl")
    with pytest.raises(ValueError, match=r"symbol\+decision_time"):
        validate_unique_records([first, {**first, "event_id": "event-2"}])
    with pytest.raises(ValueError, match="duplicate event_id"):
        validate_unique_records([first, {**first, "symbol": "ETH_USDT_SWAP",
                                         "decision_time": "2026-04-02T00:00:00Z"}])


def test_summary_basics_and_protocol_markers() -> None:
    trades = [
        {**_trade("a", "tp"), "gross_ret": 0.01, "net_maker": 0.0094, "net_taker": 0.009},
        {**_trade("b", "sl", "2026-04-02T00:00:00Z"), "gross_ret": -0.004,
         "net_maker": -0.0046, "net_taker": -0.005},
        {**_trade("c", "timeout", "2026-04-03T00:00:00Z"), "status": "skip"},
    ]
    summary = summarize_trades(trades)
    assert summary["n_total"] == 3
    assert summary["n_closed"] == 2
    assert summary["n_skip"] == 1
    assert summary["outcomes"] == {"tp": 1, "sl": 1}
    candidate = OutcomeProtocol(threshold=0.4, tp_atr=4.0, sl_atr=1.5, horizon_bars=48)
    protocol = protocol_record(start=pd.Timestamp("2026-04-01T00:00:00Z"),
                               end=pd.Timestamp("2026-04-02T00:00:00Z"), candidate=candidate)
    assert protocol["window_bars"] == 10
    assert protocol["renderer_overlay"] is False
    assert protocol["imgsz"] == 960
    assert protocol["entry"] == "next_open"
    assert protocol["baseline_parameters"] == {
        "threshold": 0.5, "tp_atr": 5.0, "sl_atr": 2.0, "horizon_bars": 72,
    }
    assert protocol["candidate_parameters"] == {
        "threshold": 0.4, "tp_atr": 4.0, "sl_atr": 1.5, "horizon_bars": 48,
    }
    assert protocol["parameter_status"] == "preholdout_candidate_not_promoted"
    assert protocol["min_gap_bars"] == 18
    assert protocol["atr_pct_min"] == ATR_PCT_MIN
    assert protocol["atr_eligibility_protocol"] == ATR_ELIGIBILITY_PROTOCOL
    assert protocol["holdout_read"] is False
    assert protocol["future_data_in_causal_input"] is False


def _optimization_frame(start: str, bars: int = 40) -> pd.DataFrame:
    times = pd.date_range(start, periods=bars, freq="15min", tz="UTC")
    return pd.DataFrame({
        "open_time": times,
        "open": np.arange(bars, dtype=float) + 100.0,
        "high": np.arange(bars, dtype=float) + 101.0,
        "low": np.arange(bars, dtype=float) + 99.0,
        "close": np.arange(bars, dtype=float) + 100.5,
        "atr14": np.full(bars, 1.25),
        "atr_pct": np.full(bars, 0.01),
    })


def _prediction(event: str, symbol: str, frame: pd.DataFrame, index: int, score: float) -> dict:
    decision = pd.Timestamp(frame["open_time"].iloc[index]).isoformat()
    return {
        "event_id": event, "symbol": symbol, "decision_i": index,
        "decision_time": decision, "p_signal": score,
        "pred": "SIGNAL" if score >= 0.5 else "NO_SIGNAL",
        "causal_window_start_time": pd.Timestamp(frame["open_time"].iloc[max(0, index-9)]).isoformat(),
    }


def test_optimization_export_is_physical_interval_safe_and_threshold_independent() -> None:
    symbol = "BTC_USDT_SWAP"
    frame = _optimization_frame("2026-03-30T23:00:00Z")
    inner_low_score = _prediction("a", symbol, frame, 4, 0.10)
    # This decision is inside inner, but its 4-bar path crosses the interval end.
    cropped = _prediction("b", symbol, frame, 9, 0.90)
    intervals = (
        OptimizationInterval.from_values(
            "inner", "2026-03-30T00:00:00Z", "2026-03-31T01:30:00Z"),
        OptimizationInterval.from_values(
            "sealed", "2026-04-02T00:00:00Z", "2026-05-04T00:00:00Z"),
    )
    rows, stats = build_optimization_rows(
        [inner_low_score, cropped], {symbol: frame}, intervals, max_horizon=4,
    )
    assert [row["event_id"] for row in rows["inner"]] == [
        # Export identity deliberately ignores input event IDs and score thresholds.
        rows["inner"][0]["event_id"]
    ]
    assert rows["sealed"] == []
    assert rows["inner"][0]["p_signal"] == 0.10
    assert rows["inner"][0]["future_path"][0]["time"] == frame["open_time"].iloc[5].isoformat()
    assert rows["inner"][0]["entry_price"] == frame["open"].iloc[5]
    assert len(rows["inner"][0]["future_path"]) == 4
    assert rows["inner"][0]["future_review_only"] is True
    assert rows["inner"][0]["future_data_in_causal_input"] is False
    assert rows["inner"][0]["holdout_read"] is False
    assert stats["interval_end_cropped"] == 1


def test_optimization_export_physically_separates_inner_purge_and_sealed() -> None:
    symbol = "ETH_USDT_SWAP"
    frame = _optimization_frame("2026-03-31T06:30:00Z", bars=800)
    inner = _prediction("inner", symbol, frame, 1, 0.8)
    purge = _prediction("purge", symbol, frame, 100, 0.8)
    sealed_i = int((pd.Timestamp("2026-04-02T00:00:00Z") - frame["open_time"].iloc[0]) / pd.Timedelta(minutes=15))
    sealed = _prediction("sealed", symbol, frame, sealed_i, 0.8)
    rows, stats = build_optimization_rows(
        [inner, purge, sealed], {symbol: frame},
        (
            OptimizationInterval.from_values("inner", "2026-01-01", "2026-03-31T07:30:00Z"),
            OptimizationInterval.from_values("sealed", "2026-04-02", "2026-05-04"),
        ),
        max_horizon=2,
    )
    assert len(rows["inner"]) == 1
    assert len(rows["sealed"]) == 1
    assert stats["outside_intervals"] == 1
    assert {row["event_id"] for row in rows["inner"]}.isdisjoint(
        {row["event_id"] for row in rows["sealed"]}
    )


def test_sealed_predictions_are_export_only_not_resolved_or_summarized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.backtest_fixed_w10_cls_preholdout as backtest

    symbol = "ETH_USDT_SWAP"
    frame = _optimization_frame("2026-03-31T06:00:00Z", bars=240)
    inner = _prediction("inner", symbol, frame, 3, 0.80)
    sealed_i = int((pd.Timestamp("2026-04-02T00:00:00Z") - frame["open_time"].iloc[0]) / pd.Timedelta(minutes=15))
    sealed = _prediction("sealed-sl-if-leaked", symbol, frame, sealed_i, 0.99)
    candidate = OutcomeProtocol(threshold=0.5, horizon_bars=2)
    inner_end = pd.Timestamp("2026-03-31T07:30:00Z")

    optimization_rows, _ = build_optimization_rows(
        [inner, sealed], {symbol: frame},
        (
            OptimizationInterval.from_values("inner", "2026-01-01", inner_end),
            OptimizationInterval.from_values("sealed", "2026-04-02", "2026-05-04"),
        ),
        max_horizon=2,
    )
    assert [row["decision_time"] for row in optimization_rows["sealed"]] == [sealed["decision_time"]]

    resolved: list[str] = []

    def fake_outcome_for(_frame: pd.DataFrame, decision_i: int, _candidate: OutcomeProtocol) -> dict:
        decision_time = pd.Timestamp(_frame["open_time"].iloc[decision_i])
        resolved.append(decision_time.isoformat())
        if decision_time >= pd.Timestamp("2026-04-02T00:00:00Z"):
            raise AssertionError("sealed resolver leak")
        return {
            "status": "closed", "outcome": "tp", "entry_i": decision_i + 1,
            "gross_ret": 0.01, "net_maker": 0.0094, "net_taker": 0.009,
        }

    monkeypatch.setattr(backtest, "outcome_for", fake_outcome_for)
    outputs = build_inner_backtest_outputs([inner, sealed], {symbol: frame}, candidate, inner_end)
    assert [row["decision_time"] for row in outputs["predictions"]] == [inner["decision_time"]]
    assert [row["decision_time"] for row in outputs["signals_raw"]] == [inner["decision_time"]]
    assert [row["decision_time"] for row in outputs["signals_dedup"]] == [inner["decision_time"]]
    assert outputs["stop_losses"] == []
    assert outputs["l1_review_queue"] == []
    assert outputs["l2_outcome_negatives"] == []
    assert resolved == [inner["decision_time"], inner["decision_time"]]

    raw_summary = summarize_trades(outputs["signals_raw"])
    dedup_summary = summarize_trades(outputs["signals_dedup"])
    assert raw_summary["n_total"] == 1
    assert dedup_summary["n_total"] == 1
    assert raw_summary["outcomes"] == {"tp": 1}
    public_exports = public_optimization_exports_summary(optimization_rows)
    assert public_exports["sealed"]["file"] == "optimization_sealed.jsonl"
    assert "rows" not in public_exports["sealed"]
    assert "selected_by_threshold" not in public_exports
    assert "p_signal" not in json.dumps({
        "raw": raw_summary,
        "deduped": dedup_summary,
        "optimization_exports": public_exports,
    }, sort_keys=True)


def test_optimization_export_rejects_holdout_interval() -> None:
    symbol = "SOL_USDT_SWAP"
    frame = _optimization_frame("2026-05-03T22:00:00Z")
    prediction = _prediction("unsafe", symbol, frame, 4, 0.9)
    unsafe = OptimizationInterval("unsafe", pd.Timestamp("2026-05-03T00:00:00Z"),
                                  pd.Timestamp("2026-05-05T00:00:00Z"))
    with pytest.raises(ValueError, match="reaches holdout"):
        build_optimization_rows([prediction], {symbol: frame}, [unsafe], max_horizon=4)


def test_symbol_sample_is_seed_hash_only_and_zero_means_all() -> None:
    declarations = [
        {"path": f"{symbol}.csv", "symbol": symbol, "rows": rows}
        for symbol, rows in (("BTC", 1), ("ETH", 999), ("SOL", 5))
    ]
    sampled, metadata = sample_declarations(declarations, 2, "seed")
    expected = sorted(
        (hashlib.sha256(f"seed\0{row['symbol']}".encode()).hexdigest(), row["symbol"])
        for row in declarations
    )[:2]
    assert [row["symbol"] for row in sampled] == [symbol for _, symbol in expected]
    assert metadata["population_symbols"] == 3
    assert metadata["sampled_symbols"] == 2
    assert "rows" not in metadata["algorithm"]
    all_rows, all_metadata = sample_declarations(declarations, 0, "seed")
    assert len(all_rows) == 3
    assert all_metadata["sampled_symbols"] == 3


def test_gold_manifest_marks_exact_safe_signal_identity(tmp_path: Path) -> None:
    symbol = "BTC_USDT_SWAP"
    frame = _optimization_frame("2026-03-01T00:00:00Z")
    prediction = _prediction("event", symbol, frame, 10, 0.1)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "label": "SIGNAL", "symbol": symbol,
        "decision_time": prediction["decision_time"], "holdout_read": False,
        "future_used_in_model_input": False,
    }) + "\n" + json.dumps({
        "label": "NO_SIGNAL", "symbol": symbol,
        "decision_time": "2026-03-02T00:00:00Z",
    }) + "\n", encoding="utf-8")
    identities, metadata = load_gold_identities(manifest)
    rows, stats = build_optimization_rows(
        [prediction], {symbol: frame},
        [OptimizationInterval.from_values("inner", "2026-03-01", "2026-04-01")],
        max_horizon=4, gold_identities=identities,
    )
    assert rows["inner"][0]["gold"] is True
    assert stats["gold_matches"] == 1
    assert metadata["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_gold_manifest_rejects_unsafe_or_duplicate_signals(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    row = {"label": "SIGNAL", "symbol": "BTC", "decision_time": "2026-03-01T00:00:00Z",
           "holdout_read": False, "future_used_in_model_input": False}
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate gold identity"):
        load_gold_identities(manifest)
    manifest.write_text(json.dumps({**row, "decision_time": "2026-05-04T00:00:00Z"}) + "\n",
                        encoding="utf-8")
    with pytest.raises(ValueError, match="reaches holdout"):
        load_gold_identities(manifest)


def test_optimization_min_score_drops_low_non_gold_but_keeps_gold() -> None:
    symbol = "BTC_USDT_SWAP"
    frame = _optimization_frame("2026-03-01T00:00:00Z")
    low = _prediction("low", symbol, frame, 10, 0.10)
    gold = _prediction("gold", symbol, frame, 20, 0.05)
    gold_identity = (symbol, pd.Timestamp(gold["decision_time"]).isoformat())
    rows, stats = build_optimization_rows(
        [low, gold], {symbol: frame},
        [OptimizationInterval.from_values("inner", "2026-03-01", "2026-04-01")],
        max_horizon=4, gold_identities={gold_identity}, min_score=0.25,
        thresholds=(0.25, 0.5),
    )
    assert len(rows["inner"]) == 1
    assert rows["inner"][0]["p_signal"] == 0.05
    assert rows["inner"][0]["gold"] is True
    assert stats["below_min_score"] == 1
    assert stats["gold_matches"] == 1
    assert stats["gold_force_included"] == 1
    assert rows["inner"][0]["selected_at_thresholds"] == []


def test_parse_optimization_thresholds_and_minimum_consistency() -> None:
    assert parse_optimization_thresholds("0.5,0.2,0.25") == (0.2, 0.25, 0.5)
    with pytest.raises(ValueError, match="unique"):
        parse_optimization_thresholds("0.2,0.2")
    with pytest.raises(ValueError, match="non-empty"):
        parse_optimization_thresholds("")
    with pytest.raises(ValueError, match="must equal"):
        build_optimization_rows([], {}, [OptimizationInterval.from_values(
            "inner", "2026-03-01", "2026-04-01")], max_horizon=4,
            min_score=0.2, thresholds=(0.25, 0.5))


def _optimizer_gap_ids(rows: list[dict], threshold: float) -> set[str]:
    eligible = sorted(
        (row for row in rows if row["p_signal"] >= threshold),
        key=lambda row: (row["symbol"], pd.Timestamp(row["decision_time"]), row["event_id"]),
    )
    selected, last = set(), {}
    for row in eligible:
        stamp = pd.Timestamp(row["decision_time"])
        if row["symbol"] not in last or stamp - last[row["symbol"]] >= pd.Timedelta(minutes=270):
            selected.add(row["event_id"])
            last[row["symbol"]] = stamp
    return selected


def test_threshold_union_preserves_each_optimizer_gap_result() -> None:
    symbol = "BTC_USDT_SWAP"
    frame = _optimization_frame("2026-03-01T00:00:00Z", bars=80)
    predictions = [
        _prediction("first-low", symbol, frame, 10, 0.30),
        # Within 18 bars of first; suppressed at 0.2, selected at 0.5.
        _prediction("later-high", symbol, frame, 20, 0.60),
        _prediction("later-low", symbol, frame, 40, 0.35),
    ]
    rows, stats = build_optimization_rows(
        predictions, {symbol: frame},
        [OptimizationInterval.from_values("inner", "2026-03-01", "2026-04-01")],
        max_horizon=4, min_score=0.2, thresholds=(0.2, 0.5),
    )
    exported = rows["inner"]
    exported_by_time = {row["decision_time"]: row for row in exported}
    assert predictions[1]["decision_time"] in exported_by_time
    assert exported_by_time[predictions[1]["decision_time"]]["selected_at_thresholds"] == [0.5]
    # Compare exact stable identities using the full-population algorithm.
    full_rows = [{**row, "event_id": exported_by_time[row["decision_time"]]["event_id"]}
                 for row in predictions]
    for threshold in (0.2, 0.5):
        assert _optimizer_gap_ids(exported, threshold) == _optimizer_gap_ids(full_rows, threshold)
    assert stats["selected_by_threshold"] == {"0.2": 2, "0.5": 1}
    assert stats["threshold_union_selected"] == 3


def test_atr_floor_precedes_threshold_gap_and_is_exported_with_contract() -> None:
    symbol = "BTC_USDT_SWAP"
    frame = _optimization_frame("2026-03-01T00:00:00Z", bars=80)
    frame.loc[10, "atr_pct"] = ATR_PCT_MIN - 1e-12
    frame.loc[20, "atr_pct"] = ATR_PCT_MIN
    predictions = [
        _prediction("low-atr-early", symbol, frame, 10, 0.9),
        _prediction("valid-later", symbol, frame, 20, 0.9),
    ]
    rows, stats = build_optimization_rows(
        predictions, {symbol: frame},
        [OptimizationInterval.from_values("inner", "2026-03-01", "2026-04-01")],
        max_horizon=4, min_score=0.5, thresholds=(0.5,),
    )
    assert len(rows["inner"]) == 1
    assert rows["inner"][0]["decision_time"] == predictions[1]["decision_time"]
    assert rows["inner"][0]["atr_pct"] == ATR_PCT_MIN
    assert rows["inner"][0]["atr_pct_min"] == ATR_PCT_MIN
    assert rows["inner"][0]["atr_eligibility_protocol"] == ATR_ELIGIBILITY_PROTOCOL
    assert stats["atr_floor_excluded"] == 1
    assert stats["selected_by_threshold"] == {"0.5": 1}


def test_inner_backtest_atr_floor_cannot_suppress_later_eligible_signal() -> None:
    symbol = "BTC_USDT_SWAP"
    frame = _optimization_frame("2026-03-01T00:00:00Z", bars=80)
    frame.loc[10, "atr_pct"] = ATR_PCT_MIN / 2
    frame.loc[20, "atr_pct"] = ATR_PCT_MIN
    early = _prediction("low-atr-early", symbol, frame, 10, 0.9)
    later = _prediction("valid-later", symbol, frame, 20, 0.9)
    outputs = build_inner_backtest_outputs(
        [early, later], {symbol: frame}, OutcomeProtocol(horizon_bars=4),
        pd.Timestamp("2026-04-01T00:00:00Z"),
    )
    assert outputs["atr_floor_excluded"] == 1
    assert [row["decision_time"] for row in outputs["signals_raw"]] == [later["decision_time"]]
    assert [row["decision_time"] for row in outputs["signals_dedup"]] == [later["decision_time"]]
    assert all(row.get("outcome") != "atr_floor" for row in outputs["signals_raw"])


class _FakeClassifier:
    def model(self, tensors):
        import torch

        signal_logit = tensors[:, 0] / 10.0
        return torch.stack((torch.zeros_like(signal_logit), signal_logit), dim=1)


def _classification_frame(bars: int = 300) -> pd.DataFrame:
    times = pd.date_range("2026-01-01T00:00:00Z", periods=bars, freq="15min", tz="UTC")
    return pd.DataFrame({
        "open_time": times,
        "open": np.arange(bars, dtype=float) + 100.0,
        "high": np.arange(bars, dtype=float) + 101.0,
        "low": np.arange(bars, dtype=float) + 99.0,
        "close": np.arange(bars, dtype=float),
        "volume": np.ones(bars, dtype=float),
    })


def test_render_workers_preserve_prediction_order_skips_and_scores(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    assert cv2.COLOR_BGR2RGB is not None
    import tools.backtest_fixed_w10_cls_preholdout as backtest

    frame = _classification_frame()
    indices = [240, 241, 242, 243, 244, 245]
    skipped_i = 243
    rendered: list[tuple[int, str]] = []

    def fake_render(window: pd.DataFrame, *, y_pad_frac: float, overlay: bool):
        assert y_pad_frac == backtest.Y_PAD_FRAC
        assert overlay is False
        decision_i = int(window["close"].iloc[-1])
        rendered.append((decision_i, hashlib.sha256(window.to_csv(index=False).encode()).hexdigest()))
        if decision_i == skipped_i:
            raise backtest.SpanError("bad span")
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        image[:, :, 0] = decision_i % 256
        return image, {"decision_i": decision_i}

    def transform(image):
        array = np.asarray(image)
        return torch.tensor([float(array[0, 0, 2])], dtype=torch.float32)

    monkeypatch.setattr(backtest, "render_w10", fake_render)
    baseline_rows, baseline_skips = classify_frame(
        _FakeClassifier(), transform, frame, "BTC_USDT_SWAP", indices,
        device="cpu", signal_idx=1, threshold=0.5, batch_size=2, render_workers=0,
    )
    baseline_rendered = list(rendered)
    rendered.clear()
    one_worker_rows, one_worker_skips = classify_frame(
        _FakeClassifier(), transform, frame, "BTC_USDT_SWAP", indices,
        device="cpu", signal_idx=1, threshold=0.5, batch_size=2, render_workers=1,
    )
    one_worker_rendered = list(rendered)
    rendered.clear()
    worker_rows, worker_skips = classify_frame(
        _FakeClassifier(), transform, frame, "BTC_USDT_SWAP", indices,
        device="cpu", signal_idx=1, threshold=0.5, batch_size=2, render_workers=3,
    )
    worker_rendered = list(rendered)
    captured = capsys.readouterr()

    assert baseline_skips == one_worker_skips == worker_skips == 1
    assert [row["decision_i"] for row in baseline_rows] == [240, 241, 242, 244, 245]
    assert [row["decision_i"] for row in one_worker_rows] == [240, 241, 242, 244, 245]
    assert [row["decision_i"] for row in worker_rows] == [240, 241, 242, 244, 245]
    assert baseline_rows == one_worker_rows
    assert baseline_rows == worker_rows
    assert [item[0] for item in baseline_rendered] == indices
    assert sorted(one_worker_rendered) == sorted(baseline_rendered)
    assert sorted(worker_rendered) == sorted(baseline_rendered)
    assert captured.out == ""
    assert "[BTC_USDT_SWAP] predictions 6/6 elapsed" in captured.err
