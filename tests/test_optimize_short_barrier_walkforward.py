from __future__ import annotations

import json

import pandas as pd
import pytest

import tools.optimize_short_barrier_walkforward as optimizer
from tools.optimize_short_barrier_walkforward import (
    BAR_DURATION,
    BASELINE,
    HOLDOUT_START,
    Fold,
    Guards,
    SearchConfig,
    build_grid,
    consume_sealed_once,
    evaluate_fold,
    gap_deduplicate,
    load_single_variable_grid,
    main,
    select_config,
    validate_candidates,
)
from yoyo.contracts.costs import SWAP_MAKER


def _path(decision: str, closes: list[float], *, high: float = 100.1, low: float = 99.9):
    start = pd.Timestamp(decision) + BAR_DURATION
    return [{"time": (start + index * BAR_DURATION).isoformat(), "open": 100.0,
             "high": high, "low": low, "close": close}
            for index, close in enumerate(closes)]


def _row(event_id: str, decision: str, score: float, closes: list[float], **extra):
    return {"event_id": event_id, "symbol": extra.pop("symbol", "BTC-USDT-SWAP"),
            "decision_time": decision, "p_signal": score, "entry_price": 100.0,
            "atr14": 1.0, "atr_pct": extra.pop("atr_pct", optimizer.ATR_PCT_MIN),
            "atr_pct_min": extra.pop("atr_pct_min", optimizer.ATR_PCT_MIN),
            "atr_eligibility_protocol": extra.pop(
                "atr_eligibility_protocol", optimizer.ATR_ELIGIBILITY_PROTOCOL),
            "future_review_only": True,
            "causal_metadata": {"visible_end_time": decision},
            "future_path": _path(decision, closes, high=extra.pop("high", 100.1),
                                 low=extra.pop("low", 99.9)), **extra}


def _fold(name="f1", start="2026-01-01T00:00:00Z", end="2026-01-03T00:00:00Z"):
    return Fold(name, pd.Timestamp("2025-01-01T00:00:00Z"),
                pd.Timestamp(start) - 162 * BAR_DURATION, pd.Timestamp(start), pd.Timestamp(end))


def _validated(rows, horizon=72):
    return validate_candidates(rows, max_horizon=horizon)


def test_baseline_is_required_but_another_config_can_be_selected():
    with pytest.raises(ValueError, match="baseline"):
        build_grid([0.6], [5], [2], [72])
    grid = build_grid([0.5, 0.6], [5], [2], [72])
    rows = _validated([
        _row("bad", "2026-01-01T01:00:00Z", .55, [101.0] * 72),
        _row("good", "2026-01-01T08:00:00Z", .65, [99.0] * 72),
    ])
    selected, _ = select_config(rows, [_fold()], grid, Guards(1, 100, 100))
    assert selected["selected_config"]["threshold"] == .6
    assert tuple(BASELINE) == (.5, 5.0, 2.0, 72)


def test_single_variable_grid_accepts_one_field_arms_and_rejects_multi_field_change():
    raw = {
        "stage": "threshold-1", "variable_under_test": "threshold",
        "locked_config": {"threshold": .5, "tp_atr": 5, "sl_atr": 2, "horizon": 72},
        "configs": [
            {"threshold": .5, "tp_atr": 5, "sl_atr": 2, "horizon": 72},
            {"threshold": .6, "tp_atr": 5, "sl_atr": 2, "horizon": 72},
        ],
    }
    grid, discipline = load_single_variable_grid(raw)
    assert [config.threshold for config in grid] == [.5, .6]
    assert discipline["experimental_discipline"] == "single_variable"
    raw["configs"].append({"threshold": .6, "tp_atr": 4, "sl_atr": 2, "horizon": 72})
    with pytest.raises(ValueError, match="outside variable_under_test"):
        load_single_variable_grid(raw)


def test_high_total_arm_is_rejected_by_per_fold_min_trades():
    folds = [_fold("one", "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"),
             _fold("two", "2026-01-04T00:00:00Z", "2026-01-06T00:00:00Z")]
    # High threshold catches a lucrative event in fold one but has no fold-two
    # trade.  It must not win merely because its aggregate return is highest.
    rows = _validated([
        _row("a", "2026-01-01T01:00:00Z", .9, [90.0] * 72),
        _row("b", "2026-01-01T08:00:00Z", .55, [99.0] * 72),
        _row("c", "2026-01-04T01:00:00Z", .55, [99.0] * 72),
    ])
    grid = build_grid([.5, .8], [5], [2], [72])
    selected, results = select_config(rows, folds, grid, Guards(1, 100, 100))
    assert selected["selected_config"]["threshold"] == .5
    assert any(item["config"]["threshold"] == .8 and "min_trades" in item["guard_violations"]
               for item in results)


def test_equal_pooled_mean_uses_worst_fold_stability_tie_break():
    folds = [_fold("one", "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"),
             _fold("two", "2026-01-04T00:00:00Z", "2026-01-06T00:00:00Z")]
    rows = _validated([
        _row("one-high", "2026-01-01T01:00:00Z", .9, [99.0] * 72),
        _row("one-low", "2026-01-01T08:00:00Z", .55, [97.0] * 72),
        _row("two-high", "2026-01-04T01:00:00Z", .9, [99.0] * 72),
        _row("two-low", "2026-01-04T08:00:00Z", .55, [101.0] * 72),
    ])
    selected, _ = select_config(
        rows, folds, build_grid([.5, .8], [5], [2], [72]), Guards(1, 100, 100)
    )
    assert selected["selected_config"]["threshold"] == .8


def test_time_leakage_is_rejected_and_cross_fold_label_is_cropped(monkeypatch):
    leaked = _row("leak", "2026-01-01T00:00:00Z", .8, [100.0] * 72)
    leaked["causal_metadata"]["visible_end_time"] = "2026-01-01T00:15:00Z"
    with pytest.raises(ValueError, match="looks beyond"):
        _validated([leaked])

    row = _validated([_row("cross", "2026-01-01T12:00:00Z", .8, [100.0] * 72)])[0]
    monkeypatch.setattr(optimizer, "_resolve",
                        lambda *_args, **_kwargs: pytest.fail("cropped label was resolved"))
    result = evaluate_fold([row], _fold(end="2026-01-02T00:00:00Z"),
                           SearchConfig(.5, 5, 2, 72), Guards(1, 100, 100))
    assert result["n_label_end_cropped"] == 1
    assert result["n_usable_candidates"] == 0
    assert result["gold_count"] == 0


def test_label_crop_happens_before_threshold_and_gap(monkeypatch):
    # The later event is within the same-symbol gap and its label crosses the
    # fold.  Cropping all exposures first must report it even though gap dedup
    # would otherwise hide it from outcome processing.
    rows = _validated([
        _row("usable", "2026-01-01T12:00:00Z", .9, [99.0] * 72),
        _row("cropped", "2026-01-01T13:00:00Z", .9, [100.0] * 72),
    ])
    fold = _fold(end="2026-01-02T07:00:00Z")
    resolved: list[str] = []
    real_resolve = optimizer._resolve

    def tracking_resolve(row, config):
        resolved.append(row["event_id"])
        return real_resolve(row, config)

    monkeypatch.setattr(optimizer, "_resolve", tracking_resolve)
    result = evaluate_fold(rows, fold, SearchConfig(.5, 5, 2, 72), Guards(1, 100, 100))
    assert result["n_label_end_cropped"] == 1
    assert resolved == ["usable"]


def test_holdout_decision_or_future_bar_is_rejected():
    with pytest.raises(ValueError, match="decision reaches holdout"):
        _validated([_row("decision", HOLDOUT_START.isoformat(), .8, [100.0] * 72)])
    row = _row("future", "2026-05-03T06:00:00Z", .8, [100.0] * 72)
    with pytest.raises(ValueError, match="future_path reaches holdout"):
        _validated([row])


def test_cost_is_deducted_exactly_once():
    row = _validated([_row("cost", "2026-01-01T01:00:00Z", .8, [99.0] * 72)])[0]
    result = evaluate_fold([row], _fold(), SearchConfig(.5, 5, 2, 72), Guards(1, 100, 100))
    # Timeout gross short return is 1%; maker route is one round-trip deduction.
    assert result["mean_net_maker"] == pytest.approx(.01 - SWAP_MAKER)
    assert result["outcomes"] == {"timeout": 1}
    assert result["win_rate_maker"] == 1.0
    assert result["worst_trade_maker"] == pytest.approx(.01 - SWAP_MAKER)
    assert result["max_drawdown_maker"] == 0.0
    assert result["profit_factor_maker"] is None


def test_trigger_density_is_per_exposed_symbol_day_not_fold_day():
    fold = _fold()
    config = SearchConfig(.5, 5, 2, 72)
    one_symbol = _validated([
        _row("btc-1", "2026-01-01T01:00:00Z", .8, [99.0] * 72),
        _row("btc-2", "2026-01-01T06:00:00Z", .8, [99.0] * 72),
        _row("btc-3", "2026-01-01T11:00:00Z", .8, [99.0] * 72),
    ])
    duplicated_pattern = _validated([
        *one_symbol,
        _row("eth-1", "2026-01-01T01:00:00Z", .8, [99.0] * 72,
             symbol="ETH-USDT-SWAP"),
        _row("eth-2", "2026-01-01T06:00:00Z", .8, [99.0] * 72,
             symbol="ETH-USDT-SWAP"),
        _row("eth-3", "2026-01-01T11:00:00Z", .8, [99.0] * 72,
             symbol="ETH-USDT-SWAP"),
    ])
    doubled_same_symbol = _validated([
        *one_symbol,
        _row("btc-4", "2026-01-01T16:00:00Z", .8, [99.0] * 72),
        _row("btc-5", "2026-01-01T21:00:00Z", .8, [99.0] * 72),
        _row("btc-6", "2026-01-02T02:00:00Z", .8, [99.0] * 72),
    ])

    base = evaluate_fold(one_symbol, fold, config, Guards(1, 100, 2.0))
    cross_symbol = evaluate_fold(duplicated_pattern, fold, config, Guards(1, 100, 2.0))
    same_symbol = evaluate_fold(doubled_same_symbol, fold, config, Guards(1, 100, 2.0))

    assert base["n_trades"] == 3
    assert cross_symbol["n_trades"] == 6
    assert same_symbol["n_trades"] == 6
    assert base["n_exposure_symbols"] == 1
    assert cross_symbol["n_exposure_symbols"] == 2
    assert same_symbol["n_exposure_symbols"] == 1
    assert base["trigger_density_per_symbol_day"] == pytest.approx(1.5)
    assert cross_symbol["trigger_density_per_symbol_day"] == pytest.approx(1.5)
    assert same_symbol["trigger_density_per_symbol_day"] == pytest.approx(3.0)
    assert base["eligible"] is True
    assert cross_symbol["eligible"] is True
    assert same_symbol["eligible"] is False
    assert "max_trigger_density" in same_symbol["guard_violations"]


def test_low_score_gold_forced_rows_count_as_symbol_exposure_before_threshold():
    rows = _validated([
        _row("trade", "2026-01-01T01:00:00Z", .8, [99.0] * 72),
        _row("gold-low", "2026-01-01T01:00:00Z", .1, [99.0] * 72,
             symbol="ETH-USDT-SWAP", gold=True),
    ])

    result = evaluate_fold(rows, _fold(), SearchConfig(.5, 5, 2, 72), Guards(1, 100, 100))

    assert result["n_trades"] == 1
    assert result["n_exposure_symbols"] == 2
    assert result["candidate_exposure_symbol_days"] == pytest.approx(4.0)
    assert result["trigger_density_per_symbol_day"] == pytest.approx(0.25)
    assert result["gold_count"] == 1
    assert result["gold_recall"] == 0.0


def test_aggregate_stability_guards_make_arm_ineligible():
    folds = [_fold("one", "2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z"),
             _fold("two", "2026-01-04T00:00:00Z", "2026-01-06T00:00:00Z")]
    rows = _validated([
        _row("one", "2026-01-01T01:00:00Z", .9, [99.0] * 72),
        _row("two", "2026-01-04T01:00:00Z", .9, [101.0] * 72),
    ])
    with pytest.raises(ValueError, match="no parameter"):
        select_config(rows, folds, build_grid([.5], [5], [2], [72]),
                      Guards(1, 100, 100, min_positive_folds=2,
                             min_worst_fold_maker_bp=0))


def test_threshold_precedes_same_symbol_gap_dedup():
    rows = _validated([
        _row("low", "2026-01-01T01:00:00Z", .55, [100.0] * 72),
        _row("high", "2026-01-01T01:15:00Z", .85, [100.0] * 72),
    ])
    assert [row["event_id"] for row in gap_deduplicate(rows, .5)] == ["low"]
    assert [row["event_id"] for row in gap_deduplicate(rows, .8)] == ["high"]


@pytest.mark.parametrize("mutation,match", [
    ({"atr_pct": optimizer.ATR_PCT_MIN / 2}, "frozen floor"),
    ({"atr_pct": float("nan")}, "finite"),
    ({"atr_pct_min": 0.0}, "does not match frozen floor"),
    ({"atr_eligibility_protocol": "legacy"}, "protocol mismatch"),
])
def test_candidate_atr_contract_rejects_polluted_or_forged_rows(mutation, match):
    row = _row("bad-atr", "2026-01-01T01:00:00Z", .8, [100.0] * 72)
    row.update(mutation)
    with pytest.raises(ValueError, match=match):
        _validated([row])


def test_candidate_atr_exact_boundary_is_allowed_and_preregistered():
    row = _validated([_row(
        "boundary", "2026-01-01T01:00:00Z", .8, [100.0] * 72,
        atr_pct=optimizer.ATR_PCT_MIN,
    )])[0]
    assert row["atr_pct"] == optimizer.ATR_PCT_MIN
    payload = optimizer._protocol(
        [SearchConfig(*BASELINE)], Guards(1, 100, 100), optimizer.DEFAULT_FOLD_DOCUMENT,
        {"experimental_discipline": "test"},
    )
    assert payload["atr_pct_min"] == optimizer.ATR_PCT_MIN
    assert payload["atr_eligibility_protocol"] == optimizer.ATR_ELIGIBILITY_PROTOCOL


def test_sealed_source_can_only_be_consumed_once(tmp_path):
    source = tmp_path / "sealed.jsonl"
    source.write_text(json.dumps({"event_id": "x"}) + "\n", encoding="utf-8")
    registry = tmp_path / "global.json"
    sealed = {"name": "april", "start": pd.Timestamp("2026-04-02T00:00:00Z"),
              "end": pd.Timestamp("2026-05-04T00:00:00Z")}
    consume_sealed_once(registry, source_path=source, sealed=sealed,
                        result_path=tmp_path / "first.json", payload={"ok": True})
    copied_source = tmp_path / "copied-sealed.jsonl"
    copied_source.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="already been consumed"):
        consume_sealed_once(registry, source_path=copied_source, sealed=sealed,
                            result_path=tmp_path / "second.json", payload={"ok": True})


def test_sealed_claim_blocks_retry_when_artifact_write_fails(tmp_path, monkeypatch):
    source = tmp_path / "sealed.jsonl"
    source.write_text(json.dumps({"event_id": "x"}) + "\n", encoding="utf-8")
    registry = tmp_path / "global.json"
    result = tmp_path / "sealed_eval.json"
    sealed = {"name": "april", "start": pd.Timestamp("2026-04-02T00:00:00Z"),
              "end": pd.Timestamp("2026-05-04T00:00:00Z")}

    def fail_write_new_json(path, payload):
        raise OSError("simulated artifact write failure")

    monkeypatch.setattr(optimizer, "write_new_json", fail_write_new_json)
    with pytest.raises(OSError, match="simulated artifact"):
        consume_sealed_once(registry, source_path=source, sealed=sealed,
                            result_path=result, payload={"ok": True})
    assert not result.exists()
    entries = json.loads(registry.read_text(encoding="utf-8"))["sealed_consumptions"]
    assert len(entries) == 1
    assert next(iter(entries.values()))["status"] == "pending"

    monkeypatch.undo()
    with pytest.raises(ValueError, match="already been consumed"):
        consume_sealed_once(registry, source_path=source, sealed=sealed,
                            result_path=result, payload={"ok": True})
    assert not result.exists()


def test_sealed_claim_blocks_retry_when_complete_registry_write_fails(tmp_path, monkeypatch):
    source = tmp_path / "sealed.jsonl"
    source.write_text(json.dumps({"event_id": "x"}) + "\n", encoding="utf-8")
    registry = tmp_path / "global.json"
    result = tmp_path / "sealed_eval.json"
    sealed = {"name": "april", "start": pd.Timestamp("2026-04-02T00:00:00Z"),
              "end": pd.Timestamp("2026-05-04T00:00:00Z")}
    real_write_atomic = optimizer._write_json_atomic
    calls = {"n": 0}

    def fail_second_registry_write(path, payload):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated registry complete failure")
        real_write_atomic(path, payload)

    monkeypatch.setattr(optimizer, "_write_json_atomic", fail_second_registry_write)
    with pytest.raises(OSError, match="simulated registry complete"):
        consume_sealed_once(registry, source_path=source, sealed=sealed,
                            result_path=result, payload={"ok": True})
    assert json.loads(result.read_text(encoding="utf-8")) == {"ok": True}
    entries = json.loads(registry.read_text(encoding="utf-8"))["sealed_consumptions"]
    assert len(entries) == 1
    assert next(iter(entries.values()))["status"] == "pending"

    monkeypatch.undo()
    with pytest.raises(ValueError, match="already been consumed"):
        consume_sealed_once(registry, source_path=source, sealed=sealed,
                            result_path=tmp_path / "second.json", payload={"ok": True})


def test_sealed_registry_lock_is_exclusive_and_fail_closed(tmp_path):
    source = tmp_path / "sealed.jsonl"
    source.write_text(json.dumps({"event_id": "x"}) + "\n", encoding="utf-8")
    registry = tmp_path / "global.json"
    lock = registry.with_name(registry.name + ".lock")
    lock.write_text("held", encoding="utf-8")
    sealed = {"name": "april", "start": pd.Timestamp("2026-04-02T00:00:00Z"),
              "end": pd.Timestamp("2026-05-04T00:00:00Z")}

    with pytest.raises(ValueError, match="registry is locked"):
        consume_sealed_once(registry, source_path=source, sealed=sealed,
                            result_path=tmp_path / "sealed_eval.json", payload={"ok": True})
    assert lock.exists()
    assert not registry.exists()
    assert not (tmp_path / "sealed_eval.json").exists()


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _cli_common(out_dir):
    return ["--out-dir", str(out_dir), "--thresholds", ".5,.6", "--tp-atrs", "5",
            "--sl-atrs", "2", "--horizons", "72", "--min-trades", "1",
            "--max-gold-recall-drop-pp", "100", "--max-trigger-density-per-day", "100",
            "--min-positive-folds", "0", "--min-worst-fold-maker-bp", "-10000"]


def _fold_doc():
    return {
        "schema_version": 1, "purge_bars": 162,
        "folds": [{"name": "inner", "calibration_start": "2025-01-01T00:00:00Z",
                   "calibration_end": "2026-02-26T06:00:00Z",
                   "validation_start": "2026-03-01T00:00:00Z",
                   "validation_end": "2026-03-31T07:30:00Z"}],
        "sealed_eval": {"name": "april", "start": "2026-04-02T00:00:00Z",
                        "end": "2026-05-04T00:00:00Z"},
    }


def _write_folds(path):
    path.write_text(json.dumps(_fold_doc()), encoding="utf-8")


def _write_selecting_inner_source(path):
    _write_jsonl(path, [
        _row("inner-bad", "2026-03-01T01:00:00Z", .55, [101.0] * 72),
        _row("inner-good", "2026-03-01T08:00:00Z", .65, [99.0] * 72),
    ])


def _sealed_source_spy(monkeypatch, sealed_path):
    calls = {"sealed": 0}
    real_read_jsonl = optimizer.read_jsonl

    def read_jsonl(path):
        if path.resolve() == sealed_path.resolve():
            calls["sealed"] += 1
            pytest.fail("sealed source was opened before lineage validation failed")
        return real_read_jsonl(path)

    monkeypatch.setattr(optimizer, "read_jsonl", read_jsonl)
    return calls


def test_cli_rejects_nonempty_output_directory(tmp_path):
    out = tmp_path / "used"
    out.mkdir()
    (out / "foreign.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        main(_cli_common(out))


def test_sealed_cli_reports_selected_baseline_and_delta_once(tmp_path):
    folds = tmp_path / "folds.json"
    _write_folds(folds)
    inner_source = tmp_path / "inner.jsonl"
    _write_selecting_inner_source(inner_source)
    inner_out = tmp_path / "inner-out"
    assert main(_cli_common(inner_out) + ["--fold-config", str(folds),
                                          "--candidates", str(inner_source)]) == 0

    source = tmp_path / "sealed.jsonl"
    _write_jsonl(source, [_row("sealed", "2026-04-02T01:00:00Z", .55, [99.0] * 72,
                                  gold=True)])
    registry = tmp_path / "registry.json"
    out = tmp_path / "sealed-out"
    argv = _cli_common(out) + ["--fold-config", str(folds), "--evaluate-sealed", str(source),
                               "--selected-config", str(inner_out / "config_selection.json"),
                               "--global-registry", str(registry)]
    assert main(argv) == 0
    payload = json.loads((out / "sealed_eval.json").read_text(encoding="utf-8"))
    assert payload["selected"]["n_trades"] == 0
    assert payload["baseline"]["n_trades"] == 1
    assert payload["delta_selected_minus_baseline"]["n_trades"] == -1
    assert len(json.loads(registry.read_text(encoding="utf-8"))["sealed_consumptions"]) == 1
    with pytest.raises(ValueError, match="non-empty"):
        main(argv)


def test_sealed_cli_rejects_manual_selection_before_opening_sealed(tmp_path, monkeypatch):
    folds = tmp_path / "folds.json"
    _write_folds(folds)
    sealed_source = tmp_path / "sealed.jsonl"
    _write_jsonl(sealed_source, [_row("sealed", "2026-04-02T01:00:00Z", .55, [99.0] * 72)])
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "schema_version": 1,
        "selected_config": {"threshold": .6, "tp_atr": 5, "sl_atr": 2, "horizon": 72},
        "selected_config_key": "threshold=0.6,tp=5,sl=2,horizon=72",
        "sealed_read": False,
    }), encoding="utf-8")
    calls = _sealed_source_spy(monkeypatch, sealed_source)

    with pytest.raises(ValueError, match="sibling prereg"):
        main(_cli_common(tmp_path / "sealed-out") + [
            "--fold-config", str(folds), "--evaluate-sealed", str(sealed_source),
            "--selected-config", str(selection), "--global-registry", str(tmp_path / "registry.json"),
        ])
    assert calls["sealed"] == 0


def test_sealed_cli_rejects_inner_lineage_mismatch_before_opening_sealed(tmp_path, monkeypatch):
    folds = tmp_path / "folds.json"
    _write_folds(folds)
    inner_source = tmp_path / "inner.jsonl"
    _write_selecting_inner_source(inner_source)
    inner_out = tmp_path / "inner-out"
    assert main(_cli_common(inner_out) + ["--fold-config", str(folds),
                                          "--candidates", str(inner_source)]) == 0
    inner_source.write_text(inner_source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    sealed_source = tmp_path / "sealed.jsonl"
    _write_jsonl(sealed_source, [_row("sealed", "2026-04-02T01:00:00Z", .55, [99.0] * 72)])
    calls = _sealed_source_spy(monkeypatch, sealed_source)

    with pytest.raises(ValueError, match="candidate_source_sha256"):
        main(_cli_common(tmp_path / "sealed-out") + [
            "--fold-config", str(folds), "--evaluate-sealed", str(sealed_source),
            "--selected-config", str(inner_out / "config_selection.json"),
            "--global-registry", str(tmp_path / "registry.json"),
        ])
    assert calls["sealed"] == 0
