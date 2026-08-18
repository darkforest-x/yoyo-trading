"""Fixed W10 / Core4 / Confirm1 migration contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yoyo.datasets.legacy_gold_migration.canonical import no_signal_slot, w10_fields
from yoyo.datasets.legacy_gold_migration.conflicts import cluster_records, resolve_cluster
from yoyo.datasets.legacy_gold_migration.intervals import (
    inclusive_to_half_open,
    map_w10,
    suggest_four_bar,
    yolo_box_to_bars,
)
from yoyo.datasets.legacy_gold_migration.io import sha256_text, write_jsonl
from yoyo.datasets.legacy_gold_migration.migration import migrate, write_migration
from yoyo.datasets.legacy_gold_migration.renderer import SpanError, draw_review_overlay, make_w10_transform, render_w10
from yoyo.datasets.legacy_gold_migration.sources import SourceRecord, load_local_8768
from yoyo.layers.l1_detection.data import add_mas
from yoyo.layers.l1_detection.render import IMG_WIDTH, MARGIN


def _frame(n: int = 400, start: str = "2026-03-01T00:00:00+00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    close = 100 + np.sin(np.linspace(0, 8, n)) * 2 + np.linspace(0, 1, n)
    df = pd.DataFrame(
        {
            "open_time": idx,
            "open": close - 0.05,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": 1.0,
        }
    )
    return add_mas(df)


def test_half_open_22_26_is_four_bars():
    start, end = 22, 26
    assert end - start == 4
    assert list(range(start, end)) == [22, 23, 24, 25]


def test_inclusive_22_26_is_five_bars():
    start, end = inclusive_to_half_open(22, 26)
    assert end - start == 5
    assert list(range(start, end)) == [22, 23, 24, 25, 26]


def test_w10_mapping():
    mapped = map_w10(100, 104)
    assert mapped["window_start_bar"] == 95
    assert mapped["window_end_exclusive_bar"] == 105
    assert mapped["window_length"] == 10
    assert mapped["confirmation_bar"] == 104
    assert mapped["local_core_start"] == 5
    assert mapped["local_core_end_exclusive"] == 9
    assert mapped["local_confirmation_position"] == 9


def test_yolo_box_to_bars_uses_centers():
    n, width = 10, IMG_WIDTH
    from yoyo.datasets.legacy_gold_migration.intervals import candle_centers

    centers = candle_centers(n, width, MARGIN)
    x_min, x_max = centers[5], centers[8]
    xc = ((x_min + x_max) / 2) / width
    bw = (x_max - x_min) / width + 1e-6
    out = yolo_box_to_bars(xc, bw, n, width, MARGIN)
    assert out["local_start"] == 5
    assert out["local_end_exclusive"] == 9


def test_8768_inclusive_adapter(tmp_path: Path):
    row = {
        "gold_id": "ETH_USDT_SWAP_15m_demo",
        "symbol": "ETH_USDT_SWAP",
        "timeframe": "15m",
        "source_path": "data/kline_fetched/x.csv",
        "decision_bar": 30,
        "decision_time": "2026-03-01T00:00:00+00:00",
        "shape_label": "POSITIVE",
        "core_start_bar": 22,
        "core_end_bar": 26,
        "reviewed_at": "2026-08-13T00:00:00+00:00",
    }
    p = tmp_path / "gold_v1.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    recs = load_local_8768(p, pd.Timestamp("2026-05-04T00:00:00+00:00"))
    assert len(recs) == 1
    assert recs[0].source_interval_semantics == "inclusive"
    assert recs[0].shape_label == "SIGNAL"
    assert recs[0].core_end_exclusive_bar - recs[0].core_start_bar == 5


def test_conflict_keeps_higher_precedence():
    a = SourceRecord(
        source_annotation_type="local_8768",
        source_interval_semantics="inclusive",
        precedence=1,
        source_dataset="a",
        source_record_id="1",
        symbol="ETH_USDT_SWAP",
        decision_bar=100,
        decision_time="2026-03-01T00:00:00+00:00",
        shape_label="IGNORE",
        core_start_bar=None,
        core_end_exclusive_bar=None,
    )
    b = SourceRecord(
        source_annotation_type="labelstudio_keypoints",
        source_interval_semantics="inclusive",
        precedence=2,
        source_dataset="b",
        source_record_id="1",
        symbol="ETH_USDT_SWAP",
        decision_bar=100,
        decision_time="2026-03-01T00:00:00+00:00",
        shape_label="NO_SIGNAL",
    )
    resolved = resolve_cluster([a, b], [0, 1])
    assert resolved["winner"] is a
    assert resolved["conflicts"][0]["resolution"] == "KEEP_HIGHER"


def test_decision_rebased_is_one_click():
    from yoyo.datasets.legacy_gold_migration.migration import _legacy_offset, _status_for_signal

    offset = _legacy_offset(107, 104)
    assert offset == 3
    status = _status_for_signal(4, offset, True, True, False, "human_gold_owner_box", False, {})
    assert status == "ONE_CLICK_REVIEW"


def test_direct_when_decision_is_confirmation():
    from yoyo.datasets.legacy_gold_migration.migration import _status_for_signal

    assert _status_for_signal(4, 0, True, True, False, "local_8768", False, {}) == "DIRECT"


def test_wicks_and_mas_in_y_scale():
    frame = _frame()
    window = frame.iloc[200:210].reset_index(drop=True)
    img, tf = render_w10(window, y_pad_frac=0.05, overlay=False)
    assert img.shape[1] == IMG_WIDTH
    raw_high = max(window["high"].max(), *(window[c].max() for c in ("sma20", "ema20", "sma60", "ema60", "sma120", "ema120")))
    raw_low = min(window["low"].min(), *(window[c].min() for c in ("sma20", "ema20", "sma60", "ema60", "sma120", "ema120")))
    span = raw_high - raw_low
    assert abs(tf.price_max - (raw_high + span * 0.05)) < 1e-9
    assert abs(tf.price_min - (raw_low - span * 0.05)) < 1e-9
    y_high = tf.y_at(window["high"].max())
    y_low = tf.y_at(window["low"].min())
    assert 0 <= y_high < img.shape[0]
    assert 0 <= y_low < img.shape[0]


def test_span_zero_raises():
    window = _frame().iloc[200:210].reset_index(drop=True)
    for col in ("open", "high", "low", "close", "sma20", "sma60", "sma120", "ema20", "ema60", "ema120"):
        window[col] = 1.0
    with pytest.raises(SpanError):
        render_w10(window)


def test_overlay_not_in_training_pixels():
    window = _frame().iloc[200:210].reset_index(drop=True)
    clean, tf = render_w10(window, overlay=False)
    marked = draw_review_overlay(clean, tf)
    assert not np.array_equal(clean, marked)


def test_last_bar_is_decision():
    mapped = w10_fields(100, 104)
    assert mapped["decision_bar"] == mapped["confirmation_bar"]
    assert mapped["window_end_exclusive_bar"] - 1 == mapped["decision_bar"]


def test_no_signal_has_null_core_and_slot():
    slot = no_signal_slot(104)
    assert slot["slot_start_bar"] == 100
    assert slot["slot_end_exclusive_bar"] == 104
    assert slot["window_length"] == 10


def test_suggest_four_bar_picks_rightmost_on_tie():
    frame = _frame()
    # Flat-ish spreads; algorithm must be deterministic.
    sug = suggest_four_bar(frame, 50, 56)
    assert sug is not None
    assert sug["core_end_exclusive_bar"] - sug["core_start_bar"] == 4


def test_idempotent_write(tmp_path: Path):
    rows = [{"gold_id": "A", "v": 1}]
    p = tmp_path / "x.jsonl"
    write_jsonl(p, rows)
    a = p.read_text(encoding="utf-8")
    write_jsonl(p, rows)
    b = p.read_text(encoding="utf-8")
    assert a == b
    assert sha256_text(a) == sha256_text(b)


def test_local_24_adapter_keeps_every_gold_id():
    path = Path("/Users/zhangzc/yoyo-trading/datasets/gold_v1.jsonl")
    if not path.exists():
        pytest.skip("gold_v1.jsonl missing")
    recs = load_local_8768(path, pd.Timestamp("2026-05-04T00:00:00+00:00"))
    ids = {json.loads(l)["gold_id"] for l in path.read_text(encoding="utf-8").splitlines() if l}
    assert {r.extra["legacy_gold_id"] for r in recs} == ids
    assert len(recs) == 24


def test_same_precedence_shape_conflict_requires_owner():
    a = SourceRecord(
        source_annotation_type="owner_review_jsonl",
        source_interval_semantics="inclusive",
        precedence=3,
        source_dataset="p1",
        source_record_id="x",
        symbol="ETH_USDT_SWAP",
        decision_time="2026-03-01T00:00:00+00:00",
        decision_bar=10,
        shape_label="SIGNAL",
        reviewed_at="2026-08-01T00:00:00+00:00",
    )
    b = SourceRecord(
        source_annotation_type="owner_review_jsonl",
        source_interval_semantics="inclusive",
        precedence=3,
        source_dataset="p2",
        source_record_id="y",
        symbol="ETH_USDT_SWAP",
        decision_time="2026-03-01T00:00:00+00:00",
        decision_bar=10,
        shape_label="NO_SIGNAL",
        reviewed_at="2026-08-01T00:00:00+00:00",
    )
    resolved = resolve_cluster([a, b], [0, 1])
    assert resolved["needs_owner"] is True


def test_freeze_skips_conflict_and_coreless_a():
    from tools.freeze_fixed_w10_gold import CORRECTED_EASY_BACKGROUND_N, freeze_rows

    easy_id = next(iter(CORRECTED_EASY_BACKGROUND_N))
    events = [
        {
            "gold_id": "CONFLICT_X",
            "symbol": "ETH_USDT_SWAP",
            "migration_status": "CONFLICT",
            "shape_label": "SIGNAL",
            "decision_bar": 104,
            "confirmation_bar": 104,
            "source_path": "data/kline_fetched/okx_ETH_USDT_SWAP_15m_1.csv",
        },
        {
            "gold_id": easy_id,
            "symbol": "BTC_USDT_SWAP",
            "migration_status": "ONE_CLICK_REVIEW",
            "shape_label": "NO_SIGNAL",
            "negative_type": "EASY_BACKGROUND",
            "decision_bar": 104,
            "confirmation_bar": 104,
            "source_path": "data/kline_fetched/okx_BTC_USDT_SWAP_15m_1.csv",
            "split": "train",
        },
        {
            "gold_id": "CORELESS_A",
            "symbol": "SOL_USDT_SWAP",
            "migration_status": "ONE_CLICK_REVIEW",
            "shape_label": "NO_SIGNAL",
            "decision_bar": 204,
            "confirmation_bar": 204,
            "source_path": "data/kline_fetched/okx_ETH_USDT_SWAP_15m_1.csv",
            "split": "train",
        },
        {
            "gold_id": "GOOD_A",
            "symbol": "ETH_USDT_SWAP",
            "migration_status": "MANUAL_ADJUST",
            "shape_label": "SIGNAL",
            "core_start_bar": 100,
            "core_end_exclusive_bar": 104,
            "decision_bar": 104,
            "source_path": "data/kline_fetched/okx_ETH_USDT_SWAP_15m_1.csv",
            "split": "train",
            "future_used_in_model_input": False,
            "holdout_read": False,
        },
    ]
    state = {
        "CONFLICT_X": {"gold_id": "CONFLICT_X", "review_action": "A", "shape_label": "SIGNAL"},
        easy_id: {
            "gold_id": easy_id,
            "review_action": "A",
            "shape_label": "SIGNAL",
            "core_start_bar": None,
            "core_end_exclusive_bar": None,
            "negative_type": "EASY_BACKGROUND",
        },
        "CORELESS_A": {
            "gold_id": "CORELESS_A",
            "review_action": "A",
            "shape_label": "SIGNAL",
            "core_start_bar": None,
            "core_end_exclusive_bar": None,
        },
        "GOOD_A": {
            "gold_id": "GOOD_A",
            "review_action": "A",
            "shape_label": "SIGNAL",
            "core_start_bar": 100,
            "core_end_exclusive_bar": 104,
        },
    }
    frozen, excluded = freeze_rows(events, state)
    by_id = {r["gold_id"]: r for r in frozen}
    assert "CONFLICT_X" not in by_id
    assert any(r["gold_id"] == "CONFLICT_X" and r["reason"] == "conflict" for r in excluded)
    assert by_id[easy_id]["shape_label"] == "NO_SIGNAL"
    assert by_id[easy_id]["core_start_bar"] is None
    assert by_id["CORELESS_A"]["shape_label"] == "NO_SIGNAL"
    assert by_id["GOOD_A"]["shape_label"] == "SIGNAL"
    assert by_id["GOOD_A"]["core_length"] == 4
    assert by_id["GOOD_A"]["window_length"] == 10
    assert by_id["GOOD_A"]["local_core_start"] == 5
