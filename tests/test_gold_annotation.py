"""Gold annotation contract tests (V1 Label Studio pipeline)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yoyo.datasets.gold_box import (
    UnstableWickError,
    bar_center_x,
    core_price_bounds,
    snap_x_to_bar,
    yolo_xywh,
)
from yoyo.datasets.gold_render import FOOTER_H, render_context, render_local
from yoyo.datasets.gold_schema import gold_id, parse_jsonl, validate_gold
from yoyo.layers.l1_detection.data import add_mas
from yoyo.layers.l1_detection.render import IMG_HEIGHT, IMG_WIDTH, MARGIN


def _frame(n: int = 400, start: str = "2026-03-01T00:00:00+00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    close = 100 + np.sin(np.linspace(0, 8, n)) * 2 + np.linspace(0, 1, n)
    high = close + 0.4
    low = close - 0.4
    open_ = close - 0.05
    df = pd.DataFrame(
        {
            "open_time": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1.0,
        }
    )
    return add_mas(df)


def test_local_ends_at_decision_no_future():
    frame = _frame()
    decision = 200
    out = render_local(frame, decision, 30)
    assert out["local_end_bar"] == decision
    assert out["future_bars"] == 0
    assert out["local_window_length"] == 30
    assert out["local_start_bar"] == decision - 29


def test_context_has_configured_pre_post():
    frame = _frame()
    decision = 200
    meta = render_context(frame, decision, 194, 199, pre_bars=50, post_bars=120)
    assert meta["pre_bars"] == 50
    assert meta["post_bars"] == 120
    assert meta["n_bars"] == 50 + (199 - 194 + 1) + 120


def test_snap_x_to_nearest_bar_center():
    n, width = 30, IMG_WIDTH
    for i in range(n):
        x = bar_center_x(i, n, width, MARGIN)
        assert snap_x_to_bar(x, n, width, MARGIN) == i
        assert snap_x_to_bar(x + 2, n, width, MARGIN) == i


def test_positive_requires_core():
    row = {
        "gold_id": "X",
        "symbol": "ETH_USDT_SWAP",
        "timeframe": "15m",
        "source_repo": "fable-trading",
        "source_path": "data/x.csv",
        "candidate_source": "legacy_gold",
        "decision_bar": 30,
        "decision_time": "2026-03-01T00:00:00+00:00",
        "local_start_bar": 1,
        "local_end_bar": 30,
        "local_window_length": 30,
        "shape_label": "POSITIVE",
        "box_rule": "full_wicks_plus_six_ma",
        "box_status": "owner_bar_range",
        "reviewer": "owner",
        "holdout_read": False,
    }
    with pytest.raises(ValueError, match="core_start"):
        validate_gold(row)
    row["core_start_bar"] = 20
    row["core_end_bar"] = 24
    validate_gold(row)


def test_negative_must_not_carry_box():
    row = {
        "gold_id": "Y",
        "symbol": "ETH_USDT_SWAP",
        "timeframe": "15m",
        "source_repo": "fable-trading",
        "source_path": "data/x.csv",
        "candidate_source": "random_background",
        "decision_bar": 30,
        "decision_time": "2026-03-01T00:00:00+00:00",
        "local_start_bar": 1,
        "local_end_bar": 30,
        "local_window_length": 30,
        "shape_label": "NEGATIVE",
        "core_start_bar": 10,
        "core_end_bar": 12,
        "box_rule": "full_wicks_plus_six_ma",
        "box_status": "none",
        "reviewer": "owner",
        "holdout_read": False,
    }
    with pytest.raises(ValueError, match="must not carry"):
        validate_gold(row)


def test_auto_box_covers_wicks_and_mas():
    frame = _frame()
    window = frame.iloc[171:201].reset_index(drop=True)
    start, end = 10, 16
    sl = window.iloc[start : end + 1]
    high, low = core_price_bounds(window, start, end)
    assert high >= sl["high"].max() - 1e-9
    assert low <= sl["low"].min() + 1e-9
    for col in ("sma20", "ema20", "sma60", "ema60", "sma120", "ema120"):
        assert high >= sl[col].max() - 1e-9
        assert low <= sl[col].min() + 1e-9
    box = yolo_xywh(window, start, end)
    assert 0 < box["w"] < 1
    assert 0 < box["h"] < 1


def test_extreme_wick_not_silently_clipped():
    frame = _frame()
    window = frame.iloc[171:201].reset_index(drop=True)
    window.loc[12, "high"] = float(window.loc[12, "low"]) + 80
    with pytest.raises(UnstableWickError):
        yolo_xywh(window, 10, 16)


def test_parse_jsonl_dedupes_gold_id():
    base = {
        "gold_id": "ETH_USDT_SWAP_15m_1",
        "symbol": "ETH_USDT_SWAP",
        "timeframe": "15m",
        "source_repo": "fable-trading",
        "source_path": "p",
        "candidate_source": "legacy_gold",
        "decision_bar": 30,
        "decision_time": "2026-03-01T00:00:00+00:00",
        "local_start_bar": 1,
        "local_end_bar": 30,
        "local_window_length": 30,
        "shape_label": "NEGATIVE",
        "box_rule": "full_wicks_plus_six_ma",
        "box_status": "none",
        "reviewer": "owner",
        "holdout_read": False,
    }
    text = json.dumps(base) + "\n" + json.dumps(base) + "\n"
    rows = parse_jsonl(text)
    assert len(rows) == 1


def test_holdout_flag_rejected():
    with pytest.raises(ValueError, match="holdout"):
        validate_gold(
            {
                "gold_id": "Z",
                "symbol": "ETH_USDT_SWAP",
                "timeframe": "15m",
                "source_repo": "fable-trading",
                "source_path": "p",
                "candidate_source": "legacy_gold",
                "decision_bar": 30,
                "decision_time": "2026-03-01T00:00:00+00:00",
                "local_start_bar": 1,
                "local_end_bar": 30,
                "local_window_length": 30,
                "shape_label": "IGNORE",
                "box_rule": "full_wicks_plus_six_ma",
                "box_status": "none",
                "reviewer": "owner",
                "holdout_read": True,
            }
        )


def test_local_image_has_footer_not_future():
    frame = _frame()
    out = render_local(frame, 200, 30)
    assert out["image"].shape[0] == IMG_HEIGHT + FOOTER_H
    assert out["image"].shape[1] == IMG_WIDTH


def test_gold_id_stable():
    assert gold_id("ETH_USDT_SWAP", "15m", "2026-04-01T12:00:00+00:00").startswith(
        "ETH_USDT_SWAP_15m_"
    )


def test_labelstudio_export_maps_keypoints_and_rejects_pos_without_core():
    from tools.convert_labelstudio_export import convert_task

    data = {
        "gold_id": "ETH_USDT_SWAP_15m_demo",
        "symbol": "ETH_USDT_SWAP",
        "source_path": "data/kline_fetched/x.csv",
        "decision_bar": 200,
        "decision_time": "2026-03-01T00:00:00+00:00",
        "local_start_bar": 171,
        "local_end_bar": 200,
        "local_window_length": 30,
        "local_image_width": IMG_WIDTH,
    }
    pos_no_pts = {
        "data": data,
        "annotations": [
            {"result": [{"from_name": "shape", "value": {"choices": ["POSITIVE"]}}]}
        ],
    }
    with pytest.raises(ValueError, match="START/END"):
        convert_task(pos_no_pts, {})
    start_x = bar_center_x(10, 30, IMG_WIDTH, MARGIN) / IMG_WIDTH * 100
    end_x = bar_center_x(16, 30, IMG_WIDTH, MARGIN) / IMG_WIDTH * 100
    task = {
        "data": data,
        "annotations": [
            {
                "result": [
                    {"from_name": "shape", "value": {"choices": ["POSITIVE"]}},
                    {
                        "from_name": "core",
                        "value": {"x": start_x, "y": 90, "keypointlabels": ["START"]},
                    },
                    {
                        "from_name": "core",
                        "value": {"x": end_x, "y": 90, "keypointlabels": ["END"]},
                    },
                ]
            }
        ],
    }
    row = convert_task(task, {})
    assert row is not None
    assert row["core_start_bar"] == 181
    assert row["core_end_bar"] == 187
    assert convert_task(task, {row["gold_id"]: row}) is None


def test_negative_export_has_no_box():
    from tools.convert_labelstudio_export import convert_task

    row = convert_task(
        {
            "data": {
                "gold_id": "ETH_USDT_SWAP_15m_neg",
                "symbol": "ETH_USDT_SWAP",
                "source_path": "p",
                "decision_bar": 200,
                "decision_time": "2026-03-01T00:00:00+00:00",
                "local_start_bar": 171,
                "local_end_bar": 200,
                "local_window_length": 30,
            },
            "annotations": [
                {"result": [{"from_name": "shape", "value": {"choices": ["NEGATIVE"]}}]}
            ],
        },
        {},
    )
    assert row is not None
    assert row["core_start_bar"] is None

