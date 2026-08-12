"""The window/box arithmetic that defines every Dataset V3 label."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yoyo.datasets.window_render import (
    SourceError,
    enrich,
    label_text,
    load_prefix,
    render_sample,
    window_slice,
    yolo_box_from_bars,
)


def synthetic_csv(path, rows: int = 200) -> None:
    times = pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC")
    close = np.linspace(100.0, 130.0, rows)
    frame = pd.DataFrame(
        {
            "open_time": times,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.arange(rows, dtype=float),
            "confirm": 1,
        }
    )
    frame.to_csv(path, index=False)


def test_load_prefix_stops_at_required_end(tmp_path) -> None:
    path = tmp_path / "series.csv"
    synthetic_csv(path)
    frame = load_prefix(path, 49)
    assert len(frame) == 50


def test_load_prefix_refuses_a_series_too_short_for_the_window(tmp_path) -> None:
    path = tmp_path / "series.csv"
    synthetic_csv(path, rows=30)
    with pytest.raises(SourceError, match="prefix index mismatch"):
        load_prefix(path, 40)


def test_unconfirmed_bar_inside_the_prefix_fails_instead_of_shifting_indices(tmp_path) -> None:
    """An unconfirmed bar removes a row, so every global index after it moves.

    Silently returning the shortened frame would draw a window one bar off from
    the one the manifest names. The guard turns that into an error.
    """
    path = tmp_path / "series.csv"
    synthetic_csv(path)
    raw = pd.read_csv(path)
    raw.loc[10, "confirm"] = 0
    raw.to_csv(path, index=False)
    with pytest.raises(SourceError, match="prefix index mismatch"):
        load_prefix(path, 49)


def test_mas_do_not_depend_on_where_the_window_starts(tmp_path) -> None:
    path = tmp_path / "series.csv"
    synthetic_csv(path)
    enriched = enrich(load_prefix(path, 160))
    wide = window_slice(enriched, 140, 160)
    narrow = window_slice(enriched, 149, 160)
    assert wide["sma120"].iloc[-1] == narrow["sma120"].iloc[-1]


def test_window_slice_rejects_an_end_beyond_the_series(tmp_path) -> None:
    path = tmp_path / "series.csv"
    synthetic_csv(path, rows=50)
    enriched = enrich(load_prefix(path, 49))
    with pytest.raises(SourceError, match="beyond series length"):
        window_slice(enriched, 40, 60)


def test_render_sample_produces_a_box_for_positives_only(tmp_path) -> None:
    path = tmp_path / "series.csv"
    synthetic_csv(path)
    enriched = enrich(load_prefix(path, 160))
    negative = render_sample(enriched, 149, 160)
    assert negative["box"] is None and negative["label"] == ""
    positive = render_sample(enriched, 149, 160, core_global=(152, 156))
    assert positive["box"] is not None
    assert positive["label"].startswith("0 ") and positive["label"].endswith("\n")
    assert positive["bars"] == 12


def test_render_sample_rejects_a_core_that_cannot_form_a_box(tmp_path) -> None:
    path = tmp_path / "series.csv"
    synthetic_csv(path)
    enriched = enrich(load_prefix(path, 160))
    with pytest.raises(SourceError, match="empty positive box"):
        render_sample(enriched, 149, 160, core_global=(200, 210))


def test_yolo_box_is_normalised_and_ordered() -> None:
    class Transform:
        width = 960
        height = 640
        candle_half_w = 3

        def x_at(self, index: int) -> int:
            return 10 + index * 20

        def y_at(self, price: float) -> int:
            return int(600 - price)

    window = pd.DataFrame({"high": [110.0, 120.0, 115.0], "low": [100.0, 105.0, 102.0]})
    box = yolo_box_from_bars(Transform(), window, 0, 2)
    assert box is not None
    xc, yc, bw, bh = box
    assert 0 < xc < 1 and 0 < yc < 1 and 0 < bw <= 1 and 0 < bh <= 1


def test_label_text_is_empty_for_negatives() -> None:
    assert label_text(None) == ""
    assert label_text((0.5, 0.5, 0.1, 0.2)) == "0 0.500000 0.500000 0.100000 0.200000\n"
