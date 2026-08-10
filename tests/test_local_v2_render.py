"""Regression tests for the opt-in Local-Signal V2 blank-slot renderer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yoyo.layers.l1_detection.data import ALL_MA_COLS
from yoyo.layers.l1_detection.local_v2_render import render_causal_chart
from yoyo.layers.l1_detection.render import BG, render_chart


def _frame(n: int = 30) -> pd.DataFrame:
    phase = np.arange(n, dtype=float)
    close = 100.0 + np.sin(phase / 4.0) + phase * 0.03
    frame = pd.DataFrame(
        {
            "open": close - 0.12,
            "high": close + 0.45,
            "low": close - 0.40,
            "close": close,
        }
    )
    for offset, column in enumerate(ALL_MA_COLS):
        frame[column] = close + (offset - 2.5) * 0.04
    return frame


def test_zero_blank_slots_preserve_frozen_legacy_pixels() -> None:
    frame = _frame()
    legacy, legacy_transform = render_chart(frame)
    local_v2, local_transform = render_causal_chart(frame, right_blank_slots=0)

    assert np.array_equal(local_v2, legacy)
    assert local_transform == legacy_transform


def test_blank_slots_change_layout_without_adding_visible_bars() -> None:
    frame = _frame()
    image, transform = render_causal_chart(frame, right_blank_slots=12)

    assert transform.n_bars == 42
    assert transform.x_at(29) / transform.width < 0.72
    blank_start = transform.x_at(30)
    assert np.all(image[:, blank_start + transform.candle_half_w + 2 :] == BG)


@pytest.mark.parametrize("value", [-1, -5])
def test_negative_blank_slots_are_rejected(value: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        render_causal_chart(_frame(), right_blank_slots=value)


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_non_integer_blank_slots_are_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        render_causal_chart(_frame(), right_blank_slots=value)  # type: ignore[arg-type]
