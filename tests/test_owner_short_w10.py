"""Owner-short → W10 rebuild: center-4 cores and nearest internal empty."""

from yoyo.datasets.legacy_gold_migration.intervals import (
    center_four_bar,
    map_w10,
    nearest_empty_confirmation,
    window_overlaps_any,
)


def test_center_four_already_four():
    assert center_four_bar(10, 13) == (10, 14)


def test_center_four_seven_bars():
    # inclusive 10..16 is 7 bars; centre 4 is 11..14 half-open [11, 15)
    assert center_four_bar(10, 16) == (11, 15)
    assert map_w10(11, 15)["window_length"] == 10
    assert map_w10(11, 15)["decision_bar"] == 15


def test_center_four_six_bars():
    assert center_four_bar(10, 15) == (11, 15)


def test_nearest_empty_skips_protected_and_self():
    # decision at 300, protect [290, 320)
    protected = [(290, 320)]
    cand = nearest_empty_confirmation(decision=300, n=500, warmup=240, protected=protected, radius=40)
    assert cand is not None
    assert cand != 300
    ws, we = cand - 9, cand + 1
    assert not window_overlaps_any(ws, we, protected)


def test_no_empty_when_fully_guarded():
    protected = [(0, 500)]
    assert (
        nearest_empty_confirmation(decision=300, n=400, warmup=240, protected=protected, radius=20)
        is None
    )
