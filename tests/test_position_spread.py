from yoyo.datasets.position_spread import (
    assign_one,
    bin_of,
    core_fits,
    placements,
    rel_position,
)


def test_rel_shorter_w_moves_near_decision_core_leftward():
    # core 6 bars before decision: typical gold-center geometry
    decision, core_mid = 100, 94.0
    rel12 = rel_position(core_mid, decision, 12)
    rel19 = rel_position(core_mid, decision, 19)
    assert rel12 < rel19
    assert bin_of(rel12) == "middle"
    assert bin_of(rel19) == "right"


def test_left_bin_requires_core_far_from_decision():
    decision = 100
    # 8 bars before decision, W=12 → rel = 1 - 8/11 ≈ 0.273 left
    assert bin_of(rel_position(92.0, decision, 12)) == "left"
    # 4 bars before, even shortest W stays middle/right
    assert bin_of(rel_position(96.0, decision, 12)) in {"middle", "right"}


def test_core_must_fit_inside_window():
    assert core_fits(90, 95, 100, 12)  # start=89
    assert not core_fits(88, 95, 100, 12)  # box starts before window


def test_assign_one_prefers_starved_left():
    opts = placements(88, 92, 100)  # far enough for left at W=12
    assert any(o["bin"] == "left" for o in opts)
    pick = assign_one(opts, {"left": 0, "middle": 10, "right": 10})
    assert pick is not None
    assert pick["bin"] == "left"


def test_assign_all_keeps_every_left_capable_sample_on_the_left():
    from yoyo.datasets.position_spread import assign_all

    rows = [
        # far from decision → can be left
        {"sample_id": "L", "box_start_bar": 88, "box_end_bar": 92, "decision_bar": 100, "window_length": 16},
        # typical near-decision core → middle/right only
        {"sample_id": "M", "box_start_bar": 96, "box_end_bar": 99, "decision_bar": 100, "window_length": 16},
    ]
    out = {r["sample_id"]: r for r in assign_all(rows)}
    assert out["L"]["chosen_bin"] == "left"
    assert out["M"]["chosen_bin"] in {"middle", "right"}
