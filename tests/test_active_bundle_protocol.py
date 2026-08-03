"""Acceptance C-01..C-08 and A-05/A-06: one exact bundle, verified or refused.

Every case here is a way production could end up running something other than
what the bundle claims. The repo has already been bitten by the mild version of
this -- an artifact loader that globs for the newest JSON, skips a corrupt one and
serves an older model, while never reading models/ACTIVE at all.

Fixtures are tiny files in tmp_path rather than the real artifacts: the actual
dataset is 12MB and hashing it in every test would buy nothing, since what is
under test is the checking, not the files.

Takeover plan: docs/protocol_repair/ACCEPTANCE_MATRIX.md sections 1 and 3.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import pandas as pd

from yoyo.contracts.protocol import file_sha256
from yoyo.contracts.protocol import (
    BundleError,
    REQUIRED_FIELDS,
    load_active_bundle,
    load_bundle,
    require_active_bundle,
    runtime_artifact,
)


def _artefacts(tmp_path: Path) -> dict[str, Path]:
    models = tmp_path / "models"
    data = tmp_path / "data"
    models.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    files = {
        "model": models / "m.txt",
        "dataset": data / "d.csv",
        "detector": models / "det.pt",
    }
    files["model"].write_text("tree\n", encoding="utf-8")
    files["dataset"].write_text("signal_time,symbol\n", encoding="utf-8")
    files["detector"].write_bytes(b"\x00weights")
    return files


def _bundle_dict(tmp_path: Path, **overrides) -> dict:
    f = _artefacts(tmp_path)
    d = {
        "bundle_version": 1,
        "protocol_version": "short_test_v1",
        "strategy_id": "dense_start_short_15m",
        "side": "short",
        "timeframe": "15m",
        "window_bars": 200,
        "candidate_source": "yolo",
        "max_tip_age_bars": 2,
        "feature_schema": "judgment_28_v1",
        "feature_semantics": "legacy_unaligned",
        "model_objective": "regression",
        "model_num_iteration": 1,
        "score_semantics": "predicted_net_barrier_taker",
        "threshold": -0.00044,
        "threshold_operator": ">=",
        "tie_policy": "legacy_large_tie_mass",
        "calibration_quantile": 0.9,
        "calibration_pass_rate": 0.9113407669295621,
        "threshold_equal_rate": 0.8615719336415556,
        "selector_status": "abnormal_tie_mass_audit_only",
        "research_entry_mode": "next_bar_open",
        "live_entry_mode": "none_until_p1",
        "tp_atr_mult": 5.0,
        "sl_atr_mult": 2.0,
        "horizon_bars": 72,
        "same_bar_policy": "conservative_sl",
        "gap_policy": "barrier_price",
        "return_convention": "linear_short",
        "cost_route": "swap_taker_in_target",
        "target_ret_column": "net_barrier_taker",
        "target_semantics": "net_taker",
        "target_cost_included": True,
        "reporting_route": "taker",
        "detector_path": "models/det.pt",
        "detector_sha256": file_sha256(f["detector"]),
        "model_path": "models/m.txt",
        "model_sha256": file_sha256(f["model"]),
        "dataset_path": "data/d.csv",
        "dataset_sha256": file_sha256(f["dataset"]),
        "execution_eligible": False,
        "paper_only": True,
    }
    d.update(overrides)
    return d


def _write(tmp_path: Path, **overrides) -> Path:
    path = tmp_path / "models" / "active_bundle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_bundle_dict(tmp_path, **overrides)), encoding="utf-8")
    return path


# ── C-01 ──────────────────────────────────────────────────────────────────────
def test_complete_bundle_loads(tmp_path: Path) -> None:
    p = load_bundle(_write(tmp_path), project_dir=tmp_path)
    assert p.side == "short"
    assert p.feature_semantics == "legacy_unaligned"
    assert p.execution_eligible is False
    assert p.horizon_bars == 72


# ── C-02 / C-03 / C-04 ────────────────────────────────────────────────────────
@pytest.mark.parametrize("field", ["model_sha256", "dataset_sha256", "detector_sha256"])
def test_tampered_hash_fails_to_load(tmp_path: Path, field: str) -> None:
    path = _write(tmp_path, **{field: "0" * 64})
    with pytest.raises(BundleError, match="mismatch"):
        load_bundle(path, project_dir=tmp_path)


def test_hash_is_checked_against_content_not_just_presence(tmp_path: Path) -> None:
    """The defect being removed is "the path exists, therefore it is right"."""
    path = _write(tmp_path)
    (tmp_path / "models" / "m.txt").write_text("different tree\n", encoding="utf-8")
    with pytest.raises(BundleError, match="model_sha256 mismatch"):
        load_bundle(path, project_dir=tmp_path)


# ── C-05 ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "field", ["side", "return_convention", "cost_route", "research_entry_mode",
              "live_entry_mode", "feature_semantics", "threshold_operator",
              "target_ret_column", "target_semantics", "reporting_route"]
)
def test_missing_required_field_fails(tmp_path: Path, field: str) -> None:
    d = _bundle_dict(tmp_path)
    d.pop(field)
    path = tmp_path / "models" / "active_bundle.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(BundleError, match="missing required field"):
        load_bundle(path, project_dir=tmp_path)


def test_every_declared_required_field_is_actually_enforced(tmp_path: Path) -> None:
    """Guards the guard: a name in REQUIRED_FIELDS that nothing checks is a lie."""
    for field in REQUIRED_FIELDS:
        d = _bundle_dict(tmp_path)
        d.pop(field)
        path = tmp_path / "models" / f"b_{field}.json"
        path.write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(BundleError, match="missing required field"):
            load_bundle(path, project_dir=tmp_path)


@pytest.mark.parametrize(
    ("field", "bad"),
    [("side", "sideways"), ("feature_semantics", "aligned_v2"),
     ("threshold_operator", "=="), ("same_bar_policy", "favourable"),
     ("candidate_source", "handmade"), ("gap_policy", "open_price"),
     ("return_convention", "guess"), ("target_semantics", "net"),
     ("reporting_route", "best_available")],
)
def test_unknown_enum_value_fails(tmp_path: Path, field: str, bad: str) -> None:
    with pytest.raises(BundleError, match="not in"):
        load_bundle(_write(tmp_path, **{field: bad}), project_dir=tmp_path)


# ── C-06 ──────────────────────────────────────────────────────────────────────
def test_corrupt_bundle_does_not_fall_back(tmp_path: Path) -> None:
    """A corrupt bundle raises. There is no older artifact to quietly serve."""
    path = tmp_path / "models" / "active_bundle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(BundleError, match="unreadable"):
        load_active_bundle(project_dir=tmp_path)


def test_absent_bundle_is_none_for_audit_but_refused_by_production(tmp_path: Path) -> None:
    """No bundle means the owner has not switched this on -- not "pick something".

    The optional reader exposes absence to audit code. The production reader must
    not translate it into permission to use models/ACTIVE or glob for a model.
    """
    assert load_active_bundle(project_dir=tmp_path) is None
    with pytest.raises(BundleError, match="requires an explicit active bundle"):
        require_active_bundle(project_dir=tmp_path)




def test_missing_artefact_file_fails_rather_than_degrading(tmp_path: Path) -> None:
    path = _write(tmp_path)
    (tmp_path / "models" / "det.pt").unlink()
    with pytest.raises(BundleError, match="detector_path does not exist"):
        load_bundle(path, project_dir=tmp_path)




@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("model_num_iteration", 0, "must be positive"),
        ("threshold", float("nan"), "must be finite"),
        ("feature_schema", "unknown_28", "not in"),
        ("model_objective", "ranking", "not in"),
    ],
)
def test_runtime_semantics_are_not_defaulted(tmp_path: Path, field: str, bad, message: str) -> None:
    with pytest.raises(BundleError, match=message):
        load_bundle(_write(tmp_path, **{field: bad}), project_dir=tmp_path)


# ── C-08 / A-06 / D-03 ────────────────────────────────────────────────────────
def test_legacy_semantics_can_never_be_execution_eligible(tmp_path: Path) -> None:
    """The exact half-step that produced the 2026-08-03 live fault.

    A model fitted on unaligned features does not become executable because the
    serving side was repaired; it becomes executable when it is retrained.
    """
    path = _write(tmp_path, execution_eligible=True, paper_only=False,
                  feature_semantics="legacy_unaligned")
    with pytest.raises(BundleError, match="different coordinate system"):
        load_bundle(path, project_dir=tmp_path)


def test_execution_eligible_and_paper_only_are_mutually_exclusive(tmp_path: Path) -> None:
    path = _write(tmp_path, execution_eligible=True, paper_only=True,
                  feature_semantics="side_aligned_v1")
    with pytest.raises(BundleError, match="cannot both be true"):
        load_bundle(path, project_dir=tmp_path)


def test_aligned_bundle_may_be_execution_eligible(tmp_path: Path) -> None:
    """The rule is about semantics, not a blanket ban -- otherwise P2 could never ship."""
    p = load_bundle(
        _write(tmp_path, execution_eligible=True, paper_only=False,
               feature_semantics="side_aligned_v1", selector_status="calibrated",
               calibration_pass_rate=0.1, threshold_equal_rate=0.0),
        project_dir=tmp_path,
    )
    assert p.execution_eligible is True


def test_abnormal_selector_can_be_audited_but_never_execution_eligible(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="selector_status=calibrated"):
        load_bundle(
            _write(
                tmp_path,
                execution_eligible=True,
                paper_only=False,
                feature_semantics="side_aligned_v1",
            ),
            project_dir=tmp_path,
        )


def test_eligible_selector_rate_must_match_declared_quantile(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="abnormal pass/equality rates"):
        load_bundle(
            _write(
                tmp_path,
                execution_eligible=True,
                paper_only=False,
                feature_semantics="side_aligned_v1",
                selector_status="calibrated",
                calibration_pass_rate=0.5,
                threshold_equal_rate=0.0,
            ),
            project_dir=tmp_path,
        )


def test_production_bundle_tip_age_may_not_exceed_two(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="may not exceed 2"):
        load_bundle(_write(tmp_path, max_tip_age_bars=3), project_dir=tmp_path)


def test_target_cost_flag_must_match_target_semantics(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="contradicts target_semantics"):
        load_bundle(
            _write(tmp_path, target_semantics="net_taker", target_cost_included=False),
            project_dir=tmp_path,
        )


def test_side_and_return_convention_must_agree(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="short side requires"):
        load_bundle(_write(tmp_path, return_convention="linear_long"), project_dir=tmp_path)


@pytest.mark.parametrize("value", ["true", 1, "yes"])
def test_eligibility_must_be_a_real_boolean(tmp_path: Path, value) -> None:
    """"false" is a non-empty string and therefore truthy. Not a theoretical risk."""
    with pytest.raises(BundleError, match="must be a JSON boolean"):
        load_bundle(_write(tmp_path, execution_eligible=value), project_dir=tmp_path)


# ── A-05 ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("row_side", "accepted"),
    [("short", True), ("SHORT", True), (" short ", True),
     ("long", False), ("", False), (None, False), ("unknown", False)],
)
def test_row_side_must_match_strategy_side(tmp_path: Path, row_side, accepted: bool) -> None:
    p = load_bundle(_write(tmp_path), project_dir=tmp_path)
    assert p.accepts_row_side(row_side) is accepted


# ── threshold operator ────────────────────────────────────────────────────────
def test_threshold_operator_decides_the_boundary_case(tmp_path: Path) -> None:
    """The plan reports the production gate passing ~91.2% of val on ties.

    Which operator a bundle declares is therefore load-bearing, and a bundle that
    declares ">" must not behave like ">=".
    """
    ge = load_bundle(_write(tmp_path, threshold=0.5, threshold_operator=">="),
                     project_dir=tmp_path)
    gt = load_bundle(_write(tmp_path, threshold=0.5, threshold_operator=">"),
                     project_dir=tmp_path)
    assert ge.passes_threshold(0.5) is True
    assert gt.passes_threshold(0.5) is False
    assert ge.passes_threshold(0.6) is gt.passes_threshold(0.6) is True


# ── the shipped example ───────────────────────────────────────────────────────
def test_repo_example_bundle_is_valid_and_honest() -> None:
    """models/active_bundle.example.json must load, and must not claim executability.

    It describes the current v10, which is legacy-semantics and audit-only. If
    this ever passes while claiming execution_eligible, the plan's D-03 has been
    quietly reversed.
    """
    from yoyo.contracts.paths import data_root
    # The example describes real v10 artefacts, so it lives with the data.
    example = data_root() / "models" / "active_bundle.example.json"
    p = load_bundle(example)
    assert p.side == "short"
    assert p.feature_semantics == "legacy_unaligned"
    assert p.execution_eligible is False
    assert p.paper_only is True


def test_example_is_not_wired_up_as_the_active_bundle() -> None:
    """P0 must not activate anything; models/active_bundle.json stays absent.

    Presence of that file is the owner's switch (iron rule 10, plan D-07/O-03).
    """
    from yoyo.contracts.paths import data_root
    project = data_root()
    assert not (project / "models" / "active_bundle.json").exists()
    assert load_active_bundle(project_dir=project) is None
    with pytest.raises(BundleError, match="requires an explicit active bundle"):
        require_active_bundle(project_dir=project)
