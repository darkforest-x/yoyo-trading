"""Safety, provenance, and resume tests for the fixed-W10 round controller."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import tools.run_fixed_w10_evolution_round as controller


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_script(tmp_path: Path) -> Path:
    script = tmp_path / "emit_stage.py"
    script.write_text(
        """#!/usr/bin/env python3
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--counter', required=True)
p.add_argument('--mode', default='ok')
p.add_argument('--device')
a = p.parse_args()
counter = Path(a.counter)
counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else '1')
if a.mode == 'fail':
    raise SystemExit(7)
payload = {
    'holdout_rows_read': 0,
    'future_rows_in_causal_input': 0,
    'auto_promote': False,
    'nested': {'holdout_rows_read': 0, 'auto_promote': False},
}
Path(a.out).write_text(json.dumps(payload, sort_keys=True) + '\\n')
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _plan(
    tmp_path: Path,
    *,
    fail_kind: str | None = None,
    manual_kind: str | None = None,
) -> tuple[Path, Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = _fixture_script(tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest = snapshot / "manifest.json"
    manifest.write_text('{"scope":"inner"}\n', encoding="utf-8")
    weights = tmp_path / "parent.pt"
    weights.write_bytes(b"pinned-parent-l1")
    state_dir = tmp_path / "round-state"
    counters = tmp_path / "counters"
    counters.mkdir()
    kinds = [
        "preholdout_backtest",
        "preholdout_backtest",
        "merge_inner_candidates",
        "optimizer_l2_expansion",
        "stoploss_feedback",
        "frozen_crosssection_confirmation",
    ]
    stages = []
    previous_artifact: Path | None = None
    backtest_ids: list[str] = []
    previous_id: str | None = None
    for index, kind in enumerate(kinds):
        stage_id = f"s{index}-{kind}"
        artifact = state_dir / "artifacts" / f"{stage_id}.json"
        command = [
            sys.executable,
            str(script),
            "--out", str(artifact),
            "--counter", str(counters / stage_id),
            "--mode", "fail" if fail_kind == kind else "ok",
            "--device", "arbitrary-device-name",
        ]
        if kind == "preholdout_backtest":
            depends_on: list[str] = []
            inputs: list[dict] = []
            backtest_ids.append(stage_id)
        elif kind == "merge_inner_candidates":
            depends_on = list(backtest_ids)
            inputs = [
                {"path": stages[item]["artifacts"][0]["path"]}
                for item in range(len(backtest_ids))
            ]
        else:
            depends_on = [previous_id] if previous_id is not None else []
            inputs = [] if previous_artifact is None else [{"path": str(previous_artifact)}]
        stages.append({
            "id": stage_id,
            "kind": kind,
            "depends_on": depends_on,
            "command": command,
            "inputs": inputs,
            "artifacts": [{"path": str(artifact), "type": "json"}],
            "expected_markers": {
                "holdout_rows_read": 0,
                "future_rows_in_causal_input": 0,
                "auto_promote": False,
            },
            "manual_review_required": manual_kind == kind,
        })
        previous_id, previous_artifact = stage_id, artifact
    raw = {
        "schema_version": 1,
        "round_id": "fixture-round",
        "single_variable": {
            "name": "threshold",
            "baseline": 0.5,
            "candidate": 0.55,
            "only_change": True,
        },
        "snapshot": {
            "path": str(snapshot),
            "manifest_path": "manifest.json",
            "manifest_sha256": _sha(manifest),
            "inner_only": True,
        },
        "parent_l1_weights": {"path": str(weights), "sha256": _sha(weights)},
        "stages": stages,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan_path, state_dir, raw


def _write_plan(path: Path, raw: dict) -> None:
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _controller_process(plan: Path, state_dir: Path, mode: str = "--check") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(Path(controller.__file__)), "--plan", str(plan),
         "--state-dir", str(state_dir), mode],
        check=False, capture_output=True, text=True,
    )


def test_check_prints_ordered_dag_without_creating_state(tmp_path: Path, capsys) -> None:
    plan, state_dir, _ = _plan(tmp_path)
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--check"]) == 0
    output = capsys.readouterr().out
    assert "preholdout_backtest" in output
    assert "frozen_crosssection_confirmation" in output
    assert not state_dir.exists()


@pytest.mark.parametrize("mutation", ["post_merge_backtest", "missing_merge_dependency", "duplicate_merge"])
def test_strict_dag_rejects_review_reproductions(tmp_path: Path, mutation: str) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    if mutation == "post_merge_backtest":
        raw["stages"][2], raw["stages"][1] = raw["stages"][1], raw["stages"][2]
    elif mutation == "missing_merge_dependency":
        raw["stages"][2]["depends_on"] = raw["stages"][2]["depends_on"][:1]
        raw["stages"][2]["inputs"] = raw["stages"][2]["inputs"][:1]
    else:
        duplicate = dict(raw["stages"][2])
        duplicate["id"] = "duplicate-merge"
        duplicate["artifacts"] = [{"path": str(state_dir / "duplicate.json"), "type": "json"}]
        raw["stages"].insert(3, duplicate)
    _write_plan(plan, raw)
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--check"]) == 2


def test_strict_dag_rejects_nonroot_backtest_and_skipped_chain_dependency(tmp_path: Path) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    raw["stages"][1]["depends_on"] = [raw["stages"][0]["id"]]
    _write_plan(plan, raw)
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--check"]) == 2

    plan, state_dir, raw = _plan(tmp_path / "skip")
    raw["stages"][4]["depends_on"] = [raw["stages"][2]["id"]]
    _write_plan(plan, raw)
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--check"]) == 2


def test_run_and_resume_skip_completed_commands(tmp_path: Path) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    argv = ["--plan", str(plan), "--state-dir", str(state_dir), "--run"]
    assert controller.main(argv) == 0
    state = json.loads((state_dir / "state.json").read_text())
    assert state["status"] == "completed"
    assert all(record["exit_code"] == 0 for record in state["stages"].values())
    assert all(record["command_hash"] for record in state["stages"].values())
    assert all(record["input_sha256"] for record in state["stages"].values())
    assert all(record["artifact_sha256"] for record in state["stages"].values())
    assert all(record["command_files"]["executable"]["sha256"] for record in state["stages"].values())
    assert all(record["command_files"]["script"]["sha256"] for record in state["stages"].values())
    counters = [Path(stage["command"][5]).read_text() for stage in raw["stages"]]
    assert counters == ["1"] * len(raw["stages"])

    assert controller.main(argv) == 0
    assert [Path(stage["command"][5]).read_text() for stage in raw["stages"]] == counters
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--status"]) == 0


def test_resume_rejects_changed_script_bytes_in_fresh_process(tmp_path: Path) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    command = [sys.executable, str(Path(controller.__file__)), "--plan", str(plan),
               "--state-dir", str(state_dir), "--run"]
    assert subprocess.run(command, check=False).returncode == 0
    script = Path(raw["stages"][0]["command"][1])
    script.write_text(script.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    retried = subprocess.run(command, check=False, capture_output=True, text=True)
    assert retried.returncode == 2
    assert "command" in retried.stderr and "changed" in retried.stderr


@pytest.mark.parametrize("options", [["-u"], ["-u", "-B"], ["-uB"], ["-W", "ignore"], ["-Wignore"]])
def test_interpreter_options_do_not_unbind_the_script(tmp_path: Path, options: list[str]) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    script = Path(raw["stages"][0]["command"][1])
    for stage in raw["stages"]:
        stage["command"][1:1] = options
    _write_plan(plan, raw)
    normalized = controller.validate_plan(plan, state_dir)
    assert normalized["stages"][0]["command_files"]["script"] == {
        "path": str(script), "sha256": _sha(script),
    }


def test_resume_rejects_changed_script_bytes_behind_an_interpreter_option(tmp_path: Path) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    script = Path(raw["stages"][0]["command"][1])
    for stage in raw["stages"]:
        stage["command"].insert(1, "-u")
    _write_plan(plan, raw)
    command = [sys.executable, str(Path(controller.__file__)), "--plan", str(plan),
               "--state-dir", str(state_dir), "--run"]
    assert subprocess.run(command, check=False).returncode == 0
    script.write_text(script.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    retried = subprocess.run(command, check=False, capture_output=True, text=True)
    assert retried.returncode == 2
    assert "command" in retried.stderr and "changed" in retried.stderr


@pytest.mark.parametrize("options", [["-c"], ["-m"], ["-uc"], ["-um"]])
def test_interpreter_code_options_are_refused(tmp_path: Path, options: list[str]) -> None:
    """``-c``/``-m`` run bytes with no file to bind, so the journal could not cover them."""
    plan, state_dir, raw = _plan(tmp_path)
    for stage in raw["stages"]:
        stage["command"][1:1] = options
    _write_plan(plan, raw)
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--check"]) == 2


def test_python_script_without_extension_is_sha_bound(tmp_path: Path) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    original = Path(raw["stages"][0]["command"][1])
    extensionless = original.with_name("stage_runner")
    extensionless.write_bytes(original.read_bytes())
    for stage in raw["stages"]:
        stage["command"][1] = extensionless.name
    _write_plan(plan, raw)
    normalized = controller.validate_plan(plan, state_dir)
    assert normalized["stages"][0]["command_files"]["script"] == {
        "path": str(extensionless), "sha256": _sha(extensionless),
    }


def test_resume_rejects_executable_bytes_change(tmp_path: Path) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    executable = tmp_path / "python-wrapper"
    executable.write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n", encoding="utf-8")
    executable.chmod(0o755)
    for stage in raw["stages"]:
        stage["command"][0] = str(executable)
    _write_plan(plan, raw)
    argv = ["--plan", str(plan), "--state-dir", str(state_dir), "--run"]
    assert controller.main(argv) == 0
    executable.write_text(executable.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
    assert controller.main(argv) == 2


def test_resume_rejects_tampered_artifact_and_input(tmp_path: Path, capsys) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    argv = ["--plan", str(plan), "--state-dir", str(state_dir), "--run"]
    assert controller.main(argv) == 0
    first = Path(raw["stages"][0]["artifacts"][0]["path"])
    first.write_text('{"holdout_rows_read":0}\n', encoding="utf-8")
    assert controller.main(argv) == 2
    assert "artifact" in capsys.readouterr().err

    # A pinned root input is checked before any process can be resumed.
    weights = Path(raw["parent_l1_weights"]["path"])
    weights.write_bytes(b"tampered")
    assert controller.main(argv) == 2


def test_failed_stage_is_journaled_and_later_stages_do_not_run(tmp_path: Path) -> None:
    plan, state_dir, raw = _plan(tmp_path, fail_kind="optimizer_l2_expansion")
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--run"]) == 1
    state = json.loads((state_dir / "state.json").read_text())
    failed_id = next(stage["id"] for stage in raw["stages"] if stage["kind"] == "optimizer_l2_expansion")
    assert state["stages"][failed_id]["status"] == "failed"
    assert state["stages"][failed_id]["exit_code"] == 7
    assert Path(state["stages"][failed_id]["stdout_log"]).is_file()
    later = raw["stages"][raw["stages"].index(next(s for s in raw["stages"] if s["id"] == failed_id)) + 1]
    assert not Path(later["command"][5]).exists()


@pytest.mark.parametrize(
    "token",
    [
        "--evaluate-sealed",
        "--source=optimization_sealed.jsonl",
        "--promote",
        "--order",
        "ACTIVE",
        "--holdout",
        "--source=holdout_losers3d",
        "2026-04-02T00:00:00Z",
        "2026-05-04",
    ],
)
def test_forbidden_arguments_and_dates_are_rejected(tmp_path: Path, token: str) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    raw["stages"][0]["command"].append(token)
    _write_plan(plan, raw)
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--check"]) == 2


@pytest.mark.parametrize("token", ["--scope=preholdout_inner_only", "preholdout_backtest"])
def test_preholdout_identity_survives_the_holdout_token_guard(tmp_path: Path, token: str) -> None:
    """The pipeline's own stages are named ``*preholdout*``; the guard must not eat them."""
    plan, state_dir, raw = _plan(tmp_path)
    raw["stages"][0]["command"].append(token)
    _write_plan(plan, raw)
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--check"]) == 0


def test_date_override_never_allows_holdout(tmp_path: Path) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    raw["allow_post_sealed_dates"] = True
    raw["stages"][0]["command"].append("2026-04-03")
    _write_plan(plan, raw)
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--check"]) == 0
    raw["stages"][0]["command"][-1] = "2026-05-04"
    _write_plan(plan, raw)
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--check"]) == 2


@pytest.mark.parametrize(
    "date",
    ["20260402", "2026/04/02", "2026_04_02", "2026-04-02T12:00:00Z", "20260504"],
)
def test_all_review_date_formats_are_rejected_recursively(tmp_path: Path, date: str) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    raw["single_variable"]["review_note"] = f"boundary={date}"
    _write_plan(plan, raw)
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--check"]) == 2


def test_date_guard_covers_plan_file_path_in_fresh_process(tmp_path: Path) -> None:
    plan, state_dir, _ = _plan(tmp_path)
    dated = tmp_path / "2026_04_02" / "plan.json"
    dated.parent.mkdir()
    dated.write_bytes(plan.read_bytes())
    result = _controller_process(dated, state_dir)
    assert result.returncode == 2
    assert "sealed range" in result.stderr


@pytest.mark.parametrize("location", ["plan", "snapshot", "weights", "input", "artifact", "argv"])
def test_forbidden_identity_is_rejected_at_every_path_surface(tmp_path: Path, location: str) -> None:
    root = tmp_path / "ordinary-parent-with-unsealed-substring"
    plan, state_dir, raw = _plan(root)
    if location == "plan":
        target = root / "sealed" / "plan.json"
        target.parent.mkdir()
        target.write_bytes(plan.read_bytes())
        plan = target
    elif location == "snapshot":
        raw["snapshot"]["path"] = str(root / "sealed" / "snapshot")
    elif location == "weights":
        raw["parent_l1_weights"]["path"] = str(root / "ACTIVE" / "parent.pt")
    elif location == "input":
        raw["stages"][0]["inputs"] = [{"path": str(root / "orders" / "config.json"), "sha256": "a" * 64}]
    elif location == "artifact":
        raw["stages"][0]["artifacts"][0]["path"] = str(state_dir / "promote" / "summary.json")
    else:
        raw["stages"][0]["command"].append(str(root / "evaluate-sealed" / "candidate.json"))
    _write_plan(plan, raw)
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--check"]) == 2


def test_identity_guard_avoids_substring_false_positives(tmp_path: Path) -> None:
    plan, state_dir, raw = _plan(tmp_path / "unsealed-orderbook-activewear")
    raw["stages"][0]["command"].append("inactive-candidate")
    _write_plan(plan, raw)
    assert _controller_process(plan, state_dir).returncode == 0

    plan, state_dir, _ = _plan(tmp_path / "test-mentions-sealed-path")
    assert _controller_process(plan, state_dir).returncode == 0


def test_review_block_reproductions_fail_in_fresh_process(tmp_path: Path) -> None:
    # DAG: a backtest moved behind merge.
    plan, state_dir, raw = _plan(tmp_path / "dag")
    raw["stages"][1], raw["stages"][2] = raw["stages"][2], raw["stages"][1]
    _write_plan(plan, raw)
    assert _controller_process(plan, state_dir).returncode == 2

    # Path identity: forbidden output destination.
    plan, state_dir, raw = _plan(tmp_path / "path")
    raw["stages"][0]["artifacts"][0]["path"] = str(state_dir / "ACTIVE" / "summary.json")
    _write_plan(plan, raw)
    assert _controller_process(plan, state_dir).returncode == 2

    # Date: compact holdout value nested outside argv.
    plan, state_dir, raw = _plan(tmp_path / "date")
    raw["single_variable"]["note"] = {"nested": "20260504"}
    _write_plan(plan, raw)
    assert _controller_process(plan, state_dir).returncode == 2


def test_marker_contract_and_recursive_artifact_violations_fail(tmp_path: Path) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    raw["stages"][0]["expected_markers"]["auto_promote"] = True
    _write_plan(plan, raw)
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--check"]) == 2

    plan, state_dir, raw = _plan(tmp_path / "artifact-case")
    script = Path(raw["stages"][0]["command"][1])
    source = script.read_text().replace("'auto_promote': False", "'auto_promote': True", 1)
    script.write_text(source, encoding="utf-8")
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--run"]) == 2
    state = json.loads((state_dir / "state.json").read_text())
    assert next(iter(state["stages"].values()))["status"] == "failed_validation"


def test_relative_command_path_escape_and_nonempty_fresh_state_are_rejected(tmp_path: Path) -> None:
    plan, state_dir, raw = _plan(tmp_path)
    raw["stages"][0]["command"][1] = "../emit_stage.py"
    _write_plan(plan, raw)
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--check"]) == 2

    plan, state_dir, _ = _plan(tmp_path / "nonempty")
    state_dir.mkdir(parents=True)
    (state_dir / "owner-file").write_text("preserve", encoding="utf-8")
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--run"]) == 2
    assert (state_dir / "owner-file").read_text() == "preserve"
    assert not (state_dir / "state.json").exists()


def test_manual_review_stops_cleanly_without_running_following_stage(tmp_path: Path) -> None:
    plan, state_dir, raw = _plan(tmp_path, manual_kind="merge_inner_candidates")
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--run"]) == 0
    state = json.loads((state_dir / "state.json").read_text())
    assert state["status"] == "manual_review_required"
    stop_index = next(i for i, stage in enumerate(raw["stages"]) if stage["kind"] == "merge_inner_candidates")
    assert Path(raw["stages"][stop_index]["command"][5]).read_text() == "1"
    assert not Path(raw["stages"][stop_index + 1]["command"][5]).exists()
    # A plain resume cannot silently turn review into approval.
    assert controller.main(["--plan", str(plan), "--state-dir", str(state_dir), "--run"]) == 0
    assert not Path(raw["stages"][stop_index + 1]["command"][5]).exists()


def test_atomic_state_write_preserves_old_file_if_replace_fails(tmp_path: Path, monkeypatch) -> None:
    state = tmp_path / "state.json"
    state.write_text('{"old":true}\n', encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(controller.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        controller.atomic_write_state(state, {"new": True})
    assert json.loads(state.read_text()) == {"old": True}
    assert list(tmp_path.glob(".state.*.tmp")) == []
