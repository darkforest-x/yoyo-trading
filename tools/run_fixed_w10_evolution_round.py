#!/usr/bin/env python3
"""Run one pinned, INNER-only fixed-W10 evolution round fail-closed.

This is an orchestration and provenance tool, not a model or research tool.
The plan owns the exact argv and artifact contract for every existing tool in
the round.  This controller only validates immutable inputs, executes argv
without a shell, checks the common causal/holdout/non-promotion markers, and
atomically journals enough SHA256 evidence for a safe resume.

Plan schema (version 1)::

    {
      "schema_version": 1,
      "round_id": "r1",
      "single_variable": {
        "name": "l1_threshold", "baseline": 0.5,
        "candidate": 0.55, "only_change": true
      },
      "snapshot": {
        "path": "/.../inner_snapshot", "manifest_path": "manifest.json",
        "manifest_sha256": "...", "inner_only": true
      },
      "parent_l1_weights": {"path": "/.../best.pt", "sha256": "..."},
      "stages": [{
        "id": "backtest-a", "kind": "preholdout_backtest",
        "depends_on": [], "command": ["python3", "tool.py", "..."],
        "inputs": [{"path": "/.../config.json", "sha256": "..."}],
        "artifacts": [{"path": "/.../summary.json", "type": "json"}],
        "expected_markers": {
          "holdout_rows_read": 0,
          "future_rows_in_causal_input": 0,
          "auto_promote": false
        },
        "manual_review_required": false
      }]
    }

Relative snapshot/input/command paths are relative to the plan directory.
Relative generated artifact paths are relative to ``--state-dir`` and may not
escape it.  A generated artifact can be a later input without a plan SHA; its
SHA is then taken only from a completed dependency's state record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
SEALED_START = datetime(2026, 4, 2, tzinfo=timezone.utc)
HOLDOUT_START = datetime(2026, 5, 4, tzinfo=timezone.utc)
REQUIRED_MARKERS: dict[str, Any] = {
    "holdout_rows_read": 0,
    "future_rows_in_causal_input": 0,
    "auto_promote": False,
}
STAGE_ALIASES = {
    "preholdout_backtest": "preholdout_backtest",
    "merge_inner_candidates": "merge_inner_candidates",
    "optimizer_l2_expansion": "optimizer_or_l2_gate",
    "optimization_or_l2": "optimizer_or_l2_gate",
    "optimizer_or_l2": "optimizer_or_l2_gate",
    "optimizer_or_l2_gate": "optimizer_or_l2_gate",
    "l2_walkforward_gate": "optimizer_or_l2_gate",
    "walkforward_gate": "optimizer_or_l2_gate",
    "stoploss_feedback": "stoploss_feedback",
    "frozen_crosssection_confirmation": "frozen_crosssection_confirmation",
    "crosssection_confirmation": "frozen_crosssection_confirmation",
}
REQUIRED_KIND_ORDER = (
    "preholdout_backtest",
    "merge_inner_candidates",
    "optimizer_or_l2_gate",
    "stoploss_feedback",
    "frozen_crosssection_confirmation",
)
DATE_RE = re.compile(
    r"(?<!\d)(?P<year>20\d{2})(?:"
    r"(?P<separator>[-/_])(?P<month>\d{2})(?P=separator)(?P<day>\d{2})"
    r"|(?P<compact_month>\d{2})(?P<compact_day>\d{2})"
    r")(?!\d)"
)
# "holdout" is a token, not a substring: `_identity_tokens` splits on non-alphanumerics,
# so the pipeline's own `*_preholdout*` identities stay one word and are not caught here.
# Without it the date guard was the only holdout defence, and it only sees dated strings --
# a plan carrying `--holdout <name>` passed `--check` clean.
FORBIDDEN_IDENTITY_TOKENS = frozenset({
    "sealed", "promote", "order", "orders", "active", "holdout",
})
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ControllerError(ValueError):
    """A fail-closed plan, provenance, or execution contract violation."""


def sha256_path(path: Path) -> str:
    """Hash a file, or a directory as sorted relative names plus file bytes."""
    if path.is_symlink():
        raise ControllerError(f"symlinks are not accepted as pinned paths: {path}")
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if any(item.is_symlink() for item in path.rglob("*")):
            raise ControllerError(f"directory contains a symlink: {path}")
        for item in files:
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        return digest.hexdigest()
    raise ControllerError(f"path does not exist: {path}")


def _require_sha(value: Any, label: str) -> str:
    digest = str(value or "").lower()
    if not SHA_RE.fullmatch(digest):
        raise ControllerError(f"{label} must be a 64-character SHA256")
    return digest


def _checked_file(path: Path, expected: Any, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise ControllerError(f"{label} is not a regular file: {path}")
    wanted = _require_sha(expected, f"{label} sha256")
    actual = sha256_path(path)
    if actual != wanted:
        raise ControllerError(f"{label} SHA256 mismatch: {path}")
    return actual


def _resolve_existing(raw: Any, base: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ControllerError(f"{label} must be a non-empty path")
    path = Path(raw).expanduser()
    path = (path if path.is_absolute() else base / path).resolve()
    _validate_identity(path, label)
    if not path.exists():
        raise ControllerError(f"{label} does not exist: {path}")
    return path


def _resolve_artifact(raw: Any, state_dir: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ControllerError(f"{label} must be a non-empty path")
    original = Path(raw).expanduser()
    path = (original if original.is_absolute() else state_dir / original).resolve()
    _validate_identity(path, label)
    if not original.is_absolute() and not path.is_relative_to(state_dir):
        raise ControllerError(f"relative artifact path escapes --state-dir: {raw}")
    logs_dir = state_dir / "logs"
    if path in {state_dir, state_dir / "state.json", logs_dir} or logs_dir in path.parents:
        raise ControllerError(f"artifact path collides with controller state: {path}")
    return path


def _matched_date(match: re.Match[str]) -> datetime:
    month = match.group("month") or match.group("compact_month")
    day = match.group("day") or match.group("compact_day")
    try:
        return datetime(int(match.group("year")), int(month), int(day), tzinfo=timezone.utc)
    except ValueError as exc:
        raise ControllerError(f"invalid date in plan: {match.group(0)}") from exc


def _validate_dates(values: Iterable[str], *, allow_post_sealed_dates: bool) -> None:
    for value in values:
        if SHA_RE.fullmatch(value.lower()):
            continue
        for match in DATE_RE.finditer(value):
            stamp = _matched_date(match)
            if stamp >= HOLDOUT_START:
                raise ControllerError(f"date reaches forbidden holdout (>=2026-05-04): {match.group(0)}")
            if stamp >= SEALED_START and not allow_post_sealed_dates:
                raise ControllerError(f"date reaches sealed range by default (>=2026-04-02): {match.group(0)}")


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _command_path(token: str, plan_dir: Path) -> Path | None:
    candidate = Path(token).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if "/" in token or token.endswith((".py", ".sh")):
        return (plan_dir / candidate).resolve()
    return None


def _identity_tokens(value: str | Path) -> set[str]:
    """Return semantic identity tokens without mining incidental path parents."""
    if isinstance(value, Path) or "/" in str(value) or "\\" in str(value):
        path = Path(value)
        parts = list(path.parts)
        tokens = {part.lower() for part in parts}
        for index, part in enumerate(parts):
            words = re.findall(r"[a-z0-9]+", part.lower())
            # The leaf is the declared identity.  Ancestors are identities only
            # when the forbidden word is their leading/trailing qualifier;
            # this avoids treating an incidental pytest parent such as
            # ``test_rejects_sealed_path`` as the plan's intended identity.
            if index == len(parts) - 1:
                tokens.update(words)
            elif words and (words[0] in FORBIDDEN_IDENTITY_TOKENS or
                            words[-1] in FORBIDDEN_IDENTITY_TOKENS):
                tokens.update(words)
        return tokens
    raw = str(value)
    if raw.startswith("-"):
        flag, separator, argument = raw.partition("=")
        tokens = set(re.findall(r"[a-z0-9]+", flag.lower()))
        if separator:
            tokens.update(_identity_tokens(argument))
        return tokens
    return set(re.findall(r"[a-z0-9]+", raw.lower()))


def _validate_identity(value: str | Path, label: str) -> None:
    forbidden = sorted(_identity_tokens(value) & FORBIDDEN_IDENTITY_TOKENS)
    if forbidden:
        raise ControllerError(f"{label} has forbidden identity {forbidden[0]!r}: {value}")


def _command_file_records(command: Sequence[str], script_index: int | None) -> dict[str, dict[str, str]]:
    executable = Path(command[0])
    records = {
        "executable": {"path": str(executable), "sha256": sha256_path(executable)},
    }
    if script_index is not None:
        script = Path(command[script_index])
        if script.is_absolute() and script.is_file():
            records["script"] = {"path": str(script), "sha256": sha256_path(script)}
    return records


def _is_python_executable(path: Path) -> bool:
    return re.fullmatch(r"(?:python|pypy)\d*(?:\.\d+)*", path.name.lower()) is not None


# Interpreter options that consume the rest of their cluster, or the next token.
_PYTHON_OPTIONS_WITH_ARGUMENT = frozenset({"W", "X"})
# Options that replace the script with unbindable code: nothing on disk to SHA.
_PYTHON_OPTIONS_WITHOUT_SCRIPT = frozenset({"c", "m"})


def _python_script_index(command: Sequence[str]) -> int | None:
    """Return the argv index of the interpreter's script, skipping its own options.

    ``python -u stage.py`` used to leave ``argv[1] == "-u"``, so the real script
    never entered ``command_files`` and its bytes could change between a completed
    run and a resume without the journal noticing.  ``-c``/``-m`` run code that has
    no file to bind, so they are refused rather than silently left unbound.
    """
    index = 1
    while index < len(command):
        token = command[index]
        if not token.startswith("-") or token == "-":
            return index
        if token.startswith("--"):
            flag = token.partition("=")[0]
            if flag == "--check-hash-based-pycs" and "=" not in token:
                index += 1
            index += 1
            continue
        cluster = token[1:]
        for position, letter in enumerate(cluster):
            if letter in _PYTHON_OPTIONS_WITHOUT_SCRIPT:
                raise ControllerError(
                    f"interpreter option -{letter} runs code that cannot be SHA-bound: {token}"
                )
            if letter in _PYTHON_OPTIONS_WITH_ARGUMENT:
                # ``-Wignore`` carries its argument; bare ``-W`` takes the next token.
                if position == len(cluster) - 1:
                    index += 1
                break
        index += 1
    return None


def _validate_command(
    command: Any, plan_dir: Path, allow_post_sealed_dates: bool,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
        raise ControllerError("stage command must be a non-empty list of non-empty strings")
    for token in command:
        if "\x00" in token:
            raise ControllerError("stage command contains a NUL byte")
        if not Path(token).is_absolute() and ".." in Path(token).parts:
            raise ControllerError(f"relative command path escaping is forbidden: {token}")
        _validate_identity(token, "command argument")
    _validate_dates(command, allow_post_sealed_dates=allow_post_sealed_dates)

    executable = command[0]
    executable_path = _command_path(executable, plan_dir)
    if executable_path is None:
        found = shutil.which(executable)
        if found is None:
            raise ControllerError(f"command executable not found: {executable}")
        executable_path = Path(found).resolve()
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise ControllerError(f"command executable is not executable: {executable_path}")
    _validate_identity(executable_path, "resolved command executable")

    resolved = list(command)
    resolved[0] = str(executable_path)
    # Only the script slot can be an interpreter's script path -- argv[1] for a
    # plain executable, the first non-option token for an interpreter.  Later
    # path-looking values can legitimately be not-yet-created artifact
    # destinations; those are validated through the explicit input/artifact
    # declarations instead.
    is_python = _is_python_executable(executable_path)
    script_index = _python_script_index(command) if is_python else (1 if len(command) > 1 else None)
    bound_index: int | None = None
    if script_index is not None:
        token = command[script_index]
        path = _command_path(token, plan_dir)
        if path is None and is_python and not token.startswith("-"):
            path = (plan_dir / Path(token).expanduser()).resolve()
        if path is not None:
            if ".." in Path(token).parts and not Path(token).is_absolute():
                raise ControllerError(f"relative command path escaping is forbidden: {token}")
            if not path.is_file() or path.is_symlink():
                raise ControllerError(f"command script path does not exist: {path}")
            _validate_identity(path, "resolved command script")
            resolved[script_index] = str(path)
            bound_index = script_index
    return resolved, _command_file_records(resolved, bound_index)


def _json_object(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(raw, Mapping):
        raise ControllerError(f"JSON artifact root must be an object: {path}")
    return raw


def _json_documents(path: Path) -> Iterable[tuple[Path, Mapping[str, Any]]]:
    files: list[Path]
    if path.is_dir():
        files = sorted([*path.rglob("*.json"), *path.rglob("*.jsonl")])
    elif path.suffix.lower() in {".json", ".jsonl"}:
        files = [path]
    else:
        files = []
    for file in files:
        if file.suffix.lower() == ".json":
            yield file, _json_object(file)
            continue
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise ControllerError(f"invalid JSONL artifact: {file}") from exc
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ControllerError(f"invalid JSONL artifact: {file}:{number}") from exc
            if not isinstance(raw, Mapping):
                raise ControllerError(f"JSONL row must be an object: {file}:{number}")
            yield file, raw


def _marker_values(value: Any, found: dict[str, list[Any]]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in REQUIRED_MARKERS:
                found[key].append(item)
            _marker_values(item, found)
    elif isinstance(value, list):
        for item in value:
            _marker_values(item, found)


def _verify_markers(paths: Sequence[Path], expected: Mapping[str, Any]) -> None:
    found = {key: [] for key in REQUIRED_MARKERS}
    documents = 0
    for artifact in paths:
        for _, raw in _json_documents(artifact):
            documents += 1
            _marker_values(raw, found)
    if not documents:
        raise ControllerError("stage produced no declared JSON/JSONL artifact to prove safety markers")
    for key, wanted in REQUIRED_MARKERS.items():
        if expected.get(key) != wanted or type(expected.get(key)) is not type(wanted):
            raise ControllerError(f"expected_markers must declare {key}={wanted!r}")
        if not found[key]:
            raise ControllerError(f"JSON artifacts do not declare required marker {key}")
        if any(value != wanted or type(value) is not type(wanted) for value in found[key]):
            raise ControllerError(f"JSON artifact marker violation: {key} must be {wanted!r}")


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_state(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace state without exposing a partial JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp.exists():
            temp.unlink()


def _read_plan(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ControllerError(f"plan does not exist: {path}")
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ControllerError(f"cannot read plan JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ControllerError("plan root must be a JSON object")
    return raw, hashlib.sha256(raw_bytes).hexdigest()


def validate_plan(plan_path: Path, state_dir: Path) -> dict[str, Any]:
    """Validate and normalize a complete immutable round plan."""
    plan_path = plan_path.resolve()
    state_dir = state_dir.expanduser().resolve()
    _validate_identity(plan_path, "plan path")
    _validate_identity(state_dir, "state directory")
    plan, plan_sha = _read_plan(plan_path)
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ControllerError(f"plan schema_version must be {SCHEMA_VERSION}")
    round_id = plan.get("round_id")
    if not isinstance(round_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", round_id):
        raise ControllerError("round_id must be a safe non-empty identifier")
    discipline = plan.get("single_variable")
    if not isinstance(discipline, Mapping) or discipline.get("only_change") is not True:
        raise ControllerError("single_variable.only_change must be true")
    if not isinstance(discipline.get("name"), str) or not discipline["name"]:
        raise ControllerError("single_variable.name is required")
    if "baseline" not in discipline or "candidate" not in discipline:
        raise ControllerError("single_variable requires explicit baseline and candidate values")

    allow_post_sealed = plan.get("allow_post_sealed_dates", False)
    if type(allow_post_sealed) is not bool:
        raise ControllerError("allow_post_sealed_dates must be boolean")
    _validate_dates(
        (str(plan_path), str(state_dir)),
        allow_post_sealed_dates=allow_post_sealed,
    )
    _validate_dates(_strings(plan), allow_post_sealed_dates=allow_post_sealed)

    snapshot = plan.get("snapshot")
    if not isinstance(snapshot, Mapping) or snapshot.get("inner_only") is not True:
        raise ControllerError("snapshot.inner_only must be true")
    snapshot_dir = _resolve_existing(snapshot.get("path"), plan_path.parent, "snapshot.path")
    if not snapshot_dir.is_dir():
        raise ControllerError("snapshot.path must be a directory")
    manifest_raw = snapshot.get("manifest_path", "manifest.json")
    if not isinstance(manifest_raw, str) or not manifest_raw:
        raise ControllerError("snapshot.manifest_path must be a path")
    manifest_candidate = Path(manifest_raw).expanduser()
    manifest = (manifest_candidate if manifest_candidate.is_absolute() else snapshot_dir / manifest_candidate).resolve()
    _validate_identity(manifest, "snapshot manifest")
    if not manifest.is_relative_to(snapshot_dir):
        raise ControllerError("snapshot manifest escapes snapshot.path")
    snapshot_sha = _checked_file(manifest, snapshot.get("manifest_sha256"), "snapshot manifest")

    weights = plan.get("parent_l1_weights")
    if not isinstance(weights, Mapping):
        raise ControllerError("parent_l1_weights object is required")
    weights_path = _resolve_existing(weights.get("path"), plan_path.parent, "parent_l1_weights.path")
    weights_sha = _checked_file(weights_path, weights.get("sha256"), "parent L1 weights")

    raw_stages = plan.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ControllerError("stages must be a non-empty list")
    normalized_stages: list[dict[str, Any]] = []
    ids: set[str] = set()
    kinds: list[str] = []
    produced: dict[Path, str] = {}
    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, Mapping):
            raise ControllerError(f"stage {index} must be an object")
        stage_id = raw.get("id")
        if not isinstance(stage_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", stage_id):
            raise ControllerError(f"stage {index} has an unsafe id")
        if stage_id in ids:
            raise ControllerError(f"duplicate stage id: {stage_id}")
        kind = STAGE_ALIASES.get(str(raw.get("kind", "")))
        if kind is None:
            raise ControllerError(f"unsupported stage kind: {raw.get('kind')}")
        depends = raw.get("depends_on")
        if not isinstance(depends, list) or any(not isinstance(item, str) for item in depends):
            raise ControllerError(f"{stage_id}.depends_on must be a string list")
        if len(set(depends)) != len(depends) or any(item not in ids for item in depends):
            raise ControllerError(f"{stage_id}.depends_on must name unique earlier stages")
        command, command_files = _validate_command(
            raw.get("command"), plan_path.parent, allow_post_sealed,
        )
        markers = raw.get("expected_markers")
        if not isinstance(markers, Mapping) or dict(markers) != REQUIRED_MARKERS:
            raise ControllerError(f"{stage_id}.expected_markers must exactly equal the required safety markers")
        manual = raw.get("manual_review_required", False)
        if type(manual) is not bool:
            raise ControllerError(f"{stage_id}.manual_review_required must be boolean")

        artifacts_raw = raw.get("artifacts")
        if not isinstance(artifacts_raw, list) or not artifacts_raw:
            raise ControllerError(f"{stage_id}.artifacts must be a non-empty list")
        artifacts: list[dict[str, Any]] = []
        for number, declaration in enumerate(artifacts_raw):
            if not isinstance(declaration, Mapping):
                raise ControllerError(f"{stage_id} artifact {number} must be an object")
            path = _resolve_artifact(declaration.get("path"), state_dir, f"{stage_id} artifact")
            if path in produced:
                raise ControllerError(f"artifact path declared by multiple stages: {path}")
            artifact_type = declaration.get("type")
            if artifact_type not in {"file", "json", "jsonl", "directory"}:
                raise ControllerError(f"{stage_id} artifact type is invalid: {artifact_type}")
            expected_sha = declaration.get("sha256")
            if expected_sha is not None:
                expected_sha = _require_sha(expected_sha, f"{stage_id} artifact sha256")
            artifacts.append({"path": path, "type": artifact_type, "sha256": expected_sha})
            produced[path] = stage_id

        inputs_raw = raw.get("inputs", [])
        if not isinstance(inputs_raw, list):
            raise ControllerError(f"{stage_id}.inputs must be a list")
        inputs: list[dict[str, Any]] = []
        seen_inputs: set[Path] = set()
        for number, declaration in enumerate(inputs_raw):
            if not isinstance(declaration, Mapping):
                raise ControllerError(f"{stage_id} input {number} must be an object")
            path_raw = declaration.get("path")
            if not isinstance(path_raw, str) or not path_raw:
                raise ControllerError(f"{stage_id} input path must be non-empty")
            candidate = Path(path_raw).expanduser()
            if candidate.is_absolute():
                path = candidate.resolve()
            else:
                generated_candidate = (state_dir / candidate).resolve()
                path = generated_candidate if generated_candidate in produced else (plan_path.parent / candidate).resolve()
            _validate_identity(path, f"{stage_id} input")
            if path in seen_inputs:
                raise ControllerError(f"duplicate input path in {stage_id}: {path}")
            expected_sha = declaration.get("sha256")
            producer = produced.get(path)
            if expected_sha is None:
                if producer is None or producer not in depends:
                    raise ControllerError(f"{stage_id} input without SHA must be an artifact of a dependency: {path}")
            else:
                expected_sha = _require_sha(expected_sha, f"{stage_id} input sha256")
                actual_sha = sha256_path(path)
                if actual_sha != expected_sha:
                    raise ControllerError(f"{stage_id} input SHA256 mismatch: {path}")
            inputs.append({"path": path, "sha256": expected_sha, "producer": producer})
            seen_inputs.add(path)

        command_hash = _canonical_hash({"argv": command, "files": command_files})
        normalized_stages.append({
            "id": stage_id, "kind": kind, "depends_on": list(depends),
            "command": command, "command_hash": command_hash,
            "command_files": command_files,
            "inputs": inputs, "artifacts": artifacts,
            "expected_markers": dict(markers), "manual_review_required": manual,
        })
        ids.add(stage_id)
        kinds.append(kind)

    backtest_count = 0
    while backtest_count < len(kinds) and kinds[backtest_count] == "preholdout_backtest":
        backtest_count += 1
    if backtest_count == 0 or kinds != [
        *(["preholdout_backtest"] * backtest_count), *REQUIRED_KIND_ORDER[1:],
    ]:
        raise ControllerError("stages must be ordered: backtest(s), merge, optimizer/L2 gate, stoploss, frozen confirmation")
    backtests = normalized_stages[:backtest_count]
    if any(stage["depends_on"] for stage in backtests):
        raise ControllerError("preholdout backtests must be independent DAG roots")
    backtest_ids = {stage["id"] for stage in backtests}
    merge = normalized_stages[backtest_count]
    if set(merge["depends_on"]) != backtest_ids:
        raise ControllerError("merge stage must directly depend on every preholdout backtest")
    previous = merge
    for stage in normalized_stages[backtest_count + 1:]:
        if stage["depends_on"] != [previous["id"]]:
            raise ControllerError(f"{stage['id']} must directly and only depend on prior stage {previous['id']}")
        previous = stage

    return {
        "round_id": round_id, "plan_path": plan_path, "plan_sha256": plan_sha,
        "state_dir": state_dir, "snapshot_path": snapshot_dir,
        "snapshot_manifest_path": manifest, "snapshot_manifest_sha256": snapshot_sha,
        "weights_path": weights_path, "weights_sha256": weights_sha,
        "allow_post_sealed_dates": allow_post_sealed, "stages": normalized_stages,
    }


def _load_state(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControllerError(f"cannot read state: {path}") from exc
    if not isinstance(raw, dict):
        raise ControllerError("state root must be an object")
    return raw


def _new_state(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "round_id": plan["round_id"],
        "plan_path": str(plan["plan_path"]),
        "plan_sha256": plan["plan_sha256"],
        "status": "ready",
        "stages": {},
    }


def _bind_state(state: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ControllerError("state schema version mismatch")
    if state.get("round_id") != plan["round_id"] or state.get("plan_sha256") != plan["plan_sha256"]:
        raise ControllerError("state belongs to a different round or plan bytes")
    if not isinstance(state.get("stages"), Mapping):
        raise ControllerError("state stages is invalid")


def _artifact_sha_map(stage: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    paths: list[Path] = []
    for declaration in stage["artifacts"]:
        path = declaration["path"]
        kind = declaration["type"]
        if kind == "directory" and not path.is_dir():
            raise ControllerError(f"expected artifact directory is missing: {path}")
        if kind != "directory" and not path.is_file():
            raise ControllerError(f"expected artifact file is missing: {path}")
        actual = sha256_path(path)
        if declaration["sha256"] is not None and actual != declaration["sha256"]:
            raise ControllerError(f"artifact SHA256 mismatch: {path}")
        result[str(path)] = actual
        paths.append(path)
    _verify_markers(paths, stage["expected_markers"])
    return result


def _input_sha_map(stage: Mapping[str, Any], state: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, str]:
    result = {
        str(plan["snapshot_manifest_path"]): _checked_file(
            plan["snapshot_manifest_path"], plan["snapshot_manifest_sha256"], "snapshot manifest"
        ),
        str(plan["weights_path"]): _checked_file(plan["weights_path"], plan["weights_sha256"], "parent L1 weights"),
    }
    for record in stage["command_files"].values():
        path = Path(record["path"])
        actual = sha256_path(path)
        if actual != record["sha256"]:
            raise ControllerError(f"command file SHA256 mismatch: {path}")
        result[str(path)] = actual
    for declaration in stage["inputs"]:
        path = declaration["path"]
        expected = declaration["sha256"]
        if expected is None:
            producer_record = state["stages"].get(declaration["producer"])
            if not isinstance(producer_record, Mapping) or producer_record.get("status") != "completed":
                raise ControllerError(f"dynamic input producer is not complete: {declaration['producer']}")
            expected = producer_record.get("artifact_sha256", {}).get(str(path))
            if expected is None:
                raise ControllerError(f"dynamic input is not bound by producer state: {path}")
        actual = sha256_path(path)
        if actual != expected:
            raise ControllerError(f"input SHA256 mismatch: {path}")
        result[str(path)] = actual
    return dict(sorted(result.items()))


def _verify_completed(stage: Mapping[str, Any], record: Mapping[str, Any], state: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    if record.get("command_hash") != stage["command_hash"]:
        raise ControllerError(f"completed stage command changed: {stage['id']}")
    if record.get("command_files") != stage["command_files"]:
        raise ControllerError(f"completed stage command files changed: {stage['id']}")
    inputs = _input_sha_map(stage, state, plan)
    if record.get("input_sha256") != inputs:
        raise ControllerError(f"completed stage inputs changed: {stage['id']}")
    artifacts = _artifact_sha_map(stage)
    if record.get("artifact_sha256") != artifacts:
        raise ControllerError(f"completed stage artifacts changed: {stage['id']}")


def _prepare_state(plan: Mapping[str, Any], *, create: bool) -> tuple[Path, dict[str, Any]]:
    state_dir: Path = plan["state_dir"]
    state_path = state_dir / "state.json"
    if state_path.exists():
        state = _load_state(state_path)
        _bind_state(state, plan)
        return state_path, state
    if not create:
        raise ControllerError(f"state does not exist: {state_path}")
    if state_dir.exists() and any(state_dir.iterdir()):
        raise ControllerError("fresh --state-dir must be empty; refusing overwrite")
    state_dir.mkdir(parents=True, exist_ok=True)
    state = _new_state(plan)
    atomic_write_state(state_path, state)
    return state_path, state


def run_round(plan: Mapping[str, Any]) -> int:
    state_path, state = _prepare_state(plan, create=True)
    logs_dir = plan["state_dir"] / "logs"
    logs_dir.mkdir(exist_ok=True)
    for stage in plan["stages"]:
        stage_id = stage["id"]
        record = state["stages"].get(stage_id)
        if isinstance(record, Mapping) and record.get("status") == "completed":
            _verify_completed(stage, record, state, plan)
            if stage["manual_review_required"] and not record.get("manual_review_approved", False):
                state["status"] = "manual_review_required"
                atomic_write_state(state_path, state)
                print(f"manual review required after {stage_id}; owner approval must be recorded outside this controller")
                return 0
            continue
        for dependency in stage["depends_on"]:
            dependency_record = state["stages"].get(dependency)
            if not isinstance(dependency_record, Mapping) or dependency_record.get("status") != "completed":
                raise ControllerError(f"stage dependency is not complete: {stage_id} <- {dependency}")
        input_shas = _input_sha_map(stage, state, plan)
        for artifact in stage["artifacts"]:
            if artifact["path"].exists():
                raise ControllerError(f"refusing to overwrite pre-existing artifact: {artifact['path']}")
            artifact["path"].parent.mkdir(parents=True, exist_ok=True)
        attempt = int(record.get("attempt", 0)) + 1 if isinstance(record, Mapping) else 1
        stdout_path = logs_dir / f"{stage_id}.{attempt}.stdout.log"
        stderr_path = logs_dir / f"{stage_id}.{attempt}.stderr.log"
        if stdout_path.exists() or stderr_path.exists():
            raise ControllerError(f"refusing to overwrite stage logs: {stage_id} attempt {attempt}")
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            completed = subprocess.run(
                stage["command"], cwd=plan["plan_path"].parent,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                shell=False, check=False,
            )
        record = {
            "status": "failed" if completed.returncode else "validating",
            "attempt": attempt,
            "command": stage["command"],
            "command_hash": stage["command_hash"],
            "command_files": stage["command_files"],
            "input_sha256": input_shas,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "exit_code": completed.returncode,
            "artifact_sha256": {},
        }
        state["stages"][stage_id] = record
        state["status"] = "failed" if completed.returncode else "validating"
        atomic_write_state(state_path, state)
        if completed.returncode:
            print(f"stage failed: {stage_id} (exit {completed.returncode})", file=sys.stderr)
            return 1
        try:
            record["artifact_sha256"] = _artifact_sha_map(stage)
        except ControllerError:
            record["status"] = "failed_validation"
            state["status"] = "failed_validation"
            atomic_write_state(state_path, state)
            raise
        record["status"] = "completed"
        record["manual_review_approved"] = False
        state["status"] = "running"
        atomic_write_state(state_path, state)
        if stage["manual_review_required"]:
            state["status"] = "manual_review_required"
            atomic_write_state(state_path, state)
            print(f"manual review required after {stage_id}; owner approval must be recorded outside this controller")
            return 0
    state["status"] = "completed"
    atomic_write_state(state_path, state)
    print(f"round completed: {plan['round_id']}")
    return 0


def print_dag(plan: Mapping[str, Any]) -> None:
    print(f"round {plan['round_id']} ({plan['plan_sha256']})")
    for stage in plan["stages"]:
        dependencies = ",".join(stage["depends_on"]) or "ROOT"
        print(f"{stage['id']} [{stage['kind']}] <- {dependencies}")


def print_status(plan: Mapping[str, Any]) -> None:
    _, state = _prepare_state(plan, create=False)
    for stage in plan["stages"]:
        record = state["stages"].get(stage["id"])
        if isinstance(record, Mapping) and record.get("status") == "completed":
            _verify_completed(stage, record, state, plan)
        print(f"{stage['id']}: {record.get('status', 'pending') if isinstance(record, Mapping) else 'pending'}")
    print(f"round: {state.get('status', 'unknown')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--status", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = validate_plan(args.plan, args.state_dir)
        if args.check:
            print_dag(plan)
            return 0
        if args.status:
            print_status(plan)
            return 0
        return run_round(plan)
    except ControllerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
