"""Atomic JSONL I/O, SHA, git, and the locked-config loader."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

LOCKED_KEYS = frozenset(
    {
        "visible_bars",
        "core_bars",
        "confirmation_bars",
        "local_core_start",
        "local_core_end_exclusive",
        "local_core_positions",
        "local_confirmation_position",
        "sma",
        "ema",
        "indicator_warmup_bars",
        "y_pad_frac",
        "timeframe",
        "pre_context_bars",
    }
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in ("task_name", "visible_bars", "core_bars") if key not in cfg]
    if missing:
        raise ValueError(f"config missing {missing}")
    return cfg


def config_sha(path: Path) -> str:
    return sha256_file(path)


def git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def git_status_short(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "status", "--short"], text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return ""


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def jsonl_dumps(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, jsonl_dumps(rows))


def add_common_flags(parser, *, default_config: Path) -> None:
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
