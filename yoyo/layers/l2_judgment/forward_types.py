"""Typed forward-log records and run summaries."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

import lightgbm as lgb
import pandas as pd

from yoyo.layers.l2_judgment.frozen import FrozenArtifact

if TYPE_CHECKING:
    from yoyo.contracts.protocol import StrategyProtocol

PROJECT_DIR: Final = Path(__file__).resolve().parents[2]

# The log schema, its legacy markers and its readers moved to contracts on
# 2026-08-03: a ledger L2 writes and L4 reads belongs to neither. Re-exported here
# so existing L2 callers and the 519 tests keep working during migration.
from yoyo.contracts.forward_log import (  # noqa: E402,F401
    FORWARD_COLUMNS, FORWARD_LOG_H1_SCALED_PATH, FORWARD_LOG_MAKER_TRIAL_PATH,
    FORWARD_LOG_PATH, LEGACY_PROTOCOL, LEGACY_SEMANTICS, LEGACY_STRATEGY,
    OUTCOME_COLUMNS, ForwardRecord, MergeResult,
)
FORWARD_START: Final = pd.Timestamp("2026-07-31 00:00:00", tz="UTC")
PROTOCOL_VERSION: Final = "short_v10_p0fix_20260731"
# Candidate provenance is part of the runtime safety contract. Production may
# only discover from the validated YOLO path; legacy rules remain available to
# explicitly marked offline/research callers.
RUNTIME_MODE: Final = os.environ.get("FABLE_RUNTIME_MODE", "production").strip().lower() or "production"
CANDIDATE_SOURCE: Final = os.environ.get("FABLE_CANDIDATE_SOURCE", "yolo").strip().lower() or "yolo"
VALID_RUNTIME_MODES: Final = frozenset({"production", "research"})
VALID_CANDIDATE_SOURCES: Final = frozenset({"yolo", "rules"})
BAR: Final = pd.Timedelta(minutes=15)
TP_MULT: Final = 5.0
SL_MULT: Final = 2.0
# H1 scaled exit params (single-variable vs mainline TP5/SL2).
SCALED_TP1_MULT: Final = 2.5
SCALED_TRAIL_MULT: Final = 3.0
SCALED_SL_MULT: Final = 2.0


def validate_candidate_source(candidate_source: str, runtime_mode: str) -> str:
    """Validate candidate provenance before any forward scan or model load.

    Production accepts only ``yolo``. ``rules`` is intentionally retained for
    reproducible offline/research diagnostics, which must opt in with
    ``FABLE_RUNTIME_MODE=research``.
    """
    source = str(candidate_source).strip().lower()
    mode = str(runtime_mode).strip().lower()
    if mode not in VALID_RUNTIME_MODES:
        raise ValueError(
            f"invalid FABLE_RUNTIME_MODE={runtime_mode!r}; "
            f"expected one of {sorted(VALID_RUNTIME_MODES)}"
        )
    if source not in VALID_CANDIDATE_SOURCES:
        raise ValueError(
            f"invalid FABLE_CANDIDATE_SOURCE={candidate_source!r}; "
            f"expected one of {sorted(VALID_CANDIDATE_SOURCES)}"
        )
    if mode == "production" and source != "yolo":
        raise RuntimeError(
            "production forward tracking requires candidate_source=yolo; "
            "legacy rules are research-only"
        )
    return source


class ForwardSummaryJson(TypedDict):
    model_path: str
    threshold: float
    start_time: str
    scanned_series: int
    candidates_seen: int
    threshold_signals_seen: int
    new_signals: int
    closed_updates: int
    total_rows: int
    open_rows: int
    closed_rows: int
    output: str


@dataclass(frozen=True)
class ForwardExit:
    __slots__ = ("status", "outcome", "label", "exit_offset", "exit_time", "realized_ret")

    status: str
    outcome: str
    label: int
    exit_offset: int
    exit_time: str
    realized_ret: float


@dataclass(frozen=True)
class ForwardScanInput:
    __slots__ = (
        "artifact", "booster", "detected_at", "start_time", "existing_log", "protocol"
    )

    artifact: FrozenArtifact
    booster: lgb.Booster
    detected_at: str
    start_time: pd.Timestamp
    existing_log: pd.DataFrame
    protocol: StrategyProtocol | None


@dataclass(frozen=True)
class ForwardScanResult:
    __slots__ = ("records", "scanned_series", "candidates_seen", "threshold_signals_seen")

    records: list[ForwardRecord]
    scanned_series: int
    candidates_seen: int
    threshold_signals_seen: int


@dataclass(frozen=True)
class ForwardRunSummary:
    __slots__ = (
        "artifact",
        "start_time",
        "scanned_series",
        "candidates_seen",
        "threshold_signals_seen",
        "new_signals",
        "closed_updates",
        "total_rows",
        "open_rows",
        "closed_rows",
        "output",
    )

    artifact: FrozenArtifact
    start_time: pd.Timestamp
    scanned_series: int
    candidates_seen: int
    threshold_signals_seen: int
    new_signals: int
    closed_updates: int
    total_rows: int
    open_rows: int
    closed_rows: int
    output: Path

    def to_json(self) -> ForwardSummaryJson:
        return {
            "model_path": self.artifact.relative_model_path,
            "threshold": self.artifact.threshold,
            "start_time": str(self.start_time),
            "scanned_series": self.scanned_series,
            "candidates_seen": self.candidates_seen,
            "threshold_signals_seen": self.threshold_signals_seen,
            "new_signals": self.new_signals,
            "closed_updates": self.closed_updates,
            "total_rows": self.total_rows,
            "open_rows": self.open_rows,
            "closed_rows": self.closed_rows,
            "output": str(self.output),
        }
