"""The forward log IS the L2 -> L4 interface, so it lives in contracts.

Discovered by the layer split rather than designed: executor.py (L4) was importing
read_forward_log, actionable_rows and LEGACY_PROTOCOL straight out of the judgment
package (L2), and tests/test_layer_boundaries.py would have rejected that once
execution moved. The fix is not to break the import -- L4 genuinely needs to read
what L2 wrote. The fix is to notice that a ledger written by one layer and read by
another is not either layer's private business.

So the schema, the legacy markers, the identity key and the readers all sit here.
L2 writes through this contract; L4 reads through the same one; neither can drift
from the other without the change being visible in one file.

Moved from src/judgment/forward_types.py and src/judgment/forward_records.py on
2026-08-03 (owner's four-layer restructure). Behaviour unchanged -- this commit
moves code, it does not alter what the log means.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import pandas as pd
from yoyo.contracts.paths import data_path, data_root  # noqa: F401

PROJECT_DIR: Final = data_root()

FORWARD_LOG_PATH: Final = PROJECT_DIR / "data" / "forward_log.csv"
# H1 scaled shadow paper book — never mixed into mainline 100-trade gate.
FORWARD_LOG_H1_SCALED_PATH: Final = PROJECT_DIR / "data" / "forward_log_h1_scaled.csv"
# A2 maker-entry trial bucket — isolated ledger; never mixed into mainline.
# Written only by scripts/forward_maker_trial.py (requires FABLE_MAKER_TRIAL=1).
FORWARD_LOG_MAKER_TRIAL_PATH: Final = PROJECT_DIR / "data" / "forward_log_maker_trial.csv"
# YOLO mainline cutover (owner 2026-07-15): new candidate source → new forward clock.
# Pre-cutover rule-scan log archived as data/forward_log_rules_pre_yolo_20260715.csv
# Owner 2026-07-18/19: clear pre-v11 mixed book and restart gate for clean retest.
# Archived: data/forward_log_pre_v11_retest_20260719.csv (VPS + local).
# Owner 2026-07-31: short-protocol reset (side-aware scan + ACTIVE authority).
# Archived polluted long-geometry book:
#   data/forward_log_pre_short_protocol_20260731.csv
# Protocol tag for dashboards/docs (not a CSV column yet):
#   protocol_version = short_v10_p0fix_20260731
# Use last *closed* bar open (not wall-clock "now") so live YOLO is not skipped
# while the current 15m candle is still forming.

FORWARD_COLUMNS: Final = (
    "source",
    "symbol",
    "signal_time",
    "detected_at",
    "status",
    "score",
    "threshold",
    "model_path",
    "dataset_sha256",
    "signal_i",
    "entry_time",
    "entry_price",
    "maker_filled",
    "outcome",
    "label",
    "exit_offset",
    "exit_time",
    "realized_ret",
    "atr_pct",
    "dense_run_len",
    # Tiered sizing (owner 2026-07-20). Appended LAST so pre-tier readers of
    # positional CSVs are unaffected; legacy rows read back as NaN → 1x.
    "tier",
    "size_mult",
    # Direction contract. Legacy rows normalize to NaN and remain long-only;
    # explicit non-long rows are rejected by the current executor.
    "side",
    # Appended, never reordered: an old CSV must keep reading back correctly.
    "protocol_version",
    "strategy_id",
    "feature_semantics",
    "decision_at",
    "execution_eligible",
    # Immutable artifact identity. Appended for CSV compatibility.
    "model_sha256",
    "detector_sha256",
    # P0.6 causal execution timeline. These are appended for old CSV readers.
    # The historical entry_* fields above are legacy compatibility only; new
    # protocol rows never put a signal-close proxy or research next-open there.
    "candidate_detected_at",
    "signal_closed_at",
    "entry_mode",
    "entry_status",
    "entry_requested_at",
    "fill_source",
    "fill_at",
    "fill_px",
    "reference_px",
    # Research outcome is explicitly separate from actual broker/paper PnL.
    "research_status",
    "research_outcome",
    "research_label",
    "research_exit_offset",
    "research_exit_time",
    "research_gross_ret",
    "actual_outcome",
    "actual_exit_at",
    "actual_exit_px",
    "actual_realized_ret",
    "actual_return_semantics",
    "return_convention",
    "target_ret_column",
    "target_semantics",
    "target_cost_included",
    "reporting_route",
)
OUTCOME_COLUMNS: Final = (
    "status", "outcome", "label", "exit_offset", "exit_time", "realized_ret",
    "research_status", "research_outcome", "research_label",
    "research_exit_offset", "research_exit_time", "research_gross_ret",
    "actual_outcome", "actual_exit_at", "actual_exit_px", "actual_realized_ret",
)

# Rows written before provenance existed. Not a placeholder to be filled in later:
# it is the honest name for "produced under a protocol nobody recorded", and it
# keeps those rows out of any new protocol's 100-trade clock (acceptance H-01/H-02).
LEGACY_PROTOCOL: Final = "legacy_pre_20260803"
LEGACY_STRATEGY: Final = "legacy_unknown"
LEGACY_SEMANTICS: Final = "legacy_unaligned"



class ForwardRecord(TypedDict):
    source: str
    symbol: str
    signal_time: str
    detected_at: str
    status: str
    score: float
    threshold: float
    model_path: str
    dataset_sha256: str
    signal_i: int
    entry_time: str
    entry_price: float
    # None while a tip-recorded row awaits its entry-bar backfill
    maker_filled: bool | None
    outcome: str
    label: int
    exit_offset: int
    exit_time: str
    realized_ret: float
    atr_pct: float
    dense_run_len: int
    # score→size tier of the frozen val distribution (q90_q95/q95_q99/q99_plus)
    tier: str
    size_mult: float
    side: str
    # Provenance (P0.3). Which contract produced this row, and when the decision
    # was actually made -- not when the batch that contained it started.
    protocol_version: str
    strategy_id: str
    feature_semantics: str
    decision_at: str
    execution_eligible: bool
    model_sha256: str
    detector_sha256: str
    candidate_detected_at: str
    signal_closed_at: str
    entry_mode: str
    entry_status: str
    entry_requested_at: str
    fill_source: str
    fill_at: str
    fill_px: float
    reference_px: float
    research_status: str
    research_outcome: str
    research_label: int
    research_exit_offset: int
    research_exit_time: str
    research_gross_ret: float
    actual_outcome: str
    actual_exit_at: str
    actual_exit_px: float
    actual_realized_ret: float
    actual_return_semantics: str
    return_convention: str
    target_ret_column: str
    target_semantics: str
    target_cost_included: bool
    reporting_route: str



@dataclass(frozen=True)
class MergeResult:
    __slots__ = ("frame", "new_signals", "closed_updates")

    frame: pd.DataFrame
    new_signals: int
    closed_updates: int


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



# side is part of the key, so it needs its own marker for rows that predate it
LEGACY_SIDE = "legacy_unknown_side"
ForwardKey = tuple[str, str, str, str, str]


def read_forward_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=FORWARD_COLUMNS)
    return normalize_log(pd.read_csv(path))


def write_forward_log(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _entry_pending(record: dict) -> bool:
    """True while a tip-recorded row still carries proxy entry fields."""
    value = record.get("maker_filled")
    return value is None or (isinstance(value, float) and np.isnan(value))


def merge_forward_log(existing: pd.DataFrame, new_records: list[ForwardRecord]) -> MergeResult:
    current = normalize_log(existing)
    rows = {}
    for record in current.to_dict("records"):
        rows[row_key(record)] = record
    new_signals = 0
    closed_updates = 0
    for record in new_records:
        key = forward_key(
            record["source"], record["symbol"], pd.Timestamp(record["signal_time"]),
            record.get("side"), record.get("protocol_version"),
        )
        previous = rows.get(key)
        if previous is None:
            rows[key] = record
            new_signals += 1
            continue
        if str(previous["status"]) == "closed":
            continue
        merged = dict(previous)
        changed = False
        # Legacy-only entry backfill (2026-07-20 real-time tip path): a row recorded at the
        # tip carries a PROXY entry (signal-bar close) and empty maker_filled.
        # Once the true entry bar has printed, overwrite entry fields with the
        # real next-bar values. detected_at stays first-seen (lag accounting).
        if (
            str(previous.get("protocol_version")) == LEGACY_PROTOCOL
            and _entry_pending(previous)
            and not _entry_pending(record)
        ):
            for column in ("entry_time", "entry_price", "maker_filled"):
                merged[column] = record[column]
            changed = True
        if record["status"] == "closed":
            for column in OUTCOME_COLUMNS:
                # Old-schema compatibility: legacy updates do not carry the
                # appended P0.6 research/actual columns.
                merged[column] = record.get(column, merged.get(column, np.nan))
            closed_updates += 1
            changed = True
        if changed:
            rows[key] = merged
    if not rows:
        return MergeResult(pd.DataFrame(columns=FORWARD_COLUMNS), new_signals, closed_updates)
    frame = pd.DataFrame(rows.values())
    frame = normalize_log(frame).sort_values(["signal_time", "symbol"]).reset_index(drop=True)
    return MergeResult(frame, new_signals, closed_updates)


def normalize_log(frame: pd.DataFrame) -> pd.DataFrame:
    """Read any vintage of the log without crashing, and without lying about it.

    A row from before provenance existed gets LEGACY_* markers rather than the
    current protocol's values. Inheriting today's protocol_version would silently
    fold old long-resolver rows into the repaired short book, which is the exact
    contamination acceptance H-01/H-02 forbids.
    """
    out = frame.copy()
    for column in FORWARD_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    for column, marker in (
        ("protocol_version", LEGACY_PROTOCOL),
        ("strategy_id", LEGACY_STRATEGY),
        ("feature_semantics", LEGACY_SEMANTICS),
    ):
        out[column] = out[column].fillna(marker).replace("", marker)
    # Absent eligibility is not eligibility. Built from a truth test rather than
    # fillna+astype: an object column of NaN downcasts with a FutureWarning, and
    # more to the point the string "False" read back from CSV is truthy under
    # astype(bool) -- which would flip an ineligible row into an actionable one.
    out["execution_eligible"] = [
        str(v).strip().lower() in ("true", "1", "1.0") for v in out["execution_eligible"]
    ]
    return out[list(FORWARD_COLUMNS)]


def rows_for_protocol(frame: pd.DataFrame, protocol_version: str) -> pd.DataFrame:
    """One protocol's rows only. Summaries must never span two (H-01/H-02)."""
    if frame.empty:
        return frame
    return normalize_log(frame).pipe(
        lambda f: f[f["protocol_version"].astype(str) == str(protocol_version)]
    )


def actionable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows an executor may act on: execution-eligible only (H-03).

    Scored-but-ineligible rows stay visible in the log on purpose -- they are
    evidence -- but they are not an order queue.
    """
    if frame.empty:
        return frame
    out = normalize_log(frame)
    return out[out["execution_eligible"]]


def actual_closed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Only causally filled, closed trades; legacy proxies never qualify."""
    if frame.empty:
        return frame
    out = normalize_log(frame)
    filled = out["entry_status"].astype(str).isin({"paper_filled", "broker_filled"})
    closed = out["status"].astype(str) == "closed"
    pnl = pd.to_numeric(out["actual_realized_ret"], errors="coerce").notna()
    semantics = out["actual_return_semantics"].astype(str).isin(
        {"gross", "net_taker", "net_maker"}
    )
    return out[filled & closed & pnl & semantics]


def open_keys(frame: pd.DataFrame) -> set[ForwardKey]:
    if frame.empty:
        return set()
    active = frame[frame["status"] != "closed"]
    return {row_key(record) for record in active.to_dict("records")}


def row_key(record) -> ForwardKey:
    return forward_key(
        str(record["source"]),
        str(record["symbol"]),
        pd.Timestamp(record["signal_time"]),
        record.get("side"),
        record.get("protocol_version"),
    )


def forward_key(
    source: str,
    symbol: str,
    signal_time: pd.Timestamp,
    side: object = None,
    protocol_version: object = None,
) -> ForwardKey:
    """Identity of one signal event, including which contract produced it.

    side and protocol_version are part of the key (acceptance B-03/B-04): the same
    bar under two protocols is two events, and merging them would let a legacy
    long-resolver row absorb the outcome of a repaired short one. Both default to
    their LEGACY_* marker rather than to a live value, so an old row keeps its own
    identity instead of being adopted by whatever is running now.
    """
    side_s = LEGACY_SIDE if side is None or _blank(side) else str(side).strip().lower()
    proto_s = LEGACY_PROTOCOL if protocol_version is None or _blank(protocol_version) \
        else str(protocol_version).strip()
    return source, symbol, str(signal_time), side_s, proto_s


def _blank(value: object) -> bool:
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip() == "" or str(value).lower() == "nan"
