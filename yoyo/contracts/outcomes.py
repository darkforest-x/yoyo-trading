"""Canonical fixed-barrier outcome and return contract.

This resolver is pure: it never reads a dataset, artifact, ACTIVE pointer, order
client, or wall clock. Callers must pass the side, actual/research entry index and
price, signal-bar ATR, TP/SL multipliers, horizon, same-bar policy, gap policy,
return convention, and whether a partial future is allowed. Only OHLC rows from
``entry_i`` through ``entry_i + horizon_bars - 1`` are examined; no earlier bar
can become an exit and no later bar can affect the result.

P0 freezes existing mainline economics rather than tuning them: TP5/SL2/72 is
supplied by the protocol, same-bar is conservative SL, and gaps use the declared
``barrier_price`` idealization. Return conventions are explicit so linear short
(``1 - exit/entry``) cannot be mixed with inverse short (``entry/exit - 1``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Side = Literal["long", "short"]
ReturnConvention = Literal["linear_long", "linear_short", "inverse_short"]
SameBarPolicy = Literal["conservative_sl"]
GapPolicy = Literal["barrier_price"]

RETURN_CONVENTIONS = ("linear_long", "linear_short", "inverse_short")
SAME_BAR_POLICIES = ("conservative_sl",)
GAP_POLICIES = ("barrier_price",)


class OutcomeContractError(ValueError):
    """The requested outcome cannot be resolved under a trustworthy contract."""


@dataclass(frozen=True)
class BarrierResolution:
    status: Literal["open", "closed"]
    outcome: str
    label: int | None
    entry_price: float
    exit_offset: int
    exit_price: float | None
    exit_time: str | None
    gross_ret: float | None


def gross_return(entry_price: float, exit_price: float, convention: str) -> float:
    """Return before costs under one explicit price-return convention."""
    entry = float(entry_price)
    exit_ = float(exit_price)
    if not np.isfinite(entry) or entry <= 0:
        raise OutcomeContractError("entry_price must be finite and positive")
    if not np.isfinite(exit_) or exit_ <= 0:
        raise OutcomeContractError("exit_price must be finite and positive")
    if convention == "linear_long":
        return exit_ / entry - 1.0
    if convention == "linear_short":
        return 1.0 - exit_ / entry
    if convention == "inverse_short":
        return entry / exit_ - 1.0
    raise OutcomeContractError(f"unknown return_convention={convention!r}")


def _validate_side_convention(side: str, convention: str) -> None:
    if side not in ("long", "short"):
        raise OutcomeContractError(f"side must be long|short, got {side!r}")
    if convention not in RETURN_CONVENTIONS:
        raise OutcomeContractError(f"unknown return_convention={convention!r}")
    if side == "long" and convention != "linear_long":
        raise OutcomeContractError("long side requires return_convention=linear_long")
    if side == "short" and convention == "linear_long":
        raise OutcomeContractError("short side requires an explicit short return convention")


def resolve_barrier_outcome(
    frame: pd.DataFrame,
    *,
    side: str,
    entry_i: int,
    entry_price: float,
    atr: float,
    tp_atr_mult: float,
    sl_atr_mult: float,
    horizon_bars: int,
    same_bar_policy: str,
    gap_policy: str,
    return_convention: str,
    allow_partial: bool,
    bar_duration: pd.Timedelta = pd.Timedelta(minutes=15),
) -> BarrierResolution:
    """Resolve TP/SL/timeout from the entry bar forward.

    Required OHLC columns: ``open/high/low/close``; ``open_time`` is used only
    for the exit timestamp. With ``allow_partial=True``, an unresolved path with
    fewer than ``horizon_bars`` rows returns ``status=open``. A barrier hit inside
    a partial path still closes. Invalid prices, ATR, policies, or side/return
    combinations raise instead of becoming TIMEOUT.
    """
    _validate_side_convention(side, return_convention)
    if same_bar_policy not in SAME_BAR_POLICIES:
        raise OutcomeContractError(f"unsupported same_bar_policy={same_bar_policy!r}")
    if gap_policy not in GAP_POLICIES:
        raise OutcomeContractError(f"unsupported gap_policy={gap_policy!r}")
    entry = float(entry_price)
    atr_value = float(atr)
    if not np.isfinite(entry) or entry <= 0:
        raise OutcomeContractError("entry_price must be finite and positive")
    if not np.isfinite(atr_value) or atr_value <= 0:
        raise OutcomeContractError("atr must be finite and positive")
    if not np.isfinite(tp_atr_mult) or float(tp_atr_mult) <= 0:
        raise OutcomeContractError("tp_atr_mult must be finite and positive")
    if not np.isfinite(sl_atr_mult) or float(sl_atr_mult) <= 0:
        raise OutcomeContractError("sl_atr_mult must be finite and positive")
    if int(horizon_bars) <= 0:
        raise OutcomeContractError("horizon_bars must be positive")
    if entry_i < 0 or entry_i >= len(frame):
        if allow_partial and entry_i == len(frame):
            return BarrierResolution("open", "", None, entry, 0, None, None, None)
        raise OutcomeContractError("entry_i is outside the available frame")

    available = min(int(horizon_bars), len(frame) - int(entry_i))
    if available <= 0:
        return BarrierResolution("open", "", None, entry, 0, None, None, None)
    path = frame.iloc[entry_i : entry_i + available]
    highs = pd.to_numeric(path["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(path["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(path["close"], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(highs).all() and np.isfinite(lows).all()):
        raise OutcomeContractError("barrier path high/low must be finite")

    if side == "long":
        tp_price = entry + float(tp_atr_mult) * atr_value
        sl_price = entry - float(sl_atr_mult) * atr_value
        hit_tp = highs >= tp_price
        hit_sl = lows <= sl_price
    else:
        tp_price = entry - float(tp_atr_mult) * atr_value
        sl_price = entry + float(sl_atr_mult) * atr_value
        if tp_price <= 0:
            raise OutcomeContractError("short TP price must remain positive")
        hit_tp = lows <= tp_price
        hit_sl = highs >= sl_price

    tp_first = int(np.argmax(hit_tp)) if hit_tp.any() else available
    sl_first = int(np.argmax(hit_sl)) if hit_sl.any() else available
    if tp_first < sl_first:
        outcome, label, offset, exit_price = "tp", 1, tp_first + 1, tp_price
    elif sl_first < tp_first:
        outcome, label, offset, exit_price = "sl", 0, sl_first + 1, sl_price
    elif tp_first == sl_first < available:
        outcome, label, offset, exit_price = "sl_ambiguous", 0, sl_first + 1, sl_price
    elif available < int(horizon_bars):
        if not allow_partial:
            raise OutcomeContractError("full horizon is unavailable")
        return BarrierResolution("open", "", None, entry, 0, None, None, None)
    else:
        exit_price = float(closes[int(horizon_bars) - 1])
        if not np.isfinite(exit_price) or exit_price <= 0:
            raise OutcomeContractError("timeout close must be finite and positive")
        outcome, label, offset = "timeout", 0, int(horizon_bars)

    exit_time = None
    if "open_time" in path.columns:
        exit_open_time = pd.Timestamp(path["open_time"].iloc[offset - 1])
        exit_time = str(exit_open_time + bar_duration)
    return BarrierResolution(
        "closed",
        outcome,
        label,
        entry,
        offset,
        float(exit_price),
        exit_time,
        gross_return(entry, float(exit_price), return_convention),
    )


# Barrier parameters. Owner decisions -- CLAUDE.md forbids code changing them, and
# they live here rather than in labeling.py because L1's tip gate, L2's labels and
# L3's replay must all read the same numbers or they are measuring different
# strategies. Moved from src/judgment/labeling.py on 2026-08-03.
HORIZON_BARS = 72
ATR_PCT_MIN = 0.0015
