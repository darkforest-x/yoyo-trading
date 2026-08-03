"""Triple-barrier labeling for dense-MA candidates (long and short).

Long path (`label_candidate`):
  For each candidate at bar i (rule inputs use data <= i):
  - entry = open of i+1 (`entry="next_open"`) or close of i (`signal_close`);
  - path starts at i+1; upper = entry + TP*ATR, lower = entry - SL*ATR;
  - label 1 iff upper is touched first; SL/timeout/ambiguous → 0.

Short path (`label_short_candidate`):
  Same entry/horizon/ATR floor contract, barriers mirrored: TP is the lower
  barrier (price fall), SL is the upper barrier (rally against the short).
  `realized_ret` is short conventional PnL ``1 - exit/entry`` (positive when
  price falls to TP) — same as dump/v10 wide `net_barrier_*` builders and
  `resolve_forward_exit_short`. Do not feed short candidates into
  `label_candidate` — that would score long barrier wins and mix sides.

Variants (trailing / MA exit / structure) have long and short twins; changing
TP/SL/cost presets requires owner approval. Intra-bar both-touch → SL
(conservative). Holdout is never labeled here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from yoyo.contracts.outcomes import OutcomeContractError, resolve_barrier_outcome

# v2 barriers (owner decision 2026-07-07): wider barriers so the median gross
# TP (~4 x atr_pct ~= 1.1%) clears the 0.2% round-trip cost by >5x. The v1
# barriers (TP 2xATR / SL 1xATR) produced labels dominated by micro noise
# (median exit 3 bars) and top-decile net returns below cost -- see
# analysis/p2b_judgment_report.md.
TP_ATR_MULT = 4.0
SL_ATR_MULT = 2.0
from yoyo.contracts.outcomes import ATR_PCT_MIN, HORIZON_BARS  # noqa: F401
# Candidates whose atr_pct is below this floor are skipped entirely: their
# barrier scale cannot cover trading costs no matter what the model says.

EntryMode = Literal["next_open", "signal_close"]
ENTRY_MODES: tuple[str, ...] = ("next_open", "signal_close")


@dataclass(frozen=True)
class BarrierOutcome:
    label: int          # 1 = TP first, 0 = SL first or timeout
    outcome: str        # "tp" | "sl" | "timeout" | "sl_ambiguous"
    exit_offset: int    # bars after entry bar when exit happened (1-based)
    entry_price: float
    realized_ret: float  # gross return under the explicit side convention


def _resolve_entry_price(
    frame: pd.DataFrame,
    signal_i: int,
    entry: EntryMode,
) -> float | None:
    """Fill price for a signal at `signal_i`. Path always starts at signal_i+1."""
    if entry == "next_open":
        entry_i = signal_i + 1
        if entry_i >= len(frame):
            return None
        price = float(frame["open"].iloc[entry_i])
    elif entry == "signal_close":
        price = float(frame["close"].iloc[signal_i])
    else:
        raise ValueError(f"unknown entry mode {entry!r}; expected one of {ENTRY_MODES}")
    if not np.isfinite(price) or price <= 0:
        return None
    return price


def label_candidate(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    tp_mult: float = TP_ATR_MULT,
    sl_mult: float = SL_ATR_MULT,
    atr_pct_min: float = ATR_PCT_MIN,
    horizon: int = HORIZON_BARS,
    entry: EntryMode = "next_open",
) -> BarrierOutcome | None:
    """Label one candidate at position `signal_i` of an indicator frame.

    Requires columns open/high/low/close and atr14. Returns None when there
    are not enough future bars for entry + full horizon, or when atr_pct at
    the signal bar is below `atr_pct_min` (barriers too narrow to trade).

    `entry="next_open"` (default): fill at open of bar i+1.
    `entry="signal_close"`: fill at close of bar i; barrier path still starts
    at bar i+1 (nothing left to trade after the close print).
    """
    return _label_fixed_barrier(
        frame,
        signal_i,
        side="long",
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        atr_pct_min=atr_pct_min,
        horizon=horizon,
        entry=entry,
        return_convention="linear_long",
    )


def label_candidate_trailing(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    trail_mult: float,
    atr_pct_min: float = ATR_PCT_MIN,
    horizon: int = HORIZON_BARS,
    entry: EntryMode = "next_open",
) -> BarrierOutcome | None:
    """Trend-following exit: no fixed TP, stop trails `trail_mult` x ATR14(i)
    below the running high (seeded at entry). Conservative intra-bar rule: the
    stop for bar j uses the running high up to bar j-1, so a new high and a
    stop-out inside the same bar never help the trade; a gap below the stop
    fills at the bar open. Timeout exits at the horizon close.

    Label = 1 iff realized_ret > 0 (no barrier order to encode here).
    """
    ctx = _entry_context(frame, signal_i, horizon, atr_pct_min, entry=entry)
    if ctx is None:
        return None
    fill, atr, highs, lows, opens, timeout_close = ctx
    run_max = fill
    for j in range(horizon):
        stop = run_max - trail_mult * atr
        if lows[j] <= stop:
            exit_price = min(stop, float(opens[j]))
            ret = exit_price / fill - 1
            return BarrierOutcome(int(ret > 0), "trail", j + 1, fill, ret)
        run_max = max(run_max, float(highs[j]))
    ret = timeout_close / fill - 1
    return BarrierOutcome(int(ret > 0), "timeout", horizon, fill, ret)


def label_short_candidate_trailing(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    trail_mult: float,
    atr_pct_min: float = ATR_PCT_MIN,
    horizon: int = HORIZON_BARS,
    entry: EntryMode = "next_open",
) -> BarrierOutcome | None:
    """Short mirror of `label_candidate_trailing`: stop trails above running low."""
    ctx = _entry_context(frame, signal_i, horizon, atr_pct_min, entry=entry)
    if ctx is None:
        return None
    fill, atr, highs, lows, opens, timeout_close = ctx
    run_min = fill
    for j in range(horizon):
        stop = run_min + trail_mult * atr
        if highs[j] >= stop:
            exit_price = max(stop, float(opens[j]))
            if exit_price <= 0:
                return None
            ret = fill / exit_price - 1
            return BarrierOutcome(int(ret > 0), "trail", j + 1, fill, ret)
        run_min = min(run_min, float(lows[j]))
    if timeout_close <= 0:
        return None
    ret = fill / timeout_close - 1
    return BarrierOutcome(int(ret > 0), "timeout", horizon, fill, ret)


def _entry_context(
    frame: pd.DataFrame,
    signal_i: int,
    horizon: int,
    atr_pct_min: float,
    entry: EntryMode = "next_open",
):
    """Shared guards for exit-variant labelers; None when the trade is untakeable."""
    path_i = signal_i + 1
    last_i = path_i + horizon - 1
    if last_i >= len(frame):
        return None
    atr = float(frame["atr14"].iloc[signal_i])
    fill = _resolve_entry_price(frame, signal_i, entry)
    if not np.isfinite(atr) or atr <= 0 or fill is None:
        return None
    atr_pct = float(frame["atr_pct"].iloc[signal_i])
    if atr_pct_min > 0 and (not np.isfinite(atr_pct) or atr_pct < atr_pct_min):
        return None
    sl = slice(path_i, last_i + 1)
    return (fill, atr, frame["high"].to_numpy()[sl], frame["low"].to_numpy()[sl],
            frame["open"].to_numpy()[sl], float(frame["close"].iloc[last_i]))


def label_short_candidate(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    tp_mult: float = TP_ATR_MULT,
    sl_mult: float = SL_ATR_MULT,
    atr_pct_min: float = ATR_PCT_MIN,
    horizon: int = HORIZON_BARS,
    entry: EntryMode = "next_open",
) -> BarrierOutcome | None:
    return _label_fixed_barrier(
        frame,
        signal_i,
        side="short",
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        atr_pct_min=atr_pct_min,
        horizon=horizon,
        entry=entry,
        return_convention="linear_short",
    )


def _label_fixed_barrier(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    side: str,
    tp_mult: float,
    sl_mult: float,
    atr_pct_min: float,
    horizon: int,
    entry: EntryMode,
    return_convention: str,
) -> BarrierOutcome | None:
    """Full-horizon adapter over the canonical resolver.

    Reads signal-bar ``atr14``/``atr_pct``, the declared entry price, and exactly
    ``horizon`` OHLC rows starting at ``signal_i + 1``. Dataset labeling refuses
    partial horizons; forward uses the same resolver with partials enabled.
    """
    path_i = signal_i + 1
    if path_i + int(horizon) > len(frame):
        return None
    try:
        atr = float(frame["atr14"].iloc[signal_i])
        atr_pct = float(frame["atr_pct"].iloc[signal_i])
    except (IndexError, TypeError, ValueError):
        return None
    fill = _resolve_entry_price(frame, signal_i, entry)
    if fill is None or not np.isfinite(atr) or atr <= 0:
        return None
    if atr_pct_min > 0 and (not np.isfinite(atr_pct) or atr_pct < atr_pct_min):
        return None
    try:
        resolved = resolve_barrier_outcome(
            frame,
            side=side,
            entry_i=path_i,
            entry_price=fill,
            atr=atr,
            tp_atr_mult=tp_mult,
            sl_atr_mult=sl_mult,
            horizon_bars=horizon,
            same_bar_policy="conservative_sl",
            gap_policy="barrier_price",
            return_convention=return_convention,
            allow_partial=False,
        )
    except OutcomeContractError:
        return None
    if resolved.status != "closed" or resolved.gross_ret is None or resolved.label is None:
        return None
    return BarrierOutcome(
        resolved.label,
        resolved.outcome,
        resolved.exit_offset,
        resolved.entry_price,
        resolved.gross_ret,
    )


def label_candidate_scaled(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    tp1_mult: float = 2.5,
    trail_mult: float = 3.0,
    sl_mult: float = 2.0,
    horizon: int = HORIZON_BARS,
    atr_pct_min: float = ATR_PCT_MIN,
) -> BarrierOutcome | None:
    """H1 scaled exit: half the position banks at entry + tp1_mult x ATR; the
    remainder trails trail_mult x ATR below the running high (seeded at the
    TP1 price). Hard stop entry - sl_mult x ATR protects the whole position
    until TP1. Conservative intra-bar ordering: the stop is always checked
    BEFORE the target within a bar, and the trailing stop for bar j uses the
    running high up to bar j-1. realized_ret is the half-and-half blend.
    """
    ctx = _entry_context(frame, signal_i, horizon, atr_pct_min)
    if ctx is None:
        return None
    entry, atr, highs, lows, opens, timeout_close = ctx
    hard_stop = entry - sl_mult * atr
    tp1 = entry + tp1_mult * atr
    ret1: float | None = None  # booked half, set once TP1 fills
    run_max = tp1
    for j in range(horizon):
        if ret1 is None:
            if lows[j] <= hard_stop:  # stop first: conservative
                exit_price = min(hard_stop, float(opens[j]))
                ret = exit_price / entry - 1
                return BarrierOutcome(0, "sl", j + 1, entry, ret)
            if highs[j] >= tp1:
                ret1 = tp1 / entry - 1
            continue  # phase-2 trailing starts on the NEXT bar (conservative)
        stop = max(run_max - trail_mult * atr, hard_stop)
        if lows[j] <= stop:
            exit_price = min(stop, float(opens[j]))
            ret = 0.5 * ret1 + 0.5 * (exit_price / entry - 1)
            return BarrierOutcome(int(ret > 0), "scaled", j + 1, entry, ret)
        run_max = max(run_max, float(highs[j]))
    if ret1 is None:
        ret = timeout_close / entry - 1
        return BarrierOutcome(int(ret > 0), "timeout", horizon, entry, ret)
    ret = 0.5 * ret1 + 0.5 * (timeout_close / entry - 1)
    return BarrierOutcome(int(ret > 0), "scaled_timeout", horizon, entry, ret)


def label_candidate_breakeven(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    tp_mult: float = 5.0,
    sl_mult: float = 2.0,
    be_trigger: float = 1.5,
    horizon: int = HORIZON_BARS,
    atr_pct_min: float = ATR_PCT_MIN,
) -> BarrierOutcome | None:
    """H2 breakeven shift: classic TP/SL, but once price has traded
    be_trigger x ATR in favor the stop moves to the entry price. Conservative
    intra-bar ordering: the CURRENT stop is checked before the trigger or the
    target inside the same bar, so a spike-up-then-down bar can't grant the
    breakeven protection retroactively.
    """
    ctx = _entry_context(frame, signal_i, horizon, atr_pct_min)
    if ctx is None:
        return None
    entry, atr, highs, lows, opens, timeout_close = ctx
    upper = entry + tp_mult * atr
    trigger = entry + be_trigger * atr
    stop = entry - sl_mult * atr
    armed = False
    for j in range(horizon):
        if lows[j] <= stop:  # current stop first: conservative
            exit_price = min(stop, float(opens[j]))
            ret = exit_price / entry - 1
            return BarrierOutcome(int(ret > 0), "be" if armed else "sl", j + 1, entry, ret)
        if highs[j] >= upper:
            return BarrierOutcome(1, "tp", j + 1, entry, upper / entry - 1)
        if not armed and highs[j] >= trigger:
            armed = True
            stop = entry
    ret = timeout_close / entry - 1
    return BarrierOutcome(int(ret > 0), "timeout", horizon, entry, ret)


def label_candidate_ma_exit(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    ma_col: str = "ema21",
    atr_pct_min: float = ATR_PCT_MIN,
    horizon: int = HORIZON_BARS,
    entry: EntryMode = "next_open",
) -> BarrierOutcome | None:
    """Trend MA exit (long): leave when close drops below `ma_col` (default EMA21).

    Entry / ATR floor / horizon match the mainline contract. No fixed TP/SL —
    the MA cross defines the pulse end. Label = 1 iff realized_ret > 0 at exit.
    Callers that want a slower trend filter should pass ma_col='ema55'.
    """
    ctx = _entry_context(frame, signal_i, horizon, atr_pct_min, entry=entry)
    if ctx is None or ma_col not in frame.columns:
        return None
    fill, _, _, _, _, timeout_close = ctx
    path_i = signal_i + 1
    closes = frame["close"].to_numpy()[path_i : path_i + horizon]
    ma_values = frame[ma_col].to_numpy()[path_i : path_i + horizon]
    for j in range(horizon):
        close = float(closes[j])
        ma_value = float(ma_values[j])
        if np.isfinite(close) and np.isfinite(ma_value) and close < ma_value:
            ret = close / fill - 1
            return BarrierOutcome(int(ret > 0), "ma_exit", j + 1, fill, ret)
    ret = timeout_close / fill - 1
    return BarrierOutcome(int(ret > 0), "timeout", horizon, fill, ret)


def label_short_candidate_ma_exit(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    ma_col: str = "ema21",
    atr_pct_min: float = ATR_PCT_MIN,
    horizon: int = HORIZON_BARS,
    entry: EntryMode = "next_open",
) -> BarrierOutcome | None:
    """Trend MA exit (short): leave when close rises above `ma_col`."""
    ctx = _entry_context(frame, signal_i, horizon, atr_pct_min, entry=entry)
    if ctx is None or ma_col not in frame.columns:
        return None
    fill, _, _, _, _, timeout_close = ctx
    if timeout_close <= 0:
        return None
    path_i = signal_i + 1
    closes = frame["close"].to_numpy()[path_i : path_i + horizon]
    ma_values = frame[ma_col].to_numpy()[path_i : path_i + horizon]
    for j in range(horizon):
        close = float(closes[j])
        ma_value = float(ma_values[j])
        if np.isfinite(close) and np.isfinite(ma_value) and close > ma_value:
            if close <= 0:
                return None
            ret = fill / close - 1
            return BarrierOutcome(int(ret > 0), "ma_exit", j + 1, fill, ret)
    ret = fill / timeout_close - 1
    return BarrierOutcome(int(ret > 0), "timeout", horizon, fill, ret)


def label_candidate_structure_exit(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    direction: int,
    redense_fast_max: float = 0.0028,
    atr_pct_min: float = ATR_PCT_MIN,
    horizon: int = HORIZON_BARS,
    entry: EntryMode = "next_open",
) -> BarrierOutcome | None:
    """Structure / re-dense exit for long (+1) or short (−1).

    Exit at bar close when either:
      - close returns to the opposite side of contemporaneous cluster_mid, or
      - fast_spread re-contracts to ≤ `redense_fast_max` (dense again).
    Timeout at horizon close. Label = 1 iff realized_ret > 0.
    """
    if direction == 0:
        return None
    need = {"cluster_max", "cluster_min", "fast_spread"}
    if not need.issubset(frame.columns):
        return None
    ctx = _entry_context(frame, signal_i, horizon, atr_pct_min, entry=entry)
    if ctx is None:
        return None
    fill, _, _, _, _, timeout_close = ctx
    if timeout_close <= 0:
        return None
    path_i = signal_i + 1
    closes = frame["close"].to_numpy(dtype=float)[path_i : path_i + horizon]
    cmax = frame["cluster_max"].to_numpy(dtype=float)[path_i : path_i + horizon]
    cmin = frame["cluster_min"].to_numpy(dtype=float)[path_i : path_i + horizon]
    fast = frame["fast_spread"].to_numpy(dtype=float)[path_i : path_i + horizon]
    for j in range(horizon):
        close = float(closes[j])
        mid = (float(cmax[j]) + float(cmin[j])) / 2.0
        fs = float(fast[j])
        if not np.isfinite(close) or close <= 0:
            continue
        mid_flip = (
            np.isfinite(mid)
            and mid > 0
            and ((direction > 0 and close < mid) or (direction < 0 and close > mid))
        )
        redense = np.isfinite(fs) and fs <= redense_fast_max
        if mid_flip or redense:
            ret = (close / fill - 1) if direction > 0 else (fill / close - 1)
            return BarrierOutcome(int(ret > 0), "structure", j + 1, fill, ret)
    ret = (
        (timeout_close / fill - 1) if direction > 0 else (fill / timeout_close - 1)
    )
    return BarrierOutcome(int(ret > 0), "timeout", horizon, fill, ret)


def label_candidate_time_stop(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    direction: int,
    hold_bars: int,
    atr_pct_min: float = ATR_PCT_MIN,
    entry: EntryMode = "next_open",
) -> BarrierOutcome | None:
    """Force flat at the close of path bar `hold_bars` (1-based). No TP/SL."""
    if direction == 0 or hold_bars < 1:
        return None
    ctx = _entry_context(frame, signal_i, hold_bars, atr_pct_min, entry=entry)
    if ctx is None:
        return None
    fill, _, _, _, _, timeout_close = ctx
    if timeout_close <= 0:
        return None
    ret = (timeout_close / fill - 1) if direction > 0 else (fill / timeout_close - 1)
    return BarrierOutcome(int(ret > 0), "time_stop", hold_bars, fill, ret)


def label_candidate_sl_only(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    direction: int,
    sl_mult: float = 2.0,
    atr_pct_min: float = ATR_PCT_MIN,
    horizon: int = HORIZON_BARS,
    entry: EntryMode = "next_open",
) -> BarrierOutcome | None:
    """Classic trend: fixed SL only, no TP; timeout at horizon close."""
    if direction == 0:
        return None
    ctx = _entry_context(frame, signal_i, horizon, atr_pct_min, entry=entry)
    if ctx is None:
        return None
    fill, atr, highs, lows, opens, timeout_close = ctx
    if timeout_close <= 0:
        return None
    if direction > 0:
        stop = fill - sl_mult * atr
        for j in range(horizon):
            if lows[j] <= stop:
                exit_price = min(stop, float(opens[j]))
                ret = exit_price / fill - 1
                return BarrierOutcome(int(ret > 0), "sl", j + 1, fill, ret)
        ret = timeout_close / fill - 1
        return BarrierOutcome(int(ret > 0), "timeout", horizon, fill, ret)
    stop = fill + sl_mult * atr
    for j in range(horizon):
        if highs[j] >= stop:
            exit_price = max(stop, float(opens[j]))
            if exit_price <= 0:
                return None
            ret = fill / exit_price - 1
            return BarrierOutcome(int(ret > 0), "sl", j + 1, fill, ret)
    ret = fill / timeout_close - 1
    return BarrierOutcome(int(ret > 0), "timeout", horizon, fill, ret)


def label_candidate_time_decay(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    tp_mult: float = 5.0,
    sl_mult: float = 2.0,
    tighten_every: int = 12,
    tighten_step: float = 0.25,
    horizon: int = HORIZON_BARS,
    atr_pct_min: float = ATR_PCT_MIN,
) -> BarrierOutcome | None:
    """H4 time-decay stop: TP fixed, SL tightens by tighten_step x ATR every
    tighten_every bars held (long: stop moves up). Conservative intra-bar
    order: current stop checked before TP. Ambiguous same-bar hit → SL.
    """
    ctx = _entry_context(frame, signal_i, horizon, atr_pct_min)
    if ctx is None:
        return None
    entry, atr, highs, lows, opens, timeout_close = ctx
    upper = entry + tp_mult * atr
    for j in range(horizon):
        steps = j // max(1, tighten_every)
        sl_dist = max(sl_mult - steps * tighten_step, 0.0) * atr
        stop = entry - sl_dist
        if lows[j] <= stop:
            exit_price = min(stop, float(opens[j]))
            ret = exit_price / entry - 1
            return BarrierOutcome(int(ret > 0), "sl_decay", j + 1, entry, ret)
        if highs[j] >= upper:
            return BarrierOutcome(1, "tp", j + 1, entry, upper / entry - 1)
    ret = timeout_close / entry - 1
    return BarrierOutcome(int(ret > 0), "timeout", horizon, entry, ret)


def vol_adaptive_mults(
    atr_pct: float,
    q33: float,
    q66: float,
    *,
    tp_by_tertile: tuple[float, float, float] = (4.0, 5.0, 6.0),
    sl_by_tertile: tuple[float, float, float] = (1.6, 2.0, 2.4),
) -> tuple[float, float, int]:
    """Map atr_pct to (tp_mult, sl_mult, tertile_index 0/1/2). Low vol → narrow."""
    if not np.isfinite(atr_pct):
        return tp_by_tertile[1], sl_by_tertile[1], 1
    if atr_pct <= q33:
        return tp_by_tertile[0], sl_by_tertile[0], 0
    if atr_pct <= q66:
        return tp_by_tertile[1], sl_by_tertile[1], 1
    return tp_by_tertile[2], sl_by_tertile[2], 2


def label_candidate_vol_adaptive(
    frame: pd.DataFrame,
    signal_i: int,
    *,
    q33: float,
    q66: float,
    tp_by_tertile: tuple[float, float, float] = (4.0, 5.0, 6.0),
    sl_by_tertile: tuple[float, float, float] = (1.6, 2.0, 2.4),
    horizon: int = HORIZON_BARS,
    atr_pct_min: float = ATR_PCT_MIN,
) -> BarrierOutcome | None:
    """H5 vol-adaptive barriers: TP/SL mults from atr_pct tertiles (low=narrow).

    Tertile edges (q33, q66) must be precomputed on train/dev only — caller
    responsibility. Delegates to label_candidate with the chosen mults.
    """
    atr_pct = float(frame["atr_pct"].iloc[signal_i])
    tp, sl, _ = vol_adaptive_mults(
        atr_pct, q33, q66, tp_by_tertile=tp_by_tertile, sl_by_tertile=sl_by_tertile
    )
    return label_candidate(
        frame, signal_i, tp_mult=tp, sl_mult=sl, horizon=horizon, atr_pct_min=atr_pct_min
    )
