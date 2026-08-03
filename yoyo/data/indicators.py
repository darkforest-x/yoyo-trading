"""Bar-level indicators and the rules-based candidate scan. Below every layer.

Sixteen modules across detection, judgment, the dashboard and three side projects
import this, which is the tell: EMA computation over bars is not any one layer's
property. It sits in data/ so that L1 and L2 can both use it without importing
each other.

One honest loose end. `scan_candidates` / `short_mask` / the threshold presets are
the RULES-based detector -- an alternative to YOLO, and arguably L1 rather than
data. They stay here for now because iron rule 12 makes production YOLO-only and
the rules path is research-only, so promoting it to a layer would overstate its
role. When the rules path is either promoted or retired, that decision moves it;
fudging the boundary now to look tidy would be the worse error.

Thresholds in this file are owner decisions (CLAUDE.md). Code does not change them.

Moved from src/judgment/candidates.py on 2026-08-03. Behaviour unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EMA_PERIODS = (8, 13, 21, 34, 55, 144, 200)
CLUSTER_EMAS = ("ema8", "ema13", "ema21", "ema34", "ema55")
ALL_MAS = (*CLUSTER_EMAS, "ema144", "ema200")

# Warmup: pre_range168 needs 168 bars; the old pipeline effectively required
# ~290 bars of history before the first candidate. Keep the same order.
WARMUP_BARS = 288
# Min bars between two candidates on the same series (old strict min_gap).
MIN_GAP_BARS = 18

STRICT_THRESHOLDS = {
    "fast_spread_max": 0.0028,
    "full_spread_max": 0.0055,
    "fast_slow_gap_max": 0.0035,
    "full_ratio_min48_max": 1.45,
    "pre_range48_max": 0.032,
    "pre_range168_max": 0.075,
    "drawdown24_max": 0.007,
    "ext_up_min": -0.0015,
    "ext_up_max": 0.0075,
    "order_score_min": 3,
    "slow_slope_abs_max": 0.0009,
    "zero_volume96_max": 0.02,
    "volume_ratio_min": 0.7,
}

# "expanded" pool (owner decision 2026-07-07): a ~1.6x relaxation of the
# strict spread/range gates, defined HERE (not ported from the old project's
# expanded mode, whose exact thresholds we chose not to inherit). Purpose:
# grow the candidate pool several-fold so the judgment model has more to
# learn from; the in-pool base win rate is expected to drop, which is fine --
# the model's job is ranking within the pool.
EXPANDED_THRESHOLDS = {
    "fast_spread_max": 0.0045,
    "full_spread_max": 0.0088,
    "fast_slow_gap_max": 0.0056,
    "full_ratio_min48_max": 1.80,
    "pre_range48_max": 0.048,
    "pre_range168_max": 0.110,
    "drawdown24_max": 0.012,
    "ext_up_min": -0.0030,
    "ext_up_max": 0.0120,
    "order_score_min": 3,
    "slow_slope_abs_max": 0.0015,
    "zero_volume96_max": 0.02,
    "volume_ratio_min": 0.5,
}

THRESHOLD_PRESETS = {"strict": STRICT_THRESHOLDS, "expanded": EXPANDED_THRESHOLDS}


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add EMAs, ATR, volume stats and strict-rule metrics (no lookahead)."""
    out = frame.copy()
    close = out["close"].replace(0, np.nan)
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    for span in EMA_PERIODS:
        out[f"ema{span}"] = out["close"].ewm(span=span, adjust=False).mean()

    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["atr_pct"] = out["atr14"] / close

    vol_mean = volume.rolling(96, min_periods=20).mean()
    vol_std = volume.rolling(96, min_periods=20).std().replace(0, np.nan)
    out["volume_z"] = ((volume - vol_mean) / vol_std).fillna(0)
    vol_ratio_base = volume.rolling(20, min_periods=5).mean().replace(0, np.nan)
    out["volume_ratio"] = (volume / vol_ratio_base).fillna(0)

    out["cluster_max"] = out[list(CLUSTER_EMAS)].max(axis=1)
    out["cluster_min"] = out[list(CLUSTER_EMAS)].min(axis=1)
    out["ma_spread_pct"] = (out["cluster_max"] - out["cluster_min"]) / close

    # strict-rule metrics (identical formulas to the old _with_visual_metrics)
    out["fast_spread"] = out["ma_spread_pct"]
    out["full_spread"] = (out[list(ALL_MAS)].max(axis=1) - out[list(ALL_MAS)].min(axis=1)) / close
    fast_mid = (out["cluster_max"] + out["cluster_min"]) / 2
    slow_mid = (out["ema144"] + out["ema200"]) / 2
    out["fast_slow_gap"] = (fast_mid - slow_mid).abs() / close
    out["full_min48"] = out["full_spread"].rolling(48, min_periods=24).min()
    out["full_ratio_min48"] = out["full_spread"] / out["full_min48"].replace(0, np.nan)
    out["pre_range48"] = (high.rolling(48).max() - low.rolling(48).min()) / close
    out["pre_range168"] = (high.rolling(168).max() - low.rolling(168).min()) / close
    out["drawdown24"] = high.rolling(24).max() / close - 1
    out["runup24"] = close / low.rolling(24).min().replace(0, np.nan) - 1
    out["ext_up"] = close / out["cluster_max"].replace(0, np.nan) - 1
    out["ext_down"] = out["cluster_min"] / close - 1
    out["slow_slope_12"] = out["ema200"].pct_change(12)
    out["slow_slope_abs"] = out["slow_slope_12"].abs()
    out["zero_volume96"] = (volume <= 0).rolling(96).mean()
    out["order_score"] = (
        (out["ema8"] >= out["ema13"]).astype(int)
        + (out["ema13"] >= out["ema21"]).astype(int)
        + (out["ema21"] >= out["ema34"]).astype(int)
        + (out["ema34"] >= out["ema55"]).astype(int)
    )
    out["down_order_score"] = (
        (out["ema8"] <= out["ema13"]).astype(int)
        + (out["ema13"] <= out["ema21"]).astype(int)
        + (out["ema21"] <= out["ema34"]).astype(int)
        + (out["ema34"] <= out["ema55"]).astype(int)
    )
    out["trend_order_score"] = out[["order_score", "down_order_score"]].max(axis=1)
    # same shape score the old strict mode used to rank/dedupe candidates
    out["shape_score"] = (
        (0.0028 - out["fast_spread"]).clip(lower=0) * 9000
        + (0.0055 - out["full_spread"]).clip(lower=0) * 6500
        + (0.0035 - out["fast_slow_gap"]).clip(lower=0) * 5000
        + (1.45 - out["full_ratio_min48"]).clip(lower=0) * 8
        + out["volume_ratio"].clip(upper=4)
    )
    out["short_shape_score"] = (
        (0.0028 - out["fast_spread"]).clip(lower=0) * 9000
        + (0.0055 - out["full_spread"]).clip(lower=0) * 6500
        + (0.0035 - out["fast_slow_gap"]).clip(lower=0) * 5000
        + (1.45 - out["full_ratio_min48"]).clip(lower=0) * 8
        + out["volume_ratio"].clip(upper=4)
    )
    return out.replace([np.inf, -np.inf], np.nan)


def strict_mask(enriched: pd.DataFrame, mode: str = "strict") -> pd.Series:
    """The old strict rule, minus its lookahead (future_*) filters.

    `mode` selects a threshold preset: "strict" or "expanded".
    """
    t = THRESHOLD_PRESETS[mode]
    return (
        (enriched["fast_spread"] <= t["fast_spread_max"])
        & (enriched["full_spread"] <= t["full_spread_max"])
        & (enriched["fast_slow_gap"] <= t["fast_slow_gap_max"])
        & (enriched["full_ratio_min48"] <= t["full_ratio_min48_max"])
        & (enriched["pre_range48"] <= t["pre_range48_max"])
        & (enriched["pre_range168"] <= t["pre_range168_max"])
        & (enriched["drawdown24"] <= t["drawdown24_max"])
        & (enriched["ext_up"].between(t["ext_up_min"], t["ext_up_max"]))
        & (enriched["order_score"] >= t["order_score_min"])
        & (enriched["slow_slope_abs"] <= t["slow_slope_abs_max"])
        & (enriched["zero_volume96"] <= t["zero_volume96_max"])
        & (enriched["volume_ratio"] >= t["volume_ratio_min"])
    )


def short_mask(enriched: pd.DataFrame, mode: str = "strict") -> pd.Series:
    t = THRESHOLD_PRESETS[mode]
    return (
        (enriched["fast_spread"] <= t["fast_spread_max"])
        & (enriched["full_spread"] <= t["full_spread_max"])
        & (enriched["fast_slow_gap"] <= t["fast_slow_gap_max"])
        & (enriched["full_ratio_min48"] <= t["full_ratio_min48_max"])
        & (enriched["pre_range48"] <= t["pre_range48_max"])
        & (enriched["pre_range168"] <= t["pre_range168_max"])
        & (enriched["runup24"] <= t["drawdown24_max"])
        & (enriched["ext_down"].between(t["ext_up_min"], t["ext_up_max"]))
        & (enriched["down_order_score"] >= t["order_score_min"])
        & (enriched["slow_slope_abs"] <= t["slow_slope_abs_max"])
        & (enriched["zero_volume96"] <= t["zero_volume96_max"])
        & (enriched["volume_ratio"] >= t["volume_ratio_min"])
    )


def scan_candidates(
    enriched: pd.DataFrame, *, horizon_bars: int = 72, mode: str = "strict"
) -> list[int]:
    """Return deduplicated candidate bar indices (positions in `enriched`).

    Requires `horizon_bars + 1` future bars so triple-barrier labeling is
    always feasible (entry at next open + horizon window).
    """
    if len(enriched) < WARMUP_BARS + horizon_bars + 2:
        return []
    mask = strict_mask(enriched, mode).fillna(False)
    idx = np.flatnonzero(mask.to_numpy())
    idx = idx[(idx >= WARMUP_BARS) & (idx + horizon_bars + 1 < len(enriched))]
    if len(idx) == 0:
        return []
    # Greedy dedupe by shape score, same as the old strict pack (min_gap=18).
    scores = enriched["shape_score"].to_numpy()
    selected: list[int] = []
    for i in sorted(idx, key=lambda j: scores[j], reverse=True):
        if all(abs(i - prev) >= MIN_GAP_BARS for prev in selected):
            selected.append(int(i))
    return sorted(selected)


def scan_short_candidates(
    enriched: pd.DataFrame, *, horizon_bars: int = 72, mode: str = "strict"
) -> list[int]:
    if len(enriched) < WARMUP_BARS + horizon_bars + 2:
        return []
    mask = short_mask(enriched, mode).fillna(False)
    idx = np.flatnonzero(mask.to_numpy())
    idx = idx[(idx >= WARMUP_BARS) & (idx + horizon_bars + 1 < len(enriched))]
    if len(idx) == 0:
        return []
    scores = enriched["short_shape_score"].to_numpy()
    selected: list[int] = []
    for i in sorted(idx, key=lambda j: scores[j], reverse=True):
        if all(abs(i - prev) >= MIN_GAP_BARS for prev in selected):
            selected.append(int(i))
    return sorted(selected)
