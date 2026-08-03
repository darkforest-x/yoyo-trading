"""Feature engineering for the judgment classifier.

All features are computed causally: at bar i only data from bars <= i is used
(rolling / shift / pct_change on past values only). Features are added as
columns on the enriched indicator frame from candidates.add_indicators, then
sampled at candidate positions.

Long vs short column semantics (2026-07-24 short-only path):
  FEATURE_COLUMNS keeps stable names for LightGBM, but short pools MUST pass
  through align_short_feature_rows so directional columns carry short meaning:
    ext_up ← ext_down, order_score ← down_order_score, drawdown24 ← runup24,
    close_vs_ema* flipped to MA/close-1, ret_* and slow_slope_12 negated.
  See docs/learnings/short-mirrors-need-directional-feature-semantics.md.
  Causal inputs for those remaps live on the indicator frame from
  candidates.add_indicators (ext_down, runup24, down_order_score, emas, close).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DENSE_SPREAD_MAX = 0.0028  # strict fast_spread threshold, reused for duration

FEATURE_COLUMNS = [
    # MA-spread level and convergence dynamics
    "ma_spread_pct",
    "full_spread",
    "fast_slow_gap",
    "full_ratio_min48",
    "spread_mean8",
    "spread_mean24",
    "spread_chg8",
    "spread_chg24",
    "spread_pos96",
    # dense-state persistence
    "dense_run_len",
    "dense_frac48",
    # price position relative to MAs (long semantics unless short-aligned)
    "ext_up",
    "close_vs_ema55",
    "close_vs_ema200",
    "order_score",
    "slow_slope_12",
    # volume
    "volume_ratio",
    "volume_z",
    "vol_ratio_mean8",
    # volatility
    "atr_pct",
    "atr_pct_ratio96",
    "pre_range48",
    "pre_range168",
    "drawdown24",
    # recent momentum
    "ret_4",
    "ret_12",
    "ret_24",
    "ret_48",
]

# Columns rewritten by align_short_feature_rows (same names, short causal meaning).
SHORT_DIRECTIONAL_COLUMNS = (
    "ext_up",
    "close_vs_ema55",
    "close_vs_ema200",
    "order_score",
    "slow_slope_12",
    "drawdown24",
    "ret_4",
    "ret_12",
    "ret_24",
    "ret_48",
)


def add_features(enriched: pd.DataFrame) -> pd.DataFrame:
    """Add FEATURE_COLUMNS to an indicator frame (causal only)."""
    out = enriched.copy()
    close = out["close"].replace(0, np.nan)
    spread = out["ma_spread_pct"]

    out["spread_mean8"] = spread.rolling(8).mean()
    out["spread_mean24"] = spread.rolling(24).mean()
    # convergence speed: negative = spread shrinking over the last N bars
    out["spread_chg8"] = spread - spread.shift(8)
    out["spread_chg24"] = spread - spread.shift(24)
    # where the current spread sits inside its trailing 96-bar range
    roll_min = spread.rolling(96, min_periods=48).min()
    roll_max = spread.rolling(96, min_periods=48).max()
    out["spread_pos96"] = (spread - roll_min) / (roll_max - roll_min).replace(0, np.nan)

    dense = (spread <= DENSE_SPREAD_MAX).astype(int)
    # consecutive dense bars ending at i
    grp = (dense == 0).cumsum()
    out["dense_run_len"] = dense.groupby(grp).cumsum()
    out["dense_frac48"] = dense.rolling(48, min_periods=24).mean()

    out["close_vs_ema55"] = close / out["ema55"].replace(0, np.nan) - 1
    out["close_vs_ema200"] = close / out["ema200"].replace(0, np.nan) - 1

    out["vol_ratio_mean8"] = out["volume_ratio"].rolling(8).mean()
    atr_mean96 = out["atr_pct"].rolling(96, min_periods=48).mean().replace(0, np.nan)
    out["atr_pct_ratio96"] = out["atr_pct"] / atr_mean96

    for n in (4, 12, 24, 48):
        out[f"ret_{n}"] = close.pct_change(n)

    return out.replace([np.inf, -np.inf], np.nan)


def extract_feature_rows(featured: pd.DataFrame, signal_indices: list[int]) -> pd.DataFrame:
    """Take feature vectors at the given candidate positions (long column semantics)."""
    rows = featured.iloc[signal_indices]
    return rows[FEATURE_COLUMNS].reset_index(drop=True)


def align_short_feature_rows(
    feature_rows: pd.DataFrame,
    featured: pd.DataFrame,
    signal_indices: list[int],
) -> pd.DataFrame:
    """Rewrite directional FEATURE_COLUMNS so names keep short causal meaning.

    Requires indicator columns on `featured`: ext_down, runup24, down_order_score,
    ema55, ema200, close, plus ret_* / slow_slope_12 from add_features.
    Non-directional columns (spreads, volume, ATR, ranges) are left unchanged.
    """
    aligned = feature_rows.copy()
    source = featured.iloc[signal_indices].reset_index(drop=True)
    close = source["close"].replace(0, np.nan)
    aligned["ext_up"] = source["ext_down"]
    aligned["close_vs_ema55"] = source["ema55"] / close - 1
    aligned["close_vs_ema200"] = source["ema200"] / close - 1
    aligned["order_score"] = source["down_order_score"]
    aligned["slow_slope_12"] = -source["slow_slope_12"]
    aligned["drawdown24"] = source["runup24"]
    for bars in (4, 12, 24, 48):
        aligned[f"ret_{bars}"] = -source[f"ret_{bars}"]
    return aligned.replace([np.inf, -np.inf], np.nan)


def extract_feature_rows_for_side(
    featured: pd.DataFrame,
    signal_indices: list[int],
    side: str,
) -> pd.DataFrame:
    """Extract FEATURE_COLUMNS; short side applies directional remapping."""
    if side not in ("long", "short"):
        raise ValueError(f"side must be long|short, got {side!r}")
    rows = extract_feature_rows(featured, signal_indices)
    if side == "short":
        return align_short_feature_rows(rows, featured, signal_indices)
    return rows


def extract_feature_rows_for_semantics(
    featured: pd.DataFrame,
    signal_indices: list[int],
    *,
    feature_semantics: str,
    side: str,
) -> pd.DataFrame:
    """Extract one declared model coordinate system, with no implicit default.

    Inputs are the 28 ``FEATURE_COLUMNS`` already computed causally by
    :func:`add_features` plus, for ``side_aligned_v1`` short rows, the same-bar
    indicator columns documented by :func:`align_short_feature_rows`. No row
    after any requested signal index is read. ``legacy_unaligned`` preserves the
    historical column values; ``side_aligned_v1`` applies the declared trade-side
    transform. Unknown semantics fail rather than guessing.
    """
    semantics = str(feature_semantics).strip()
    if semantics == "legacy_unaligned":
        return extract_feature_rows(featured, signal_indices)
    if semantics == "side_aligned_v1":
        return extract_feature_rows_for_side(featured, signal_indices, side)
    raise ValueError(
        f"unknown feature_semantics={feature_semantics!r}; "
        "expected legacy_unaligned|side_aligned_v1"
    )
