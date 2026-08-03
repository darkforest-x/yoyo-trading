"""Train and evaluate the judgment-layer LightGBM model.

Usage:
  python3 -m src.judgment.train --data PATH --tag TAG [--side long|short|auto]
  python3 -m src.judgment.train --data PATH --tag TAG --objective regression
  # Never pass --eval-holdout unless the owner explicitly authorizes a holdout burn.

Split discipline (strict time-based, no shuffling):
- HOLDOUT_START is frozen: samples with signal_time >= 2026-05-04 00:00 UTC
  are never touched by training or tuning; they are evaluated only when
  --eval-holdout is passed (once, results reported as-is).
- The remaining samples are split by time into train (first 80%) and
  val (last 20%).

Objective (ACTIVE ≈ v11 mainline):
- binary: classify label (TP vs not) — historical default / CLI default
- regression: predict realized_ret; rank by score; entry gate = val score q90
  (same philosophy as frozen_tp5_sl2_swap_yolo_v11_reg)

Side discipline (2026-07-24 short-only):
- Datasets with a `side` column must be homogeneous (no long/short mix).
- `--side short` (or auto-detect from the column / tag containing "short")
  asserts every row is short and refuses mixed pools.
- Short tags should include `short` so outputs stay pool-tagged
  (e.g. p2b_yolo_short_30_6m_reg).

Outputs metrics JSON to analysis/output/{tag}_metrics.json and feature
importance to analysis/output/{tag}_feature_importance.csv.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from yoyo.data.bars import BAR_CHOICES, purge_window
from yoyo.layers.l2_judgment.features import FEATURE_COLUMNS

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "judgment_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "analysis" / "output"

HOLDOUT_START = pd.Timestamp("2026-05-04 00:00:00", tz="UTC")  # frozen, do not tune on >= this
TRAIN_FRACTION = 0.8
THRESHOLDS = (0.4, 0.5, 0.6, 0.7)
# Align with frozen.DEFAULT_SCORE_QUANTILE (v11 ACTIVE entry gate).
SCORE_QUANTILE = 0.9
from yoyo.contracts.costs import LEGACY_P0_ROUND_TRIP as ROUND_TRIP_COST  # reporting-only, see src/costs.py
SEED = 42

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "verbosity": -1,
}


DEFAULT_HORIZON_BARS = 72
DEFAULT_BAR = "15m"
PURGE_WINDOW = purge_window(DEFAULT_HORIZON_BARS, DEFAULT_BAR)


def resolve_side(data: pd.DataFrame, *, side_arg: str, tag: str) -> str:
    """Resolve training side and refuse mixed long/short pools.

    side_arg:
      - long|short: assert column (if present) matches; fail on mix
      - auto: prefer unique `side` column; else infer from tag containing 'short'
    """
    if side_arg not in ("long", "short", "auto"):
        raise ValueError(f"side must be long|short|auto, got {side_arg!r}")
    col_side: str | None = None
    if "side" in data.columns:
        sides = sorted({str(s).lower() for s in data["side"].dropna().unique()})
        if not sides:
            col_side = None
        elif len(sides) > 1:
            raise SystemExit(
                f"mixed side values in dataset: {sides}; "
                "judgment main tables must be long-only or short-only"
            )
        else:
            col_side = sides[0]
            if col_side not in ("long", "short"):
                raise SystemExit(f"unknown side value {col_side!r}; expected long|short")

    tag_implies_short = "short" in tag.lower()
    if side_arg == "auto":
        if col_side is not None:
            resolved = col_side
        elif tag_implies_short:
            resolved = "short"
        else:
            resolved = "long"
    else:
        resolved = side_arg
        if col_side is not None and col_side != resolved:
            raise SystemExit(
                f"--side {resolved} but dataset side column is {col_side!r}"
            )

    if resolved == "short" and not tag_implies_short:
        raise SystemExit(
            "short-only training requires --tag containing 'short' "
            f"(got {tag!r}) so outputs stay pool-tagged"
        )
    if resolved == "long" and tag_implies_short and col_side == "long":
        raise SystemExit(
            f"tag {tag!r} implies short but dataset/side is long; refuse ambiguous run"
        )
    return resolved


def load_splits(
    dataset_path: Path = DATASET_PATH, *, horizon_bars: int = DEFAULT_HORIZON_BARS, bar: str = DEFAULT_BAR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    purge = purge_window(horizon_bars, bar)
    data = pd.read_csv(dataset_path, parse_dates=["signal_time"])
    data = data.sort_values("signal_time").reset_index(drop=True)
    dev = data[data["signal_time"] < HOLDOUT_START - purge].reset_index(drop=True)
    holdout = data[data["signal_time"] >= HOLDOUT_START].reset_index(drop=True)
    split_i = int(len(dev) * TRAIN_FRACTION)
    train, val = dev.iloc[:split_i], dev.iloc[split_i:]
    # purge train samples whose triple-barrier window overlaps the val period
    val_start = val["signal_time"].min()
    train = train[train["signal_time"] < val_start - purge]
    return train, val, holdout


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    returns: np.ndarray,
    *,
    objective: str = "binary",
    returns_are_net: bool = False,
) -> dict:
    """Rank/threshold metrics. Primary economic gate = top-decile net.

    For regression, y_score is predicted realized_ret; AUC/PR are secondary
    rank diagnostics against the binary label, not the success criterion.

    If ``returns_are_net`` (target already net of fees, e.g. net_barrier_taker),
    do not subtract ROUND_TRIP_COST again.
    """
    out = {
        "n": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "thresholds": {},
    }
    for threshold in THRESHOLDS:
        pred = (y_score >= threshold).astype(int)
        out["thresholds"][str(threshold)] = {
            "n_signals": int(pred.sum()),
            "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        }
    # top-decile economic metrics. When `returns` is already a net column
    # (e.g. v10 target_column = net_barrier_taker), do **not** subtract cost
    # again — that double-counted RT fees (H10 / GPT review 2026-07-31).
    # Opt in via returns_are_net=True from train_model for regression nets.
    cost = 0.0 if returns_are_net else float(ROUND_TRIP_COST)
    k = max(1, len(y_score) // 10)
    top_idx = np.argsort(y_score)[-k:]
    out["top_decile"] = {
        "n": int(k),
        "mean_realized_ret": round(float(returns[top_idx].mean()), 5),
        "mean_net_ret": round(float(returns[top_idx].mean() - cost), 5),
        "win_rate": round(float(y_true[top_idx].mean()), 4),
        "returns_are_net": bool(returns_are_net),
        "cost_subtracted": cost,
    }
    out["all_mean_net_ret"] = round(float(returns.mean() - cost), 5)
    if objective == "regression":
        rho = spearmanr(y_score, returns).statistic
        out["spearman_score_vs_ret"] = None if rho is None or np.isnan(rho) else round(float(rho), 4)
        thr = float(np.quantile(y_score, SCORE_QUANTILE))
        q_mask = y_score >= thr
        out["threshold_val_q90"] = round(thr, 8)
        out["score_quantile"] = SCORE_QUANTILE
        out["above_q90"] = {
            "n": int(q_mask.sum()),
            "pass_rate": round(float(q_mask.mean()), 4),
            "mean_realized_ret": round(float(returns[q_mask].mean()), 5) if q_mask.any() else None,
            "mean_net_ret": (
                round(float(returns[q_mask].mean() - cost), 5) if q_mask.any() else None
            ),
            "win_rate": round(float(y_true[q_mask].mean()), 4) if q_mask.any() else None,
        }
    return out


def permutation_pvalue(y_true: np.ndarray, y_prob: np.ndarray, *, n_perm: int = 1000) -> float:
    """P(label-permuted AUC >= observed AUC); tests AUC > 0.5 significance."""
    rng = np.random.default_rng(SEED)
    observed = roc_auc_score(y_true, y_prob)
    hits = 0
    for _ in range(n_perm):
        if roc_auc_score(rng.permutation(y_true), y_prob) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def train_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    objective: str = "binary",
) -> lgb.Booster:
    """Train judgment model.

    objective:
      - binary: classify label (TP vs not) — historical default
      - regression: rank by predicted realized_ret (economic target; 2026-07-15+)
    """
    cols = list(feature_columns)
    params = dict(LGB_PARAMS)
    if objective == "regression":
        params["objective"] = "regression"
        y_train, y_val = train["realized_ret"], val["realized_ret"]
    elif objective == "binary":
        params["objective"] = "binary"
        y_train, y_val = train["label"], val["label"]
    else:
        raise ValueError(f"unknown objective {objective!r}; expected binary|regression")
    dtrain = lgb.Dataset(train[cols], label=y_train)
    dval = lgb.Dataset(val[cols], label=y_val, reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=600,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )


def train_baseline(train: pd.DataFrame) -> tuple[StandardScaler, LogisticRegression]:
    """Naive baseline: logistic regression on ma_spread_pct alone."""
    scaler = StandardScaler()
    x = scaler.fit_transform(train[["ma_spread_pct"]].fillna(0))
    model = LogisticRegression()
    model.fit(x, train["label"])
    return scaler, model


def baseline_prob(scaler: StandardScaler, model: LogisticRegression, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(scaler.transform(frame[["ma_spread_pct"]].fillna(0)))[:, 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-holdout", action="store_true", help="Evaluate the frozen holdout (run once).")
    parser.add_argument("--data", type=Path, default=DATASET_PATH, help="Dataset CSV from build_dataset.")
    parser.add_argument(
        "--tag",
        default="p2b",
        help="Output file prefix; short-only runs must include 'short' (e.g. p2b_v2_strict_short).",
    )
    parser.add_argument(
        "--side",
        choices=("long", "short", "auto"),
        default="auto",
        help="Force side assertion. auto = unique side column, else tag containing 'short'.",
    )
    parser.add_argument(
        "--objective",
        choices=("binary", "regression"),
        default="binary",
        help="binary=label classifier (legacy CLI default); "
        "regression=predict realized_ret (v11 ACTIVE philosophy).",
    )
    parser.add_argument("--bar", choices=BAR_CHOICES, default=DEFAULT_BAR)
    parser.add_argument("--horizon-bars", type=int, default=DEFAULT_HORIZON_BARS)
    parser.add_argument(
        "--features-file",
        type=Path,
        default=None,
        help="Optional text file of feature names (one per line; # comments ok). "
        "Must be a non-empty subset of FEATURE_COLUMNS. Single-variable ablations only.",
    )
    args = parser.parse_args()

    feature_columns = list(FEATURE_COLUMNS)
    if args.features_file is not None:
        if not args.features_file.exists():
            raise SystemExit(f"--features-file missing: {args.features_file}")
        wanted: list[str] = []
        for ln in args.features_file.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            wanted.append(s)
        unknown = [c for c in wanted if c not in FEATURE_COLUMNS]
        if unknown:
            raise SystemExit(f"--features-file has unknown columns: {unknown}")
        if not wanted:
            raise SystemExit("--features-file is empty")
        # Preserve file order (importance rank) but de-dupe.
        seen: set[str] = set()
        feature_columns = []
        for c in wanted:
            if c not in seen:
                seen.add(c)
                feature_columns.append(c)

    raw = pd.read_csv(args.data, parse_dates=["signal_time"])
    side = resolve_side(raw, side_arg=args.side, tag=args.tag)

    train, val, holdout = load_splits(args.data, horizon_bars=args.horizon_bars, bar=args.bar)
    model = train_model(
        train, val, feature_columns=feature_columns, objective=args.objective
    )
    scaler, base = train_baseline(train)

    val_score = model.predict(val[feature_columns], num_iteration=model.best_iteration)
    # Auto-detect: realized_ret already equals net_barrier_taker → no second cost cut.
    returns_are_net = False
    if "net_barrier_taker" in val.columns and "realized_ret" in val.columns:
        a = val["realized_ret"].to_numpy(dtype=float)
        b = val["net_barrier_taker"].to_numpy(dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        returns_are_net = bool(m.any() and np.allclose(a[m], b[m], rtol=0, atol=1e-12))
    results = {
        "dataset": str(args.data),
        "side": side,
        "objective": args.objective,
        "score_semantics": (
            "predicted_realized_ret" if args.objective == "regression" else "class_probability"
        ),
        "returns_are_net": returns_are_net,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "bar": args.bar,
        "horizon_bars": args.horizon_bars,
        "purge_window": str(purge_window(args.horizon_bars, args.bar)),
        "holdout_start": str(HOLDOUT_START),
        "holdout_policy": "holdout excluded from training and threshold selection; not evaluated"
        if not args.eval_holdout
        else "holdout evaluated once (owner-authorized)",
        "splits": {
            "train": {"n": len(train), "range": [str(train["signal_time"].min()), str(train["signal_time"].max())]},
            "val": {"n": len(val), "range": [str(val["signal_time"].min()), str(val["signal_time"].max())]},
            "holdout": {"n": len(holdout), "range": [str(holdout["signal_time"].min()), str(holdout["signal_time"].max())]},
        },
        "best_iteration": model.best_iteration,
        "val": evaluate(
            val["label"].to_numpy(),
            val_score,
            val["realized_ret"].to_numpy(),
            objective=args.objective,
            returns_are_net=returns_are_net,
        ),
        "val_permutation_p": permutation_pvalue(val["label"].to_numpy(), val_score),
        "val_baseline_ma_spread_logreg": evaluate(
            val["label"].to_numpy(),
            baseline_prob(scaler, base, val),
            val["realized_ret"].to_numpy(),
            objective="binary",
            returns_are_net=returns_are_net,
        ),
    }

    importance = pd.DataFrame({
        "feature": feature_columns,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    results["feature_importance_top10"] = importance.head(10)[["feature", "gain"]].to_dict("records")

    if args.eval_holdout:
        hold_score = model.predict(holdout[feature_columns], num_iteration=model.best_iteration)
        results["holdout"] = evaluate(
            holdout["label"].to_numpy(),
            hold_score,
            holdout["realized_ret"].to_numpy(),
            objective=args.objective,
            returns_are_net=returns_are_net,
        )
        results["holdout_permutation_p"] = permutation_pvalue(
            holdout["label"].to_numpy(), hold_score
        )
        results["holdout_baseline_ma_spread_logreg"] = evaluate(
            holdout["label"].to_numpy(),
            baseline_prob(scaler, base, holdout),
            holdout["realized_ret"].to_numpy(),
            objective="binary",
            returns_are_net=returns_are_net,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(OUTPUT_DIR / f"{args.tag}_feature_importance.csv", index=False)
    (OUTPUT_DIR / f"{args.tag}_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
