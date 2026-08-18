"""Live paper pulse for the W10 classifier + LightGBM gate. PAPER ONLY.

An orchestrator, not a layer (CLAUDE.md): it calls L1, then L2, then writes its
own ledger. That is exactly why forward_scan could not simply be "moved" -- the
thing doing the calling belongs above the layers, not inside one.

Why this is a separate entry point from the migrated `l1_detection/scan.py`:
that module is the YOLO *detection* path (v10 mainline). This pipeline is a
*classifier* over rendered W10 windows and a different L2 coordinate system.
Sharing an entry point would mean one flag silently choosing between two
different models' semantics -- the 2026-08-03 fault, rebuilt.

The concurrency that made forward_scan hard to split is absent here by
construction: iron rule 9 restricts live to tip/tip-1/tip-2, so a full 344-symbol
universe is ~1,200 windows, measured at 1.1 min single-threaded against the
fifteen-minute budget. No thread pool is introduced for a problem that does not
exist.

Safety, all asserted in `SAFETY` and re-checked at run time:
  - places no orders (no executor/okx_client import exists in this module)
  - never reads or writes models/ACTIVE
  - never appends data/forward_log.csv (that file is the mainline gate's)
  - refuses to score a bar whose data is older than the freshness gate

Usage:
  PYTHONPATH=. python3 -m yoyo.cli.forward_pulse_w10 --config configs/live_w10_pulse_v1.json --dry-run
  PYTHONPATH=. python3 -m yoyo.cli.forward_pulse_w10 --config configs/live_w10_pulse_v1.json --send
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yoyo.contracts.outcomes import ATR_PCT_MIN, resolve_barrier_outcome  # noqa: E402
from yoyo.contracts.paths import data_root  # noqa: E402
from yoyo.data.indicators import MIN_GAP_BARS, add_indicators  # noqa: E402
from yoyo.data.loader import list_series, load_series  # noqa: E402
from yoyo.data.universe import is_stockish  # noqa: E402
from yoyo.layers.l1_detection.data import ALL_MA_COLS, add_mas  # noqa: E402
from yoyo.layers.l2_judgment.features import (  # noqa: E402
    FEATURE_COLUMNS,
    add_features,
    extract_feature_rows_for_semantics,
)

PROTOCOL = "live_w10_pulse_v1"
BAR = pd.Timedelta(minutes=15)
WINDOW_BARS = 10
WARMUP_BARS = 240

SAFETY = {
    "paper_only": True,
    "places_orders": False,
    "touches_active_bundle": False,
    "writes_forward_log": False,
    "auto_promote": False,
}


class PulseError(RuntimeError):
    """A fail-closed live-pulse contract violation."""


@dataclass(frozen=True)
class Candidate:
    """One scored decision bar that passed both gates but has not been entered."""

    __slots__ = ("symbol", "decision_i", "decision_time", "p_signal", "l2_score",
                 "atr", "atr_pct", "entry_projection")

    symbol: str
    decision_i: int
    decision_time: pd.Timestamp
    p_signal: float
    l2_score: float
    atr: float
    atr_pct: float
    entry_projection: float


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def event_id(symbol: str, decision_time: pd.Timestamp) -> str:
    """Identity from protocol + symbol + decision bar only -- never from wall clock."""
    material = f"{PROTOCOL}\0{symbol}\0{decision_time.isoformat()}".encode("utf-8")
    return "w10live_" + hashlib.sha256(material).hexdigest()[:24]


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise PulseError(f"config protocol must be {PROTOCOL}, got {config.get('protocol')!r}")
    if config.get("side") != "short":
        raise PulseError("this pipeline is short-only (Owner V5 stage lock)")
    for key, expected in SAFETY.items():
        if config.get("safety", {}).get(key) != expected:
            raise PulseError(f"config safety.{key} must be {expected}")
    barrier = config["barrier"]
    if float(barrier["atr_pct_min"]) != float(ATR_PCT_MIN):
        raise PulseError(
            f"config atr_pct_min={barrier['atr_pct_min']} disagrees with the frozen "
            f"contract {ATR_PCT_MIN}; the contract is the authority"
        )
    return config


def live_decision_indices(frame: pd.DataFrame, *, max_tip_age_bars: int) -> list[int]:
    """The only bars live may score: tip, tip-1, tip-2 (iron rule 9).

    A bar qualifies only with a complete causal W10 window and no NaN in the MA
    bundle, which is the same admission test the training renderer applies.
    """
    if max_tip_age_bars < 0:
        raise PulseError("max_tip_age_bars must not be negative")
    out: list[int] = []
    first = len(frame) - 1 - max_tip_age_bars
    for index in range(max(first, WARMUP_BARS), len(frame)):
        window = frame.iloc[index - WINDOW_BARS + 1 : index + 1]
        if len(window) != WINDOW_BARS:
            continue
        if window[list(ALL_MA_COLS)].isna().any().any():
            continue
        out.append(int(index))
    return out


def check_freshness(frame: pd.DataFrame, *, now: pd.Timestamp, max_minutes: float,
                    symbol: str) -> float:
    """Return tip age in minutes, or raise when the series is too stale to score.

    Stale data does not produce a wrong signal, it produces a *late* one, which
    the executor would treat as current. Fail closed per symbol rather than
    letting one lagging series poison a pulse.
    """
    tip = pd.Timestamp(frame["open_time"].iloc[-1])
    tip = tip.tz_localize("UTC") if tip.tzinfo is None else tip.tz_convert("UTC")
    age = (now - tip).total_seconds() / 60.0
    if age > max_minutes:
        raise PulseError(f"{symbol}: tip {tip.isoformat()} is {age:.0f} min old "
                         f"(gate {max_minutes:.0f} min)")
    return age


def atr_eligible(frame: pd.DataFrame, decision_i: int) -> tuple[bool, float, float]:
    """Frozen ATR floor, read at the decision bar and before any threshold gap."""
    atr = float(frame["atr14"].iloc[decision_i])
    atr_pct = float(frame["atr_pct"].iloc[decision_i])
    ok = bool(np.isfinite(atr) and np.isfinite(atr_pct) and atr_pct >= ATR_PCT_MIN)
    return ok, atr, atr_pct


def score_l2(booster, frame: pd.DataFrame, decision_i: int, *, feature_semantics: str) -> float:
    """L2 score for one decision bar, on a frame physically cut at that bar.

    The cut is the causality proof: no later row exists while feature code runs.
    Semantics come from the config, which reflects the artifact -- never from the
    trade side (that collapse is the 2026-08-03 fault).
    """
    featured = add_features(frame.iloc[: decision_i + 1].copy())
    vector = extract_feature_rows_for_semantics(
        featured, [len(featured) - 1], feature_semantics=feature_semantics, side="short",
    )
    if list(vector.columns) != FEATURE_COLUMNS or vector.shape != (1, len(FEATURE_COLUMNS)):
        raise PulseError("feature extraction does not match the frozen 28-column schema")
    values = vector.iloc[0].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [n for n in FEATURE_COLUMNS if not np.isfinite(float(vector.iloc[0][n]))]
        raise PulseError(f"non-finite features at decision bar: {bad}")
    kwargs: dict[str, Any] = {}
    best = getattr(booster, "best_iteration", None)
    if best:
        kwargs["num_iteration"] = int(best)
    score = float(np.asarray(booster.predict(vector[FEATURE_COLUMNS], **kwargs), dtype=float)[0])
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise PulseError(f"L2 produced an invalid probability: {score}")
    return score


def gap_filter(candidates: Sequence[Candidate], *, gap_bars: int,
               seen: Mapping[str, int]) -> list[Candidate]:
    """Keep the frozen 18-bar spacing, counting bars already claimed in the ledger."""
    kept: list[Candidate] = []
    last = dict(seen)
    for candidate in sorted(candidates, key=lambda c: (c.symbol, c.decision_i)):
        previous = last.get(candidate.symbol)
        if previous is not None and candidate.decision_i - previous < gap_bars:
            continue
        kept.append(candidate)
        last[candidate.symbol] = candidate.decision_i
    return kept


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_ledger(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_open_rows(rows: Sequence[dict[str, Any]], frames: Mapping[str, pd.DataFrame],
                      *, barrier: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Fill exits for rows whose barrier has since resolved. No retraining, no orders."""
    updated: list[dict[str, Any]] = []
    filled = 0
    for row in rows:
        out = dict(row)
        if out.get("outcome") not in (None, "", "open"):
            updated.append(out)
            continue
        frame = frames.get(str(out["symbol"]))
        if frame is None:
            updated.append(out)
            continue
        times = pd.to_datetime(frame["open_time"], utc=True)
        entry_time = pd.Timestamp(out["entry_time"])
        matches = np.flatnonzero((times == entry_time).to_numpy())
        if len(matches) != 1:
            updated.append(out)
            continue
        entry_i = int(matches[0])
        # The signal was emitted before the entry bar existed, so entry_price was a
        # projection off the decision close. The frozen contract says next_open, and
        # the ledger has to settle on the real one or the 100-trade result is scored
        # against a price that was never available.
        if out.get("entry_price_is_projection"):
            out["entry_price_projected"] = float(out["entry_price"])
            out["entry_price"] = float(frame["open"].iloc[entry_i])
            out["entry_price_is_projection"] = False
        resolution = resolve_barrier_outcome(
            frame, side="short", entry_i=entry_i,
            entry_price=float(out["entry_price"]), atr=float(out["atr"]),
            tp_atr_mult=float(barrier["tp_atr"]), sl_atr_mult=float(barrier["sl_atr"]),
            horizon_bars=int(barrier["horizon_bars"]),
            same_bar_policy=str(barrier["same_bar_policy"]),
            gap_policy=str(barrier["gap_policy"]),
            return_convention=str(barrier["return_convention"]),
            allow_partial=True, bar_duration=BAR,
        )
        # `status`, not `outcome`: an unresolved partial path comes back as
        # status="open" with outcome="" and every price field None.
        if resolution.status != "closed":
            updated.append(out)
            continue
        out["outcome"] = str(resolution.outcome)
        out["label"] = resolution.label
        out["exit_price"] = float(resolution.exit_price)
        out["exit_time"] = str(resolution.exit_time)
        out["exit_offset"] = int(resolution.exit_offset)
        out["gross_return"] = float(resolution.gross_ret)
        filled += 1
        updated.append(out)
    return updated, filled


def load_universe(cache_dir: Path, *, exclude_stockish: bool) -> list[tuple[str, list[Path]]]:
    series = list_series(cache_dir, bar="15m")
    out = []
    for (_source, symbol), paths in series.items():
        if not symbol.endswith("_USDT_SWAP"):
            continue
        if exclude_stockish and is_stockish(symbol):
            continue
        out.append((symbol, paths))
    return sorted(out)


def build_frame(paths: Sequence[Path]) -> pd.DataFrame:
    return add_indicators(add_mas(load_series(list(paths))))


def _format_caption(candidate: Candidate, config: Mapping[str, Any], age_min: float) -> str:
    barrier = config["barrier"]
    tp = candidate.entry_projection * (1 - float(barrier["tp_atr"]) * candidate.atr_pct)
    sl = candidate.entry_projection * (1 + float(barrier["sl_atr"]) * candidate.atr_pct)
    return (
        f"🔻 {candidate.symbol}  做空 PAPER\n"
        f"决策 bar {candidate.decision_time:%m-%d %H:%M} UTC ({age_min:.0f} 分钟前)\n"
        f"L1 p(SIGNAL) {candidate.p_signal:.3f} / 门 {config['l1']['threshold']}\n"
        f"L2 score {candidate.l2_score:.3f} / 门 {config['l2']['threshold']}\n"
        f"预计入场 {candidate.entry_projection:.6g} (下根开盘)\n"
        f"TP {tp:.6g}  SL {sl:.6g}  ATR {candidate.atr_pct * 100:.2f}%\n"
        f"障碍 TP{barrier['tp_atr']}×/SL{barrier['sl_atr']}×/{barrier['horizon_bars']} 根\n"
        f"⚠️ 纸面信号,未下单,未 promote"
    )


# Same candidate list as scripts/live_signal_tg.py: a chart that renders tofu is
# worse than an English one, because tofu still looks like a label.
_CJK_FONTS = ("PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti", "Songti SC",
              "Arial Unicode MS", "Noto Sans CJK SC", "Noto Sans CJK JP",
              "Source Han Sans SC", "WenQuanYi Zen Hei", "Droid Sans Fallback")


def _use_cjk_font(plt) -> bool:
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def draw_signal(frame: pd.DataFrame, candidate: Candidate, config: Mapping[str, Any],
                out_path: Path) -> None:
    """Context chart with the scored W10 window marked and the barrier projected.

    The classifier only ever saw the 10 shaded bars; the surrounding 110 are for
    the reader, and are drawn in a lighter weight so the two are not confused.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    zh = _use_cjk_font(plt)

    def L(chinese: str, english: str) -> str:
        return chinese if zh else english

    decision_i = candidate.decision_i
    low_i = max(0, decision_i - 110)
    segment = frame.iloc[low_i : decision_i + 1]
    x = mdates.date2num(pd.to_datetime(segment["open_time"], utc=True).dt.tz_localize(None))
    o, h, l, c = (segment[k].astype(float).to_numpy() for k in ("open", "high", "low", "close"))

    barrier = config["barrier"]
    entry = candidate.entry_projection
    tp = entry * (1 - float(barrier["tp_atr"]) * candidate.atr_pct)
    sl = entry * (1 + float(barrier["sl_atr"]) * candidate.atr_pct)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    width = (x[1] - x[0]) * 0.7 if len(x) > 1 else 0.005
    up = c >= o
    ax.vlines(x, l, h, color="#888", linewidth=0.8, zorder=2)
    ax.bar(x[up], (c - o)[up], width, bottom=o[up], color="#26a69a", zorder=3)
    ax.bar(x[~up], (o - c)[~up], width, bottom=c[~up], color="#ef5350", zorder=3)

    for name in ALL_MA_COLS:
        if name in segment:
            ax.plot(x, segment[name].astype(float).to_numpy(), lw=0.7, alpha=0.55)

    window_x0 = x[max(0, len(x) - WINDOW_BARS)]
    ax.axvspan(window_x0, x[-1], color="#42a5f5", alpha=0.16, zorder=1,
               label=L(f"W10 模型窗口 ({WINDOW_BARS} 根)", f"W10 model window ({WINDOW_BARS} bars)"))
    ax.axhline(entry, color="#1e88e5", lw=1.2, ls="--",
               label=L(f"入场投影 {entry:.6g}", f"entry projection {entry:.6g}"))
    ax.axhline(tp, color="#2e7d32", lw=1.2, ls="--", label=f"TP {tp:.6g}")
    ax.axhline(sl, color="#c62828", lw=1.2, ls="--", label=f"SL {sl:.6g}")

    ax.set_title(
        f"{candidate.symbol}  {L('做空', 'SHORT')} PAPER   "
        f"L1 {candidate.p_signal:.3f}  L2 {candidate.l2_score:.3f}   "
        f"ATR {candidate.atr_pct * 100:.2f}%\n"
        f"{L('决策 bar', 'decision bar')} {candidate.decision_time:%Y-%m-%d %H:%M} UTC   "
        f"TP{barrier['tp_atr']}× / SL{barrier['sl_atr']}× / {barrier['horizon_bars']} "
        f"{L('根', 'bars')}   "
        f"{L('未下单 · 未 promote', 'no order placed, not promoted')}",
        fontsize=11, loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)
    ax.set_ylabel(L("价格", "Price"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def scan(config: Mapping[str, Any], *, cache_dir: Path, limit: int,
         verbose: bool = True) -> tuple[list[Candidate], dict[str, Any], dict[str, pd.DataFrame]]:
    """One pulse of discovery: L1 classify, then L2 gate. Returns survivors."""
    from tools.backtest_fixed_w10_cls_preholdout import (
        _full_frame_transform, _load_classifier, _pick_device, classify_frame,
    )
    import lightgbm as lgb

    fable = Path(config.get("fable_root", "/Users/zhangzc/fable-trading"))
    weights = fable / config["l1"]["weights"]
    declared = str(config["l1"].get("weights_sha256", "")).lower()
    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    if declared and declared != actual:
        raise PulseError(f"L1 weights sha256 mismatch: config={declared} actual={actual}")

    device = _pick_device(config["l1"].get("device"))
    model, names, signal_idx = _load_classifier(weights, device)
    transform = _full_frame_transform()
    booster = lgb.Booster(model_file=str(fable / config["l2"]["model"]))
    if list(booster.feature_name()) != list(FEATURE_COLUMNS):
        raise PulseError("L2 model feature names disagree with the frozen 28-column schema")

    universe = load_universe(cache_dir, exclude_stockish=bool(config["live"]["exclude_stockish"]))
    if limit:
        universe = universe[:limit]
    now = _utc_now()
    stats = {
        "protocol": PROTOCOL, "pulse_at": now.isoformat(), "device": device,
        "l1_classes": names, "symbols_total": len(universe), "symbols_scanned": 0,
        "symbols_stale": 0, "symbols_failed": 0, "windows_scored": 0,
        "l1_passed": 0, "atr_rejected": 0, "l2_passed": 0,
        "weights_sha256": actual,
    }
    survivors: list[Candidate] = []
    frames: dict[str, pd.DataFrame] = {}
    started = time.monotonic()

    for symbol, paths in universe:
        try:
            frame = build_frame(paths)
            age = check_freshness(
                frame, now=now, max_minutes=float(config["live"]["freshness_max_minutes"]),
                symbol=symbol,
            )
        except PulseError:
            stats["symbols_stale"] += 1
            continue
        except Exception:  # a single unreadable series must not end the pulse
            stats["symbols_failed"] += 1
            continue
        frames[symbol] = frame
        indices = live_decision_indices(
            frame, max_tip_age_bars=int(config["live"]["max_tip_age_bars"]),
        )
        if not indices:
            stats["symbols_scanned"] += 1
            continue
        rows, _ = classify_frame(
            model, transform, frame, symbol, indices, device=device,
            signal_idx=signal_idx, threshold=float(config["l1"]["threshold"]),
            batch_size=int(config["l1"].get("batch_size", 8)), render_workers=0,
        )
        stats["symbols_scanned"] += 1
        stats["windows_scored"] += len(indices)
        for row in rows:
            if float(row["p_signal"]) < float(config["l1"]["threshold"]):
                continue
            stats["l1_passed"] += 1
            decision_i = int(row["decision_i"])
            ok, atr, atr_pct = atr_eligible(frame, decision_i)
            if not ok:
                stats["atr_rejected"] += 1
                continue
            l2 = score_l2(booster, frame, decision_i,
                          feature_semantics=str(config["l2"]["feature_semantics"]))
            if l2 < float(config["l2"]["threshold"]):
                continue
            stats["l2_passed"] += 1
            survivors.append(Candidate(
                symbol=symbol, decision_i=decision_i,
                decision_time=pd.Timestamp(frame["open_time"].iloc[decision_i]),
                p_signal=float(row["p_signal"]), l2_score=l2, atr=atr, atr_pct=atr_pct,
                entry_projection=float(frame["close"].iloc[decision_i]),
            ))
        if verbose and stats["symbols_scanned"] % 50 == 0:
            print(f"  ... {stats['symbols_scanned']}/{len(universe)} symbols "
                  f"({time.monotonic() - started:.0f}s)", flush=True)

    stats["age_gate_minutes"] = float(config["live"]["freshness_max_minutes"])
    stats["scan_wall_seconds"] = round(time.monotonic() - started, 1)
    return survivors, stats, frames


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/Users/zhangzc/fable-trading/data/kline_fetched"))
    parser.add_argument("--ledger", type=Path,
                        default=Path("/Users/zhangzc/fable-trading/analysis/output/"
                                     "live_w10_pulse_v1/ledger.jsonl"))
    parser.add_argument("--charts", type=Path,
                        default=Path("/Users/zhangzc/fable-trading/analysis/output/"
                                     "live_w10_pulse_v1/charts"))
    parser.add_argument("--limit", type=int, default=0, help="debug: only first N symbols")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--send", action="store_true", help="push signals to Telegram")
    group.add_argument("--dry-run", action="store_true", help="render only, never send")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(f"=== {PROTOCOL} pulse {_utc_now().isoformat()} ===", flush=True)
    print(f"L1 门 {config['l1']['threshold']}  L2 门 {config['l2']['threshold']}  "
          f"新鲜度门 {config['live']['freshness_max_minutes']} min  "
          f"{'SEND' if args.send else 'DRY-RUN'}", flush=True)

    survivors, stats, frames = scan(config, cache_dir=args.cache_dir, limit=args.limit)

    ledger = read_ledger(args.ledger)
    resolved, filled = resolve_open_rows(ledger, frames, barrier=config["barrier"])
    # Settling a projected entry price changes a row without filling it, and that
    # correction has to survive the pulse or the next one re-reads the projection.
    if resolved != ledger:
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        with args.ledger.open("w", encoding="utf-8") as handle:
            for row in resolved:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    known = {str(row["event_id"]) for row in resolved}
    last_seen: dict[str, int] = {}
    for row in resolved:
        symbol = str(row["symbol"])
        last_seen[symbol] = max(last_seen.get(symbol, -10**9), int(row["decision_i"]))
    fresh = [c for c in survivors if event_id(c.symbol, c.decision_time) not in known]
    fresh = gap_filter(fresh, gap_bars=int(config["barrier"]["min_gap_bars"]), seen=last_seen)
    fresh.sort(key=lambda c: c.l2_score, reverse=True)

    target = int(config["live"]["target_trades"])
    print(f"扫描 {stats['symbols_scanned']}/{stats['symbols_total']} symbol  "
          f"{stats['windows_scored']} 窗口  {stats['scan_wall_seconds']}s", flush=True)
    print(f"过期跳过 {stats['symbols_stale']}  读取失败 {stats['symbols_failed']}  "
          f"L1 过 {stats['l1_passed']}  ATR 拒 {stats['atr_rejected']}  "
          f"L2 过 {stats['l2_passed']}  去重后新信号 {len(fresh)}", flush=True)
    print(f"账本 {len(resolved)} 行 (本轮补平 {filled});目标 {target} 笔", flush=True)

    if len(resolved) >= target:
        print(f"已达 {target} 笔目标,不再开新信号。", flush=True)
        fresh = []

    room = max(0, target - len(resolved))
    cap = min(int(config["live"]["max_send_per_pulse"]), room)
    to_emit, new_rows = fresh[:cap], []
    now_iso = _utc_now().isoformat()
    for candidate in to_emit:
        frame = frames[candidate.symbol]
        chart = args.charts / (
            f"{candidate.symbol}_{candidate.decision_time:%Y%m%dT%H%M%SZ}.png"
        )
        draw_signal(frame, candidate, config, chart)
        age = (pd.Timestamp(now_iso) - candidate.decision_time).total_seconds() / 60.0
        caption = _format_caption(candidate, config, age)
        sent = False
        if args.send:
            from yoyo import notify
            sent = bool(notify.send_photo(chart, caption))
        print(f"  {'SENT' if sent else 'DRY '} {candidate.symbol} "
              f"L1={candidate.p_signal:.3f} L2={candidate.l2_score:.3f} "
              f"-> {chart.name}", flush=True)
        new_rows.append({
            "protocol": PROTOCOL,
            "event_id": event_id(candidate.symbol, candidate.decision_time),
            "symbol": candidate.symbol,
            "decision_i": candidate.decision_i,
            "decision_time": candidate.decision_time.isoformat(),
            "entry_time": (candidate.decision_time + BAR).isoformat(),
            "entry_price": candidate.entry_projection,
            "entry_price_is_projection": True,
            "p_signal": candidate.p_signal,
            "l2_score": candidate.l2_score,
            "atr": candidate.atr,
            "atr_pct": candidate.atr_pct,
            "atr_pct_min": float(ATR_PCT_MIN),
            "side": "short",
            "outcome": "open",
            "chart": str(chart),
            "telegram_sent": sent,
            "logged_at": now_iso,
            **{f"safety_{k}": v for k, v in SAFETY.items()},
        })
    if new_rows and args.send:
        append_ledger(args.ledger, new_rows)
        print(f"账本 +{len(new_rows)} 行 -> {args.ledger}", flush=True)
    elif new_rows:
        print(f"DRY-RUN: 未写账本 ({len(new_rows)} 行本可入账)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
