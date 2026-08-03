"""Poll forward_log → place market + TP/SL bracket.

Hard rules:
- OkxDemoClient reads environment from keys file (demo|live).
- Kill switch file blocks new entries.
- Circuit breaker: consecutive closed losses pause new entries.
- Invalid TP/SL refuses entry (never leave a naked position).
"""
from __future__ import annotations

import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from yoyo.layers.l4_execution.config import (
    SL_ATR_MULT,
    TP_ATR_MULT,
    ExecutorConfig,
    kill_switch_active,
)
from yoyo.layers.l4_execution import ledger as led
from yoyo.layers.l4_execution.okx_client import OkxDemoClient, OkxDemoError
from yoyo.layers.l4_execution.symbols import round_price, size_for_notional, to_okx_inst_id
from yoyo.contracts.forward_log import actionable_rows, read_forward_log
from yoyo.contracts.forward_log import LEGACY_PROTOCOL
from yoyo.contracts.protocol import StrategyProtocol, require_active_bundle
from yoyo.contracts.paths import data_path, data_root  # noqa: F401


def signal_key(row: pd.Series) -> str:
    """Idempotency key for one signal event.

    Intentionally **excludes score and model hash**: re-scoring one protocol
    event must not open a second position. Side and protocol stay in the key so
    a legacy long row can never absorb a repaired short event (P0 B-01..B-04).
    """
    side = signal_trade_side(row)
    raw_protocol = row.get("protocol_version")
    protocol = (
        LEGACY_PROTOCOL
        if raw_protocol is None or pd.isna(raw_protocol) or not str(raw_protocol).strip()
        else str(raw_protocol).strip()
    )
    return "|".join(
        (
            str(row.get("source", "okx")),
            str(row.get("symbol")),
            str(row.get("signal_time")),
            side,
            protocol,
        )
    )


_NOTIFY_EVENTS = {
    "order_placed": "🟢 <b>实盘开仓</b>",
    "order_partial": "🟡 <b>开仓成功·括号失败</b>(需人工补止损!)",
    "order_failed": "🔴 <b>下单失败</b>",
    "skipped_invalid_barriers": "⚠️ <b>拒单</b>(止盈止损价不可用)",
    "skipped_unsupported_side": "⚠️ <b>拒单</b>(执行器不支持该方向)",
    "timeout_close": "⏱ <b>超时平仓</b>(72bar 到期,按验证策略出场)",
    "timeout_close_failed": "🔴 <b>超时平仓失败</b>(需人工处理!)",
}


def _notify_event(ev: dict) -> None:
    """Push trade events to Telegram. Fire-and-forget: the trading loop must
    never stall or die because a notification did."""
    label = _NOTIFY_EVENTS.get(str(ev.get("event")))
    if label is None:
        return
    try:
        from yoyo.notify import send

        parts = [label, f"品种: <b>{ev.get('inst_id') or ev.get('symbol')}</b>"]
        if ev.get("mark_px"):
            parts.append(f"价格: {ev['mark_px']}")
        if ev.get("tp_px") and ev.get("sl_px"):
            parts.append(f"止盈 {ev['tp_px']} / 止损 {ev['sl_px']}")
        if ev.get("sz"):
            parts.append(f"数量: {ev['sz']}  名义: {ev.get('notional_usdt', '?')}U")
        if ev.get("error"):
            parts.append(f"错误: {str(ev['error'])[:160]}")
        if ev.get("note"):
            parts.append(str(ev["note"])[:160])
        send("\n".join(parts))
    except Exception as exc:  # noqa: BLE001
        print(f"executor notify failed: {exc}")


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return data_root() / p


def load_actionable_signals(
    cfg: ExecutorConfig,
    protocol: StrategyProtocol,
) -> pd.DataFrame:
    """Return rows from exactly one execution-eligible protocol.

    Side mismatch is intentionally left for ``open_one`` so the refusal is
    recorded once in the ledger. All other provenance mismatches disappear from
    the order queue before a client exists.
    """
    path = _resolve(cfg.forward_log)
    if not path.exists():
        return pd.DataFrame()
    if not protocol.execution_eligible:
        return pd.DataFrame()
    df = actionable_rows(read_forward_log(path))
    if df.empty:
        return df
    for column, expected in (
        ("protocol_version", protocol.protocol_version),
        ("strategy_id", protocol.strategy_id),
        ("feature_semantics", protocol.feature_semantics),
        ("model_sha256", protocol.model_sha256),
        ("detector_sha256", protocol.detector_sha256),
        ("dataset_sha256", protocol.dataset_sha256),
    ):
        df = df[df[column].astype(str) == str(expected)]
    if "status" in df.columns:
        df = df[df["status"].astype(str).isin(cfg.open_statuses)]
    if cfg.require_score_ge_threshold:
        score = pd.to_numeric(df["score"], errors="coerce")
        row_threshold = pd.to_numeric(df["threshold"], errors="coerce")
        # CSV decimal round-trips may move a float by one ULP. Permit only that
        # representation noise, not a semantically different gate.
        threshold_tolerance = max(1e-15, abs(float(protocol.threshold)) * 1e-12)
        exact_threshold = (row_threshold - float(protocol.threshold)).abs() <= threshold_tolerance
        passes = score.map(
            lambda value: protocol.passes_threshold(float(value)) if pd.notna(value) else False
        )
        df = df[exact_threshold & passes]
    # Freshness gate: a signal stays status=open until its barrier resolves (up
    # to 18h), but the EDGE is the launch moment -- entering hours late is a
    # different, untested trade. Only rows younger than max_signal_age_min may
    # open positions (the backtest enters at the very next bar).
    if "signal_time" in df.columns:
        age_cap = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=cfg.max_signal_age_min)
        ts = pd.to_datetime(df["signal_time"], errors="coerce", utc=True)
        df = df[ts >= age_cap]
        df = df.sort_values("signal_time")
    return df.reset_index(drop=True)


# Owner-approved tier cap (2026-07-20, analysis/p_weight_centric_val.md):
# q90-q95 / q95-q99 / q99+ → 1x / 1.5x / 2x. The cap guards against a corrupt
# forward_log ever inflating risk past the approved maximum.
TIER_SIZE_MULT_CAP = 2.0


def signal_size_mult(row: pd.Series) -> float:
    """Tiered sizing multiplier from the forward-log row.

    Legacy rows (pre-tier log, missing column or NaN) trade the historic 1x.
    A stamped 0.0 (below-threshold, should never be logged) yields notional 0,
    which the min_notional gate then skips — corrupt data can only shrink
    exposure, never inflate it beyond TIER_SIZE_MULT_CAP.
    """
    raw = row.get("size_mult")
    try:
        mult = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(mult):
        return 1.0
    return min(max(mult, 0.0), TIER_SIZE_MULT_CAP)


MISSING_SIDE = "missing"


def signal_trade_side(row: pd.Series) -> str:
    """Normalized strategy direction. A missing side is NOT long.

    The current execution implementation is long-only. Explicit ``short``
    (mainline v10 ledger after P0 side fix) must remain visible so ``open_one``
    rejects it — never silently convert a short model signal into a buy.
    Short market execution needs a separate owner-approved path.

    P0 fix 2026-08-03: absent/blank/NaN side used to resolve to "long", which is
    the one default that can turn an unlabelled row into a real buy. The mainline
    is short, so the row most likely to arrive without a side is a short one. It
    now resolves to ``missing`` and open_one refuses it, per the takeover plan's
    P0-02. Legacy analysis that genuinely wants the old reading must ask for it
    explicitly rather than inherit it from the production path.
    """
    raw = row.get("side")
    if raw is None or pd.isna(raw):
        return MISSING_SIDE
    side = str(raw).strip().lower()
    return side or MISSING_SIDE


def barriers(entry: float, atr_pct: float) -> tuple[float, float]:
    try:
        atr = abs(entry * float(atr_pct))
    except (TypeError, ValueError):
        atr = float("nan")
    # `atr <= 0` misses NaN (all NaN comparisons are False): a forward row with
    # atr_pct=None sailed through here on 2026-07-16, produced tp=sl=NaN -> 0.0
    # after tick rounding, OKX rejected the bracket (51250) and a REAL DOGE long
    # sat naked. `not (atr > 0)` is True for NaN, zero, and negatives alike.
    if not (atr > 0) or not math.isfinite(atr):
        atr = entry * 0.01  # 1% proxy so the position is never unprotected
    tp = entry + TP_ATR_MULT * atr
    sl = entry - SL_ATR_MULT * atr
    return tp, sl


def compute_entry_notional(
    client: OkxDemoClient | None,
    cfg: ExecutorConfig,
    *,
    open_n: int,
    open_notional: float = 0.0,
) -> dict[str, Any]:
    """How much USDT notional to open for the next slot.

    equity_times_leverage: remaining_budget / slots_left
      remaining = equity * leverage - open_notional
    fixed: cfg.notional_usdt
    """
    mode = (cfg.sizing_mode or "fixed").strip().lower()
    out: dict[str, Any] = {
        "sizing_mode": mode,
        "leverage": cfg.leverage,
        "open_n": open_n,
        "open_notional": open_notional,
    }
    if mode in {"equity_times_leverage", "equity_x_leverage", "equity_leverage"}:
        if client is None:
            out["notional_usdt"] = float(cfg.notional_usdt)
            out["note"] = "no client — fell back to fixed notional_usdt"
            return out
        equity = client.usdt_equity()
        target = max(0.0, float(equity) * float(cfg.leverage))
        remaining = max(0.0, target - max(0.0, float(open_notional)))
        slots_left = max(1, int(cfg.max_concurrent) - int(open_n))
        notional = remaining / slots_left
        out.update({
            "equity_usdt": equity,
            "target_gross_usdt": target,
            "remaining_budget_usdt": remaining,
            "slots_left": slots_left,
            "notional_usdt": notional,
        })
        return out
    out["notional_usdt"] = float(cfg.notional_usdt)
    return out


def open_one(
    client: OkxDemoClient | None,
    cfg: ExecutorConfig,
    row: pd.Series,
    *,
    dry_run: bool,
    notional_usdt: float | None = None,
    sizing_meta: dict[str, Any] | None = None,
    protocol: StrategyProtocol | None = None,
) -> dict[str, Any]:
    """Place one long paper trade (+ OCO). Returns ledger event dict."""
    sk = signal_key(row)
    symbol = str(row["symbol"])
    inst_id = to_okx_inst_id(symbol)
    trade_side = signal_trade_side(row)
    atr_pct = float(row["atr_pct"]) if pd.notna(row.get("atr_pct")) else 0.01
    notional = float(notional_usdt if notional_usdt is not None else cfg.notional_usdt)
    event: dict[str, Any] = {
        "event": "dry_run" if dry_run else "order_placed",
        "signal_key": sk,
        "symbol": symbol,
        "inst_id": inst_id,
        "score": row.get("score"),
        "threshold": row.get("threshold"),
        "signal_side": trade_side,
        # Populated only after protocol + side guards. A rejected row must not
        # even be represented as an intended buy in the audit event.
        "side": None,
        "tp_atr_mult": TP_ATR_MULT,
        "sl_atr_mult": SL_ATR_MULT,
        "td_mode": cfg.td_mode,
        "signal_time": row.get("signal_time"),
        "candidate_detected_at": row.get("candidate_detected_at", row.get("detected_at")),
        "decision_at": row.get("decision_at"),
        "entry_requested_at": None,
        # Order acceptance is not fill evidence. A separate broker-ledger
        # reconciliation must populate these fields from actual fills.
        "fill_source": None,
        "fill_at": None,
        "fill_px": None,
    }
    if sizing_meta:
        event["sizing"] = sizing_meta

    if protocol is None:
        event["event"] = "skipped_protocol_mismatch"
        event["note"] = "missing verified strategy protocol"
        return event
    if not protocol.execution_eligible:
        event["event"] = "skipped_ineligible_protocol"
        event["note"] = f"protocol {protocol.protocol_version!r} is execution-ineligible"
        return event
    if not protocol.accepts_row_side(row.get("side")):
        event["event"] = "skipped_protocol_mismatch"
        event["note"] = (
            f"row side={trade_side!r} does not match protocol side={protocol.side!r}"
        )
        return event

    if trade_side != "long":
        event["event"] = "skipped_unsupported_side"
        event["note"] = (
            f"current executor is long-only; refused signal side={trade_side!r}"
        )
        return event

    event["side"] = "buy"

    if dry_run or client is None:
        # Estimate without keys when dry-run and no client
        mark = float(row["entry_price"]) if pd.notna(row.get("entry_price")) else None
        event.update({
            "mark_px": mark,
            "notional_usdt": notional,
            "note": "dry-run: no order sent",
        })
        if mark:
            tp, sl = barriers(mark, atr_pct)
            event["tp_px"], event["sl_px"] = tp, sl
        return event

    if notional < float(cfg.min_notional_usdt):
        event["event"] = "skipped"
        event["note"] = (
            f"notional {notional:.4f} < min_notional_usdt {cfg.min_notional_usdt}"
        )
        return event

    inst = client.instrument(inst_id)
    mark = client.mark_px(inst_id)
    sz = size_for_notional(notional, mark, inst)
    tick = inst.get("tickSz") or "0.01"
    tp_raw, sl_raw = barriers(mark, atr_pct)
    tp_px = round_price(tp_raw, tick)
    sl_px = round_price(sl_raw, tick)
    event.update({
        "mark_px": mark,
        "sz": sz,
        "tp_px": tp_px,
        "sl_px": sl_px,
        "notional_usdt": notional,
        "leverage": cfg.leverage,
    })

    # The bracket IS the risk control: if these numbers are unusable, there is
    # nothing safe to place afterwards, so refuse the ENTRY -- do not discover
    # the problem with a live position already open (2026-07-16 DOGE incident).
    if not (math.isfinite(tp_px) and math.isfinite(sl_px) and 0 < sl_px < mark < tp_px):
        event["event"] = "skipped_invalid_barriers"
        event["note"] = f"tp/sl unusable: tp={tp_px} sl={sl_px} mark={mark}"
        return event

    try:
        client.set_leverage(inst_id, str(cfg.leverage), mgn_mode=cfg.td_mode)
    except OkxDemoError as exc:
        # leverage may already be set; log and continue
        event["leverage_warn"] = str(exc)

    # Account may be net_mode or long_short_mode (hedge).
    mode = client.pos_mode()
    pos_side = "long" if mode == "long_short_mode" else "net"
    event["pos_mode"] = mode
    event["pos_side"] = pos_side

    cl_id = f"f{abs(hash(sk)) % 10**10}"
    event["entry_requested_at"] = datetime.now(timezone.utc).isoformat()
    order = client.place_market(
        inst_id, "buy", sz, td_mode=cfg.td_mode, cl_ord_id=cl_id, pos_side=pos_side
    )
    event["order_resp"] = order.get("data")
    # closing side for long = sell; same posSide in hedge mode
    # Retry bracket: a transient OKX 5xx after fill must not leave us naked.
    retries = max(0, int(getattr(cfg, "bracket_retries", 2)))
    sleep_s = float(getattr(cfg, "bracket_retry_sleep_sec", 1.5))
    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            algo = client.place_bracket(
                inst_id, "sell", sz, tp_px, sl_px, td_mode=cfg.td_mode, pos_side=pos_side
            )
            event["algo_resp"] = algo.get("data")
            event["bracket_attempts"] = attempt + 1
            last_err = None
            break
        except OkxDemoError as exc:
            last_err = str(exc)
            event["algo_error"] = last_err
            if attempt < retries:
                time.sleep(max(0.2, sleep_s))
    if last_err is not None:
        event["event"] = "order_partial"  # entry ok, bracket failed — owner must watch
        event["bracket_attempts"] = retries + 1
    return event


def enforce_timeout_exits(client, cfg: ExecutorConfig, ledger_path: Path) -> int:
    """Close positions older than the validated 72-bar horizon (18h).

    The strategy every backtest and the forward gate validated has exactly three
    exits: TP, SL, or timeout at 72 bars. Live had only the OCO bracket, so a
    position that touched neither barrier would linger indefinitely -- an
    untested trade. Closing is reduce-only, and the bracket algo is cancelled
    FIRST: a leftover OCO on a flat position would otherwise fire later and
    open a naked short.
    """
    import src.execution.ledger as led

    timeout = pd.Timedelta(hours=float(getattr(cfg, "timeout_hours", 18.0)))
    now = pd.Timestamp.now(tz="UTC")
    rows = led.load_all(ledger_path)
    # last entry event + algo id per instrument, minus anything already closed
    entries: dict[str, dict] = {}
    for r in rows:
        inst = r.get("inst_id")
        if not inst:
            continue
        ev = r.get("event")
        if ev in {"order_placed", "order_partial"}:
            algo_id = None
            for a in r.get("algo_resp") or []:
                algo_id = a.get("algoId") or algo_id
            entries[inst] = {"ts": r.get("ts"), "algo_id": algo_id}
        elif ev == "timeout_close":
            entries.pop(inst, None)
    closed = 0
    try:
        positions = client.positions()
    except Exception as exc:  # noqa: BLE001 -- position read failing must not kill the loop
        print(f"timeout check: positions read failed: {exc}")
        return 0
    for p in positions:
        try:
            pos_sz = abs(float(p.get("pos") or 0))
            if pos_sz <= 0:
                continue
            inst = p.get("instId")
            meta = entries.get(inst) or {}
            # Entry time: ledger first, OKX cTime fallback, explicit NaT checks.
            # pd.Timestamp(None) returns NaT WITHOUT raising, and `now - NaT <
            # timeout` is False -- the unit test caught this closing every
            # position that lacked a ledger row. Unknown age => never close.
            entry_ts = pd.NaT
            if meta.get("ts"):
                entry_ts = pd.Timestamp(meta["ts"])
                entry_ts = (entry_ts.tz_localize("UTC") if entry_ts.tzinfo is None
                            else entry_ts.tz_convert("UTC"))
            if pd.isna(entry_ts):
                ctime = int(p.get("cTime") or 0)
                if ctime <= 0:
                    continue
                entry_ts = pd.Timestamp(ctime, unit="ms", tz="UTC")
            if pd.isna(entry_ts) or now - entry_ts < timeout:
                continue
            if meta.get("algo_id"):
                try:
                    client.cancel_algo(inst, meta["algo_id"])
                except Exception as exc:  # noqa: BLE001 -- may be gone already
                    print(f"timeout close {inst}: cancel_algo: {exc}")
            side = "sell" if str(p.get("posSide", "net")) != "short" else "buy"
            resp = client.place_market(
                inst, side, str(pos_sz), td_mode=cfg.td_mode,
                pos_side=(p.get("posSide") if p.get("posSide") in {"long", "short"} else None),
                reduce_only=True,
            )
            ev = {
                "event": "timeout_close", "inst_id": inst, "sz": str(pos_sz),
                "held_hours": round((now - entry_ts).total_seconds() / 3600, 1),
                "order_resp": resp.get("data"),
            }
            led.append(ledger_path, ev)
            _notify_event(ev)
            closed += 1
        except Exception as exc:  # noqa: BLE001 -- one bad position must not skip the rest
            ev = {"event": "timeout_close_failed", "inst_id": p.get("instId"), "error": str(exc)}
            led.append(ledger_path, ev)
            _notify_event(ev)
    return closed


def run_once(
    cfg: ExecutorConfig,
    *,
    dry_run: bool = False,
    protocol: StrategyProtocol | None = None,
) -> dict[str, Any]:
    """Single poll cycle. Returns summary counters."""
    # Resolve the exact contract before any trading client can be constructed.
    # Tests for legacy mechanics may inject a fully explicit fixture protocol;
    # the production CLI never does and therefore requires active_bundle.json.
    protocol = protocol if protocol is not None else require_active_bundle()
    ledger_path = _resolve(cfg.ledger)
    summary: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "opened": 0,
        "skipped": 0,
        "errors": 0,
        "paused": None,
    }

    if kill_switch_active(cfg):
        summary["paused"] = f"kill switch: {cfg.kill_switch_file}"
        # Do not append paused every 30–60s — it bloated the ledger to 300+ noise rows.
        return summary

    # Enforce the validated 72-bar exit BEFORE any early return: a position ages
    # past its horizon precisely on quiet cycles, when no-signal returns would
    # otherwise skip the check. Runs under circuit-breaker pause too (it reduces
    # exposure); only the explicit kill switch above silences everything.
    if not dry_run:
        try:
            n_to = enforce_timeout_exits(OkxDemoClient(), cfg, ledger_path)
            if n_to:
                summary["timeout_closed"] = n_to
        except Exception as exc:  # noqa: BLE001
            print(f"timeout enforcement failed: {exc}")

    losses = led.consecutive_losses(ledger_path)
    if losses >= cfg.max_consecutive_losses:
        summary["paused"] = f"circuit breaker: {losses} consecutive losses"
        return summary

    taken = led.signal_keys_already_taken(ledger_path)
    signals = load_actionable_signals(cfg, protocol)
    if signals.empty:
        summary["note"] = "no actionable rows in forward_log"
        return summary

    client: OkxDemoClient | None = None
    open_n = 0
    open_notional = 0.0
    if not dry_run:
        client = OkxDemoClient()
        try:
            positions = client.positions("SWAP")
            open_n = sum(1 for p in positions if abs(float(p.get("pos") or 0)) > 0)
            open_notional = client.open_swap_notional_usd()
        except OkxDemoError as exc:
            summary["errors"] += 1
            summary["error"] = str(exc)
            led.append(ledger_path, {"event": "error", "where": "positions", "error": str(exc)})
            return summary
    else:
        # dry-run: count opens from ledger order_placed without closed
        placed = {r["signal_key"] for r in led.load_all(ledger_path) if r.get("event") == "order_placed"}
        closed = {r["signal_key"] for r in led.load_all(ledger_path) if r.get("event") == "closed"}
        open_n = len(placed - closed)
        open_notional = float(cfg.notional_usdt) * open_n

    slots = max(0, cfg.max_concurrent - open_n)
    summary["open_n"] = open_n
    summary["open_notional_usd"] = open_notional
    summary["max_concurrent"] = cfg.max_concurrent
    if slots <= 0:
        summary["note"] = f"at max_concurrent={cfg.max_concurrent} (open={open_n})"
        return summary

    for _, row in signals.iterrows():
        if slots <= 0:
            break
        sk = signal_key(row)
        if sk in taken:
            continue
        try:
            sizing = compute_entry_notional(
                client, cfg, open_n=open_n, open_notional=open_notional
            )
            base_notional = float(sizing.get("notional_usdt") or cfg.notional_usdt)
            # Tiered sizing (owner 2026-07-20): per-signal multiplier stamped
            # by the forward pulse; legacy rows without the column trade 1x.
            # Headroom (owner deploy option ①, 2026-07-21): unit = full-slot
            # budget / TIER_SIZE_MULT_CAP so q99+ (2x) fills equity*leverage
            # and never trips OKX 51008. 1x trades at half budget.
            size_mult = signal_size_mult(row)
            unit_notional = base_notional / TIER_SIZE_MULT_CAP
            notional = unit_notional * size_mult
            sizing["tier"] = row.get("tier")
            sizing["size_mult"] = size_mult
            sizing["base_notional_usdt"] = base_notional
            sizing["unit_notional_usdt"] = unit_notional
            sizing["notional_usdt"] = notional
            sizing["tier_headroom"] = True
            summary["last_sizing"] = sizing
            ev = open_one(
                client, cfg, row, dry_run=dry_run,
                notional_usdt=notional, sizing_meta=sizing, protocol=protocol,
            )
            led.append(ledger_path, ev)
            _notify_event(ev)
            if ev.get("event") in {"order_placed", "order_partial", "dry_run"}:
                summary["opened"] += 1
                slots -= 1
                open_n += 1
                open_notional += notional
                taken.add(sk)
            else:
                summary["skipped"] += 1
        except Exception as exc:  # noqa: BLE001 — one bad symbol must not kill the loop
            summary["errors"] += 1
            fail_ev = {
                "event": "order_failed",
                "signal_key": sk,
                "symbol": row.get("symbol"),
                "error": str(exc),
                "trace": traceback.format_exc(limit=4),
            }
            led.append(ledger_path, fail_ev)
            _notify_event(fail_ev)
    return summary


def run_loop(cfg: ExecutorConfig, *, dry_run: bool = False, once: bool = False) -> None:
    while True:
        summary = run_once(cfg, dry_run=dry_run)
        print(json_dumps(summary), flush=True)
        if once:
            return
        time.sleep(max(5, int(cfg.poll_seconds)))


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, default=str)
