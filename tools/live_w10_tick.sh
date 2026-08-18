#!/usr/bin/env bash
# One live W10 tick: refresh tips, then score them. PAPER ONLY, places no orders.
#
# Refresh and scan must be adjacent. The freshness gate is 30 minutes against 15m
# bars, so a scan run even one bar after its refresh starts rejecting the whole
# universe as stale -- observed 2026-08-14: a pulse 30 min behind its refresh
# skipped 341 of 344 symbols. This is the gate working, not a bug, but it means
# the two steps belong in one tick rather than two schedules.
#
# Install (15 min cadence):
#   */15 * * * * /Users/zhangzc/yoyo-trading/tools/live_w10_tick.sh --send
set -uo pipefail

YOYO="$(cd "$(dirname "$0")/.." && pwd)"
FABLE="${FABLE_ROOT:-/Users/zhangzc/fable-trading}"
MODE="${1:---dry-run}"

PY="$FABLE/.venv/bin/python"
[ -x "$PY" ] || PY=python3

# yoyo.notify reads data/tg_config.json relative to the data root, and this repo
# holds code only -- the credentials live in the fable-trading tree.
export YOYO_DATA_ROOT="$FABLE"

LOG_DIR="$FABLE/analysis/output/live_w10_pulse_v1"
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/tick.log" 2>&1
echo "=== tick $(date -u +%Y-%m-%dT%H:%M:%SZ) mode=$MODE ==="

# Step 1: pull confirmed tip bars. workers=2 -- OKX returned HTTP 429 for 47 of
# 401 symbols at workers=6 on 2026-08-14.
( cd "$FABLE" && PYTHONPATH=. "$PY" scripts/refresh_kline_tip.py --workers 2 ) \
  || echo "refresh returned $? (continuing; the freshness gate is the real guard)"

# Step 2: score tip / tip-1 / tip-2 and emit.
cd "$YOYO" && PYTHONPATH=. "$PY" -m yoyo.cli.forward_pulse_w10 \
  --config configs/live_w10_pulse_v1.json "$MODE"
echo "=== tick done rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
