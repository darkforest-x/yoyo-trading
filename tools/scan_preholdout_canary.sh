#!/bin/bash
# Frozen pre-holdout canary (V5 §8 / EVALUATION_PROTOCOL).
# Reuses fable-trading scanner; writes into yoyo-trading. Never holdout.
set -euo pipefail
YOYO="$(cd "$(dirname "$0")/.." && pwd)"
FABLE="${FABLE_REPO:-/Users/zhangzc/fable-trading}"
PY="${FABLE}/.venv/bin/python"

WEIGHTS="${1:?usage: scan_preholdout_canary.sh WEIGHTS.pt OUT_DIR}"
OUT="${2:?}"

SNAP="${FABLE}/analysis/output/owner_short_gold_center_preholdout_canary_20260503_v1/kline_snapshot"
mkdir -p "$OUT"
exec env PYTHONPATH="${FABLE}:${YOYO}" "$PY" -u \
  "${FABLE}/scripts/backtest_owner_short_gold_center_recent.py" scan \
  --snapshot-dir "$SNAP" \
  --out-dir "$OUT" \
  --weights "$WEIGHTS" \
  --hours 12 \
  --window-min 12 --window-max 19 \
  --conf 0.25 --iou 0.7 --imgsz 960 \
  --device mps \
  --evaluation-scope preholdout_postval_canary
