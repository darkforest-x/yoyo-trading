#!/bin/bash
# V5 §10 steps 9–10 leftover: same-contract canary for V3.1 then R3B.
# Home repo: yoyo-trading. No holdout, no promote.
set -euo pipefail
YOYO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$YOYO"
SCAN="$YOYO/tools/scan_preholdout_canary.sh"
CMP="$YOYO/tools/compare_preholdout_canary.py"
PY="${FABLE_REPO:-/Users/zhangzc/fable-trading}/.venv/bin/python"
LOGDIR="$YOYO/reports"
mkdir -p "$LOGDIR"

V31_W="$YOYO/runs/r3a_v31_spread_ft/weights/best.pt"
V31_OUT="$YOYO/runs/r3a_v31_spread_ft/canary_am_20260503"
R3B_W="$YOYO/runs/r3b_v3gold_cold/weights/best.pt"
R3B_OUT="$YOYO/runs/r3b_v3gold_cold/canary_am_20260503"
R3A_OUT="$YOYO/runs/r3a_v3gold_ft_r1/canary_am_20260503"
R1_OUT="/Users/zhangzc/fable-trading/analysis/output/owner_short_gold_center_preholdout_canary_20260503_v1/merged_hardneg"
BASE_OUT="/Users/zhangzc/fable-trading/analysis/output/owner_short_gold_center_preholdout_canary_20260503_v1/merged_baseline"

status() {
  printf '%s\n' "$1" | tee -a "$LOGDIR/v5_screen.log"
  printf '%s\n' "$1" > "$LOGDIR/v5_screen_status.txt"
}

if [[ ! -f "$V31_W" ]]; then
  status "FAIL missing V3.1 weights"
  exit 1
fi

if [[ ! -f "$V31_OUT/_SUCCESS.json" ]]; then
  status "V3.1 canary scanning"
  mkdir -p "$V31_OUT"
  bash "$SCAN" "$V31_W" "$V31_OUT" 2>&1 | tee "$V31_OUT/scan.log"
  [[ -f "$V31_OUT/_SUCCESS.json" ]] || { status "FAIL V3.1 canary"; exit 1; }
fi
status "V3.1 canary done"

if [[ -f "$R3B_W" && ! -f "$R3B_OUT/_SUCCESS.json" ]]; then
  status "R3B canary scanning"
  mkdir -p "$R3B_OUT"
  bash "$SCAN" "$R3B_W" "$R3B_OUT" 2>&1 | tee "$R3B_OUT/scan.log"
  [[ -f "$R3B_OUT/_SUCCESS.json" ]] || { status "FAIL R3B canary"; exit 1; }
fi

MODELS=(--model baseline_v1_ft "$BASE_OUT" --model R1 "$R1_OUT" --model R3A_V3 "$R3A_OUT" --model R3A_V31 "$V31_OUT")
if [[ -f "$R3B_OUT/_SUCCESS.json" ]]; then
  MODELS+=(--model R3B_V3 "$R3B_OUT")
fi
status "writing compare table"
"$PY" "$CMP" "${MODELS[@]}" --ref R1 --out "$YOYO/manifests/runs/canary_am_20260503_screen.json"
status "SCREEN_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
