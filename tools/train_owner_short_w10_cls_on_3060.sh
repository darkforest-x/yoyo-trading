#!/bin/bash
# Ship and start the fixed-W10 owner-short classifier on the Windows RTX 3060.
#
# This launcher freezes the dataset/model/trainer inputs used for this one
# experiment, validates the classification tree against its manifest, and
# starts exactly one detached WMI job.  It never reads holdout data, retries a
# failed run, evaluates, promotes, or changes ACTIVE.
set -euo pipefail

FABLE="/Users/zhangzc/fable-trading"
HOST="${FABLE_3060_HOST:-zzc@192.168.1.4}"
REMOTE="C:/fable"
REMOTE_WIN='C:\fable'
LOCAL_PY="$FABLE/.venv/bin/python"
RPC="$FABLE/scripts/ssh_ps.sh"
DATASET="$FABLE/datasets/fixed_w10_gold500_short_ownerhn_v1"
MODEL="$FABLE/models/yolo11n-cls.pt"
TRAINER="$FABLE/scripts/train_fixed_w10_cls.py"
NAME="fixed_w10_gold500_short_ownerhn_v1_cls"
MODE="run"

# Frozen, independently-auditable inputs for this training arm.
EXPECTED_DATASET_META_SHA="3d2470fc323bde5641b4ec3243a28962b0c4cd429e1149c99186f9c7b84fee65"
EXPECTED_MANIFEST_SHA="9897f34ee9c27003cff5f8d6db8d26c04889229a555a04308402f2445af92e1e"
EXPECTED_MODEL_SHA="c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7"
EXPECTED_TRAINER_SHA="81eac7daea14af8717542dd8c0df21fe862f228a1f4f3c843c63c9ab82fe948b"
EXPECTED_IMAGES=820

SCP=(scp -o BatchMode=yes -o ConnectTimeout=15 -q)
TMP_TAR=""
TMP_CMD=""
cleanup() {
  set +e
  [[ -n "$TMP_TAR" && -f "$TMP_TAR" ]] && rm -f -- "$TMP_TAR"
  [[ -n "$TMP_CMD" && -f "$TMP_CMD" ]] && rm -f -- "$TMP_CMD"
}
trap cleanup EXIT

say() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
die() { printf '\033[1;31m[X] %s\033[0m\n' "$*" >&2; exit 1; }
usage() {
  cat <<'EOF'
Usage: bash tools/train_owner_short_w10_cls_on_3060.sh [--check|--status]

  run       validate frozen inputs, stage them, and start one detached job
  --check   validate local inputs plus remote CUDA/run availability; no training
  --status  show matching process, log tail, exit receipt, and best.pt

Environment:
  FABLE_3060_HOST  override the default zzc@192.168.1.4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) [[ "$MODE" == run ]] || die "choose one mode"; MODE="check"; shift ;;
    --status) [[ "$MODE" == run ]] || die "choose one mode"; MODE="status"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$HOST" =~ ^[^[:space:]@]+@[^[:space:]]+$ ]] || die "unsafe host: $HOST"

remote_ps() {
  local program
  program="$(cat)"
  [[ -n "$program" ]] || die "refusing to execute an empty remote PowerShell program"
  bash "$RPC" "$HOST" "$program"
}

sha256() { shasum -a 256 "$1" | awk '{print $1}'; }

show_status() {
  remote_ps <<PS | tr -d '\r'
\$ErrorActionPreference = 'Stop'
Write-Output '=== matching classifier processes ==='
\$p = @(Get-CimInstance Win32_Process | Where-Object {
  \$_.CommandLine -and \$_.CommandLine -like '*$NAME*'
})
if (\$p.Count) {
  \$p | Select-Object ProcessId,ParentProcessId,CreationDate,Name,CommandLine |
    Format-List | Out-String -Width 4096 | Write-Output
} else { Write-Output '(none)' }
Write-Output '=== log tail ==='
\$log = '$REMOTE/logs/$NAME.log'
if (Test-Path -LiteralPath \$log) { Get-Content -LiteralPath \$log -Tail 60 } else { Write-Output '(missing)' }
Write-Output '=== exit code ==='
\$exit = '$REMOTE/logs/$NAME.exit_code'
if (Test-Path -LiteralPath \$exit) { Get-Content -LiteralPath \$exit } else { Write-Output '(running or not started)' }
Write-Output '=== best.pt ==='
\$best = '$REMOTE/runs/classify/$NAME/weights/best.pt'
if (Test-Path -LiteralPath \$best) {
  Get-Item -LiteralPath \$best | Select-Object FullName,Length,LastWriteTime |
    Format-List | Out-String -Width 4096 | Write-Output
} else { Write-Output '(missing)' }
PS
}

if [[ "$MODE" == "status" ]]; then
  [[ -f "$RPC" ]] || die "missing Windows SSH RPC helper: $RPC"
  show_status
  exit 0
fi

say "local frozen-input gates"
[[ -x "$LOCAL_PY" ]] || die "missing local Python: $LOCAL_PY"
[[ -f "$RPC" ]] || die "missing Windows SSH RPC helper: $RPC"
[[ -d "$DATASET" ]] || die "missing dataset: $DATASET"
[[ -s "$MODEL" ]] || die "missing pretrained model: $MODEL"
[[ -s "$TRAINER" ]] || die "missing trainer: $TRAINER"
[[ "$(sha256 "$DATASET/dataset_manifest.json")" == "$EXPECTED_DATASET_META_SHA" ]] || die "dataset_manifest.json SHA drift"
[[ "$(sha256 "$DATASET/manifest.jsonl")" == "$EXPECTED_MANIFEST_SHA" ]] || die "manifest.jsonl SHA drift"
[[ "$(sha256 "$MODEL")" == "$EXPECTED_MODEL_SHA" ]] || die "yolo11n-cls.pt SHA drift"
[[ "$(sha256 "$TRAINER")" == "$EXPECTED_TRAINER_SHA" ]] || die "trainer SHA drift"

AUDIT_OUTPUT="$(
  "$LOCAL_PY" - "$DATASET" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]).resolve()
meta = json.loads((root / "dataset_manifest.json").read_text())
if meta.get("training_eligible") is not True:
    raise SystemExit("dataset is not training_eligible")
if meta.get("holdout_read") is not False:
    raise SystemExit("dataset metadata records a holdout read")
if meta.get("split", {}).get("random_split") is not False:
    raise SystemExit("dataset is not time-split")
if meta.get("gates", {}).get("future_input_rows") != 0:
    raise SystemExit("dataset contains future model input")

rows = []
with (root / "manifest.jsonl").open() as fh:
    for line_no, line in enumerate(fh, 1):
        row = json.loads(line)
        split, label = row["split"], row["label"]
        if split not in {"train", "val", "test"} or label not in {"SIGNAL", "NO_SIGNAL"}:
            raise SystemExit(f"bad manifest class at line {line_no}")
        path = root / split / label / Path(row["image_path"]).name
        if not path.is_file():
            raise SystemExit(f"missing image at line {line_no}: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["image_sha256"]:
            raise SystemExit(f"image SHA mismatch at line {line_no}: {path}")
        if row.get("future_used_in_model_input") is not False or row.get("holdout_read") is not False:
            raise SystemExit(f"unsafe lineage at line {line_no}")
        rows.append(row)

physical = sorted(root.glob("train/*/*.png")) + sorted(root.glob("val/*/*.png")) + sorted(root.glob("test/*/*.png"))
if len(physical) != len(rows):
    raise SystemExit(f"physical/manifest count mismatch: {len(physical)} != {len(rows)}")
counts = Counter((row["split"], row["label"]) for row in rows)
expected = {
    ("train", "SIGNAL"): 139, ("train", "NO_SIGNAL"): 380,
    ("val", "SIGNAL"): 36, ("val", "NO_SIGNAL"): 130,
    ("test", "SIGNAL"): 61, ("test", "NO_SIGNAL"): 74,
}
if counts != expected:
    raise SystemExit(f"class counts drifted: {dict(counts)}")
print(len(physical), len(rows), *(counts[key] for key in expected))
PY
)"
read -r N_IMAGES N_ROWS TRAIN_SIGNAL TRAIN_NO_SIGNAL VAL_SIGNAL VAL_NO_SIGNAL TEST_SIGNAL TEST_NO_SIGNAL <<<"$AUDIT_OUTPUT"
[[ "$N_IMAGES" == "$EXPECTED_IMAGES" && "$N_ROWS" == "$EXPECTED_IMAGES" ]] || die "expected $EXPECTED_IMAGES images/rows, got $N_IMAGES/$N_ROWS"
printf '  images=%s manifest_rows=%s\n' "$N_IMAGES" "$N_ROWS"
printf '  train SIGNAL=%s NO_SIGNAL=%s\n' "$TRAIN_SIGNAL" "$TRAIN_NO_SIGNAL"
printf '  val   SIGNAL=%s NO_SIGNAL=%s\n' "$VAL_SIGNAL" "$VAL_NO_SIGNAL"
printf '  test  SIGNAL=%s NO_SIGNAL=%s\n' "$TEST_SIGNAL" "$TEST_NO_SIGNAL"
printf '  manifest_sha=%s\n  model_sha=%s\n  trainer_sha=%s\n' "$EXPECTED_MANIFEST_SHA" "$EXPECTED_MODEL_SHA" "$EXPECTED_TRAINER_SHA"

LOCAL_RUN="$FABLE/runs/classify/$NAME"
[[ ! -e "$LOCAL_RUN" ]] || die "refusing to overwrite existing local run: $LOCAL_RUN"

say "remote CUDA and no-overwrite preflight"
REMOTE_PREFLIGHT="$(
  remote_ps <<PS | tr -d '\r'
\$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath '$REMOTE/.venv/Scripts/python.exe' -PathType Leaf)) { throw 'remote Python missing' }
\$run = '$REMOTE/runs/classify/$NAME'
\$log = '$REMOTE/logs/$NAME.log'
\$exit = '$REMOTE/logs/$NAME.exit_code'
\$active = @(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -and \$_.CommandLine -like '*$NAME*' })
if ((Test-Path -LiteralPath \$run) -or (Test-Path -LiteralPath \$log) -or (Test-Path -LiteralPath \$exit) -or \$active.Count) {
  throw 'refusing to overwrite or duplicate an existing remote run/log/exit/process'
}
\$probe = & '$REMOTE/.venv/Scripts/python.exe' -c 'import torch,ultralytics;print(torch.cuda.is_available(),torch.cuda.get_device_name(0),torch.__version__,ultralytics.__version__,sep=chr(124))'
if (\$LASTEXITCODE -ne 0) { throw 'remote environment probe failed' }
if (-not \$probe.StartsWith('True|')) { throw ('remote CUDA unavailable: ' + \$probe) }
Write-Output ('REMOTE_OK|' + \$probe)
PS
)"
printf '%s\n' "$REMOTE_PREFLIGHT"
[[ "$REMOTE_PREFLIGHT" == *"REMOTE_OK|True|"* ]] || die "remote preflight returned no exact CUDA success marker"

if [[ "$MODE" == "check" ]]; then
  printf '\nCheck passed; no dataset/model/trainer was uploaded and no training was started.\n'
  exit 0
fi

DATASET_BASE="$(basename "$DATASET")"
REMOTE_DATASET="$REMOTE/datasets/$DATASET_BASE"
REMOTE_MODEL="$REMOTE/inputs/$EXPECTED_MODEL_SHA/yolo11n-cls.pt"
REMOTE_TRAINER="$REMOTE/scripts/train_fixed_w10_cls_${NAME}.py"
REMOTE_BATCH="$REMOTE/launch_${NAME}.cmd"
REMOTE_LOG="$REMOTE/logs/$NAME.log"

say "package immutable 820-image dataset"
TMP_TAR="$(mktemp -t fixed_w10_gold500_short).tar"
COPYFILE_DISABLE=1 tar -cf "$TMP_TAR" --exclude='*.cache' --exclude='*.npy' --exclude='._*' \
  -C "$(dirname "$DATASET")" "$DATASET_BASE"
TAR_SHA="$(sha256 "$TMP_TAR")"
printf '  tar_bytes=%s tar_sha=%s\n' "$(stat -f %z "$TMP_TAR")" "$TAR_SHA"

TMP_CMD="$(mktemp -t fixed_w10_gold500_short_cmd)"
{
  printf '@echo off\r\nsetlocal\r\n'
  printf '> %s\\logs\\%s.log echo [launcher] started %%DATE%% %%TIME%%\r\n' "$REMOTE_WIN" "$NAME"
  printf 'cd /d %s\r\n' "$REMOTE_WIN"
  printf '%s\\.venv\\Scripts\\python.exe -u %s\\scripts\\train_fixed_w10_cls_%s.py --data C:/fable/datasets/%s --model C:/fable/inputs/%s/yolo11n-cls.pt --name %s --project C:/fable/runs/classify --epochs 100 --patience 20 --batch 8 --imgsz 960 --seed 0 --workers 2 --device 0 >> %s\\logs\\%s.log 2>&1\r\n' \
    "$REMOTE_WIN" "$REMOTE_WIN" "$NAME" "$DATASET_BASE" "$EXPECTED_MODEL_SHA" "$NAME" "$REMOTE_WIN" "$NAME"
  printf 'set RC=%%ERRORLEVEL%%\r\n'
  printf '>> %s\\logs\\%s.log echo [launcher] exit_code=%%RC%% %%DATE%% %%TIME%%\r\n' "$REMOTE_WIN" "$NAME"
  printf '> %s\\logs\\%s.exit_code echo %%RC%%\r\n' "$REMOTE_WIN" "$NAME"
  printf 'exit /b %%RC%%\r\n'
} >"$TMP_CMD"
BATCH_SHA="$(sha256 "$TMP_CMD")"
STAGE_SENTINEL="STAGE_OK|$NAME|$EXPECTED_MANIFEST_SHA|$BATCH_SHA"

say "upload to unique incoming paths"
"${SCP[@]}" "$TMP_TAR" "$HOST:$REMOTE/incoming_${NAME}.tar" || die "dataset upload failed"
"${SCP[@]}" "$MODEL" "$HOST:$REMOTE/incoming_${NAME}_model.pt" || die "model upload failed"
"${SCP[@]}" "$TRAINER" "$HOST:$REMOTE/incoming_${NAME}_trainer.py" || die "trainer upload failed"
"${SCP[@]}" "$TMP_CMD" "$HOST:$REMOTE/incoming_${NAME}.cmd" || die "launcher upload failed"

say "verify and atomically stage remote inputs"
STAGE_OUTPUT="$(
  remote_ps <<PS | tr -d '\r'
\$ErrorActionPreference = 'Stop'
\$run = '$REMOTE/runs/classify/$NAME'
\$log = '$REMOTE_LOG'
\$exit = '$REMOTE/logs/$NAME.exit_code'
\$active = @(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -and \$_.CommandLine -like '*$NAME*' })
if ((Test-Path -LiteralPath \$run) -or (Test-Path -LiteralPath \$log) -or (Test-Path -LiteralPath \$exit) -or \$active.Count) {
  throw 'run/log/exit/process appeared after preflight; refusing to stage or launch'
}
New-Item -ItemType Directory -Force -Path '$REMOTE/datasets','$REMOTE/inputs/$EXPECTED_MODEL_SHA','$REMOTE/scripts','$REMOTE/logs' | Out-Null

function Assert-Hash([string]\$Path, [string]\$Expected, [string]\$Label) {
  if (-not (Test-Path -LiteralPath \$Path -PathType Leaf)) { throw (\$Label + ' missing: ' + \$Path) }
  \$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath \$Path).Hash.ToLowerInvariant()
  if (\$actual -ne \$Expected) { throw (\$Label + ' hash mismatch: ' + \$actual) }
}
function Finalize-Immutable([string]\$Incoming, [string]\$Target, [string]\$Expected, [string]\$Label) {
  Assert-Hash \$Incoming \$Expected ('incoming ' + \$Label)
  if (Test-Path -LiteralPath \$Target) {
    Assert-Hash \$Target \$Expected ('existing ' + \$Label)
    Remove-Item -LiteralPath \$Incoming -Force
  } else {
    Move-Item -LiteralPath \$Incoming -Destination \$Target
    Assert-Hash \$Target \$Expected ('final ' + \$Label)
  }
}
function Assert-Dataset([string]\$Root) {
  if (-not (Test-Path -LiteralPath \$Root -PathType Container)) { throw ('dataset missing: ' + \$Root) }
  Assert-Hash (Join-Path \$Root 'dataset_manifest.json') '$EXPECTED_DATASET_META_SHA' 'dataset metadata'
  Assert-Hash (Join-Path \$Root 'manifest.jsonl') '$EXPECTED_MANIFEST_SHA' 'dataset manifest'
  \$images = @(Get-ChildItem -LiteralPath \$Root -Recurse -File -Filter '*.png').Count
  \$rows = @(Get-Content -LiteralPath (Join-Path \$Root 'manifest.jsonl')).Count
  if (\$images -ne $EXPECTED_IMAGES -or \$rows -ne $EXPECTED_IMAGES) {
    throw ('remote dataset count mismatch: images=' + \$images + ' rows=' + \$rows)
  }
}

Assert-Hash '$REMOTE/incoming_${NAME}.tar' '$TAR_SHA' 'dataset tar'
if (Test-Path -LiteralPath '$REMOTE_DATASET') {
  Assert-Dataset '$REMOTE_DATASET'
  Remove-Item -LiteralPath '$REMOTE/incoming_${NAME}.tar' -Force
} else {
  \$stage = '$REMOTE/datasets/.stage_$NAME'
  try {
    Remove-Item -LiteralPath \$stage -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path \$stage | Out-Null
    & tar.exe -xf '$REMOTE/incoming_${NAME}.tar' -C \$stage
    if (\$LASTEXITCODE -ne 0) { throw 'dataset tar extraction failed' }
    \$incomingDataset = Join-Path \$stage '$DATASET_BASE'
    Assert-Dataset \$incomingDataset
    Move-Item -LiteralPath \$incomingDataset -Destination '$REMOTE_DATASET'
    Assert-Dataset '$REMOTE_DATASET'
  } finally {
    Remove-Item -LiteralPath \$stage -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath '$REMOTE/incoming_${NAME}.tar' -Force -ErrorAction SilentlyContinue
  }
}
Finalize-Immutable '$REMOTE/incoming_${NAME}_model.pt' '$REMOTE_MODEL' '$EXPECTED_MODEL_SHA' 'pretrained model'
Finalize-Immutable '$REMOTE/incoming_${NAME}_trainer.py' '$REMOTE_TRAINER' '$EXPECTED_TRAINER_SHA' 'trainer'
Finalize-Immutable '$REMOTE/incoming_${NAME}.cmd' '$REMOTE_BATCH' '$BATCH_SHA' 'batch launcher'
Write-Output '$STAGE_SENTINEL'
PS
)"
printf '%s\n' "$STAGE_OUTPUT"
[[ "$STAGE_OUTPUT" == *"$STAGE_SENTINEL"* ]] || die "remote staging returned no exact success sentinel"

rm -f -- "$TMP_TAR" "$TMP_CMD"
TMP_TAR=""
TMP_CMD=""

say "start one detached WMI job"
START_OUTPUT="$(
  remote_ps <<PS | tr -d '\r'
\$ErrorActionPreference = 'Stop'
\$run = '$REMOTE/runs/classify/$NAME'
\$log = '$REMOTE/logs/$NAME.log'
\$exit = '$REMOTE/logs/$NAME.exit_code'
\$active = @(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -and \$_.CommandLine -like '*$NAME*' })
if ((Test-Path -LiteralPath \$run) -or (Test-Path -LiteralPath \$log) -or (Test-Path -LiteralPath \$exit) -or \$active.Count) {
  throw 'run/log/exit/process appeared before launch; refusing duplicate start'
}
\$cmd = 'cmd.exe /d /c "$REMOTE_WIN\\launch_${NAME}.cmd"'
\$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=\$cmd}
if (\$result.ReturnValue -ne 0) { throw ('WMI create failed: ' + \$result.ReturnValue) }
Write-Output ('PID=' + \$result.ProcessId)
PS
)"
printf '%s\n' "$START_OUTPUT"
[[ "$START_OUTPUT" =~ PID=([0-9]+) ]] || die "WMI returned no PID"
printf '  run=%s\n  log=%s\n' "$NAME" "$REMOTE_LOG"
printf '  status: FABLE_3060_HOST=%q bash %q --status\n' "$HOST" "$0"
printf '\nStarted only. No retry, evaluation, promotion, ACTIVE change, or holdout read was performed.\n'
