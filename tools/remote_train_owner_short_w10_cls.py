#!/usr/bin/env python3
"""Stage and train the frozen Owner-SHORT W10 classifier on Windows.

This script is shipped to ``C:\\fable`` and started through a detached WMI
process.  It verifies every immutable input and every rendered image before it
imports CUDA or starts Ultralytics.  It never reads OHLCV/holdout data, retries
training, promotes a model, or changes ACTIVE.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(r"C:\fable")
NAME = "fixed_w10_gold500_short_ownerhn_v1_cls"
DATASET_NAME = "fixed_w10_gold500_short_ownerhn_v1"

INCOMING_TAR = ROOT / f"incoming_{DATASET_NAME}.tar"
INCOMING_MODEL = ROOT / f"incoming_{NAME}_model.pt"
INCOMING_TRAINER = ROOT / f"incoming_{NAME}_trainer.py"

DATASET = ROOT / "datasets" / DATASET_NAME
MODEL_SHA = "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7"
TRAINER_SHA = "81eac7daea14af8717542dd8c0df21fe862f228a1f4f3c843c63c9ab82fe948b"
DATASET_META_SHA = "3d2470fc323bde5641b4ec3243a28962b0c4cd429e1149c99186f9c7b84fee65"
MANIFEST_SHA = "9897f34ee9c27003cff5f8d6db8d26c04889229a555a04308402f2445af92e1e"

MODEL = ROOT / "inputs" / MODEL_SHA / "yolo11n-cls.pt"
TRAINER = ROOT / f"train_fixed_w10_cls_{DATASET_NAME}.py"
RUN = ROOT / "runs" / "classify" / NAME
RECEIPT = ROOT / "logs" / f"{NAME}.launch.json"
EXIT_RECEIPT = ROOT / "logs" / f"{NAME}.exit_code"
HOLDOUT_START = datetime(2026, 5, 4, tzinfo=timezone.utc)
EXPECTED_COUNTS = {
    ("train", "SIGNAL"): 139,
    ("train", "NO_SIGNAL"): 380,
    ("val", "SIGNAL"): 36,
    ("val", "NO_SIGNAL"): 130,
    ("test", "SIGNAL"): 61,
    ("test", "NO_SIGNAL"): 74,
}


def note(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA mismatch: {actual} != {expected}")


def parse_utc(value: object) -> datetime:
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    stamp = datetime.fromisoformat(text)
    if stamp.tzinfo is None:
        raise ValueError(f"naive decision time: {value}")
    return stamp.astimezone(timezone.utc)


def safe_extract_dataset() -> None:
    if DATASET.exists():
        note("dataset target exists; verifying immutable contents")
        return
    if not INCOMING_TAR.is_file():
        raise FileNotFoundError(f"missing dataset tar: {INCOMING_TAR}")
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{DATASET_NAME}.", dir=DATASET.parent))
    try:
        with tarfile.open(INCOMING_TAR, "r") as archive:
            members = archive.getmembers()
            for member in members:
                parts = PurePosixPath(member.name).parts
                if (
                    not parts
                    or parts[0] != DATASET_NAME
                    or member.name.startswith(("/", "\\"))
                    or ".." in parts
                ):
                    raise RuntimeError(f"unsafe tar member: {member.name}")
            archive.extractall(stage)
        extracted = stage / DATASET_NAME
        if not extracted.is_dir():
            raise RuntimeError("dataset tar has no expected top-level directory")
        os.replace(extracted, DATASET)
        note("dataset extracted to immutable target")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def validate_dataset() -> None:
    require_sha(DATASET / "dataset_manifest.json", DATASET_META_SHA, "dataset metadata")
    require_sha(DATASET / "manifest.jsonl", MANIFEST_SHA, "sample manifest")
    meta = json.loads((DATASET / "dataset_manifest.json").read_text(encoding="utf-8"))
    if meta.get("training_eligible") is not True:
        raise RuntimeError("dataset is not training_eligible")
    if meta.get("holdout_read") is not False or meta.get("auto_promote") is not False:
        raise RuntimeError("dataset holdout/promote contract is unsafe")
    if meta.get("split", {}).get("random_split") is not False:
        raise RuntimeError("dataset is not chronologically split")

    rows = []
    for line_number, line in enumerate(
        (DATASET / "manifest.jsonl").read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        split, label = str(row["split"]), str(row["label"])
        if (split, label) not in EXPECTED_COUNTS:
            raise RuntimeError(f"unexpected split/label at line {line_number}")
        if row.get("holdout_read") is not False:
            raise RuntimeError(f"holdout flag at line {line_number}")
        if row.get("future_used_in_model_input") is not False:
            raise RuntimeError(f"future input flag at line {line_number}")
        if parse_utc(row["decision_time"]) >= HOLDOUT_START:
            raise RuntimeError(f"holdout timestamp at line {line_number}")
        if int(row["visible_end_bar"]) > int(row["decision_bar"]):
            raise RuntimeError(f"future visible bar at line {line_number}")
        path = DATASET / split / label / Path(str(row["image_path"])).name
        require_sha(path, str(row["image_sha256"]), f"image line {line_number}")
        rows.append((split, label, path.resolve()))

    counts = Counter((split, label) for split, label, _ in rows)
    if counts != Counter(EXPECTED_COUNTS):
        raise RuntimeError(f"class-count mismatch: {dict(counts)}")
    physical = {
        path.resolve()
        for split, label in EXPECTED_COUNTS
        for path in (DATASET / split / label).glob("*.png")
    }
    declared = {path for _, _, path in rows}
    if len(rows) != 820 or len(declared) != 820 or physical != declared:
        raise RuntimeError(
            f"dataset conservation failed: rows={len(rows)} "
            f"declared={len(declared)} physical={len(physical)}"
        )
    note(f"dataset verified: images={len(rows)} counts={dict(counts)}")


def finalize_input(incoming: Path, target: Path, expected: str, label: str) -> None:
    require_sha(incoming, expected, f"incoming {label}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require_sha(target, expected, f"existing {label}")
    else:
        shutil.copy2(incoming, target)
        require_sha(target, expected, f"staged {label}")
    note(f"{label} verified: {target}")


def write_receipt(payload: dict) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if RUN.exists() or EXIT_RECEIPT.exists():
        raise RuntimeError(f"refusing existing run/exit receipt: {RUN}")
    safe_extract_dataset()
    validate_dataset()
    finalize_input(INCOMING_MODEL, MODEL, MODEL_SHA, "pretrained model")
    finalize_input(INCOMING_TRAINER, TRAINER, TRAINER_SHA, "trainer")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    gpu = torch.cuda.get_device_name(0)
    command = [
        sys.executable,
        "-u",
        str(TRAINER),
        "--data",
        str(DATASET),
        "--model",
        str(MODEL),
        "--name",
        NAME,
        "--project",
        str(ROOT / "runs" / "classify"),
        "--epochs",
        "100",
        "--patience",
        "20",
        "--batch",
        "8",
        "--imgsz",
        "960",
        "--seed",
        "0",
        "--workers",
        "2",
        "--device",
        "0",
    ]
    receipt = {
        "protocol": "fixed_w10_gold500_short_ownerhn_v1_cls_train_v1",
        "state": "training",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "name": NAME,
        "gpu": gpu,
        "dataset": str(DATASET),
        "dataset_manifest_sha256": MANIFEST_SHA,
        "model_sha256": MODEL_SHA,
        "trainer_sha256": TRAINER_SHA,
        "command": command,
        "holdout_read": False,
        "auto_promote": False,
    }
    write_receipt(receipt)
    note(f"CUDA verified: {gpu}")
    note("starting frozen W10 classifier training")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    receipt["state"] = "complete" if completed.returncode == 0 else "failed"
    receipt["exit_code"] = completed.returncode
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_receipt(receipt)
    return completed.returncode


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        note("bootstrap failed\n" + traceback.format_exc())
        try:
            write_receipt(
                {
                    "protocol": "fixed_w10_gold500_short_ownerhn_v1_cls_train_v1",
                    "state": "bootstrap_failed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "traceback": traceback.format_exc(),
                    "holdout_read": False,
                    "auto_promote": False,
                }
            )
        except Exception:
            pass
    finally:
        EXIT_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        EXIT_RECEIPT.write_text(f"{code}\n", encoding="ascii")
    raise SystemExit(code)
