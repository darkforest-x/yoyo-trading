#!/usr/bin/env python3
"""Build Dataset V3 Gold Core from the migration table.

One variable changes against R1: the negatives. Positives, window protocol,
renderer and the whole val split are carried over unchanged -- and "unchanged"
is proved, not asserted: every reused sample is re-rendered through the yoyo
chain and its SHA compared with the frozen manifest, so a silent pixel drift
fails the build instead of quietly producing a different dataset.

What actually changes:

* 916 Owner-long mirrors leave the negatives (an unconfirmed mirror is neither
  a positive nor a negative);
* 1,370 model-mined hard negatives leave the negatives (never reviewed, so
  they cannot be certified shape-NO);
* 765 Owner-confirmed shape-NO detections enter as hard negatives.

Owner-YES events do **not** become positives: their class is confirmed but
their geometry is still model-proposed, and this project does not promote a
model's rectangle to a gold box.

Writes into this repo's own datasets/ directory; fable-trading stays read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from yoyo.datasets.legacy_audit import parse_time
from yoyo.datasets.window_render import SourceError, enrich, load_prefix, render_sample

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CONFIG = ROOT / "configs/source_repo.json"
DEFAULT_MIGRATION = ROOT / "manifests/legacy_label_migration_v3.jsonl"
DEFAULT_OUT_DIR = ROOT / "datasets/dataset_v3_gold_core_v1"
DEFAULT_MANIFEST = ROOT / "manifests/dataset_v3_gold_core_v1.json"
LEGACY_DATASET = "datasets/owner_short_gold_center_hardneg_r1"
CLASS_NAME = "ma_dense_core"
BAR = timedelta(minutes=15)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def csv_index(repo: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted((repo / "data/kline_fetched").glob("okx_*_15m_*.csv")):
        symbol = path.name.split("_15m_")[0][len("okx_") :]
        index.setdefault(symbol, path)
    return index


def resolve_event_windows(
    events: list[dict[str, Any]], repo: Path, holdout_start: datetime
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn (symbol, decision_time, window_len) into global bar indices."""
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_symbol[str(event["symbol"])].append(event)
    paths = csv_index(repo)
    resolved: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for symbol, rows in sorted(by_symbol.items()):
        path = paths.get(symbol)
        if path is None:
            dropped += [{**row, "drop_reason": "no source csv"} for row in rows]
            continue
        frame = pd.read_csv(path, usecols=["open_time"], encoding_errors="replace")
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["open_time"]).reset_index(drop=True)
        position = {time: index for index, time in enumerate(frame["open_time"])}
        for row in rows:
            decision = parse_time(row["decision_time"])
            if decision >= holdout_start:
                dropped.append({**row, "drop_reason": "decision at or after holdout"})
                continue
            index = position.get(decision)
            if index is None:
                dropped.append({**row, "drop_reason": "decision bar not in source series"})
                continue
            length = int(row.get("window_len") or 12)
            start = index - (length - 1)
            if start < 0:
                dropped.append({**row, "drop_reason": "window starts before series"})
                continue
            resolved.append(
                {
                    **row,
                    "source_csv": str(path.relative_to(repo)),
                    "win_start": start,
                    "win_end": index,
                    "win_len": length,
                    "decision_bar": index,
                    "visible_end_bar": index,
                }
            )
    return resolved, dropped


def render_group(
    items: list[dict[str, Any]], repo: Path, out_dir: Path, builder_commit: str
) -> list[dict[str, Any]]:
    """Render every sample of one symbol from a single prefix load."""
    if not items:
        return []
    symbol = str(items[0]["symbol"])
    source = repo / str(items[0]["source_csv"])
    required_end = max(int(item["win_end"]) for item in items)
    enriched = enrich(load_prefix(source, required_end))
    written: list[dict[str, Any]] = []
    for item in items:
        split = str(item["split"])
        core = item.get("core_global")
        rendered = render_sample(
            enriched,
            int(item["win_start"]),
            int(item["win_end"]),
            tuple(core) if core else None,
        )
        image_path = out_dir / "images" / split / f"{item['sample_id']}.png"
        label_path = out_dir / "labels" / split / f"{item['sample_id']}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), rendered["image"]):
            raise OSError(image_path)
        label_path.write_text(rendered["label"], encoding="utf-8")
        image_sha = sha256_bytes(image_path.read_bytes())
        label_sha = sha256_bytes(rendered["label"].encode("utf-8"))
        window = enriched.iloc[int(item["win_start"]) : int(item["win_end"]) + 1]
        high, low = float(window["high"].max()), float(window["low"].min())
        written.append(
            {
                "sample_id": item["sample_id"],
                "window_start_time": window["open_time"].iloc[0].isoformat(),
                "decision_time": window["open_time"].iloc[-1].isoformat(),
                "causal_span_pct": round((high / low - 1.0) * 100, 4) if low else None,
                "event_id": item.get("event_id"),
                "symbol": symbol,
                "timeframe": "15m",
                "source_repo": str(repo),
                "source_commit": item.get("source_commit"),
                "source_dataset": item.get("source_dataset"),
                "source_path": str(item["source_csv"]),
                "window_start_bar": int(item["win_start"]),
                "window_end_bar": int(item["win_end"]),
                "window_length": int(item["win_len"]),
                "anchor_bar": item.get("anchor_bar"),
                "decision_bar": int(item.get("decision_bar", item["win_end"])),
                "visible_end_bar": int(item["win_end"]),
                "box_start_bar": core[0] if core else None,
                "box_end_bar": core[1] if core else None,
                "class_name": CLASS_NAME if core else None,
                "class_status": item["class_status"],
                "box_status": item["box_status"],
                "split": split,
                "is_hard_negative": bool(item.get("is_hard_negative")),
                "ignore_reason": None,
                "future_used_for_l1_label": False,
                "image_path": str(image_path.relative_to(ROOT)),
                "image_sha256": image_sha,
                "label_sha256": label_sha,
                "expected_image_sha256": item.get("expected_image_sha256"),
                "parity": (
                    image_sha == item["expected_image_sha256"]
                    if item.get("expected_image_sha256")
                    else None
                ),
                "builder_commit": builder_commit,
                "origin": item["origin"],
            }
        )
    return written


def build(
    source_config: Path,
    migration_path: Path,
    out_dir: Path,
    manifest_path: Path,
    builder_commit: str,
) -> dict[str, Any]:
    source = json.loads(source_config.read_text(encoding="utf-8"))
    repo = Path(source["source_repo"])
    holdout_start = parse_time(source["holdout_start"])
    legacy = repo / LEGACY_DATASET

    migration = read_jsonl(migration_path)
    decision_by_id = {str(row["sample_id"]): row for row in migration}

    items: list[dict[str, Any]] = []
    for name, origin in (
        ("positive_manifest", "legacy_positive"),
        ("negative_manifest", "legacy_negative"),
    ):
        for row in read_jsonl(legacy / f"{name}.jsonl"):
            decision = decision_by_id.get(str(row["sample_id"]))
            if decision is None or decision["migration"] not in {"KEEP_POSITIVE", "KEEP_NEGATIVE"}:
                continue
            items.append(
                {
                    **row,
                    "origin": origin,
                    "class_status": decision["class_status"],
                    "box_status": decision["box_status"],
                    "is_hard_negative": False,
                    "expected_image_sha256": row["image_sha256"],
                    "source_dataset": LEGACY_DATASET,
                }
            )

    owner_no = [
        row
        for row in migration
        if row["population"] == "owner_review_event" and row["migration"] == "KEEP_NEGATIVE"
    ]
    resolved, dropped = resolve_event_windows(owner_no, repo, holdout_start)
    for row in resolved:
        items.append(
            {
                **row,
                "sample_id": "ownerno_" + str(row["sample_id"]).replace(":", "_"),
                "split": "train",
                "origin": "owner_confirmed_shape_no",
                "is_hard_negative": True,
                "expected_image_sha256": None,
                "source_dataset": row["pack"],
                "core_global": None,
            }
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item["symbol"])].append(item)
    rows: list[dict[str, Any]] = []
    for symbol in sorted(groups):
        rows += render_group(groups[symbol], repo, out_dir, builder_commit)

    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: {CLASS_NAME}\n",
        encoding="utf-8",
    )
    rows_path = out_dir / "manifest.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )

    reused = [row for row in rows if row["parity"] is not None]
    parity_failures = [row for row in reused if not row["parity"]]
    windows = Counter(
        (row["symbol"], row["window_start_bar"], row["window_end_bar"]) for row in rows
    )
    duplicates = {key: count for key, count in windows.items() if count > 1}
    events_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["event_id"]:
            events_by_split[row["split"]].add(str(row["event_id"]))
    cross_split = events_by_split.get("train", set()) & events_by_split.get("val", set())
    positives = [row for row in rows if row["class_name"]]
    negatives = [row for row in rows if not row["class_name"]]
    train_positive = sum(1 for row in positives if row["split"] == "train")
    train_negative = sum(1 for row in negatives if row["split"] == "train")

    summary = {
        "protocol": "yoyo_dataset_v3_gold_core_v1_20260812",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "class_name": CLASS_NAME,
        "direction_scope": "short_pipeline",
        "window_protocol": "owner_gold_center_crop_w12_19",
        "dataset_dir": str(out_dir),
        "builder_commit": builder_commit,
        "source_repo": str(repo),
        "legacy_dataset": LEGACY_DATASET,
        "counts": {
            "total": len(rows),
            "positive": len(positives),
            "negative": len(negatives),
            "train_positive": train_positive,
            "train_negative": train_negative,
            "val_positive": sum(1 for row in positives if row["split"] == "val"),
            "val_negative": sum(1 for row in negatives if row["split"] == "val"),
            "hard_negative": sum(1 for row in rows if row["is_hard_negative"]),
        },
        "by_origin": dict(Counter(row["origin"] for row in rows)),
        "by_box_status": dict(Counter(row["box_status"] for row in rows)),
        "by_class_status": dict(Counter(row["class_status"] for row in rows)),
        "train_negative_to_positive_ratio": round(train_negative / train_positive, 3)
        if train_positive
        else None,
        "symbols": len({row["symbol"] for row in rows}),
        "window_lengths": dict(sorted(Counter(row["window_length"] for row in rows).items())),
        "by_month": dict(sorted(Counter(row["decision_time"][:7] for row in rows).items())),
        "by_causal_span_bucket": dict(
            sorted(
                Counter(
                    "unknown"
                    if row["causal_span_pct"] is None
                    else "lt1"
                    if row["causal_span_pct"] < 1
                    else "1to2"
                    if row["causal_span_pct"] < 2
                    else "2to4"
                    if row["causal_span_pct"] < 4
                    else "ge4"
                    for row in rows
                ).items()
            )
        ),
        "positive_core_position": dict(
            sorted(
                Counter(
                    "left"
                    if position < 1 / 3
                    else "middle"
                    if position < 2 / 3
                    else "right"
                    for position in (
                        ((row["box_start_bar"] + row["box_end_bar"]) / 2 - row["window_start_bar"])
                        / max(1, row["window_end_bar"] - row["window_start_bar"])
                        for row in rows
                        if row["class_name"]
                    )
                ).items()
            )
        ),
        "reused_samples_checked_for_parity": len(reused),
        "parity_failures": len(parity_failures),
        "duplicate_windows": len(duplicates),
        "cross_split_event_ids": sorted(cross_split),
        "owner_no_events_resolved": len(resolved),
        "owner_no_events_dropped": len(dropped),
        "drop_reasons": dict(Counter(row["drop_reason"] for row in dropped)),
        "future_used_for_l1_label": False,
        "holdout_start": source["holdout_start"],
        "holdout_read": False,
        "manifest_path": str(rows_path),
        "manifest_sha256": sha256_bytes(rows_path.read_bytes()),
        "training_eligible": not parity_failures and not duplicates and not cross_split,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--migration", type=Path, default=DEFAULT_MIGRATION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--builder-commit", default="unrecorded")
    args = parser.parse_args()
    summary = build(
        args.source_config, args.migration, args.out_dir, args.manifest, args.builder_commit
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["training_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
