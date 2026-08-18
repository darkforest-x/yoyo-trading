#!/usr/bin/env python3
"""Build the reviewed SHORT-only fixed-W10 classification arm.

The source decision is deliberately narrower than the later 1,345-row
derived-positive pool.  The first Owner direction-review sample contained 500
boxes; 257 were explicitly marked SHORT.  Those exact box IDs are joined back
to the Owner-confirmed positive manifest, then collapsed by dependency event
so a cluster cannot vote multiple times.  Negative examples come only from the
584 false fires explicitly rejected by the Owner in four train-time review
packs.  LONG/SKIP boxes are not negatives: direction is not an L1 shape label.

Every image is re-rendered from a causal OHLCV prefix with the fixed
W10/Core4/Confirm1 renderer.  The old W12--W19 review PNGs and review-only
future charts are never copied.  Time splits are fixed before rendering, with
162 bars purged on both sides of each boundary.  The builder refuses holdout,
future-input, guard-overlap, duplicate-event, duplicate-image, and residual
output state instead of silently dropping questionable rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import pandas as pd

from yoyo.datasets.gold_render import load_causal_prefix
from yoyo.datasets.legacy_gold_migration.intervals import center_four_bar
from yoyo.datasets.legacy_gold_migration.io import git_head, sha256_file, write_jsonl
from yoyo.datasets.legacy_gold_migration.canonical import w10_fields
from yoyo.datasets.legacy_gold_migration.renderer import render_w10


ROOT = Path(__file__).resolve().parents[1]
FABLE = Path("/Users/zhangzc/fable-trading")
HOLDOUT_START = pd.Timestamp("2026-05-04T00:00:00Z")
CUT1 = pd.Timestamp("2025-12-01T00:00:00Z")
CUT2 = pd.Timestamp("2026-02-01T00:00:00Z")
PURGE_BARS = 162
BAR_DELTA = pd.Timedelta(minutes=15)
PURGE_DELTA = BAR_DELTA * PURGE_BARS
OWNER_GUARD_BARS = 12

DEFAULT_SHEET = FABLE / "analysis/output/owner_side_review/review_sheet.csv"
DEFAULT_POSITIVES = FABLE / "datasets/owner_short_gold_center_v1/positive_manifest.jsonl"
DEFAULT_OUT = FABLE / "datasets/fixed_w10_gold500_short_ownerhn_v1"

HARD_NEGATIVE_SOURCES = (
    FABLE
    / "analysis/output/owner_short_train_hardneg_review200_v1/owner_review_labeled_manifest.jsonl",
    FABLE
    / "analysis/output/owner_short_train_positive_retrieval100_v1/owner_review_labeled_manifest.jsonl",
    FABLE
    / "analysis/output/owner_short_train_hardneg_expansion200_v2/owner_review_labeled_manifest.jsonl",
    FABLE
    / "analysis/output/owner_short_train_hardneg_newblocks200_v3/owner_review_labeled_manifest.jsonl",
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_gold500_short_ids(sheet_path: Path) -> set[str]:
    """Return unique box IDs from the original 500-box review sample marked SHORT."""
    with Path(sheet_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"box_id", "in_sample", "owner_side"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"bad direction-review sheet: {sheet_path}")
    sample = [row for row in rows if str(row["in_sample"]).strip() == "1"]
    if len(sample) != 500:
        raise ValueError(f"expected original review sample=500, got {len(sample)}")
    allowed = {"long", "short", "skip"}
    bad = sorted({str(row["owner_side"]).strip().lower() for row in sample} - allowed)
    if bad:
        raise ValueError(f"unresolved direction labels in gold500 sample: {bad}")
    return {
        str(row["box_id"]).strip()
        for row in sample
        if str(row["owner_side"]).strip().lower() == "short"
    }


def select_positive_rows(rows: Sequence[dict], short_ids: set[str]) -> list[dict]:
    """Join reviewed SHORT box IDs to derived rows, requiring exact one-row coverage."""
    by_annotation: dict[str, list[int]] = defaultdict(list)
    selected_indices: set[int] = set()
    for index, row in enumerate(rows):
        if row.get("source_owner_gold_confirmed") is not True:
            continue
        annotations = {str(value) for value in row.get("owner_annotation_ids") or []}
        hits = annotations & short_ids
        if hits:
            selected_indices.add(index)
            for annotation in hits:
                by_annotation[annotation].append(index)
    missing = sorted(short_ids - set(by_annotation))
    ambiguous = {key: value for key, value in by_annotation.items() if len(set(value)) != 1}
    if missing or ambiguous:
        raise ValueError(
            f"SHORT positive join failed: missing={len(missing)} ambiguous={len(ambiguous)}"
        )
    selected = [dict(rows[index]) for index in sorted(selected_indices)]
    sample_ids = [str(row.get("sample_id")) for row in selected]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate selected positive sample_id")
    return selected


def split_for_time(value: object) -> str | None:
    """Return the fixed chronological split, or None inside either purge band."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    if stamp >= HOLDOUT_START:
        raise ValueError(f"holdout timestamp: {stamp.isoformat()}")
    if abs(stamp - CUT1) < PURGE_DELTA or abs(stamp - CUT2) < PURGE_DELTA:
        return None
    if stamp < CUT1:
        return "train"
    if stamp < CUT2:
        return "val"
    return "test"


def load_confirmed_hard_negatives(sources: Iterable[Path]) -> list[dict]:
    """Load only explicit Owner false fires and reject any unsafe source row."""
    out: list[dict] = []
    seen_events: set[str] = set()
    seen_images: set[str] = set()
    seen_intervals: set[tuple[str, str, str]] = set()
    for source in sources:
        source = Path(source)
        for row in _read_jsonl(source):
            if row.get("owner_decision") != "hard_negative":
                continue
            event_id = str(row.get("event_id") or "")
            unsafe = {
                "owner_confirmed": row.get("owner_confirmed") is not True,
                "holdout_read": row.get("holdout_read") is not False,
                "future_data_in_causal_input": row.get("future_data_in_causal_input") is not False,
                "future_data_in_training_image": row.get("future_data_in_training_image", False) is not False,
                "touches_owner_box_guard": row.get("touches_owner_box_guard") is not False,
            }
            for causal_field in (
                "selection_future_used",
                "hard_negative_expansion_future_used",
                "hard_negative_newblocks_future_used",
            ):
                if causal_field in row:
                    unsafe[causal_field] = row[causal_field] is not False
            if not event_id or any(unsafe.values()):
                failed = sorted(key for key, failed in unsafe.items() if failed)
                raise ValueError(f"unsafe Owner hard negative {event_id or '<missing>'}: {failed}")
            decision = pd.Timestamp(row["decision_time"])
            if decision.tzinfo is None:
                decision = decision.tz_localize("UTC")
            else:
                decision = decision.tz_convert("UTC")
            if decision >= HOLDOUT_START:
                raise ValueError(f"hard negative in holdout: {event_id}")
            image_sha = str(row.get("causal_input_sha256") or "")
            interval = (str(row.get("symbol")), str(row.get("window_start_time")), decision.isoformat())
            if event_id in seen_events or (image_sha and image_sha in seen_images) or interval in seen_intervals:
                raise ValueError(f"duplicate Owner hard negative: {event_id}")
            seen_events.add(event_id)
            if image_sha:
                seen_images.add(image_sha)
            seen_intervals.add(interval)
            item = dict(row)
            item["source_review_manifest"] = str(source)
            out.append(item)
    return out


def _snapshot_path(row: dict) -> Path:
    block = str(row["candidate_block"])
    version = "v1" if block.startswith("B") else "v2" if block.startswith("C") else None
    if version is None:
        raise ValueError(f"unknown hard-negative block: {block}")
    return (
        FABLE
        / f"analysis/output/owner_short_train_hardneg_blocks_{version}"
        / block
        / "audit_snapshot/kline_snapshot"
        / f"{row['symbol']}.csv"
    )


def _positive_geometry(row: dict) -> dict:
    lo, hi = (int(value) for value in row["core_global"])
    core_start, core_end = center_four_bar(lo, hi)
    mapped = w10_fields(core_start, core_end)
    source = FABLE / str(row["source_csv"])
    frame = load_causal_prefix(source, int(mapped["decision_bar"]))
    times = pd.to_datetime(frame["open_time"], utc=True)
    decision_time = times.iloc[int(mapped["decision_bar"])]
    return {
        **row,
        **mapped,
        "source_path": str(source),
        "decision_time_w10": decision_time.isoformat(),
        "split_w10": split_for_time(decision_time),
        "core_start_bar_w10": core_start,
        "core_end_exclusive_bar_w10": core_end,
    }


def _collapse_dependencies(rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the earliest causal decision per dependency event; retain lineage for the rest."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["dependency_id"])].append(dict(row))
    selected: list[dict] = []
    dropped: list[dict] = []
    for dependency, group in sorted(groups.items()):
        ranked = sorted(
            group,
            key=lambda row: (pd.Timestamp(row["decision_time_w10"]), str(row["sample_id"])),
        )
        winner = dict(ranked[0])
        winner["dependency_event_id"] = dependency
        winner["dependency_group_size"] = len(ranked)
        winner["dependency_source_sample_ids"] = [str(row["sample_id"]) for row in ranked]
        selected.append(winner)
        for row in ranked[1:]:
            dropped.append(
                {
                    "sample_id": row["sample_id"],
                    "dependency_event_id": dependency,
                    "kept_sample_id": winner["sample_id"],
                    "reason": "same_dependency_event_keep_earliest_decision",
                }
            )
    return selected, dropped


def _owner_protected_intervals(rows: Sequence[dict]) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    by_symbol: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(list)
    time_cache: dict[Path, pd.Series] = {}
    for row in rows:
        source = FABLE / str(row["source_csv"])
        lo, hi = (int(value) for value in row["source_owner_global"])
        times = time_cache.get(source)
        if times is None or len(times) <= hi:
            raw = pd.read_csv(source, usecols=["open_time"], nrows=hi + 1)
            times = pd.to_datetime(raw["open_time"], utc=True, errors="raise")
            if len(times) <= hi:
                raise ValueError(f"Owner positive index out of bounds: {source} need {hi}")
            time_cache[source] = times
        by_symbol[str(row["symbol"])].append(
            (
                times.iloc[lo] - BAR_DELTA * OWNER_GUARD_BARS,
                times.iloc[hi] + BAR_DELTA * (OWNER_GUARD_BARS + 1),
            )
        )
    return by_symbol


def _assert_not_owner_protected(
    row: dict, protected: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]
) -> None:
    decision = pd.Timestamp(row["decision_time"])
    if decision.tzinfo is None:
        decision = decision.tz_localize("UTC")
    else:
        decision = decision.tz_convert("UTC")
    start = decision - BAR_DELTA * 9
    for lo, hi in protected.get(str(row["symbol"]), []):
        if start < hi and decision + BAR_DELTA > lo:
            raise ValueError(f"hard negative overlaps Owner positive guard: {row['event_id']}")


def _render_positive(row: dict, out_path: Path) -> dict:
    decision = int(row["decision_bar"])
    frame = load_causal_prefix(Path(row["source_path"]), decision)
    window = frame.iloc[int(row["window_start_bar"]) : int(row["window_end_exclusive_bar"])].reset_index(drop=True)
    image, _ = render_w10(window, y_pad_frac=0.05, overlay=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image):
        raise RuntimeError(f"failed to write {out_path}")
    return {
        "sample_id": str(row["sample_id"]),
        "source_kind": "gold500_owner_short",
        "label": "SIGNAL",
        "split": str(row["split_w10"]),
        "symbol": str(row["symbol"]),
        "decision_time": str(row["decision_time_w10"]),
        "decision_bar": decision,
        "window_start_bar": int(row["window_start_bar"]),
        "window_end_exclusive_bar": int(row["window_end_exclusive_bar"]),
        "visible_end_bar": decision,
        "core_start_bar": int(row["core_start_bar_w10"]),
        "core_end_exclusive_bar": int(row["core_end_exclusive_bar_w10"]),
        "dependency_event_id": str(row["dependency_event_id"]),
        "dependency_group_size": int(row["dependency_group_size"]),
        "source_sample_ids": list(row["dependency_source_sample_ids"]),
        "owner_annotation_ids": list(row.get("owner_annotation_ids") or []),
        "source_path": str(row["source_path"]),
        "image_path": str(out_path),
        "image_sha256": sha256_file(out_path),
        "future_used_in_model_input": False,
        "holdout_read": False,
    }


def _render_negative(row: dict, out_path: Path) -> dict:
    decision = int(row["decision_i"])
    source = _snapshot_path(row)
    frame = load_causal_prefix(source, decision)
    times = pd.to_datetime(frame["open_time"], utc=True)
    actual_time = times.iloc[decision]
    expected_time = pd.Timestamp(row["decision_time"])
    if expected_time.tzinfo is None:
        expected_time = expected_time.tz_localize("UTC")
    else:
        expected_time = expected_time.tz_convert("UTC")
    if actual_time != expected_time:
        raise ValueError(f"hard-negative index/time mismatch: {row['event_id']}")
    start = decision - 9
    if start < 0:
        raise ValueError(f"hard-negative W10 underflow: {row['event_id']}")
    window = frame.iloc[start : decision + 1].reset_index(drop=True)
    image, _ = render_w10(window, y_pad_frac=0.05, overlay=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image):
        raise RuntimeError(f"failed to write {out_path}")
    return {
        "sample_id": f"hn_{row['event_id']}",
        "source_kind": "owner_confirmed_false_fire",
        "label": "NO_SIGNAL",
        "split": str(split_for_time(actual_time)),
        "symbol": str(row["symbol"]),
        "decision_time": actual_time.isoformat(),
        "decision_bar": decision,
        "window_start_bar": start,
        "window_end_exclusive_bar": decision + 1,
        "visible_end_bar": decision,
        "core_start_bar": None,
        "core_end_exclusive_bar": None,
        "dependency_event_id": str(row["event_id"]),
        "source_review_manifest": str(row["source_review_manifest"]),
        "source_path": str(source),
        "source_causal_input_sha256": str(row.get("causal_input_sha256") or ""),
        "owner_decision": "hard_negative",
        "image_path": str(out_path),
        "image_sha256": sha256_file(out_path),
        "future_used_in_model_input": False,
        "holdout_read": False,
    }


def _source_sha(path: Path) -> str:
    return sha256_file(path)


def build(*, sheet: Path, positives_path: Path, hard_sources: Sequence[Path], out: Path) -> dict:
    """Build into a fresh directory and return the frozen dataset manifest."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {out}")
    short_ids = load_gold500_short_ids(sheet)
    all_positives = _read_jsonl(positives_path)
    joined = select_positive_rows(all_positives, short_ids)
    geometries = [_positive_geometry(row) for row in joined]
    positive_rows, dependency_drops = _collapse_dependencies(geometries)
    hard_rows = load_confirmed_hard_negatives(hard_sources)
    protected = _owner_protected_intervals(all_positives)
    for row in hard_rows:
        _assert_not_owner_protected(row, protected)

    source_counts = Counter(str(Path(row["source_review_manifest"])) for row in hard_rows)
    if len(short_ids) != 257 or len(joined) != 256 or len(positive_rows) != 244:
        raise ValueError(
            "gold500 lineage drift: "
            f"short_ids={len(short_ids)} joined={len(joined)} dependency_events={len(positive_rows)}"
        )
    if len(hard_rows) != 584:
        raise ValueError(f"Owner hard-negative count drift: {len(hard_rows)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.building-", dir=out.parent) as temp_name:
        temp = Path(temp_name)
        rendered: list[dict] = []
        purged: list[dict] = []
        for row in positive_rows:
            split = row["split_w10"]
            if split is None:
                purged.append(
                    {
                        "sample_id": row["sample_id"],
                        "source_kind": "gold500_owner_short",
                        "decision_time": row["decision_time_w10"],
                        "reason": "split_boundary_purge",
                    }
                )
                continue
            target = temp / split / "SIGNAL" / f"pos_{row['sample_id']}.png"
            rendered.append(_render_positive(row, target))
        for row in hard_rows:
            split = split_for_time(row["decision_time"])
            if split is None:
                purged.append(
                    {
                        "sample_id": row["event_id"],
                        "source_kind": "owner_confirmed_false_fire",
                        "decision_time": row["decision_time"],
                        "reason": "split_boundary_purge",
                    }
                )
                continue
            target = temp / split / "NO_SIGNAL" / f"hn_{row['event_id']}.png"
            rendered.append(_render_negative(row, target))

        identities = [(row["source_kind"], row["dependency_event_id"]) for row in rendered]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate rendered dependency event")
        image_shas = [row["image_sha256"] for row in rendered]
        if len(image_shas) != len(set(image_shas)):
            duplicates = len(image_shas) - len(set(image_shas))
            raise ValueError(f"duplicate rendered image SHA: {duplicates}")
        by_sha_split: dict[str, set[str]] = defaultdict(set)
        for row in rendered:
            by_sha_split[row["image_sha256"]].add(row["split"])
        cross_split = [sha for sha, splits in by_sha_split.items() if len(splits) > 1]
        if cross_split:
            raise ValueError(f"cross-split image SHA: {len(cross_split)}")
        if any(row["visible_end_bar"] != row["window_end_exclusive_bar"] - 1 for row in rendered):
            raise ValueError("visible-end/decision mismatch")
        if any(pd.Timestamp(row["decision_time"]) >= HOLDOUT_START for row in rendered):
            raise ValueError("holdout row rendered")

        counts = Counter((row["split"], row["label"]) for row in rendered)
        for split in ("train", "val", "test"):
            for label in ("SIGNAL", "NO_SIGNAL"):
                if counts[(split, label)] < 1:
                    raise ValueError(f"empty class: {split}/{label}")
        for row in rendered:
            row["image_path"] = str(out / Path(row["image_path"]).relative_to(temp))
        write_jsonl(temp / "manifest.jsonl", rendered)
        write_jsonl(temp / "purged.jsonl", purged)
        write_jsonl(temp / "dependency_dedup.jsonl", dependency_drops)
        (temp / "lineage").mkdir(parents=True, exist_ok=True)
        (temp / "lineage/gold500_short_box_ids.json").write_text(
            json.dumps(sorted(short_ids), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "protocol": "fixed_w10_gold500_short_ownerhn_v1_20260813",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "task": "binary_classification",
            "classes": {"0": "NO_SIGNAL", "1": "SIGNAL"},
            "direction_scope": "short_pipeline",
            "window_contract": "W10/Core4/Confirm1",
            "renderer": "yoyo.datasets.legacy_gold_migration.renderer.render_w10",
            "y_pad_frac": 0.05,
            "split": {
                "cut1": CUT1.isoformat(),
                "cut2": CUT2.isoformat(),
                "purge_bars_each_side": PURGE_BARS,
                "random_split": False,
            },
            "counts": {f"{split}/{label}": counts[(split, label)] for split in ("train", "val", "test") for label in ("SIGNAL", "NO_SIGNAL")},
            "n_images": len(rendered),
            "n_purged": len(purged),
            "source_audit": {
                "direction_review_total": 2525,
                "gold500_sample": 500,
                "gold500_short_box_ids": len(short_ids),
                "gold500_short_ids_covered": len(short_ids),
                "mapped_positive_rows": len(joined),
                "positive_dependency_events": len(positive_rows),
                "positive_dependency_rows_dropped": len(dependency_drops),
                "owner_confirmed_hard_negatives": len(hard_rows),
                "hard_negatives_by_source": dict(sorted(source_counts.items())),
            },
            "gates": {
                "holdout_rows": 0,
                "future_input_rows": 0,
                "owner_positive_guard_overlaps": 0,
                "duplicate_dependency_events": 0,
                "duplicate_image_sha": 0,
                "cross_split_image_sha": 0,
                "visible_end_le_decision": len(rendered),
                "model_proposed_positive_boxes": 0,
            },
            "sources": {
                "direction_sheet": str(sheet),
                "direction_sheet_sha256": _source_sha(sheet),
                "owner_positive_manifest": str(positives_path),
                "owner_positive_manifest_sha256": _source_sha(positives_path),
                "hard_negative_manifests": [
                    {"path": str(path), "sha256": _source_sha(path)} for path in hard_sources
                ],
            },
            "code": {
                "yoyo_commit": git_head(ROOT),
                "fable_commit": git_head(FABLE),
                "builder": str(Path(__file__).resolve()),
                "builder_sha256": sha256_file(Path(__file__).resolve()),
            },
            "training_eligible": True,
            "auto_promote": False,
            "holdout_read": False,
        }
        (temp / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest["manifest_jsonl_sha256"] = sha256_file(temp / "manifest.jsonl")
        manifest["dataset_manifest_pre_sha256"] = sha256_file(temp / "dataset_manifest.json")
        (temp / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temp.rename(out)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = build(
        sheet=args.sheet,
        positives_path=args.positives,
        hard_sources=HARD_NEGATIVE_SOURCES,
        out=args.out,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
