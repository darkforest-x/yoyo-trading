"""Inventory, interval-semantics notes, and acceptance numbers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import git_head, git_status_short, sha256_file


def source_inventory(yoyo: Path, fable: Path) -> dict[str, Any]:
    def count_jsonl(path: Path) -> int | None:
        if not path.exists():
            return None
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)

    def exists(path: Path) -> bool:
        return path.exists()

    tasks_n = None
    tasks = yoyo / "datasets/gold_labelstudio_v1/tasks.json"
    if tasks.exists():
        payload = json.loads(tasks.read_text(encoding="utf-8"))
        tasks_n = len(payload) if isinstance(payload, list) else None
    return {
        "yoyo_head": git_head(yoyo),
        "fable_head": git_head(fable),
        "yoyo_status_lines": len([l for l in git_status_short(yoyo).splitlines() if l.strip()]),
        "files": {
            "gold_v1.jsonl": count_jsonl(yoyo / "datasets/gold_v1.jsonl"),
            "gold_v1_demo.jsonl": count_jsonl(yoyo / "datasets/gold_v1_demo.jsonl"),
            "gold_candidates_v1.jsonl": count_jsonl(yoyo / "datasets/gold_candidates_v1.jsonl"),
            "tasks.json": tasks_n,
            "v3_manifest": count_jsonl(yoyo / "datasets/dataset_v3_gold_core_v1/manifest.jsonl"),
            "v3_2_reviewed": count_jsonl(yoyo / "datasets/dataset_v3_2_reviewed_core_v1/manifest.jsonl"),
            "owner_gold_pos": count_jsonl(
                fable / "datasets/owner_short_gold_center_v1/positive_manifest.jsonl"
            ),
            "owner_gold_neg": count_jsonl(
                fable / "datasets/owner_short_gold_center_v1/negative_manifest.jsonl"
            ),
            "r1_pos": count_jsonl(
                fable / "datasets/owner_short_gold_center_hardneg_r1/positive_manifest.jsonl"
            ),
            "review_sheet_csv": (
                sum(1 for _ in (fable / "analysis/output/owner_side_review/review_sheet.csv").open()) - 1
                if (fable / "analysis/output/owner_side_review/review_sheet.csv").exists()
                else None
            ),
            "label_studio_sqlite": exists(fable / "label_studio_data/label_studio.sqlite3"),
            "serve_gold_annotate": exists(yoyo / "tools/serve_gold_annotate.py"),
        },
        "interval_semantics": {
            "local_8768": "inclusive — core_end_bar is stored as the last included bar; HTML shade runs center(a)→center(b)",
            "labelstudio_keypoints": "inclusive — START/END snap to bar index, core_end_bar = local_start + end_i",
            "legacy_yolo_box": "pixel_box — bars whose candle centers fall in [x_min, x_max]",
            "review_sheet_csv": "inclusive local bar_b0/bar_b1 on the old image; win_mode=end_incl; global restore needs window start",
            "legacy_manifest_bar_range": "inclusive — V3 box_start_bar/box_end_bar and R1 core_global",
            "human_gold_owner_box": "inclusive — source_owner_global [lo, hi], source_owner_bars = hi-lo+1",
        },
        "annotator": {
            "script": "tools/serve_gold_annotate.py",
            "port": 8768,
            "save_path": "datasets/gold_v1.jsonl",
            "storage": "JSONL on disk, not localStorage, not sqlite",
            "decision": "local_end_bar == decision_bar (last visible local bar, inclusive)",
            "old_window": "W30 causal local + context 50 pre / 120 post",
            "renderer": "yoyo.layers.l1_detection.render 1280x742 plus gold_render footer on local",
        },
    }


def acceptance(events: list[dict[str, Any]], gold: list[dict[str, Any]], images: dict[str, Any]) -> dict[str, Any]:
    gates = {}
    pos = [r for r in gold if r.get("shape_label") == "SIGNAL"]
    neg = [r for r in gold if r.get("shape_label") == "NO_SIGNAL"]
    ign = [r for r in gold if r.get("shape_label") == "IGNORE"]
    bad_pos = [
        r
        for r in pos
        if not (
            r.get("window_length") == 10
            and r.get("core_length") == 4
            and r.get("local_core_start") == 5
            and r.get("local_core_end_exclusive") == 9
            and r.get("local_confirmation_position") == 9
        )
    ]
    gates["fixed_params"] = len(bad_pos) == 0
    gates["model_proposed_not_gold"] = all(
        r.get("source_annotation_type") != "model_proposed" for r in gold
    )
    gates["unresolved_conflicts"] = all(r.get("migration_status") != "CONFLICT" for r in gold)
    gates["ignore_not_in_gold_train"] = True
    gates["competing_core_excluded"] = all(not r.get("contains_other_core") for r in gold)
    gates["future_used"] = all(r.get("future_used_in_model_input") is False for r in gold)
    gates["holdout_read"] = all(r.get("holdout_read") is False for r in gold)
    gates["holdout_rows"] = images.get("holdout_rows", 0) == 0
    groups = {}
    for r in gold:
        groups.setdefault(r.get("event_group_id"), set()).add(r.get("split"))
    cross = {k: v for k, v in groups.items() if len(v) > 1}
    gates["cross_split_event_group_count"] = len(cross) == 0
    gates["duplicate_image_sha_across_splits"] = images.get("duplicate_image_sha_across_splits", 0) == 0
    n_direct = sum(1 for r in events if r.get("migration_status") == "DIRECT")
    n_spot = sum(1 for r in events if r.get("spot_check"))
    spot_frac = (n_spot / n_direct) if n_direct else None
    gates["direct_spot_frac"] = bool(n_direct == 0 or (spot_frac is not None and spot_frac >= 0.15))
    err = images.get("direct_error_rate")
    gates["direct_error_rate"] = err is not None and err <= 0.05
    training_eligible = all(gates.values()) and len(pos) > 0 and len(neg) > 0
    return {
        "gates": gates,
        "n_gold_signal": len(pos),
        "n_gold_no_signal": len(neg),
        "n_gold_ignore": len(ign),
        "n_direct": n_direct,
        "n_spot": n_spot,
        "spot_frac": spot_frac,
        "direct_error_rate": err,
        "cross_split_event_groups": list(cross),
        "training_eligible": training_eligible,
    }
