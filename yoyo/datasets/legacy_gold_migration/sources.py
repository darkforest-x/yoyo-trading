"""One adapter per legacy source. Each declares original interval semantics.

Do not interpret every start/end the same way. 8768 and Label Studio keypoints
store an inclusive end. Owner gold ``source_owner_global`` / V3 ``box_*`` /
``core_global`` are also inclusive. YOLO xywh is a pixel box.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from yoyo.datasets.gold_box import snap_x_to_bar
from yoyo.layers.l1_detection.render import IMG_WIDTH, MARGIN

from .canonical import gold_id_from_time, is_holdout, parse_time
from .intervals import inclusive_to_half_open, yolo_box_to_bars
from .io import read_jsonl

SOURCE_SEMANTICS = {
    "local_8768": "inclusive",
    "labelstudio_keypoints": "inclusive",
    "legacy_yolo_box": "pixel_box",
    "review_sheet_csv": "inclusive",
    "legacy_manifest_bar_range": "inclusive",
    "owner_review_jsonl": "inclusive",
    "human_gold_owner_box": "inclusive",
    "rule_verified_core": "inclusive",
    "model_proposed": "pixel_box",
}

PRECEDENCE = {
    "local_8768": 1,
    "labelstudio_keypoints": 2,
    "owner_review_jsonl": 3,
    "review_sheet_csv": 3,
    "human_gold_owner_box": 4,
    "rule_verified_core": 5,
    "legacy_manifest_bar_range": 5,
    "model_proposed": 6,
    "easy_negative_pool": 6,
}


@dataclass
class SourceRecord:
    source_annotation_type: str
    source_interval_semantics: str
    precedence: int
    source_dataset: str
    source_record_id: str
    symbol: str
    timeframe: str = "15m"
    source_path: str | None = None
    decision_bar: int | None = None
    decision_time: str | None = None
    shape_label: str | None = None
    core_start_bar: int | None = None
    core_end_exclusive_bar: int | None = None
    yolo_box: list | None = None
    reviewed_at: str | None = None
    holdout_skipped: bool = False
    ambiguous_boundary: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "source_annotation_type": self.source_annotation_type,
            "source_interval_semantics": self.source_interval_semantics,
            "precedence": self.precedence,
            "source_dataset": self.source_dataset,
            "source_record_id": self.source_record_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "source_path": self.source_path,
            "decision_bar": self.decision_bar,
            "decision_time": self.decision_time,
            "shape_label": self.shape_label,
            "core_start_bar": self.core_start_bar,
            "core_end_exclusive_bar": self.core_end_exclusive_bar,
            "yolo_box": self.yolo_box,
            "reviewed_at": self.reviewed_at,
            "holdout_skipped": self.holdout_skipped,
            "ambiguous_boundary": self.ambiguous_boundary,
            "gold_id": gold_id_from_time(self.symbol, self.timeframe, self.decision_time)
            if self.decision_time
            else None,
        }
        d.update(self.extra)
        return d


def _shape_from_legacy(value: Any) -> str | None:
    raw = str(value or "").upper()
    mapping = {
        "POSITIVE": "SIGNAL",
        "NEGATIVE": "NO_SIGNAL",
        "IGNORE": "IGNORE",
        "SIGNAL": "SIGNAL",
        "NO_SIGNAL": "NO_SIGNAL",
        "YES": "SIGNAL",
        "NO": "NO_SIGNAL",
        "SKIP": "IGNORE",
        "TARGET": "SIGNAL",
        "HARD_NEGATIVE": "NO_SIGNAL",
        "REBOX": "SIGNAL",
        "SHORT": "SIGNAL",
        "LONG": "IGNORE",
    }
    return mapping.get(raw, None)


def _skip_holdout(decision_time: str | None, holdout) -> bool:
    return bool(decision_time) and is_holdout(decision_time, holdout)


def load_local_8768(path: Path, holdout) -> list[SourceRecord]:
    rows = read_jsonl(path)
    out: list[SourceRecord] = []
    for row in rows:
        if _skip_holdout(row.get("decision_time"), holdout):
            continue
        shape = _shape_from_legacy(row.get("shape_label"))
        start = end = None
        if shape == "SIGNAL" and row.get("core_start_bar") is not None:
            start, end = inclusive_to_half_open(row["core_start_bar"], row["core_end_bar"])
        rec = SourceRecord(
            source_annotation_type="local_8768",
            source_interval_semantics="inclusive",
            precedence=1,
            source_dataset=str(path),
            source_record_id=row["gold_id"],
            symbol=row["symbol"],
            timeframe=row.get("timeframe", "15m"),
            source_path=row.get("source_path"),
            decision_bar=int(row["decision_bar"]),
            decision_time=row["decision_time"],
            shape_label=shape,
            core_start_bar=start,
            core_end_exclusive_bar=end,
            reviewed_at=row.get("reviewed_at"),
            extra={"legacy_gold_id": row["gold_id"], "legacy_shape": row.get("shape_label")},
        )
        out.append(rec)
    return out


def load_labelstudio_sqlite(db_path: Path, holdout, project_title: str = "yoyo_gold_v1") -> list[SourceRecord]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        proj = con.execute("SELECT id FROM project WHERE title=?", (project_title,)).fetchone()
        if not proj:
            return []
        pid = proj[0]
        out: list[SourceRecord] = []
        q = """
        SELECT t.data, tc.result, tc.updated_at
        FROM task_completion tc
        JOIN task t ON t.id = tc.task_id
        WHERE t.project_id=?
        """
        for data_s, result_s, updated in con.execute(q, (pid,)):
            data = json.loads(data_s)
            results = json.loads(result_s) if result_s else []
            if _skip_holdout(data.get("decision_time"), holdout):
                continue
            shape = None
            pts: dict[str, int] = {}
            width = int(data.get("local_image_width") or IMG_WIDTH)
            n_bars = int(data.get("local_window_length") or 30)
            for item in results:
                if item.get("from_name") == "shape":
                    choices = (item.get("value") or {}).get("choices") or []
                    if choices:
                        shape = _shape_from_legacy(choices[0])
                if item.get("from_name") == "core":
                    labels = (item.get("value") or {}).get("keypointlabels") or []
                    if labels:
                        x_pct = float(item["value"]["x"])
                        x_px = x_pct / 100.0 * width
                        pts[str(labels[0])] = snap_x_to_bar(x_px, n_bars, width, MARGIN)
            start = end = None
            if shape == "SIGNAL" and "START" in pts and "END" in pts:
                a, b = pts["START"], pts["END"]
                local_start = int(data["local_start_bar"])
                start, end = inclusive_to_half_open(local_start + a, local_start + b)
            out.append(
                SourceRecord(
                    source_annotation_type="labelstudio_keypoints",
                    source_interval_semantics="inclusive",
                    precedence=2,
                    source_dataset=f"{db_path}#{project_title}",
                    source_record_id=data.get("gold_id") or "",
                    symbol=data["symbol"],
                    timeframe=data.get("timeframe", "15m"),
                    source_path=data.get("source_path"),
                    decision_bar=int(data["decision_bar"]),
                    decision_time=data["decision_time"],
                    shape_label=shape,
                    core_start_bar=start,
                    core_end_exclusive_bar=end,
                    reviewed_at=str(updated) if updated else None,
                    extra={"legacy_gold_id": data.get("gold_id")},
                )
            )
        return out
    finally:
        con.close()


def load_labelstudio_export(path: Path, holdout) -> list[SourceRecord]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload if isinstance(payload, list) else payload.get("tasks") or []
    # Reuse sqlite-shaped conversion via a temporary wrap.
    out: list[SourceRecord] = []
    for task in tasks:
        data = task.get("data") or {}
        if _skip_holdout(data.get("decision_time"), holdout):
            continue
        anns = task.get("annotations") or []
        if not anns:
            continue
        results = anns[-1].get("result") or []
        fake = SourceRecord(
            source_annotation_type="labelstudio_keypoints",
            source_interval_semantics="inclusive",
            precedence=2,
            source_dataset=str(path),
            source_record_id=data.get("gold_id") or "",
            symbol=data["symbol"],
            timeframe=data.get("timeframe", "15m"),
            source_path=data.get("source_path"),
            decision_bar=int(data["decision_bar"]),
            decision_time=data["decision_time"],
            shape_label=None,
        )
        shape = None
        pts: dict[str, int] = {}
        width = int(data.get("local_image_width") or IMG_WIDTH)
        n_bars = int(data.get("local_window_length") or 30)
        for item in results:
            if item.get("from_name") == "shape":
                choices = (item.get("value") or {}).get("choices") or []
                if choices:
                    shape = _shape_from_legacy(choices[0])
            if item.get("from_name") == "core":
                labels = (item.get("value") or {}).get("keypointlabels") or []
                if labels:
                    x_px = float(item["value"]["x"]) / 100.0 * width
                    pts[str(labels[0])] = snap_x_to_bar(x_px, n_bars, width, MARGIN)
        fake.shape_label = shape
        if shape == "SIGNAL" and "START" in pts and "END" in pts:
            local_start = int(data["local_start_bar"])
            fake.core_start_bar, fake.core_end_exclusive_bar = inclusive_to_half_open(
                local_start + pts["START"], local_start + pts["END"]
            )
        out.append(fake)
    return out


def _inclusive_core(pair) -> tuple[int, int] | tuple[None, None]:
    if not pair or pair[0] is None or pair[1] is None:
        return None, None
    return inclusive_to_half_open(pair[0], pair[1])


def load_human_gold_owner_boxes(path: Path, holdout) -> list[SourceRecord]:
    if not path.exists():
        return []
    out: list[SourceRecord] = []
    for row in read_jsonl(path):
        dt = row.get("end_time") or row.get("source_owner_cut_time")
        if _skip_holdout(dt, holdout):
            continue
        start, end = _inclusive_core(row.get("source_owner_global"))
        decision_bar = int(row["win_end"])
        out.append(
            SourceRecord(
                source_annotation_type="human_gold_owner_box",
                source_interval_semantics="inclusive",
                precedence=4,
                source_dataset=str(path),
                source_record_id=row["sample_id"],
                symbol=row["symbol"],
                source_path=row.get("source_csv") or row.get("source_path"),
                decision_bar=decision_bar,
                decision_time=dt,
                shape_label="SIGNAL",
                core_start_bar=start,
                core_end_exclusive_bar=end,
                extra={
                    "sample_id": row["sample_id"],
                    "source_owner_bars": row.get("source_owner_bars"),
                    "rule_core_global": row.get("core_global"),
                },
            )
        )
    return out


def load_rule_verified_cores(path: Path, holdout, *, source_dataset: str | None = None) -> list[SourceRecord]:
    if not path.exists():
        return []
    out: list[SourceRecord] = []
    for row in read_jsonl(path):
        if row.get("class") not in (None, "positive") and not row.get("class_name"):
            continue
        is_pos = row.get("class") == "positive" or row.get("class_name") == "ma_dense_core"
        if not is_pos:
            continue
        dt = row.get("end_time") or row.get("decision_time")
        if _skip_holdout(dt, holdout):
            continue
        core = row.get("core_global")
        if core is None and row.get("box_start_bar") is not None:
            core = [row["box_start_bar"], row["box_end_bar"]]
        start, end = _inclusive_core(core)
        decision_bar = int(row.get("win_end") or row.get("decision_bar"))
        yolo = row.get("yolo_box")
        out.append(
            SourceRecord(
                source_annotation_type="rule_verified_core",
                source_interval_semantics="inclusive",
                precedence=5,
                source_dataset=source_dataset or str(path),
                source_record_id=row.get("sample_id") or "",
                symbol=row["symbol"],
                source_path=row.get("source_csv") or row.get("source_path"),
                decision_bar=decision_bar,
                decision_time=dt,
                shape_label="SIGNAL",
                core_start_bar=start,
                core_end_exclusive_bar=end,
                yolo_box=list(yolo) if yolo else None,
                extra={"sample_id": row.get("sample_id"), "core_bars": row.get("core_bars")},
            )
        )
    return out


def load_easy_negatives(path: Path, holdout) -> list[SourceRecord]:
    if not path.exists():
        return []
    out: list[SourceRecord] = []
    for row in read_jsonl(path):
        dt = row.get("end_time") or row.get("decision_time")
        if _skip_holdout(dt, holdout):
            continue
        out.append(
            SourceRecord(
                source_annotation_type="easy_negative_pool",
                source_interval_semantics="source_declared",
                precedence=6,
                source_dataset=str(path),
                source_record_id=row.get("sample_id") or "",
                symbol=row["symbol"],
                source_path=row.get("source_csv") or row.get("source_path"),
                decision_bar=int(row.get("win_end") or row.get("decision_bar")),
                decision_time=dt,
                shape_label="NO_SIGNAL",
                extra={
                    "negative_type": "EASY_BACKGROUND",
                    "negative_sampling_rule": row.get("selection_method"),
                    "gold_eligible": False,
                },
            )
        )
    return out


OWNER_PACKS = (
    (
        "analysis/output/owner_short_train_hardneg_review200_v1/owner_review_labeled_manifest.jsonl",
        "owner_decision",
        {"target": "SIGNAL", "hard_negative": "NO_SIGNAL", "rebox": "SIGNAL"},
    ),
    (
        "analysis/output/owner_short_train_positive_retrieval100_v1/owner_review_labeled_manifest.jsonl",
        "owner_decision",
        {"target": "SIGNAL", "hard_negative": "NO_SIGNAL", "rebox": "SIGNAL"},
    ),
    (
        "analysis/output/owner_short_train_hardneg_expansion200_v2/owner_review_labeled_manifest.jsonl",
        "owner_decision",
        {"target": "SIGNAL", "hard_negative": "NO_SIGNAL", "rebox": "SIGNAL"},
    ),
    (
        "analysis/output/owner_short_train_hardneg_newblocks200_v3/owner_review_labeled_manifest.jsonl",
        "owner_decision",
        {"target": "SIGNAL", "hard_negative": "NO_SIGNAL", "rebox": "SIGNAL"},
    ),
    (
        "analysis/output/local_signal_v2_positive_semantic_review200_v2/owner_review_joined.jsonl",
        "owner_verdict",
        {"YES": "SIGNAL", "NO": "NO_SIGNAL", "SKIP": "IGNORE"},
    ),
    (
        "analysis/output/local_signal_v2_early_frontier_review300_v1/owner_review_joined.jsonl",
        "owner_verdict",
        {"YES": "SIGNAL", "NO": "NO_SIGNAL", "SKIP": "IGNORE"},
    ),
)


def load_owner_review_packs(fable: Path, holdout) -> list[SourceRecord]:
    out: list[SourceRecord] = []
    for rel, field, mapping in OWNER_PACKS:
        path = fable / rel
        if not path.exists():
            continue
        for row in read_jsonl(path):
            dt = row.get("decision_time")
            if _skip_holdout(dt, holdout):
                continue
            raw = str(row.get(field))
            shape = mapping.get(raw) or _shape_from_legacy(raw)
            start = end = None
            if row.get("core_start_i") is not None and row.get("core_end_i") is not None:
                start, end = inclusive_to_half_open(row["core_start_i"], row["core_end_i"])
            decision_bar = row.get("decision_i") or row.get("decision_bar")
            extra = {
                "raw_verdict": raw,
                "review_id": row.get("review_id"),
                "event_id": row.get("event_id"),
                "negative_type": "MODEL_FALSE_POSITIVE" if shape == "NO_SIGNAL" else None,
            }
            if raw.lower() == "rebox":
                extra["needs_rebox"] = True
            out.append(
                SourceRecord(
                    source_annotation_type="owner_review_jsonl",
                    source_interval_semantics="inclusive",
                    precedence=3,
                    source_dataset=rel,
                    source_record_id=str(row.get("review_id") or row.get("event_id") or ""),
                    symbol=row["symbol"],
                    source_path=row.get("source_csv") or row.get("source_path"),
                    decision_bar=int(decision_bar) if decision_bar is not None else None,
                    decision_time=dt,
                    shape_label=shape,
                    core_start_bar=start,
                    core_end_exclusive_bar=end,
                    extra=extra,
                )
            )
    return out


def load_v32_reviewed(path: Path, holdout) -> list[SourceRecord]:
    if not path.exists():
        return []
    out: list[SourceRecord] = []
    for row in read_jsonl(path):
        if _skip_holdout(row.get("decision_time"), holdout):
            continue
        verdict = str(row.get("review_r1_owner") or "").upper()
        if verdict == "YES":
            shape = "SIGNAL"
        elif verdict == "NO":
            shape = "NO_SIGNAL"
        else:
            shape = "IGNORE"
        start = end = None
        if row.get("box_start_bar") is not None and row.get("box_end_bar") is not None:
            start, end = inclusive_to_half_open(row["box_start_bar"], row["box_end_bar"])
        extra = {
            "sample_id": row.get("sample_id"),
            "origin": row.get("origin"),
            "negative_type": None,
        }
        if shape == "NO_SIGNAL":
            extra["negative_type"] = (
                "MODEL_FALSE_POSITIVE" if row.get("is_hard_negative") else "EASY_BACKGROUND"
            )
        out.append(
            SourceRecord(
                source_annotation_type="owner_review_jsonl",
                source_interval_semantics="inclusive",
                precedence=3,
                source_dataset=str(path),
                source_record_id=row.get("sample_id") or "",
                symbol=row["symbol"],
                source_path=row.get("source_path"),
                decision_bar=int(row["decision_bar"]),
                decision_time=row["decision_time"],
                shape_label=shape,
                core_start_bar=start,
                core_end_exclusive_bar=end,
                extra=extra,
            )
        )
    return out


def load_review_sheet(path: Path, holdout) -> list[SourceRecord]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    out: list[SourceRecord] = []
    for row in df.itertuples(index=False):
        side = str(getattr(row, "owner_side", "")).lower()
        dt = str(getattr(row, "cut_time"))
        if _skip_holdout(dt, holdout):
            continue
        shape = _shape_from_legacy(side)
        if shape is None:
            shape = "IGNORE"
        # Local inclusive bars on the old image; global cut is the window end.
        b0 = int(getattr(row, "bar_b0"))
        b1 = int(getattr(row, "bar_b1"))
        cut = int(getattr(row, "cut_global"))
        # win_mode=end_incl: last local index maps to cut_global. Local length
        # is not stored; recover global by assuming the local end of the *box*
        # is not the window end. Join quality is weak without window length, so
        # keep local bars in extra and only emit global if width_bars matches.
        width_bars = int(getattr(row, "width_bars"))
        start = end = None
        if b1 - b0 + 1 == width_bars:
            # Still cannot map to global without window start. Leave cores empty
            # unless we later join on box_id == sample_id in human_gold.
            pass
        yolo = [
            float(getattr(row, "yolo_xc")),
            float(getattr(row, "yolo_yc")),
            float(getattr(row, "yolo_w")),
            float(getattr(row, "yolo_h")),
        ]
        out.append(
            SourceRecord(
                source_annotation_type="review_sheet_csv",
                source_interval_semantics="inclusive",
                precedence=3,
                source_dataset=str(path),
                source_record_id=str(getattr(row, "box_id")),
                symbol=str(getattr(row, "symbol")),
                decision_bar=cut,
                decision_time=dt,
                shape_label=shape,
                core_start_bar=start,
                core_end_exclusive_bar=end,
                yolo_box=yolo,
                extra={
                    "local_b0": b0,
                    "local_b1": b1,
                    "width_bars": width_bars,
                    "owner_side": side,
                    "win_mode": getattr(row, "win_mode", None),
                    "needs_window_restore": True,
                },
            )
        )
    return out


def load_final_labels(path: Path, holdout) -> list[SourceRecord]:
    if not path.exists():
        return []
    out: list[SourceRecord] = []
    for row in read_jsonl(path):
        if _skip_holdout(row.get("decision_time"), holdout):
            continue
        shape = _shape_from_legacy(row.get("final"))
        out.append(
            SourceRecord(
                source_annotation_type="owner_review_jsonl",
                source_interval_semantics="inclusive",
                precedence=3,
                source_dataset=str(path),
                source_record_id=row.get("sample_id") or "",
                symbol=row["symbol"],
                decision_bar=None,
                decision_time=row.get("decision_time"),
                shape_label=shape,
                extra={"sample_id": row.get("sample_id"), "via": row.get("via")},
            )
        )
    return out


def load_all_sources(yoyo: Path, fable: Path, holdout) -> tuple[list[SourceRecord], dict[str, Any]]:
    groups: dict[str, list[SourceRecord]] = {
        "local_8768": load_local_8768(yoyo / "datasets/gold_v1.jsonl", holdout),
        "labelstudio_yoyo_gold_v1": load_labelstudio_sqlite(
            fable / "label_studio_data/label_studio.sqlite3", holdout
        ),
        "v3_2_reviewed": load_v32_reviewed(
            yoyo / "datasets/dataset_v3_2_reviewed_core_v1/manifest.jsonl", holdout
        ),
        "v3_dispute_final_labels": load_final_labels(
            yoyo / "reviews/v3_dispute_r1/final_labels.jsonl", holdout
        ),
        "owner_review_packs": load_owner_review_packs(fable, holdout),
        "human_gold_owner_box": load_human_gold_owner_boxes(
            fable / "datasets/owner_short_gold_center_v1/positive_manifest.jsonl", holdout
        ),
        "rule_verified_r1": load_rule_verified_cores(
            fable / "datasets/owner_short_gold_center_hardneg_r1/positive_manifest.jsonl",
            holdout,
            source_dataset="owner_short_gold_center_hardneg_r1",
        ),
        "rule_verified_v3": load_rule_verified_cores(
            yoyo / "datasets/dataset_v3_gold_core_v1/manifest.jsonl",
            holdout,
            source_dataset="dataset_v3_gold_core_v1",
        ),
        "easy_negatives_r1": load_easy_negatives(
            fable / "datasets/owner_short_gold_center_v1/negative_manifest.jsonl", holdout
        ),
        "review_sheet_csv": load_review_sheet(
            fable / "analysis/output/owner_side_review/review_sheet.csv", holdout
        ),
    }
    inventory = {name: len(rows) for name, rows in groups.items()}
    flat: list[SourceRecord] = []
    for rows in groups.values():
        flat.extend(rows)
    return flat, inventory
