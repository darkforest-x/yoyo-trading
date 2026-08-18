"""Turn resolved source clusters into half-open W10 migration candidates."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from yoyo.datasets.gold_render import load_causal_prefix
from yoyo.datasets.window_render import SourceError

from .canonical import base_event, gold_id_from_time, is_holdout, no_signal_slot, w10_fields
from .conflicts import cluster_records, resolve_cluster
from .intervals import core_length, suggest_four_bar
from .io import git_head, write_jsonl
from .sources import SourceRecord, load_all_sources


def _legacy_offset(decision_bar: int | None, core_end_exclusive: int | None) -> int | None:
    if decision_bar is None or core_end_exclusive is None:
        return None
    return int(decision_bar) - int(core_end_exclusive)


def _status_for_signal(
    length: int | None,
    offset: int | None,
    ohlc_ok: bool,
    warmup_ok: bool,
    conflict: bool,
    source_type: str,
    ambiguous: bool,
    extra: dict[str, Any],
) -> str:
    if conflict:
        return "CONFLICT"
    if extra.get("needs_rebox") or extra.get("needs_window_restore"):
        return "MANUAL_ADJUST"
    if not ohlc_ok:
        return "IGNORE"
    if length != 4 or ambiguous:
        return "MANUAL_ADJUST"
    if not warmup_ok:
        return "IGNORE"
    if offset is None:
        return "MANUAL_ADJUST"
    if offset == 0:
        return "DIRECT"
    return "ONE_CLICK_REVIEW"


def _status_for_negative(
    gold_eligible: bool,
    length: int | None,
    conflict: bool,
    ohlc_ok: bool,
    source_type: str,
) -> str:
    if conflict:
        return "CONFLICT"
    if not gold_eligible:
        return "IGNORE"
    if not ohlc_ok:
        return "IGNORE"
    if source_type in {"local_8768", "labelstudio_keypoints", "owner_review_jsonl"}:
        return "ONE_CLICK_REVIEW"
    return "IGNORE"


class FrameCache:
    def __init__(self, fable_root: Path):
        self.fable_root = fable_root
        self._frames: dict[str, tuple[int, pd.DataFrame]] = {}
        self.missing = 0

    def get(self, source_path: str | None, last_bar: int) -> pd.DataFrame | None:
        if not source_path or last_bar < 0:
            self.missing += 1
            return None
        path = Path(source_path)
        if not path.is_absolute():
            path = self.fable_root / source_path
        if not path.exists():
            self.missing += 1
            return None
        key = str(path)
        cached = self._frames.get(key)
        if cached is not None and cached[0] >= int(last_bar):
            return cached[1]
        try:
            frame = load_causal_prefix(path, int(last_bar))
        except (SourceError, FileNotFoundError, ValueError):
            self.missing += 1
            return None
        self._frames[key] = (int(last_bar), frame)
        return frame


def migrate(
    yoyo: Path,
    fable: Path,
    cfg: dict[str, Any],
    *,
    limit: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    holdout = pd.Timestamp(cfg["holdout_start"])
    warmup = int(cfg["indicator_warmup_bars"])
    records, inventory = load_all_sources(yoyo, fable, holdout)
    # Review-sheet rows without a recoverable global core stay in inventory only.
    records = [
        r
        for r in records
        if not (r.source_annotation_type == "review_sheet_csv" and r.core_start_bar is None)
    ]
    if limit:
        records = records[: int(limit)]
    groups = cluster_records(records, int(cfg["event_group_center_max_bars"]))
    groups.sort(
        key=lambda idxs: max((records[i].decision_bar or 0) for i in idxs),
        reverse=True,
    )
    cache = FrameCache(fable)
    builder = git_head(yoyo)
    source_commit = git_head(fable)
    cfg_sha = None
    out_rows: list[dict[str, Any]] = []
    conflicts_out: list[dict[str, Any]] = []
    signal_cores: list[tuple[str, int, int]] = []

    for gi, idxs in enumerate(groups):
        resolved = resolve_cluster(records, idxs)
        winner: SourceRecord = resolved["winner"]
        geometry: SourceRecord = resolved["geometry"]
        if is_holdout(winner.decision_time, holdout):
            continue
        gold_id = (
            winner.extra.get("legacy_gold_id")
            or (gold_id_from_time(winner.symbol, winner.timeframe, winner.decision_time) if winner.decision_time else f"anon_{gi}")
        )
        shape = winner.shape_label or "IGNORE"
        core_s = geometry.core_start_bar
        core_e = geometry.core_end_exclusive_bar
        length = core_length(core_s, core_e) if core_s is not None and core_e is not None else None
        decision_bar = winner.decision_bar if winner.decision_bar is not None else geometry.decision_bar
        offset = _legacy_offset(decision_bar, core_e)
        last_need = decision_bar if decision_bar is not None else (core_e or 0)
        frame = cache.get(winner.source_path or geometry.source_path, int(last_need)) if last_need else None
        ohlc_ok = frame is not None
        warmup_ok = False
        suggestion = None
        mapped = None
        if shape == "SIGNAL" and length == 4 and core_s is not None:
            mapped = w10_fields(core_s, core_e)
            warmup_ok = mapped["window_start_bar"] >= warmup
        elif shape == "SIGNAL" and core_s is not None and core_e is not None and frame is not None:
            suggestion = suggest_four_bar(frame, core_s, core_e)
            if suggestion:
                try:
                    mapped = w10_fields(suggestion["core_start_bar"], suggestion["core_end_exclusive_bar"])
                    warmup_ok = mapped["window_start_bar"] >= warmup
                except ValueError:
                    mapped = None
        elif shape == "NO_SIGNAL" and decision_bar is not None:
            mapped = no_signal_slot(int(decision_bar))
            warmup_ok = mapped["window_start_bar"] >= warmup
            core_s = core_e = length = None
        conflict = resolved["needs_owner"]
        gold_eligible = bool(winner.extra.get("gold_eligible", True))
        if shape == "SIGNAL":
            status = _status_for_signal(
                length,
                offset,
                ohlc_ok,
                warmup_ok,
                conflict,
                winner.source_annotation_type,
                geometry.ambiguous_boundary,
                winner.extra,
            )
        elif shape == "NO_SIGNAL":
            status = _status_for_negative(
                gold_eligible, length, conflict, ohlc_ok, winner.source_annotation_type
            )
        else:
            status = "CONFLICT" if conflict else "IGNORE"

        decision_rebased = bool(offset is not None and offset != 0 and shape == "SIGNAL")
        if decision_rebased and status == "DIRECT":
            status = "ONE_CLICK_REVIEW"
        owner_conf = "REQUIRED"
        if status == "DIRECT":
            owner_conf = "MIGRATED"
        elif status == "IGNORE":
            owner_conf = "NOT_REQUIRED"

        event = base_event(
            gold_id=gold_id,
            event_id=f"{winner.symbol}_{winner.timeframe}_core_{core_s if core_s is not None else decision_bar}",
            symbol=winner.symbol,
            timeframe=winner.timeframe,
            source_repo="fable-trading",
            source_commit=source_commit,
            source_dataset=winner.source_dataset,
            source_record_id=winner.source_record_id,
            source_annotation_type=winner.source_annotation_type,
            source_interval_semantics=winner.source_interval_semantics,
            shape_label=shape,
            core_start_bar=core_s if shape == "SIGNAL" else None,
            core_end_exclusive_bar=core_e if shape == "SIGNAL" else None,
            core_length=length if shape == "SIGNAL" else None,
            confirmation_bar=(mapped or {}).get("confirmation_bar"),
            decision_bar=(mapped or {}).get("decision_bar") or decision_bar,
            window_start_bar=(mapped or {}).get("window_start_bar"),
            window_end_exclusive_bar=(mapped or {}).get("window_end_exclusive_bar"),
            window_length=(mapped or {}).get("window_length"),
            slot_start_bar=(mapped or {}).get("slot_start_bar") if shape != "SIGNAL" else None,
            slot_end_exclusive_bar=(mapped or {}).get("slot_end_exclusive_bar") if shape != "SIGNAL" else None,
            migration_status=status,
            owner_confirmation=owner_conf,
            legacy_decision_offset=offset,
            decision_rebased=decision_rebased,
            suggested_core_start_bar=(suggestion or {}).get("core_start_bar"),
            suggested_core_end_exclusive_bar=(suggestion or {}).get("core_end_exclusive_bar"),
            negative_type=winner.extra.get("negative_type"),
            negative_sampling_rule=winner.extra.get("negative_sampling_rule"),
            holdout_read=False,
            future_used_in_model_input=False,
            config_sha256=cfg_sha,
            builder_commit=builder,
            source_path=winner.source_path or geometry.source_path,
            decision_time=winner.decision_time,
            cluster_size=len(idxs),
            member_ids=resolved["member_ids"],
        )
        if mapped and shape == "SIGNAL":
            event["local_core_start"] = 5
            event["local_core_end_exclusive"] = 9
            event["local_confirmation_position"] = 9
        out_rows.append(event)
        if shape == "SIGNAL" and core_s is not None and core_e is not None:
            signal_cores.append((winner.symbol, core_s, core_e))
        for c in resolved["conflicts"]:
            conflicts_out.append({"event_id": event["event_id"], "gold_id": gold_id, **c})

    # Competing cores inside a W10: another SIGNAL core overlapping pre-context or confirm.
    for event in out_rows:
        ws, we = event.get("window_start_bar"), event.get("window_end_exclusive_bar")
        if ws is None or we is None:
            continue
        slot_s = event.get("core_start_bar") or event.get("slot_start_bar")
        slot_e = event.get("core_end_exclusive_bar") or event.get("slot_end_exclusive_bar")
        for sym, cs, ce in signal_cores:
            if sym != event["symbol"]:
                continue
            if slot_s is not None and cs == slot_s and ce == slot_e:
                continue
            if ce <= ws or cs >= we:
                continue
            # Overlaps the W10. Flag if it is not the slot itself.
            event["contains_other_core"] = True
            if event["migration_status"] in {"DIRECT", "ONE_CLICK_REVIEW"} and event["shape_label"] == "NO_SIGNAL":
                event["migration_status"] = "IGNORE"
            break

    # Time split on event groups.
    from .intervals import cluster_event_groups

    group_lists = cluster_event_groups(out_rows, center_max_bars=int(cfg["event_group_center_max_bars"]))
    labeled: list[tuple[int, Any, list[int]]] = []
    for grp in group_lists:
        times = []
        for i in grp:
            dt = out_rows[i].get("decision_bar")
            ts = out_rows[i].get("decision_time") or out_rows[i].get("source_record_id")
            times.append(ts or "")
        labeled.append((min(times), min(out_rows[i].get("decision_bar") or 0 for i in grp), grp))
    labeled.sort(key=lambda x: (str(x[0]), x[1]))
    n = max(len(labeled), 1)
    q_train, q_val = cfg["split_quantiles"]
    for rank, (_, _, grp) in enumerate(labeled):
        frac = rank / n
        split = "train" if frac < q_train else ("val" if frac < q_val else "test")
        gname = f"{out_rows[grp[0]]['symbol']}_{out_rows[grp[0]]['timeframe']}_group_{min((out_rows[i].get('core_start_bar') or out_rows[i].get('decision_bar') or 0) for i in grp)}"
        for i in grp:
            out_rows[i]["split"] = split
            out_rows[i]["event_group_id"] = gname

    out_rows.sort(key=lambda r: (r.get("gold_id") or "", r.get("event_id") or ""))
    conflicts_out.sort(key=lambda r: (r.get("gold_id") or "", r.get("event_id") or ""))
    by_status = Counter(r["migration_status"] for r in out_rows)
    by_shape = Counter(r["shape_label"] for r in out_rows)
    core_len_dist = Counter()
    offset_dist = Counter()
    for r in out_rows:
        if r["shape_label"] == "SIGNAL" and r.get("core_length") is not None:
            ln = int(r["core_length"])
            core_len_dist[str(ln if ln < 7 else "7+")] += 1
        off = r.get("legacy_decision_offset")
        if off is None:
            continue
        if off < 0:
            offset_dist["neg"] += 1
        elif off >= 4:
            offset_dist["4+"] += 1
        else:
            offset_dist[str(off)] += 1
    return {
        "inventory": inventory,
        "n_source_records": len(records),
        "n_events": len(out_rows),
        "by_status": dict(by_status),
        "by_shape": dict(by_shape),
        "core_length_distribution": dict(core_len_dist),
        "decision_offset_distribution": dict(offset_dist),
        "ohlcv_missing": cache.missing,
        "holdout_rows": 0,
        "events": out_rows,
        "conflicts": conflicts_out,
        "source_records": [r.to_dict() for r in records],
        "builder_commit": builder,
        "source_commit": source_commit,
    }


def write_migration(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events = result["events"]
    write_jsonl(out_dir / "source_records.jsonl", result.get("source_records") or [])
    write_jsonl(out_dir / "migrated_candidates.jsonl", events)
    write_jsonl(out_dir / "conflicts.jsonl", result["conflicts"])
    buckets = {
        "DIRECT": "direct.jsonl",
        "ONE_CLICK_REVIEW": "one_click_review.jsonl",
        "MANUAL_ADJUST": "manual_adjust.jsonl",
        "IGNORE": "ignored.jsonl",
        "CONFLICT": "conflicts_events.jsonl",
    }
    by: dict[str, list] = {k: [] for k in buckets}
    for row in events:
        by.setdefault(row["migration_status"], []).append(row)
    write_jsonl(out_dir / "direct.jsonl", by.get("DIRECT", []))
    write_jsonl(out_dir / "one_click_review.jsonl", by.get("ONE_CLICK_REVIEW", []))
    write_jsonl(out_dir / "manual_adjust.jsonl", by.get("MANUAL_ADJUST", []))
    write_jsonl(out_dir / "ignored.jsonl", by.get("IGNORE", []))
    # conflicts.jsonl already written as pairwise diffs; also dump event rows.
    write_jsonl(out_dir / "conflict_events.jsonl", by.get("CONFLICT", []))
    summary = {k: v for k, v in result.items() if k not in {"events", "conflicts", "source_records"}}
    (out_dir / "migration_summary.json").write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
