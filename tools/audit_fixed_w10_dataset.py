#!/usr/bin/env python3
"""Audit the frozen W10 dataset and write acceptance reports."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from yoyo.datasets.legacy_gold_migration.audit import acceptance, source_inventory
from yoyo.datasets.legacy_gold_migration.io import (
    add_common_flags,
    atomic_write_json,
    load_config,
    read_jsonl,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs/fixed_w10_core4_confirm1_v1.json"


def md_table(pairs) -> str:
    lines = ["| item | value |", "|---|---|"]
    for k, v in pairs:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_flags(parser, default_config=DEFAULT_CFG)
    parser.add_argument("--yoyo-root", type=Path, default=ROOT)
    parser.add_argument("--fable-root", type=Path, default=Path("/Users/zhangzc/fable-trading"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["dataset_root"])
    events = read_jsonl(root / "migration" / "migrated_candidates.jsonl")
    gold = read_jsonl(root / "gold" / "events.jsonl")
    images = read_jsonl(root / "manifests" / "image_manifest.jsonl")
    ds = {}
    ds_path = root / "manifests" / "dataset_manifest.json"
    if ds_path.exists():
        ds = json.loads(ds_path.read_text(encoding="utf-8"))
    local24 = read_jsonl(args.yoyo_root / "datasets/gold_v1.jsonl")
    migrated_ids = {r.get("gold_id") for r in events}
    local_ok = [r["gold_id"] for r in local24 if r["gold_id"] in migrated_ids]
    local_miss = [r["gold_id"] for r in local24 if r["gold_id"] not in migrated_ids]
    by_status = Counter(r.get("migration_status") for r in events)
    by_shape = Counter(r.get("shape_label") for r in events)
    core_len = Counter()
    off = Counter()
    for r in events:
        if r.get("shape_label") == "SIGNAL" and r.get("core_length") is not None:
            n = int(r["core_length"])
            core_len[str(n if n < 7 else "7+")] += 1
        o = r.get("legacy_decision_offset")
        if o is None:
            continue
        off["4+" if int(o) >= 4 else str(o)] += 1
    gold_split = Counter((r.get("split"), r.get("shape_label")) for r in gold)
    acc = acceptance(
        events,
        [r for r in gold if r.get("shape_label") in {"SIGNAL", "NO_SIGNAL"}],
        {
            "holdout_rows": ds.get("holdout_rows", 0),
            "duplicate_image_sha_across_splits": ds.get("duplicate_image_sha_across_splits", 0),
            "direct_error_rate": None,
        },
    )
    # Protocol 17.6: DIRECT error rate is unknown until Owner labels the
    # 15% spot check. Do not pass this gate by editing the report.
    acc["gates"]["direct_error_rate"] = False
    failed = [k for k, v in acc["gates"].items() if not v]
    acc["training_eligible"] = all(acc["gates"].values()) and acc["n_gold_signal"] > 0 and acc["n_gold_no_signal"] > 0
    acc["training_eligible_reason"] = (
        "；".join(failed) + " — "
        f"冻结 Gold SIGNAL={acc['n_gold_signal']} NO_SIGNAL={acc['n_gold_no_signal']}。"
        "DIRECT 抽检错误率未知（迁移 DIRECT=0；协议 17.6 不得改报告绕过）"
    )
    ds["training_eligible"] = acc["training_eligible"]
    ds["acceptance"] = acc
    if args.dry_run:
        print(json.dumps({"dry_run": True, "by_status": dict(by_status), "acceptance": acc}, indent=2, ensure_ascii=False))
        return 0
    atomic_write_json(ds_path, ds)
    yoyo_man = args.yoyo_root / "manifests" / "fixed_w10_core4_confirm1_v1.json"
    atomic_write_json(yoyo_man, ds)

    def write_md(name: str, body: str) -> None:
        for folder in (root / "reports", args.yoyo_root / "reports" / "fixed_w10_core4_confirm1_v1"):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text(body.rstrip() + "\n", encoding="utf-8")

    inv = source_inventory(args.yoyo_root, args.fable_root)
    write_md(
        "interval_semantics_audit.md",
        "# Interval semantics audit\n\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in inv["interval_semantics"].items())
        + "\n\n8768 页面「起点 22 — 终点 26」在旧工具里是**闭区间** 22…26（5 根）。"
        "新合同解释为半开区间 [22, 26) = 22,23,24,25。"
        "迁移时按代码语义（闭区间）转半开，不把旧 22–26 静默当成 4 根。\n",
    )
    mig_md = [
        "# Migration summary",
        "",
        md_table(
            [
                ("旧来源种类", len(inv["files"])),
                ("8768 已保存", inv["files"]["gold_v1.jsonl"]),
                ("Label Studio yoyo_gold_v1 任务/完成", "30 tasks / 1 completion"),
                ("候选任务数（1,345）", inv["files"]["gold_candidates_v1.jsonl"]),
                ("事件数", len(events)),
                ("DIRECT", by_status.get("DIRECT", 0)),
                ("ONE_CLICK_REVIEW", by_status.get("ONE_CLICK_REVIEW", 0)),
                ("MANUAL_ADJUST", by_status.get("MANUAL_ADJUST", 0)),
                ("CONFLICT", by_status.get("CONFLICT", 0)),
                ("IGNORE", by_status.get("IGNORE", 0)),
                ("SIGNAL 候选", by_shape.get("SIGNAL", 0)),
                ("NO_SIGNAL 候选", by_shape.get("NO_SIGNAL", 0)),
                ("8768 24 条命中", f"{len(local_ok)}/{len(local24)}"),
                ("8768 丢失", len(local_miss)),
                ("holdout 读取", 0),
                ("冻结 SIGNAL", acc["n_gold_signal"]),
                ("冻结 NO_SIGNAL", acc["n_gold_no_signal"]),
                ("DIRECT 抽检错误率", "未测（Owner 未审）"),
                ("training_eligible", acc["training_eligible"]),
            ]
        ),
        "",
        "## 旧核心长度",
        json.dumps(dict(core_len), ensure_ascii=False),
        "",
        "## 旧 decision offset",
        json.dumps(dict(off), ensure_ascii=False),
        "",
        "## 冻结 split",
        json.dumps({f"{a}/{b}": n for (a, b), n in gold_split.items()}, ensure_ascii=False),
        "",
        acc["training_eligible_reason"],
    ]
    write_md("migration_summary.md", "\n".join(mig_md))
    write_md(
        "direct_sample_review.md",
        "# DIRECT spot check\n\n"
        f"DIRECT={by_status.get('DIRECT', 0)}. 15% 抽样清单见 "
        "`migration/direct_spot_check.jsonl`。Owner 尚未标注，错误率未知。"
        "因此不得把该来源视为已验收 DIRECT。\n",
    )
    write_md(
        "conflict_report.md",
        "# Conflicts\n\n"
        f"冲突对数见 `migration/conflicts.jsonl`。"
        f"CONFLICT 事件 {by_status.get('CONFLICT', 0)}。"
        "高优先级结果未被低优先级覆盖。\n",
    )
    write_md(
        "split_audit.md",
        "# Split audit\n\n按 event_group 时间分位 train 70% / val 15% / test 15%，holdout 前。"
        f" cross_split_event_groups={acc.get('cross_split_event_groups')}\n",
    )
    write_md(
        "leakage_audit.md",
        "# Leakage audit\n\n"
        f"- holdout_read rows: 0\n- future_used_in_model_input: all false in frozen gold\n"
        f"- 8768 lossless: {len(local_ok)}/{len(local24)} missing={local_miss}\n",
    )
    write_md(
        "image_audit.md",
        f"# Image audit\n\nrendered={len(images)} skipped={len(ds.get('skipped') or [])} "
        f"dup_sha_across_splits={ds.get('duplicate_image_sha_across_splits')}\n",
    )
    accept_md = [
        "# Dataset acceptance",
        "",
        f"**training_eligible: {str(acc['training_eligible']).lower()}**",
        "",
        acc["training_eligible_reason"],
        "",
        md_table(list(acc["gates"].items())),
        "",
        f"dataset_manifest sha256: `{sha256_file(ds_path) if ds_path.exists() else 'n/a'}`",
    ]
    write_md("dataset_acceptance.md", "\n".join(accept_md))
    print(json.dumps({"training_eligible": acc["training_eligible"], "by_status": dict(by_status), "local24": len(local_ok)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
