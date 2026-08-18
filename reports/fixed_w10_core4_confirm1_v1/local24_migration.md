# 8768 24 条无损迁移证明

保存文件：`yoyo-trading/datasets/gold_v1.jsonl`（JSONL，不是 localStorage / sqlite）。

| 项 | 值 |
|---|---|
| 原条数 | 24 |
| 迁入 `migrated_candidates.jsonl` | 24/24 |
| 丢失 gold_id | 0 |
| 胜出来源 | 全部 `local_8768`（优先级 1） |
| 23 条 POSITIVE | 闭区间长度 7（20 条，local 22–28）或 5（3 条，local 22–26）→ 半开后不是 4 根 → **MANUAL_ADJUST** |
| 1 条 IGNORE | `MOODENG_USDT_SWAP_15m_20250715T011500Z` → **IGNORE**；LS 同 id 的 NEGATIVE 被 KEEP_HIGHER 压住 |

未改写 `gold_v1.jsonl`。备份：`fable-trading/datasets/fixed_w10_core4_confirm1_v1/backups/gold_migration_20260813T111157Z/`。
