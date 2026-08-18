# Batch accept suggested 4-bar cores

Owner 2026-08-13：「全部接受」= 接受迁移 `suggest_four_bar`，写入复审 state。**未 freeze。**

| 项 | 数 |
|---|---:|
| 新接受 A | 1,233 |
| 保留已有 4 根核 A | 34 |
| 纠正误 A → NO_SIGNAL | 3 |
| 跳过 CONFLICT | 44 |
| 跳过无建议核 SIGNAL | 10 |
| 跳过其余 NO_SIGNAL | 140 |
| state 合计 | 1,270 |
| 冻结 Gold SIGNAL / NO_SIGNAL | 0 / 0 |

纠正的 3 条（EASY_BACKGROUND，不得当正例）：

- `2Z_USDT_SWAP_15m_20251017T170000Z`
- `ACH_USDT_SWAP_15m_20251113T133000Z`
- `ADA_USDT_SWAP_15m_20251015T151500Z`

备份：`~/fable-trading/datasets/fixed_w10_core4_confirm1_v1/review/state.jsonl.bak.20260813T114549915725Z`

下一步：等 Owner 说 freeze。
