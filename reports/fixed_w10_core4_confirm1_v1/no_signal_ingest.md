# NO_SIGNAL 负例写入复审 state

Owner 2026-08-13 授权补负例。**未 freeze、未开训、未 commit。** Gold 仍为 **0 / 0**。

## 结果

| 项 | 数 |
|---|---:|
| 新收 N | **1,395** |
| 其中易负例池 | 1,256 |
| 其中 Owner 已确认无核（ONE_CLICK_REVIEW） | 139 |
| 保留已有 N | 3 |
| 未改已有 A | 1,269 |
| 现在 A / N | **1,269 / 1,398** |
| 跳过 holdout | **0** |
| 跳过 CONFLICT | **44**（43 未写入；1 条已在 A，未改） |
| 跳过窗口内另有 SIGNAL 核 | 103 |
| 跳过无 K 线路径 | 666 |
| 跳过 warmup 不足 | 2 |
| 冻结 SIGNAL / NO_SIGNAL | **0 / 0** |

写入 `~/fable-trading/datasets/fixed_w10_core4_confirm1_v1/review/state.jsonl`。  
备份 `state.jsonl.bak.20260813T120319413033Z`。  
未改 `datasets/gold_v1.jsonl`，未改 `render.py`。

## 来源

1. **易负例池** `owner_short_gold_center_v1/negative_manifest.jsonl`（1,343）。迁移后仍以 `easy_negative_pool` 胜出、且可渲染的 **1,256** 条。核为空。
2. **候选层已标 NO_SIGNAL** 且 Owner 确认无密集核心：`ONE_CLICK_REVIEW` **139**（V3.2 已审 129 + Owner semantic pack 14，去掉 4 条已在 A）。`gold_candidates_v1.jsonl` 全是正例候选，没有 NO_SIGNAL 标签，未用。
3. **已有 3 条 N** 原样保留（ACH 20250831、ADA 20250723 两张）。

未把 CONFLICT、未审 model-mined hard negative、Owner-long 镜像、或任何 SIGNAL 洗成负例。

## 易负例池的坑

- 规则采样：同币种、同 split、随机落在所有 Owner 框 + 12 bar guard 之外。**没有 Owner 目视。**
- 原池 `production_eligible=false`；W10 迁移故意 `gold_eligible=false` → 标成 IGNORE，**不是**自动 Gold。本次是 Owner 授权才收进 state。
- `later_return_used` / `model_score_used` 全 false；V3 审计 1,343 条都是 `KEEP_NEGATIVE`。
- 原图是 W12–19，本任务用 decision 切 **最后 10 根**。框外背景 ≠ W10 槽里绝对没有均线密集。
- 85 条与更高优先级来源撞 `gold_id`：76 条变成 Owner ONE_CLICK NO（走桶 2），5 SIGNAL / 5 CONFLICT / 4 IGNORE 未当负例。

## 未改的 5 条无核 A

state 里仍有 5 条 A、核为空（队列标签其实是 NO_SIGNAL）。本次按「已接受 A 不改」留下：

- `2Z_USDT_SWAP_15m_20251017T170000Z`
- `ACH_USDT_SWAP_15m_20251113T133000Z`
- `ADA_USDT_SWAP_15m_20251015T151500Z`
- `ADA_USDT_SWAP_15m_20251225T101500Z`
- `AERO_USDT_SWAP_15m_20250715T011500Z`

前 3 条曾在 batch accept 时改成 N，后来又出现在 A 里。freeze 前 Owner 应决定是改回 N 还是丢掉。

## 下一步

已 freeze（见 `freeze.md`）。不要开训，等 Owner 说开训。
