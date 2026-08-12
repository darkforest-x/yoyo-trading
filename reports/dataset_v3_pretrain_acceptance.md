# Dataset V3 Gold Core — 训练前验收（V5 §6.8 / DATASET_CONTRACT §9）

日期：2026-08-13  
数据集：`datasets/dataset_v3_gold_core_v1`  
清单 SHA-256：`db9ca3eef7b1900b932af208da03bf267be46037a750c5bff9f7ff943b9d80b3`  
窗口协议（Owner 锁定）：`owner_gold_center_crop_w12_19`  
**门禁结论：未通过。按文档不得把本集当作已验收金标。**

训练已经在未过门的情况下发生。本页不追溯取消那些 run，但之后不得再用「V3 已验收」当理由开新训或 promote。

## 一页数字

| 项 | 值 |
|---|---|
| 正 / 负 / ignore（进入训练集） | 1345 / 2108 / **0**（ignore 未入 Gold Core） |
| 正负比 train | 1 : 1.669（在 1:1～1:2 内） |
| hard negative（shape NO） | 765 / 2108 ≈ 36%（建议约 50%，偏低） |
| human_gold / rule_verified / model_proposed | 训练金标框全是 **rule_verified 1345**；human_gold 未进入本集 |
| 旧样本 KEEP_POS / KEEP_NEG / IGNORE / REBOX / RELABEL | 1345 / 2108 / 1116 / 235 / 1370（见 `manifests/legacy_label_migration_v3.json`） |
| 重复审核一致率 | **75.3%**（243 张；未来辅助 + 过滤后为主） |
| Cohen's kappa | **0.50**（YES/NO；未过约 90% 门） |
| 每 split 抽查错标率 | **未做** |
| 时间覆盖 | 2025-06 … 2026-04（5 月仅 5 条，均 < holdout） |
| 币种 | 215 |
| 框位置 | **中 1345 / 左 0 / 右 0**（center-crop 已知代价，合同 §4 本轮不修） |
| 跨 split 重复 event_id | 0 |
| 未来泄漏 / holdout 行 | 0 / 0 |
| `visible_end_bar <= decision_bar` | 3453/3453 |
| lineage 缺口 | `source_commit` 全缺；`anchor_bar` 全缺；`config_sha256` 全缺；负例无 `event_id`（2688） |

## 门

- 机器门（泄漏、跨 split、因果收口）：过。
- 人工门（§6.3 分层 10%–15% 盲重审 + kappa；§6.8 错标率）：**不过**。
- 合同 §4：本轮必须保持 center-crop，位置捷径只许汇报。**V3.1 改裁剪属于下一轮单变量，不是本轮补丁。**

一致率 / kappa 只有 Owner 能做。清单见 `manifests/v3_blind_review_sample_v1.jsonl`（10%、seed=0、按 split×正负分层）。
