# V5 / 仓内协议 对照（2026-08-13）

权威：`工作/未命名文件夹/Claude提示词_…_V5.md`  
仓内落地：`docs/DATASET_CONTRACT.md`、`docs/EXPERIMENT_PROTOCOL.md`、`docs/EVALUATION_PROTOCOL.md`

## 已经按文档做对的

- 家在 `yoyo-trading`；fable 只读源。
- SHORT only，类别只有 `ma_dense_core`。
- 冻结 R1/R2 + 官方 yolo11s SHA。
- 首轮窗口按 Owner：**W12–19 center-crop**（V5 正文 W30 被合同 §4 明确降为后续臂）。
- R3A = V3 + R1 微调；R3B = V3 + yolo11s 冷启动；seed 0；增强全 0。
- 评估契约冻结：05-03 am、215 币、conf 0.25、NMS 0.7、event_gap 5。
- 未读 holdout，未 promote，未部署。

## 已经偏离文档的（停止用它们当主线）

| 偏离 | 文档 | 实际 |
|---|---|---|
| 未过 §6.3/§6.8 就训练 | 无 kappa **不得训练** | R3A/R3B/V3.1 都训了；清单曾标 `training_eligible: true` |
| 首轮改了窗口/裁剪 | 合同 §4、实验 §7：本轮 center-crop 不动；位置捷径只报不修 | 做了 V3.1 并开训 |
| 首轮对照未完成就开下一变量 | 实验 §7、V5 §10 第 12 步之后才下一刀 | V3.1 插在 R3B canary 之前 |
| 静态评估不全 | 评估 §1：L/中/右、FP/1000、event recall | 只报了训练 mAP |
| 把 V3.1 当成本轮主臂 | 本轮主臂是 V3 上的 R3A vs R3B | 对话里把 V3.1 推成主线 |

R3B 的 `optimizer=auto`（lr≈0.002）对 R3A 的 AdamW 1e-4 不是同一超参。文档要求两臂相同配方。已在 R3B 清单注明，不改写历史。

## 从现在起的主线（只做这些）

1. 把 V3 训练前验收标成 **未过门**（本文件 + `reports/dataset_v3_pretrain_acceptance.md`）。
2. V3.1 降为 **下一轮候选**，不参与本轮接受/拒绝。已在跑的 V3.1 canary 不杀掉（避免浪费），结果只作附录。
3. 本轮必须完成：R3B 同口径 canary + R1/R3A/R3B 对照表 + 合同要求的静态项（能算的先算）。
4. Owner 做盲审 kappa 之前：**不开新训练、不 promote**。
5. 正式晋升（≥3 seed、独立块）在快速筛选有优胜且 kappa 过门之后才启动。
