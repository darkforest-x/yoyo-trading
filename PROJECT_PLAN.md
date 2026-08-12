# PROJECT_PLAN — L1 Dataset V3 与 R3

目标（Owner V5 简报，2026-08-12）：**数据集准确** + **模型训练准确**。
其余一切（目录美化、全量迁移、文档扩写）不得阻塞这两件事。

## 阶段

| # | 阶段 | 交付 | 状态 |
|---|---|---|---|
| 1 | 审计两仓 + 冻结 R1/R2 基线 | `manifests/frozen_baseline_r1_r2_v1.json`、`docs/ASSET_REUSE_MAP.md` | ✅ 2026-08-12 |
| 2 | 最低项目规范 | CLAUDE/AGENTS/PROJECT_PLAN/HANDOFF + 6 份 docs | ✅ 2026-08-12 |
| 3 | 最小 L1 链路 + smoke parity | 20–50 固定样本逐字节复现 | ✅ 45/45 |
| 4 | 旧标签审计 | 迁移表 + shape/outcome 分离 + `class_status`/`box_status` | ✅ 表在；人工复审未做 |
| 5 | Dataset V3 Gold Core | 训练前验收（一致率/kappa/错标率/覆盖/泄漏） | **机器过 / 人工门未过** |
| 6 | R3A / R3B 快速筛选 | 两条臂，同数据同配方同 seed | 已训完；**在未过 kappa 门下训的** |
| 7 | 静态评估 + 连续回放 | 与 R1 同口径；R3B canary 未完 | 进行中 |
| 8 | 优胜者 3 seed + 独立块复验 | 接受 / 拒绝裁决 | 未开始 |

## 已锁定的决定

- **窗口**：沿用 R1 的 W12–19 动态 center-crop（Owner 2026-08-12）。首轮只改数据。
- **类别**：唯一 `ma_dense_core`，SHORT pipeline。
- **初始化**：R3A ← R1 `029f80a5…`；R3B ← 官方 `yolo11s.pt` `85a76fe8…`。默认不从 R2 续训。
- **配方**：40ep / patience10 / batch8 / imgsz960 / AdamW / rect / seed0，增强全 0
  （translate 0.02、scale 0.1 保持 R1 原值）。

## 当前阻塞

1. **Owner 人工审核**：V5 §6.3 分层 10%–15% 盲重审 + kappa。抽签已写好。未过门不得新训、不得 promote。
2. **本轮评估未完**：R3B 同口径 canary + 静态 L/中/右、FP/1000、event recall。

## 不做（R3 连续回放出结果前）

LONG 模型、L2 重构、收益回测、强化学习、实盘部署、全量迁移、只调 conf/NMS、
同时改窗口+renderer+标签+结构。
