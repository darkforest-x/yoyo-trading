# PROJECT_PLAN — L1 Dataset V3 与 R3

目标（Owner V5 简报，2026-08-12）：**数据集准确** + **模型训练准确**。
其余一切（目录美化、全量迁移、文档扩写）不得阻塞这两件事。

## 阶段

| # | 阶段 | 交付 | 状态 |
|---|---|---|---|
| 1 | 审计两仓 + 冻结 R1/R2 基线 | `manifests/frozen_baseline_r1_r2_v1.json`、`docs/ASSET_REUSE_MAP.md` | ✅ 2026-08-12 |
| 2 | 最低项目规范 | CLAUDE/AGENTS/PROJECT_PLAN/HANDOFF + 6 份 docs | ✅ 2026-08-12 |
| 3 | 最小 L1 链路 + smoke parity | 20–50 固定样本逐字节复现 | 进行中 |
| 4 | 旧标签审计 | 迁移表 + shape/outcome 分离 + `class_status`/`box_status` | 待办 |
| 5 | Dataset V3 Gold Core | manifest + 训练前验收报告（一致率/kappa/错标率/覆盖/泄漏） | 待办 |
| 6 | R3A / R3B 快速筛选 | 两条臂，同数据同配方同 seed | **被 3060 阻塞** |
| 7 | 静态评估 + 连续回放 | 与 R1/R2 同口径对照 | 待办 |
| 8 | 优胜者 3 seed + 独立块复验 | 接受 / 拒绝裁决 | 待办 |

## 已锁定的决定

- **窗口**：沿用 R1 的 W12–19 动态 center-crop（Owner 2026-08-12）。首轮只改数据。
- **类别**：唯一 `ma_dense_core`，SHORT pipeline。
- **初始化**：R3A ← R1 `029f80a5…`；R3B ← 官方 `yolo11s.pt` `85a76fe8…`。默认不从 R2 续训。
- **配方**：40ep / patience10 / batch8 / imgsz960 / AdamW / rect / seed0，增强全 0
  （translate 0.02、scale 0.1 保持 R1 原值）。

## 当前阻塞

1. **3060 训练机**：`192.168.1.4` ping 通、SSH 无响应 → R3 无算力。需 Owner 开机或授权替代方案。
2. **Owner 人工审核预算**：Dataset V3 质量门要求分层抽 10%–15% 盲重复审核算 kappa，
   这一步只有 Owner 能做。

## 不做（R3 连续回放出结果前）

LONG 模型、L2 重构、收益回测、强化学习、实盘部署、全量迁移、只调 conf/NMS、
同时改窗口+renderer+标签+结构。
