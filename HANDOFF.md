# HANDOFF — 当前真相

> 合同：`docs/DATASET_CONTRACT.md` · 协议：`docs/EXPERIMENT_PROTOCOL.md` /
> `docs/EVALUATION_PROTOCOL.md` · 复用：`docs/ASSET_REUSE_MAP.md` · 纪律：`CLAUDE.md`

## 2026-08-12（第二次更新）— Dataset V3 已建成，机器门全过，等 3060 与人工一致性

**Dataset V3 Gold Core**：`manifests/dataset_v3_gold_core_v1.json`，
manifest SHA `db9ca3eef7b1900b…`，数据在 `datasets/dataset_v3_gold_core_v1/`（不入 git）。

| 项 | train | val | 合计 |
|---|---:|---:|---:|
| 正例 | 1,143 | 202 | 1,345 |
| 易负例 | 1,143 | 200 | 1,343 |
| 硬负例（Owner shape-NO） | 765 | 0 | 765 |
| 负正比 | **1.67:1** | — | — |

唯一变量是训练负例：916 个 Owner-long 镜像与 1,370 个未审的模型挖掘硬负例移出，
765 个 Owner 确认的 shape-NO 进入。正例、窗口协议、renderer、整个 val split 逐字节不变
（2,688 个复用样本重建，0 SHA 漂移）。

**机器质量门全过**：重复窗口 0、跨 split event 0、`visible_end<=decision` 3,453/3,453、
future flag 全 false、val 无硬负例、purge 162 根 bar、最晚 decision 2026-05-02 23:45 UTC、
holdout 读取 0、765 个事件窗口解析失败 0。

**两道人工门未过**（只有 Owner 能过）：10%–15% 盲重复审核的一致率与 kappa、每 split 的
标签抽查错标率。审核包尚未生成，等 Owner 点头。

**已知结构缺陷**：正例核心水平位置 **middle 1,345 / left 0 / right 0**（中位 0.577）——
center-crop 的位置 shortcut 本轮修不掉，评估必须照报左/中/右召回。

**阻塞**：3060（192.168.1.4）ping 通但 SSH 挂起（探测 >25 分钟无响应），R3A/R3B 无算力。

报告：`~/fable-trading/analysis/html/p3_yoyo_dataset_v3_gold_core_prereview_20260812.html`。

---

## 2026-08-12 — 基线已冻结，规范已建，等 3060 与标签审计

**已完成**

1. **冻结基线** `manifests/frozen_baseline_r1_r2_v1.json`（SHA `2258566017153390…`），
   由 `tools/freeze_baseline.py` 生成（先入库后运行）。内容：5 个权重 SHA、R1 训练配方
   （40ep/patience10/batch8/imgsz960/AdamW/rect/seed0，增强全 0）、三个模型的静态 val、
   两个 pre-holdout canary 快照的完整契约与事件数、三个数据集 manifest 的 SHA 与分布、
   金标几何 CSV 的 SHA、六个 Owner 审核包的指针。工具会拒绝冻结跨契约的 canary 比较。
2. **窗口裁定**：Dataset V3 沿用 R1 的 **W12–19 动态 center-crop**（Owner 2026-08-12），
   首轮只改数据；V5 §7 字面的 W=30 属更早的 P1 `dense_start` 血统，留作后续单变量臂。
3. **规范**：CLAUDE.md 追加 12 条 V5 锁定；新增 `docs/{ARCHITECTURE,ASSET_REUSE_MAP,
   DATASET_CONTRACT,EXPERIMENT_PROTOCOL,EVALUATION_PROTOCOL,INTEGRATION_CONTRACT}.md`；
   `configs/source_repo.json` 定义只读源仓引用。

**冻结的对照数字**（后续任何 R3 结论都必须对着这张表说话）

| 模型 | 静态 P / R / mAP50 | am canary 事件 | pm canary 事件 |
|---|---|---:|---:|
| 1:1 baseline | .8467 / .9024 / .9206 | 732 | — |
| R1 `029f80a5` | .8626 / .7770 / .8980 | 331 | 223 |
| R2 `52cd38fd` | .8475 / .7525 / .8774 | — | 195 |

契约：conf 0.25 / NMS 0.7 / event_gap 5 bars / W12–19 / 215 币 / pre-holdout。

**下一步**：smoke parity（20–50 样本逐字节复现）→ 旧标签迁移表与 shape/outcome 分离
→ Dataset V3 Gold Core 与训练前验收 → R3A/R3B。

**阻塞**

- **3060**（`192.168.1.4`）ping 通但 SSH 无响应，R3 训练无算力；
- Dataset V3 质量门需要 Owner 做 10%–15% 盲重复审核（算一致率与 kappa）。

**纪律状态**：holdout 未读、未训练、未 promote、未部署、未下单；
本仓对 fable-trading 只读。
