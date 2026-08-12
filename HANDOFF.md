# HANDOFF — 当前真相

> 合同：`docs/DATASET_CONTRACT.md` · 协议：`docs/EXPERIMENT_PROTOCOL.md` /
> `docs/EVALUATION_PROTOCOL.md` · 复用：`docs/ASSET_REUSE_MAP.md` · 纪律：`CLAUDE.md`

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
