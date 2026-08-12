# HANDOFF — 当前真相

> 合同：`docs/DATASET_CONTRACT.md` · 协议：`docs/EXPERIMENT_PROTOCOL.md` /
> `docs/EVALUATION_PROTOCOL.md` · 复用：`docs/ASSET_REUSE_MAP.md` · 纪律：`CLAUDE.md`

---


## 2026-08-13 — 按 V5 继续：R3B 已训完（更正），V3.1+R3B canary 开扫

工作目录：`/Users/zhangzc/yoyo-trading`。扫描脚本入口也写在本仓 `tools/`；底层复用 fable scanner。

- **更正**：R3B 不是卡死。日志是 EarlyStopping 20/40，fitness-best ep10 mAP50 **0.844**。权重已拉回 `runs/r3b_v3gold_cold/weights/best.pt` SHA `b890997e…`
- V3.1 R3A 训完：fitness-best ep15 mAP50 **0.664** · `f7dc007f…`
- 已启动冻结 canary：05-03 00:00→12:00 UTC · 215 币 · W12–19 · conf 0.25 · scope=`preholdout_postval_canary`
  1. V3.1 `runs/r3a_v31_spread_ft/canary_am_20260503/`
  2. 随后 R3B `runs/r3b_v3gold_cold/canary_am_20260503/`
- 对照表将写入 `manifests/runs/canary_am_20260503_screen.json`
- 进度：`reports/v5_screen_status.txt` / `reports/v5_screen.log`
- 未 promote、未读 holdout


## 2026-08-12（V5 数据刀）— 在 yoyo-trading 改正例位置分布

工作目录：`/Users/zhangzc/yoyo-trading`。

**单一变量**：因果窗口长度仍在 W12–19、仍在 decision 收口、仍画 **SMA+EMA 20/60/120**；只改正例裁剪，让框尽量落在左/中/右。

| 清单 | 路径 |
|------|------|
| 几何计划 | `manifests/dataset_v3_1_position_spread_plan.json` |
| 逐样本分配 | `manifests/dataset_v3_1_position_spread_assignments.jsonl` |
| 新数据集（构建中） | `datasets/dataset_v3_1_position_spread_v1/` |
| 构建日志 | `reports/build_v3_1_position_spread.log` |

计划分配（1345 正例）：**left 72 / middle 636 / right 637**。  
几何上限：只有 72 个核心离 decision 足够远，才能在 W≥12 且无未来 K 时进入左 1/3。V3 的 0/1345/0 不是渲染漏了 EMA。

---

## 2026-08-12（V5 纠偏）— 以 V5 为准：R3A 登记进 yoyo，R3B 已开训，canary 进行中

**权威文档**：`工作/未命名文件夹/Claude提示词_yoyo-trading数据集准确性与模型准确性_R3训练_V5.md`

### 纠正内容

此前 R3A 产物主要落在 `fable-trading/analysis/output/...`，不完全符合 V5「目标仓 = yoyo-trading」。
现已：

1. **R3A 登记到 yoyo-trading**
   - 权重副本：`runs/r3a_v3gold_ft_r1/weights/best.pt`
   - 运行清单：`manifests/runs/r3a_v3gold_ft_r1.json`
   - best SHA：`3042c16d766ede358e89db0a33fa2983ec7a2d22893731d2b34ed44c39f010f7`
   - 基座 R1 SHA：`029f80a5…`（与 frozen baseline 一致）
   - 静态 best：epoch 33 · mAP50 **0.903**（**不作晋升依据**，V5 §8.5）

2. **R3B 已在 3060 启动**（V5 §8.2 冷启动臂）
   - name：`yoyo_r3b_v3gold_cold`
   - 初始化：官方 `yolo11s.pt` SHA `85a76fe8…`（与 frozen 一致）
   - 数据：同一 `dataset_v3_gold_core_v1`
   - 配方意图：40ep / pat10 / batch8 / seed0 / imgsz960 / `--no-finetune`
   - 日志：`C:\fable\logs\yoyo_r3b_v3gold_cold.train.log`
   - 注意：冷启动日志出现 `optimizer=auto`→AdamW(lr=0.002)；R3A 微调为 lr=0.0001。需在结果中诚实标注，是否强制同 lr 待对照后决定（首轮变量应仅为初始化权重）。

3. **连续 pre-holdout canary（R3A）**
   - 协议：`owner_short_gold_center_recent2d_v1_20260811`
   - 块：2026-05-03 00:00→12:00 UTC，W12–19，conf 0.25，215 币
   - 状态：Mac MPS 扫描中（约 95+/215 币）
   - 输出目录：`fable-trading/analysis/output/yoyo_r3a_v3gold_ft_r1_preholdout_canary_20260503_am/`
   - （扫描脚本仍在 fable-trading，因历史 canary 工具链在此；结果将写回 yoyo manifests）

### 窗口说明（V5 vs Owner）

V5 §8.2 字面写 W30；Owner 2026-08-12 与 `EXPERIMENT_PROTOCOL.md` 锁定 **W12–19** 为 V3 首轮协议。本阶段按 **Owner / EXPERIMENT_PROTOCOL** 执行。

### 仍未做（V5 清单）

- R3B 训完 + 拉回 SHA 登记
- R3A/R3B/R1 同口径 canary 对照表
- 完整静态指标（event recall、FP/1000、左中右）
- 人工盲审一致率 / kappa（Owner 门）
- 多 seed / 独立时间块（正式晋升，非快速筛选）

### 纪律

holdout 未读、未 promote、未部署、未下单。


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
