# ASSET_REUSE_MAP — 从 fable-trading 复用什么，为什么

审计日期：2026-08-12 · fable-trading commit `92b2571` · yoyo-trading commit `f431c41`

优先级：**REUSE → ADAPT → REFERENCE → 证据充分才 REWRITE**。
本仓对 fable-trading 只读；路径见 `configs/source_repo.json`。

## 1. 权重（REUSE，冻结不动）

| 资产 | 路径（相对 fable-trading） | SHA256 前 16 | 用途 |
|---|---|---|---|
| Stage A base | `analysis/output/lsv2_stagea/owner_lsv2_stagea_randomcrop_v1_cold/weights/best.pt` | `c0e94f47df125e29` | R1/R2 的初始化来源 |
| 1:1 baseline | `analysis/output/lsv2_stageb/owner_lsv2_short_gold_center_v1_ft/weights/best.pt` | `da278820f2d96a64` | 连续密度的第一条基线 |
| **R1** | `analysis/output/lsv2_stageb/owner_lsv2_short_gold_center_hardneg_r1_ft/weights/best.pt` | `029f80a52b5beda2` | 当前候选挖掘器 / R3A 初始化 |
| **R2** | `analysis/output/lsv2_stageb/owner_lsv2_short_gold_center_hardneg_r2_ownerconfirmed_ft/weights/best.pt` | `52cd38fda253f052` | 对照基线，不作为唯一初始化 |
| 官方预训练 | `models/yolo11s.pt` | `85a76fe86dd8afe3` | R3B 干净初始化 |

R1 在 3060 上训练时的 `model: C:/fable/models/yolo11s_w20.pt` 只是 `--base` 的远端副本，
不是另一个权重（见 `scripts/train_w20_midbox_on_3060.sh`）。

## 2. 金标几何（REUSE，唯一几何源）

`analysis/output/owner_side_review/review_sheet.csv`（SHA `bb7081e7e1821c5f…`）：
2,525 个原始 Label Studio 框 × 228 币，Owner 方向裁决 **short 1,361 / long 1,152 / skip 12**。

这是 Dataset V3 唯一允许的框几何来源。**禁止二次目测重画**
（`docs/learnings/original-gold-geometry-beats-secondary-manual-reboxing.md`）。

## 3. 数据集（ADAPT：复用样本与几何，重建标签与切分）

| 数据集 | 组成 | 处理 |
|---|---|---|
| `datasets/owner_short_gold_center_v1` | pos 1,345 / easy-neg 1,343 | 正例血统与 crop 协议 REUSE |
| `datasets/owner_short_gold_center_hardneg_r1` | + hard-neg 2,286 | R1 训练集；hard-neg 需重新审计 shape vs outcome |
| `datasets/owner_short_gold_center_hardneg_r2_ownerconfirmed` | 同上，替换 hard-neg 来源 | 对照 |

正例窗口分布 W12–19（12:213 / 13:335 / 14:278 / 15:130 / 16:231 / 17:86 / 18:25 / 19:47），
core 4–7 根，train 1,143 / val 202。**Dataset V3 沿用同一窗口协议**（Owner 2026-08-12 裁定）。

## 4. 代码（REUSE：渲染器已在本仓）

| 能力 | 位置 | 说明 |
|---|---|---|
| K线/均线 | `yoyo/data/bars.py`、`indicators.py`、`loader.py` | 已迁移 |
| 渲染器 | `yoyo/layers/l1_detection/render.py` | **像素不得改动一位**（铁律 6）；fable 的构建脚本已 import 本仓 |
| 局部渲染 v2 | `yoyo/layers/l1_detection/local_v2_render.py` | Local Signal V2 专用 |
| 扫描 | `yoyo/layers/l1_detection/scan.py` | 连续扫描相位 1 |
| 候选 | `yoyo/layers/l1_detection/candidates.py` | 事件去重 |

仍在 fable-trading、本轮以 REFERENCE 方式使用（不迁移）：

- `scripts/build_owner_short_gold_center_dataset.py`：正例/易负例构建（crop 协议的事实来源）；
- `scripts/build_owner_short_gold_center_hardneg.py` / `_r2.py`：hard negative 装配；
- `scripts/compare_owner_short_canary.py`：连续回放同口径比较（拒绝跨契约比较）；
- `scripts/serve_local_signal_v2_semantic_review.py`：Owner 审核页。

## 5. Owner 审核结果（REUSE 为审计输入，**不是**直接标签）

| 包 | n | Owner YES | 说明 |
|---|---:|---:|---|
| `owner_short_train_hardneg_review200_v1` | 200 | 18 | 硬负例检索偏向 |
| `owner_short_train_positive_retrieval100_v1` | 100 | 45 | 正例检索偏向 |
| `owner_short_train_hardneg_expansion200_v2` | 200 | 25 | 硬负例扩展 |
| `owner_short_train_hardneg_newblocks200_v3` | 200 | 26 | 硬负例新块 |
| `local_signal_v2_positive_semantic_review200_v2` | 200 | 96 | Positive 85% / Canary 11% |
| `local_signal_v2_early_frontier_review300_v1` | 300 | 119 | 2026-08-12 完成，检索分层零区分度 |

**这 1,200 条裁决全部 `training_eligible=false`。** 转标签前必须逐条区分
shape verdict 与 outcome verdict（V5 §6.2），并分开记录 `class_status` / `box_status`。

## 6. 冻结基线

`manifests/frozen_baseline_r1_r2_v1.json`（SHA `2258566017153390…`），由
`tools/freeze_baseline.py` 生成，含权重 SHA、训练配方、静态 val、连续 canary 契约。

连续 canary 契约（两个快照均为 conf 0.25 / NMS 0.7 / event_gap 5 bars / W12–19 / 215 币）：

| 快照 | 模型 | bar endpoints | raw | 去重事件 |
|---|---|---:|---:|---:|
| am 2026-05-03 00:00→12:00 | baseline | 10,320 | 22,037 | 732 |
| am | **R1** | 10,320 | 8,268 | **331** |
| pm 2026-05-03 12:00→23:45 | **R1** | 10,105 | 3,964 | **223** |
| pm | **R2** | 10,105 | 3,538 | **195** |

静态固定 val：baseline P .8467 / R .9024 / mAP50 .9206；R1 P .8626 / R .7770 / mAP50 .8980；
R2 P .8475 / R .7525 / mAP50 .8774。

## 7. 不复用 / 不迁移

- L2 LightGBM、L3 回测、L4 执行：本轮不扩展（`yoyo/layers/l2_judgment`、`l4_execution`
  在 2026-08-03 拆分时已迁入，保留原样，不删不改）；
- 200-bar / v10–v16 旧检测器：只作历史参照；
- P1 `dense_start` W30 血统（`datasets/local_signal_v2_p1_b2_w30` 等）：与当前类别语义不同，
  首轮不混用；
- `analysis/` 207 份报告与 `docs/learnings/` 234 条：**一个字都不改**。

## 8. 未解决的资产风险

1. **3060 训练机连不上**：`192.168.1.4` ping 通但 SSH 无响应，R3 训练臂目前无算力
   （Mac MPS 实测过慢，历史上已放弃过一次）。
2. `.claude/worktrees/peaceful-ardinghelli-b84300/` 是 fable-trading 里遗留的 worktree
   副本（违反单分支纪律），内含一份 Stage A 权重；**未删除，等 Owner 处置**。
