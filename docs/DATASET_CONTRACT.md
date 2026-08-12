# DATASET_CONTRACT — Dataset V3（L1 `ma_dense_core`）

版本：v3.2-owner-r1-20260813 · 方向范围：`short_pipeline` · 唯一类别：`ma_dense_core`

## 0. L1 只回答一个问题

> decision 时刻及以前的局部图里，是否存在 Owner 认可的均线密集核心，以及它在哪里。

L1 不回答未来涨跌、是否值得开空、收益、止损、仓位。**未来结果只能作为 metadata，
不得反向决定 L1 标签。**

## 1. 类别合同

### POSITIVE

必须同时成立：

- 六条均线（SMA+EMA 20/60/120）在 **decision 及以前** 形成一个能指出来的局部粘合/收口，不是「看起来挤在一起」；
- 粘合持续到肉眼可数的若干根，不是一两根擦肩；
- 小框只包这束线，不包前面的普通趋势，也不包已经打开的主跌段；
- **不靠 decision 之后的涨跌来决定是不是密集。**

2026-08-13 Owner 复审（243 张，κ=0.50）之后收紧：旧 gold-center 正例里，Owner 明确否掉并选择 FLIP 的 28 张，视为旧框把「线靠近」或「纵轴压扁」当成了密集，**不再当正例**。

### NEGATIVE

以下一律为负，即使旧集标过正：

- 纵轴被窗口 min-max 拉满造成的伪粘合（线几乎水平贴在一起，换带价格轴就散开）；
- 普通交叉、平行、已经明显张开；
- 只有一两根偶然靠近；
- 框里主要是下跌本身，不是下跌前的收口；
- Owner 本轮复审定为「没有密集」的样本（28 张 FLIP）。

**只因为后面没跌、或虚线不高于紫带均价，而把当时成立的密集改成负，仍是违约。**
那条「虚线收盘 > 紫带均收」是 Owner 的 **审图/做空前置过滤**，属于未来辅助，**不是 L1 标签规则**。

### IGNORE / 本轮直接丢掉

- Owner 无法稳定判断、图异常、灰区、框不可靠、两次审核不一致。
- **2026-08-13 Owner 裁定：22 张 REBOX（含 8 张想改成正但没有金框）本轮直接丢掉，不画框、不进训、不进 IGNORE 池扩量。**

IGNORE / 丢掉的样本不参与训练，也不计入正负比例。

## 2. 框几何合同

框只包住均线密集核心：不大面积含前置普通趋势、不大面积含后续突破或主跌段、
不使用 decision 之后的 K 线、横向边界对应真实 bar、纵向覆盖目标均线束与必要 K 线结构。
过大、过小、明显偏移的框不得作为金标。

几何唯一来源：`analysis/output/owner_side_review/review_sheet.csv` 的原始 Label Studio 坐标。
外层窗口可重裁，内框只能从已批准坐标派生。**禁止二次目测重画。**
Owner 2026-08-13：需要重画框的样本本轮全部丢弃，不启动新的画框流程。

## 3. 两个状态分开记录

```text
class_status = confirmed | rejected | ignore
box_status   = human_gold | rule_verified | model_proposed | rejected
```

- 协议级确认 ≠ 逐样本确认：`owner_protocol_confirmed` 与 `sample_owner_confirmed` 分层记录；
- `model_proposed` 框未经审核**不得**进入训练金标；
- Owner 在审核页上的 YES 是**类别**确认，不自动等于框几何确认。

## 4. 窗口协议（Owner 2026-08-12 裁定）

沿用 R1 的 **W12–19 动态 center-crop**：core 4–7 根，pre 5–7 根，post 3–5 根。
理由是首轮只改数据、不改窗口，保证 R3 vs R1 是单变量对照。

已知代价：center-crop 让核心集中在窗口中部，**位置 shortcut 本轮修不掉**；
左/中/右召回必须照报，不得声称已解决。V5 §7 的 W=30 属更早的 `dense_start` 血统，
留作后续单变量臂。

## 5. 负样本结构

- 约 50% 普通负样本 + 约 50% 相似 hard negative；
- 正负比例 1:1 ～ 1:2；
- hard negative 必须是 **shape NO**，不能只是 outcome NO；
- 每轮新增 hard negative 后必须复查真实正样本召回是否受损；
- 禁止一次性灌入巨量 hard negative。

配对比例是软目标；**Owner 框保护区、同币同时间块、时间隔离是硬约束**。找不到安全背景就诚实缺样，
不得跨币凑数或靠近金标取样。

## 6. 时间切分与去重

- 按时间块划分 train / val / canary，split 之间设 purge（现行 150 根 15m bar）；
- 同一 `event_id` 不得跨 split；同一事件的多个裁剪不得跨 split；
- 相邻重叠窗口按依赖块合并，不得制造隐性重复；
- holdout（≥2026-05-04）保持冻结，本仓不读。

## 7. 因果合同

每个样本必须满足：

```text
visible_end_bar <= decision_bar
future_used_for_l1_label = false
```

## 8. lineage 字段（每样本必填）

```text
sample_id event_id symbol timeframe
source_repo source_commit source_dataset source_path
window_start_bar window_end_bar window_length anchor_bar decision_bar visible_end_bar
box_start_bar box_end_bar
class_name class_status box_status split is_hard_negative ignore_reason
future_used_for_l1_label
image_path image_sha256 label_sha256 builder_commit config_sha256
```

## 9. 训练前质量门（不过就修数据，不许用训练结果掩盖）

必须在一页内给出：

- 正 / 负 / ignore 数量；
- human_gold / rule_verified / model_proposed 框数量；
- 旧样本 KEEP / RELABEL / REBOX / IGNORE / REJECT 数量；
- 重复审核一致率与 Cohen's kappa（分层抽 10%–15% 盲重审）；
- 标签抽查错标率（每个 split）；
- 时间块 / 币种 / 波动 / 框位置覆盖；
- 跨 split 重复事件数（必须为 0）；
- 未来泄漏数（必须为 0）；
- Dataset manifest SHA256。

一致率低于约 90% 或 kappa 明显不足 → 先修订标签合同并复审争议样本，**不得直接训练**。
