# DATASET_CONTRACT — Dataset V3（L1 `ma_dense_core`）

版本：v3-draft-20260812 · 方向范围：`short_pipeline` · 唯一类别：`ma_dense_core`

## 0. L1 只回答一个问题

> decision 时刻及以前的局部图里，是否存在 Owner 认可的均线密集核心，以及它在哪里。

L1 不回答未来涨跌、是否值得开空、收益、止损、仓位。**未来结果只能作为 metadata，
不得反向决定 L1 标签。**

## 1. 类别合同

### POSITIVE

- 均线在局部区间内真实收敛/粘合/密集；
- 结构持续到可辨识程度；
- 小框能明确包住密集核心；
- 判断不依赖 decision 之后的行情；
- **后来上涨 / 横盘 / 做空失败都不取消正类身份**（那是 L2 的 outcome，不是 L1 的 shape）。

### NEGATIVE

局部图确实不存在目标密集核心：普通交叉、平行趋势线、已明显发散、仅一两根偶然靠近、
纵轴压缩造成的伪密集、外观接近但核心不成立、历史模型误报且人工确认无密集。

**只因未来结果为负而标 NEGATIVE 是违约。**

### IGNORE

Owner 无法稳定判断、数据或图片异常、类别灰区、框范围无法可靠确认、多次审核结论不一致。
IGNORE 不参与首轮训练，也不计入正负比例。

## 2. 框几何合同

框只包住均线密集核心：不大面积含前置普通趋势、不大面积含后续突破或主跌段、
不使用 decision 之后的 K 线、横向边界对应真实 bar、纵向覆盖目标均线束与必要 K 线结构。
过大、过小、明显偏移的框不得作为金标。

几何唯一来源：`analysis/output/owner_side_review/review_sheet.csv` 的原始 Label Studio 坐标
（Owner 已逐框裁定方向）。外层窗口可重裁，内框只能按已批准的中心截取规则从原坐标派生。
**禁止二次目测重画。**

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
