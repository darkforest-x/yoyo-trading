# EVALUATION_PROTOCOL — R3 怎么算赢

## 0. mAP 不是裁决量

`val mAP 更高` 不构成晋升理由。裁决靠**同口径连续行情**与**人工确认的 Gold Core 召回**。

## 1. 静态评估（固定 val，逐文件相同）

必报：Precision、Recall、F1、mAP50、mAP50-95、event-level recall、FP / 1000 windows、
框定位误差、**左/中/右位置召回**、分时间块 / 分币种 / 分波动桶表现。

冻结对照（`manifests/frozen_baseline_r1_r2_v1.json`）：

| 模型 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 1:1 baseline | 0.8467 | 0.9024 | 0.9206 | 0.7294 |
| R1 | 0.8626 | 0.7770 | 0.8980 | 0.7405 |
| R2 | 0.8475 | 0.7525 | 0.8774 | 0.7413 |

## 2. 连续行情回放（真正的裁决场）

契约必须逐项相同，否则拒绝比较（`scripts/compare_owner_short_canary.py` 已强制）：
symbols / 时间范围 / bar endpoints / window exposures / conf / NMS / window_lengths /
event_gap_bars / evaluation_scope。

现行契约：conf 0.25、NMS 0.7、event_gap 5 bars、W12–19、215 币、pre-holdout。

| 快照 | 模型 | endpoints | raw | 去重事件 |
|---|---|---:|---:|---:|
| am 05-03 00:00→12:00 | baseline | 10,320 | 22,037 | 732 |
| am | R1 | 10,320 | 8,268 | 331 |
| pm 05-03 12:00→23:45 | R1 | 10,105 | 3,964 | 223 |
| pm | R2 | 10,105 | 3,538 | 195 |

必报：scanned symbols、bar endpoints、window exposures、raw detections、去重事件、
events/day、触发币数、重复触发次数、平均检测延迟、分时间块事件密度、
**人工抽样 shape YES 率**、Owner NO 的主要结构类型。

## 3. 晋升门（全部满足才算候选）

1. 人工确认 Gold Core 上真实正样本召回相对 R1 不明显下降；
2. 同召回口径下连续错误候选 / FP-per-1000 明显下降；
3. 左/中/右召回无明显失衡；
4. 不依赖单一时间块；
5. 多 seed 稳定（≥3 seed，报均值/标准差/最差 seed）；
6. 独立、未参与调参的 pre-holdout 块仍复现改进；
7. 无未来泄漏、split 泄漏、event 重复；
8. 改进不能仅来自提高 confidence。

工程参考门槛（须在报告中对照冻结基线确认）：

```text
连续错误候选或 FP/1000 相对 R1 至少下降约 30%
且 Gold Core event recall 相对 R1 下降不超过约 3–5 个百分点
```

没有候选满足 → **明确拒绝 R3**，不得包装成成功。

## 4. 失败定位顺序（不许盲训下一轮）

| 现象 | 优先检查 |
|---|---|
| val 高、连续误报多 | 负样本覆盖、时间泄漏、分布、renderer shortcut |
| 误报降、召回崩 | hard negative 过量、真密集被误标负、框过窄 |
| R3A 好 R3B 差 | R1 视觉特征可复用，数据量可能不足 |
| R3B 好 R3A 差 | R1 权重带历史偏见 |
| 左中右差异大 | crop 分布、位置 shortcut |
| 时间块间波动大 | 市场环境偏置、覆盖不足、split 泄漏 |
| 框差类别对 | 框几何合同或检测任务定义 |
| 类别反复争议 | 标签定义不清 → 扩大 ignore，不是继续训练 |

## 5. 诚实口径

- 人工抽样 YES 率若在看得到未来的页面上产生，一律称「未来辅助语义裁决」，不得称 precision；
- 池内指标看不见 beta，方向性结论必须带匹配随机对照；
- holdout（≥2026-05-04）本仓不读；任何 holdout 评估需 Owner 逐次批准并记录第 N 次消耗。
