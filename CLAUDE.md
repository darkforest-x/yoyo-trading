# CLAUDE.md — yoyo-trading 工作规范

四层结构:**YOLO 检测(L1)+ LightGBM 判断(L2)+ 回测(L3)+ 执行(L4)**,
15m OKX USDT-SWAP,**做空单向**。

本仓只放**代码**。数据、模型、实验报告、learnings 全部留在 `~/fable-trading`
(见「与 fable-trading 的关系」)。

## 唯一的结构约束

**`yoyo/layers/` 四层禁止互相 import,只能经 `yoyo/contracts/` 和 `yoyo/data/`。**

由 `tests/test_layer_boundaries.py` 用 AST 强制,**不是口头约定**。

病因:2026-08-03 一个故障横跨 `forward_scan` + `frozen` + `executor` 三个文件 ——
L2 的事实(模型用什么坐标系训的)被 L1 的事实(这单做多还是做空)决定了,
而代码里没有任何东西反对。分层做完后又立刻抓出两处同类问题:
forward log(L2 写、L4 读)和 barrier 参数(L1/L2/L3 都要读)本来就不属于任何一层。

**遇到"我需要 import 另一层"时,正确反应不是绕过测试,是问这个东西是不是本来就该在 contracts。**

## 铁律(继承自 fable-trading,违反 = 返工)

1. **holdout 纪律**:holdout(≥2026-05-04)只在最终验收评估,每次动用需 owner 明确批准
   并记录"第 N 次消耗"。**本仓不做评估,所以本仓不应出现任何 holdout 读取。**
2. **时间切分**:所有评估按时间切分,禁止随机切分。
   **研究侧默认 walkforward;CPCV 只作上界参考并注明** —— 实测其乐观幅度 21.93bp,
   比本项目要找的效应(5~20bp)还大。
3. **无前视**:特征只能用信号 bar 及之前的数据;只有标签允许看未来。
   新增特征必须在 docstring 写明用到的列与窗口。
4. **单变量纪律**:一次实验只改一个变量;结果无论成败都写入报告。
5. **YOLO 增强禁用**:fliplr/flipud/mosaic/mixup/hsv 全关 —— 破坏时间方向与红绿 K 线语义。
6. **`yoyo/layers/l1_detection/render.py` 的像素不得改动一位。** 检测器绑死在那些像素上。
   任何改动前后必须比对 sha256。
7. **不自动 promote**:ACTIVE / bundle 的切换需 owner 点头。
8. **真金操作**(下单/撤单/kill 开关/改仓位/改 API key)只有 owner 亲手做或逐次授权。
9. **检测只认盘口**:live 扫描只扫 tip/tip-1/tip-2;凡"只能产出事后信号"的路径一律不得存在。
10. **单分支纪律**:**只有 `main`,不开新分支、不建 worktree。**
    提交前先 `git branch --show-current` 确认。
11. **层间契约**:见上。
12. **无新增重型依赖**:python3 + pandas/lightgbm/ultralytics,`pyproject.toml` 的
    `dependencies` 故意为空 —— 装本包不得把求解器拖进一个正常的 venv。

## 与 fable-trading 的关系

`~/fable-trading` 是**实验日志与数据仓**,不会被删:

- `analysis/` 207 份报告、`docs/learnings/` 234 条 —— **一个字都不改**。
  它们记录「当时发生了什么」,里面的路径就是当时真实的路径。
- `data/` `models/` `datasets/` `runs/` —— 数据与权重。
- `src/` —— 尚未迁移的模块(L2 判断、L3 回测、看板、支线)+ 指向本仓的转发壳。

`fable-trading` 通过 `pip install -e ~/yoyo-trading` 引用本仓,所以那边的
`src/` 转发壳与 528 个测试无需改动。

**新旧路径对照**:`~/fable-trading/docs/RESTRUCTURE_MAP.md`。

## 迁移状态

| 层 | 状态 |
|---|---|
| `yoyo/contracts/` | 已迁 — protocol · outcomes · costs · forward_log |
| `yoyo/data/` | 已迁 — indicators |
| `yoyo/layers/l1_detection/` | 已迁 — render · data · candidates |
| `yoyo/layers/l4_execution/` | 已迁 — config · executor · ledger · okx_client · symbols |
| `yoyo/layers/l2_judgment/` | 部分 — features · frozen · train · labeling · forward_types 已迁;`forward_scan` / `forward` 未迁 |
| `yoyo/layers/l3_backtest/` | **未迁** |
| `yoyo/cli/` | 空 — 真入口待建(不是 253 个脚本) |

## 剩下的那台手术:`forward_scan.py`

`~/fable-trading/src/judgment/forward_scan.py`(725 行)+ `forward.py`(326 行)**故意没搬**。
它们是这次重构里唯一不能"搬"、只能"重新设计"的部分,原因写在这里,免得下一个人以为是遗漏。

**为什么不能直接搬**:`scan_forward_records` 用 **3 线程池并行跑 L1 候选发现**,
再**顺序跑 L2 打分**,中间的 `discover_wall` / `phase2_wall` 是**铁律「脉冲 <15min」的监控点**。
把 L1/L2 拆进不同模块 = 重构这套并发。超预算的后果不是慢,是**结构性挡住 tip 信号**。

**已确定的设计**(不必重新讨论,直接照做):

```
yoyo/layers/l1_detection/scan.py     series → 候选下标(现 _discover / forward_candidate_indices
                                     / _rule_candidate_indices,约 60 行)
yoyo/layers/l2_judgment/score.py     候选 + frame → 分数 + ForwardExit
                                     (现 _artifact_* / _extract_rows_for_artifact
                                     / resolve_forward_exit* 适配器,约 250 行)
yoyo/cli/forward_pulse.py            编排:调 L1 → 调 L2 → 写 contracts/forward_log
                                     **编排器不是层** —— 这就是它现在卡住的原因
```

**动手前必须做的两件事**:
1. 先量一次当前 `discover_wall` / `phase2_wall` 的基线,拆完再量,**不得变慢**。
2. 出场解算**已经没有分叉**:`resolve_forward_exit_short` 只是把
   `contracts/outcomes.resolve_barrier_outcome` 适配成 `ForwardExit`(P0.5 已治好)。
   **不要再"统一"一次**,那会造出第三套。

`src/judgment` 下另有 17 个研究/工具模块(barrier_sweep · p1_build · p2_l2 · shadow_* 等),
**不属于四层骨架**,不要因为它们同在一个目录就升进 `layers/`。

## 代码标准

python3 + pandas/lightgbm/ultralytics。模块级 docstring 说明**来源与决策依据**,
不只说做了什么 —— 现有代码都是这个风格,照着写。
注释与代码用英文,与 owner 交流用中文。

## 不确定时停下来问

holdout、阈值预设、障碍参数(TP/SL 倍数、atr 下限)、成本假设、
ACTIVE/bundle 切换、promote、真下单、新鲜度门 —— **一律先问 owner,不要"先试试"。**

---

## L1 Dataset V3 / R3 阶段锁定(Owner V5 简报,2026-08-12)

本阶段只抓两个结果:**数据集准确**、**模型训练准确**。目录美化、完整历史迁移、
文档扩写和无关重构**不得阻塞**这两件事。

1. 只服务 **SHORT pipeline**,第一阶段不开发 LONG,不新增方向类别。
2. L1 唯一类别 `ma_dense_core`;L1 只回答"decision 及以前有没有 Owner 认可的均线密集核心、在哪"。
3. **L1 标签不由未来涨跌决定**。当时密集成立、后来上涨/横盘/做空失败 —— 仍是 L1 正样本,
   只可能是 L2 的 outcome 负样本。
4. 优先复用 `fable-trading`(REUSE → ADAPT → REFERENCE → 有证据才 REWRITE),不得无理由重写。
5. 数据集准确与模型准确 **优先于** 完整重构。
6. **类别确认与框几何确认必须分开**:`class_status` 与 `box_status` 分列;
   `model_proposed` 框未经审核不得进训练金标;协议级确认 ≠ 逐样本确认。
7. 时间切分优先,**禁止随机图片切分**;split 之间 purge,同一 event 不跨 split。
8. 禁止未来 K、事件重复、增强泄漏;`visible_end_bar <= decision_bar`。
9. **不得只看 mAP 宣布成功**;裁决靠连续行情事件密度 + Gold Core 召回(见 `docs/EVALUATION_PROTOCOL.md`)。
10. 每个实验必须有 lineage、配置、日志与 SHA(见 `docs/EXPERIMENT_PROTOCOL.md`)。
11. 不得擅自读取 holdout、部署或下单。
12. 关键结果及时更新 `HANDOFF.md`,但**报告先给简洁结论**,细节放附件。

**窗口协议(Owner 2026-08-12 裁定)**:Dataset V3 沿用 R1 的 **W12–19 动态 center-crop**,
首轮只改数据。V5 §7 字面的 W=30 属更早的 P1 `dense_start` 血统,留作后续单变量臂;
代价是 center-crop 的位置 shortcut 本轮修不掉,左/中/右召回必须照报。

**本仓存放范围的修订**:原则仍是"本仓只放代码",但本阶段允许 `configs/`、`manifests/`、
`reports/` 存放**小体积 JSON / Markdown**(冻结基线、数据集 manifest、给 Owner 的结论页)。
图片、权重、K 线、runs 仍然只在 `fable-trading`。
