# ARCHITECTURE — 本仓在做什么

## 1. L1 链路（本轮唯一主线）

```text
K线与均线 → 局部窗口 → 图像渲染 → 标签审核 → Dataset V3
          → YOLO训练 → 连续扫描 → event-level评估 → L1候选事件交付
```

## 2. 现有目录（沿用既有 `yoyo/` 包，不为对齐文档做大搬家）

```text
yoyo/
├── contracts/          层间共享事实（protocol / outcomes / costs / paths / forward_log）
├── data/               bars / indicators / loader / universe
├── layers/
│   ├── l1_detection/   render(像素冻结) · local_v2_render · scan · candidates · data
│   ├── l2_judgment/    2026-08-03 拆分时已迁入，本轮不扩展
│   ├── l3_backtest/    空
│   └── l4_execution/   同 l2，本轮不扩展
├── cli/
configs/                源仓引用、冻结基线规格
manifests/              冻结产物（JSON，小文件）
tools/                  可执行入口（freeze_baseline.py …）
tests/                  pytest
docs/                   合同与协议
reports/                给 Owner 的结论页
```

V5 §4.2 的 `src/yoyo/{render,labels,datasets,training,inference,evaluation}` 与现状的差异
按 **ADAPT** 处理：新 L1 数据集/评估代码作为 `yoyo/` 下的新子包加入，不重排既有模块——
重构不得阻塞 Dataset V3 与 R3。

## 3. 唯一结构约束

`yoyo/layers/` 四层禁止互相 import，只能经 `yoyo/contracts/` 与 `yoyo/data/`，
由 `tests/test_layer_boundaries.py` 用 AST 强制。

新增的非 layer 子包（数据集构建、评估）**可以** import layers；反向不行。

## 4. 与 fable-trading 的关系

fable-trading 是数据、权重、历史实验日志的事实来源，本仓只读引用
（`configs/source_repo.json`）。复用优先级 REUSE → ADAPT → REFERENCE → REWRITE。
详见 `ASSET_REUSE_MAP.md`。
