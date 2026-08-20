# ARCHIVED

This repository has been consolidated into `darkforest-x/fable-trading`.

- Canonical repository: `darkforest-x/fable-trading`
- Source frozen commit: `784766de45a3b876c986d3ba672779124b46a66f`
- Migration PR: https://github.com/darkforest-x/fable-trading/pull/1
- Migration merge commit: `31d6b2aa39cc8cc454144228980c8eb427f868c4` (PR #1, merged into main)
- Canonical module: `yoyo/ (whole package), datasets/manifests/, configs/labelstudio/, tools/review/, experiments/historical/yoyo_trading/`
- Status: read-only historical research — superseded — the canonical yoyo package returned to fable-trading

No further development occurs in this repository. Its conclusions, including the
negative ones, are registered in `experiments/registry.yaml` and summarised in
`experiments/historical/` in the canonical repository. Nothing here was deleted.

---

# fable — 四层结构

owner 2026-08-03 决定重构并独立成仓。逐层迁移,每层跑通测试再进下一层。
数据、实验报告与 learnings 留在 `~/fable-trading`,那边通过 `pip install -e ~/yoyo-trading` 引用本仓。

## 为什么重构

不是文件多,是**层与层之间没有契约**。量出来的事实:

```
src/       120 个 py  25,238 行
scripts/   253 个 py  62,852 行    ← 是 src 的 2.5 倍,其中 43 个 diag_*、35 个 build_*
四层入口可达的模块只有 28 个 ≈ 7,700 行    ← 系统本体
src/ 里 80 个模块(74%)在四条链之外
```

`forward_scan.py` 一个文件 725 行,同时做**调检测器 + 抽特征 + 打分 + 解 barrier**。
所以 2026-08-03 那个 side/feature-semantics 的故障能横跨
`forward_scan` + `frozen` + `executor` 三个文件 —— 没有边界,改一处要靠人记得另两处。

## 结构

```
yoyo/
  contracts/       跨层契约。层与层只经这里对话
    protocol.py      StrategyProtocol + bundle 校验(谁在跑)
    outcomes.py      canonical barrier / return(结果怎么算)
    costs.py         成本常数(owner 决策,代码不得自改)
    schema.py        特征列与语义、forward row 列(数据长什么样)
  layers/
    l1_detection/    YOLO:渲染 → 扫描 → 候选
    l2_judgment/     LightGBM:候选 → 分数
    l3_backtest/     同一 outcome 合约下的回放与验收
    l4_execution/    分数 → 订单(fail-closed)
  data/            K 线读取 / universe / bar 工具
  cli/             真入口(不是 253 个脚本)

tools/dashboard/   看板。观察工具,不是交易层
archive/           支线包与一次性诊断脚本
```

## 唯一的硬约束

**`layers/` 之间禁止互相 import,只能经 `contracts/` 和 `data/`。**

由 `tests/test_layer_boundaries.py` 强制,不是口头约定。违反即测试失败。

理由:上面那个跨三文件的故障,本质是 L2 的一个事实(模型用什么坐标系训的)
被 L1 的一个事实(这单做多还是做空)决定了。有契约就不会。

## 迁移状态

| 层 | 状态 |
|---|---|
| contracts | 进行中 |
| l1_detection | 未开始 |
| l2_judgment | 未开始 |
| l3_backtest | 未开始 |
| l4_execution | 未开始 |

迁移期间 `src/` 下的模块改为**转发到 `yoyo/`** 的薄壳,所以现有 scripts 与 499 个测试
不需要同步改动;等调用方全部切过去再删壳。
