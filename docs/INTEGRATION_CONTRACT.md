# INTEGRATION_CONTRACT — yoyo-trading 交给 fable-trading 什么

## 1. 只输出 L1 候选事件，不输出交易动作

```json
{
  "schema_version": "l1_ma_dense_event_v1",
  "direction_scope": "short_pipeline",
  "detection_class": "ma_dense_core",
  "model_version": "r3a",
  "weights_sha256": "...",
  "dataset_sha256": "...",
  "symbol": "ETH_USDT_SWAP",
  "timeframe": "15m",
  "window_start_time": "...",
  "decision_time": "...",
  "confidence": 0.82,
  "box_start_bar": 18,
  "box_end_bar": 23,
  "event_id": "..."
}
```

`decision_time` 必须等于该事件可见窗口的右端 bar 时间；**不得把使用了核心形态之后 K 线的
输出记为盘口时间**。

## 2. fable-trading 负责

空头方向判断、多因子与多周期、outcome 概率、风险收益、仓位与执行、部署与真金操作。

## 3. 生产资格

本仓产物默认 `production_eligible=false`。进入 tip-smoke / forward / ACTIVE / 部署
需要 Owner 单独批准，并另行批准延迟预算与执行架构。

## 4. 边界

- 本仓不写 fable-trading 的任何路径（只读，见 `configs/source_repo.json`）；
- 本仓不读 holdout（≥2026-05-04）；
- 本仓不下单、不改仓位、不碰 API key、不 promote ACTIVE、不清 forward log。
