# V5 Agent Loop — 已启动

## 模式说明

本会话按 **V5 为准** 自主推进，不等待每一步人工确认（仍遵守：不 promote、不读 holdout、不部署/下单）。

## 当前并行任务

1. **R3B 训练**（3060）`yoyo_r3b_v3gold_cold` — seed0 · yolo11s 冷启动  
2. **R3A pre-holdout canary**（Mac MPS）am 块 215 币 · W12–19 · conf 0.25  
3. **Watcher** `logs/v5_agent_watch.out` / `reports/r3_agent_watch_status.json` — 每 3 分钟轮询  

## 两者都完成后自动做

1. 拉回 R3B `best.pt` → `yoyo-trading/runs/r3b_v3gold_cold/` + SHA 清单  
2. finalize R3A canary（若需）  
3. 同口径对照表 R1 / R3A / R3B（events、raw、密度）  
4. 更新 `HANDOFF.md` + 一页裁决（快速筛选，不 promote）  

## 查进度

```bash
cat ~/yoyo-trading/reports/r3_agent_watch_status.json
tail -20 ~/fable-trading/logs/v5_agent_watch.log
```

或对我说「进度」。
