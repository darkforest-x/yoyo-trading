# Freeze — 固定 W10 / Core4 / Confirm1 金标

Owner 2026-08-13 授权 freeze。**未开训、未 commit、未 promote、未读 holdout。**  
`training_eligible: false`。复审页 PID 32906 未杀。

## 结论

金标已冻、训练图已出，**还不能开训**。唯一没过的门是协议 17.6：迁移 DIRECT=0，抽检错误率未知，不得改报告绕过。

| 项 | 数 |
|---|---:|
| 冻结 SIGNAL | **1,247**（半开 4 根核 + confirm 1 + W10） |
| 冻结 NO_SIGNAL | **1,402**（核为空） |
| IGNORE | 0 |
| 训练图 | **2,649**（skipped 0） |
| train | 1,849（SIGNAL 869 / NO_SIGNAL 980） |
| val | 350（SIGNAL 179 / NO_SIGNAL 171） |
| test | 450（SIGNAL 199 / NO_SIGNAL 251） |
| 跨 split 事件组 | 0 |
| 跨 split 重复图 SHA | 0 |
| holdout 行 | 0 |

图：`~/fable-trading/datasets/fixed_w10_core4_confirm1_v1/classification/{train,val,test}/{SIGNAL,NO_SIGNAL}/`  
金标：`~/fable-trading/datasets/fixed_w10_core4_confirm1_v1/gold/`  
训练图 `render_w10 overlay=False`，1280×742，无黄带。未改 `render.py`。未改 `datasets/gold_v1.jsonl`。

| SHA | 值 |
|---|---|
| dataset_manifest | `20686feba41d15b82e34109402840c2d640fe1e2daea0392b35e1ea79320a7fc` |
| gold/events.jsonl | `344212f8e5ef1fac3616b2026d19d6e721ce29984b3bbda194d4071c9fc327c4` |
| config | `7db6c09930d122389b3648ff70551b41cfc6b26286ed8497510f2e3daf0d7652` |

## 排除 / 改标（未改复审 state）

**CONFLICT 44 未冻**：43 条从未进 state；1 条在 state 里当 A，freeze 丢掉：

- `AERO_USDT_SWAP_15m_20250715T011500Z`（无核 A + CONFLICT + 无 source_path）

**5 条无核 A 未冻成 SIGNAL：**

| gold_id | freeze |
|---|---|
| `2Z_USDT_SWAP_15m_20251017T170000Z` | 改成 NO_SIGNAL（已纠正 EASY_BACKGROUND） |
| `ACH_USDT_SWAP_15m_20251113T133000Z` | 改成 NO_SIGNAL（已纠正 EASY_BACKGROUND） |
| `ADA_USDT_SWAP_15m_20251015T151500Z` | 改成 NO_SIGNAL（已纠正 EASY_BACKGROUND） |
| `ADA_USDT_SWAP_15m_20251225T101500Z` | 改成 NO_SIGNAL（队列本是 NO_SIGNAL） |
| `AERO_USDT_SWAP_15m_20250715T011500Z` | 排除（CONFLICT） |

**3 条已纠正 EASY_BACKGROUND 保持负例**（上表前三）。原有 3 条 N（ACH 20250831、ADA 20250723 两张）仍是 NO_SIGNAL。

**17 条 SIGNAL** 因冻后 4 根核几何上 W10 内另有不同核心而排除（Gold V1 竞争核规则）：

BLUR 20251111 1845/1930 · DOT 20251130 2345 / 20251201 0015 · INIT 20260203 1730/1815 · MOVE 20260318 1115/1215 · NEO 20250925 0200 · RVN 20260326 0430 · SNX 20250723 1000/1100 · W 20251113 1700 · XTZ 20250811 1145/1230 · ZETA 20250723 1030/1045

易负例池按 Owner 授权收进负例。`ACH_USDT_SWAP_15m_20260215T031500Z` 缺 `source_path`，已从同币种唯一 K 线路径补回并渲染。

## 下一步

等 Owner 说开训。不要自行训 YOLO。若要先过 `training_eligible`，需补 DIRECT 15% 盲抽检（错误率 ≤5%），或由 Owner 明确豁免 17.6。
