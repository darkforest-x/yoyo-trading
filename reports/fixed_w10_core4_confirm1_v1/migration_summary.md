# Migration summary

**Freeze 已完成。training_eligible = false。** 金标 SIGNAL **1,247** / NO_SIGNAL **1,402**；训练图 2,649。  
不能开训：协议 17.6 DIRECT 抽检错误率未知。结论页：`freeze.md`。

---


| item | value |
|---|---|
| 旧来源种类 | 12 |
| 8768 已保存 | 24 |
| Label Studio yoyo_gold_v1 任务/完成 | 30 tasks / 1 completion |
| 候选任务数（1,345） | 1345 |
| 事件数 | 3723 |
| DIRECT | 0 |
| ONE_CLICK_REVIEW | 152 |
| MANUAL_ADJUST | 1268 |
| CONFLICT | 44 |
| IGNORE | 2259 |
| SIGNAL 候选 | 1511 |
| NO_SIGNAL 候选 | 2189 |
| 8768 24 条命中 | 24/24 |
| 8768 丢失 | 0 |
| holdout 读取 | 0 |
| 冻结 SIGNAL | 1247 |
| 冻结 NO_SIGNAL | 1402 |
| DIRECT 抽检错误率 | 未测（Owner 未审） |
| training_eligible | False |

## 旧核心长度
{"6": 48, "7+": 1256, "4": 45, "5": 38}

## 旧 decision offset
{"1": 48, "2": 451, "-1": 798, "0": 379, "-3": 11, "3": 128, "4+": 186, "-2": 9, "-6": 1, "-4": 3, "-8": 1}

## 冻结 split
{"train/NO_SIGNAL": 980, "train/SIGNAL": 869, "val/SIGNAL": 179, "val/NO_SIGNAL": 171, "test/NO_SIGNAL": 251, "test/SIGNAL": 199}

direct_error_rate — 冻结 Gold SIGNAL=1247 NO_SIGNAL=1402。DIRECT 抽检错误率未知（迁移 DIRECT=0；协议 17.6 不得改报告绕过）
