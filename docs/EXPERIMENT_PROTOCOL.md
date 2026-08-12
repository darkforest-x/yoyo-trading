# EXPERIMENT_PROTOCOL — R3 两条臂怎么跑

## 1. 先冻结，再训练

`manifests/frozen_baseline_r1_r2_v1.json` 必须先存在（`tools/freeze_baseline.py`）。
没有冻结基线就没有"更好"可言。

## 2. 两条严格对照臂

| 臂 | 数据 | 初始化 | 目的 |
|---|---|---|---|
| **R3A** | Dataset V3 | 冻结 R1 权重 `029f80a5…` 微调 | 复用 R1 已学到的均线视觉特征 |
| **R3B** | Dataset V3 | 官方 `yolo11s.pt` `85a76fe8…` | 判断 R1 权重是否带历史标签偏见 |

默认不从 R2 单线继续。R2 只作对照。

两臂必须相同：Dataset V3、train/val 划分、窗口协议（W12–19）、renderer、imgsz 960、
epochs 40、patience 10、batch 8、AdamW、rect、seed、增强配置、评估配置。

## 3. 增强（沿用冻结配方，全 0）

`hsv_* / degrees / shear / perspective / flipud / fliplr / mosaic / mixup / cutmix /
copy_paste / erasing` 全为 0；`translate 0.02`、`scale 0.1` 为 R1 原值，**不得改动**。
翻转会破坏时间方向与红绿 K 线语义；mosaic/mixup 会拼接不连续行情。

## 4. 快速筛选 → 正式晋升

- 快速筛选：R3A / R3B 各一个固定 seed（0），跑冻结 val + pre-holdout 连续 canary，
  只用于剔除明显失败或确认明显更优的方向；
- 正式晋升：优胜者补 ≥3 seed，报均值 / 标准差 / 最差 seed，再用一个**未参与调参**的
  独立 pre-holdout 时间块复验 + 人工事件抽查。

## 5. 每次训练必须记录

```text
yoyo-trading commit / fable-trading source commit
Dataset manifest SHA / Dataset content SHA
初始化权重 SHA / 完整训练配置 / 依赖版本与硬件 / seed
最佳 epoch / 训练日志 / best.pt SHA / 评估配置 SHA / 失败或中断原因
```

## 6. 算力

训练一律走局域网 RTX 3060（`zzc@192.168.1.4`，`scripts/train_w20_midbox_on_3060.sh`）。
Mac 只做数据、评估、决策。**当前 3060 ping 通但 SSH 无响应，训练臂被卡住。**
Mac MPS 跑完整 W12–19 训练实测过慢，历史上已放弃过一次，不作为默认替代。

## 7. 单变量纪律

一次实验只改一个变量。首轮变量 = **数据**（Dataset V3 的标签清理与 shape-NO 硬负例）。
窗口、renderer、标签语义、超参数、模型结构在首轮 R3A/R3B 对照完成前一律不动。
