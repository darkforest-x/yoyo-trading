# VERDICT — yoyo_r3a_v3gold_ft_r1/weights/best.pt

**状态：R3A 训练完成；静态 val 仅作过程记录。禁止 promote / ACTIVE / 部署。holdout 未动用。**

写于 2026-08-12。本文件记录权重身份与静态指标；晋升门只认 pre-holdout 连续 canary
（协议 `owner_short_gold_center_recent2d_v1_20260811`）与后续 tip/因果评估。

## 绑定

| 项 | 值 |
|---|---|
| run name | `yoyo_r3a_v3gold_ft_r1` |
| best.pt SHA-256 | `3042c16d766ede358e89db0a33fa2983ec7a2d22893731d2b34ed44c39f010f7` |
| best.pt bytes | 19185818 |
| base (R1 hardneg) SHA-256 | `029f80a52b5beda2e32f6bb5a188a39fd7f74fe0a3fef4dffa79ae620384f537` |
| dataset | `dataset_v3_gold_core_v1` |
| dataset manifest SHA-256 | `6d21a9b960cd314d2a2db70d34424193530fa339bcf7eda273a68cee217b45d8` |
| finetune | True · 40ep / patience10 / batch8 / seed0 / imgsz960 |

## 静态 val（results.csv，非裁决）

| | epoch | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| **best** | 33 | 0.8279 | 0.8573 | **0.9032** | 0.7969 |
| final | 40 | 0.8146 | 0.8483 | 0.8951 | 0.7893 |

对照冻结基线（同表，不得直接比作升级）：R1 static mAP50=0.8980；本 run best mAP50=0.9032。

## holdout 记账

- **holdout 读取：0 次**
- 训练与 val 均未触及 ≥2026-05-04

## 下一步（进行中）

1. pre-holdout 连续 canary（am 2026-05-03 00:00→12:00，W12–19，conf 0.25，215 币）
2. 与 frozen baseline 的 baseline_v1_ft / R1 事件密度对照
3. **未**授权 holdout、**未** promote

## 路径

- weights: `analysis/output/lsv2_stageb/yoyo_r3a_v3gold_ft_r1/weights/best.pt`
- manifest: `analysis/output/lsv2_stageb/yoyo_r3a_v3gold_ft_r1/manifest.json`
