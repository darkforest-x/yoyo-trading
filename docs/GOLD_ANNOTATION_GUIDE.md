# Gold 标注指南（Label Studio V1）

Owner 用完整走势判断形态；模型只学 decision 及以前的 W30 局部图。  
你只点 **P / N / I**，正例再点 **START / END**；程序统一生成含完整影线 + 六条均线的框。

## 启动 Label Studio

本机已经装过：`fable-trading/.venv_label_studio`（label-studio **1.13.1**，2025-07-10）。不要再 pip。

```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/Users/zhangzc/yoyo-trading \
/Users/zhangzc/fable-trading/.venv_label_studio/bin/label-studio start \
  --internal-host 127.0.0.1 --port 8081
```

或：`bash /Users/zhangzc/fable-trading/scripts/start_label_studio_review.sh`

浏览器打开 `http://127.0.0.1:8081/`。

## 导入任务

1. 新建项目，Labeling Setup 粘贴 `configs/labelstudio_gold_v1.xml`
2. Import `datasets/gold_labelstudio_v1/tasks.json`（当前 30 张示例）
3. 若图片不显示：Settings → Cloud Storage → add Local Files，根目录设为  
   `/Users/zhangzc/yoyo-trading/datasets/gold_labelstudio_v1`

## 怎么标

| 键 | 含义 |
|---|---|
| `P` | POSITIVE：整体图是目标形态，且局部图在 decision 前能指出密集核心 |
| `N` | NEGATIVE：普通交叉 / 平行 / 已张开 / 纵轴伪粘合 / 后面跌但当时没结构 |
| `I` | IGNORE：看不清、decision 不对、灰区。宁可 I 不要硬标 |
| `1` / `2` | 只在 **右侧局部图** 点 START / END（吸附到最近 K） |

左边是参考图（可以看未来，青虚线是 decision，紫带是之后）。  
右边是模型真实输入（W30，无未来，底部有 0–29 和点击条）。

不要只因为后面暴跌就把没有结构的交叉标成正。

## 标注保存在哪

Label Studio 自己的项目库。导出 JSON 后转成金标：

```bash
PYTHONPATH=. python3 tools/convert_labelstudio_export.py \
  --export /path/to/export.json \
  --out datasets/gold_v1.jsonl \
  --existing datasets/gold_v1.jsonl
```

重复导入同一 `gold_id` 不会复制。POSITIVE 缺 START/END 会失败。

## 生成 YOLO 数据集

```bash
PYTHONPATH=. python3 tools/build_gold_yolo_dataset.py \
  --gold datasets/gold_v1.jsonl \
  --out-dir datasets/gold_yolo_v1
PYTHONPATH=. python3 tools/audit_gold_dataset.py \
  --gold datasets/gold_v1.jsonl \
  --yolo-dir datasets/gold_yolo_v1
```

参考图永远不会进入 `images/`。local 图截止 decision。holdout（≥2026-05-04）不读。

## 重建任务批次

```bash
PYTHONPATH=. python3 tools/build_gold_candidate_pool.py --n 30 --seed 0
PYTHONPATH=. python3 tools/build_labelstudio_tasks.py
```

候选来源对你隐藏。不用未来收益或模型 conf 预筛。
