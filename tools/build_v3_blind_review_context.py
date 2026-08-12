#!/usr/bin/env python3
"""Owner-requested context charts for the frozen V3 blind pack.

Training PNGs stay untouched. These extra charts add a price axis, a cyan
model-input band, and a purple post-decision band. Bars at or after holdout
are never loaded. This pass is future-assisted review, not causal L1 pixels.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from yoyo.datasets.window_render import load_prefix
from yoyo.layers.l1_detection.data import ALL_MA_COLS, add_mas

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "reviews/v3_blind_r1"
SOURCE = json.loads((ROOT / "configs/source_repo.json").read_text())
FABLE = Path(SOURCE["source_repo"])
HOLDOUT = pd.Timestamp(SOURCE["holdout_start"])
FUTURE_BARS = 24
PRE_BARS = 20
MANIFEST = {
    json.loads(line)["sample_id"]: json.loads(line)
    for line in (ROOT / "datasets/dataset_v3_gold_core_v1/manifest.jsonl").read_text().splitlines()
    if line
}


def render_one(key: dict, out_path: Path) -> dict:
    sample = MANIFEST[key["sample_id"]]
    decision = int(sample["decision_bar"])
    win_start = int(sample["window_start_bar"])
    source_csv = FABLE / sample["source_path"]
    required_end = decision + FUTURE_BARS
    frame = add_mas(load_prefix(source_csv, required_end))
    times = pd.to_datetime(frame["open_time"], utc=True)
    allowed = times < HOLDOUT
    frame = frame.loc[allowed].reset_index(drop=True)
    times = pd.to_datetime(frame["open_time"], utc=True)
    if decision >= len(frame):
        raise RuntimeError(f"{key['sample_id']} decision clipped by holdout")
    lo = max(0, win_start - PRE_BARS)
    hi = min(len(frame) - 1, decision + FUTURE_BARS)
    segment = frame.iloc[lo : hi + 1].reset_index(drop=True)
    local_decision = decision - lo
    local_win = win_start - lo
    cst = times.iloc[lo : hi + 1].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    x = mdates.date2num(cst)
    o = segment["open"].to_numpy(float)
    h = segment["high"].to_numpy(float)
    low = segment["low"].to_numpy(float)
    c = segment["close"].to_numpy(float)
    width = (x[1] - x[0]) * 0.70 if len(x) > 1 else 0.01
    up = c >= o
    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    ax.vlines(x, low, h, color="#888888", lw=0.8, zorder=2)
    ax.bar(x[up], (c - o)[up], width, bottom=o[up], color="#26a69a", zorder=3)
    ax.bar(x[~up], (o - c)[~up], width, bottom=c[~up], color="#ef5350", zorder=3)
    colors = {
        "sma20": "#303f9f",
        "ema20": "#ef6c00",
        "sma60": "#039be5",
        "ema60": "#7cb342",
        "sma120": "#8e24aa",
        "ema120": "#d81b60",
    }
    for column in ALL_MA_COLS:
        ax.plot(x, segment[column], color=colors[column], lw=1.0, alpha=0.9)
    if 0 <= local_win <= local_decision < len(x):
        ax.axvspan(x[local_win], x[local_decision] + width / 2, color="#00acc1", alpha=0.10)
    if local_decision + 1 < len(x):
        ax.axvspan(x[local_decision + 1] - width / 2, x[-1] + width / 2, color="#7e57c2", alpha=0.08)
    ax.axvline(x[local_decision], color="#00838f", lw=2.0, ls="--")
    ax.set_title(
        f"{key['review_id']}  decision {cst.iloc[local_decision]:%Y-%m-%d %H:%M} CST\n"
        "青带=模型输入（decision及以前）· 紫带=仅供你看的未来 · 不要用涨跌当 L1 标签",
        loc="left",
        fontsize=11,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.grid(alpha=0.15)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    future_n = max(0, hi - decision)
    return {
        "review_id": key["review_id"],
        "future_bars_shown": int(future_n),
        "holdout_clipped": bool(times.iloc[-1] >= HOLDOUT - pd.Timedelta(minutes=15)),
    }


def main() -> int:
    keys = [
        json.loads(line)
        for line in (PACK / "answer_key.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    out_dir = PACK / "public" / "images_future"
    if out_dir.exists():
        for path in out_dir.glob("*.png"):
            path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = []
    for i, key in enumerate(keys, 1):
        audit.append(render_one(key, out_dir / f"{key['review_id']}.png"))
        if i % 25 == 0 or i == len(keys):
            print(f"context [{i}/{len(keys)}]", flush=True)
    receipt = {
        "protocol": "yoyo_v3_blind_review_r1_future_assisted_20260813",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n": len(audit),
        "future_bars_requested": FUTURE_BARS,
        "pre_bars": PRE_BARS,
        "holdout_start": HOLDOUT.isoformat(),
        "holdout_rows_materialized": 0,
        "label_name": "未来辅助语义裁决",
        "not_precision": True,
        "audit": audit,
    }
    (PACK / "context_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: receipt[k] for k in receipt if k != "audit"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
