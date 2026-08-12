#!/usr/bin/env python3
"""Materialize the frozen 10% draw as an anonymous Owner review pack.

Images are the exact V3 training pixels. No gold box, no POS/NEG in the
filename, order shuffled with a recorded seed. The answer key stays out of
the web root so the gallery cannot leak labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRAW = ROOT / "manifests/v3_blind_review_sample_v1.jsonl"
DEFAULT_OUT = ROOT / "reviews/v3_blind_r1"
SHUFFLE_SEED = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw", type=Path, default=DEFAULT_DRAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.draw.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rng = random.Random(args.shuffle_seed)
    order = list(range(len(rows)))
    rng.shuffle(order)

    public = args.out / "public"
    images = public / "images"
    if args.out.exists():
        shutil.rmtree(args.out)
    images.mkdir(parents=True)

    items = []
    keys = []
    for i, src_index in enumerate(order, 1):
        row = rows[src_index]
        review_id = f"{i:04d}"
        src = ROOT / row["image_path"]
        dst = images / f"{review_id}.png"
        shutil.copy2(src, dst)
        items.append({"review_id": review_id, "image": f"images/{review_id}.png"})
        keys.append(
            {
                "review_id": review_id,
                "sample_id": row["sample_id"],
                "kind": row["kind"],
                "split": row["split"],
                "stratum": row["stratum"],
                "symbol": row["symbol"],
                "decision_time": row["decision_time"],
                "source_image": row["image_path"],
                "source_image_sha256": sha256_file(src),
            }
        )

    items_path = public / "items.json"
    payload = {
        "protocol": "yoyo_v3_blind_review_r1_20260813",
        "question": (
            "这张局部图（decision 及以前）里，有没有你认的均线密集核心？"
            "不管后面涨跌、也不管值不值得开空。"
        ),
        "choices": ["YES", "NO", "IGNORE"],
        "n": len(items),
        "draw": str(args.draw.relative_to(ROOT)),
        "shuffle_seed": args.shuffle_seed,
        "items": items,
    }
    items_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    key_path = args.out / "answer_key.jsonl"
    key_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in keys),
        encoding="utf-8",
    )

    gallery_src = Path(__file__).with_name("v3_blind_review_gallery.html")
    shutil.copy2(gallery_src, public / "gallery.html")

    manifest = {
        "protocol": payload["protocol"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n": len(items),
        "draw_sha256": sha256_file(args.draw),
        "items_sha256": sha256_file(items_path),
        "answer_key_sha256": sha256_file(key_path),
        "shuffle_seed": args.shuffle_seed,
        "public_dir": str(public.relative_to(ROOT)),
        "answer_key": str(key_path.relative_to(ROOT)),
        "note": "Do not open answer_key.jsonl until the Owner finishes the pack.",
    }
    (args.out / "pack_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out / "README.md").write_text(
        (
            "# V3 盲审图包 r1\n\n"
            "打开方式：在仓库根执行\n\n"
            "```bash\n"
            "python3 tools/serve_v3_blind_review.py\n"
            "```\n\n"
            "浏览器进 `http://127.0.0.1:8766/`。\n\n"
            "快捷键：`Y` 有密集核心 · `N` 没有 · `I` 看不清 · `←/→` 翻页。\n"
            "不要打开 `answer_key.jsonl`。审完再说一声，我再对答案算一致率和 kappa。\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
