#!/usr/bin/env python3
"""Simple Gold annotator: P/N/I + two clicks for START/END. No LS 1/2 dance."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from yoyo.datasets.gold_box import snap_x_to_bar
from yoyo.datasets.gold_schema import validate_gold
from yoyo.layers.l1_detection.render import IMG_WIDTH, MARGIN

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "datasets/gold_labelstudio_v1/tasks.json"
IMG = ROOT / "datasets/gold_labelstudio_v1/images"
HTML = Path(__file__).with_name("gold_annotate.html")
OUT = ROOT / "datasets/gold_v1.jsonl"


def load_saved() -> dict[str, dict]:
    if not OUT.exists():
        return {}
    out: dict[str, dict] = {}
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        out[row["gold_id"]] = row
    return out


def rewrite(rows: dict[str, dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()),
        encoding="utf-8",
    )


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args, flush=True)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, code: int, payload: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._bytes(200, HTML.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            saved = load_saved()
            slim = {
                gid: {
                    "shape_label": row["shape_label"],
                    "local_start_i": (
                        int(row["core_start_bar"]) - int(row["local_start_bar"])
                        if row.get("core_start_bar") is not None
                        else None
                    ),
                    "local_end_i": (
                        int(row["core_end_bar"]) - int(row["local_start_bar"])
                        if row.get("core_end_bar") is not None
                        else None
                    ),
                }
                for gid, row in saved.items()
            }
            tasks = json.loads(TASKS.read_text(encoding="utf-8"))
            return self._json(200, {"tasks": tasks, "saved": slim})
        if path == "/api/export":
            saved = load_saved()
            return self._json(200, {"n": len(saved), "rows": list(saved.values())})
        if path == "/api/snap":
            q = parse_qs(urlparse(self.path).query)
            x = float(q.get("x", ["0"])[0])
            n = int(q.get("n", ["30"])[0])
            return self._json(200, {"bar": snap_x_to_bar(x, n, IMG_WIDTH, MARGIN)})
        if path.startswith("/img/context/") and path.endswith(".png"):
            p = IMG / "context" / path.rsplit("/", 1)[-1]
            if p.is_file():
                return self._bytes(200, p.read_bytes(), "image/png")
        if path.startswith("/img/local/") and path.endswith(".png"):
            p = IMG / "local" / path.rsplit("/", 1)[-1]
            if p.is_file():
                return self._bytes(200, p.read_bytes(), "image/png")
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/save":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        tasks = {t["data"]["gold_id"]: t["data"] for t in json.loads(TASKS.read_text())}
        data = tasks.get(body.get("gold_id"))
        if not data:
            return self._json(400, {"ok": False, "error": "unknown gold_id"})
        shape = body.get("shape_label")
        row = {
            "gold_id": data["gold_id"],
            "symbol": data["symbol"],
            "timeframe": data.get("timeframe", "15m"),
            "source_repo": data.get("source_repo", "fable-trading"),
            "source_commit": data.get("source_commit"),
            "source_path": data["source_path"],
            "candidate_source": "label_batch_hidden",
            "decision_bar": int(data["decision_bar"]),
            "decision_time": data["decision_time"],
            "context_start_bar": data.get("context_start_bar"),
            "context_end_bar": data.get("context_end_bar"),
            "local_start_bar": int(data["local_start_bar"]),
            "local_end_bar": int(data["local_end_bar"]),
            "local_window_length": int(data["local_window_length"]),
            "shape_label": shape,
            "core_start_bar": None,
            "core_end_bar": None,
            "box_rule": "full_wicks_plus_six_ma",
            "box_status": "none",
            "reviewer": "owner",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "notes": None,
            "context_image_sha256": data.get("context_image_sha256"),
            "local_image_sha256": data.get("local_image_sha256"),
            "task_config_sha256": data.get("task_config_sha256"),
            "holdout_read": False,
        }
        if shape == "POSITIVE":
            a, b = body.get("local_start_i"), body.get("local_end_i")
            if a is None or b is None:
                return self._json(400, {"ok": False, "error": "POSITIVE 需要点两根 K"})
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            row["core_start_bar"] = int(data["local_start_bar"]) + a
            row["core_end_bar"] = int(data["local_start_bar"]) + b
            row["box_status"] = "owner_bar_range"
        try:
            validate_gold(row)
        except ValueError as exc:
            return self._json(400, {"ok": False, "error": str(exc)})
        saved = load_saved()
        saved[row["gold_id"]] = row
        rewrite(saved)
        return self._json(200, {"ok": True, "n": len(saved)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    print(f"open http://127.0.0.1:{args.port}/", flush=True)
    print(f"gold -> {OUT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
