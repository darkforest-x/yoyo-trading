#!/usr/bin/env python3
"""8768 review server: left context, right W10 overlay, A/N/I/[ / ]/S."""

from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import pandas as pd

from yoyo.datasets.gold_render import load_causal_prefix, render_context
from yoyo.datasets.legacy_gold_migration.canonical import no_signal_slot, w10_fields
from yoyo.datasets.legacy_gold_migration.io import add_common_flags, load_config, read_jsonl
from yoyo.datasets.legacy_gold_migration.renderer import post_bars_allowed, render_w10
from yoyo.datasets.legacy_gold_migration.review_queue import apply_action, load_state, save_review
from yoyo.datasets.window_render import SourceError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs/fixed_w10_core4_confirm1_v1.json"
HTML = Path(__file__).with_name("gold_migrate_review.html")


def _png(img) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


class ReviewHandler(SimpleHTTPRequestHandler):
    cfg: dict
    fable: Path
    queue: list
    by_id: dict
    state_path: Path
    history_dir: Path
    holdout: pd.Timestamp

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
        if path in ("/", "/index.html", "/review"):
            return self._bytes(200, HTML.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/review/state":
            saved = load_state(self.state_path)
            slim = []
            for row in self.queue:
                item = {
                    "gold_id": row.get("gold_id"),
                    "symbol": row.get("symbol"),
                    "shape_label": row.get("shape_label"),
                    "migration_status": row.get("queue_status") or row.get("migration_status"),
                    "source_annotation_type": row.get("source_annotation_type"),
                    "core_length": row.get("core_length"),
                    "legacy_decision_offset": row.get("legacy_decision_offset"),
                    "core_start_bar": row.get("suggested_core_start_bar") or row.get("core_start_bar"),
                    "core_end_exclusive_bar": row.get("suggested_core_end_exclusive_bar")
                    or row.get("core_end_exclusive_bar"),
                    "conflict": row.get("migration_status") == "CONFLICT",
                    "saved": saved.get(row["gold_id"]) if row.get("gold_id") in saved else None,
                }
                slim.append(item)
            return self._json(200, {"tasks": slim, "n_saved": len(saved)})
        if path == "/api/export":
            saved = list(load_state(self.state_path).values())
            return self._json(200, {"n": len(saved), "rows": saved})
        if path.startswith("/img/") and path.endswith(".png"):
            return self._image(path)
        self.send_error(404)

    def _image(self, path: str) -> None:
        q = parse_qs(urlparse(self.path).query)
        kind = "w10"
        if "/context/" in path:
            kind = "context"
        gid = path.rsplit("/", 1)[-1].removesuffix(".png")
        row = self.by_id.get(gid)
        if not row:
            return self.send_error(404)
        shift = int((q.get("shift") or ["0"])[0])
        try:
            img = self._render(row, kind, shift)
        except (SourceError, ValueError, FileNotFoundError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._bytes(200, _png(img), "image/png")

    def _render(self, row: dict, kind: str, shift: int):
        src = row.get("source_path")
        if not src:
            raise FileNotFoundError("no source_path")
        path = self.fable / src
        core_s = row.get("suggested_core_start_bar") or row.get("core_start_bar")
        core_e = row.get("suggested_core_end_exclusive_bar") or row.get("core_end_exclusive_bar")
        if core_s is not None:
            core_s = int(core_s) + shift
            core_e = int(core_s) + 4
            mapped = w10_fields(core_s, core_e)
        else:
            mapped = no_signal_slot(int(row["decision_bar"]) + shift)
            core_s = mapped["slot_start_bar"]
            core_e = mapped["slot_end_exclusive_bar"]
        last = int(mapped["window_end_exclusive_bar"]) - 1
        post = 0
        if kind == "context":
            post = post_bars_allowed(row["decision_time"], self.holdout, int(self.cfg["review_context_post_bars"]))
            last = max(last, int(row.get("decision_bar") or last) + post)
        frame = load_causal_prefix(path, last)
        times = pd.to_datetime(frame["open_time"], utc=True)
        if (times >= self.holdout).any():
            raise ValueError("holdout")
        if kind == "context":
            from tempfile import NamedTemporaryFile

            import numpy as np

            with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                render_context(
                    frame,
                    int(mapped["confirmation_bar"]),
                    int(core_s) if core_s is not None else None,
                    int(core_e) - 1 if core_e is not None else None,
                    pre_bars=int(self.cfg["review_context_pre_bars"]),
                    post_bars=post,
                    out_path=tmp_path,
                )
                arr = cv2.imdecode(np.frombuffer(tmp_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            finally:
                tmp_path.unlink(missing_ok=True)
            if arr is None:
                raise ValueError("context render failed")
            return arr
        window = frame.iloc[mapped["window_start_bar"] : mapped["window_end_exclusive_bar"]].reset_index(drop=True)
        img, _tf = render_w10(window, y_pad_frac=float(self.cfg["y_pad_frac"]), overlay=True)
        return img

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/review/save":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        gid = body.get("gold_id")
        row = self.by_id.get(gid)
        if not row:
            return self._json(400, {"ok": False, "error": "unknown gold_id"})
        try:
            saved = apply_action(row, str(body.get("action")), body.get("core_start_bar"))
            state = save_review(self.state_path, self.history_dir, saved)
        except ValueError as exc:
            return self._json(400, {"ok": False, "error": str(exc)})
        return self._json(200, {"ok": True, "n": len(state)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_flags(parser, default_config=DEFAULT_CFG)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--queue", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["dataset_root"])
    queue_path = args.queue or root / "review" / "queue.jsonl"
    queue = read_jsonl(queue_path) if queue_path.exists() else []
    if args.limit:
        queue = queue[: args.limit]
    ReviewHandler.cfg = cfg
    ReviewHandler.fable = Path(cfg["paths"]["fable_root"])
    ReviewHandler.queue = queue
    ReviewHandler.by_id = {r["gold_id"]: r for r in queue if r.get("gold_id")}
    ReviewHandler.state_path = root / "review" / "state.jsonl"
    ReviewHandler.history_dir = root / "review" / "history"
    ReviewHandler.holdout = pd.Timestamp(cfg["holdout_start"])
    print(f"open http://127.0.0.1:{args.port}/  mode=legacy_migration_review  n={len(queue)}", flush=True)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "n": len(queue)}))
        return 0
    ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
