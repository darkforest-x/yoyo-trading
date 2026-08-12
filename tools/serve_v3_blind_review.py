#!/usr/bin/env python3
"""Local server for the V3 blind review pack. Does not serve the answer key."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "reviews/v3_blind_r1"
VALID = frozenset({"YES", "NO", "IGNORE"})


def load_verdicts(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        out[str(row["review_id"])] = str(row["verdict"])
    return out


class Handler(SimpleHTTPRequestHandler):
    pack: Path
    public: Path
    verdicts_path: Path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.public), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args), flush=True)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/verdicts":
            self._json(200, {"verdicts": load_verdicts(self.verdicts_path)})
            return
        if path in ("/", "/index.html"):
            self.path = "/gallery.html"
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/label":
            self._json(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        review_id = str(data.get("review_id", ""))
        verdict = str(data.get("verdict", ""))
        if not review_id or verdict not in VALID:
            self._json(400, {"error": "need review_id and YES|NO|IGNORE"})
            return
        self.verdicts_path.parent.mkdir(parents=True, exist_ok=True)
        with self.verdicts_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "review_id": review_id,
                        "verdict": verdict,
                        "at": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        labeled = load_verdicts(self.verdicts_path)
        self._json(200, {"ok": True, "n_labeled": len(labeled)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    public = args.pack / "public"
    if not (public / "gallery.html").exists():
        raise SystemExit(f"missing gallery under {public}; run tools/build_v3_blind_review_pack.py")
    Handler.pack = args.pack
    Handler.public = public
    Handler.verdicts_path = args.pack / "verdicts.jsonl"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"open http://127.0.0.1:{args.port}/", flush=True)
    print(f"verdicts -> {Handler.verdicts_path}", flush=True)
    print("answer key is NOT served", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
