#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PACK = Path(__file__).resolve().parents[1] / "reviews/v3_dispute_r1"
PUBLIC = PACK / "public"
OUT = PACK / "resolutions.jsonl"

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(PUBLIC), **k)
    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)
    def _json(self, code, payload):
        b = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if urlparse(self.path).path in ("/", "/index.html"):
            self.path = "/gallery.html"
        return super().do_GET()
    def do_POST(self):
        if urlparse(self.path).path != "/api/label":
            return self._json(404, {"error": "no"})
        n = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(n).decode() or "{}")
        data["at"] = datetime.now(timezone.utc).isoformat()
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        self._json(200, {"ok": True})

if __name__ == "__main__":
    print("open http://127.0.0.1:8767/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8767), H).serve_forever()
