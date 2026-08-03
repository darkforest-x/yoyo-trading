"""Telegram notifier. Credentials live in data/tg_config.json (gitignored,
owner-created; agents never read or echo the token):

    {"bot_token": "...", "chat_id": "..."}

Falls back to env TG_BOT_TOKEN / TG_CHAT_ID. Missing config -> warn + no-op,
so pipelines never crash because of notification plumbing.
"""
from __future__ import annotations

import json
import mimetypes
import os
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from yoyo.contracts.paths import data_path, data_root  # noqa: F401

CONFIG_PATH = data_path("data", "tg_config.json")


def _load() -> tuple[str, str] | None:
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
        return cfg["bot_token"], str(cfg["chat_id"])
    tok, chat = os.environ.get("TG_BOT_TOKEN"), os.environ.get("TG_CHAT_ID")
    if tok and chat:
        return tok, chat
    return None


def send(text: str) -> bool:
    """Send a Telegram message (HTML parse mode). Returns delivery success."""
    creds = _load()
    if creds is None:
        print("tg_notify: no config (data/tg_config.json) -- message not sent")
        return False
    token, chat_id = creds
    data = urllib.parse.urlencode({
        "chat_id": chat_id, "text": text[:4000],
        "parse_mode": "HTML", "disable_web_page_preview": "true",
    }).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage", data=data)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read()).get("ok", False)
    except Exception as exc:  # noqa: BLE001 -- notification must never crash the caller
        print(f"tg_notify: send failed: {exc}")
        return False


def send_photo(image_path: Path, caption: str = "") -> bool:
    """Send a local image with optional HTML caption (Telegram limit ~1024)."""
    creds = _load()
    if creds is None:
        print("tg_notify: no config (data/tg_config.json) -- photo not sent")
        return False
    path = Path(image_path)
    if not path.exists():
        print(f"tg_notify: photo missing: {path}")
        return False
    token, chat_id = creds
    boundary = f"----fable{uuid.uuid4().hex}"
    caption_safe = (caption or "")[:1024]
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    file_bytes = path.read_bytes()

    def part(name: str, value: bytes, filename: str | None = None, content_type: str | None = None) -> bytes:
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            disposition += f'; filename="{filename}"'
        headers = [disposition.encode()]
        if content_type:
            headers.append(f"Content-Type: {content_type}".encode())
        return b"\r\n".join([f"--{boundary}".encode(), *headers, b"", value, b""])

    body = b"".join([
        part("chat_id", str(chat_id).encode()),
        part("caption", caption_safe.encode("utf-8")),
        part("parse_mode", b"HTML"),
        part("photo", file_bytes, filename=path.name, content_type=mime),
        f"--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendPhoto",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()).get("ok", False)
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300] if hasattr(exc, "read") else b""
        print(f"tg_notify: sendPhoto failed: {exc} {detail!r}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"tg_notify: sendPhoto failed: {exc}")
        return False


def send_document(file_path: Path, caption: str = "") -> bool:
    """Send a local file as a document.

    Reports and appendices run to tens of KB, well past sendMessage's 4096-char
    ceiling, and splitting a markdown report across a dozen chat bubbles makes it
    unreadable. sendDocument keeps it one openable file.
    """
    creds = _load()
    if creds is None:
        print("tg_notify: no config (data/tg_config.json) -- document not sent")
        return False
    path = Path(file_path)
    if not path.exists():
        print(f"tg_notify: document missing: {path}")
        return False
    token, chat_id = creds
    boundary = f"----fable{uuid.uuid4().hex}"
    # mimetypes has no entry for .md, so a report went out as octet-stream with no
    # charset and Telegram decoded the Chinese with whatever default it picked.
    # Text needs its charset stated in the type or the viewer has to guess.
    TEXT_EXT = {".md": "text/markdown", ".txt": "text/plain", ".csv": "text/csv",
                ".json": "application/json", ".html": "text/html"}
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        mime = f"{TEXT_EXT[ext]}; charset=utf-8"
    else:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    file_bytes = path.read_bytes()

    def part(name: str, value: bytes, filename: str | None = None,
             content_type: str | None = None) -> bytes:
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            disposition += f'; filename="{filename}"'
        headers = [disposition.encode()]
        if content_type:
            headers.append(f"Content-Type: {content_type}".encode())
        return b"\r\n".join([f"--{boundary}".encode(), *headers, b"", value, b""])

    body = b"".join([
        part("chat_id", str(chat_id).encode()),
        part("caption", (caption or "")[:1024].encode("utf-8")),
        part("parse_mode", b"HTML"),
        part("document", file_bytes, filename=path.name, content_type=mime),
        f"--{boundary}--\r\n".encode(),
    ])
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendDocument",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read()).get("ok", False)
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300] if hasattr(exc, "read") else b""
        print(f"tg_notify: sendDocument failed: {exc} {detail!r}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"tg_notify: sendDocument failed: {exc}")
        return False


BARK_CONFIG_PATH = data_path("data", "bark_config.json")


def _load_bark() -> str | None:
    """Load Bark device key from data/bark_config.json or BARK_KEY env.

    bark_config.json example (gitignored):
        {"key": "xxxxxxxxxxxx"}
    """
    if BARK_CONFIG_PATH.exists():
        try:
            cfg = json.loads(BARK_CONFIG_PATH.read_text())
            k = cfg.get("key") or cfg.get("device_key")
            if k:
                return str(k).strip()
        except Exception:  # noqa: BLE001
            pass
    k = os.environ.get("BARK_KEY")
    if k:
        return k.strip()
    return None


def bark_send(title: str, body: str = "", *, group: str = "fable", level: str = "active",
              sound: str = "", icon: str = "", url: str = "", image: str = "") -> bool:
    """Push to Bark (iOS). Returns True on HTTP 200 from server.

    Uses https://api.day.app/<key>/<title>/<body> or JSON /push when extra fields needed.
    Falls back to env/key config; missing config -> no-op + print.
    """
    key = _load_bark()
    if not key:
        print("bark_notify: no key (data/bark_config.json or BARK_KEY) -- message not sent")
        return False
    title = (title or "fable")[:256]
    body = (body or "")[:4000]
    # Build payload. Prefer JSON /push to carry group/level/sound/url/image.
    data = {
        "title": title,
        "body": body,
        "group": group,
        "level": level or "active",
    }
    if sound:
        data["sound"] = sound
    if icon:
        data["icon"] = icon
    if url:
        data["url"] = url
    if image:
        data["image"] = image
    body_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.day.app/{key}/push",
        data=body_bytes,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            try:
                j = json.loads(raw)
                return bool(j.get("code") == 200 or j.get("success") is True)
            except Exception:
                return resp.status == 200
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300] if hasattr(exc, "read") else b""
        print(f"bark_notify: push failed: {exc} {detail!r}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"bark_notify: push failed: {exc}")
        return False


if __name__ == "__main__":
    import sys
    ok = send(sys.argv[1] if len(sys.argv) > 1 else "fable-trading 通知链路测试 ✅")
    bark_send("测试", "Bark 链路就绪 ✅")
    raise SystemExit(0 if ok else 1)
