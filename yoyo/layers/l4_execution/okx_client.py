"""Minimal OKX v5 REST client.

Keys: data/okx_demo_keys.json (gitignored). Optional field:
  "environment": "live" | "demo"   (default demo)

- demo → header x-simulated-trading: 1
- live → no simulated header (owner-authorized real account)

Endpoints used: instruments, ticker, balance, positions, place order,
place algo (OCO TP/SL). Nothing that withdraws.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from yoyo.contracts.paths import data_path, data_root  # noqa: F401

BASE = "https://www.okx.com"
SIMULATED_HEADER = ("x-simulated-trading", "1")
KEYS_PATH = data_path("data", "okx_demo_keys.json")
_USER_AGENT = "fable-trading/1.0 (+https://www.okx.com)"


def _prefer_ipv4() -> None:
    """Prefer A records: some VPS IPv6 egress is CF-blocked while IPv4 works."""
    if getattr(socket, "_fable_ipv4_patched", False):
        return
    _orig = socket.getaddrinfo

    def _wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
        res = _orig(*args, **kwargs)
        v4 = [r for r in res if r[0] == socket.AF_INET]
        return v4 or res

    socket.getaddrinfo = _wrapped  # type: ignore[assignment]
    socket._fable_ipv4_patched = True  # type: ignore[attr-defined]


_prefer_ipv4()


class OkxDemoError(RuntimeError):
    pass


# Back-compat alias
OkxError = OkxDemoError


def _load_keys(path: Path | None = None) -> dict[str, str]:
    keys_path = Path(path) if path else KEYS_PATH
    if not keys_path.exists():
        raise OkxDemoError(
            f"缺少 {keys_path}。"
            '格式: {"api_key":"..","secret_key":"..","passphrase":"..","environment":"live|demo"}'
        )
    k = json.loads(keys_path.read_text(encoding="utf-8"))
    missing = [f for f in ("api_key", "secret_key", "passphrase") if not k.get(f)]
    if missing:
        raise OkxDemoError(f"keys json 缺字段: {missing}")
    env = str(k.get("environment") or "demo").strip().lower()
    if env not in {"demo", "live", "simulated", "real"}:
        raise OkxDemoError(f"environment 必须是 demo|live，收到: {env}")
    if env in {"simulated"}:
        env = "demo"
    if env in {"real"}:
        env = "live"
    return {
        "api_key": str(k["api_key"]).strip(),
        "secret_key": str(k["secret_key"]).strip(),
        "passphrase": str(k["passphrase"]).strip(),
        "environment": env,
    }


def _ts() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class OkxDemoClient:
    """OKX REST client. environment from keys file: demo | live."""

    def __init__(self, *, keys_path: Path | None = None) -> None:
        self._keys_path = Path(keys_path) if keys_path else KEYS_PATH
        self._k = _load_keys(self._keys_path)
        self.environment = self._k["environment"]
        self.is_demo = self.environment == "demo"

    def _request(self, method: str, path: str, body: dict | None = None) -> dict[str, Any]:
        ts = _ts()
        body_str = json.dumps(body) if body else ""
        msg = f"{ts}{method}{path}{body_str}"
        sign = base64.b64encode(
            hmac.new(self._k["secret_key"].encode(), msg.encode(), hashlib.sha256).digest()
        ).decode()
        headers = {
            "OK-ACCESS-KEY": self._k["api_key"],
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self._k["passphrase"],
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        }
        if self.is_demo:
            headers[SIMULATED_HEADER[0]] = SIMULATED_HEADER[1]
        req = urllib.request.Request(
            BASE + path,
            data=body_str.encode() if body_str else None,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                out = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raw = exc.read()[:400]
            raise OkxDemoError(f"HTTP {exc.code}: {raw!r}") from exc
        if str(out.get("code")) != "0":
            raise OkxDemoError(
                f"OKX code {out.get('code')}: {out.get('msg')} / {out.get('data')}"
            )
        return out

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        if params:
            q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            path = f"{path}?{q}"
        return self._request("GET", path)

    def balance(self) -> dict:
        return self._get("/api/v5/account/balance")

    def usdt_equity(self) -> float:
        """USDT equity (eq) for sizing; falls back to totalEq / availEq."""
        raw = self.balance()
        data = (raw.get("data") or [{}])[0]
        # Prefer account totalEq (USD) when present
        try:
            te = data.get("totalEq")
            if te not in (None, ""):
                return float(te)
        except (TypeError, ValueError):
            pass
        for det in data.get("details") or []:
            if str(det.get("ccy") or "").upper() != "USDT":
                continue
            for k in ("eq", "eqUsd", "availEq", "cashBal"):
                try:
                    v = det.get(k)
                    if v not in (None, ""):
                        return float(v)
                except (TypeError, ValueError):
                    continue
        raise OkxDemoError("no USDT equity in balance response")

    def open_swap_notional_usd(self) -> float:
        """Sum |notionalUsd| of open SWAP positions."""
        total = 0.0
        for p in self.positions("SWAP"):
            try:
                pos = abs(float(p.get("pos") or 0))
            except (TypeError, ValueError):
                continue
            if pos <= 0:
                continue
            try:
                n = float(p.get("notionalUsd") or 0)
            except (TypeError, ValueError):
                n = 0.0
            total += abs(n)
        return total

    def account_config(self) -> dict:
        data = self._get("/api/v5/account/config").get("data") or []
        return data[0] if data else {}

    def pos_mode(self) -> str:
        """net_mode | long_short_mode"""
        return str(self.account_config().get("posMode") or "net_mode")

    def positions(self, inst_type: str = "SWAP") -> list[dict]:
        return self._get("/api/v5/account/positions", {"instType": inst_type}).get("data") or []

    def instrument(self, inst_id: str) -> dict:
        data = self._get(
            "/api/v5/public/instruments",
            {"instType": "SWAP", "instId": inst_id},
        ).get("data") or []
        if not data:
            raise OkxDemoError(f"unknown instrument {inst_id}")
        return data[0]

    def mark_px(self, inst_id: str) -> float:
        data = self._get("/api/v5/public/mark-price", {"instType": "SWAP", "instId": inst_id}).get(
            "data"
        ) or []
        if not data:
            raise OkxDemoError(f"no mark price for {inst_id}")
        return float(data[0]["markPx"])

    def ticker_last(self, inst_id: str) -> float:
        data = self._get("/api/v5/market/ticker", {"instId": inst_id}).get("data") or []
        if not data:
            raise OkxDemoError(f"no ticker for {inst_id}")
        return float(data[0]["last"])

    def set_leverage(self, inst_id: str, lever: str, *, mgn_mode: str = "cross") -> dict:
        return self._request(
            "POST",
            "/api/v5/account/set-leverage",
            {"instId": inst_id, "lever": str(lever), "mgnMode": mgn_mode},
        )

    def place_market(
        self,
        inst_id: str,
        side: str,
        size: str,
        *,
        td_mode: str = "cross",
        cl_ord_id: str | None = None,
        pos_side: str | None = None,
        reduce_only: bool = False,
    ) -> dict:
        body: dict[str, Any] = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "ordType": "market",
            "sz": size,
        }
        # hedge mode requires long|short; net mode uses net (or omit)
        if pos_side:
            body["posSide"] = pos_side
        if reduce_only:
            # closing orders must never be able to flip into a fresh position
            body["reduceOnly"] = "true"
        if cl_ord_id:
            body["clOrdId"] = cl_ord_id[:32]
        return self._request("POST", "/api/v5/trade/order", body)

    def place_bracket(
        self,
        inst_id: str,
        side: str,
        size: str,
        tp_px: float,
        sl_px: float,
        *,
        td_mode: str = "cross",
        pos_side: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "ordType": "oco",
            "sz": size,
            "tpTriggerPx": _fmt_px(tp_px),
            "tpOrdPx": "-1",
            "slTriggerPx": _fmt_px(sl_px),
            "slOrdPx": "-1",
        }
        if pos_side:
            body["posSide"] = pos_side
        # Prefer last price triggers for swap OCO (mark can fail band checks).
        body["tpTriggerPxType"] = "last"
        body["slTriggerPxType"] = "last"
        return self._request("POST", "/api/v5/trade/order-algo", body)



    def cancel_algo(self, inst_id: str, algo_id: str) -> dict:
        return self._request(
            "POST",
            "/api/v5/trade/cancel-algos",
            [{"instId": inst_id, "algoId": algo_id}],
        )


def _fmt_px(px: float) -> str:
    s = f"{px:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"
