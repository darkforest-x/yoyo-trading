"""Symbol helpers: internal fable names ↔ OKX instId."""
from __future__ import annotations

import math
from typing import Any


def to_okx_inst_id(symbol: str) -> str:
    """BTC_USDT_SWAP -> BTC-USDT-SWAP ; BTC_USDT -> BTC-USDT."""
    s = str(symbol).upper().strip()
    if s.endswith("_USDT_SWAP"):
        base = s[: -len("_USDT_SWAP")]
        return f"{base}-USDT-SWAP"
    if s.endswith("_USDT"):
        base = s[: -len("_USDT")]
        return f"{base}-USDT"
    if "-" in s:
        return s
    return s.replace("_", "-")


def round_size(sz: float, lot_sz: float, min_sz: float) -> str:
    """Round contract size down to lotSz, enforce minSz."""
    if lot_sz <= 0:
        lot_sz = 1.0
    n = math.floor(sz / lot_sz + 1e-12) * lot_sz
    if n < min_sz:
        n = min_sz
    # trim float noise
    if abs(n - round(n)) < 1e-9:
        return str(int(round(n)))
    text = f"{n:.8f}".rstrip("0").rstrip(".")
    return text


def size_for_notional(
    notional_usdt: float,
    mark_px: float,
    instrument: dict[str, Any],
) -> str:
    """Convert target notional (USDT) to contract sz for a linear USDT-margined swap.

    OKX linear swap: ctVal is coin per contract; notional ≈ sz * ctVal * mark_px.
    """
    ct_val = float(instrument.get("ctVal") or 1)
    lot_sz = float(instrument.get("lotSz") or 1)
    min_sz = float(instrument.get("minSz") or lot_sz)
    if mark_px <= 0 or ct_val <= 0:
        raise ValueError("bad mark/ctVal")
    raw = notional_usdt / (ct_val * mark_px)
    return round_size(raw, lot_sz, min_sz)


def round_price(px: float, tick_sz: str | float) -> float:
    """Round price to exchange tick. tick_sz as str preferred (avoids 1e-05 float trap)."""
    raw = str(tick_sz if tick_sz not in (None, "") else "0.01")
    tick = float(raw)
    if tick <= 0:
        return px
    n = round(px / tick) * tick
    # decimals from string form only — float(0.00001) stringifies to '1e-05' and used to
    # collapse prices to 0.0 via round(..., 0).
    if "e" in raw.lower() or "E" in raw:
        # scientific: 1e-5 -> 5 decimals
        try:
            decimals = abs(int(raw.lower().split("e")[1]))
        except (IndexError, ValueError):
            decimals = 8
    elif "." in raw:
        decimals = len(raw.split(".")[-1].rstrip("0") or "0")
        # keep full trailing zeros of tick ("0.00001" -> 5)
        decimals = len(raw.split(".")[-1])
    else:
        decimals = 0
    decimals = max(0, min(12, decimals))
    return round(n, decimals)

