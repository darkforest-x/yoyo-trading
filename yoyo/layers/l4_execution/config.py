"""Executor knobs (no secrets). Trading environment is in the keys file."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from yoyo.contracts.paths import data_path, data_root  # noqa: F401

PROJECT = data_root()
DEFAULT_CONFIG_PATH = PROJECT / "data" / "executor_config.json"
# Relative paths so example JSON is portable across machines.
DEFAULT_KILL_PATH = "data/executor_KILL"
DEFAULT_LEDGER = "data/executor_ledger.jsonl"
DEFAULT_FORWARD_LOG = "data/forward_log.csv"

# Mainline barriers (freeze name tp5_sl2; HANDOFF TP5/SL2).
TP_ATR_MULT = 5.0
SL_ATR_MULT = 2.0


@dataclass
class ExecutorConfig:
    """Paper/live executor knobs (secrets stay in okx keys file).

    Sizing (owner 2026-07-17):
      sizing_mode=equity_times_leverage → target gross notional = equity * leverage
      e.g. 100U equity, leverage 3 → ~300U total notional budget (cross).
      Concurrent slots share the remaining budget (not 3x each).
      sizing_mode=fixed → always use notional_usdt per entry (legacy).
    """

    max_concurrent: int = 1
    notional_usdt: float = 20.0  # fixed mode, or floor/fallback when equity missing
    leverage: int = 3
    # equity_times_leverage | fixed
    sizing_mode: str = "equity_times_leverage"
    # min notional per entry (USDT); skip if remaining budget below this
    min_notional_usdt: float = 5.0
    max_consecutive_losses: int = 5
    # validated strategy exits at 72 bars (18h); live must too
    timeout_hours: float = 18.0
    # A forward row stays "open" for up to the 18h barrier horizon, but the edge
    # is the launch moment: refuse to open positions on signals older than this.
    # Arithmetic (2026-07-20 tip path): age counts from the signal bar OPEN, so
    # a tip detection is already 16 min old at the first possible pulse (:01/
    # :16/:31/:46) and the 344-symbol scan adds up to ~7 min before the log is
    # written. 30 = 15 (bar) + 7 (pulse+scan) + headroom; 20 would drop real
    # tip signals scanned late in the pulse, and the pre-tip pipeline could not
    # record ANYTHING younger than 31 min. Align with TG + dashboard verdict.
    max_signal_age_min: int = 30
    poll_seconds: int = 30
    # Retry OCO bracket this many times after market entry (0 = no retry).
    bracket_retries: int = 2
    bracket_retry_sleep_sec: float = 1.5
    td_mode: str = "cross"  # full cross margin
    kill_switch_file: str = DEFAULT_KILL_PATH
    forward_log: str = DEFAULT_FORWARD_LOG
    ledger: str = DEFAULT_LEDGER
    # Only take signals with score >= row threshold (already filtered in log)
    # and status in these sets:
    open_statuses: tuple[str, ...] = ("open", "pending")
    require_score_ge_threshold: bool = True

    @classmethod
    def load(cls, path: Path | None = None) -> "ExecutorConfig":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text(encoding="utf-8"))
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        if "open_statuses" in kwargs and isinstance(kwargs["open_statuses"], list):
            kwargs["open_statuses"] = tuple(kwargs["open_statuses"])
        return cls(**kwargs)

    def save_example(self, path: Path | None = None) -> Path:
        p = Path(path) if path else DEFAULT_CONFIG_PATH.with_suffix(".example.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return p


def kill_switch_active(cfg: ExecutorConfig) -> bool:
    p = Path(cfg.kill_switch_file)
    if not p.is_absolute():
        p = PROJECT / p
    return p.exists()
