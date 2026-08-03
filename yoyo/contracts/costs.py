"""Trading-cost assumptions: the route table (single source of truth).

CLAUDE.md lists cost assumptions among the four things only the project owner
may change -- yet by 2026-07-16 they had grown ~30 independent copies across
src/ and scripts/, with SWAP_MAKER_COST=0.0006 hard-coded seven times. Nothing
contradicted anything (different routes legitimately cost differently), but no
single place said which number belongs to which route, and a fee change would
have meant hunting down every copy. Owner approved consolidation 2026-07-16.

Live code under src/ imports from here. Experiment scripts under scripts/ that
back published reports keep their inline copies ON PURPOSE: editing them would
break reproduction of the numbers those reports state.

All values are ROUND-TRIP fractions of notional (entry + exit combined),
including the slippage allowance noted per route. Owner decisions, do not tune.
"""
from __future__ import annotations

from typing import Final

# ---- Spot routes -----------------------------------------------------------
# Stage-3 base case: taker both sides + slippage allowance (P0 assumption
# was 0.002; stage 3 tightened to 0.003 after the PF 1.01 failure showed the
# strategy's breakeven sat near 0.30%).
SPOT_TAKER: Final = 0.003   # 0.15%/side incl. slippage
SPOT_MAKER: Final = 0.0016  # 0.08%/side limit orders, owner route D

# ---- Swap (perpetual) routes -- the mainline universe -----------------------
SWAP_MAKER: Final = 0.0006  # OKX swap maker 0.02%/side + slippage allowance
SWAP_TAKER: Final = 0.0010  # OKX swap taker 0.05%/side (fills always)

# ---- Legacy / reporting-only -------------------------------------------------
# P0-era blanket assumption; kept because judgment train.py reports net returns
# with it for continuity with published p2b numbers. Not used for decisions.
LEGACY_P0_ROUND_TRIP: Final = 0.002

# ---- Stage-3 backtest knobs --------------------------------------------------
BASE_COST: Final = SPOT_TAKER            # base case for acceptance checks
COST_SWEEP: Final = (0.002, 0.003, 0.004)

# Forward validation reports net at the mainline execution route.
FORWARD_COST: Final = SWAP_MAKER

# Return semantics are a type-level routing table, not aliases. A value tagged
# net_taker has already paid SWAP_TAKER; converting it to net_maker first restores
# gross, then applies SWAP_MAKER. Subtracting maker directly from net_taker is a
# double charge and is intentionally not exposed as an API.
RETURN_SEMANTIC_COST: Final = {
    "gross": 0.0,
    "net_taker": SWAP_TAKER,
    "net_maker": SWAP_MAKER,
}


def convert_return(value: float, *, source_semantics: str, target_semantics: str) -> float:
    """Convert gross/net-taker/net-maker through the unique gross bridge."""
    if source_semantics not in RETURN_SEMANTIC_COST:
        raise ValueError(f"unknown source return semantics {source_semantics!r}")
    if target_semantics not in RETURN_SEMANTIC_COST:
        raise ValueError(f"unknown target return semantics {target_semantics!r}")
    gross = float(value) + RETURN_SEMANTIC_COST[source_semantics]
    return gross - RETURN_SEMANTIC_COST[target_semantics]


def deduct_round_trip_cost_once(
    value: float,
    *,
    input_semantics: str,
    target_semantics: str,
) -> float:
    """Apply a route cost only to gross input; reject already-net input.

    Use :func:`convert_return` for legitimate net-route conversion. This stricter
    helper exists for report builders whose intended operation is "deduct cost";
    it makes a second deduction a visible error instead of a plausible number.
    """
    if input_semantics != "gross":
        raise ValueError(
            f"cost already included in {input_semantics!r}; convert routes instead of deducting again"
        )
    if target_semantics == "gross":
        raise ValueError("target_semantics must be a net route when deducting cost")
    return convert_return(
        value,
        source_semantics=input_semantics,
        target_semantics=target_semantics,
    )
