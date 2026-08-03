"""Universe taxonomy helpers.

STOCKISH_BASES: tokenized equity/ETF swaps on OKX. Pre-registered rule
(2026-07-12, PROJECT_STATUS): these keep being LOGGED by forward tracking
but are excluded from the crypto verdict counts and reported as a side
channel. Single source of truth -- import from here, do not copy the list.
"""
STOCKISH_BASES = frozenset({
    "AAPL", "ADBE", "AMAT", "AMD", "AMZN", "ASML", "AVGO", "COIN", "CRCL",
    "GOOGL", "HOOD", "INTC", "META", "MSFT", "MSTR", "NVDA", "PLTR", "TSLA",
    "QQQ", "SPY", "ALAB", "APLD", "MU", "ORCL", "NFLX", "BABA", "EWJ", "NEO0",
    "HIMS", "SBET", "RDDT", "TSM", "OPEN", "GME", "AMC", "UNH", "LLY", "FIG",
})


def is_stockish(symbol: str) -> bool:
    return symbol.split("_", 1)[0] in STOCKISH_BASES
