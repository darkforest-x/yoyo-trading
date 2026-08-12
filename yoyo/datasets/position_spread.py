"""Causal W12–19 crop placement: put the gold core left / middle / right.

The window always *ends* at the decision bar (no future K). Length stays in
12–19. That is the only degree of freedom: which W we pick.

Position of the core midpoint in the window:

    start = decision - (W - 1)
    rel   = (core_mid - start) / (W - 1)

Shorter W pushes a near-decision core left; longer W pushes it right.
A core sitting only a few bars before decision **cannot** reach the left
third at W>=12 — that is geometry, not a missing flag.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal

W_MIN = 12
W_MAX = 19
Bin = Literal["left", "middle", "right"]


def bin_of(rel: float) -> Bin:
    if rel < 1 / 3:
        return "left"
    if rel < 2 / 3:
        return "middle"
    return "right"


def rel_position(core_mid: float, decision: int, window_len: int) -> float:
    if window_len < 2:
        raise ValueError("window_len must be >= 2")
    start = decision - (window_len - 1)
    return (core_mid - start) / (window_len - 1)


def core_fits(box_start: int, box_end: int, decision: int, window_len: int) -> bool:
    start = decision - (window_len - 1)
    return start >= 0 and box_start >= start and box_end <= decision


def placements(
    box_start: int, box_end: int, decision: int, w_min: int = W_MIN, w_max: int = W_MAX
) -> list[dict]:
    """Every causal (W, bin) this gold box can occupy."""
    core_mid = (box_start + box_end) / 2.0
    out: list[dict] = []
    for length in range(w_min, w_max + 1):
        if not core_fits(box_start, box_end, decision, length):
            continue
        rel = rel_position(core_mid, decision, length)
        out.append(
            {
                "window_len": length,
                "win_start": decision - (length - 1),
                "win_end": decision,
                "rel": rel,
                "bin": bin_of(rel),
            }
        )
    return out


def assign_one(
    options: list[dict],
    counts: dict[str, int],
    preferred: Iterable[Bin] = ("left", "right", "middle"),
) -> dict | None:
    """Pick the option whose bin is most under-filled; prefer left, then right."""
    if not options:
        return None
    # among achievable bins, take the one with the lowest current count;
    # ties broken by preferred order, then by shorter W (more leftward).
    by_bin: dict[str, list[dict]] = {}
    for opt in options:
        by_bin.setdefault(opt["bin"], []).append(opt)
    target = min(
        by_bin,
        key=lambda b: (counts.get(b, 0), list(preferred).index(b) if b in preferred else 9),
    )
    chosen = min(by_bin[target], key=lambda o: (o["window_len"] if target == "left" else -o["window_len"]))
    return chosen


def _record(row: dict, pick: dict, opts: list[dict]) -> dict:
    return {
        **row,
        "spread_status": "ok",
        "chosen_window_len": pick["window_len"],
        "chosen_win_start": pick["win_start"],
        "chosen_win_end": pick["win_end"],
        "chosen_rel": round(pick["rel"], 4),
        "chosen_bin": pick["bin"],
        "achievable_bins": sorted({o["bin"] for o in opts}),
    }


def assign_all(positives: list[dict]) -> list[dict]:
    """Assign every left-capable core to left first; then balance the rest.

    Left is geometrically rare under causal W>=12. Starving it to 'balance'
    would throw away the only samples that can kill the center shortcut.
    """
    counts: dict[str, int] = Counter()
    assigned: list[dict] = []
    deferred: list[tuple[dict, list[dict]]] = []
    for row in positives:
        opts = placements(int(row["box_start_bar"]), int(row["box_end_bar"]), int(row["decision_bar"]))
        if not opts:
            assigned.append({**row, "spread_status": "no_legal_window"})
            continue
        left_opts = [o for o in opts if o["bin"] == "left"]
        if left_opts:
            pick = min(left_opts, key=lambda o: o["rel"])
            counts["left"] += 1
            assigned.append(_record(row, pick, opts))
        else:
            deferred.append((row, opts))
    for row, opts in deferred:
        pick = assign_one(opts, counts)
        assert pick is not None
        counts[pick["bin"]] += 1
        assigned.append(_record(row, pick, opts))
    return assigned
