"""MCP-facing sector calibration: does the sector scanner's ranking predict anything?

Reads the local store only (no network). Replays the live scanner's sector rules
over stored history and reports, for each rule and each continuous score, what
followed — with excess returns measured against the average sector on the same
dates, permutation p-values, Benjamini–Hochberg q-values and walk-forward folds.
See ``core.calibration.sectors`` for the method.
"""

from __future__ import annotations

from typing import Any

from tradingview_mcp.core.calibration.sectors import analyze_sectors
from tradingview_mcp.core.data.egx_sectors import EGX_SECTORS, get_sector
from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services.egx_calibration_service import _load_bars

MAX_PERMUTATIONS = 2000
MAX_SECTOR_SYMBOLS = 400


def egx_sector_calibration(
    *,
    horizon: int = 5,
    min_members: int = 3,
    permutations: int = 200,
    walk_forward_folds: int = 3,
    round_trip_cost: float = 0.0,
    lookback_days: int = 1500,
) -> dict[str, Any]:
    """Calibrate the sector scanner's Hot/Cold labels and momentum ranking.

    Args:
      horizon: sessions held after the sector is observed (entry at the next open).
      min_members: a sector needs this many members with data on a session to count.
      permutations: block-shuffle permutations per rule (0 disables p-values).
      walk_forward_folds: 0 to skip, otherwise 2-6 contiguous folds.
      round_trip_cost: fraction deducted from every event (default 0 = gross). Set
        it from your broker's commission and the stamp duty you pay; this tool does
        not assume a figure.
      lookback_days: how far back to read the store.
    """
    if not 1 <= horizon <= 60:
        return make_error(ErrorCode.INVALID_PARAMETER, "horizon must be 1-60 sessions", horizon=horizon)
    if not 2 <= min_members <= 10:
        return make_error(ErrorCode.INVALID_PARAMETER, "min_members must be 2-10", min_members=min_members)
    if not 0 <= permutations <= MAX_PERMUTATIONS:
        return make_error(ErrorCode.INVALID_PARAMETER, f"permutations must be 0-{MAX_PERMUTATIONS}")
    if walk_forward_folds != 0 and not 2 <= walk_forward_folds <= 6:
        return make_error(ErrorCode.INVALID_PARAMETER, "walk_forward_folds must be 0 or 2-6")
    if not 0.0 <= round_trip_cost < 0.2:
        return make_error(ErrorCode.INVALID_PARAMETER, "round_trip_cost must be a fraction in [0, 0.2)")
    if not 60 <= lookback_days <= 5000:
        return make_error(ErrorCode.INVALID_PARAMETER, "lookback_days must be 60-5000")

    members = sorted({sym for syms in EGX_SECTORS.values() for sym in syms})
    bars_by_symbol, skipped = _load_bars(members, lookback_days=lookback_days, max_symbols=MAX_SECTOR_SYMBOLS)
    if not bars_by_symbol:
        return make_error(
            ErrorCode.NO_DATA,
            "no sector member has enough stored history (60+ sessions). Run "
            "`python scripts/egx_data_sync.py daily --index EGX30 --days 365` first",
            members_in_map=len(members),
            lookback_days=lookback_days,
        )

    report = analyze_sectors(
        bars_by_symbol,
        sector_of=get_sector,
        horizon=horizon,
        min_members=min_members,
        permutations=permutations,
        walk_forward_folds=walk_forward_folds,
        round_trip_cost=round_trip_cost,
    )
    payload = report.as_dict()
    payload["mode"] = "offline store"
    payload["symbols"] = {"requested": len(members), "with_history": len(bars_by_symbol), "skipped_short_history": skipped}
    payload["warnings"] = list(payload["warnings"])
    if skipped:
        payload["warnings"].append(
            f"{len(skipped)} member(s) had too little history and were excluded: {', '.join(skipped[:10])}"
        )
    if not payload["validated_rules"]:
        payload["warnings"].append("no sector rule reached VALIDATED: the scanner's labels are not shown to predict outperformance")
    return payload
