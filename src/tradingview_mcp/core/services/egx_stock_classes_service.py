"""Classify stored EGX stocks by measured behaviour, and show which tools each class may use.

Read-only. Uses the local store only; it never fetches data.
"""

from __future__ import annotations

from typing import Any

from tradingview_mcp.core.analysis import stock_classes as sc
from tradingview_mcp.core.calibration.glossary import CLASSIFICATION_GLOSSARY, attach_glossary
from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services.egx_calibration_service import _load_bars
from tradingview_mcp.core.services.egx_data_service import _resolve_symbols

DISCLAIMER = (
    "Classes describe each stock's measured behaviour over the last window of sessions. "
    "They are not forecasts. No tool is allowed for any class until it has been validated "
    "for that class on held-out data; until then the allowed-tools list is empty."
)


def egx_classify_stocks(
    symbols: str = "",
    index: str = "",
    max_symbols: int = 200,
    lookback_days: int = 400,
) -> dict[str, Any]:
    try:
        requested = _resolve_symbols(symbols, index) if (symbols or index) else []
    except ValueError as exc:
        return make_error(ErrorCode.INVALID_PARAMETER, str(exc), index=index)
    if not requested:
        requested = _resolve_symbols("", "", limit=max_symbols)

    bars_by_symbol, skipped = _load_bars(requested, lookback_days=lookback_days, max_symbols=max_symbols)
    if not bars_by_symbol:
        return make_error(
            ErrorCode.NO_DATA,
            "no stored daily history for the requested symbols: run "
            "`python scripts/egx_data_sync.py daily --index EGX30 --days 365` first",
            requested=len(requested),
        )

    rules = sc.ClassRules()
    stocks: list[dict[str, Any]] = []
    groups: dict[str, list[str]] = {c: [] for c in sc.CLASSES}
    as_of = ""
    for symbol in sorted(bars_by_symbol):
        bars = bars_by_symbol[symbol]
        history = sc.classify_history(bars, rules)
        latest = history[-1]
        changes = sc.transitions(history, bars)
        as_of = max(as_of, str(bars[-1].get("date", "")))
        groups[latest.official_class].append(symbol)
        stocks.append({
            "symbol": symbol,
            "as_of": str(bars[-1].get("date", "")),
            **latest.as_dict(),
            "class_changes_recent": changes[-3:],
            "allowed_tools": list(sc.allowed_tools(latest.official_class)),
            "excluded_tools": dict(sc.CLASS_POLICY[latest.official_class]["excluded"]),
        })

    warnings: list[str] = []
    if skipped:
        warnings.append(f"{len(skipped)} symbol(s) had too little history and were not classified: {', '.join(skipped[:10])}")
    if len(bars_by_symbol) < 5:
        warnings.append("fewer than 5 stocks classified: class counts say little about the market")
    if len(groups[sc.UNCLASSIFIED]) == len(stocks):
        warnings.append("no stock has enough complete history to classify")
    warnings.append(
        "Thresholds are provisional (see ClassRules). Liquidity floor is in EGP and must be "
        "checked against real EGX turnover before results are interpreted."
    )

    payload = {
        "as_of": as_of,
        "lookback_days": lookback_days,
        "window_sessions": rules.window,
        "confirm_sessions": rules.confirm_sessions,
        "counts_by_class": {c: len(v) for c, v in groups.items() if v},
        "symbols_by_class": {c: v for c, v in groups.items() if v},
        "class_policy": {c: {"status": p["status"], "reason": p["reason"]} for c, p in sc.CLASS_POLICY.items()},
        "stocks": stocks,
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }
    return attach_glossary(payload, CLASSIFICATION_GLOSSARY)
