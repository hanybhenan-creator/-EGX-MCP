"""MCP-facing service for the EGX data acquisition layer.

Thin by design: the business logic lives in ``core/data`` (router, ingest,
store, calendar) and this module only marshals arguments and shapes the
responses the tools return. That keeps every behaviour testable without the
MCP server, which is the pattern the rest of ``core/services`` follows.

Two guarantees the tools inherit from the layer beneath:

* a refresh always returns a **report** (what was written, skipped, failed and
  why) rather than a bare count, and
* a status call always states **how current the data is** — measured against
  the EGX trading calendar, not against wall-clock time.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from tradingview_mcp.core.data.factory import offline_mode
from tradingview_mcp.core.data.ingest import (
    EGXIngestor,
    symbols_for_index,
    symbols_from_coinlist,
)
from tradingview_mcp.core.errors import ErrorCode, make_error

DEFAULT_INDEX = "EGX30"
MAX_SYMBOLS_PER_CALL = 60


def _resolve_symbols(symbols: str = "", index: str = "", limit: int = 0) -> list[str]:
    """Pick the symbol set for a refresh: explicit list > index > full universe."""
    explicit = [s for s in (symbols or "").replace(";", ",").split(",") if s.strip()]
    if explicit:
        out = [s.strip().upper().replace("EGX:", "") for s in explicit]
    elif index:
        out = symbols_for_index(index)
    else:
        out = symbols_from_coinlist()
    if limit:
        out = out[:limit]
    return out


def egx_data_status(*, index: str = DEFAULT_INDEX, sample: int = 10) -> dict[str, Any]:
    """Health of the acquisition system: sources, store, freshness, coverage.

    Answers the four questions an operator actually has:
      1. Which sources are reachable right now, and which are tripped?
      2. What is in the local store, and how old is it?
      3. Which symbols are behind the calendar, and by how many sessions?
      4. Are the holidays the calendar believes in verified for this year?
    """
    from tradingview_mcp.core.data.factory import get_layer

    layer = get_layer()
    cal = layer.calendar
    state = cal.state()

    payload: dict[str, Any] = {
        "mode": "offline" if offline_mode() else "online",
        "market_clock": {
            "now_local": state.now_local.isoformat(),
            "phase": state.phase,
            "is_trading_day": state.is_trading_day,
            "is_open": state.is_open,
            "expect_live_data": state.expect_live_data,
            "seconds_to_open": state.seconds_to_open,
            "seconds_since_close": state.seconds_since_close,
            "max_quote_age_s": cal.max_quote_age_s(),
            "newest_settled_session": _newest_settled(cal),
        },
        "calendar_warnings": cal.sync_warnings(state.now_local.year),
        "sources": layer.registry.health_report(),
        "store": layer.store.stats().as_dict(),
        "source_reliability": layer.store.source_reliability(),
    }

    try:
        symbols = symbols_for_index(index) if index else symbols_from_coinlist()
    except ValueError as exc:
        return make_error(ErrorCode.INVALID_PARAMETER, str(exc))

    ingestor = EGXIngestor(layer.router, layer.store, calendar=cal)
    payload["coverage"] = ingestor.coverage(symbols, lookback_days=30)
    payload["recent_failures"] = [
        row
        for row in layer.store.recent_audit(limit=25, failures_only=True)
        if row.get("error")
    ][:sample]
    payload["recent_runs"] = layer.store.recent_runs(limit=5)
    return payload


def egx_data_refresh_daily(
    *,
    symbols: str = "",
    index: str = "",
    days: int = 365,
    max_symbols: int = 25,
    force: bool = False,
    cross_check: bool = False,
) -> dict[str, Any]:
    """Fetch and persist settled daily OHLCV for a symbol set.

    Defaults are deliberately modest (25 symbols) because the EGX site is
    paced at half a request per second: a full 292-name refresh belongs in a
    scheduled job, not in an interactive tool call.
    """
    from tradingview_mcp.core.data.factory import get_layer

    if offline_mode():
        return make_error(
            ErrorCode.DEPENDENCY_MISSING,
            "EGX_DATA_OFFLINE=1 — the data layer is in read-only mode; "
            "unset it to allow network refreshes",
        )

    limit = max(1, min(int(max_symbols or 25), MAX_SYMBOLS_PER_CALL))
    try:
        universe = _resolve_symbols(symbols, index, limit=limit)
    except ValueError as exc:
        return make_error(ErrorCode.INVALID_PARAMETER, str(exc))

    if not universe:
        return make_error(
            ErrorCode.NO_DATA, "no symbols resolved — pass `symbols` or a known `index`"
        )

    layer = get_layer()
    ingestor = EGXIngestor(layer.router, layer.store, calendar=layer.calendar)
    report = ingestor.refresh_daily(
        universe,
        days=max(20, min(int(days or 365), 365 * 10)),
        cross_check=bool(cross_check),
        force=bool(force),
    )
    return report.as_dict()


def egx_data_refresh_universe(*, cross_check: bool = True) -> dict[str, Any]:
    """Pull the exchange's whole-market snapshot: symbol master + latest quotes."""
    from tradingview_mcp.core.data.factory import get_layer

    if offline_mode():
        return make_error(
            ErrorCode.DEPENDENCY_MISSING,
            "EGX_DATA_OFFLINE=1 — the data layer is in read-only mode; "
            "unset it to allow network refreshes",
        )

    layer = get_layer()
    ingestor = EGXIngestor(layer.router, layer.store, calendar=layer.calendar)
    return ingestor.refresh_universe(cross_check=bool(cross_check)).as_dict()


def _newest_settled(cal) -> Optional[str]:
    from tradingview_mcp.core.data.quality import _expected_bar

    try:
        return _expected_bar(cal, None).isoformat()
    except Exception:  # pragma: no cover - defensive
        return None
