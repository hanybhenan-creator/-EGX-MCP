"""Default TradingView loaders wired to the repository's existing TV access.

These functions are thin: they build a ``tradingview_screener`` query exactly
the way ``egx_service``/``stock_screener_service`` already do, route it through
``screener_provider._scan_with_retry`` (retry + TTL cache + stale-while-error),
and normalise the DataFrame into the data layer's row shape.

They are deliberately defensive. TradingView is the flakiest upstream in this
project, and a column that does not exist for the ``egypt`` market must degrade
to "fewer fields" rather than "no snapshot at all" — so each loader tries the
full column set first and falls back to a minimal one on failure.
"""

from __future__ import annotations

from typing import Any, Optional

FULL_COLUMNS = (
    "name",
    "description",
    "close",
    "change",
    "volume",
    "exchange",
    "sector",
    "market_cap_basic",
    "high",
    "low",
    "open",
    "price_52_week_high",
    "price_52_week_low",
)

MINIMAL_COLUMNS = ("name", "close", "change", "volume", "exchange")


def _df_to_rows(df: Any) -> list[dict[str, Any]]:
    if df is None:
        return []
    try:
        if getattr(df, "empty", False):
            return []
    except Exception:  # pragma: no cover
        pass
    try:
        records = df.to_dict(orient="records")
    except Exception:  # pragma: no cover
        return []
    return [r for r in records if isinstance(r, dict)]


def egx_snapshot_loader(
    *,
    symbol: Optional[str] = None,
    interval: str = "1D",
    limit: int = 300,
) -> list[dict[str, Any]]:
    """Whole-market (or single-symbol) EGX rows from the TradingView screener."""
    from .base import SourceUnavailable

    try:
        from tradingview_screener import Query  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency always present in prod
        raise SourceUnavailable("tradingview", f"tradingview-screener missing: {exc}", retryable=False)

    try:
        from tradingview_mcp.core.services.screener_provider import _scan_with_retry
    except ImportError as exc:  # pragma: no cover
        raise SourceUnavailable("tradingview", f"screener_provider missing: {exc}", retryable=False)

    attempts = [FULL_COLUMNS, MINIMAL_COLUMNS]
    last_error: Optional[BaseException] = None

    for columns in attempts:
        try:
            q = Query().set_markets("egypt").select(*columns)
            if symbol:
                ticker = symbol if ":" in symbol else f"EGX:{symbol.strip().upper()}"
                q = q.set_tickers(ticker)
            else:
                q = q.limit(int(limit))
            _total, df = _scan_with_retry(
                q, cache_key=("egx_data_layer_snapshot_v1", interval, tuple(columns), symbol or "")
            )
            rows = _df_to_rows(df)
            if rows:
                return rows
        except Exception as exc:  # try the next column set, then give up loudly
            last_error = exc

    raise SourceUnavailable(
        "tradingview",
        f"screener returned no rows after {len(attempts)} column sets"
        + (f" (last error: {last_error})" if last_error else ""),
    )


def egx_sector_loader(
    *,
    symbol: Optional[str] = None,
    interval: str = "1D",
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Sector-level rows for the EGX market."""
    return egx_snapshot_loader(symbol=None, interval=interval, limit=limit)
