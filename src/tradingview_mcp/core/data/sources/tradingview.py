"""TradingView source adapter — delegates, does not re-implement.

The repository already talks to TradingView through ``tradingview_ta`` and
``tradingview_screener`` (see ``screener_provider`` for the retry/cache layer
that exists precisely because that upstream is flaky). Re-implementing that
access on this module's transport would mean a second, differently-behaved
TradingView client in the same process — two rate-limit budgets, two cache
policies, two retry strategies.

So this adapter *delegates*: it takes loader callables (one per capability),
normalises their output into the data layer's row shape, and attaches
provenance and quality flags. The loaders are injected, which also means the
whole adapter is testable with a lambda and no network.

What it buys the chain: a **third independent observation** of EGX quotes for
:func:`quality.triangulate`. When the exchange's page and Yahoo agree, the
value is corroborated; when TradingView disagrees, the operator gets a
CONFLICT row instead of a silent average.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..provenance import QualityFlag, Sourced
from ..transport import Transport
from .base import (
    Capability,
    MarketDataSource,
    SourceDataError,
    SourceDescriptor,
    SourceUnavailable,
)

#: Loader signature: ``(symbol: str | None, interval: str, limit: int) -> list[dict]``
Loader = Callable[..., list[dict[str, Any]]]


class NullTransport:
    """Placeholder transport for delegating sources (they never call it)."""

    def request(self, *args: Any, **kwargs: Any):  # pragma: no cover - never called
        raise SourceUnavailable("tradingview", "delegating source performs no HTTP itself")


class TradingViewSource(MarketDataSource):
    """Adapter over injected TradingView loaders."""

    descriptor = SourceDescriptor(
        name="tradingview",
        capabilities=frozenset(
            {
                Capability.MARKET_SNAPSHOT,
                Capability.QUOTE,
                Capability.SECTOR_STATS,
            }
        ),
        priority=30,
        official=False,
        usage_note=(
            "Delegates to the repository's existing tradingview_ta / "
            "tradingview_screener access path (retry + TTL cache included)."
        ),
        typical_latency_ms=900,
        symbol_style="bare",
    )

    def __init__(
        self,
        loaders: dict[Capability, Loader],
        *,
        transport: Optional[Transport] = None,
        health=None,
        clock=None,
    ) -> None:
        super().__init__(transport or NullTransport(), health=health, clock=clock)
        self._loaders = dict(loaders)

    def supports(self, capability: Capability) -> bool:
        return super().supports(capability) and capability in self._loaders

    def capabilities(self) -> frozenset[Capability]:  # type: ignore[override]
        return frozenset(
            c for c in self.descriptor.capabilities if c in self._loaders
        )

    def available(self) -> bool:
        return any(callable(l) for l in self._loaders.values())

    # ── internals ───────────────────────────────────────────────────────────

    def _run(
        self,
        capability: Capability,
        *,
        symbol: Optional[str] = None,
        interval: str = "1D",
        limit: int = 300,
        attempt_index: int = 0,
    ) -> Sourced[list[dict[str, Any]]]:
        loader = self._loaders.get(capability)
        if loader is None:
            raise SourceUnavailable(
                self.name, f"no loader registered for {capability.value}", retryable=False
            )

        try:
            rows = loader(symbol=symbol, interval=interval, limit=limit)
        except TypeError:
            # Tolerate simpler loader signatures in tests and integrations.
            try:
                rows = loader(symbol, interval, limit)  # type: ignore[misc]
            except Exception as exc:
                raise SourceUnavailable(self.name, f"loader failed: {exc}") from exc
        except Exception as exc:
            raise SourceUnavailable(self.name, f"loader failed: {exc}") from exc

        if not rows:
            raise SourceDataError(self.name, f"{capability.value} loader returned no rows")

        out = self.wrap(
            rows,
            endpoint=f"tradingview:{capability.value}",
            locator={
                "exchange": "EGX",
                "interval": interval,
                "symbol": symbol,
                "rows": len(rows),
            },
            attempt_index=attempt_index,
        )
        missing_symbols = sum(1 for r in rows if not r.get("symbol"))
        if missing_symbols:
            out.add_flag(
                QualityFlag.PARTIAL,
                f"{missing_symbols}/{len(rows)} TradingView rows carry no symbol and "
                "cannot be cross-checked",
            )
        return out

    # ── capabilities ────────────────────────────────────────────────────────

    def market_snapshot(
        self, *, interval: str = "1D", limit: int = 300, attempt_index: int = 0
    ) -> Sourced[list[dict[str, Any]]]:
        return self._run(
            Capability.MARKET_SNAPSHOT, interval=interval, limit=limit, attempt_index=attempt_index
        )

    def sector_stats(self, *, attempt_index: int = 0) -> Sourced[list[dict[str, Any]]]:
        return self._run(Capability.SECTOR_STATS, attempt_index=attempt_index)

    def quote(
        self,
        symbol: str,
        *,
        interval: str = "1D",
        attempt_index: int = 0,
    ) -> Sourced[dict[str, Any]]:
        out = self._run(
            Capability.QUOTE, symbol=symbol, interval=interval, limit=1, attempt_index=attempt_index
        )
        wanted = symbol.strip().upper().replace("EGX:", "")
        for row in out.value:
            if str(row.get("symbol", "")).upper().replace("EGX:", "") == wanted:
                return Sourced(
                    value=row,
                    provenance=out.provenance,
                    flags=list(out.flags),
                    notes=list(out.notes),
                    meta={"rows_returned": len(out.value)},
                )
        if len(out.value) == 1 and str(out.value[0].get("symbol", "")).upper() in {"", wanted}:
            return Sourced(
                value=out.value[0],
                provenance=out.provenance,
                flags=list(out.flags),
                notes=list(out.notes + ["loader returned a single unlabelled row — assumed to be the request"]),
            )
        raise SourceDataError(self.name, f"{wanted} not present in TradingView rows")


# ── normalisation helpers ───────────────────────────────────────────────────


#: Keys TradingView responses use for the same canonical field.
TV_FIELD_ALIASES: dict[str, str] = {
    "ticker": "symbol",
    "name": "name",
    "description": "name",
    "close": "close",
    "last": "close",
    "price": "close",
    "change": "change",
    "change_percent": "change_pct",
    "change_abs": "change",
    "volume": "volume",
    "volume_24h": "volume",
    "value_traded": "value_traded",
    "market_cap_basic": "market_cap",
    "sector": "sector",
    "open": "open",
    "high": "high",
    "low": "low",
    "prev_close": "previous_close",
    "previous_close": "previous_close",
}


def normalise_tv_row(raw: dict[str, Any], *, exchange: str = "EGX") -> dict[str, Any]:
    """Map one TradingView record to the canonical row shape.

    TradingView returns ``EGX:COMI`` style tickers; the canonical shape uses the
    bare ticker with an explicit ``exchange`` field, because the router and the
    store key on ``(exchange, symbol)`` and must not depend on a vendor prefix.
    """
    out: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = TV_FIELD_ALIASES.get(str(key).lower())
        if canonical and canonical not in out and value is not None:
            out[canonical] = value

    symbol = str(out.get("symbol", ""))
    if ":" in symbol:
        prefix, _, bare = symbol.partition(":")
        out["symbol"] = bare.strip().upper()
        out["exchange"] = prefix.strip().upper()
    else:
        out["symbol"] = symbol.strip().upper()
        out.setdefault("exchange", exchange)
    return out


def normalise_tv_rows(rows: list[dict[str, Any]], *, exchange: str = "EGX") -> list[dict[str, Any]]:
    return [normalise_tv_row(r, exchange=exchange) for r in rows if isinstance(r, dict)]
