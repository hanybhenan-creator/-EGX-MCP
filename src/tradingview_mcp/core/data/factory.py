"""Assembly: build a wired registry + router + store from the environment.

Everything the data layer needs to run lives behind this module, so the MCP
tools never construct transports or registries inline. Configuration is
environment-first (matching the rest of the project, see ``.env.example``):

===========================  ==========================================
``EGX_DATA_DB``              SQLite path (default ``egx_data.db``)
``EGX_DATA_OFFLINE``         ``1`` → never touch the network (read store only)
``EGX_TRANSPORT_TIMEOUT``    Per-request timeout in seconds (default 15)
``EGX_POLITE_RATE``          Requests/second to egx.com.eg (default 0.5)
``EGX_ENABLE_TRADINGVIEW``   ``1`` → register the delegating TV source
``EGX_PROXY_URL``            Optional HTTP(S) proxy for all sources
===========================  ==========================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .calendar import EGXCalendar
from .router import DataRouter
from .sources.base import Capability, SourceRegistry
from .sources.egx_official import EGXOfficialSource
from .sources.yahoo import YahooFinanceSource
from .store import DataStore
from .transport import HttpxTransport, RetryPolicy, Transport

#: Hosts we pace deliberately. egx.com.eg is a public service and the whole
#: universe refresh is ~300 requests; 0.5 rps keeps that to ~10 minutes of
#: polite crawling instead of a burst that looks like an attack.
DEFAULT_HOST_RATES: dict[str, tuple[float, float]] = {
    "www.egx.com.eg": (0.5, 1.0),
    "egx.com.eg": (0.5, 1.0),
    "query1.finance.yahoo.com": (4.0, 4.0),
    "query2.finance.yahoo.com": (4.0, 4.0),
}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def offline_mode() -> bool:
    return _env_flag("EGX_DATA_OFFLINE", False)


def default_db_path() -> Path:
    return Path(os.environ.get("EGX_DATA_DB", "egx_data.db"))


def build_transport(
    *,
    transport: Optional[Transport] = None,
    proxy: Optional[str] = None,
) -> Optional[Transport]:
    """Real transport, or ``None`` in offline mode."""
    if transport is not None:
        return transport
    if offline_mode():
        return None
    from .transport import HostRateLimiter

    limiter = HostRateLimiter(default_rate=_env_float("EGX_POLITE_RATE", 2.0))
    for host, (rate, burst) in DEFAULT_HOST_RATES.items():
        limiter.set_host_rate(host, rate, burst)

    return HttpxTransport(
        retry=RetryPolicy(attempts=3, base_delay_s=0.6, max_delay_s=20.0),
        limiter=limiter,
        default_timeout_s=_env_float("EGX_TRANSPORT_TIMEOUT", 15.0),
        proxy=proxy or os.environ.get("EGX_PROXY_URL") or None,
    )


def build_registry(
    *,
    transport: Optional[Transport] = None,
    include_tradingview: Optional[bool] = None,
    extra_sources: Optional[list] = None,
) -> SourceRegistry:
    """Register the sources available in this environment.

    A source that cannot run here (no transport in offline mode, missing
    optional dependency) is **not registered** rather than registered-and-always
    failing: the router's candidate list should mean "things we can actually
    use", and the audit log stays free of noise from sources that were never
    viable.
    """
    registry = SourceRegistry()
    tx = build_transport(transport=transport)

    if tx is not None:
        registry.register(EGXOfficialSource(tx))
        registry.register(YahooFinanceSource(tx))

        want_tv = (
            include_tradingview
            if include_tradingview is not None
            else _env_flag("EGX_ENABLE_TRADINGVIEW", True)
        )
        if want_tv:
            try:
                from .sources.tradingview import TradingViewSource
                from .sources.tradingview_loaders import egx_sector_loader, egx_snapshot_loader

                registry.register(
                    TradingViewSource(
                        {
                            Capability.MARKET_SNAPSHOT: egx_snapshot_loader,
                            Capability.QUOTE: egx_snapshot_loader,
                            Capability.SECTOR_STATS: egx_sector_loader,
                        }
                    )
                )
            except Exception:  # pragma: no cover - optional dependency path
                pass

    for source in extra_sources or []:
        registry.register(source)

    return registry


def build_audit_sink(store: DataStore):
    """Adapter turning router Attempts into ``fetch_audit`` rows."""

    def sink(attempt, subject) -> None:  # noqa: ANN001 - structural typing
        store.record_attempt(
            source=attempt.source,
            capability=attempt.capability.value,
            subject=subject,
            ok=attempt.ok,
            latency_ms=attempt.latency_ms,
            rows=attempt.rows,
            http_status=attempt.http_status,
            error=attempt.error,
            error_kind=attempt.error_kind,
            flags=attempt.flags,
            endpoint=attempt.endpoint,
        )

    return sink


def build_store(*, db_path: Optional[str | Path] = None) -> DataStore:
    return DataStore(db_path or default_db_path())


def build_router(
    *,
    transport: Optional[Transport] = None,
    store: Optional[DataStore] = None,
    calendar: Optional[EGXCalendar] = None,
    registry: Optional[SourceRegistry] = None,
    with_audit: bool = True,
) -> DataRouter:
    reg = registry or build_registry(transport=transport)
    return DataRouter(
        reg,
        calendar=calendar or EGXCalendar.from_env(),
        audit_sink=build_audit_sink(store) if (store is not None and with_audit) else None,
    )


class DataLayer:
    """Convenience bundle: store + registry + router + calendar, one lifecycle."""

    def __init__(
        self,
        *,
        transport: Optional[Transport] = None,
        db_path: Optional[str | Path] = None,
        calendar: Optional[EGXCalendar] = None,
        store: Optional[DataStore] = None,
        extra_sources: Optional[list] = None,
        include_tradingview: Optional[bool] = None,
    ) -> None:
        self.calendar = calendar or EGXCalendar.from_env()
        self.store = store if store is not None else build_store(db_path=db_path)
        self.registry = build_registry(
            transport=transport,
            include_tradingview=include_tradingview,
            extra_sources=extra_sources,
        )
        self.router = build_router(
            store=self.store, calendar=self.calendar, registry=self.registry, with_audit=True
        )

    def close(self) -> None:
        self.registry.close_all()
        self.store.close()

    def __enter__(self) -> "DataLayer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── convenience wrappers ────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        return {
            "offline": offline_mode(),
            "calendar": {
                "now_local": self.calendar.now().isoformat(),
                "state": self.calendar.state().phase,
                "expect_live_data": self.calendar.state().expect_live_data,
                "warnings": self.calendar.sync_warnings(self.calendar.now().year),
            },
            "sources": self.registry.health_report(),
            "store": self.store.stats().as_dict(),
        }

    def source_reliability(self) -> list[dict[str, Any]]:
        return self.store.source_reliability()


_LAYER: Optional[DataLayer] = None


def get_layer() -> DataLayer:
    """Process-wide singleton (lazy) used by MCP tools."""
    global _LAYER
    if _LAYER is None:
        _LAYER = DataLayer()
    return _LAYER


def reset_layer() -> None:
    """Drop the singleton — used by tests and after config changes."""
    global _LAYER
    if _LAYER is not None:
        try:
            _LAYER.close()
        except Exception:  # pragma: no cover
            pass
    _LAYER = None
