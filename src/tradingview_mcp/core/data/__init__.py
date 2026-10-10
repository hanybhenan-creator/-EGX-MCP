"""EGX data acquisition layer.

Public surface
--------------
``DataLayer`` / ``get_layer()``
    Wired bundle of store + source registry + router + market calendar.
``Capability``
    What a caller asks for (a quote, daily bars, a snapshot…), independent of
    which source answers.
``Sourced`` / ``Provenance`` / ``QualityFlag``
    Every returned value carries its source, fetch time, validation verdicts
    and cross-source comparisons.
``DataRouter``
    Fallback chain with quality gates and an audit trail.

Design contract
---------------
1. Nothing in this package returns a bare number. Values arrive inside a
   :class:`~tradingview_mcp.core.data.provenance.Sourced` so downstream code
   can always answer "where did this come from?".
2. Freshness is decided by the EGX calendar, never by a constant.
3. A source declares what it can serve; the router never asks for more.
4. Failures are recorded, not swallowed — ``AllSourcesFailed`` carries every
   attempt with its reason.
"""

from .calendar import EGXCalendar, SessionState
from .factory import (
    DataLayer,
    build_registry,
    build_router,
    build_store,
    build_transport,
    default_db_path,
    get_layer,
    offline_mode,
    reset_layer,
)
from .provenance import BLOCKING_FLAGS, Provenance, QualityFlag, Sourced
from .quality import triangulate, validate_ohlcv_bars, validate_quote
from .router import AllSourcesFailed, DataRouter, Resolution
from .sources.base import (
    Capability,
    MarketDataSource,
    SourceDescriptor,
    SourceHealth,
    SourceRegistry,
    SourceUnavailable,
)
from .store import DataStore
from .transport import FakeTransport, HttpxTransport, Transport, TransportError

__all__ = [
    "EGXCalendar",
    "SessionState",
    "DataLayer",
    "DataRouter",
    "Resolution",
    "AllSourcesFailed",
    "Capability",
    "MarketDataSource",
    "SourceDescriptor",
    "SourceHealth",
    "SourceRegistry",
    "SourceUnavailable",
    "Provenance",
    "QualityFlag",
    "Sourced",
    "BLOCKING_FLAGS",
    "DataStore",
    "Transport",
    "TransportError",
    "HttpxTransport",
    "FakeTransport",
    "triangulate",
    "validate_ohlcv_bars",
    "validate_quote",
    "build_registry",
    "build_router",
    "build_store",
    "build_transport",
    "default_db_path",
    "get_layer",
    "offline_mode",
    "reset_layer",
]
