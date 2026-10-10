"""Data-source contract: capabilities, health, circuit breaking.

A source in this system is not "a website we scrape". It is an object that
declares — explicitly and machine-checkably — **what it can serve**
(:class:`Capability`), **how it fails** (:class:`SourceHealth`), and **what it
costs** (rate limits, latency). The router only ever asks a source for a
capability it declared, so a Yahoo fetcher can never be silently picked to
answer a question only the exchange's own feed can answer.

The circuit breaker is deliberately boring: N consecutive failures opens the
source for a cooldown, one trial request closes it again. What matters is that
it is *per source*, so a broken scraper does not take the whole refresh down —
it demotes that source in the chain and the router moves on.
"""

from __future__ import annotations

import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from ..provenance import Provenance, Sourced, now_utc
from ..transport import Transport, TransportError, TransportResponse


class Capability(str, Enum):
    """What a source is allowed to be asked for.

    Kept deliberately small and *semantic* (not per-endpoint) so the router's
    decisions stay readable in the audit log.
    """

    #: Symbol universe / listing metadata (authoritative name, sector, ISIN…).
    SYMBOL_MASTER = "symbol_master"
    #: One instrument's most recent print.
    QUOTE = "quote"
    #: Whole-market snapshot (every symbol, one row each).
    MARKET_SNAPSHOT = "market_snapshot"
    #: Daily OHLCV history for one instrument.
    DAILY_OHLCV = "daily_ohlcv"
    #: Intraday OHLCV history for one instrument.
    INTRADAY_OHLCV = "intraday_ohlcv"
    #: Index level and change (EGX30/70/100…).
    INDEX_QUOTE = "index_quote"
    #: Sector aggregates (value traded, volume, breadth).
    SECTOR_STATS = "sector_stats"
    #: Corporate/profile facts (market cap, shares outstanding, free float).
    COMPANY_PROFILE = "company_profile"


#: Which capabilities are *proven* only by the exchange itself. Used by the
#: router to express "prefer official, but fall back with a warning".
OFFICIAL_PREFERRED: frozenset[Capability] = frozenset(
    {
        Capability.SYMBOL_MASTER,
        Capability.MARKET_SNAPSHOT,
        Capability.SECTOR_STATS,
        Capability.INDEX_QUOTE,
    }
)


class SourceUnavailable(Exception):
    """Raised when a source cannot serve the request at all (not a data error)."""

    def __init__(self, source: str, reason: str, *, retryable: bool = True) -> None:
        super().__init__(f"{source}: {reason}")
        self.source = source
        self.reason = reason
        self.retryable = retryable


class SourceParseError(SourceUnavailable):
    """Response arrived but could not be turned into rows — the scraper's fault."""

    def __init__(self, source: str, reason: str, *, sample: str = "") -> None:
        super().__init__(source, reason, retryable=False)
        self.sample = sample[:400]


class SourceDataError(SourceUnavailable):
    """Source answered, parsing worked, but the data is structurally invalid."""

    def __init__(self, source: str, reason: str) -> None:
        super().__init__(source, reason, retryable=False)


# ── health & circuit breaking ───────────────────────────────────────────────


@dataclass
class HealthSnapshot:
    source: str
    state: str
    consecutive_failures: int
    total_successes: int
    total_failures: int
    last_success_at: Optional[datetime]
    last_failure_at: Optional[datetime]
    last_error: Optional[str]
    open_until: Optional[datetime]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "successes": self.total_successes,
            "failures": self.total_failures,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_failure_at": self.last_failure_at.isoformat() if self.last_failure_at else None,
            "last_error": self.last_error,
            "open_until": self.open_until.isoformat() if self.open_until else None,
        }


class SourceHealth:
    """Per-source reachability accounting + circuit breaker state.

    The counters measure **reachability**: did the source answer? They drive the
    circuit breaker, which exists to stop hammering a dead endpoint. A source
    that answers fine but whose page layout changed (a ``SourceParseError``) is
    *reachable*, so it does not trip the breaker — that failure shows up in
    ``fetch_audit`` (queryable through :meth:`DataStore.source_reliability`) and
    in the quality flags instead. Conflating the two would take a merely
    mis-parsed source out of the fallback chain for a full cooldown, which is
    how a one-line selector fix turns into a 30-minute data outage.

    Recording happens in :class:`~tradingview_mcp.core.data.router.DataRouter`,
    once per *source call*, so a source that answers via a fallback endpoint
    counts as a success rather than as the failure of its first attempt.
    """

    def __init__(
        self,
        source: str,
        *,
        failure_threshold: int = 3,
        cooldown_s: float = 120.0,
        clock: Callable[[], float] = _time.monotonic,
    ) -> None:
        self.source = source
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        # Reentrant on purpose: snapshot() reads the derived `state` property
        # (and therefore the same lock) from inside its own critical section.
        # A plain Lock deadlocks the first health report.
        self._lock = threading.RLock()

        self._consecutive_failures = 0
        self._successes = 0
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._last_success_at: Optional[datetime] = None
        self._last_failure_at: Optional[datetime] = None
        self._last_error: Optional[str] = None

    # ── state transitions ───────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if self._clock() - self._opened_at >= self.cooldown_s:
                # Half-open: allow one trial request through.
                return False
            return True

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            if self._clock() - self._opened_at >= self.cooldown_s:
                return "half_open"
            return "open"

    def record_success(self) -> None:
        with self._lock:
            self._successes += 1
            self._consecutive_failures = 0
            self._opened_at = None
            self._last_success_at = now_utc()
            self._last_error = None

    def record_failure(self, error: Optional[BaseException] = None) -> None:
        with self._lock:
            self._failures += 1
            self._consecutive_failures += 1
            self._last_failure_at = now_utc()
            if error is not None:
                self._last_error = f"{type(error).__name__}: {error}"[:300]
            if self._consecutive_failures >= self.failure_threshold:
                self._opened_at = self._clock()

    def snapshot(self) -> HealthSnapshot:
        import datetime as _dt

        with self._lock:
            open_until = None
            if self._opened_at is not None:
                remaining = self.cooldown_s - (self._clock() - self._opened_at)
                if remaining > 0:
                    open_until = datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=remaining)
            return HealthSnapshot(
                source=self.source,
                state=self.state,
                consecutive_failures=self._consecutive_failures,
                total_successes=self._successes,
                total_failures=self._failures,
                last_success_at=self._last_success_at,
                last_failure_at=self._last_failure_at,
                last_error=self._last_error,
                open_until=open_until,
            )


# ── source contract ─────────────────────────────────────────────────────────


@dataclass
class SourceDescriptor:
    """Static, declarative facts about a source (no I/O needed to read them)."""

    name: str
    capabilities: frozenset[Capability]
    #: Lower wins when several sources serve the same capability.
    priority: int
    #: Is this the authoritative publisher of the data (the exchange itself)?
    official: bool = False
    #: Human-facing licence/usage note; surfaced in docs and in the audit row.
    usage_note: str = ""
    #: Approximate per-request latency in ms, used for planning only.
    typical_latency_ms: int = 800
    #: Symbol namespace this source expects: "bare" (COMI) or "yahoo" (COMI.CA).
    symbol_style: str = "bare"

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities


class MarketDataSource:
    """Base class: descriptor + health + honest ``available()``."""

    descriptor: SourceDescriptor

    def __init__(
        self,
        transport: Transport,
        *,
        health: Optional[SourceHealth] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.transport = transport
        self.health = health or SourceHealth(self.descriptor.name)
        self._clock = clock or now_utc
        self._request_count = 0
        self._count_lock = threading.Lock()

    # ── identity ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self.descriptor.capabilities

    def supports(self, capability: Capability) -> bool:
        return self.descriptor.supports(capability)

    # ── lifecycle ───────────────────────────────────────────────────────────

    def available(self) -> bool:
        """Cheap, offline check: is this source configured well enough to try?

        Must not perform network I/O — the router calls it for every request.
        Returning False is a statement about configuration, not about upstream
        health (that is what the circuit breaker is for).
        """
        return self.transport is not None

    def close(self) -> None:
        closer = getattr(self.transport, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # pragma: no cover
                pass

    # ── instrumentation ─────────────────────────────────────────────────────

    def _next_request_index(self) -> int:
        with self._count_lock:
            self._request_count += 1
            return self._request_count

    def provenance(
        self,
        *,
        fetched_at: Optional[datetime] = None,
        attempt_index: int = 0,
        latency_ms: Optional[float] = None,
        http_status: Optional[int] = None,
        endpoint: Optional[str] = None,
        locator: Optional[dict[str, Any]] = None,
    ) -> Provenance:
        return Provenance(
            source=self.name,
            fetched_at=fetched_at or self._clock(),
            attempt_index=attempt_index,
            latency_ms=latency_ms,
            http_status=http_status,
            endpoint=endpoint,
            locator=dict(locator or {}),
        )

    def wrap(
        self,
        value: Any,
        *,
        response: Optional[TransportResponse] = None,
        endpoint: Optional[str] = None,
        locator: Optional[dict[str, Any]] = None,
        attempt_index: int = 0,
    ) -> Sourced[Any]:
        return Sourced(
            value=value,
            provenance=self.provenance(
                latency_ms=response.latency_ms if response else None,
                http_status=response.status if response else None,
                endpoint=endpoint or (response.final_url or response.url if response else None),
                locator=locator,
                attempt_index=attempt_index,
            ),
        )

    def _get(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
    ) -> TransportResponse:
        """GET, normalising transport failures into :class:`SourceUnavailable`.

        Health is deliberately **not** accounted here. A source may try several
        endpoints internally before answering (see ``EGXOfficialSource``), and
        counting per-request would report a source as failing when it actually
        succeeded on the second candidate. The router sees the final outcome of
        the whole call, so that is where health is recorded.
        """
        try:
            resp = self.transport.request(
                "GET", url, headers=headers, params=params, timeout_s=timeout_s
            )
        except TransportError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise SourceUnavailable(self.name, f"transport error: {exc}") from exc
        if not resp.ok:
            raise SourceUnavailable(
                self.name,
                f"HTTP {resp.status}",
                retryable=resp.status >= 500 or resp.status == 429,
            )
        return resp

    def health_snapshot(self) -> HealthSnapshot:
        return self.health.snapshot()


# ── registry ────────────────────────────────────────────────────────────────


class SourceRegistry:
    """Holds the installed sources and answers capability queries."""

    def __init__(self) -> None:
        self._sources: dict[str, MarketDataSource] = {}
        self._lock = threading.Lock()

    def register(self, source: MarketDataSource, *, replace: bool = False) -> None:
        with self._lock:
            if source.name in self._sources and not replace:
                raise ValueError(f"source already registered: {source.name}")
            self._sources[source.name] = source

    def unregister(self, name: str) -> None:
        with self._lock:
            self._sources.pop(name, None)

    def get(self, name: str) -> Optional[MarketDataSource]:
        return self._sources.get(name)

    def all(self) -> list[MarketDataSource]:
        return list(self._sources.values())

    def for_capability(
        self,
        capability: Capability,
        *,
        include_open_circuits: bool = False,
    ) -> list[MarketDataSource]:
        """Sources declaring *capability*, best-first.

        Ordering: official sources first (an exchange's own print beats a
        third-party mirror), then declared priority, then healthy before
        tripped. Sources whose circuit is open are excluded unless asked for.
        """
        with self._lock:
            candidates = [
                s
                for s in self._sources.values()
                if s.supports(capability) and s.available()
            ]
        if not include_open_circuits:
            candidates = [s for s in candidates if not s.health.is_open]

        def rank(s: MarketDataSource) -> tuple[int, int, str]:
            official_rank = 0 if s.descriptor.official else 1
            if capability in OFFICIAL_PREFERRED and s.descriptor.official:
                official_rank = -1
            return (official_rank, s.descriptor.priority, s.name)

        return sorted(candidates, key=rank)

    def health_report(self) -> list[dict[str, Any]]:
        return [s.health_snapshot().as_dict() for s in self.all()]

    def close_all(self) -> None:
        for s in self.all():
            s.close()
