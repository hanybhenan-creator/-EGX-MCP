"""Router: fallback order, quality gating, corroboration, audit trail."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from tradingview_mcp.core.data.calendar import EGXCalendar
from tradingview_mcp.core.data.provenance import QualityFlag
from tradingview_mcp.core.data.router import AllSourcesFailed, DataRouter
from tradingview_mcp.core.data.sources.base import (
    Capability,
    MarketDataSource,
    SourceDescriptor,
    SourceRegistry,
    SourceUnavailable,
)
from tradingview_mcp.core.data.transport import FakeTransport


class StubSource(MarketDataSource):
    """Minimal source whose behaviour is fully controlled by the test."""

    def __init__(
        self,
        name: str,
        *,
        capabilities: tuple[Capability, ...] = (Capability.QUOTE, Capability.DAILY_OHLCV),
        priority: int = 50,
        official: bool = False,
        responder: Optional[Any] = None,
    ) -> None:
        self.descriptor = SourceDescriptor(
            name=name,
            capabilities=frozenset(capabilities),
            priority=priority,
            official=official,
        )
        super().__init__(FakeTransport())
        self._responder = responder

    def _call(self, kind: str, subject: Optional[str], **kwargs: Any):
        if self._responder is None:
            raise SourceUnavailable(self.name, "no responder configured", retryable=False)
        out = self._responder(kind, subject, **kwargs)
        if isinstance(out, Exception):
            raise out
        return self.wrap(out, locator={"kind": kind, "subject": subject})

    def quote(self, symbol: str, *, attempt_index: int = 0):
        return self._call("quote", symbol, attempt_index=attempt_index)

    def daily_ohlcv(self, symbol: str, *, days: int = 365, attempt_index: int = 0):
        return self._call("daily_ohlcv", symbol, days=days, attempt_index=attempt_index)

    def market_snapshot(self, *, attempt_index: int = 0):
        return self._call("market_snapshot", None, attempt_index=attempt_index)


def make_quote(price: float, *, volume: int = 1000, previous_close: float | None = None) -> dict[str, Any]:
    quote = {"symbol": "COMI", "price": price, "volume": volume, "close": price}
    if previous_close is not None:
        quote["previous_close"] = previous_close
    return quote


@pytest.fixture
def calendar() -> EGXCalendar:
    return EGXCalendar(holidays={})


class TestFallbackChain:
    def test_falls_back_to_the_next_source(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        registry.register(
            StubSource("primary", priority=10, responder=lambda *a, **k: SourceUnavailable("primary", "down"))
        )
        registry.register(StubSource("secondary", priority=20, responder=lambda *a, **k: make_quote(86.2)))

        router = DataRouter(registry, calendar=calendar)
        res = router.fetch(Capability.QUOTE, "COMI")

        assert res.ok
        assert res.result.provenance.source == "secondary"
        assert [a.source for a in res.attempts] == ["primary", "secondary"]
        assert res.attempts[0].ok is False
        assert res.attempts[1].ok is True

    def test_official_source_wins_over_a_higher_priority_number(self, calendar: EGXCalendar) -> None:
        snapshot = [{"symbol": "COMI", "close": 86.2}]
        registry = SourceRegistry()
        registry.register(
            StubSource("mirror", priority=1, capabilities=(Capability.MARKET_SNAPSHOT,), responder=lambda *a, **k: list(snapshot))
        )
        registry.register(
            StubSource("exchange", priority=99, official=True, capabilities=(Capability.MARKET_SNAPSHOT,), responder=lambda *a, **k: list(snapshot))
        )
        res = DataRouter(registry, calendar=calendar).fetch(Capability.MARKET_SNAPSHOT, None)
        assert res.result.provenance.source == "exchange"

    def test_all_sources_failing_raises_with_reasons(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        registry.register(StubSource("a", priority=1, responder=lambda *a, **k: SourceUnavailable("a", "http 503")))
        registry.register(StubSource("b", priority=2, responder=lambda *a, **k: SourceUnavailable("b", "timeout")))
        router = DataRouter(registry, calendar=calendar)

        with pytest.raises(AllSourcesFailed) as exc:
            router.fetch_or_raise(Capability.QUOTE, "COMI")
        assert "http 503" in str(exc.value)
        assert len(exc.value.attempts) == 2

    def test_missing_result_is_reported_not_raised_by_fetch(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        registry.register(StubSource("a", responder=lambda *a, **k: SourceUnavailable("a", "down")))
        res = DataRouter(registry, calendar=calendar).fetch(Capability.QUOTE, "COMI")
        assert not res.ok
        assert res.value is None
        assert res.as_dict()["ok"] is False

    def test_no_source_for_capability_raises(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        registry.register(StubSource("a", capabilities=(Capability.SYMBOL_MASTER,)))
        with pytest.raises(AllSourcesFailed):
            DataRouter(registry, calendar=calendar).fetch(Capability.QUOTE, "COMI")

    def test_sources_without_the_capability_are_never_called(self, calendar: EGXCalendar) -> None:
        calls: list[str] = []

        def responder(kind, subject, **kwargs):
            calls.append(kind)
            return make_quote(86.2)

        registry = SourceRegistry()
        registry.register(StubSource("only_bars", capabilities=(Capability.DAILY_OHLCV,), responder=responder))
        with pytest.raises(AllSourcesFailed):
            DataRouter(registry, calendar=calendar).fetch(Capability.QUOTE, "COMI")
        assert calls == []


class TestQualityGating:
    def test_invalid_value_is_rejected_and_the_next_source_used(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        registry.register(StubSource("broken", priority=1, responder=lambda *a, **k: make_quote(0.0)))
        registry.register(StubSource("good", priority=2, responder=lambda *a, **k: make_quote(86.2)))

        res = DataRouter(registry, calendar=calendar).fetch(Capability.QUOTE, "COMI")
        assert res.result.provenance.source == "good"
        assert res.attempts[0].flags and "INVALID" in ",".join(res.attempts[0].flags)

    def test_outlier_is_accepted_but_flagged(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        registry.register(
            StubSource("egx", responder=lambda *a, **k: make_quote(200.0, previous_close=100.0))
        )
        res = DataRouter(registry, calendar=calendar).fetch(Capability.QUOTE, "COMI")
        assert res.ok
        assert QualityFlag.OUTLIER in res.flags

    def test_resolution_exposes_quality_flags(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        registry.register(StubSource("egx", responder=lambda *a, **k: make_quote(86.2)))
        res = DataRouter(registry, calendar=calendar).fetch(Capability.QUOTE, "COMI")
        assert res.result is not None
        assert res.result.is_usable
        assert "provenance" in res.as_dict()

    def test_snapshot_with_too_many_unpriced_rows_is_invalid(self, calendar: EGXCalendar) -> None:
        rows = [{"symbol": f"S{i}", "close": None} for i in range(10)]
        registry = SourceRegistry()
        registry.register(
            StubSource("egx", capabilities=(Capability.MARKET_SNAPSHOT,), responder=lambda *a, **k: rows)
        )
        registry.register(
            StubSource(
                "yahoo",
                capabilities=(Capability.MARKET_SNAPSHOT,),
                priority=60,
                responder=lambda *a, **k: [{"symbol": "S1", "close": 10.0}],
            )
        )
        res = DataRouter(registry, calendar=calendar).fetch(Capability.MARKET_SNAPSHOT, None)
        assert res.result.provenance.source == "yahoo"


class TestCrossChecks:
    def test_two_agreeing_sources_are_marked_agreement(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        registry.register(StubSource("egx", priority=10, responder=lambda *a, **k: make_quote(86.2, volume=1_000_000)))
        registry.register(StubSource("yahoo", priority=20, responder=lambda *a, **k: make_quote(86.35, volume=1_005_000)))

        res = DataRouter(registry, calendar=calendar).fetch(
            Capability.QUOTE, "COMI", cross_check=True
        )
        assert QualityFlag.AGREEMENT in res.flags
        assert res.corroborated
        assert res.result.cross_checks
        assert res.result.cross_checks[0]["verdict"] == "AGREEMENT"

    def test_disagreeing_sources_are_marked_conflict(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        registry.register(StubSource("egx", priority=10, responder=lambda *a, **k: make_quote(86.2)))
        registry.register(StubSource("yahoo", priority=20, responder=lambda *a, **k: make_quote(95.0)))

        res = DataRouter(registry, calendar=calendar).fetch(Capability.QUOTE, "COMI", cross_check=True)
        assert QualityFlag.CONFLICT in res.flags
        assert not res.result.is_usable  # a conflicted value is not tradable

    def test_single_source_is_not_claimed_as_verified(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        registry.register(StubSource("egx", responder=lambda *a, **k: make_quote(86.2)))
        res = DataRouter(registry, calendar=calendar).fetch(Capability.QUOTE, "COMI", cross_check=True)
        assert not res.corroborated
        assert not res.result.cross_checks

    def test_snapshots_are_compared_symbol_wise(self, calendar: EGXCalendar) -> None:
        left = [{"symbol": "COMI", "close": 86.2}, {"symbol": "TMGH", "close": 38.95}]
        right = [{"symbol": "COMI", "close": 86.3}, {"symbol": "TMGH", "close": 38.90}]
        registry = SourceRegistry()
        registry.register(StubSource("egx", capabilities=(Capability.MARKET_SNAPSHOT,), priority=10, responder=lambda *a, **k: left))
        registry.register(StubSource("yahoo", capabilities=(Capability.MARKET_SNAPSHOT,), priority=20, responder=lambda *a, **k: right))

        res = DataRouter(registry, calendar=calendar).fetch(Capability.MARKET_SNAPSHOT, None, cross_check=True)
        assert QualityFlag.AGREEMENT in res.flags
        assert "2/2" in " ".join(res.result.notes)

    def test_snapshot_disagreement_on_overlap_is_reported(self, calendar: EGXCalendar) -> None:
        left = [{"symbol": "COMI", "close": 86.2}, {"symbol": "TMGH", "close": 38.95}]
        right = [{"symbol": "COMI", "close": 86.2}, {"symbol": "TMGH", "close": 45.00}]
        registry = SourceRegistry()
        registry.register(StubSource("egx", capabilities=(Capability.MARKET_SNAPSHOT,), priority=10, responder=lambda *a, **k: left))
        registry.register(StubSource("yahoo", capabilities=(Capability.MARKET_SNAPSHOT,), priority=20, responder=lambda *a, **k: right))

        res = DataRouter(registry, calendar=calendar).fetch(Capability.MARKET_SNAPSHOT, None, cross_check=True)
        assert QualityFlag.DIVERGENCE in res.flags or QualityFlag.CONFLICT in res.flags
        assert any(c["symbol"] == "TMGH" for c in res.result.cross_checks)


class TestCircuitBreaker:
    def test_repeated_failures_open_the_circuit(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        dead = StubSource("dead", priority=1, responder=lambda *a, **k: SourceUnavailable("dead", "connection refused"))
        registry.register(dead)
        registry.register(StubSource("alive", priority=2, responder=lambda *a, **k: make_quote(86.2)))
        router = DataRouter(registry, calendar=calendar)

        for _ in range(3):
            router.fetch(Capability.QUOTE, "COMI")

        assert dead.health.is_open
        assert [s.name for s in registry.for_capability(Capability.QUOTE)] == ["alive"]

    def test_open_circuit_can_still_be_forced_for_diagnostics(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        dead = StubSource("dead", responder=lambda *a, **k: SourceUnavailable("dead", "boom"))
        registry.register(dead)
        for _ in range(3):
            try:
                DataRouter(registry, calendar=calendar).fetch(Capability.QUOTE, "COMI")
            except AllSourcesFailed:
                pass
        assert dead.health.is_open
        assert [s.name for s in registry.for_capability(Capability.QUOTE, include_open_circuits=True)] == ["dead"]


class TestAuditTrail:
    def test_every_attempt_is_reported_to_the_sink(self, calendar: EGXCalendar) -> None:
        seen: list[tuple[str, bool, str | None]] = []

        def sink(attempt, subject):
            seen.append((attempt.source, attempt.ok, attempt.error))

        registry = SourceRegistry()
        registry.register(StubSource("primary", priority=1, responder=lambda *a, **k: SourceUnavailable("primary", "503")))
        registry.register(StubSource("secondary", priority=2, responder=lambda *a, **k: make_quote(86.2)))

        router = DataRouter(registry, calendar=calendar, audit_sink=sink)
        router.fetch(Capability.QUOTE, "COMI")

        assert seen[0][0] == "primary" and seen[0][1] is False
        assert seen[1][0] == "secondary" and seen[1][1] is True

    def test_a_broken_sink_never_breaks_fetching(self, calendar: EGXCalendar) -> None:
        def sink(attempt, subject):
            raise RuntimeError("database on fire")

        registry = SourceRegistry()
        registry.register(StubSource("a", responder=lambda *a, **k: make_quote(86.2)))
        res = DataRouter(registry, calendar=calendar, audit_sink=sink).fetch(Capability.QUOTE, "COMI")
        assert res.ok

    def test_health_report_is_serialisable(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        registry.register(StubSource("a", responder=lambda *a, **k: make_quote(86.2)))
        report = DataRouter(registry, calendar=calendar).health_report()
        assert report[0]["source"] == "a"
        assert report[0]["state"] == "closed"


class TestCapabilityRouting:
    def test_daily_bars_route_to_the_bars_method(self, calendar: EGXCalendar) -> None:
        registry = SourceRegistry()
        registry.register(
            StubSource(
                "yahoo",
                capabilities=(Capability.DAILY_OHLCV,),
                responder=lambda kind, subject, **k: _recent_bars(calendar),
            )
        )
        res = DataRouter(registry, calendar=calendar).fetch(Capability.DAILY_OHLCV, "COMI", days=30)
        assert res.ok
        assert res.attempts[0].rows and res.attempts[0].rows >= 2

    def test_subject_is_passed_positionally(self, calendar: EGXCalendar) -> None:
        captured: dict[str, Any] = {}

        def responder(kind, subject, **kwargs):
            captured["subject"] = subject
            captured["kwargs"] = kwargs
            return make_quote(86.2)

        registry = SourceRegistry()
        registry.register(StubSource("a", responder=responder))
        DataRouter(registry, calendar=calendar).fetch(Capability.QUOTE, "COMI", days=42)
        assert captured["subject"] == "COMI"
        # QUOTE's responder takes no ``days``; the router must not force it in.
        assert "days" not in captured["kwargs"]


def _recent_bars(calendar: EGXCalendar) -> list[dict[str, Any]]:
    """Bars ending on the newest settled session, so the freshness gate passes."""
    from datetime import date, timedelta

    from tradingview_mcp.core.data.quality import _expected_bar

    last = _expected_bar(calendar, None)
    prev = calendar.previous_trading_day(last)
    return [
        {"date": prev.isoformat(), "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000},
        {"date": last.isoformat(), "open": 10.5, "high": 12, "low": 10, "close": 11.5, "volume": 1200},
    ]
