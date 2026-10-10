"""Ingestion: refresh reports, calendar-aware skipping, coverage accounting."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

import pytest

from tradingview_mcp.core.data.calendar import EGXCalendar
from tradingview_mcp.core.data.factory import build_audit_sink
from tradingview_mcp.core.data.ingest import EGXIngestor, symbols_for_index
from tradingview_mcp.core.data.quality import _expected_bar
from tradingview_mcp.core.data.router import DataRouter
from tradingview_mcp.core.data.sources.base import (
    Capability,
    MarketDataSource,
    SourceDescriptor,
    SourceRegistry,
    SourceUnavailable,
)
from tradingview_mcp.core.data.store import DataStore
from tradingview_mcp.core.data.transport import FakeTransport


class FixedSource(MarketDataSource):
    """Source with deterministic payloads, driven entirely by the test."""

    def __init__(self, name: str, *, responder, capabilities, priority: int = 10) -> None:
        self.descriptor = SourceDescriptor(
            name=name, capabilities=frozenset(capabilities), priority=priority, official=False
        )
        super().__init__(FakeTransport())
        self._responder = responder
        self.calls = 0

    def _call(self, kind: str, subject: Optional[str], **kwargs):
        self.calls += 1
        out = self._responder(kind, subject, **kwargs)
        if isinstance(out, Exception):
            raise out
        return self.wrap(out, locator={"kind": kind, "subject": subject})

    def daily_ohlcv(self, symbol: str, *, days: int = 365, attempt_index: int = 0):
        return self._call("daily_ohlcv", symbol, days=days, attempt_index=attempt_index)

    def quote(self, symbol: str, *, attempt_index: int = 0):
        return self._call("quote", symbol, attempt_index=attempt_index)

    def market_snapshot(self, *, attempt_index: int = 0):
        return self._call("market_snapshot", None, attempt_index=attempt_index)


def bars_through(calendar: EGXCalendar, sessions: int = 3) -> list[dict[str, Any]]:
    """Bars ending on the newest settled session (never stale by construction)."""
    last = _expected_bar(calendar, None)
    days = [last]
    while len(days) < sessions:
        days.insert(0, calendar.previous_trading_day(days[0]))
    return [
        {
            "date": d.isoformat(),
            "open": 10.0 + i,
            "high": 11.0 + i,
            "low": 9.5 + i,
            "close": 10.5 + i,
            "volume": 1000 * (i + 1),
        }
        for i, d in enumerate(days)
    ]


@pytest.fixture
def calendar() -> EGXCalendar:
    return EGXCalendar(holidays={})


@pytest.fixture
def store(tmp_path) -> DataStore:
    db = DataStore(tmp_path / "ingest.db")
    yield db
    db.close()


def build(calendar: EGXCalendar, store: DataStore, responder, *, capabilities=None) -> tuple[EGXIngestor, FixedSource]:
    source = FixedSource(
        "fixed",
        responder=responder,
        capabilities=capabilities or (Capability.DAILY_OHLCV, Capability.QUOTE, Capability.MARKET_SNAPSHOT),
    )
    registry = SourceRegistry()
    registry.register(source)
    router = DataRouter(registry, calendar=calendar, audit_sink=build_audit_sink(store))
    return EGXIngestor(router, store, calendar=calendar), source


class TestDailyRefresh:
    def test_bars_are_written_with_provenance(self, calendar, store) -> None:
        payload = bars_through(calendar, 3)
        ingestor, _ = build(calendar, store, lambda *a, **k: payload)

        report = ingestor.refresh_daily(["COMI"])
        assert report.status == "ok"
        assert report.counts == {"ok": 1}
        assert report.rows_written == 3

        rows = store.get_ohlcv("COMI")
        assert len(rows) == 3
        assert rows[-1]["source"] == "fixed"

    def test_run_row_is_closed_with_a_status(self, calendar, store) -> None:
        ingestor, _ = build(calendar, store, lambda *a, **k: bars_through(calendar, 2))
        report = ingestor.refresh_daily(["COMI"])
        run = store.get_run(report.run_id)
        assert run["kind"] == "ohlcv_daily"
        assert run["status"] == "ok"
        assert run["finished_at"] is not None
        assert run["rows_written"] == 2
        assert run["sources_used"] == "fixed"

    def test_second_run_skips_instead_of_refetching(self, calendar, store) -> None:
        ingestor, source = build(calendar, store, lambda *a, **k: bars_through(calendar, 2))
        ingestor.refresh_daily(["COMI"])
        calls_after_first = source.calls

        report = ingestor.refresh_daily(["COMI"])
        assert report.counts == {"skipped": 1}
        assert report.outcomes[0].reason
        assert source.calls == calls_after_first  # no request was made

    def test_force_refetches(self, calendar, store) -> None:
        ingestor, source = build(calendar, store, lambda *a, **k: bars_through(calendar, 2))
        ingestor.refresh_daily(["COMI"])
        before = source.calls
        report = ingestor.refresh_daily(["COMI"], force=True)
        assert report.counts == {"ok": 1}
        assert source.calls > before

    def test_partial_failure_is_reported_per_symbol(self, calendar, store) -> None:
        payload = bars_through(calendar, 2)

        def responder(kind, subject, **kwargs):
            if subject == "BAD":
                return SourceUnavailable("fixed", "no data")
            return payload

        ingestor, _ = build(calendar, store, responder)
        report = ingestor.refresh_daily(["COMI", "BAD", "TMGH"])

        assert report.status == "partial"
        assert report.counts == {"ok": 2, "failed": 1}
        problem = next(o for o in report.outcomes if o.status == "failed")
        assert problem.symbol == "BAD"
        assert "no data" in (problem.error or "")
        assert report.as_dict()["problems_total"] == 1

    def test_duplicate_symbols_are_requested_once(self, calendar, store) -> None:
        ingestor, source = build(calendar, store, lambda *a, **k: bars_through(calendar, 2))
        report = ingestor.refresh_daily(["COMI", "comi", "EGX:COMI"])
        assert report.requested == 1
        assert source.calls == 1

    def test_empty_symbol_list_closes_the_run(self, calendar, store) -> None:
        ingestor, _ = build(calendar, store, lambda *a, **k: bars_through(calendar, 1))
        report = ingestor.refresh_daily([])
        assert report.status == "empty"
        assert store.get_run(report.run_id)["status"] == "empty"

    def test_calendar_warnings_are_attached(self, calendar, store) -> None:
        ingestor, _ = build(calendar, store, lambda *a, **k: bars_through(calendar, 1))
        report = ingestor.refresh_daily(["COMI"])
        # The shipped holiday file is unverified for the current year, and the
        # run says so instead of pretending the calendar is authoritative.
        assert report.warnings


class TestQuoteRefresh:
    def test_quotes_are_appended(self, calendar, store) -> None:
        ingestor, _ = build(
            calendar, store, lambda *a, **k: {"symbol": "COMI", "price": 86.2, "volume": 1000}
        )
        ingestor.refresh_quotes(["COMI"])
        ingestor.refresh_quotes(["COMI"])
        assert store.stats().quote_rows == 2

    def test_quote_failures_do_not_stop_the_batch(self, calendar, store) -> None:
        def responder(kind, subject, **kwargs):
            if subject == "BAD":
                return SourceUnavailable("fixed", "timeout")
            return {"symbol": subject, "price": 10.0, "volume": 5}

        ingestor, _ = build(calendar, store, responder)
        report = ingestor.refresh_quotes(["COMI", "BAD"])
        assert report.counts == {"ok": 1, "failed": 1}


class TestUniverseRefresh:
    def test_snapshot_populates_symbols_and_quotes(self, calendar, store) -> None:
        rows = [
            {"symbol": "COMI", "name": "CIB", "close": 86.2, "volume": 1000},
            {"symbol": "TMGH", "name": "Telecom Egypt", "close": 38.95, "volume": 900},
        ]
        ingestor, _ = build(calendar, store, lambda *a, **k: rows)

        report = ingestor.refresh_universe(cross_check=False)
        assert report.status == "ok"
        assert report.rows_written == 2
        assert {s["symbol"] for s in store.get_symbols()} == {"COMI", "TMGH"}
        assert store.stats().quote_rows == 2

    def test_snapshot_failure_closes_the_run_as_failed(self, calendar, store) -> None:
        ingestor, _ = build(calendar, store, lambda *a, **k: SourceUnavailable("fixed", "503"))
        report = ingestor.refresh_universe(cross_check=False)
        assert report.status == "failed"
        assert store.get_run(report.run_id)["status"] == "failed"


class TestCoverage:
    def test_complete_symbols_are_counted(self, calendar, store) -> None:
        # Coverage asks whether every trading session in the window is stored,
        # so the symbol needs history that spans the whole window.
        payload = bars_through(calendar, 30)
        ingestor, _ = build(calendar, store, lambda *a, **k: payload)
        ingestor.refresh_daily(["COMI"])

        cover = ingestor.coverage(["COMI"], lookback_days=30)
        assert cover["symbols_checked"] == 1
        assert cover["complete"] == 1
        assert cover["current_ratio"] == 1.0

    def test_missing_history_is_visible(self, calendar, store) -> None:
        ingestor, _ = build(calendar, store, lambda *a, **k: bars_through(calendar, 1))
        cover = ingestor.coverage(["COMI"], lookback_days=30)
        assert cover["complete"] == 0
        assert cover["worst_offenders"][0]["symbol"] == "COMI"
        assert cover["worst_offenders"][0]["first_missing"]

    def test_calendar_drives_the_expected_session_count(self, calendar, store) -> None:
        ingestor, _ = build(calendar, store, lambda *a, **k: bars_through(calendar, 1))
        cover = ingestor.coverage(["COMI"], lookback_days=7)
        last_settled = _expected_bar(calendar, None)
        expected = calendar.trading_days_between(last_settled - timedelta(days=7), last_settled)
        assert cover["window"]["expected_sessions"] == len(expected)
        # Today's (unsettled) session is never counted as a gap.
        assert cover["window"]["to"] <= last_settled.isoformat()


class TestSymbolHelpers:
    def test_index_constituents_are_resolved(self) -> None:
        symbols = symbols_for_index("EGX30")
        assert "COMI" in symbols
        assert len(symbols) >= 25

    def test_limit_is_applied(self) -> None:
        assert len(symbols_for_index("EGX30", limit=5)) == 5

    def test_unknown_index_raises(self) -> None:
        with pytest.raises(ValueError):
            symbols_for_index("NOT_AN_INDEX")
