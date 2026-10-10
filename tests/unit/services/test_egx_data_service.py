"""MCP data-layer service: status shape, refresh reports, offline behaviour.

Hermetic end to end — the MCP tools are exercised against a real ``DataLayer``
whose transport is a fake and whose only usable source is a deterministic stub,
so nothing here depends on the network or on a live EGX page.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest

from tradingview_mcp.core.data.calendar import EGXCalendar
from tradingview_mcp.core.data.factory import DataLayer
from tradingview_mcp.core.data.quality import _expected_bar
from tradingview_mcp.core.data.sources.base import (
    Capability,
    MarketDataSource,
    SourceDescriptor,
    SourceUnavailable,
)
from tradingview_mcp.core.data.transport import FakeTransport


class FixedSource(MarketDataSource):
    def __init__(self, *, bars, quotes=None, snapshot=None, fail_on=()):
        self.descriptor = SourceDescriptor(
            name="fixed",
            capabilities=frozenset(
                {Capability.DAILY_OHLCV, Capability.QUOTE, Capability.MARKET_SNAPSHOT}
            ),
            priority=1,
        )
        super().__init__(FakeTransport())
        self._bars = bars
        self._quotes = quotes or {}
        self._snapshot = snapshot or []
        self._fail_on = set(fail_on)

    def _guard(self, symbol: Optional[str]) -> None:
        if symbol in self._fail_on:
            raise SourceUnavailable("fixed", f"simulated failure for {symbol}")

    def daily_ohlcv(self, symbol: str, *, days: int = 365, attempt_index: int = 0):
        self._guard(symbol)
        return self.wrap(list(self._bars), locator={"subject": symbol})

    def quote(self, symbol: str, *, attempt_index: int = 0):
        self._guard(symbol)
        return self.wrap({"symbol": symbol, "price": 86.2, "volume": 1000})

    def market_snapshot(self, *, attempt_index: int = 0):
        self._guard(None)
        return self.wrap(list(self._snapshot))


@pytest.fixture
def calendar() -> EGXCalendar:
    return EGXCalendar(holidays={})


@pytest.fixture
def bars(calendar: EGXCalendar) -> list[dict[str, Any]]:
    last = _expected_bar(calendar, None)
    prev = calendar.previous_trading_day(last)
    return [
        {"date": prev.isoformat(), "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000},
        {"date": last.isoformat(), "open": 10.5, "high": 12, "low": 10, "close": 11.5, "volume": 1200},
    ]


@pytest.fixture
def layer(tmp_path, calendar, bars, monkeypatch):
    source = FixedSource(
        bars=bars,
        snapshot=[{"symbol": "COMI", "name": "CIB", "close": 86.2, "volume": 1000}],
        fail_on={"BAD"},
    )
    built = DataLayer(
        transport=FakeTransport(),
        db_path=tmp_path / "svc.db",
        calendar=calendar,
        extra_sources=[source],
        include_tradingview=False,
    )
    monkeypatch.setattr(
        "tradingview_mcp.core.data.factory.get_layer", lambda: built
    )
    yield built
    built.close()


class TestStatus:
    def test_reports_the_market_clock_and_store(self, layer) -> None:
        from tradingview_mcp.core.services.egx_data_service import egx_data_status

        payload = egx_data_status(index="EGX30")
        assert "error" not in payload
        assert payload["market_clock"]["phase"]
        assert payload["market_clock"]["newest_settled_session"]
        assert "store" in payload and "sources" in payload
        assert "fixed" in {row["source"] for row in payload["sources"]}

    def test_coverage_is_measured_against_the_calendar(self, layer) -> None:
        from tradingview_mcp.core.services.egx_data_service import egx_data_status

        payload = egx_data_status(index="EGX30")
        assert payload["coverage"]["symbols_checked"] >= 25
        assert payload["coverage"]["window"]["to"] <= payload["market_clock"]["newest_settled_session"]
        assert payload["coverage"]["complete"] == 0  # nothing ingested yet

    def test_unknown_index_is_an_error_envelope(self, layer) -> None:
        from tradingview_mcp.core.services.egx_data_service import egx_data_status

        payload = egx_data_status(index="NOPE")
        assert payload["error"]["code"] == "INVALID_PARAMETER"

    def test_calendar_warnings_are_surfaced(self, layer) -> None:
        from tradingview_mcp.core.services.egx_data_service import egx_data_status

        assert egx_data_status()["calendar_warnings"]


class TestRefreshDaily:
    def test_explicit_symbols_are_fetched_and_stored(self, layer, bars) -> None:
        from tradingview_mcp.core.services.egx_data_service import egx_data_refresh_daily

        report = egx_data_refresh_daily(symbols="comi, tmgh")
        assert report["status"] == "ok"
        assert report["requested"] == 2
        assert report["counts"] == {"ok": 2}
        assert report["rows_written"] == 2 * len(bars)
        assert layer.store.get_ohlcv("COMI")

    def test_index_symbols_are_resolved(self, layer) -> None:
        from tradingview_mcp.core.services.egx_data_service import egx_data_refresh_daily

        report = egx_data_refresh_daily(index="EGX30", max_symbols=5)
        assert report["requested"] == 5

    def test_failures_are_reported_per_symbol(self, layer) -> None:
        from tradingview_mcp.core.services.egx_data_service import egx_data_refresh_daily

        report = egx_data_refresh_daily(symbols="COMI,BAD")
        assert report["status"] == "partial"
        assert report["counts"] == {"ok": 1, "failed": 1}
        assert report["problems"][0]["symbol"] == "BAD"

    def test_second_call_skips_without_refetching(self, layer) -> None:
        from tradingview_mcp.core.services.egx_data_service import egx_data_refresh_daily

        egx_data_refresh_daily(symbols="COMI")
        report = egx_data_refresh_daily(symbols="COMI")
        assert report["counts"] == {"skipped": 1}

    def test_max_symbols_is_clamped(self, layer) -> None:
        from tradingview_mcp.core.services.egx_data_service import (
            MAX_SYMBOLS_PER_CALL,
            egx_data_refresh_daily,
        )

        report = egx_data_refresh_daily(index="EGX70", max_symbols=10_000)
        assert report["requested"] <= MAX_SYMBOLS_PER_CALL

    def test_unknown_index_returns_an_envelope(self, layer) -> None:
        from tradingview_mcp.core.services.egx_data_service import egx_data_refresh_daily

        assert egx_data_refresh_daily(index="NOPE")["error"]["code"] == "INVALID_PARAMETER"

    def test_no_symbols_returns_no_data(self, layer, monkeypatch) -> None:
        from tradingview_mcp.core.services import egx_data_service as svc

        # Only reachable when the fallback universe is empty (e.g. a stripped
        # install with no coinlist shipped).
        monkeypatch.setattr(svc, "symbols_from_coinlist", lambda *a, **k: [])
        payload = svc.egx_data_refresh_daily(symbols=" , ")
        assert payload["error"]["code"] == "NO_DATA"

    def test_whitespace_symbols_fall_back_to_the_full_universe(self, layer) -> None:
        from tradingview_mcp.core.services.egx_data_service import egx_data_refresh_daily

        report = egx_data_refresh_daily(symbols=" ", max_symbols=3)
        assert report["requested"] == 3

    def test_offline_mode_refuses_to_fetch(self, layer, monkeypatch) -> None:
        from tradingview_mcp.core.services.egx_data_service import egx_data_refresh_daily

        monkeypatch.setenv("EGX_DATA_OFFLINE", "1")
        payload = egx_data_refresh_daily(symbols="COMI")
        assert payload["error"]["code"] == "DEPENDENCY_MISSING"
        assert "offline" in payload["error"]["message"].lower()


class TestRefreshUniverse:
    def test_snapshot_populates_symbols_and_quotes(self, layer) -> None:
        from tradingview_mcp.core.services.egx_data_service import egx_data_refresh_universe

        report = egx_data_refresh_universe(cross_check=False)
        assert report["status"] == "ok"
        assert report["rows_written"] == 1
        assert layer.store.get_symbols()[0]["symbol"] == "COMI"
        assert layer.store.stats().quote_rows == 1

    def test_offline_mode_refuses_to_fetch(self, layer, monkeypatch) -> None:
        from tradingview_mcp.core.services.egx_data_service import egx_data_refresh_universe

        monkeypatch.setenv("EGX_DATA_OFFLINE", "1")
        assert egx_data_refresh_universe()["error"]["code"] == "DEPENDENCY_MISSING"


class TestSymbolResolution:
    def test_explicit_list_wins_over_index(self) -> None:
        from tradingview_mcp.core.services.egx_data_service import _resolve_symbols

        assert _resolve_symbols("COMI,TMGH", "EGX30") == ["COMI", "TMGH"]

    def test_semicolons_and_prefixes_are_accepted(self) -> None:
        from tradingview_mcp.core.services.egx_data_service import _resolve_symbols

        assert _resolve_symbols("egx:COMI; TMGH") == ["COMI", "TMGH"]

    def test_empty_falls_back_to_the_full_universe(self) -> None:
        from tradingview_mcp.core.services.egx_data_service import _resolve_symbols

        symbols = _resolve_symbols()
        assert "COMI" in symbols
        assert len(symbols) > 100
