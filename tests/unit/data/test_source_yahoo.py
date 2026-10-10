"""Yahoo adapter: symbol mapping, null-candle handling, error surfaces."""

from __future__ import annotations

import pytest

from tradingview_mcp.core.data.provenance import QualityFlag
from tradingview_mcp.core.data.sources.base import (
    Capability,
    SourceDataError,
    SourceParseError,
    SourceUnavailable,
)
from tradingview_mcp.core.data.sources.yahoo import (
    YahooFinanceSource,
    bare_symbol,
    yahoo_symbol,
)
from tradingview_mcp.core.data.transport import FakeTransport, json_response


@pytest.fixture
def transport(yahoo_chart_payload, yahoo_error_payload) -> FakeTransport:
    return (
        FakeTransport()
        .add_json("COMI.CA", yahoo_chart_payload)
        .add_json("^CASE30", yahoo_chart_payload)
        .add_json("DELISTED.CA", yahoo_error_payload)
        .add_text("BADJSON.CA", "<html>not json</html>")
    )


@pytest.fixture
def source(transport: FakeTransport) -> YahooFinanceSource:
    return YahooFinanceSource(transport)


class TestSymbolMapping:
    def test_egx_bare_ticker_gets_cairo_suffix(self) -> None:
        assert yahoo_symbol("comi", "EGX") == "COMI.CA"

    def test_already_qualified_symbol_is_untouched(self) -> None:
        assert yahoo_symbol("COMI.CA", "EGX") == "COMI.CA"

    def test_index_caret_symbol_is_untouched(self) -> None:
        assert yahoo_symbol("^CASE30", "EGX") == "^CASE30"

    def test_other_exchanges_map_to_their_suffix(self) -> None:
        assert yahoo_symbol("THYAO", "BIST") == "THYAO.IS"
        assert yahoo_symbol("2222", "TADAWUL") == "2222.SR"

    def test_unknown_exchange_passes_through(self) -> None:
        assert yahoo_symbol("AAPL", "NASDAQ") == "AAPL"
        assert yahoo_symbol("BTC-USD", "YAHOO") == "BTC-USD"

    def test_empty_symbol_rejected(self) -> None:
        with pytest.raises(ValueError):
            yahoo_symbol("", "EGX")

    def test_bare_symbol_strips_suffix(self) -> None:
        assert bare_symbol("COMI.CA") == "COMI"
        assert bare_symbol("AAPL") == "AAPL"


class TestDailyBars:
    def test_bars_are_returned_oldest_first(self, source) -> None:
        result = source.daily_ohlcv("COMI")
        dates = [b["date"] for b in result.value]
        assert dates == sorted(dates)

    def test_null_candles_are_dropped_not_zeroed(self, source) -> None:
        result = source.daily_ohlcv("COMI")
        # The fixture ships 8 timestamps, one of which is entirely null.
        assert len(result.value) == 7
        assert all(b["close"] > 0 for b in result.value)
        assert all(b["volume"] is not None for b in result.value)

    def test_bars_carry_expected_fields(self, source) -> None:
        bar = source.daily_ohlcv("COMI").value[-1]
        assert set(bar) >= {"date", "open", "high", "low", "close", "volume", "datetime"}

    def test_adjusted_close_is_carried_when_present(self, source) -> None:
        bars = source.daily_ohlcv("COMI").value
        assert any("adj_close" in b for b in bars)

    def test_provenance_records_the_request(self, source) -> None:
        result = source.daily_ohlcv("COMI", days=30)
        assert result.provenance.source == "yahoo"
        assert result.provenance.locator["yahoo_symbol"] == "COMI.CA"
        assert result.provenance.locator["range"] == "1mo"
        assert result.provenance.http_status == 200

    def test_meta_is_attached(self, source) -> None:
        result = source.daily_ohlcv("COMI")
        assert result.meta["currency"] == "EGP"
        assert result.meta["name"] is None or isinstance(result.meta["name"], str)

    def test_delisted_symbol_raises_data_error(self, source) -> None:
        with pytest.raises(SourceDataError):
            source.daily_ohlcv("DELISTED")

    def test_non_json_body_raises_parse_error(self, source) -> None:
        with pytest.raises(SourceParseError):
            source.daily_ohlcv("BADJSON")


class TestQuote:
    def test_quote_uses_chart_metadata(self, source) -> None:
        quote = source.quote("COMI").value
        assert quote["symbol"] == "COMI"
        assert quote["price"] == pytest.approx(86.2)
        assert quote["previous_close"] == pytest.approx(84.1)

    def test_change_percent_is_derived(self, source) -> None:
        quote = source.quote("COMI").value
        expected = (86.2 / 84.1 - 1) * 100
        assert quote["change_percent"] == pytest.approx(expected, rel=1e-3)

    def test_source_timestamp_is_carried(self, source) -> None:
        quote = source.quote("COMI").value
        assert quote["source_timestamp"] is not None


class TestIndexQuote:
    def test_unverified_mapping_is_flagged_not_hidden(self, source) -> None:
        result = source.index_quote("EGX30")
        assert result.has(QualityFlag.UNCROSSCHECKED)
        assert result.provenance.locator["mapping_verified"] is False
        assert "unverified" in " ".join(result.notes)

    def test_unknown_index_is_unavailable(self, source) -> None:
        with pytest.raises(SourceUnavailable):
            source.index_quote("NOT_AN_INDEX")


class TestWindows:
    def test_intraday_window_is_clamped_to_upstream_limit(self, source, transport) -> None:
        source.intraday_ohlcv("COMI", interval="15m", days=365)
        call = transport.calls[-1]
        assert call.params["interval"] == "15m"
        span_days = (call.params["period2"] - call.params["period1"]) / 86_400
        assert span_days <= 61  # 60-day upstream limit + end-of-day padding

    def test_unsupported_interval_is_rejected(self, source) -> None:
        with pytest.raises(SourceUnavailable):
            source.intraday_ohlcv("COMI", interval="3s")


class TestCapabilities:
    def test_declares_only_what_it_can_serve(self, source) -> None:
        assert source.supports(Capability.DAILY_OHLCV)
        assert source.supports(Capability.QUOTE)
        assert not source.supports(Capability.MARKET_SNAPSHOT)
        assert not source.supports(Capability.SYMBOL_MASTER)

    def test_source_calls_do_not_self_account_health(self, source) -> None:
        # Health is recorded by the router, once per source call, so a source
        # that succeeds on a fallback endpoint is not counted as a failure.
        source.daily_ohlcv("COMI")
        assert source.health_snapshot().total_successes == 0

    def test_parse_bars_is_pure(self) -> None:
        result = {"timestamp": [1, 2], "indicators": {"quote": [{"open": [1, 2], "high": [2, 3], "low": [0.5, 1], "close": [1.5, 2.5], "volume": [10, 20]}]}}
        bars = YahooFinanceSource.parse_bars(result)
        assert len(bars) == 2
        assert bars[0]["close"] == 1.5


def test_non_object_payload_is_a_parse_error(source, transport) -> None:
    transport.add_text("WEIRD.CA", "[1,2,3]")
    with pytest.raises(SourceParseError):
        source.daily_ohlcv("WEIRD")


def test_transport_failure_bubbles_up(transport, source) -> None:
    from tradingview_mcp.core.data.transport import TransportError

    transport.add("TIMEOUT.CA", TransportError("timeout", transient=True))
    with pytest.raises(TransportError):
        source.daily_ohlcv("TIMEOUT")
