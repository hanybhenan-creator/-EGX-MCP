"""EGX official scraper: endpoint fallback, column mapping, honest reporting."""

from __future__ import annotations

import pytest

from tradingview_mcp.core.data.provenance import QualityFlag
from tradingview_mcp.core.data.sources.base import (
    Capability,
    SourceDataError,
    SourceParseError,
    SourceUnavailable,
)
from tradingview_mcp.core.data.sources.egx_official import EGXOfficialSource
from tradingview_mcp.core.data.transport import FakeTransport, TransportError

INDEX_PAGE = """
<html><body><table>
<tr><th>Index</th><th>Close</th><th>Change %</th></tr>
<tr><td>EGX 30</td><td>32,456.78</td><td>0.85%</td></tr>
<tr><td>EGX 70</td><td>9,123.44</td><td>-0.32%</td></tr>
</table></body></html>
"""

JSON_ONLY_PAGE = """
<html><body>
<div id="grid"></div>
<script>
  window.__GRID__ = [{"symbol":"COMI","close":86.2,"volume":1234567},
                     {"symbol":"TMGH","close":38.95,"volume":845120}];
</script>
</body></html>
"""


@pytest.fixture
def source() -> EGXOfficialSource:
    return EGXOfficialSource(FakeTransport())


class TestSnapshotParsing:
    def test_reads_the_price_grid(self, source, egx_market_watch_html) -> None:
        parsed = source.parse_snapshot(egx_market_watch_html)
        assert parsed.report.rows == 7
        assert parsed.report.strategy == "html_table"

    def test_values_are_typed_not_strings(self, source, egx_market_watch_html) -> None:
        row = source.parse_snapshot(egx_market_watch_html).rows[0]
        assert row["symbol"] == "COMI"
        assert row["open"] == pytest.approx(84.1)
        assert row["high"] == pytest.approx(86.75)
        assert row["low"] == pytest.approx(83.95)
        assert row["close"] == pytest.approx(86.2)
        assert row["volume"] == 1_234_567
        assert row["change_pct"] == pytest.approx(2.5)

    def test_totals_row_is_not_a_symbol(self, source, egx_market_watch_html) -> None:
        symbols = {r["symbol"] for r in source.parse_snapshot(egx_market_watch_html).rows}
        assert "COMI" in symbols
        assert not any("TOTAL" in s for s in symbols)

    def test_all_columns_are_mapped(self, source, egx_market_watch_html) -> None:
        report = source.parse_snapshot(egx_market_watch_html).report
        assert set(report.fields_present) >= {"symbol", "name", "open", "high", "low", "close", "change_pct", "volume"}
        assert report.fields_missing == []

    def test_arabic_locale_page_is_read(self, source, egx_market_watch_arabic_html) -> None:
        parsed = source.parse_snapshot(egx_market_watch_arabic_html)
        assert parsed.report.rows == 2
        row = parsed.rows[0]
        assert row["symbol"] == "HRHO"
        assert row["close"] == pytest.approx(22.15)
        assert row["volume"] == 5_432_100
        assert row["change_pct"] == pytest.approx(3.02)
        assert row["sector"] == "خدمات مالية"

    def test_arabic_parenthesised_negative_is_signed(self, source, egx_market_watch_arabic_html) -> None:
        rows = source.parse_snapshot(egx_market_watch_arabic_html).rows
        assert rows[1]["change_pct"] == pytest.approx(-2.04)

    def test_embedded_json_is_used_when_tables_fail(self, source) -> None:
        parsed = source.parse_snapshot(JSON_ONLY_PAGE)
        assert parsed.report.strategy == "embedded_json"
        assert {r["symbol"] for r in parsed.rows} == {"COMI", "TMGH"}

    def test_empty_page_reports_why(self, source) -> None:
        parsed = source.parse_snapshot("<html><body><p>maintenance</p></body></html>")
        assert parsed.report.rows == 0
        assert parsed.report.strategy == "none"
        assert parsed.report.notes


class TestEndpointFallback:
    def test_moves_to_the_next_candidate_on_failure(self, egx_market_watch_html) -> None:
        tx = (FakeTransport()
              .add("StocksData.aspx", TransportError("connection reset", transient=True))
              .add_text("MarketWatch.aspx", egx_market_watch_html))
        source = EGXOfficialSource(tx)

        result = source.market_snapshot()
        assert len(result.value) == 7
        assert result.provenance.locator["endpoint_id"] == "/en/MarketWatch.aspx"

    def test_working_endpoint_becomes_sticky(self, egx_market_watch_html) -> None:
        tx = (FakeTransport()
              .add("StocksData.aspx", TransportError("gone", transient=True))
              .add_text("MarketWatch.aspx", egx_market_watch_html))
        source = EGXOfficialSource(tx)
        source.market_snapshot()
        before = tx.route_count("StocksData.aspx")
        source.market_snapshot()
        # The second call must try the known-good endpoint first.
        assert tx.route_count("StocksData.aspx") == before
        assert tx.urls()[-1].endswith("MarketWatch.aspx")

    def test_page_reached_but_unmappable_is_a_parse_error(self) -> None:
        tx = FakeTransport().add_text("StocksData.aspx", "<html><table><tr><td>x</td></tr></table></html>")
        source = EGXOfficialSource(tx)
        with pytest.raises(SourceParseError):
            source.market_snapshot()

    def test_every_candidate_failing_is_unavailable(self) -> None:
        tx = FakeTransport().add("egx.com.eg", TransportError("dns failure", transient=True))
        source = EGXOfficialSource(tx)
        with pytest.raises(SourceUnavailable) as exc:
            source.market_snapshot()
        assert "no candidate produced rows" in str(exc.value)

    def test_not_parsing_failures_are_visible_in_the_error(self) -> None:
        tx = FakeTransport().add("egx.com.eg", TransportError("dns failure", transient=True))
        source = EGXOfficialSource(tx)
        with pytest.raises(SourceUnavailable) as exc:
            source.market_snapshot()
        assert "StocksData.aspx" in str(exc.value)


class TestSnapshotSourced:
    def test_partial_fields_are_flagged(self) -> None:
        page = """
        <table><tr><th>Symbol</th><th>Company Name</th><th>Close</th></tr>
        <tr><td>COMI</td><td>CIB</td><td>86.20</td></tr></table>
        """
        source = EGXOfficialSource(FakeTransport().add_text("StocksData.aspx", page))
        result = source.market_snapshot()
        assert result.value
        assert result.has(QualityFlag.PARTIAL)

    def test_parse_report_is_recorded_in_provenance(self, egx_market_watch_html) -> None:
        source = EGXOfficialSource(FakeTransport().add_text("StocksData.aspx", egx_market_watch_html))
        result = source.market_snapshot()
        parse = result.provenance.locator["parse"]
        assert parse["strategy"] == "html_table"
        assert parse["rows"] == 7


class TestQuoteAndIndexes:
    def test_quote_is_resolved_from_the_snapshot(self, egx_market_watch_html) -> None:
        source = EGXOfficialSource(FakeTransport().add_text("StocksData.aspx", egx_market_watch_html))
        quote = source.quote("COMI")
        assert quote.value["close"] == pytest.approx(86.2)
        assert quote.provenance.locator["resolved_from"] == "market_snapshot"

    def test_quote_accepts_a_prebuilt_snapshot(self, egx_market_watch_html) -> None:
        source = EGXOfficialSource(FakeTransport())
        rows = source.parse_snapshot(egx_market_watch_html).rows
        assert source.quote("TMGH", snapshot=rows).value["close"] == pytest.approx(38.95)

    def test_unknown_symbol_is_a_data_error(self, egx_market_watch_html) -> None:
        source = EGXOfficialSource(FakeTransport().add_text("StocksData.aspx", egx_market_watch_html))
        with pytest.raises(SourceDataError):
            source.quote("NOSUCH")

    def test_index_quote_is_extracted_from_an_index_table(self) -> None:
        source = EGXOfficialSource(FakeTransport().add_text("IndicesData.aspx", INDEX_PAGE))
        result = source.index_quote("EGX30")
        assert result.value["index"] == "EGX30"
        assert result.value["close"] == pytest.approx(32456.78)

    def test_missing_index_raises_parse_error(self) -> None:
        source = EGXOfficialSource(FakeTransport().add_text("IndicesData.aspx", INDEX_PAGE))
        with pytest.raises(SourceParseError):
            source.index_quote("EGX100")


class TestCapabilityHonesty:
    def test_does_not_claim_daily_history(self, source) -> None:
        # EGX's history page is an ASP.NET postback grid; claiming DAILY_OHLCV
        # here would let the router pick a source that cannot deliver.
        assert not source.supports(Capability.DAILY_OHLCV)
        assert not hasattr(source, "daily_ohlcv")

    def test_declares_the_market_wide_capabilities(self, source) -> None:
        assert source.supports(Capability.MARKET_SNAPSHOT)
        assert source.supports(Capability.SYMBOL_MASTER)
        assert source.supports(Capability.INDEX_QUOTE)
        assert source.descriptor.official is True


class TestColumnMapping:
    def test_arabic_and_english_synonyms_resolve_to_the_same_field(self) -> None:
        en = EGXOfficialSource._map_columns(["Symbol", "Close", "Volume"])
        ar = EGXOfficialSource._map_columns(["الرمز", "آخر سعر", "الكمية المتداولة"])
        assert en == ar == {0: "symbol", 1: "close", 2: "volume"}

    def test_price_group_does_not_hijack_the_open_column(self) -> None:
        mapping = EGXOfficialSource._map_columns(
            ["Symbol", "Prices (EGP) Open", "High", "Low", "Close"]
        )
        assert mapping[1] == "open"
        assert mapping[4] == "close"

    def test_change_percent_beats_change_absolute(self) -> None:
        mapping = EGXOfficialSource._map_columns(["Symbol", "Change", "Change %"])
        assert mapping[1] == "change"
        assert mapping[2] == "change_pct"

    def test_duplicate_field_is_not_assigned_twice(self) -> None:
        mapping = EGXOfficialSource._map_columns(["Symbol", "Close", "Last Price"])
        assert list(mapping.values()).count("close") == 1
