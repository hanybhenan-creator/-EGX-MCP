"""HTML table extraction: column geometry must survive merged cells."""

from __future__ import annotations

from tradingview_mcp.core.data.html_tables import (
    extract_json_blobs,
    extract_tables,
    tables_with_headers,
)

MERGED_HEADER_TABLE = """
<table>
  <tr><th rowspan="2">Symbol</th><th rowspan="2">Name</th><th colspan="2">Price</th><th rowspan="2">Volume</th></tr>
  <tr><th>Open</th><th>Close</th></tr>
  <tr><td>COMI</td><td>CIB</td><td>84.10</td><td>86.20</td><td>1,234,567</td></tr>
  <tr><td>TMGH</td><td>Telecom Egypt</td><td>38.50</td><td>38.95</td><td>845,120</td></tr>
</table>
"""


class TestMergedCells:
    def test_two_level_header_merges_positionally(self) -> None:
        table = tables_with_headers(MERGED_HEADER_TABLE)[0]
        assert table.headers == ["Symbol", "Name", "Price Open", "Close", "Volume"]

    def test_data_rows_keep_column_alignment(self) -> None:
        table = tables_with_headers(MERGED_HEADER_TABLE)[0]
        assert table.rows[0] == ["COMI", "CIB", "84.10", "86.20", "1,234,567"]
        assert table.rows[1][3] == "38.95"  # TMGH close, not shifted

    def test_rowspan_does_not_leak_into_later_rows(self) -> None:
        table = tables_with_headers(MERGED_HEADER_TABLE)[0]
        assert len(table.rows[1]) == len(table.rows[0])

    def test_dict_rows_are_keyed_by_header(self) -> None:
        table = tables_with_headers(MERGED_HEADER_TABLE)[0]
        row = table.dict_rows()[0]
        assert row["symbol"] == "COMI"
        assert row["price_open"] == "84.10"
        assert row["volume"] == "1,234,567"


class TestStructure:
    def test_table_without_headers_is_skipped_by_tables_with_headers(self) -> None:
        html = "<table><tr><td>a</td><td>b</td></tr></table>"
        assert tables_with_headers(html) == []
        assert len(extract_tables(html)) == 1

    def test_entities_are_unescaped_once(self) -> None:
        html = "<table><tr><th>Name</th></tr><tr><td>Fawry &amp;amp; Co</td></tr></table>"
        table = tables_with_headers(html)[0]
        assert table.rows[0][0] == "Fawry &amp; Co"

    def test_caption_is_captured(self) -> None:
        html = "<table><caption>EGX 30 constituents</caption><tr><th>A</th></tr><tr><td>1</td></tr></table>"
        assert tables_with_headers(html)[0].caption == "EGX 30 constituents"

    def test_scripts_are_not_read_as_table_content(self) -> None:
        html = """
        <table><tr><th>Symbol</th><th>Close</th></tr>
        <tr><td>COMI</td><td>86.20<script>var x = "<td>99</td>";</script></td></tr></table>
        """
        table = tables_with_headers(html)[0]
        assert table.rows[0] == ["COMI", "86.20"]

    def test_empty_table_yields_nothing(self) -> None:
        assert tables_with_headers("<table></table>") == []


class TestEmbeddedJson:
    def test_var_assignment_is_extracted(self) -> None:
        html = '<script>var data = {"a": [1, 2, 3]};</script>'
        blobs = extract_json_blobs(html)
        assert {"a": [1, 2, 3]} in blobs

    def test_json_with_braces_inside_strings(self) -> None:
        html = '<script>window.__S__ = {"name": "a}b{c", "n": 1};</script>'
        blobs = extract_json_blobs(html)
        assert any(isinstance(b, dict) and b.get("n") == 1 for b in blobs)

    def test_non_json_scripts_are_ignored(self) -> None:
        html = "<script>trackPageView(); console.log('hello');</script>"
        assert extract_json_blobs(html) == []
