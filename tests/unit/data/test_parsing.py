"""Value parsing: the difference between a wrong number and no number."""

from __future__ import annotations

import pytest

from tradingview_mcp.core.data.parsing import (
    clean_text,
    header_key,
    looks_like_symbol,
    normalize_arabic,
    normalize_header,
    parse_int,
    parse_number,
    parse_percent,
    parse_ratio_percent,
    to_ascii_digits,
)


class TestDigitsAndArabic:
    def test_arabic_indic_digits_fold(self) -> None:
        assert to_ascii_digits("١٩٩٥") == "1995"
        assert to_ascii_digits("۳۲۵") == "325"

    def test_arabic_orthography_folds(self) -> None:
        assert normalize_arabic("أعلى") == normalize_arabic("اعلى")
        assert normalize_arabic("الشركة") == normalize_arabic("الشركه")
        assert normalize_arabic("مصـــر") == normalize_arabic("مصر")

    def test_headers_fold_case_and_punctuation(self) -> None:
        assert normalize_header("Change %") == "change %"
        assert header_key("Change %") == "change_pct"
        assert header_key("Volume (Shares)") == "volume_(shares)"


class TestNumbers:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1234", 1234.0),
            ("1,234,567", 1234567.0),
            ("84.10", 84.1),
            ("0.85", 0.85),
            ("6.70", 6.7),
            ("١٢٣٤", 1234.0),
            ("٥٬٤٣٢٬١٠٠", 5432100.0),
            ("1٬234٬567", 1234567.0),
            ("-1.14", -1.14),
            ("(12.5)", -12.5),
            ("1.2M", 1_200_000.0),
            ("3K", 3000.0),
            ("1,205.00", 1205.0),
            ("18,976,217", 18976217.0),
        ],
    )
    def test_reads_market_page_formats(self, raw: str, expected: float) -> None:
        assert parse_number(raw) == pytest.approx(expected)

    def test_european_style_both_separators(self) -> None:
        assert parse_number("1.234,56") == pytest.approx(1234.56)

    def test_ambiguous_thousands_case_is_grouped(self) -> None:
        # "1.234" on an Egyptian price page is 1234, not 1.234 pounds.
        assert parse_number("1.234") == pytest.approx(1234.0)

    def test_arabic_decimal_separator(self) -> None:
        assert parse_number("١٢٫٥") == pytest.approx(12.5)

    @pytest.mark.parametrize("raw", ["", "  ", "n/a", "N/A", "-", "—", None, "abc", True, "12.34.56.78"])
    def test_refuses_to_guess(self, raw: object) -> None:
        assert parse_number(raw) is None

    def test_numeric_passthrough(self) -> None:
        assert parse_number(42) == 42.0
        assert parse_number(1.5) == 1.5

    def test_infinity_and_nan_rejected(self) -> None:
        assert parse_number(float("nan")) is None
        assert parse_number(float("inf")) is None

    def test_parse_int_rounds(self) -> None:
        assert parse_int("1,234,567") == 1234567
        assert parse_int("n/a") is None


class TestPercents:
    def test_percent_keeps_scale(self) -> None:
        assert parse_percent("2.50%") == pytest.approx(2.5)
        assert parse_percent("+0.85%") == pytest.approx(0.85)

    def test_arabic_percent_sign(self) -> None:
        assert parse_percent("٣.٠٢٪") == pytest.approx(3.02)

    def test_parenthesised_negative_percent(self) -> None:
        assert parse_percent("(2.04%)") == pytest.approx(-2.04)

    def test_ratio_form(self) -> None:
        assert parse_ratio_percent("12.5%") == pytest.approx(0.125)


class TestSymbols:
    @pytest.mark.parametrize("raw", ["COMI", "TMGH", "FWRY", "ABUK", "CIEB", "EGX30", "AALR"])
    def test_accepts_egx_tickers(self, raw: str) -> None:
        assert looks_like_symbol(raw)

    @pytest.mark.parametrize(
        "raw",
        ["", "Total traded value (EGP)", "1,234", "comi bank", "TOOLONGSYMBOL", None],
    )
    def test_rejects_non_tickers(self, raw: object) -> None:
        assert not looks_like_symbol(raw)

    def test_clean_text_collapses_whitespace(self) -> None:
        assert clean_text("  Fawry   Banking &  Payment ") == "Fawry Banking & Payment"
        assert clean_text(None) == ""
