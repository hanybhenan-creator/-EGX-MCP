"""Quality gates: hard validity, plausibility flags, and cross-source agreement."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from tradingview_mcp.core.data.calendar import POST_CLOSE_GRACE_S, EGXCalendar
from tradingview_mcp.core.data.provenance import QualityFlag
from tradingview_mcp.core.data.quality import (
    SourceObservation,
    check_freshness,
    check_price_jump,
    check_volume_anomaly,
    triangulate,
    validate_ohlcv_bars,
    validate_quote,
)


class TestQuoteValidation:
    def test_valid_quote_passes(self) -> None:
        verdicts = validate_quote(price=86.2, volume=1_234_567, previous_close=84.1)
        assert [v.flag for v in verdicts] == [QualityFlag.VALID]

    def test_zero_price_is_invalid(self) -> None:
        verdicts = validate_quote(price=0)
        assert any(v.flag is QualityFlag.INVALID for v in verdicts)

    def test_missing_price_is_invalid(self) -> None:
        verdicts = validate_quote(price=None)
        assert any(v.flag is QualityFlag.INVALID for v in verdicts)

    def test_negative_volume_is_invalid(self) -> None:
        verdicts = validate_quote(price=10, volume=-5)
        assert any(v.flag is QualityFlag.INVALID for v in verdicts)

    def test_crossed_book_is_flagged_not_rejected(self) -> None:
        verdicts = validate_quote(price=10, bid=11, ask=10.5)
        assert any(v.flag is QualityFlag.OUTLIER for v in verdicts)
        assert not any(v.flag is QualityFlag.INVALID for v in verdicts)

    def test_implausible_move_is_an_outlier(self) -> None:
        verdicts = validate_quote(price=200, previous_close=100)
        assert any(v.flag is QualityFlag.OUTLIER for v in verdicts)


class TestOHLCVValidation:
    def _bar(self, **overrides):
        bar = {"date": "2026-03-02", "open": 84.1, "high": 86.75, "low": 83.95, "close": 86.2, "volume": 1234}
        bar.update(overrides)
        return bar

    def test_clean_series_passes(self) -> None:
        verdicts = validate_ohlcv_bars([self._bar(), self._bar(date="2026-03-03")])
        assert [v.flag for v in verdicts] == [QualityFlag.VALID]

    def test_high_below_low_is_invalid(self) -> None:
        verdicts = validate_ohlcv_bars([self._bar(high=80, low=90, open=85, close=85)])
        assert any(v.flag is QualityFlag.INVALID and "high < low" in v.note for v in verdicts)

    def test_close_above_high_is_invalid(self) -> None:
        verdicts = validate_ohlcv_bars([self._bar(high=85, close=90)])
        assert any("max(open, close)" in v.note for v in verdicts)

    def test_negative_price_is_invalid(self) -> None:
        verdicts = validate_ohlcv_bars([self._bar(low=-1)])
        assert any("negative" in v.note for v in verdicts)

    def test_non_finite_is_invalid(self) -> None:
        verdicts = validate_ohlcv_bars([self._bar(close=None)])
        assert any("non-finite" in v.note for v in verdicts)

    def test_mostly_zero_volume_is_partial(self) -> None:
        bars = [self._bar(volume=0) for _ in range(4)] + [self._bar(volume=100)]
        verdicts = validate_ohlcv_bars(bars)
        assert any(v.flag is QualityFlag.PARTIAL for v in verdicts)


class TestPlausibility:
    def test_price_jump_flagged(self) -> None:
        closes = [10.0, 10.2, 10.1, 18.0]
        verdicts = check_price_jump(closes)
        assert any(v.flag is QualityFlag.OUTLIER for v in verdicts)

    def test_normal_volatility_not_flagged(self) -> None:
        closes = [10.0, 10.5, 10.1, 10.9, 10.4]
        assert [v.flag for v in check_price_jump(closes)] == [QualityFlag.VALID]

    def test_volume_spike_flagged_against_median(self) -> None:
        volumes = [1000] * 20 + [50_000]
        verdicts = check_volume_anomaly(volumes, multiple=5)
        assert any(v.flag is QualityFlag.OUTLIER for v in verdicts)

    def test_volume_within_range_not_flagged(self) -> None:
        volumes = [1000, 1100, 900, 1050, 1200, 950, 1000]
        assert [v.flag for v in check_volume_anomaly(volumes, multiple=5)] == [QualityFlag.VALID]


class TestTriangulation:
    def _obs(self, source: str, close: float, volume: float | None = None):
        values = {"close": close}
        if volume is not None:
            values["volume"] = volume
        return SourceObservation(source, values, datetime.now(timezone.utc))

    def test_single_source_is_uncrosschecked_not_verified(self) -> None:
        flag, checks, note = triangulate([self._obs("a", 86.2)])
        assert flag is QualityFlag.UNCROSSCHECKED
        assert "single source" in note

    def test_two_sources_within_tolerance_agree(self) -> None:
        flag, checks, note = triangulate([self._obs("egx", 86.20), self._obs("yahoo", 86.35)])
        assert flag is QualityFlag.AGREEMENT
        assert checks[0]["verdict"] == "AGREEMENT"
        assert checks[0]["relative_spread"] < 0.01

    def test_small_gap_diverges(self) -> None:
        flag, _, _ = triangulate([self._obs("egx", 86.20), self._obs("yahoo", 87.30)])
        assert flag is QualityFlag.DIVERGENCE

    def test_large_gap_conflicts(self) -> None:
        flag, _, _ = triangulate([self._obs("egx", 86.20), self._obs("yahoo", 95.00)])
        assert flag is QualityFlag.CONFLICT

    def test_per_field_evidence_is_recorded(self) -> None:
        flag, checks, _ = triangulate(
            [self._obs("egx", 86.2, 1000), self._obs("yahoo", 86.2, 1000)]
        )
        fields = {c["field"] for c in checks}
        assert fields == {"close", "volume"}
        assert all(c["verdict"] == "AGREEMENT" for c in checks)


class TestFreshnessGate:
    def test_stale_quote_is_flagged(self, calendar: EGXCalendar, sourced_factory) -> None:
        moment = datetime.combine(date(2026, 3, 2), datetime.min.time(), tzinfo=calendar.tz).replace(hour=11)
        result = sourced_factory({"price": 10.0}, age_s=0)
        result.provenance = result.provenance.__class__(
            source="test",
            fetched_at=moment - timedelta(hours=4),
        )
        check_freshness(result, kind="quote", calendar=calendar, now=moment)
        assert result.has(QualityFlag.STALE)

    def test_fresh_quote_passes(self, calendar: EGXCalendar, sourced_factory) -> None:
        moment = datetime.combine(date(2026, 3, 2), datetime.min.time(), tzinfo=calendar.tz).replace(hour=11)
        result = sourced_factory({"price": 10.0}, age_s=0)
        check_freshness(result, kind="quote", calendar=calendar, now=moment)
        assert not result.has(QualityFlag.STALE)
        assert result.is_usable

    def test_closed_market_tolerates_an_old_quote(self, calendar: EGXCalendar, sourced_factory) -> None:
        # Friday 10:00: the newest possible print is Thursday's close.
        friday = datetime.combine(date(2026, 3, 6), datetime.min.time(), tzinfo=calendar.tz).replace(hour=10)
        thursday_close = friday - timedelta(hours=20)
        result = sourced_factory({"price": 10.0}, age_s=0)
        result.provenance = result.provenance.__class__(source="test", fetched_at=thursday_close)
        check_freshness(result, kind="quote", calendar=calendar, now=friday)
        assert not result.has(QualityFlag.STALE)

    def test_daily_bar_lag_is_flagged(self, calendar: EGXCalendar, sourced_factory) -> None:
        moment = datetime.combine(date(2026, 3, 2), datetime.min.time(), tzinfo=calendar.tz).replace(hour=11)
        bars = [{"date": "2026-02-26", "close": 10.0}]
        result = sourced_factory(bars, age_s=0)
        check_freshness(result, kind="daily_bar", calendar=calendar, now=moment)
        assert result.has(QualityFlag.STALE)

    def test_daily_bar_on_expected_date_passes(self, calendar: EGXCalendar, sourced_factory) -> None:
        moment = datetime.combine(date(2026, 3, 2), datetime.min.time(), tzinfo=calendar.tz).replace(hour=11)
        # Mid-session Monday: the newest settled bar is Sunday 03-01.
        bars = [{"date": "2026-02-26", "close": 10.0}, {"date": "2026-03-01", "close": 10.5}]
        result = sourced_factory(bars, age_s=0)
        check_freshness(result, kind="daily_bar", calendar=calendar, now=moment)
        assert not result.has(QualityFlag.STALE)

    def test_unknown_kind_raises(self, sourced_factory) -> None:
        with pytest.raises(ValueError):
            check_freshness(sourced_factory({}), kind="nonsense")
