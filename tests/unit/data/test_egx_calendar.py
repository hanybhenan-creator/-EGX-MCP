"""EGX trading calendar: the clock every freshness judgement is measured against."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tradingview_mcp.core.data.calendar import (
    DEFAULT_SESSIONS,
    LIVE_QUOTE_TTL_S,
    POST_CLOSE_GRACE_S,
    TRADING_WEEKDAYS,
    EGXCalendar,
)


# 2026-03-01 is a Sunday; the EGX week runs Sunday → Thursday.
SUNDAY = date(2026, 3, 1)
MONDAY = date(2026, 3, 2)
THURSDAY = date(2026, 3, 5)
FRIDAY = date(2026, 3, 6)
SATURDAY = date(2026, 3, 7)


def at(cal: EGXCalendar, day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, datetime.min.time(), tzinfo=cal.tz).replace(
        hour=hour, minute=minute
    )


class TestTradingWeek:
    def test_sunday_to_thursday_are_trading_days(self, calendar: EGXCalendar) -> None:
        for day in (SUNDAY, MONDAY, THURSDAY):
            assert calendar.is_trading_day(day), day

    def test_friday_and_saturday_are_not(self, calendar: EGXCalendar) -> None:
        assert not calendar.is_trading_day(FRIDAY)
        assert not calendar.is_trading_day(SATURDAY)

    def test_weekday_set_matches_egx_week(self) -> None:
        # Monday=0 … Sunday=6
        assert TRADING_WEEKDAYS == frozenset({6, 0, 1, 2, 3})

    def test_previous_trading_day_skips_the_weekend(self, calendar: EGXCalendar) -> None:
        # Sunday 03-01 → back over Fri/Sat → Thursday 02-26
        assert calendar.previous_trading_day(SUNDAY) == date(2026, 2, 26)
        assert calendar.previous_trading_day(FRIDAY) == THURSDAY
        assert calendar.previous_trading_day(MONDAY) == SUNDAY

    def test_next_trading_day_skips_the_weekend(self, calendar: EGXCalendar) -> None:
        # Thursday 03-05 → forward over Fri/Sat → Sunday 03-08
        assert calendar.next_trading_day(THURSDAY) == date(2026, 3, 8)
        assert calendar.next_trading_day(FRIDAY) == date(2026, 3, 8)

    def test_trading_days_between_counts_only_weekdays(self, calendar: EGXCalendar) -> None:
        days = calendar.trading_days_between(SUNDAY, SATURDAY)
        assert days == [SUNDAY, MONDAY, date(2026, 3, 3), date(2026, 3, 4), THURSDAY]

    def test_holidays_remove_days(self) -> None:
        cal = EGXCalendar(holidays={2026: frozenset({MONDAY})})
        assert not cal.is_trading_day(MONDAY)
        assert cal.previous_trading_day(date(2026, 3, 3)) == SUNDAY


class TestSessionClock:
    def test_open_during_continuous_session(self, calendar: EGXCalendar) -> None:
        state = calendar.state(at(calendar, MONDAY, 11, 30))
        assert state.is_open
        assert state.phase == "continuous"
        assert state.expect_live_data

    def test_auction_phases_are_not_live(self, calendar: EGXCalendar) -> None:
        pre_open = calendar.state(at(calendar, MONDAY, 9, 45))
        assert pre_open.phase == "pre_open"
        assert not pre_open.is_open

        post_close = calendar.state(at(calendar, MONDAY, 15, 0))
        assert post_close.phase == "post_close"
        assert not post_close.expect_live_data

    def test_closing_auction_is_live(self, calendar: EGXCalendar) -> None:
        state = calendar.state(at(calendar, MONDAY, 14, 25))
        assert state.phase == "closing_auction"
        assert state.is_open

    def test_after_post_close_window(self, calendar: EGXCalendar) -> None:
        # 15:30 is the end of the post-close window; after that the day is done.
        state = calendar.state(at(calendar, MONDAY, 16, 0))
        assert state.phase == "closed_intraday_break"
        assert state.is_trading_day
        assert not state.expect_live_data

    def test_weekend_state(self, calendar: EGXCalendar) -> None:
        state = calendar.state(at(calendar, FRIDAY, 11, 0))
        assert not state.is_trading_day
        assert state.phase == "closed_weekend_or_holiday"
        assert state.seconds_to_open > 0

    def test_seconds_to_open_points_at_continuous_session(self, calendar: EGXCalendar) -> None:
        state = calendar.state(at(calendar, MONDAY, 8, 0))
        assert state.seconds_to_open == 2 * 3600  # 08:00 → 10:00 continuous

    def test_seconds_since_close_after_the_bell(self, calendar: EGXCalendar) -> None:
        state = calendar.state(at(calendar, MONDAY, 15, 0))
        assert state.seconds_since_close == 30 * 60


class TestDstHandling:
    """Egypt reinstated DST in 2023; hard-coded UTC offsets silently break."""

    def test_offsets_follow_the_tz_database(self, calendar: EGXCalendar) -> None:
        if calendar.tz_degraded:
            pytest.skip("Africa/Cairo tz data unavailable in this environment")

        winter = calendar.to_local(datetime(2026, 1, 15, 12, tzinfo=timezone.utc))
        summer = calendar.to_local(datetime(2026, 7, 15, 12, tzinfo=timezone.utc))
        assert winter.utcoffset() == timedelta(hours=2)
        assert summer.utcoffset() == timedelta(hours=3)

    def test_session_boundary_is_local_not_utc(self, calendar: EGXCalendar) -> None:
        if calendar.tz_degraded:
            pytest.skip("Africa/Cairo tz data unavailable in this environment")
        summer_noon = calendar.to_local(datetime(2026, 7, 15, 9, 30, tzinfo=timezone.utc))
        assert summer_noon.hour == 12  # 09:30 UTC == 12:30 Cairo (EEST)


class TestFreshnessContract:
    def test_live_ttl_while_open(self, calendar: EGXCalendar) -> None:
        moment = at(calendar, MONDAY, 11, 0)
        assert calendar.max_quote_age_s(moment) == LIVE_QUOTE_TTL_S

    def test_closed_ttl_grows_with_time_since_close(self, calendar: EGXCalendar) -> None:
        moment = at(calendar, MONDAY, 16, 0)
        expected = 90 * 60 + POST_CLOSE_GRACE_S
        assert calendar.max_quote_age_s(moment) == expected

    def test_a_fresh_quote_is_never_stale(self, calendar: EGXCalendar) -> None:
        now = datetime.now(timezone.utc)
        assert not calendar.is_stale(now, now)

    def test_yesterdays_fetch_is_stale_during_a_live_session(self, calendar: EGXCalendar) -> None:
        moment = at(calendar, MONDAY, 11, 0)
        fetched = moment - timedelta(hours=3)
        assert calendar.is_stale(fetched, moment)

    def test_thursday_close_is_current_on_friday(self, calendar: EGXCalendar) -> None:
        moment = at(calendar, FRIDAY, 10, 0)
        assert calendar.data_is_current(THURSDAY, moment)

    def test_wednesday_bar_is_stale_on_friday(self, calendar: EGXCalendar) -> None:
        moment = at(calendar, FRIDAY, 10, 0)
        assert not calendar.data_is_current(date(2026, 3, 4), moment)

    def test_settled_bar_expectation_before_and_after_close(
        self, calendar: EGXCalendar
    ) -> None:
        before = at(calendar, MONDAY, 11, 0)
        after = at(calendar, MONDAY, 17, 0)

        # Mid-session the newest *settled* bar is the previous session (Sunday).
        assert calendar.data_is_current(SUNDAY, before)
        assert not calendar.data_is_current(date(2026, 2, 26), before)  # prior week is old
        assert not calendar.data_is_current(SUNDAY, after)  # today has settled by now
        assert calendar.data_is_current(MONDAY, after)


class TestWarnings:
    def test_unverified_year_is_reported(self) -> None:
        cal = EGXCalendar(holidays={})
        warnings = cal.sync_warnings(2026)
        assert any("2026" in w for w in warnings)

    def test_no_warning_when_holidays_recorded(self) -> None:
        cal = EGXCalendar(holidays={2026: frozenset({MONDAY})})
        assert cal.sync_warnings(2026) == []


def test_sessions_cover_a_full_trading_day() -> None:
    kinds = [s.kind for s in DEFAULT_SESSIONS]
    assert kinds == ["pre_open", "continuous", "closing_auction", "post_close"]
    starts = [s.start for s in DEFAULT_SESSIONS]
    assert starts == sorted(starts)
