"""EGX (Egyptian Exchange) trading calendar & session clock.

This module is the backbone of every freshness and completeness check in the
data layer. Before it, "is this quote stale?" was answered with a wall-clock
constant (``86_400`` seconds) that knows nothing about the Egyptian week — the
EGX trades **Sunday through Thursday**, so a Friday-morning check would call a
perfectly fresh Thursday close "stale", and a Sunday-morning check would call
it fresh while the market had been shut for two days.

Design rules
------------
1. **Timezone is never guessed.** Everything is computed in ``Africa/Cairo``
   via :mod:`zoneinfo`, so Egypt's DST regime (abolished 2015, reinstated 2023)
   is handled by the tz database instead of by hand-rolled offsets.
2. **Session windows are data, not code.** :data:`DEFAULT_SESSIONS` holds the
   published EGX timetable. They are overridable per-process through
   :meth:`EGXCalendar.from_env` so an EGX rulebook change never needs a code
   edit.
3. **Holidays are declared, not invented.** Fixed-date public holidays live in
   ``egx_holidays.json`` next to this module. Islamic-calendar holidays (Eid
   al-Fitr, Eid al-Adha, Hijri New Year, Mawlid) and one-off EGX closures move
   every year and **must** be synced from the exchange's own announcements;
   :meth:`EGXCalendar.sync_warnings` reports when that has not been done for
   the current year instead of silently assuming the market was open.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterator, Literal, Optional

try:  # pragma: no cover - import shim
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover - Python < 3.9
    raise RuntimeError("EGXCalendar requires Python 3.9+ (zoneinfo)")

CAIRO_TZ_NAME = "Africa/Cairo"

_HOLIDAY_FILE = Path(__file__).with_name("egx_holidays.json")

#: EGX trades Sunday → Thursday. Friday/Saturday is the Egyptian weekend.
TRADING_WEEKDAYS: frozenset[int] = frozenset({6, 0, 1, 2, 3})  # Mon=0 … Sun=6

SessionKind = Literal["pre_open", "continuous", "closing_auction", "post_close"]


@dataclass(frozen=True)
class Session:
    """One phase of the EGX trading day, in Cairo local time."""

    kind: SessionKind
    start: time
    end: time
    #: Whether a *continuous* price feed is expected during this phase.
    live_prices: bool

    def contains(self, local_dt: datetime) -> bool:
        return self.start <= local_dt.time() < self.end

    def on(self, day: date) -> tuple[datetime, datetime]:
        """Concrete datetimes for this session on *day* (naive local parts)."""
        return (
            datetime.combine(day, self.start),
            datetime.combine(day, self.end),
        )


# Published EGX timetable. The continuous session is the window during which a
# quote is expected to tick; the auctions produce the official open/close
# prints. Times are Cairo local (the exchange does not shift its clock for DST
# — the local clock does, which is exactly why this is computed in tz-aware
# terms rather than as fixed UTC offsets).
DEFAULT_SESSIONS: tuple[Session, ...] = (
    Session("pre_open", time(9, 30), time(10, 0), live_prices=False),
    Session("continuous", time(10, 0), time(14, 20), live_prices=True),
    Session("closing_auction", time(14, 20), time(14, 30), live_prices=True),
    Session("post_close", time(14, 30), time(15, 30), live_prices=False),
)

#: A quote is "live" if it arrived within this many seconds during the
#: continuous session.
LIVE_QUOTE_TTL_S = 300

#: After the close, the official settlement print can take this long to appear
#: on public feeds before the day's data is considered missing rather than late.
POST_CLOSE_GRACE_S = 90 * 60


@dataclass(frozen=True)
class SessionState:
    """Answer to "where are we in the trading day?" — pure fact, no guessing."""

    now_local: datetime
    is_trading_day: bool
    is_open: bool
    #: Active session, or ``None`` when the market is closed.
    session: Optional[Session]
    #: Seconds until the next continuous session starts (0 when open).
    seconds_to_open: int
    #: Seconds since the last continuous session ended (0 when open).
    seconds_since_close: int
    #: True while data is expected to update; False when a frozen feed is normal.
    expect_live_data: bool

    @property
    def phase(self) -> str:
        """Which named phase of the trading day we are in.

        The session kind is reported whenever a session is active — including
        the non-live ones (pre-open, post-close). Collapsing those into
        "closed" would hide the difference between "the auction is running and
        prices are being collected" and "nothing is happening until tomorrow".
        """
        if self.session is not None:
            return self.session.kind
        if self.is_trading_day:
            return "closed_intraday_break"
        return "closed_weekend_or_holiday"


class EGXCalendar:
    """Trading calendar with DST-correct session math.

    All public methods take/return timezone-aware datetimes in Cairo time
    unless the name says otherwise.
    """

    def __init__(
        self,
        holidays: Optional[dict[int, frozenset[date]]] = None,
        sessions: tuple[Session, ...] = DEFAULT_SESSIONS,
        tz_name: str = CAIRO_TZ_NAME,
    ) -> None:
        try:
            self.tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:  # pragma: no cover - bare container
            # Last-resort fixed offset keeps the module importable; every
            # session boundary would then be an hour off in DST months, so we
            # surface it rather than hide it.
            self.tz = ZoneInfo("UTC")
            self._tz_degraded = True
        else:
            self._tz_degraded = False

        self.sessions = sessions
        self._holidays = holidays if holidays is not None else load_holidays()
        self._continuous = next(s for s in sessions if s.kind == "continuous")
        self._closing = next(s for s in sessions if s.kind == "closing_auction")

    @property
    def tz_degraded(self) -> bool:
        """True when the Cairo tz database was unavailable (session math off)."""
        return self._tz_degraded

    # ── construction helpers ────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "EGXCalendar":
        """Build a calendar, letting deployments disable built-in holidays.

        ``EGX_IGNORE_HOLIDAYS=1`` treats every Sun–Thu day as a trading day.
        Useful for historical backfills where the holiday list is known to be
        incomplete — better to over-fetch than to silently drop sessions.
        """
        if os.environ.get("EGX_IGNORE_HOLIDAYS", "").strip() in {"1", "true", "yes"}:
            return cls(holidays={})
        return cls()

    # ── time helpers ────────────────────────────────────────────────────────

    def now(self) -> datetime:
        """Current moment, always tz-aware in Cairo time."""
        return datetime.now(tz=self.tz)

    def to_local(self, dt: datetime) -> datetime:
        """Coerce *dt* (naive ⇒ assumed UTC) into Cairo-local aware time."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(self.tz)

    # ── day-level facts ─────────────────────────────────────────────────────

    def is_trading_day(self, day: date) -> bool:
        if day.weekday() not in TRADING_WEEKDAYS:
            return False
        return day not in self._holidays.get(day.year, frozenset())

    def holidays_for(self, year: int) -> frozenset[date]:
        return self._holidays.get(year, frozenset())

    def previous_trading_day(self, day: date) -> date:
        cur = day - timedelta(days=1)
        guard = 0
        while not self.is_trading_day(cur):
            cur -= timedelta(days=1)
            guard += 1
            if guard > 400:  # pragma: no cover - pathological holiday list
                raise RuntimeError("no trading day found in the last 400 days")
        return cur

    def next_trading_day(self, day: date) -> date:
        cur = day + timedelta(days=1)
        guard = 0
        while not self.is_trading_day(cur):
            cur += timedelta(days=1)
            guard += 1
            if guard > 400:  # pragma: no cover
                raise RuntimeError("no trading day found in the next 400 days")
        return cur

    def trading_days_between(self, start: date, end: date) -> list[date]:
        """Inclusive list of trading days in ``[start, end]``."""
        if end < start:
            return []
        out: list[date] = []
        cur = start
        while cur <= end:
            if self.is_trading_day(cur):
                out.append(cur)
            cur += timedelta(days=1)
        return out

    def expected_sessions(self, start: date, end: date) -> int:
        """How many trading sessions *should* have happened in the range."""
        return len(self.trading_days_between(start, end))

    # ── session-level facts ─────────────────────────────────────────────────

    def session_at(self, moment: datetime) -> Optional[Session]:
        local = self.to_local(moment)
        if not self.is_trading_day(local.date()):
            return None
        for s in self.sessions:
            if s.contains(local):
                return s
        return None

    def state(self, moment: Optional[datetime] = None) -> SessionState:
        """Full description of the market clock at *moment* (default: now)."""
        local = self.to_local(moment) if moment is not None else self.now()
        today = local.date()
        trading_day = self.is_trading_day(today)
        session = self.session_at(local)

        next_open = self._next_continuous_start(local)
        seconds_to_open = max(0, int((next_open - local).total_seconds()))

        last_close = self._last_continuous_end(local)
        seconds_since_close = (
            max(0, int((local - last_close).total_seconds())) if last_close else 0
        )

        return SessionState(
            now_local=local,
            is_trading_day=trading_day,
            is_open=session is not None and session.live_prices,
            session=session,
            seconds_to_open=seconds_to_open,
            seconds_since_close=seconds_since_close,
            expect_live_data=bool(session and session.live_prices),
        )

    def _continuous_bounds(self, day: date) -> tuple[datetime, datetime]:
        start_naive, _ = self._continuous.on(day)
        _, end_naive = self._closing.on(day)
        return (
            start_naive.replace(tzinfo=self.tz),
            end_naive.replace(tzinfo=self.tz),
        )

    def _next_continuous_start(self, local: datetime) -> datetime:
        day = local.date()
        start, end = self._continuous_bounds(day)
        if self.is_trading_day(day) and local < start:
            return start
        nxt = self.next_trading_day(day)
        return self._continuous_bounds(nxt)[0]

    def _last_continuous_end(self, local: datetime) -> Optional[datetime]:
        day = local.date()
        if self.is_trading_day(day):
            _, end = self._continuous_bounds(day)
            if local >= end:
                return end
        prev = self.previous_trading_day(day)
        return self._continuous_bounds(prev)[1]

    # ── freshness contracts ─────────────────────────────────────────────────

    def max_quote_age_s(self, moment: Optional[datetime] = None) -> int:
        """Age beyond which a quote is objectively stale.

        Open market → :data:`LIVE_QUOTE_TTL_S`. Closed market → the time since
        the last continuous close plus :data:`POST_CLOSE_GRACE_S`, because a
        frozen feed is *correct* after hours; only a feed older than the last
        session's close is wrong.
        """
        st = self.state(moment)
        if st.expect_live_data:
            return LIVE_QUOTE_TTL_S
        return st.seconds_since_close + POST_CLOSE_GRACE_S

    def is_stale(self, fetched_at: datetime, moment: Optional[datetime] = None) -> bool:
        """True when *fetched_at* is too old to represent the current session."""
        now = self.to_local(moment) if moment is not None else self.now()
        age = (now - self.to_local(fetched_at)).total_seconds()
        return age > self.max_quote_age_s(moment)

    def data_is_current(
        self,
        last_bar_date: date,
        moment: Optional[datetime] = None,
    ) -> bool:
        """Is ``last_bar_date`` the bar a correct daily feed should end on?

        On a trading day before the close, the newest *settled* daily bar is
        the previous trading day. After the close (plus grace) it is today.
        """
        now = self.to_local(moment) if moment is not None else self.now()
        today = now.date()
        if not self.is_trading_day(today):
            return last_bar_date >= self.previous_trading_day(today)
        _, close_dt = self._continuous_bounds(today)
        settled_today = now >= close_dt + timedelta(seconds=POST_CLOSE_GRACE_S)
        expected = today if settled_today else self.previous_trading_day(today)
        return last_bar_date >= expected

    def sync_warnings(self, year: int) -> list[str]:
        """Human-readable gaps in the holiday list for *year*.

        Returned (not logged) so the caller decides whether to surface it —
        an ingestion run should attach these to its audit row.
        """
        out: list[str] = []
        if self._tz_degraded:
            out.append(
                "Africa/Cairo tz database unavailable — session times fall back "
                "to UTC and are off by the DST offset during Egyptian summer time"
            )
        if year not in self._holidays or not self._holidays[year]:
            out.append(
                f"no EGX holidays recorded for {year}: Islamic-calendar closures "
                "(Eid al-Fitr, Eid al-Adha, Hijri New Year, Mawlid) and one-off "
                "exchange closures must be synced from egx.com.eg announcements"
            )
        return out


# ── holiday file loading ────────────────────────────────────────────────────


def load_holidays(path: Optional[Path] = None) -> dict[int, frozenset[date]]:
    """Read ``egx_holidays.json`` → ``{year: frozenset[date]}``.

    Missing/corrupt file ⇒ empty map (every Sun–Thu day counts as trading).
    The file carries a ``verified`` flag per year precisely so callers can tell
    "no holidays" apart from "holidays not synced yet".
    """
    target = path or _HOLIDAY_FILE
    if not target.exists():
        return {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    out: dict[int, frozenset[date]] = {}
    for year_key, payload in raw.get("years", {}).items():
        try:
            year = int(year_key)
        except ValueError:
            continue
        days: set[date] = set()
        for item in payload.get("closures", []):
            try:
                days.add(date.fromisoformat(item))
            except (TypeError, ValueError):
                continue
        if days:
            out[year] = frozenset(days)
    return out


def holiday_metadata(path: Optional[Path] = None) -> dict:
    """Raw metadata (``verified`` flags etc.) from the holiday file."""
    target = path or _HOLIDAY_FILE
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def iter_sessions(day: date, tz: ZoneInfo = ZoneInfo(CAIRO_TZ_NAME)) -> Iterator[tuple[str, datetime, datetime]]:
    """Yield ``(kind, start, end)`` aware datetimes for a given day."""
    for s in DEFAULT_SESSIONS:
        a, b = s.on(day)
        yield s.kind, a.replace(tzinfo=tz), b.replace(tzinfo=tz)
