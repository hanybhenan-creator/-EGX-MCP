"""Wyckoff and Volume-Spread-Analysis accumulation events on OHLCV bars.

Four event definitions, each evaluated once per session, causally: the flag for
session ``i`` reads only sessions ``0..i``. No parameter here was tuned on EGX
results. They are written down before measurement, and the calibration engine
decides whether any of them predicts anything. The practitioner literature that
names these patterns (Wyckoff; Tom Williams on VSA) is not evidence that they
work, so nothing in this module treats them as established.

Events
------
``selling_climax``  a down session that makes a new ``LOOKBACK``-session low on
                    at least ``SC_VOLUME_MULT`` × average volume, and closes in
                    the upper half of its range ("stopping volume").
``spring``          a session that undercuts the prior ``LOOKBACK``-session low
                    and closes back above it, on below-average volume (supply
                    tested and absent).
``no_supply``       a narrow-spread down session on volume below
                    ``NS_VOLUME_MULT`` × average, closing in the upper half
                    (VSA "no supply").
``secondary_test``  a retest of the most recent selling climax within
                    ``ST_WINDOW`` sessions: low within ``ST_TOLERANCE`` above
                    the climax low, on volume below ``ST_VOLUME_MULT`` × the
                    climax's volume, and the climax low never undercut.

Averages use only the sessions before ``i`` (no look-ahead into the bar itself).
"""

from __future__ import annotations

from statistics import fmean
from typing import Any, Mapping, Optional, Sequence

Bar = Mapping[str, Any]

LOOKBACK = 20
SC_VOLUME_MULT = 2.0
SC_CLOSE_LOCATION = 0.5
NS_VOLUME_MULT = 0.7
ST_WINDOW = 30
ST_VOLUME_MULT = 0.8
ST_TOLERANCE = 0.03

EVENTS = ("selling_climax", "spring", "no_supply", "secondary_test")


def _num(bar: Bar, key: str) -> Optional[float]:
    value = bar.get(key)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _ohlcv(bar: Bar) -> Optional[tuple[float, float, float, float, float]]:
    o, h, l, c = (_num(bar, k) for k in ("open", "high", "low", "close"))
    v = _num(bar, "volume")
    if None in (o, h, l, c, v) or h < l or h <= 0:
        return None
    return o, h, l, c, v  # type: ignore[return-value]


def _close_location(h: float, l: float, c: float) -> Optional[float]:
    return (c - l) / (h - l) if h > l else None


def _prior_stats(rows: Sequence[Optional[tuple[float, float, float, float, float]]], i: int):
    """Support, average volume and average spread over the LOOKBACK sessions before i."""
    window = rows[i - LOOKBACK : i]
    if any(r is None for r in window):
        return None
    support = min(r[2] for r in window)  # type: ignore[index]
    avg_volume = fmean(r[4] for r in window)  # type: ignore[index]
    avg_spread = fmean(r[1] - r[2] for r in window)  # type: ignore[index]
    return support, avg_volume, avg_spread


def _flags(bars: Sequence[Bar], check) -> list[Optional[bool]]:
    rows = [_ohlcv(b) for b in bars]
    out: list[Optional[bool]] = []
    for i in range(len(rows)):
        if i < LOOKBACK or rows[i] is None:
            out.append(None)
            continue
        stats = _prior_stats(rows, i)
        if stats is None:
            out.append(None)
            continue
        out.append(bool(check(rows[i], stats)))
    return out


def _is_selling_climax(row, stats) -> bool:
    o, h, l, c, v = row
    support, avg_volume, _ = stats
    loc = _close_location(h, l, c)
    return (
        l < support
        and c < o
        and avg_volume > 0
        and v >= SC_VOLUME_MULT * avg_volume
        and loc is not None
        and loc >= SC_CLOSE_LOCATION
    )


def selling_climax_flags(bars: Sequence[Bar]) -> list[Optional[bool]]:
    return _flags(bars, _is_selling_climax)


def spring_flags(bars: Sequence[Bar]) -> list[Optional[bool]]:
    def check(row, stats) -> bool:
        o, h, l, c, v = row
        support, avg_volume, _ = stats
        return l < support and c > support and avg_volume > 0 and v < avg_volume

    return _flags(bars, check)


def no_supply_flags(bars: Sequence[Bar]) -> list[Optional[bool]]:
    def check(row, stats) -> bool:
        o, h, l, c, v = row
        _, avg_volume, avg_spread = stats
        loc = _close_location(h, l, c)
        return (
            c < o
            and avg_volume > 0
            and v < NS_VOLUME_MULT * avg_volume
            and (h - l) < avg_spread
            and loc is not None
            and loc >= SC_CLOSE_LOCATION
        )

    return _flags(bars, check)


def secondary_test_flags(bars: Sequence[Bar]) -> list[Optional[bool]]:
    """Causal state machine: the most recent selling climax, while still valid."""
    rows = [_ohlcv(b) for b in bars]
    out: list[Optional[bool]] = []
    sc_index: Optional[int] = None
    sc_low = sc_volume = 0.0
    for i in range(len(rows)):
        if i < LOOKBACK or rows[i] is None:
            out.append(None)
            continue
        stats = _prior_stats(rows, i)
        if stats is None:
            out.append(None)
            continue
        row = rows[i]
        o, h, l, c, v = row
        fired = False
        if sc_index is not None:
            if i - sc_index > ST_WINDOW or l < sc_low:
                sc_index = None  # expired, or the climax low was undercut
            else:
                fired = (
                    not _is_selling_climax(row, stats)
                    and sc_low <= l <= sc_low * (1.0 + ST_TOLERANCE)
                    and v < ST_VOLUME_MULT * sc_volume
                )
        out.append(fired)
        if _is_selling_climax(row, stats):
            sc_index, sc_low, sc_volume = i, l, v
    return out


FLAG_BUILDERS = {
    "selling_climax": selling_climax_flags,
    "spring": spring_flags,
    "no_supply": no_supply_flags,
    "secondary_test": secondary_test_flags,
}

DEFINITIONS = {
    "selling_climax": (
        f"new {LOOKBACK}-session low on ≥{SC_VOLUME_MULT:g}× average volume, down session "
        f"closing in the upper half of its range"
    ),
    "spring": (
        f"undercuts the prior {LOOKBACK}-session low and closes back above it on below-average volume"
    ),
    "no_supply": (
        f"narrow-spread down session on volume <{NS_VOLUME_MULT:g}× average, closing in the upper half"
    ),
    "secondary_test": (
        f"retests a selling climax within {ST_WINDOW} sessions, low within {ST_TOLERANCE:.0%} above "
        f"its low, volume <{ST_VOLUME_MULT:g}× the climax, climax low not undercut"
    ),
}
