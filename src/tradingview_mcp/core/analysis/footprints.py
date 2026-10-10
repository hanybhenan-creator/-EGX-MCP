"""Institutional footprint detection — the "smart money leaves objective
traces" layer, implemented over stored bars and made measurable.

The premise (from the project's design document): institutions accumulating a
position do not want price to move, so the observable trace is **volume rising
while price stays flat**. That claim is a hypothesis about the future, not a
fact about the past, so this module does two separate jobs:

1. **Detection** — describe what the bars objectively contain: volume
   anomalies relative to a rolling average, consolidation zones with real
   support/resistance volumes, absorption days (a dip bought back on volume),
   and breakout preparation (range position + volatility compression + volume
   build). Nothing here needs indicators, moving averages or RSI.

2. **Measurability** — every detector is expressed so that the calibration
   engine can evaluate it on *every* historical session and report what
   actually followed. The composite scores below (``footprint_score``,
   ``preparation_score``, ``accumulation_strength``) are weighted flags, and
   their weights are the design document's. They are named and documented as
   *scores*, never as probabilities: the probability is what
   ``core.calibration`` measures
   (``python scripts/egx_calibrate.py --signals hidden_accumulation``).

Deviations from the reference implementation, each one deliberate:

* ``buy_volume``/``sell_volume`` are derived from the candle direction
  (the reference multiplies volume by 0.7/0.3). That is an approximation, not
  observed order flow — EGX publishes no per-side volume — so the field is
  documented as an estimate and the calibration report is what decides whether
  it carries information.
* A consolidation zone that is still **active** at the end of the window is
  emitted. The reference only emits zones that have already ended, which drops
  the most relevant zone (the one price is in right now).
* ``volume_at_support``/``volume_at_resistance`` are measured from the bars
  that actually traded in those bands, rather than reconstructed as
  ``total_volume × strength / 100``.
* ``consecutive_accumulation_days`` walks the dates and stops at the first
  gap. The reference increments a counter for every anomaly regardless of date,
  so it always returns the anomaly count.
* ``data_quality_score`` and ``source_reliability`` are supplied by the data
  layer's real store statistics; the reference hard-codes ``85.0`` and
  ``"HIGH"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from statistics import fmean, pstdev
from typing import Any, Mapping, Optional, Sequence

Bar = Mapping[str, Any]


# ── vocabularies (the design document's, kept verbatim) ─────────────────────


class SmartMoneySignal(str, Enum):
    """Institutional activity signal. Describes state, not probability."""

    STRONG_ACCUMULATION = "strong_accumulation"
    MILD_ACCUMULATION = "mild_accumulation"
    HIDDEN_ACCUMULATION = "hidden_accumulation"
    BASE_BUILDING = "base_building"
    BREAKOUT_IMMINENT = "breakout_imminent"
    SMART_DISTRIBUTION = "smart_distribution"
    TRAP_SHAKEOUT = "trap_shakeout"
    ACCUMULATION_COMPLETE = "accumulation_complete"
    DISTRIBUTION_PHASE = "distribution_phase"
    NEUTRAL = "neutral"


class AccumulationPhase(str, Enum):
    PHASE_1_SILENT = "phase_1_silent"
    PHASE_2_QUIET = "phase_2_quiet"
    PHASE_3_PREPARATION = "phase_3_preparation"
    PHASE_4_ABSORPTION = "phase_4_absorption"
    PHASE_4B_SHAKEOUT = "phase_4b_shakeout"
    PHASE_5_ACCUMULATION_COMPLETE = "phase_5_complete"
    PHASE_6_INITIAL_BREAKOUT = "phase_6_initial"
    PHASE_7_TREND = "phase_7_trend"
    PHASE_8_DISTRIBUTION_START = "phase_8_dist"
    PHASE_9_DISTRIBUTION_END = "phase_9_dist_end"


class PositionSize(str, Enum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    MASSIVE = "MASSIVE"


# ── helpers ─────────────────────────────────────────────────────────────────


def _f(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _c(bar: Bar, key: str) -> Optional[float]:
    return _f(bar.get(key))


def _day(bar: Bar) -> Optional[str]:
    value = bar.get("date")
    return str(value) if value else None


def _ordinal(day: str) -> Optional[int]:
    """Proleptic Gregorian day number of an ISO date, or None if unparsable."""
    try:
        return date.fromisoformat(day[:10]).toordinal()
    except (TypeError, ValueError):
        return None


def _is_adjacent(previous: Optional[str], current: Optional[str]) -> bool:
    """True when ``current`` is at most four calendar days after ``previous``.

    Four days covers the EGX week (Thursday close → Sunday open is three days)
    plus one holiday. Exact session adjacency is the calendar's job, not this
    module's; the point of the test is to stop counting a March anomaly and a
    June anomaly as "consecutive".
    """
    if not previous or not current:
        return False
    a, b = _ordinal(previous), _ordinal(current)
    if a is None or b is None:
        return False
    return 1 <= (b - a) <= 4


def rolling_mean(series: Sequence[float], window: int, index: int) -> Optional[float]:
    if index + 1 < window:
        return None
    return fmean(series[index + 1 - window : index + 1])


def _score(*weighted_flags: tuple[bool, float]) -> float:
    """Sum the weights of the flags that are true, capped at 100."""
    return min(100.0, sum(weight for flag, weight in weighted_flags if flag))


# ── 1. volume anomalies ─────────────────────────────────────────────────────


@dataclass
class VolumeAnomaly:
    """One session's institutional footprint measurements."""

    date: str
    symbol: str
    total_volume: int
    avg_volume_20d: int
    volume_ratio: float
    buy_volume: int
    sell_volume: int
    buy_sell_ratio: float
    price_open: float
    price_close: float
    price_high: float
    price_low: float
    price_range: float
    price_move_pct: float
    volume_price_divergence: float
    hidden_volume_strength: float
    pressure_index: float
    is_volume_spike: bool
    is_price_flat: bool
    is_hidden_accumulation: bool
    is_buying_pressure: bool
    footprint_score: float  # the guide's weighted flags (0-100), NOT a probability

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def volume_anomaly(bars: Sequence[Bar], index: int, symbol: str = "", *, lookback: int = 20) -> Optional[VolumeAnomaly]:
    """Measure one session against the ``lookback`` sessions before it."""
    if index < lookback or index >= len(bars):
        return None
    bar = bars[index]
    volume = _c(bar, "volume")
    close, open_, high, low = (_c(bar, k) for k in ("close", "open", "high", "low"))
    if volume is None or close is None or open_ is None or high is None or low is None or open_ <= 0:
        return None
    window = [_c(b, "volume") or 0.0 for b in bars[index - lookback : index]]
    avg_volume = fmean(window) if window else 0.0
    volume_ratio = volume / (avg_volume + 1.0)

    price_range = high - low
    price_move_pct = abs((close - open_) / (open_ + 1.0)) * 100.0

    # Direction-implied split, not observed order flow: EGX publishes no
    # per-side volume. Documented approximation (reference: ×0.7 / ×0.3).
    if close > open_:
        buy_vol, sell_vol = volume * 0.7, volume * 0.3
    elif close < open_:
        buy_vol, sell_vol = volume * 0.3, volume * 0.7
    else:
        buy_vol, sell_vol = volume * 0.5, volume * 0.5
    buy_sell_ratio = (buy_vol + 1.0) / (sell_vol + 1.0)

    is_spike = volume_ratio > 2.0
    is_flat = price_move_pct < 2.0
    is_hidden = volume_ratio > 2.0 and price_move_pct < 1.5
    is_buying = buy_sell_ratio > 1.5

    footprint = _score(
        (volume_ratio > 2.0, 30.0),
        (price_move_pct < 1.0, 25.0),
        (buy_sell_ratio > 1.5, 25.0),
        (volume_ratio if price_move_pct < 2.0 else 0.0 > 1.5, 20.0),
    )
    return VolumeAnomaly(
        date=_day(bar) or str(index),
        symbol=symbol,
        total_volume=int(volume),
        avg_volume_20d=int(avg_volume),
        volume_ratio=round(volume_ratio, 4),
        buy_volume=int(buy_vol),
        sell_volume=int(sell_vol),
        buy_sell_ratio=round(buy_sell_ratio, 4),
        price_open=open_,
        price_close=close,
        price_high=high,
        price_low=low,
        price_range=round(price_range, 6),
        price_move_pct=round(price_move_pct, 4),
        volume_price_divergence=round(volume_ratio - price_move_pct / 2.0, 4),
        hidden_volume_strength=round(volume_ratio if price_move_pct < 2.0 else 0.0, 4),
        pressure_index=round(buy_vol - sell_vol, 2),
        is_volume_spike=is_spike,
        is_price_flat=is_flat,
        is_hidden_accumulation=is_hidden,
        is_buying_pressure=is_buying,
        footprint_score=footprint,
    )


def detect_volume_anomalies(bars: Sequence[Bar], *, symbol: str = "", lookback: int = 20) -> list[VolumeAnomaly]:
    out: list[VolumeAnomaly] = []
    for index in range(lookback, len(bars)):
        anomaly = volume_anomaly(bars, index, symbol, lookback=lookback)
        if anomaly and (anomaly.is_volume_spike or anomaly.is_hidden_accumulation or anomaly.is_buying_pressure):
            out.append(anomaly)
    return out


# ── 2. accumulation zones ───────────────────────────────────────────────────


@dataclass
class AccumulationZone:
    symbol: str
    start_date: str
    end_date: str
    zone_low: float
    zone_high: float
    zone_midpoint: float
    zone_width: float
    total_zone_volume: int
    avg_zone_price: float
    volume_at_support: int
    volume_at_resistance: int
    days_in_zone: int
    price_touches_low: int
    price_touches_high: int
    accumulation_strength: float
    support_strength: float
    resistance_quality: float
    phase: AccumulationPhase
    expected_breakout_direction: str
    active: bool
    breakout_score: float

    def as_dict(self) -> dict[str, Any]:
        return {k: (v.value if isinstance(v, Enum) else v) for k, v in self.__dict__.items()}


def _close_volatility(bars: Sequence[Bar], index: int, window: int = 5) -> Optional[float]:
    if index + 1 < window:
        return None
    closes = [_c(b, "close") for b in bars[index + 1 - window : index + 1]]
    values = [c for c in closes if c is not None]
    if len(values) < 2:
        return None
    return pstdev(values)


def detect_accumulation_zones(
    bars: Sequence[Bar], *, symbol: str = "", lookback_days: int = 30
) -> list[AccumulationZone]:
    """Consolidation zones where price stops moving and volume keeps coming.

    Zones are built from runs of low 5-session close volatility (below the 30th
    percentile of the window), then measured: how often each end of the range
    was tested, and how much volume traded in the bottom and top deciles of the
    range *while inside the zone* (measured, not reconstructed from a score).
    """
    window_bars = list(bars[-lookback_days:])
    if len(window_bars) < 8:
        return []
    vols: list[Optional[float]] = [_close_volatility(window_bars, i, 5) for i in range(len(window_bars))]
    defined = [v for v in vols if v is not None]
    if len(defined) < 5:
        return []
    threshold = sorted(defined)[max(0, int(len(defined) * 0.30) - 1)]

    zones: list[AccumulationZone] = []
    start: Optional[int] = None
    low = high = 0.0
    for index in range(len(window_bars)):
        quiet = vols[index] is not None and vols[index] <= threshold
        if quiet and start is None:
            start, low, high = index, _c(window_bars[index], "low") or 0.0, _c(window_bars[index], "high") or 0.0
        if quiet:
            low = min(low, _c(window_bars[index], "low") or low)
            high = max(high, _c(window_bars[index], "high") or high)
            continue
        if start is not None:
            zone = _build_zone(window_bars, start, index - 1, low, high, symbol)
            if zone:
                zones.append(zone)
            start = None
    if start is not None:  # the zone price is in RIGHT NOW — the reference drops it
        zone = _build_zone(window_bars, start, len(window_bars) - 1, low, high, symbol, active=True)
        if zone:
            zones.append(zone)
    return zones


def _build_zone(
    window: Sequence[Bar],
    start: int,
    end: int,
    low: float,
    high: float,
    symbol: str,
    *,
    active: bool = False,
    min_days: int = 5,
) -> Optional[AccumulationZone]:
    days = end - start + 1
    if days < min_days or high <= low:
        return None
    members = list(window[start : end + 1])
    width = high - low
    support_band = low + width * 0.10
    resistance_band = high - width * 0.10

    volume_at_support = 0.0
    volume_at_resistance = 0.0
    touches_low = touches_high = 0
    for bar in members:
        volume = _c(bar, "volume") or 0.0
        close = _c(bar, "close")
        bar_low, bar_high = _c(bar, "low"), _c(bar, "high")
        if bar_low is not None and abs(bar_low - low) <= width * 0.05:
            touches_low += 1
        if bar_high is not None and abs(bar_high - high) <= width * 0.05:
            touches_high += 1
        if close is not None and close <= support_band:
            volume_at_support += volume
        elif close is not None and close >= resistance_band:
            volume_at_resistance += volume

    total_volume = sum(_c(bar, "volume") or 0.0 for bar in members)
    closes = [c for c in (_c(bar, "close") for bar in members) if c is not None]
    avg_price = fmean(closes) if closes else 0.0
    support_strength = (touches_low / days) * 100.0
    resistance_quality = (touches_high / days) * 100.0

    # Two corrections to the reference formula, both documented because they
    # change what the number means:
    #
    #   * It multiplied the zone's share of window volume by 1000, so any zone
    #     holding a tenth of the window's volume already scored the 100 cap and
    #     the term stopped discriminating. Share is a fraction; the scale is 100.
    #   * A range with no volume activity inside it is not accumulation — the
    #     reference's own criteria require "volume is present (institutions
    #     active)". Without this gate a dead flat line scores 30/100 and reads
    #     as a base, which is exactly the kind of number this project is trying
    #     to stop producing. Activity means at least one session traded 1.2x the
    #     zone's median volume.
    zone_volumes = [(_c(bar, "volume") or 0.0) for bar in members]
    mean_volume = fmean(zone_volumes) if zone_volumes else 0.0
    has_activity = (
        bool(zone_volumes)
        and mean_volume > 0
        and max(zone_volumes) >= 1.2 * mean_volume
        and max(zone_volumes) > min(zone_volumes)
    )
    if not has_activity:
        return None
    window_volume = sum(_c(b, "volume") or 0.0 for b in window)
    share = total_volume / (window_volume + 1.0)
    accumulation_strength = min(100.0, share * 100.0 + support_strength * 0.5)

    current = closes[-1] if closes else avg_price
    midpoint = (low + high) / 2.0
    if current > midpoint:
        phase, direction = AccumulationPhase.PHASE_6_INITIAL_BREAKOUT, "up"
    elif current < midpoint:
        phase, direction = AccumulationPhase.PHASE_4_ABSORPTION, "up"
    else:
        phase, direction = AccumulationPhase.PHASE_5_ACCUMULATION_COMPLETE, "unclear"

    # Both ends tested, and more volume traded at the bottom than the top, is
    # what a base being absorbed looks like; the score is a flag sum, and the
    # real breakout frequency is what the calibration engine measures.
    breakout_score = _score(
        (touches_low >= 2, 25.0),
        (touches_high >= 2, 25.0),
        (volume_at_support > volume_at_resistance, 30.0),
        (accumulation_strength >= 60.0, 20.0),
    )
    return AccumulationZone(
        symbol=symbol,
        start_date=_day(members[0]) or str(start),
        end_date=_day(members[-1]) or str(end),
        zone_low=low,
        zone_high=high,
        zone_midpoint=midpoint,
        zone_width=width,
        total_zone_volume=int(total_volume),
        avg_zone_price=avg_price,
        volume_at_support=int(volume_at_support),
        volume_at_resistance=int(volume_at_resistance),
        days_in_zone=days,
        price_touches_low=touches_low,
        price_touches_high=touches_high,
        accumulation_strength=round(accumulation_strength, 2),
        support_strength=round(support_strength, 2),
        resistance_quality=round(resistance_quality, 2),
        phase=phase,
        expected_breakout_direction=direction,
        active=active,
        breakout_score=breakout_score,
    )


# ── 3. absorption ───────────────────────────────────────────────────────────


@dataclass
class AbsorptionResult:
    absorption_detected: bool
    absorption_days: int
    average_absorption_power: float
    institutional_strength: float
    days: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "absorption_detected": self.absorption_detected,
            "absorption_days": self.absorption_days,
            "average_absorption_power": round(self.average_absorption_power, 4),
            "institutional_strength": self.institutional_strength,
            "days": self.days,
        }


def absorption_on_day(bars: Sequence[Bar], index: int, *, lookback: int = 20) -> Optional[float]:
    """Absorption power on one session, or None when the pattern is absent.

    A dip is opened below the previous close, on rising volume, and the session
    closes back above that previous close: somebody bought the fall. Returns
    ``recovery% × volume ratio``.
    """
    if index < 1 or index < lookback:
        return None
    bar, previous = bars[index], bars[index - 1]
    prev_close = _c(previous, "close")
    open_, close = _c(bar, "open"), _c(bar, "close")
    volume = _c(bar, "volume")
    if None in (prev_close, open_, close, volume) or open_ is None or open_ <= 0:
        return None
    if not (open_ < prev_close and close > prev_close):
        return None
    avg_volume = fmean([_c(b, "volume") or 0.0 for b in bars[index - lookback : index]])
    if avg_volume <= 0:
        return None
    volume_ratio = volume / avg_volume
    if volume_ratio <= 1.5:
        return None
    return (close - open_) / open_ * 100.0 * volume_ratio


def detect_absorption_pattern(bars: Sequence[Bar], *, window: int = 20, lookback: int = 20) -> AbsorptionResult:
    """Count absorbed dips in the last ``window`` sessions.

    The volume baseline for each session comes from the sessions *before it in
    the full series*, never from the tail being scanned: measuring a session
    against a one-bar window makes its volume ratio exactly 1.0 and silently
    discards the pattern (the first implementation of this port did that).
    """
    powers: list[float] = []
    days: list[str] = []
    start = max(1, len(bars) - window)
    for index in range(start, len(bars)):
        power = absorption_on_day(bars, index, lookback=lookback)
        if power is not None:
            powers.append(power)
            days.append(_day(bars[index]) or str(index))
    count = len(powers)
    return AbsorptionResult(
        absorption_detected=count >= 3,
        absorption_days=count,
        average_absorption_power=fmean(powers) if powers else 0.0,
        institutional_strength=min(100.0, count * 15.0),
        days=days,
    )


# ── 4. breakout preparation ─────────────────────────────────────────────────


@dataclass
class BreakoutPreparation:
    breakout_ready: bool
    preparation_score: float
    is_at_range_top: bool
    is_volatility_compressed: bool
    is_volume_building: bool
    range_position: float
    range_high: float
    range_low: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def detect_breakout_preparation(bars: Sequence[Bar], *, window: int = 30) -> BreakoutPreparation:
    """Range position + volatility compression + volume build over ``window``."""
    if len(bars) < 10:
        return BreakoutPreparation(False, 0.0, False, False, False, 0.5, 0.0, 0.0)
    tail = list(bars[-window:])
    highs = [_c(b, "high") for b in tail]
    lows = [_c(b, "low") for b in tail]
    closes = [c for c in (_c(b, "close") for b in tail) if c is not None]
    if not closes or not highs or not lows or None in highs or None in lows:
        return BreakoutPreparation(False, 0.0, False, False, False, 0.5, 0.0, 0.0)
    recent_high, recent_low, current = max(highs), min(lows), closes[-1]
    width = recent_high - recent_low
    position = 0.5 if width <= 0 else 1.0 - ((recent_high - current) / width)

    last5 = closes[-5:]
    compression = (pstdev(last5) / pstdev(closes)) if len(closes) >= 10 and pstdev(closes) > 0 else 1.0
    vol_5 = fmean([_c(b, "volume") or 0.0 for b in tail[-5:]])
    vol_all = fmean([_c(b, "volume") or 0.0 for b in tail])

    at_top = position > 0.7
    compressed = compression < 0.7
    building = vol_all > 0 and vol_5 > vol_all * 1.2
    score = _score((at_top, 30.0), (compressed, 30.0), (building, 40.0))
    return BreakoutPreparation(
        breakout_ready=score > 60,
        preparation_score=score,
        is_at_range_top=at_top,
        is_volatility_compressed=compressed,
        is_volume_building=building,
        range_position=round(position, 4),
        range_high=recent_high,
        range_low=recent_low,
    )


# ── 5. shakeout ─────────────────────────────────────────────────────────────


def detect_shakeout(bars: Sequence[Bar], *, window: int = 10) -> bool:
    """A low that was broken and reclaimed within the last ``window`` sessions.

    The reference vocabulary lists a TRAP_SHAKEOUT signal but never produces
    one; the pattern is objective enough to detect: price closes below the
    lowest low of the prior 20 sessions, then closes back above it within
    ``window`` sessions on higher volume.
    """
    if len(bars) < 22:
        return False
    for index in range(max(21, len(bars) - window), len(bars)):
        close = _c(bars[index], "close")
        if close is None:
            continue
        prior_lows = [_c(b, "low") for b in bars[index - 20 : index]]
        prior_lows = [v for v in prior_lows if v is not None]
        if not prior_lows:
            continue
        floor = min(prior_lows)
        if close < floor:
            after = [_c(b, "close") for b in bars[index + 1 : index + 1 + window]]
            after = [v for v in after if v is not None]
            if any(v > floor for v in after):
                return True
    return False


# ── 6. composite fingerprint ────────────────────────────────────────────────


@dataclass
class SmartMoneyFingerprint:
    symbol: str
    analysis_date: str
    lookback_days: int
    signal: SmartMoneySignal
    footprint_score: float
    phase: AccumulationPhase
    zones: list[AccumulationZone]
    anomalies: list[VolumeAnomaly]
    absorption: AbsorptionResult
    breakout: BreakoutPreparation
    days_with_footprint: int
    consecutive_footprint_days: int
    total_footprint_volume: int
    estimated_position_size: PositionSize
    acceleration: bool
    shakeout: bool
    targets: dict[str, float]
    recommendation: str
    rationale: list[str]
    data_quality_score: Optional[float]
    source_reliability: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "analysis_date": self.analysis_date,
            "lookback_days": self.lookback_days,
            "signal": self.signal.value,
            "phase": self.phase.value,
            "footprint_score": self.footprint_score,
            "days_with_footprint": self.days_with_footprint,
            "consecutive_footprint_days": self.consecutive_footprint_days,
            "total_footprint_volume": self.total_footprint_volume,
            "estimated_position_size": self.estimated_position_size.value,
            "acceleration": self.acceleration,
            "shakeout": self.shakeout,
            "absorption": self.absorption.as_dict(),
            "breakout": self.breakout.as_dict(),
            "zone": self.zones[-1].as_dict() if self.zones else None,
            "anomalies": [a.as_dict() for a in self.anomalies[-10:]],
            "anomalies_total": len(self.anomalies),
            "targets": self.targets,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "data_quality_score": self.data_quality_score,
            "source_reliability": self.source_reliability,
            "note": (
                "footprint_score and preparation scores are weighted flag sums, not "
                "probabilities. Run scripts/egx_calibrate.py --signals hidden_accumulation "
                "for the measured hit rate of these patterns on your stored history."
            ),
        }


def _consecutive_footprint_days(anomalies: Sequence[VolumeAnomaly]) -> int:
    """Footprint days that actually run continuously up to the newest session.

    The reference increments a counter for every anomaly, so the field reads as
    "how many anomalies" regardless of when they happened. Counting backwards
    and stopping at the first gap is the only version of this number that can
    mean "the buying is happening right now".
    """
    if not anomalies:
        return 0
    ordered = sorted(anomalies, key=lambda a: a.date)
    run = 1
    for previous, current in zip(ordered, ordered[1:]):
        if _is_adjacent(previous.date, current.date):
            run += 1
        else:
            run = 1
    return run


def analyze_footprint(
    bars: Sequence[Bar],
    symbol: str,
    *,
    lookback_days: int = 20,
    zone_lookback: int = 30,
    data_quality_score: Optional[float] = None,
    source_reliability: Optional[str] = None,
) -> SmartMoneyFingerprint:
    """Combine every detector into one described state (no probabilities)."""
    if len(bars) < lookback_days + 5:
        raise ValueError(f"need at least {lookback_days + 5} bars, got {len(bars)}")

    anomalies = detect_volume_anomalies(bars, symbol=symbol, lookback=lookback_days)
    zones = detect_accumulation_zones(bars, symbol=symbol, lookback_days=zone_lookback)
    absorption = detect_absorption_pattern(bars, window=20, lookback=lookback_days)
    breakout = detect_breakout_preparation(bars, window=zone_lookback)
    shakeout = detect_shakeout(bars)

    triggered = 0
    score = 0.0
    if len(anomalies) >= 3:
        score += fmean(a.footprint_score for a in anomalies) * 0.3
        triggered += 1
    if zones:
        score += max(z.accumulation_strength for z in zones) * 0.3
        triggered += 1
    if absorption.absorption_detected:
        score += absorption.institutional_strength * 0.2
        triggered += 1
    if breakout.breakout_ready:
        score += breakout.preparation_score * 0.2
        triggered += 1
    score = min(100.0, score)

    distribution = _detect_distribution(bars, lookback=lookback_days)
    if distribution:
        signal, phase = SmartMoneySignal.SMART_DISTRIBUTION, AccumulationPhase.PHASE_8_DISTRIBUTION_START
    elif shakeout and (absorption.absorption_detected or len(anomalies) >= 2):
        signal, phase = SmartMoneySignal.TRAP_SHAKEOUT, AccumulationPhase.PHASE_4B_SHAKEOUT
    elif breakout.breakout_ready and (triggered >= 2):
        signal, phase = SmartMoneySignal.BREAKOUT_IMMINENT, AccumulationPhase.PHASE_5_ACCUMULATION_COMPLETE
    elif triggered >= 3 and score >= 70:
        signal, phase = SmartMoneySignal.STRONG_ACCUMULATION, AccumulationPhase.PHASE_5_ACCUMULATION_COMPLETE
    elif triggered >= 2 and score >= 50:
        signal, phase = SmartMoneySignal.MILD_ACCUMULATION, AccumulationPhase.PHASE_3_PREPARATION
    elif zones and zones[-1].days_in_zone >= 30:
        signal, phase = SmartMoneySignal.BASE_BUILDING, AccumulationPhase.PHASE_3_PREPARATION
    elif any(a.is_hidden_accumulation for a in anomalies):
        signal, phase = SmartMoneySignal.HIDDEN_ACCUMULATION, AccumulationPhase.PHASE_2_QUIET
    else:
        signal, phase = SmartMoneySignal.NEUTRAL, AccumulationPhase.PHASE_1_SILENT

    closes = [c for c in (_c(b, "close") for b in bars) if c is not None]
    current = closes[-1] if closes else 0.0
    zone = zones[-1] if zones else None
    if zone:
        height = zone.zone_width
        targets = {
            "conservative": round(zone.zone_high + height * 0.5, 4),
            "breakout": round(zone.zone_high + height * 1.5, 4),
            "aggressive": round(zone.zone_high + height * 2.5, 4),
            "risk_zone_low": round(zone.zone_low, 4),
            "current_price": current,
        }
    else:
        highs = [_c(b, "high") for b in bars[-20:]]
        lows = [_c(b, "low") for b in bars[-20:]]
        highs = [v for v in highs if v is not None]
        lows = [v for v in lows if v is not None]
        top, bottom = (max(highs), min(lows)) if highs and lows else (current, current)
        height = top - bottom
        targets = {
            "conservative": round(top + height * 0.5, 4),
            "breakout": round(top + height * 1.5, 4),
            "aggressive": round(top + height * 2.5, 4),
            "risk_zone_low": round(bottom, 4),
            "current_price": current,
        }

    total_volume = sum(a.total_volume for a in anomalies)
    avg_volume = fmean([_c(b, "volume") or 0.0 for b in bars[-lookback_days:]]) if bars else 0.0
    ratio = total_volume / (avg_volume * max(1, len(anomalies))) if avg_volume else 0.0
    size = (
        PositionSize.MASSIVE
        if ratio > 3
        else PositionSize.LARGE
        if ratio > 2
        else PositionSize.MEDIUM
        if ratio > 1.2
        else PositionSize.SMALL
    )

    recommendation, rationale = _recommend(signal, anomalies, absorption, breakout, zone, distribution)
    return SmartMoneyFingerprint(
        symbol=symbol,
        analysis_date=_day(bars[-1]) or "",
        lookback_days=lookback_days,
        signal=signal,
        footprint_score=round(score, 2),
        phase=phase,
        zones=zones,
        anomalies=anomalies,
        absorption=absorption,
        breakout=breakout,
        days_with_footprint=len(anomalies),
        consecutive_footprint_days=_consecutive_footprint_days(anomalies),
        total_footprint_volume=total_volume,
        estimated_position_size=size,
        acceleration=len(anomalies) > lookback_days * 0.5,
        shakeout=shakeout,
        targets=targets,
        recommendation=recommendation,
        rationale=rationale,
        data_quality_score=data_quality_score,
        source_reliability=source_reliability,
    )


def _detect_distribution(bars: Sequence[Bar], *, lookback: int = 20) -> bool:
    """The mirror footprint: heavy volume while price closes near the session low.

    Institutions distributing want the opposite of accumulating — they want to
    sell into strength without breaking the price, so volume spikes while the
    close sits in the lower part of the range.
    """
    tail = list(bars[-lookback:])
    if len(tail) < lookback:
        return False
    avg_volume = fmean([_c(b, "volume") or 0.0 for b in tail[:-1]]) if len(tail) > 1 else 0.0
    hits = 0
    for bar in tail[-5:]:
        volume, close = _c(bar, "volume"), _c(bar, "close")
        high, low = _c(bar, "high"), _c(bar, "low")
        if None in (volume, close, high, low) or avg_volume <= 0 or high <= low:
            continue
        location = (close - low) / (high - low)
        if volume / avg_volume > 1.5 and location < 0.35:
            hits += 1
    return hits >= 3


def _recommend(signal, anomalies, absorption, breakout, zone, distribution) -> tuple[str, list[str]]:
    rationale: list[str] = []
    if signal is SmartMoneySignal.SMART_DISTRIBUTION:
        rationale.append("Volume spikes with closes in the lower third of the session range")
        rationale.append("That is the mirror of the accumulation footprint, not a dip to buy")
        return "🔴 DISTRIBUTION — volume is leaving into strength", rationale
    if signal is SmartMoneySignal.STRONG_ACCUMULATION:
        rationale.append(f"{len(anomalies)} footprint sessions (volume above 1.5× average)")
        rationale.append("Absorption active: dips are being bought back on volume")
        if breakout.breakout_ready:
            rationale.append("Range position, volatility compression and volume build all align")
        return "🟢 STRONG ACCUMULATION — multiple independent footprints agree", rationale
    if signal is SmartMoneySignal.BREAKOUT_IMMINENT:
        rationale.append("Price is in the top of its range on compressed volatility")
        rationale.append("Volume is building rather than fading")
        return "🟠 BREAKOUT PREPARATION — base looks complete, direction unproven", rationale
    if signal is SmartMoneySignal.TRAP_SHAKEOUT:
        rationale.append("A prior low was broken and reclaimed — weak hands were shaken out")
        return "🟡 SHAKEOUT — re-test the level before acting", rationale
    if signal is SmartMoneySignal.MILD_ACCUMULATION:
        rationale.append("Some footprint sessions present, not enough independent agreement")
        return "🟡 MILD ACCUMULATION — early, unconfirmed", rationale
    if signal is SmartMoneySignal.BASE_BUILDING:
        rationale.append(f"Price has been range-bound for {zone.days_in_zone} sessions" if zone else "")
        return "⚪ BASE BUILDING — consolidation, no breakout evidence yet", [r for r in rationale if r]
    if signal is SmartMoneySignal.HIDDEN_ACCUMULATION:
        rationale.append("Volume above 2× average with a sub-1.5% candle — the hidden footprint")
        return "🟡 HIDDEN ACCUMULATION — one session, needs confirmation", rationale
    rationale.append("No volume/price divergence beyond the normal range")
    return "⚪ NEUTRAL — no institutional footprint detected", rationale


# ── 7. fast conditions for the calibration engine ───────────────────────────


def hidden_accumulation_flags(bars: Sequence[Bar], *, lookback: int = 20) -> list[Optional[bool]]:
    """Per-session ``is_hidden_accumulation`` for the whole series.

    ``is_hidden_accumulation`` is the design document's central claim: volume
    more than twice the trailing average while the candle moves less than
    1.5%. Returned as a series so calibration can measure what follows it.
    """
    out: list[Optional[bool]] = []
    volumes = [_c(b, "volume") or 0.0 for b in bars]
    for index, bar in enumerate(bars):
        if index < lookback:
            out.append(None)
            continue
        volume = volumes[index]
        avg = fmean(volumes[index - lookback : index]) if index >= lookback else 0.0
        open_, close = _c(bar, "open"), _c(bar, "close")
        if avg <= 0 or open_ in (None, 0) or close is None:
            out.append(None)
            continue
        ratio = volume / (avg + 1.0)
        move = abs((close - open_) / (open_ + 1.0)) * 100.0
        out.append(ratio > 2.0 and move < 1.5)
    return out


def volume_footprint_flags(bars: Sequence[Bar], *, lookback: int = 20) -> list[Optional[bool]]:
    """The looser threshold pair (ratio > 1.5, move < 2.5%) the reference uses."""
    out: list[Optional[bool]] = []
    volumes = [_c(b, "volume") or 0.0 for b in bars]
    for index, bar in enumerate(bars):
        if index < lookback:
            out.append(None)
            continue
        avg = fmean(volumes[index - lookback : index])
        open_, close = _c(bar, "open"), _c(bar, "close")
        if avg <= 0 or open_ in (None, 0) or close is None:
            out.append(None)
            continue
        ratio = volumes[index] / (avg + 1.0)
        move = abs((close - open_) / (open_ + 1.0)) * 100.0
        out.append(ratio > 1.5 and move < 2.5)
    return out


def absorption_flags(bars: Sequence[Bar], *, lookback: int = 20) -> list[Optional[bool]]:
    return [absorption_on_day(bars, i, lookback=lookback) is not None for i in range(len(bars))]
