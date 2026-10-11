"""Sector momentum scoring and status classification — pure functions.

These are the exact rules the live sector scanner (``egx_service``) has always
used. They live here, with no network or framework imports, so that two things
can share one definition:

* the live scanner, which labels sectors Hot / Warming / Cooling / Cold, and
* the sector calibration in ``core.calibration.sectors``, which replays the same
  rules over stored history and measures whether the labels predict anything.

If the two ever disagreed, the calibration would be measuring a rule nobody
runs. Keeping one copy prevents that.

Nothing here is calibrated by itself: the weights (30/25/25/20) and the status
thresholds (65 / 50 / 35) are the scanner's own, carried over unchanged. Whether
they carry information is what the calibration answers.
"""

from __future__ import annotations

from typing import Optional, Sequence


def momentum_score(
    avg_change: float,
    avg_rsi: float,
    breadth_pct: float,
    volume_flow_positive: bool,
    change_rank_pct: float,
) -> int:
    """Compute a 0–100 sector momentum score from four components.

    change_rank_pct  30 pts  cross-sectional rank of the sector's average change
    avg_rsi          25 pts  mid-band RSI scores highest (50–70)
    breadth_pct      25 pts  share of members advancing
    volume_flow      20 pts  sum of (volume − 20-session average) is positive
    """
    change_pts = round(change_rank_pct * 30)

    if 50 <= avg_rsi <= 70:
        rsi_pts = 25
    elif 40 <= avg_rsi < 50 or 70 < avg_rsi <= 80:
        rsi_pts = 15
    elif 30 <= avg_rsi < 40:
        rsi_pts = 10
    elif avg_rsi > 80:
        rsi_pts = 5
    else:
        rsi_pts = 8

    breadth_pts = round(min(breadth_pct, 100) / 100 * 25)
    volume_pts = 20 if volume_flow_positive else 0
    return max(0, min(100, change_pts + rsi_pts + breadth_pts + volume_pts))


def classify_status(
    momentum: float,
    avg_change: float,
    breadth_pct: float,
    volume_flow_positive: bool,
) -> str:
    """Hot / Warming / Cooling / Cold, exactly as the scanner has always labelled them."""
    if momentum >= 65 and volume_flow_positive:
        return "Hot"
    if momentum >= 50 or (avg_change > 0 and breadth_pct > 50):
        return "Warming"
    if momentum >= 35 and not volume_flow_positive:
        return "Cooling"
    return "Cold"


def change_rank_percentiles(changes_by_sector: dict[str, float]) -> dict[str, float]:
    """Cross-sectional rank of each sector's average change, as the scanner computes it.

    Sorted ascending, rank = index / count; with a single sector the rank is 0.5.
    Note the denominator is the count, not count − 1: the scanner has always
    used this form, and the replay must match it.
    """
    ordered = sorted(changes_by_sector, key=lambda k: changes_by_sector[k])
    n = len(ordered)
    return {k: (i / n if n > 1 else 0.5) for i, k in enumerate(ordered)}


def wilder_rsi_series(closes: Sequence[float], period: int = 14) -> list[Optional[float]]:
    """Wilder RSI at every index, computed incrementally in one pass.

    Index ``i`` is None until ``period`` changes have been observed, i.e. the
    first defined value is at index ``period``. Each value uses only closes up
    to and including its own index, so the series is safe to replay.
    """
    out: list[Optional[float]] = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def wilder_rsi(closes: Sequence[float], period: int = 14) -> Optional[float]:
    """RSI at the last close (TradingView's default Wilder smoothing)."""
    series = wilder_rsi_series(closes, period)
    return series[-1] if series else None
