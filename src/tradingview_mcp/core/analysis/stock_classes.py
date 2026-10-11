"""Stock classes: describe each stock's behaviour, then group stocks by it.

Principle: no single measurement suits every stock. A thinly traded stock, a stock
that gaps every few sessions and a stock that trends are different problems, so
each class gets its own list of allowed tools. A stock moves between classes as its
measured behaviour changes.

Everything here is computed from daily OHLCV only and is causal: the class at
session ``i`` uses bars ``0..i``. Missing values are never imputed; a window with a
missing required value is ``UNCLASSIFIED``.

Thresholds live in ``ClassRules``. The defaults are PROVISIONAL: they are reasoned
starting points, not values fitted to EGX data. They must be checked on real
history before any result is reported as a finding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median, stdev
from typing import Any, Mapping, Optional, Sequence

Bar = Mapping[str, Any]

DORMANT = "DORMANT"
GAP_PRONE = "GAP_PRONE"
TRENDING = "TRENDING"
RANGE_BOUND = "RANGE_BOUND"
MIXED = "MIXED"
UNCLASSIFIED = "UNCLASSIFIED"

CLASSES = (DORMANT, GAP_PRONE, TRENDING, RANGE_BOUND, MIXED, UNCLASSIFIED)


@dataclass(frozen=True)
class ClassRules:
    """All classification thresholds, in one place. PROVISIONAL until checked on real data."""

    window: int = 60                      # sessions used for every feature
    confirm_sessions: int = 5             # a new class must hold this many sessions before it is official
    min_median_traded_value: float = 1_000_000.0  # EGP per session (PROVISIONAL)
    max_zero_volume_share: float = 0.10   # share of sessions with no trading
    gap_threshold: float = 0.02           # |open / previous close - 1| counted as a gap
    max_gap_share: float = 0.10           # share of sessions that gap
    trend_efficiency: float = 0.30        # Kaufman efficiency ratio at or above: trending
    range_efficiency: float = 0.12        # at or below: range-bound
    high_volatility_sd: float = 0.03      # daily return standard deviation flag


@dataclass(frozen=True)
class Features:
    median_traded_value: float
    zero_volume_share: float
    gap_share: float
    efficiency_ratio: float
    daily_volatility: float
    sessions: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "median_traded_value": round(self.median_traded_value, 0),
            "zero_volume_share": round(self.zero_volume_share, 4),
            "gap_share": round(self.gap_share, 4),
            "efficiency_ratio": round(self.efficiency_ratio, 4),
            "daily_volatility": round(self.daily_volatility, 5),
            "sessions": self.sessions,
        }


@dataclass(frozen=True)
class Assessment:
    raw_class: str                        # what today's features say
    official_class: str                   # after the confirmation rule
    sessions_in_official: int
    flags: tuple[str, ...]
    features: Optional[Features]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_class": self.raw_class,
            "official_class": self.official_class,
            "sessions_in_official_class": self.sessions_in_official,
            "flags": list(self.flags),
            "features": self.features.as_dict() if self.features else None,
            "reason": self.reason,
        }


def _num(bar: Bar, key: str) -> Optional[float]:
    value = bar.get(key)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def features_at(bars: Sequence[Bar], i: int, rules: ClassRules = ClassRules()) -> Optional[Features]:
    """Features over the ``window`` sessions ending at ``i``. None if any required value is missing."""
    n = rules.window
    if i < n:
        return None
    span = bars[i - n : i + 1]  # n + 1 bars: the extra one supplies the first previous close
    closes = [_num(b, "close") for b in span]
    opens = [_num(b, "open") for b in span[1:]]
    volumes = [_num(b, "volume") for b in span[1:]]
    if any(c is None for c in closes) or any(o is None for o in opens) or any(v is None for v in volumes):
        return None

    returns = [closes[k] / closes[k - 1] - 1.0 for k in range(1, len(closes)) if closes[k - 1] > 0]
    if len(returns) < n:
        return None

    traded = [closes[k] * volumes[k - 1] for k in range(1, len(closes))]
    zero_share = sum(1 for v in volumes if v == 0) / n
    gaps = sum(
        1 for k in range(1, len(closes))
        if closes[k - 1] > 0 and abs(opens[k - 1] / closes[k - 1] - 1.0) >= rules.gap_threshold
    )
    path = sum(abs(closes[k] - closes[k - 1]) for k in range(1, len(closes)))
    net = abs(closes[-1] - closes[0])
    efficiency = 0.0 if path == 0 else net / path
    return Features(
        median_traded_value=median(traded),
        zero_volume_share=zero_share,
        gap_share=gaps / n,
        efficiency_ratio=efficiency,
        daily_volatility=stdev(returns),
        sessions=n,
    )


def raw_class(features: Optional[Features], rules: ClassRules = ClassRules()) -> tuple[str, str]:
    """Class and plain reason from one set of features. Precedence: dormant, gap-prone, trending, range, mixed."""
    if features is None:
        return UNCLASSIFIED, "not enough complete history in the window"
    if features.zero_volume_share > rules.max_zero_volume_share:
        return DORMANT, f"no trading on {features.zero_volume_share:.0%} of sessions (limit {rules.max_zero_volume_share:.0%})"
    if features.median_traded_value < rules.min_median_traded_value:
        return DORMANT, "median traded value below the liquidity floor"
    if features.gap_share > rules.max_gap_share:
        return GAP_PRONE, f"gaps of {rules.gap_threshold:.0%} or more on {features.gap_share:.0%} of sessions"
    if features.efficiency_ratio >= rules.trend_efficiency:
        return TRENDING, f"efficiency ratio {features.efficiency_ratio:.2f} (price moves mostly in one direction)"
    if features.efficiency_ratio <= rules.range_efficiency:
        return RANGE_BOUND, f"efficiency ratio {features.efficiency_ratio:.2f} (price mostly goes back and forth)"
    return MIXED, f"efficiency ratio {features.efficiency_ratio:.2f} (between trending and range-bound)"


def classify_history(bars: Sequence[Bar], rules: ClassRules = ClassRules()) -> list[Assessment]:
    """Assessment at every session, with the confirmation rule applied causally.

    A session's raw class becomes official only after ``confirm_sessions`` consecutive
    sessions agree on it. Until then the previous official class holds. This stops a
    stock flipping between classes on one noisy day.
    """
    out: list[Assessment] = []
    official: Optional[str] = None
    candidate: Optional[str] = None
    run = 0
    days_in_official = 0
    for i in range(len(bars)):
        feats = features_at(bars, i, rules)
        raw, reason = raw_class(feats, rules)
        flags: list[str] = []
        if feats is not None and feats.daily_volatility >= rules.high_volatility_sd:
            flags.append("HIGH_VOLATILITY")

        if official is None:
            if raw != UNCLASSIFIED:
                official = raw
                days_in_official = 1
        elif raw == official:
            candidate, run = None, 0
            days_in_official += 1
        else:
            if raw == candidate:
                run += 1
            else:
                candidate, run = raw, 1
            if run >= rules.confirm_sessions:
                official = raw
                candidate, run = None, 0
                days_in_official = 1
            else:
                days_in_official += 1

        out.append(Assessment(
            raw_class=raw,
            official_class=official or UNCLASSIFIED,
            sessions_in_official=days_in_official if official else 0,
            flags=tuple(flags),
            features=feats,
            reason=reason,
        ))
    return out


def transitions(history: Sequence[Assessment], bars: Sequence[Bar]) -> list[dict[str, str]]:
    """Dates on which the official class changed, with the previous and new class."""
    changes: list[dict[str, str]] = []
    prev: Optional[str] = None
    for a, b in zip(history, bars):
        if a.official_class != prev and prev is not None:
            changes.append({"date": str(b.get("date", "")), "from": prev, "to": a.official_class})
        prev = a.official_class
    return changes


# ── Which tools a class may use ───────────────────────────────────────────────
# A tool is allowed for a class only after it has been validated for that class by
# the calibration engine on held-out data. Nothing is validated yet, so every class
# currently has an empty allowlist. The exclusions below are design-level reasons
# (what a tool assumes about the stock), stated as hypotheses to be checked.

VOLUME_BASED = (
    "c1", "c2", "c4", "c5", "d1", "d2", "d4", "d5",
    "hidden_accumulation", "volume_footprint", "absorption",
    "breakout_preparation", "distribution_footprint", "shakeout_reclaim",
    "wy_selling_climax", "wy_spring", "wy_no_supply", "wy_secondary_test",
)

CLASS_POLICY: dict[str, dict[str, Any]] = {
    DORMANT: {
        "status": "NO_TOOLS_UNTIL_LIQUID",
        "reason": "Volume-based measures rest on trades that barely happen; a few prints dominate them.",
        "excluded": {name: "volume-based; unreliable when trading is sparse" for name in VOLUME_BASED},
    },
    GAP_PRONE: {
        "status": "CLOSE_BASED_PENDING_AUDIT",
        "reason": "Frequent opening gaps make open-based conditions measure the gap, not the session.",
        "excluded": {},
    },
    TRENDING: {
        "status": "NOT_YET_VALIDATED",
        "reason": "Trend and breakout conditions are the candidates; range-reversion conditions do not fit.",
        "excluded": {},
    },
    RANGE_BOUND: {
        "status": "NOT_YET_VALIDATED",
        "reason": "Range-edge conditions are the candidates; breakout-trend conditions do not fit.",
        "excluded": {},
    },
    MIXED: {
        "status": "NOT_YET_VALIDATED",
        "reason": "No specific hypothesis about this class; any tool needs its own validation here.",
        "excluded": {},
    },
    UNCLASSIFIED: {
        "status": "NO_TOOLS",
        "reason": "Not enough complete history to describe the stock.",
        "excluded": {},
    },
}

# Filled by the calibration engine once a tool passes validation for a class.
VALIDATED_BY_CLASS: dict[str, tuple[str, ...]] = {c: () for c in CLASSES}


def allowed_tools(stock_class: str) -> tuple[str, ...]:
    return VALIDATED_BY_CLASS.get(stock_class, ())


@dataclass
class ClassSummary:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, stock_class: str) -> None:
        self.counts[stock_class] = self.counts.get(stock_class, 0) + 1
