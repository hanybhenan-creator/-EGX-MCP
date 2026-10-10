"""Signal definitions — the five conditions of the EGX smart-money model,
expressed as functions over time so they can be *measured* instead of
asserted.

The original detector (``services/egx_smart_money_core.py``) evaluates its
conditions once, on the most recent bar, and returns a verdict. That is
enough to print a signal and not enough to know whether it works: there is no
way to ask how often the same conditions preceded a gain. This module
re-expresses every condition as ``f(context, index) -> bool`` — the same
arithmetic, evaluated on every historical session — which is what turns
"confidence: 87" into "fired 214 times in 2019-2026, hit rate 54% ±6% against
a base rate of 49%, q = 0.31".

Three deliberate differences from the legacy detector, each one a bug found
while making the conditions measurable:

1. ``price_structure_bullish`` (condition 3) as written compares the current
   close against the highest high of a window that *includes the current
   bar*. Since a close can never exceed its own session's high, the condition
   is unsatisfiable: measured over 4,800 synthetic sessions it fired exactly
   zero times, and the "5/5 high conviction" branch of the legacy code is
   therefore dead code. The corrected variant compares against the previous
   window's high, which is what "breaking resistance" was meant to mean. The
   legacy variant is kept registered (as ``*_legacy``) precisely so the
   calibration report can show the difference.

2. The volume profile in the legacy detector is built over whatever dataframe
   it is handed (a 6-month fetch) while the other conditions use 20- and
   30-bar windows. The composite therefore mixes timescales with no stated
   reason. Here every window is an explicit, documented parameter.

3. ``numpy``/``pandas`` are not used. The arithmetic is identical (including
   the quirk that the profile's top bin is never filled, so POC can never sit
   exactly at the window high), but the module runs on the plain bar dicts the
   data layer stores — the calibration engine must be able to run over years
   of history without building a dataframe per symbol per bar.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from tradingview_mcp.core.analysis import footprints

Bar = Mapping[str, Any]
CtxFunction = Callable[["SignalContext", int], Optional[bool]]

# Defaults mirror the configuration block of the project's master guide
# (`ANALYSIS`): lookback_periods 20, min_conditions_met 4, min_risk_reward 2.0.
DEFAULT_PARAMS: dict[str, float] = {
    "flow_period": 20,          # order-flow / OBV / ADL window (guide: 20)
    "structure_window": 20,     # breakout / swing window
    "profile_window": 120,      # volume-profile window (guide: full fetch ≈ 6mo)
    "profile_bins": 20,
    "buy_sell_ratio": 1.5,      # guide: institutional buying if ratio > 1.5
    "sell_buy_ratio": 0.67,     # guide: institutional selling if ratio < 0.67
    "min_conditions": 4,        # guide: min_conditions_met = 4 of 5
    "min_risk_reward": 2.0,     # guide: min_risk_reward = 2.0
}


# ── indicators (numpy-equivalent, stdlib only) ──────────────────────────────


def _f(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def obv_series(bars: Sequence[Bar]) -> list[Optional[float]]:
    """On-Balance Volume: cumulative sign(close change) × volume.

    The first bar has no prior close, so its contribution is 0 — the value
    pandas produces (NaN at index 0, skipped by rolling means) is equivalent
    for every window of length ≥ 2.
    """
    out: list[Optional[float]] = []
    running = 0.0
    prev: Optional[float] = None
    for bar in bars:
        close = _f(bar.get("close"))
        volume = _f(bar.get("volume")) or 0.0
        if prev is not None and close is not None:
            if close > prev:
                running += volume
            elif close < prev:
                running -= volume
        out.append(running)
        if close is not None:
            prev = close
    return out


def adl_series(bars: Sequence[Bar]) -> list[Optional[float]]:
    """Accumulation/Distribution Line: cumulative close-location-value × volume.

    CLV = ((close − low) − (high − close)) / (high − low); it answers *where in
    the session's range the price settled*, which is the closest thing to
    institutional intent obtainable from OHLCV alone.
    """
    out: list[Optional[float]] = []
    running = 0.0
    for bar in bars:
        close, high, low = _f(bar.get("close")), _f(bar.get("high")), _f(bar.get("low"))
        volume = _f(bar.get("volume")) or 0.0
        if close is None or high is None or low is None:
            out.append(None)
            continue
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        running += clv * volume
        out.append(running)
    return out


def rolling_mean(values: Sequence[Optional[float]], window: int, index: int) -> Optional[float]:
    if index + 1 < window:
        return None
    chunk = [v for v in values[index + 1 - window : index + 1] if v is not None]
    if len(chunk) < window:
        return None
    return sum(chunk) / len(chunk)


def highest_high(bars: Sequence[Bar], start: int, end: int) -> Optional[float]:
    """Highest high over the inclusive range [start, end]."""
    vals = [_f(bars[i].get("high")) for i in range(max(0, start), min(end + 1, len(bars)))]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def lowest_low(bars: Sequence[Bar], start: int, end: int) -> Optional[float]:
    vals = [_f(bars[i].get("low")) for i in range(max(0, start), min(end + 1, len(bars)))]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def volume_profile_poc(
    closes: Sequence[float], volumes: Sequence[float], *, bins: int = 20
) -> tuple[float, float]:
    """Point of control of a binned volume profile — ``(poc_price, poc_strength)``.

    Faithful to the legacy binning: ``numpy.linspace(min, max, bins)`` edges,
    a close belongs to bin ``i`` when ``edges[i] <= close < edges[i+1]``, and
    the loop stops at ``bins-1``. Consequence (preserved, not fixed): the top
    edge and anything equal to the window maximum lands in no bin at all, so
    the POC is never the highest price level of the window.
    """
    if not closes or not volumes:
        return (0.0, 0.0)
    lo, hi = min(closes), max(closes)
    if hi <= lo:
        return (lo, 0.0)
    edges = [lo + (hi - lo) * k / (bins - 1) for k in range(bins)]
    totals = [0.0] * bins
    for close, volume in zip(closes, volumes):
        idx = bisect_right(edges, close) - 1
        if 0 <= idx < bins - 1:  # the last bin is never filled (legacy quirk)
            totals[idx] += volume
    grand = sum(totals)
    best_idx, best_val = 0, -1.0
    for i, value in enumerate(totals):
        if value > best_val:  # first maximum wins, like numpy.argmax
            best_idx, best_val = i, value
    return (edges[best_idx], (best_val / grand) if grand else 0.0)


# ── evaluation context ──────────────────────────────────────────────────────


@dataclass
class SignalContext:
    """One symbol's bars plus every series the conditions need, computed once.

    Plugins get the same object: ``ctx.series("my_indicator", builder)`` results
    are memoized, so adding a condition never means recomputing OBV per bar.
    """

    bars: Sequence[Bar]
    params: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    _series: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        merged = dict(DEFAULT_PARAMS)
        merged.update({k: v for k, v in self.params.items()})
        self.params = merged

    def param(self, name: str) -> float:
        return float(self.params[name])

    @property
    def n(self) -> int:
        return len(self.bars)

    def series(self, key: str, builder: Callable[[], Any]) -> Any:
        if key not in self._series:
            self._series[key] = builder()
        return self._series[key]

    @property
    def closes(self) -> list[Optional[float]]:
        return self.series("closes", lambda: [_f(b.get("close")) for b in self.bars])

    @property
    def obv(self) -> list[Optional[float]]:
        return self.series("obv", lambda: obv_series(self.bars))

    @property
    def adl(self) -> list[Optional[float]]:
        return self.series("adl", lambda: adl_series(self.bars))

    def flow(self, index: int) -> Optional[dict[str, float]]:
        """Legacy order-flow signature over ``flow_period`` bars ending at index."""
        cache = self.series("flow", dict)
        if index not in cache:
            cache[index] = self._flow(index)
        return cache[index]

    def _flow(self, index: int) -> Optional[dict[str, float]]:
        period = int(self.param("flow_period"))
        if index + 1 < period + 1:  # need one extra bar for the first diff
            return None
        start = index + 1 - period
        closes = self.closes[start : index + 1]
        if any(c is None for c in closes):
            return None
        buy = sell = neutral = total = 0.0
        for k in range(1, len(closes)):
            volume = _f(self.bars[start + k].get("volume")) or 0.0
            total += volume
            delta = closes[k] - closes[k - 1]
            if delta > 0:
                buy += volume
            elif delta < 0:
                sell += volume
            else:
                neutral += volume
        high = highest_high(self.bars, start, index) or 0.0
        low = lowest_low(self.bars, start, index) or 0.0
        return {
            "buy": buy,
            "sell": sell,
            "neutral": neutral,
            "total": total,
            "buy_sell_ratio": (buy + 1.0) / (sell + 1.0),
            "capital_commitment": total * (high - low) / (period * 24.0),
        }

    def poc(self, index: int) -> Optional[tuple[float, float]]:
        cache = self.series("poc", dict)
        if index not in cache:
            cache[index] = self._poc(index)
        return cache[index]

    def _poc(self, index: int) -> Optional[tuple[float, float]]:
        window = int(self.param("profile_window"))
        if index + 1 < window:
            return None
        closes = [c for c in self.closes[index + 1 - window : index + 1] if c is not None]
        volumes = [_f(self.bars[i].get("volume")) or 0.0 for i in range(index + 1 - window, index + 1)]
        return volume_profile_poc(closes, volumes, bins=int(self.param("profile_bins")))

    def risk_reward(self, index: int) -> Optional[float]:
        """Legacy reward/risk: (mean target − entry low) / (entry high − stop).

        entry zone = close ±1%; targets = recent swing high and its measured
        move projection; stop = swing low −2% (the legacy formulas verbatim).
        """
        cache = self.series("risk_reward", dict)
        if index not in cache:
            cache[index] = self._risk_reward(index)
        return cache[index]

    def _risk_reward(self, index: int) -> Optional[float]:
        window = int(self.param("structure_window"))
        if index + 1 < window:
            return None
        close = self.closes[index]
        swing_high = highest_high(self.bars, index + 1 - window, index)
        swing_low = lowest_low(self.bars, index + 1 - window, index)
        if close is None or not swing_high or not swing_low or swing_low <= 0:
            return None
        entry_low, entry_high = close * 0.99, close * 1.01
        first_target = swing_high
        second_target = swing_high * (1 + (swing_high - swing_low) / swing_low)
        risk = entry_high - swing_low * 0.98
        if risk <= 0:
            return None
        reward = (first_target + second_target) / 2 - entry_low
        return reward / risk


# ── the conditions ──────────────────────────────────────────────────────────


def c1_volume_profile_bullish(ctx: SignalContext, i: int) -> Optional[bool]:
    poc = ctx.poc(i)
    close = ctx.closes[i]
    if poc is None or close is None:
        return None
    return poc[0] < close


def c2_order_flow_buying(ctx: SignalContext, i: int) -> Optional[bool]:
    flow = ctx.flow(i)
    if flow is None:
        return None
    return flow["buy_sell_ratio"] > ctx.param("buy_sell_ratio") and flow["capital_commitment"] > 0


def c3_price_structure_legacy(ctx: SignalContext, i: int) -> Optional[bool]:
    """The legacy condition: close > highest high of a window INCLUDING bar i.

    Kept because it was shipped; provably never true (close ≤ high ≤ max high).
    """
    window = int(ctx.param("structure_window"))
    close = ctx.closes[i]
    top = highest_high(ctx.bars, i + 1 - window, i)
    if close is None or top is None:
        return None
    return close > top


def c3_price_structure_corrected(ctx: SignalContext, i: int) -> Optional[bool]:
    """Breakout against the PREVIOUS window's high — the intended meaning."""
    window = int(ctx.param("structure_window"))
    close = ctx.closes[i]
    if i + 1 < window + 1:
        return None
    top = highest_high(ctx.bars, i - window, i - 1)
    if close is None or top is None:
        return None
    return close > top


def c4_obv_confirmed(ctx: SignalContext, i: int) -> Optional[bool]:
    window = int(ctx.param("flow_period"))
    obv = ctx.obv
    if i + 1 < window + 1 or obv[i] is None:
        return None
    return obv[i] > rolling_mean(obv, window, i)


def c5_adl_confirmed(ctx: SignalContext, i: int) -> Optional[bool]:
    window = int(ctx.param("flow_period"))
    adl = ctx.adl
    if i + 1 < window + 1 or adl[i] is None:
        return None
    return adl[i] > rolling_mean(adl, window, i)


# Distribution mirrors: the same measurements, asking the opposite question.
# The guide asks for a DistributionDetector and the legacy code only ever
# shipped the accumulation side, so the mirror conditions are registered here
# where they can be measured like everything else.


def d1_volume_profile_bearish(ctx: SignalContext, i: int) -> Optional[bool]:
    poc = ctx.poc(i)
    close = ctx.closes[i]
    if poc is None or close is None:
        return None
    return poc[0] > close


def d2_order_flow_selling(ctx: SignalContext, i: int) -> Optional[bool]:
    flow = ctx.flow(i)
    if flow is None:
        return None
    return flow["buy_sell_ratio"] < ctx.param("sell_buy_ratio") and flow["capital_commitment"] > 0


def d3_price_structure_bearish(ctx: SignalContext, i: int) -> Optional[bool]:
    window = int(ctx.param("structure_window"))
    close = ctx.closes[i]
    if i + 1 < window + 1:
        return None
    bottom = lowest_low(ctx.bars, i - window, i - 1)
    if close is None or bottom is None:
        return None
    return close < bottom


def d4_obv_weak(ctx: SignalContext, i: int) -> Optional[bool]:
    window = int(ctx.param("flow_period"))
    obv = ctx.obv
    if i + 1 < window + 1 or obv[i] is None:
        return None
    return obv[i] < rolling_mean(obv, window, i)


def d5_adl_weak(ctx: SignalContext, i: int) -> Optional[bool]:
    window = int(ctx.param("flow_period"))
    adl = ctx.adl
    if i + 1 < window + 1 or adl[i] is None:
        return None
    return adl[i] < rolling_mean(adl, window, i)


# ── institutional footprint conditions (core/analysis/footprints.py) ────────
#
# These are the design document's microstructure claims, evaluated per session
# so they can be measured like everything else. The series builders are
# memoized in the context: each costs one pass over the symbol's bars.


def fp_hidden_accumulation(ctx: SignalContext, i: int) -> Optional[bool]:
    flags = ctx.series(
        "fp:hidden",
        lambda: footprints.hidden_accumulation_flags(ctx.bars, lookback=int(ctx.param("flow_period"))),
    )
    return flags[i] if i < len(flags) else None


def fp_volume_footprint(ctx: SignalContext, i: int) -> Optional[bool]:
    flags = ctx.series(
        "fp:footprint",
        lambda: footprints.volume_footprint_flags(ctx.bars, lookback=int(ctx.param("flow_period"))),
    )
    return flags[i] if i < len(flags) else None


def fp_absorption(ctx: SignalContext, i: int) -> Optional[bool]:
    flags = ctx.series(
        "fp:absorption",
        lambda: footprints.absorption_flags(ctx.bars, lookback=int(ctx.param("flow_period"))),
    )
    return flags[i] if i < len(flags) else None


def fp_breakout_preparation(ctx: SignalContext, i: int) -> Optional[bool]:
    window = int(ctx.param("structure_window"))
    if i + 1 < window + 5:
        return None
    prep = footprints.detect_breakout_preparation(ctx.bars[: i + 1], window=window)
    return prep.breakout_ready


def fp_distribution_footprint(ctx: SignalContext, i: int) -> Optional[bool]:
    lookback = int(ctx.param("flow_period"))
    if i + 1 < lookback + 5:
        return None
    return footprints._detect_distribution(ctx.bars[: i + 1], lookback=lookback)


def fp_shakeout_reclaim(ctx: SignalContext, i: int) -> Optional[bool]:
    if i + 1 < 30:
        return None
    return footprints.detect_shakeout(ctx.bars[: i + 1], window=10)


LEGACY_CONDITIONS = ("c1", "c2", "c3_legacy", "c4", "c5")
CORRECTED_CONDITIONS = ("c1", "c2", "c3", "c4", "c5")
DISTRIBUTION_CONDITIONS = ("d1", "d2", "d3", "d4", "d5")


def _count(ctx: SignalContext, i: int, names: Sequence[str]) -> Optional[int]:
    met = 0
    seen = 0
    for name in names:
        spec = _REGISTRY[name]
        value = spec.fn(ctx, i)
        if value is None:
            return None
        seen += 1
        met += 1 if value else 0
    return met if seen else None


def accumulation_legacy(ctx: SignalContext, i: int) -> Optional[bool]:
    """The signal as shipped: ≥4 of the 5 legacy conditions."""
    met = _count(ctx, i, LEGACY_CONDITIONS)
    if met is None:
        return None
    return met >= int(ctx.param("min_conditions"))


def accumulation_corrected(ctx: SignalContext, i: int) -> Optional[bool]:
    """≥4 of 5 with the breakout condition actually satisfiable."""
    met = _count(ctx, i, CORRECTED_CONDITIONS)
    if met is None:
        return None
    return met >= int(ctx.param("min_conditions"))


def accumulation_high_conviction(ctx: SignalContext, i: int) -> Optional[bool]:
    """5/5 conditions plus reward/risk above the guide's 2.0 threshold."""
    met = _count(ctx, i, CORRECTED_CONDITIONS)
    if met is None:
        return None
    if met < 5:
        return False
    rr = ctx.risk_reward(i)
    return None if rr is None else rr > ctx.param("min_risk_reward")


def distribution_corrected(ctx: SignalContext, i: int) -> Optional[bool]:
    met = _count(ctx, i, DISTRIBUTION_CONDITIONS)
    if met is None:
        return None
    return met >= int(ctx.param("min_conditions"))


# ── registry (the plugin surface) ───────────────────────────────────────────


@dataclass(frozen=True)
class SignalSpec:
    name: str
    title: str
    fn: CtxFunction
    kind: str = "condition"   # condition | accumulation | distribution
    hypothesis: str = ""      # what the calibration test is actually testing
    min_history: int = 30


_REGISTRY: dict[str, SignalSpec] = {}


def register(spec: SignalSpec, *, replace: bool = False) -> SignalSpec:
    """Register a signal. Plugins add conditions here; calibration discovers them."""
    if spec.name in _REGISTRY and not replace:
        raise ValueError(f"signal already registered: {spec.name!r}")
    _REGISTRY[spec.name] = spec
    return spec


def get_signal(name: str) -> SignalSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown signal {name!r}; known: {sorted(_REGISTRY)}") from exc


def all_signals(*, kind: Optional[str] = None) -> list[SignalSpec]:
    return [s for s in _REGISTRY.values() if kind is None or s.kind == kind]


def signal_names(*, kind: Optional[str] = None) -> list[str]:
    return [s.name for s in all_signals(kind=kind)]


def register_builtins() -> None:
    builtins = [
        SignalSpec("c1", "Volume profile bullish (POC below price)", c1_volume_profile_bullish, "condition",
                   "value was accumulated below the current price"),
        SignalSpec("c2", "Order flow shows institutional buying", c2_order_flow_buying, "condition",
                   "buy/sell volume ratio > 1.5 over 20 sessions"),
        SignalSpec("c3_legacy", "Price above 20d high — LEGACY (unsatisfiable)", c3_price_structure_legacy, "condition",
                   "legacy comparison includes the current bar, so it can never fire"),
        SignalSpec("c3", "Breakout above prior 20d high (corrected)", c3_price_structure_corrected, "condition",
                   "close exceeds the highest high of the previous 20 sessions"),
        SignalSpec("c4", "OBV above its 20-session mean", c4_obv_confirmed, "condition",
                   "on-balance volume confirmes accumulation"),
        SignalSpec("c5", "ADL above its 20-session mean", c5_adl_confirmed, "condition",
                   "accumulation/distribution line confirmes accumulation"),
        SignalSpec("d1", "Volume profile bearish (POC above price)", d1_volume_profile_bearish, "condition",
                   "value was distributed above the current price"),
        SignalSpec("d2", "Order flow shows institutional selling", d2_order_flow_selling, "condition",
                   "buy/sell volume ratio < 0.67 over 20 sessions"),
        SignalSpec("d3", "Breakdown below prior 20d low", d3_price_structure_bearish, "condition",
                   "close loses the lowest low of the previous 20 sessions"),
        SignalSpec("d4", "OBV below its 20-session mean", d4_obv_weak, "condition",
                   "on-balance volume confirmes distribution"),
        SignalSpec("d5", "ADL below its 20-session mean", d5_adl_weak, "condition",
                   "accumulation/distribution line confirmes distribution"),
        SignalSpec("hidden_accumulation", "Hidden accumulation (volume spike, flat candle)", fp_hidden_accumulation,
                   "condition", "institutions buy size without moving price: volume >2x average with a <1.5% candle"),
        SignalSpec("volume_footprint", "Volume footprint (1.5x volume, <2.5% candle)", fp_volume_footprint,
                   "condition", "the looser threshold pair of the reference implementation"),
        SignalSpec("absorption", "Absorption day (dip bought back on volume)", fp_absorption,
                   "condition", "session opens below the previous close and closes back above it on >1.5x volume"),
        SignalSpec("breakout_preparation", "Breakout preparation (top of range, compressed, volume build)",
                   fp_breakout_preparation, "condition",
                   "range position >0.7 with 5-day volatility under 70% of the 30-day and rising 5-day volume"),
        SignalSpec("distribution_footprint", "Distribution footprint (volume spike, close in lower third)",
                   fp_distribution_footprint, "condition",
                   "the mirror of hidden accumulation: heavy volume while price settles near the session low"),
        SignalSpec("shakeout_reclaim", "Shakeout and reclaim of a prior low", fp_shakeout_reclaim,
                   "condition", "a 20-session low is broken and reclaimed within ten sessions"),
        SignalSpec("accumulation_legacy", "Accumulation signal as shipped (≥4/5 legacy)", accumulation_legacy,
                   "accumulation", "the composite signal the project has been reporting"),
        SignalSpec("accumulation_corrected", "Accumulation signal, corrected breakout (≥4/5)", accumulation_corrected,
                   "accumulation", "the composite with every condition satisfiable"),
        SignalSpec("accumulation_high_conviction", "5/5 conditions and reward/risk > 2.0", accumulation_high_conviction,
                   "accumulation", "the legacy 'high conviction' tier, made reachable"),
        SignalSpec("distribution_corrected", "Distribution signal (≥4/5 bearish mirrors)", distribution_corrected,
                   "distribution", "the exit side the guide asks for"),
    ]
    for spec in builtins:
        if spec.name not in _REGISTRY:
            register(spec)


register_builtins()


def evaluate(name: str, ctx: SignalContext, index: int) -> Optional[bool]:
    return get_signal(name).fn(ctx, index)


def scan(ctx: SignalContext, *, names: Optional[Sequence[str]] = None) -> dict[str, list[Optional[bool]]]:
    """Evaluate every registered signal on every bar — one pass per symbol."""
    chosen = list(names) if names else list(_REGISTRY)
    out: dict[str, list[Optional[bool]]] = {}
    for name in chosen:
        spec = get_signal(name)
        series: list[Optional[bool]] = []
        for i in range(ctx.n):
            series.append(None if i + 1 < spec.min_history else spec.fn(ctx, i))
        out[name] = series
    return out
