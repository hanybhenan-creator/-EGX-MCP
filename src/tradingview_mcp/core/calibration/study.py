"""Event study: turn a signal into measured outcomes, then try to break it.

The pipeline for one signal and one horizon:

  1. **Events.** Walk every bar of every symbol. Where the signal is true,
     record the *next* session's open as the entry and the close ``horizon``
     sessions later as the exit. Entry strictly after the signal bar is what
     removes look-ahead: a signal computed on the close of session *t* cannot
     buy at a price that existed before that close.

  2. **Benchmark.** EGX is one small market; on a day the index moves 2%, every
     "signal" that fired that day looks clever. Each event's return is reduced
     by the equal-weighted return of *all symbols in the sample over the exact
     same entry/exit dates*. The headline number is the excess, not the raw
     return.

  3. **Base rate.** The unconditional hit rate of the sample over the same
     window — "you bought a random symbol on a random Sunday and held the same
     number of sessions". A hit rate without a base rate is not a result.

  4. **Inference.** A block permutation gives a p-value for the signal's
     *timing*: every symbol's full return series has its consecutive blocks
     reordered (preserving momentum and volatility clustering) and the
     statistic is recomputed at the very same event positions. Symbols
     are shuffled independently, so market-wide co-movement of same-date
     events is not preserved — which is why every measurement carries
     ``event_dates`` next to ``events``, and why the walk-forward fold check,
     not the p-value, is the final gate.

  5. **Multiple testing.** Testing fifteen conditions on one price history
     guarantees a "significant" one by chance; Benjamini–Hochberg q-values are
     applied across the batch.

  6. **Walk-forward.** The sample is cut into contiguous time folds. A signal
     that works only in the first half of the history is a curve fit, however
     small its p-value; the out-of-sample folds say which it is.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from tradingview_mcp.core.calibration import stats
from tradingview_mcp.core.calibration.signals import (
    DEFAULT_PARAMS,
    Bar,
    SignalContext,
    get_signal,
    signal_names,
)

# ── forward returns ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Forward:
    entry_index: int
    exit_index: int
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    ret: float


def _price(bar: Bar, key: str) -> Optional[float]:
    value = bar.get(key)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def forward_return(bars: Sequence[Bar], index: int, horizon: int) -> Optional[Forward]:
    """Return from the open of ``index+1`` to the close of ``index+horizon``.

    Falls back to the entry bar's close when it has no open (some feeds omit
    it). Returns None when the future does not exist yet, when a price is
    missing, or when the computed return is outside a sanity band — a 95% loss
    or a 500% gain inside one holding period is a data error, and letting it
    through would let one bad bar dominate every statistic in the report.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1 session")
    entry_index, exit_index = index + 1, index + horizon
    if entry_index >= len(bars) or exit_index >= len(bars):
        return None
    entry_bar, exit_bar = bars[entry_index], bars[exit_index]
    entry = _price(entry_bar, "open") or _price(entry_bar, "close")
    exit_price = _price(exit_bar, "close")
    if entry is None or exit_price is None:
        return None
    ret = exit_price / entry - 1.0
    if not (-0.95 < ret < 5.0):
        return None
    return Forward(
        entry_index=entry_index,
        exit_index=exit_index,
        entry_date=str(entry_bar.get("date") or entry_index),
        exit_date=str(exit_bar.get("date") or exit_index),
        entry_price=entry,
        exit_price=exit_price,
        ret=ret,
    )


def forward_returns(bars: Sequence[Bar], horizon: int) -> list[Optional[Forward]]:
    return [forward_return(bars, i, horizon) for i in range(len(bars))]


def _reshuffle(series: Sequence[Optional[float]], *, block: int, rng: random.Random) -> list[Optional[float]]:
    """Block-shuffle the defined values of a series, keeping the gaps in place."""
    filled = [v for v in series if v is not None]
    shuffled = stats.block_shuffle(filled, block=block, rng=rng)
    out: list[Optional[float]] = []
    cursor = 0
    for value in series:
        if value is None:
            out.append(None)
        else:
            out.append(shuffled[cursor])
            cursor += 1
    return out


# ── measurement ─────────────────────────────────────────────────────────────


@dataclass
class Measurement:
    signal: str
    horizon: int
    events: int = 0
    symbols: int = 0
    event_dates: int = 0
    hit_rate: Optional[float] = None
    hit_ci: Optional[tuple[float, float]] = None
    base_rate: Optional[float] = None
    excess_hit: Optional[float] = None
    mean_return: Optional[float] = None
    mean_excess: Optional[float] = None
    median_excess: Optional[float] = None
    excess_stdev: Optional[float] = None
    p_hit: Optional[float] = None
    p_mean: Optional[float] = None
    q_hit: Optional[float] = None
    q_mean: Optional[float] = None
    first_event: Optional[str] = None
    last_event: Optional[str] = None
    best_symbols: list[dict[str, Any]] = field(default_factory=list)
    worst_symbols: list[dict[str, Any]] = field(default_factory=list)
    verdict: str = "UNMEASURED"

    def as_dict(self) -> dict[str, Any]:
        spec = None
        try:
            spec = get_signal(self.signal)
        except KeyError:
            pass
        return {
            "signal": self.signal,
            "title": spec.title if spec else self.signal,
            "hypothesis": spec.hypothesis if spec else "",
            "kind": spec.kind if spec else "unknown",
            "horizon": self.horizon,
            "events": self.events,
            "symbols": self.symbols,
            "event_dates": self.event_dates,
            "hit_rate": _r(self.hit_rate),
            "hit_rate_ci95": [_r(self.hit_ci[0]), _r(self.hit_ci[1])] if self.hit_ci else None,
            "base_rate": _r(self.base_rate),
            "excess_hit": _r(self.excess_hit),
            "mean_return": _r(self.mean_return, 4),
            "mean_excess": _r(self.mean_excess, 4),
            "median_excess": _r(self.median_excess, 4),
            "excess_stdev": _r(self.excess_stdev, 4),
            "p_value_hit": _r(self.p_hit, 4),
            "p_value_mean": _r(self.p_mean, 4),
            "q_value_hit": _r(self.q_hit, 4),
            "q_value_mean": _r(self.q_mean, 4),
            "first_event": self.first_event,
            "last_event": self.last_event,
            "top_symbols": self.best_symbols,
            "bottom_symbols": self.worst_symbols,
            "verdict": self.verdict,
        }


def _r(value: Optional[float], digits: int = 5) -> Optional[float]:
    return None if value is None else round(value, digits)


@dataclass
class Sample:
    """A universe of symbols prepared once and reused across signals."""

    bars_by_symbol: dict[str, list[Bar]]
    horizon: int
    params: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    _contexts: dict[str, SignalContext] = field(default_factory=dict, repr=False)
    _forwards: dict[str, list[Optional[Forward]]] = field(default_factory=dict, repr=False)

    def context(self, symbol: str) -> SignalContext:
        ctx = self._contexts.get(symbol)
        if ctx is None:
            ctx = SignalContext(self.bars_by_symbol[symbol], params=self.params)
            self._contexts[symbol] = ctx
        return ctx

    def forwards(self, symbol: str) -> list[Optional[Forward]]:
        series = self._forwards.get(symbol)
        if series is None:
            series = forward_returns(self.bars_by_symbol[symbol], self.horizon)
            self._forwards[symbol] = series
        return series


def measure(
    sample: Sample,
    signal: str,
    *,
    permutations: int = 500,
    block: int = 10,
    seed: int = 0,
    min_events_for_p: int = 8,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> Measurement:
    """Measure one signal's outcome statistics over a prepared sample."""
    spec = get_signal(signal)
    horizon = sample.horizon
    result = Measurement(signal=signal, horizon=horizon)

    base_returns: list[float] = []
    returns_by_window: dict[tuple[str, str], list[float]] = defaultdict(list)
    full_series: dict[str, list[Optional[float]]] = {}
    positions_by_symbol: dict[str, list[int]] = {}
    events: list[tuple[str, Forward]] = []

    for symbol, bars in sample.bars_by_symbol.items():
        if not bars:
            continue
        ctx = sample.context(symbol)
        forwards = sample.forwards(symbol)
        series: list[Optional[float]] = []
        positions: list[int] = []
        for i, fwd in enumerate(forwards):
            series.append(fwd.ret if fwd else None)
            if fwd is None:
                continue
            returns_by_window[(fwd.entry_date, fwd.exit_date)].append(fwd.ret)
            if date_start and fwd.entry_date < date_start:
                continue
            if date_end and fwd.exit_date > date_end:
                continue
            base_returns.append(fwd.ret)
            if i + 1 < spec.min_history:
                continue
            if spec.fn(ctx, i):
                positions.append(i)
                events.append((symbol, fwd))
        full_series[symbol] = series
        if positions:
            positions_by_symbol[symbol] = positions

    result.base_rate = stats.hit_rate(base_returns)
    if not events:
        result.verdict = "NO_EVENTS"
        return result

    benchmark = {key: stats.mean(vals) for key, vals in returns_by_window.items() if vals}
    returns = [fwd.ret for _, fwd in events]
    excesses = []
    for _, fwd in events:
        b = benchmark.get((fwd.entry_date, fwd.exit_date))
        excesses.append(fwd.ret - b if b is not None else 0.0)

    result.events = len(returns)
    result.event_dates = len({fwd.entry_date for _, fwd in events})
    result.symbols = len(positions_by_symbol)
    result.hit_rate = stats.hit_rate(returns)
    wins = sum(1 for r in returns if r > 0)
    result.hit_ci = stats.wilson_interval(wins, len(returns))
    result.excess_hit = (
        None if result.hit_rate is None or result.base_rate is None else result.hit_rate - result.base_rate
    )
    result.mean_return = stats.mean(returns)
    result.mean_excess = stats.mean(excesses)
    result.median_excess = stats.median(excesses)
    result.excess_stdev = stats.stdev(excesses)
    dates = [fwd.entry_date for _, fwd in events]
    result.first_event, result.last_event = min(dates), max(dates)

    by_symbol: dict[str, list[float]] = defaultdict(list)
    for symbol, fwd in events:
        by_symbol[symbol].append(fwd.ret)
    summary = [
        {
            "symbol": symbol,
            "events": len(rets),
            "hit_rate": _r(stats.hit_rate(rets)),
            "mean_return": _r(stats.mean(rets), 4),
        }
        for symbol, rets in by_symbol.items()
    ]
    summary.sort(key=lambda row: ((row["hit_rate"] or 0), row["events"]), reverse=True)
    result.best_symbols = summary[:5]
    result.worst_symbols = sorted(
        summary,
        key=lambda row: (row["hit_rate"] if row["hit_rate"] is not None else 1, -row["events"]),
    )[:5]

    if result.events >= min_events_for_p and permutations > 0:
        result.p_hit = _pooled_permutation_p(
            full_series=full_series,
            positions_by_symbol=positions_by_symbol,
            stat=stats.hit_rate,
            observed=result.hit_rate,
            block=block,
            permutations=permutations,
            seed=seed,
        )
        result.p_mean = _pooled_permutation_p(
            full_series=full_series,
            positions_by_symbol=positions_by_symbol,
            stat=stats.mean,
            observed=result.mean_return,
            block=block,
            permutations=permutations,
            seed=seed + 1,
        )

    result.verdict = call_it(result)
    return result


def _pooled_permutation_p(
    *,
    full_series: Mapping[str, Sequence[Optional[float]]],
    positions_by_symbol: Mapping[str, Sequence[int]],
    stat: Callable[[Sequence[float]], Optional[float]],
    observed: Optional[float],
    block: int,
    permutations: int,
    seed: int,
) -> Optional[float]:
    """Permutation p over pooled events, rotating each symbol's whole series.

    The null: within each symbol's own return sequence, the dates the signal
    picked carry no information beyond that sequence's local autocorrelation.
    Rotating the *whole* series (not just the event values) is what makes the
    comparison meaningful — the shuffled statistic is computed at the same
    positions from the same distribution of returns.
    """
    if observed is None or not positions_by_symbol:
        return None
    rng = random.Random(seed)
    rounds = max(1, permutations)
    hits = 0
    for _ in range(rounds):
        pooled: list[float] = []
        for symbol, positions in positions_by_symbol.items():
            shuffled = _reshuffle(full_series[symbol], block=block, rng=rng)
            pooled.extend(shuffled[i] for i in positions if shuffled[i] is not None)
        got = stat(pooled)
        if got is not None and got >= observed:
            hits += 1
    return (1 + hits) / (1 + rounds)


# ── verdicts ────────────────────────────────────────────────────────────────


def call_it(m: Measurement, *, min_events: int = 30, alpha: float = 0.10) -> str:
    """Turn a measurement into a declared state — no hedge words.

    VALIDATED  : at least ``min_events`` observations, positive excess hit rate,
                 and q ≤ ``alpha`` after multiple-testing correction.
    SUGGESTIVE : right direction, not enough evidence yet (few events, or q
                 still above alpha).
    REJECTED   : the sign is wrong, or the effect is indistinguishable from the
                 base rate.
    """
    if m.events == 0:
        return "NO_EVENTS"
    if (m.excess_hit or 0) <= 0:
        return "REJECTED"
    if m.events < min_events:
        return "SUGGESTIVE"
    q = m.q_hit if m.q_hit is not None else m.p_hit
    if q is None:
        return "SUGGESTIVE"
    return "VALIDATED" if q <= alpha else "SUGGESTIVE"


def apply_fdr(measurements: Sequence[Measurement], *, on: str = "hit") -> None:
    """Set q-values across a batch of measurements, in place."""
    field_name = "q_hit" if on == "hit" else "q_mean"
    source = "p_hit" if on == "hit" else "p_mean"
    qs = stats.benjamini_hochberg([getattr(m, source) for m in measurements])
    for m, q in zip(measurements, qs):
        setattr(m, field_name, q)


def measure_all(
    sample: Sample,
    *,
    signals: Optional[Iterable[str]] = None,
    permutations: int = 500,
    block: int = 10,
    seed: int = 0,
) -> list[Measurement]:
    """Measure a batch of signals, then correct p-values across the batch."""
    names = list(signals) if signals else signal_names()
    out = [
        measure(sample, name, permutations=permutations, block=block, seed=seed + 7 * k)
        for k, name in enumerate(names)
    ]
    apply_fdr(out, on="hit")
    apply_fdr(out, on="mean")
    for m in out:
        m.verdict = call_it(m)
    return out


# ── walk-forward ────────────────────────────────────────────────────────────


@dataclass
class Fold:
    index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train: Measurement
    test: Measurement

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.index,
            "train_window": [self.train_start, self.train_end],
            "test_window": [self.test_start, self.test_end],
            "train_events": self.train.events,
            "train_hit_rate": _r(self.train.hit_rate),
            "train_excess_hit": _r(self.train.excess_hit),
            "train_mean_excess": _r(self.train.mean_excess, 4),
            "test_events": self.test.events,
            "test_hit_rate": _r(self.test.hit_rate),
            "test_excess_hit": _r(self.test.excess_hit),
            "test_mean_excess": _r(self.test.mean_excess, 4),
            "test_hit_rate_ci95": (
                [_r(self.test.hit_ci[0]), _r(self.test.hit_ci[1])] if self.test.hit_ci else None
            ),
        }


@dataclass
class WalkForward:
    signal: str
    horizon: int
    folds: list[Fold]
    retention: Optional[float]
    verdict: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "horizon": self.horizon,
            "folds": [f.as_dict() for f in self.folds],
            "retention": _r(self.retention),
            "retention_basis": "pooled out-of-sample mean excess / pooled in-sample mean excess",
            "verdict": self.verdict,
            "reason": self.reason,
        }


MIN_SESSIONS_PER_FOLD = 20


def session_axis(sample: Sample) -> list[str]:
    dates: set[str] = set()
    for bars in sample.bars_by_symbol.values():
        for bar in bars:
            date = bar.get("date")
            if date:
                dates.add(str(date))
    return sorted(dates)


def walk_forward(
    sample: Sample,
    signal: str,
    *,
    folds: int = 4,
    seed: int = 0,
    permutations: int = 0,
    min_test_events: int = 10,
    min_train_events: int = 30,
    min_train_excess: float = 0.002,
) -> WalkForward:
    """Measure the signal in-sample and out-of-sample across contiguous folds.

    Fold *k* is trained on everything up to its start and tested on the segment
    that follows; an event can never be counted in both, because the train side
    requires its exit before the fold boundary and the test side requires its
    entry after it. Folds whose test window produced fewer than
    ``min_test_events`` events do not vote — three observations are noise in
    both directions, and saying so is more useful than a verdict built on them.
    """
    if folds < 2:
        raise ValueError("walk-forward needs at least 2 folds")
    axis = session_axis(sample)
    if len(axis) < folds * MIN_SESSIONS_PER_FOLD:
        return WalkForward(
            signal, sample.horizon, [], None, "INSUFFICIENT_HISTORY",
            f"only {len(axis)} sessions available; {folds} usable folds need at least "
            f"{folds * MIN_SESSIONS_PER_FOLD}",
        )
    bounds = [axis[min(len(axis) - 1, int(len(axis) * k / (folds + 1)))] for k in range(folds + 2)]

    out: list[Fold] = []
    for k in range(1, folds + 1):
        train = measure(sample, signal, permutations=permutations, seed=seed + k, date_end=bounds[k])
        test = measure(
            sample,
            signal,
            permutations=permutations,
            seed=seed + 100 + k,
            date_start=bounds[k],
            date_end=bounds[k + 1],
        )
        out.append(
            Fold(
                index=k,
                train_start=bounds[0],
                train_end=bounds[k],
                test_start=bounds[k],
                test_end=bounds[k + 1],
                train=train,
                test=test,
            )
        )

    # Only folds with enough evidence on both sides vote, and the in-sample edge
    # must clear BOTH a small absolute floor and one standard error of its own
    # sampling distribution. Dividing a near-zero in-sample excess into an
    # out-of-sample one produces ratios of 3x or 8x out of pure noise, which is
    # exactly how a curve fit gets reported as ROBUST.
    def _gated(fold: Fold) -> bool:
        if fold.test.events < min_test_events or fold.train.events < min_train_events:
            return False
        mean_excess = fold.train.mean_excess
        if mean_excess is None:
            return False
        se = (fold.train.excess_stdev or 0.0) / math.sqrt(max(1, fold.train.events))
        return mean_excess >= min_train_excess and mean_excess >= se

    usable = [f for f in out if _gated(f)]
    if not usable:
        return WalkForward(
            signal, sample.horizon, out, None, "NO_TRAIN_EDGE",
            "no fold had at least "
            f"{min_train_events} in-sample events with a mean excess above both "
            f"{min_train_excess:.1%} and one standard error, alongside "
            f"{min_test_events}+ out-of-sample events",
        )
    # Retention is the pooled ratio of MEAN excess, not of hit-rate excess and
    # not the mean of per-fold ratios. Two reasons, both measured on the
    # planted-footprint control: a hit-rate lift of a few points has a large
    # sampling variance, so its per-fold ratios swung between 0.19 and 0.83 on
    # one dataset, while the mean excess held its sign in every fold; and
    # averaging ratios weights a fold by nothing but its denominator. Pooling
    # the mean excess weights every fold by the edge it actually measured.
    train_edges = [f.train.mean_excess for f in usable if f.train.mean_excess is not None]
    test_edges = [f.test.mean_excess for f in usable if f.test.mean_excess is not None]
    total_train = sum(train_edges)
    retention = (sum(test_edges) / total_train) if total_train else None
    if retention is None:
        return WalkForward(signal, sample.horizon, out, None, "NO_TRAIN_EDGE", "retention not computable")
    if retention >= 0.8:
        verdict, reason = "ROBUST", "out-of-sample excess retains at least 80% of the in-sample edge"
    elif retention >= 0.5:
        verdict, reason = "MODERATE", "out-of-sample excess retains 50-80% of the in-sample edge"
    elif retention >= 0.2:
        verdict, reason = "WEAK", "most of the in-sample edge does not survive out-of-sample"
    else:
        verdict, reason = "OVERFITTED", "out-of-sample excess is under 20% of the in-sample edge"
    return WalkForward(signal, sample.horizon, out, retention, verdict, reason)
