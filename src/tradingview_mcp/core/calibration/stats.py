"""Statistical primitives for calibrated signal measurement.

The point of this module is that a number is not a claim. A hit rate of 62%
means something only next to (a) the rate you would have got by not using the
signal, (b) an interval wide enough to say how much of that 62% is noise, and
(c) a p-value from a null hypothesis that respects the way daily equity
returns actually behave.

Nothing here assumes normality. EGX daily returns are fat-tailed, volatility
clusters, and — because Egyptian sessions are one market — events on the same
date across different symbols move together. Textbook t-tests overstate
significance under those conditions, so inference goes through a **circular
block permutation**: the return series is rotated in blocks of consecutive
sessions, which preserves the local autocorrelation (momentum, volatility
clustering) while destroying the alignment between the signal's dates and the
returns. The p-value that comes out answers exactly the question that matters:
*given this sequence of returns, how surprising is the signal's timing?*
"""

from __future__ import annotations

import math
import random
from typing import Callable, Optional, Sequence

# ── descriptive ─────────────────────────────────────────────────────────────


def mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def stdev(values: Sequence[float]) -> Optional[float]:
    """Sample standard deviation (n-1). None when n < 2."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var)


def median(values: Sequence[float]) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolation quantile (numpy's default method)."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def percentile_rank(values: Sequence[float], value: float) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v <= value) / len(vals)


def hit_rate(values: Sequence[float], *, threshold: float = 0.0) -> Optional[float]:
    """Share of observations strictly above ``threshold`` (default: positive)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v > threshold) / len(vals)


def describe(values: Sequence[float]) -> dict[str, Optional[float]]:
    vals = [v for v in values if v is not None]
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "stdev": stdev(vals),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "p25": quantile(vals, 0.25),
        "p75": quantile(vals, 0.75),
        "positive_share": hit_rate(vals),
    }


# ── intervals ───────────────────────────────────────────────────────────────


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> Optional[tuple[float, float]]:
    """Wilson score interval for a binomial proportion (statsmodels).

    Preferred over the normal approximation for the sample sizes this project
    actually sees: a signal that fired 12 times with 9 wins has a genuine
    uncertainty that ``p ± 1.96·sqrt(p(1-p)/n)`` hides (it can even cross 1.0).
    ``z`` is kept for compatibility with earlier callers; ``alpha`` is derived
    from it, so the interval is identical to the textbook formula at that z.
    """
    if n <= 0:
        return None
    from scipy.stats import norm
    from statsmodels.stats.proportion import proportion_confint

    alpha = 2.0 * float(norm.sf(z))
    lo, hi = proportion_confint(successes, n, alpha=alpha, method="wilson")
    # At k=0 or k=n the Wilson bounds are exactly 0 and 1; the library can return
    # 1-1e-16 there, so pin the boundary rather than report float noise.
    lo = 0.0 if successes == 0 else max(0.0, float(lo))
    hi = 1.0 if successes == n else min(1.0, float(hi))
    return (lo, hi)


# ── resampling ──────────────────────────────────────────────────────────────


def block_shuffle(values: Sequence[float], *, block: int, rng: random.Random) -> list[float]:
    """Randomly permute the order of consecutive blocks of one series.

    The series is cut into non-overlapping blocks of ``block`` sessions (the
    last one may be short) and the blocks are reordered. Because it is a
    permutation and not a resampling, the resulting series contains exactly the
    same values as the original — so a p-value computed from it is testing the
    *timing* of the signal, not a different distribution of returns. Runs of
    up to ``block`` consecutive sessions keep their internal order.
    """
    n = len(values)
    if n == 0:
        return []
    b = max(1, min(block, n))
    blocks = [list(values[i : i + b]) for i in range(0, n, b)]
    rng.shuffle(blocks)
    return [value for chunk in blocks for value in chunk]


def permutation_p(
    *,
    values: Sequence[float],
    positions: Sequence[int],
    statistic: Callable[[Sequence[float], Sequence[int]], Optional[float]],
    observed: Optional[float] = None,
    block: int = 10,
    permutations: int = 1000,
    seed: int = 0,
    alternative: str = "greater",
) -> Optional[float]:
    """One-sided permutation p-value for a statistic evaluated at ``positions``.

    ``positions`` are the indices at which the signal fired. The null is that
    the signal's *timing* carries no information about returns: the return
    series is block-shuffled and the same statistic is recomputed at the very
    same positions.

    p = (1 + #{shuffled >= observed}) / (1 + permutations)

    The +1s are the standard correction that keeps the test exact at the
    boundary (a p-value of exactly 0 is not attainable from a finite number of
    permutations).
    """
    vals = list(values)
    pos = [i for i in positions if 0 <= i < len(vals)]
    if not vals or not pos:
        return None
    obs = observed if observed is not None else statistic(vals, pos)
    if obs is None:
        return None
    if alternative not in {"greater", "less"}:
        raise ValueError("alternative must be 'greater' or 'less'")
    rng = random.Random(seed)
    hits = 0
    for _ in range(max(1, permutations)):
        shuffled = block_shuffle(vals, block=block, rng=rng)
        got = statistic(shuffled, pos)
        if got is None:
            continue
        if (got >= obs) if alternative == "greater" else (got <= obs):
            hits += 1
    return (1 + hits) / (1 + max(1, permutations))


def permutation_p_mean(
    values: Sequence[float],
    positions: Sequence[int],
    *,
    observed: Optional[float] = None,
    block: int = 10,
    permutations: int = 1000,
    seed: int = 0,
) -> Optional[float]:
    """Permutation p for the mean return at the signal's positions."""

    def _stat(series: Sequence[float], idx: Sequence[int]) -> Optional[float]:
        return mean([series[i] for i in idx])

    return permutation_p(
        values=values,
        positions=positions,
        statistic=_stat,
        observed=observed,
        block=block,
        permutations=permutations,
        seed=seed,
    )


def permutation_p_hit(
    values: Sequence[float],
    positions: Sequence[int],
    *,
    threshold: float = 0.0,
    observed: Optional[float] = None,
    block: int = 10,
    permutations: int = 1000,
    seed: int = 0,
) -> Optional[float]:
    """Permutation p for the hit rate at the signal's positions."""

    def _stat(series: Sequence[float], idx: Sequence[int]) -> Optional[float]:
        return hit_rate([series[i] for i in idx], threshold=threshold)

    return permutation_p(
        values=values,
        positions=positions,
        statistic=_stat,
        observed=observed,
        block=block,
        permutations=permutations,
        seed=seed,
    )


# ── multiple testing ────────────────────────────────────────────────────────


def benjamini_hochberg(p_values: Sequence[Optional[float]]) -> list[Optional[float]]:
    """Benjamini–Hochberg FDR q-values, order-preserving (statsmodels ``fdr_bh``).

    Testing twenty conditions on one price history will always produce a
    "significant" one by chance. BH turns raw p-values into q-values that can be
    read as "share of false discoveries if I call this set significant". A
    condition with p=0.04 out of twenty tested is not 0.04 of evidence.
    Entries that are None (no test was run) stay None and do not count as tests.
    """
    from statsmodels.stats.multitest import multipletests

    out: list[Optional[float]] = [None] * len(p_values)
    indexed = [(i, float(p)) for i, p in enumerate(p_values) if p is not None]
    if not indexed:
        return out
    _, adjusted, _, _ = multipletests([p for _, p in indexed], method="fdr_bh")
    for (i, _), q in zip(indexed, adjusted):
        out[i] = min(1.0, float(q))
    return out


def z_score(observed: float, null: Sequence[float]) -> Optional[float]:
    """How many null-standard-deviations the observed value sits from the null mean."""
    mu = mean(null)
    sd = stdev(null)
    if mu is None or sd in (None, 0):
        return None
    return (observed - mu) / sd
