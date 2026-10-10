"""Synthetic market builders shared by the calibration tests.

Everything is deterministic: same seed, same bars, same verdict. Tests that
would otherwise be flaky (p-values, walk-forward retention) are seeded on
purpose and assert on the mechanism, not on a number that only holds today.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Optional

import pytest


def make_bars(
    closes: list[float],
    *,
    volumes: Optional[list[float]] = None,
    highs: Optional[list[float]] = None,
    lows: Optional[list[float]] = None,
    opens: Optional[list[float]] = None,
    start_day: int = 1,
) -> list[dict]:
    """Build bar dicts with explicit OHLCV values (dates just need to be unique)."""
    out = []
    for i, close in enumerate(closes):
        high = highs[i] if highs else close * 1.01
        low = lows[i] if lows else close * 0.99
        out.append(
            {
                "date": f"2024-{(start_day + i) // 28 + 1:02d}-{(start_day + i) % 28 + 1:02d}",
                "open": opens[i] if opens else close,
                "high": high,
                "low": low,
                "close": close,
                "volume": volumes[i] if volumes else 1000.0,
            }
        )
    return out


def random_walk(
    sessions: int = 300,
    *,
    seed: int = 0,
    start: float = 30.0,
    drift: float = 0.0004,
    sigma: float = 0.018,
    boost_after_breakout: float = 0.0,
    breakout_window: int = 20,
) -> list[dict]:
    """A synthetic daily series.

    ``boost_after_breakout`` plants a real edge: every time the close breaks the
    previous ``breakout_window`` sessions' high, the following five sessions get
    ``boost`` added to their drift. The calibration engine is expected to find
    it — that is the positive control for the whole pipeline.
    """
    rng = random.Random(seed)
    bars: list[dict] = []
    close = start
    highs: deque[float] = deque(maxlen=breakout_window + 1)
    boost_left = 0
    for i in range(sessions):
        extra = boost_after_breakout if boost_left > 0 else 0.0
        close = max(1.0, close * (1 + rng.gauss(drift + extra, sigma)))
        open_ = close * (1 + rng.gauss(0, 0.004))
        # Valid OHLC by construction: the data layer's structural checks
        # (high >= max(open, close), low <= min(open, close)) are part of the
        # contract, and fixtures that violate them would make the quality
        # scorer report fake problems.
        high = max(close * (1 + abs(rng.gauss(0, 0.007))), open_, close)
        low = min(close * (1 - abs(rng.gauss(0, 0.007))), open_, close)
        bars.append(
            {
                "date": f"2024-{(i // 21) + 1:02d}-{(i % 21) + 1:02d}",
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": math.exp(rng.gauss(13, 0.6)),
            }
        )
        if len(highs) == breakout_window + 1 and close > max(highs):
            boost_left = 5
        if boost_left > 0:
            boost_left -= 1
        highs.append(high)
    return bars


@pytest.fixture
def seeded_universe():
    """A small, fast universe for measurement tests (10 symbols x 240 sessions)."""

    def build(*, smart: int = 0, seed: int = 0, sessions: int = 240, n: int = 10):
        return {
            f"S{s:02d}": random_walk(
                sessions, seed=seed * 100 + s, boost_after_breakout=0.006 if s < smart else 0.0
            )
            for s in range(n)
        }

    return build
