"""Synthetic OHLCV universe for the Wyckoff event controls.

Each session has a probability of receiving an injected selling-climax bar: a
new 20-session low, on about 2.5× average volume, closing in the upper half of
its range. In the planted universe, the next ``drift_sessions`` sessions carry
an extra daily drift. In the noise universe the same bars are injected with no
drift afterwards.

The controls answer one question: when the injected pattern is followed by a
real effect, does the calibration engine see it, and when it is not, does the
engine refrain from calling it VALIDATED? They say nothing about whether the
pattern exists in EGX data.
"""

from __future__ import annotations

import math
import random
from statistics import fmean
from typing import Any


def wyckoff_universe(
    n: int = 14,
    sessions: int = 380,
    *,
    plant: bool,
    seed: int = 0,
    event_rate: float = 0.03,
    drift: float = 0.0025,
    drift_sessions: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    out: dict[str, list[dict[str, Any]]] = {}
    for s in range(n):
        close = 30.0
        bars: list[dict[str, Any]] = []
        boost_left = 0
        for i in range(sessions):
            day = {
                "date": f"{2023 + i // 252}-{(i // 21) % 12 + 1:02d}-{(i % 21) + 1:02d}",
            }
            if i >= 25 and rng.random() < event_rate:
                prior = bars[i - 20 : i]
                support = min(b["low"] for b in prior)
                avg_volume = fmean(b["volume"] for b in prior)
                open_ = close * (1 + rng.gauss(0, 0.002))
                low = support * 0.985
                high = max(open_, close) * 1.003
                close_new = low + 0.72 * (high - low)
                close_new = min(close_new, open_ * 0.998)
                day.update(
                    open=open_,
                    high=high,
                    low=min(low, close_new, open_),
                    close=close_new,
                    volume=avg_volume * 2.5,
                )
                close = close_new
                if plant:
                    boost_left = drift_sessions
            else:
                step = rng.gauss(0.0004 + (drift if boost_left > 0 else 0.0), 0.017)
                boost_left = max(0, boost_left - 1)
                close = max(1.0, close * (1 + step))
                open_ = close * (1 + rng.gauss(0, 0.003))
                high = max(close * (1 + abs(rng.gauss(0, 0.006))), open_, close)
                low = min(close * (1 - abs(rng.gauss(0, 0.006))), open_, close)
                day.update(
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=math.exp(rng.gauss(13, 0.4)),
                )
            bars.append(day)
        out[f"W{s:02d}"] = bars
    return out
