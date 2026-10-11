"""Synthetic sector universes with a known answer.

Two controls, used by both the unit tests and the manual checks:

* ``planted``  — each sector carries a slowly varying drift (an AR(1) latent
  state, stationary s.d. ≈ 4.1 × ``drift_scale`` per session). A sector that has
  been rising keeps rising for a while, so trailing returns should predict the
  next horizon's excess return. The default (0.0005, i.e. ~0.2% per session of
  latent s.d.) is near the smallest effect this design reliably detects on 420
  sessions with 6 sectors × 4 members; smaller effects are reported as
  not distinguishable from zero, which is the honest answer.
* ``noise``    — the same market and idiosyncratic structure with the drift
  removed. Nothing in the trailing data should predict anything.

Symbols are named ``<SECTOR>_<k>`` and mapped to sectors through ``sector_of``,
so no real EGX membership is involved.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import Any


def sector_universe(
    *,
    planted: bool,
    sectors: int = 6,
    members: int = 4,
    sessions: int = 420,
    seed: int = 7,
    drift_scale: float = 0.0005,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    rng = random.Random(seed)
    start = date(2022, 1, 3)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(sessions)]

    latent = [0.0] * sectors
    bars: dict[str, list[dict[str, Any]]] = {}
    sector_of: dict[str, str] = {}
    prices = {f"S{k}_{j}": 20.0 + rng.random() * 30 for k in range(sectors) for j in range(members)}
    last_close = dict(prices)
    for t in range(sessions):
        market = rng.gauss(0.0003, 0.010)
        for k in range(sectors):
            if planted:
                latent[k] = 0.97 * latent[k] + rng.gauss(0, drift_scale)
            else:
                latent[k] = 0.0
        for k in range(sectors):
            for j in range(members):
                sym = f"S{k}_{j}"
                sector_of[sym] = f"sector_{k}"
                prev = last_close[sym]
                gap = rng.gauss(0, 0.003)
                open_ = prev * (1 + gap)
                ret = market + latent[k] + rng.gauss(0, 0.015)
                close = open_ * (1 + ret)
                volume = math.exp(rng.gauss(12, 0.35)) * (1 + 3 * abs(ret))
                bars.setdefault(sym, []).append(
                    {
                        "date": dates[t],
                        "open": round(open_, 4),
                        "high": round(max(open_, close) * 1.004, 4),
                        "low": round(min(open_, close) * 0.996, 4),
                        "close": round(close, 4),
                        "volume": round(volume),
                    }
                )
                last_close[sym] = close
    return bars, sector_of
