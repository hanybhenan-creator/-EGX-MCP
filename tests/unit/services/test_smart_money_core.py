"""Regression tests for the accumulation core's price-structure condition.

Historically ``is_above_20d_high`` compared the current close against the
highest high of a window that *included* the current bar. A close can never
exceed its own session's high, so the condition was unsatisfiable: measured
over 4,800 synthetic sessions it fired zero times, which silently made the
"5/5 conditions" branch of :class:`AccumulationSignal` dead code. These tests
pin the corrected behaviour — and the fact that the current bar is excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingview_mcp.core.services.egx_smart_money_core import (
    AccumulationSignalGenerator,
    PriceStructureAnalyzer,
)


def frame(closes, highs=None, lows=None, volumes=None) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs if highs is not None else [c * 1.01 for c in closes],
            "low": lows if lows is not None else [c * 0.99 for c in closes],
            "close": closes,
            "volume": volumes if volumes is not None else [1000.0] * n,
        }
    )


class TestPriceStructureWindow:
    def test_breakout_is_detected(self) -> None:
        # 25 flat sessions at 10, then a session that closes above every prior bar
        closes = [10.0] * 25 + [10.8]
        highs = [10.1] * 25 + [10.9]
        price = PriceStructureAnalyzer.calculate_price_structure(frame(closes, highs))
        assert bool(price.is_above_20d_high) is True
        assert price.highest_high_20d == pytest.approx(10.1)

    def test_current_bar_is_excluded_from_the_extremes(self) -> None:
        # Today's high is the highest of the whole frame, but today's close is
        # below the previous window's high: not a breakout.
        closes = [10.0] * 24 + [10.0, 10.2, 10.05]
        highs = [10.1] * 25 + [10.5, 10.5]
        price = PriceStructureAnalyzer.calculate_price_structure(frame(closes, highs))
        assert bool(price.is_above_20d_high) is False

    def test_extremes_never_include_the_last_bar(self) -> None:
        closes = [10.0] * 30 + [50.0]
        highs = [10.1] * 30 + [60.0]
        lows = [9.9] * 30 + [1.0]
        price = PriceStructureAnalyzer.calculate_price_structure(frame(closes, highs, lows))
        assert price.highest_high_20d == pytest.approx(10.1)
        assert price.lowest_low_20d == pytest.approx(9.9)

    def test_swing_levels_come_from_the_previous_window(self) -> None:
        highs = [10.0] * 20 + [99.0]
        lows = [9.0] * 20 + [0.5]
        closes = [9.5] * 20 + [50.0]
        swing_high, swing_low = PriceStructureAnalyzer.find_swing_levels(frame(closes, highs, lows))
        assert swing_high == pytest.approx(10.0)
        assert swing_low == pytest.approx(9.0)

    def test_single_bar_frame_does_not_crash(self) -> None:
        price = PriceStructureAnalyzer.calculate_price_structure(frame([10.0]))
        assert price.current_price == 10.0

    def test_breakout_range_still_requires_proximity_to_resistance(self) -> None:
        closes = [10.0] * 25 + [10.05]
        price = PriceStructureAnalyzer.calculate_price_structure(frame(closes, [10.1] * 26))
        assert bool(price.is_above_20d_high) is False  # 10.05 < 10.1
        assert bool(price.is_in_breakout_range) is False


class TestFiveConditionsAreReachable:
    def test_a_strong_uptrend_produces_a_five_of_five_signal(self) -> None:
        rng = np.random.default_rng(3)
        n = 260
        steps = rng.normal(0.004, 0.012, n)  # steady uptrend with noise
        closes = list(30.0 * np.exp(np.cumsum(steps)))
        highs = [c * (1 + abs(rng.normal(0, 0.004))) for c in closes]
        lows = [c * (1 - abs(rng.normal(0, 0.004))) for c in closes]
        volumes = rng.lognormal(13, 0.3, n)
        df = frame(closes, highs, lows, list(volumes))

        best = 0
        for end in range(80, n, 4):
            signal = AccumulationSignalGenerator.generate_signal(df.iloc[: end + 1], "TEST", 20)
            if signal is None:
                continue
            best = max(
                best,
                sum(
                    [
                        signal.volume_profile_bullish,
                        signal.order_flow_buying,
                        signal.price_structure_bullish,
                        signal.obv_confirmed,
                        signal.adl_confirmed,
                    ]
                ),
            )
        assert best == 5, "the breakout condition must be satisfiable on a trending series"

    def test_signal_strength_is_a_declared_share_of_conditions(self) -> None:
        rng = np.random.default_rng(4)
        n = 200
        closes = list(30.0 * np.exp(np.cumsum(rng.normal(0.003, 0.012, n))))
        df = frame(closes, [c * 1.005 for c in closes], [c * 0.995 for c in closes])
        seen = False
        for end in range(80, n, 4):
            signal = AccumulationSignalGenerator.generate_signal(df.iloc[: end + 1], "TEST", 20)
            if signal is None:
                continue
            seen = True
            met = sum(
                [
                    signal.volume_profile_bullish,
                    signal.order_flow_buying,
                    signal.price_structure_bullish,
                    signal.obv_confirmed,
                    signal.adl_confirmed,
                ]
            )
            assert met >= 4
            assert signal.signal_strength == int(met / 5 * 100)
            assert bool(signal.is_valid()) is True
        assert seen, "the generator should fire at least once on a trending series"
