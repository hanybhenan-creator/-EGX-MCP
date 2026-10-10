"""Does the "volume up, price flat" footprint actually predict anything?

This is the scientific claim at the centre of the design document: institutions
accumulating cannot hide the volume, only the price, so a session trading far
above its average volume while closing nearly flat is a trace of accumulation —
and it is supposed to precede gains.

The test plants that behaviour (volume spikes on flat candles, followed by
drift) and requires the calibration engine to find it, then removes it and
requires the engine to reject it. A pipeline that cannot make this distinction
would report "VALIDATED" for any pattern, which is exactly the failure mode
this project exists to remove.
"""

from __future__ import annotations

import math
import random

import pytest

from tradingview_mcp.core.calibration import Sample, analyze, measure


def footprint_universe(n: int = 14, sessions: int = 380, *, plant: bool, seed: int = 0):
    """Synthetic EGX-like history. When ``plant`` is true, flat candles carry
    institutional volume and are followed by upward drift."""
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {}
    for s in range(n):
        close, drift_boost, bars = 30.0, 0.0, []
        for i in range(sessions):
            close = max(1.0, close * (1 + rng.gauss(0.0004 + drift_boost, 0.017)))
            open_ = close * (1 + rng.gauss(0, 0.003))
            high = max(close * (1 + abs(rng.gauss(0, 0.006))), open_, close)
            low = min(close * (1 - abs(rng.gauss(0, 0.006))), open_, close)
            quiet = abs(close / open_ - 1) < 0.012
            planted = plant and quiet and rng.random() < 0.25
            bars.append(
                {
                    "date": f"{2023 + i // 252}-{(i // 21) % 12 + 1:02d}-{(i % 21) + 1:02d}",
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": math.exp(rng.gauss(13, 0.4)) * (2.7 if planted else 1.0),
                }
            )
            if planted:
                drift_boost = 0.007
            elif drift_boost > 0:
                drift_boost = max(0.0, drift_boost - 0.0024)
        out[f"S{s:02d}"] = bars
    return out


@pytest.fixture(scope="module")
def planted_report():
    sample = Sample(footprint_universe(plant=True, seed=4), horizon=5)
    return analyze(
        sample,
        signals=["hidden_accumulation", "volume_footprint"],
        permutations=150,
        walk_forward_folds=3,
        seed=9,
    )


@pytest.fixture(scope="module")
def noise_report():
    sample = Sample(footprint_universe(plant=False, seed=4), horizon=5)
    return analyze(
        sample,
        signals=["hidden_accumulation", "volume_footprint"],
        permutations=150,
        walk_forward_folds=3,
        seed=9,
    )


class TestPlantedFootprint:
    def test_hidden_accumulation_fires_often_enough_to_measure(self, planted_report) -> None:
        m = planted_report.measurements[0]
        assert m.signal == "hidden_accumulation"
        assert m.events > 200
        assert m.event_dates > 100

    def test_the_edge_is_found_and_survives_correction(self, planted_report) -> None:
        m = planted_report.measurements[0]
        assert m.excess_hit is not None and m.excess_hit > 0.03
        assert m.p_hit is not None and m.p_hit < 0.05
        assert m.q_hit is not None and m.q_hit <= 0.10
        assert m.verdict == "VALIDATED"

    def test_the_edge_survives_out_of_sample(self, planted_report) -> None:
        wf = planted_report.walk_forwards["hidden_accumulation"]
        assert wf.verdict in {"ROBUST", "MODERATE"}
        assert wf.retention is not None and wf.retention > 0.5

    def test_the_looser_threshold_pair_also_works(self, planted_report) -> None:
        m = planted_report.measurements[1]
        assert m.signal == "volume_footprint"
        assert m.verdict == "VALIDATED"


class TestNoise:
    def test_the_footprint_is_not_a_winner_on_random_data(self, noise_report) -> None:
        m = noise_report.measurements[0]
        assert m.events > 100
        assert m.verdict != "VALIDATED"
        assert m.p_hit is None or m.p_hit > 0.05

    def test_the_engine_does_not_reward_a_signal_that_fires_everywhere(self, noise_report) -> None:
        # Both footprint definitions fire thousands of times on noise; a hit
        # rate close to the base rate must come back as REJECTED, not as a
        # "62% win rate" headline.
        for m in noise_report.measurements:
            if m.hit_rate is not None and m.base_rate is not None:
                assert abs(m.hit_rate - m.base_rate) < 0.06


class TestThresholdSemantics:
    def test_hidden_requires_both_conditions(self) -> None:
        from tradingview_mcp.core.analysis.footprints import hidden_accumulation_flags
        from tests.unit.calibration.conftest import make_bars

        quiet_big_volume = make_bars([10.0] * 25, volumes=[1000.0] * 24 + [3000.0], opens=[10.0] * 25)
        assert hidden_accumulation_flags(quiet_big_volume, lookback=20)[24] is True

        loud_small_volume = make_bars([10.0] * 24 + [10.6], volumes=[1000.0] * 25, opens=[10.0] * 25)
        assert hidden_accumulation_flags(loud_small_volume, lookback=20)[24] is False

    def test_measure_carries_the_payload_fields(self) -> None:
        sample = Sample(footprint_universe(n=4, sessions=140, plant=True, seed=1), horizon=3)
        payload = measure(sample, "hidden_accumulation", permutations=20, seed=2).as_dict()
        assert payload["kind"] == "condition"
        assert "volume" in payload["hypothesis"]
        assert payload["title"]
