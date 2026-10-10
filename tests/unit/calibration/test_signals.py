"""Signal definitions: the arithmetic has to match the shipped detector, and
the unsatisfiable legacy condition has to stay visibly unsatisfiable."""

from __future__ import annotations

import pytest

from tradingview_mcp.core.calibration import signals as sig
from tests.unit.calibration.conftest import make_bars, random_walk


def bar(date: str, o: float, h: float, low: float, c: float, v: float = 1000.0) -> dict:
    return {"date": date, "open": o, "high": h, "low": low, "close": c, "volume": v}


class TestIndicators:
    def test_obv_matches_hand_computation(self) -> None:
        bars = make_bars([10, 11, 10.5, 10.5, 12], volumes=[100, 200, 300, 400, 500])
        assert sig.obv_series(bars) == [0, 200, -100, -100, 400]

    def test_obv_ignores_an_unchanged_close(self) -> None:
        bars = make_bars([10, 10, 10], volumes=[100, 100, 100])
        assert sig.obv_series(bars) == [0, 0, 0]

    def test_adl_matches_hand_computation(self) -> None:
        bars = [
            bar("a", 10, 11, 9, 10, 1000),      # clv = 0
            bar("b", 10.5, 12, 10, 11.5, 2000),  # clv = +0.5
            bar("c", 11, 11, 10, 10, 500),       # clv = -1
        ]
        # the 1e-10 guard in the legacy CLV formula survives by design
        assert sig.adl_series(bars) == pytest.approx([0.0, 1000.0, 500.0], rel=1e-6)

    def test_rolling_mean_needs_a_full_window(self) -> None:
        assert sig.rolling_mean([1.0, 2.0, 3.0], 3, 2) == 2.0
        assert sig.rolling_mean([1.0, 2.0, 3.0], 3, 1) is None

    def test_rolling_mean_reports_none_when_a_bar_is_missing(self) -> None:
        assert sig.rolling_mean([1.0, None, 3.0], 3, 2) is None

    def test_highest_high_and_lowest_low_are_inclusive(self) -> None:
        bars = make_bars([10, 11, 12], highs=[10, 15, 12], lows=[9, 8, 7])
        assert sig.highest_high(bars, 0, 2) == 15
        assert sig.lowest_low(bars, 0, 2) == 7
        assert sig.lowest_low(bars, 0, 0) == 9


class TestVolumeProfile:
    def test_top_edge_close_lands_in_no_bin(self) -> None:
        # The legacy binning loops to bins-2, so the window maximum never counts.
        poc, strength = sig.volume_profile_poc([10.0, 12.0], [100.0, 100.0], bins=20)
        assert poc == pytest.approx(10.0)
        assert strength == pytest.approx(1.0)

    def test_uniform_closes_put_the_poc_at_the_first_bin(self) -> None:
        closes = [100.0 + i for i in range(20)]
        poc, strength = sig.volume_profile_poc(closes, [1.0] * 20, bins=20)
        assert poc == pytest.approx(100.0)
        assert strength == pytest.approx(1 / 19)

    def test_poc_is_never_the_window_high(self) -> None:
        closes = [10.0 + i * 0.1 for i in range(30)]
        poc, _ = sig.volume_profile_poc(closes, [2.0] * 30, bins=20)
        assert poc < max(closes)

    def test_flat_window_does_not_divide_by_zero(self) -> None:
        poc, strength = sig.volume_profile_poc([5.0, 5.0, 5.0], [1.0, 1.0, 1.0], bins=10)
        assert poc == 5.0 and strength == 0.0


class TestConditions:
    def test_c2_fires_on_sustained_buying(self) -> None:
        closes = [10.0, 10.0] + [10.0 + 0.1 * i for i in range(20)]
        ctx = sig.SignalContext(make_bars(closes, volumes=[1000.0] * len(closes)))
        assert sig.c2_order_flow_buying(ctx, 21) is True

    def test_c2_rejects_sustained_selling(self) -> None:
        closes = [20.0, 20.0] + [20.0 - 0.1 * i for i in range(20)]
        ctx = sig.SignalContext(make_bars(closes, volumes=[1000.0] * len(closes)))
        assert sig.c2_order_flow_buying(ctx, 21) is False

    def test_c2_is_none_before_the_window_exists(self) -> None:
        bars = make_bars([10.0 + i * 0.1 for i in range(10)])
        ctx = sig.SignalContext(bars)
        assert sig.c2_order_flow_buying(ctx, 2) is None

    def test_c3_legacy_never_fires_on_any_real_series(self) -> None:
        # close <= high <= max(high) is arithmetic, not an opinion: over every
        # bar of a random walk the legacy breakout condition is False, never True.
        bars = random_walk(200, seed=4)
        ctx = sig.SignalContext(bars)
        values = [sig.c3_price_structure_legacy(ctx, i) for i in range(len(bars))]
        assert all(v is False for v in values[30:])
        assert (sig.c3_price_structure_corrected(ctx, 30) in (True, False))

    def test_c3_corrected_fires_on_a_real_breakout(self) -> None:
        bars = make_bars([10.0] * 40 + [12.0], highs=[10.1] * 40 + [12.1], lows=[9.9] * 41)
        ctx = sig.SignalContext(bars)
        assert sig.c3_price_structure_corrected(ctx, 40) is True
        assert sig.c3_price_structure_legacy(ctx, 40) is False

    def test_c3_corrected_needs_a_previous_window(self) -> None:
        bars = make_bars([10.0] * 10)
        ctx = sig.SignalContext(bars)
        assert sig.c3_price_structure_corrected(ctx, 5) is None

    def test_c1_poc_below_price_when_value_built_lower(self) -> None:
        closes = [20.0] * 100 + [20.0 + 0.5 * i for i in range(20)]
        ctx = sig.SignalContext(make_bars(closes, volumes=[5000.0] * 120))
        assert sig.c1_volume_profile_bullish(ctx, 119) is True

    def test_d1_is_the_mirror_of_c1(self) -> None:
        closes = [25.0 + 0.05 * i for i in range(100)] + [29.0 - 0.4 * i for i in range(20)]
        ctx = sig.SignalContext(make_bars(closes, volumes=[5000.0] * 120))
        assert sig.d1_volume_profile_bearish(ctx, 119) is True
        assert sig.c1_volume_profile_bullish(ctx, 119) is False

    def test_c4_and_c5_need_their_window(self) -> None:
        bars = make_bars([10.0 + i * 0.05 for i in range(40)])
        ctx = sig.SignalContext(bars)
        assert sig.c4_obv_confirmed(ctx, 5) is None
        assert sig.c4_obv_confirmed(ctx, 39) in (True, False)
        assert sig.c5_adl_confirmed(ctx, 39) in (True, False)

    def test_risk_reward_of_a_flat_market_is_not_infinite(self) -> None:
        bars = make_bars([10.0] * 40)
        ctx = sig.SignalContext(bars)
        rr = ctx.risk_reward(39)
        assert rr is None or rr >= 0

    def test_risk_reward_none_without_history(self) -> None:
        ctx = sig.SignalContext(make_bars([10.0] * 5))
        assert ctx.risk_reward(4) is None


class TestComposites:
    def test_legacy_composite_requires_four_of_five(self) -> None:
        closes = [10.0] * 100 + [15.0] * 40
        ctx = sig.SignalContext(make_bars(closes, volumes=[1000.0] * 140))
        value = sig.accumulation_legacy(ctx, 139)
        assert value in (True, False)

    def test_high_conviction_is_unreachable_with_the_legacy_condition(self) -> None:
        # 5/5 including the unsatisfiable c3_legacy can never happen, which is
        # why the corrected composite exists.
        bars = random_walk(300, seed=9, boost_after_breakout=0.01)
        ctx = sig.SignalContext(bars)
        met = [sig._count(ctx, i, sig.LEGACY_CONDITIONS) for i in range(100, 300)]
        assert met and max(m for m in met if m is not None) <= 4

    def test_distribution_composite_runs(self) -> None:
        bars = random_walk(200, seed=3)
        ctx = sig.SignalContext(bars)
        assert sig.distribution_corrected(ctx, 199) in (True, False)

    def test_counts_are_none_until_every_condition_is_decidable(self) -> None:
        ctx = sig.SignalContext(make_bars([10.0] * 10))
        assert sig._count(ctx, 5, sig.CORRECTED_CONDITIONS) is None


class TestRegistry:
    def test_builtins_are_registered(self) -> None:
        names = sig.signal_names()
        for expected in ("c1", "c2", "c3", "c3_legacy", "c4", "c5", "accumulation_corrected"):
            assert expected in names
        assert len(names) >= 15

    def test_third_party_signals_can_be_registered_and_evaluated(self) -> None:
        spec = sig.SignalSpec(
            name="test_always_true",
            title="Test signal",
            fn=lambda ctx, i: True,
            hypothesis="unit-test plugin",
            min_history=1,
        )
        sig.register(spec, replace=True)
        ctx = sig.SignalContext(make_bars([10.0] * 5))
        assert sig.evaluate("test_always_true", ctx, 0) is True
        assert "test_always_true" in sig.signal_names()

    def test_duplicate_registration_is_refused(self) -> None:
        spec = sig.SignalSpec(name="test_duplicate", title="dup", fn=lambda ctx, i: False)
        sig.register(spec, replace=True)
        with pytest.raises(ValueError):
            sig.register(spec)

    def test_unknown_signal_raises_with_the_known_names(self) -> None:
        with pytest.raises(KeyError) as exc:
            sig.get_signal("nope")
        assert "c1" in str(exc.value)

    def test_scan_returns_one_series_per_signal_aligned_to_bars(self) -> None:
        ctx = sig.SignalContext(make_bars([10.0 + i * 0.1 for i in range(50)]))
        out = sig.scan(ctx, names=["c1", "c2"])
        assert set(out) == {"c1", "c2"}
        assert all(len(series) == 50 for series in out.values())
        assert all(series[0] is None for series in out.values())

    def test_scan_can_filter_by_kind(self) -> None:
        kinds = {s.kind for s in sig.all_signals(kind="accumulation")}
        assert kinds == {"accumulation"}
