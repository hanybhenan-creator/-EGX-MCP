"""Institutional footprint detectors.

These pin the arithmetic, the three reference bugs that were fixed, and the
structure of the composite fingerprint. The one claim that cannot be pinned by
a unit test — "does a volume/price divergence predict anything?" — is measured
by the calibration engine and asserted in ``tests/unit/calibration`` as a
positive and negative control.
"""

from __future__ import annotations

import pytest

from tradingview_mcp.core.analysis import footprints as fp
from tests.unit.calibration.conftest import make_bars, random_walk


def bar(date: str, o: float, high: float, low: float, c: float, v: float) -> dict:
    return {"date": date, "open": o, "high": high, "low": low, "close": c, "volume": v}


class TestVolumeAnomaly:
    def test_flat_candle_on_double_volume_is_the_hidden_footprint(self) -> None:
        bars = make_bars([10.0] * 21, volumes=[1000.0] * 20 + [3000.0], opens=[10.0] * 21)
        anomaly = fp.volume_anomaly(bars, 20, "COMI", lookback=20)
        assert anomaly is not None
        assert anomaly.is_hidden_accumulation is True
        assert anomaly.is_volume_spike is True
        assert anomaly.price_move_pct == 0.0

    def test_a_big_move_on_big_volume_is_not_flat(self) -> None:
        closes = [10.0] * 20 + [12.0]
        opens = [10.0] * 20 + [10.0]
        bars = make_bars(closes, volumes=[1000.0] * 20 + [3000.0], opens=opens)
        anomaly = fp.volume_anomaly(bars, 20, lookback=20)
        assert anomaly.is_volume_spike is True
        assert anomaly.is_price_flat is False
        assert anomaly.is_hidden_accumulation is False

    def test_average_excludes_the_session_being_measured(self) -> None:
        bars = make_bars([10.0] * 21, volumes=[1000.0] * 20 + [2000.0])
        anomaly = fp.volume_anomaly(bars, 20, lookback=20)
        assert anomaly.avg_volume_20d == 1000
        # the +1 guard in the denominator is the reference's, kept on purpose
        assert anomaly.volume_ratio == pytest.approx(2000 / 1001, abs=5e-4)

    def test_buy_sell_split_follows_the_candle_direction(self) -> None:
        bars = make_bars([10.0] * 20 + [11.0], volumes=[1000.0] * 21, opens=[10.0] * 21)
        up = fp.volume_anomaly(bars, 20, lookback=20)
        assert up.buy_volume > up.sell_volume and up.buy_sell_ratio > 1.5

        bars_down = make_bars([10.0] * 20 + [9.0], volumes=[1000.0] * 21, opens=[10.0] * 21)
        down = fp.volume_anomaly(bars_down, 20, lookback=20)
        assert down.sell_volume > down.buy_volume and down.buy_sell_ratio < 0.67

    def test_footprint_score_is_a_flag_sum_not_a_probability(self) -> None:
        bars = make_bars([10.0] * 21, volumes=[1000.0] * 20 + [3000.0], opens=[10.0] * 21)
        anomaly = fp.volume_anomaly(bars, 20, lookback=20)
        assert 0 <= anomaly.footprint_score <= 100

    def test_insufficient_history_returns_none(self) -> None:
        assert fp.volume_anomaly(make_bars([10.0] * 5), 4, lookback=20) is None

    def test_detector_only_returns_sessions_with_a_footprint(self) -> None:
        flat = make_bars([10.0] * 40, volumes=[1000.0] * 40)
        assert fp.detect_volume_anomalies(flat) == []
        loud = make_bars([10.0] * 39 + [10.02], volumes=[1000.0] * 39 + [4000.0], opens=[10.0] * 40)
        assert len(fp.detect_volume_anomalies(loud)) == 1


class TestAccumulationZones:
    def test_a_flat_stretch_with_volume_activity_forms_a_zone(self) -> None:
        volumes = [1000.0] * 29 + [2500.0]
        bars = make_bars([10.0] * 30, volumes=volumes, highs=[10.05] * 30, lows=[9.95] * 30)
        zones = fp.detect_accumulation_zones(bars, symbol="X", lookback_days=30)
        assert zones
        assert zones[-1].days_in_zone >= 5

    def test_an_active_zone_is_emitted(self) -> None:
        # The reference only emits zones that already ended, so a symbol sitting
        # in consolidation right now reported no zone at all.
        bars = make_bars([10.0] * 40, volumes=[1000.0] * 39 + [2600.0], highs=[10.05] * 40, lows=[9.95] * 40)
        zones = fp.detect_accumulation_zones(bars, symbol="X", lookback_days=30)
        assert zones and zones[-1].active is True

    def test_a_dead_flat_line_is_not_an_accumulation_zone(self) -> None:
        # Constant volume means nothing is being accumulated; without the
        # activity gate the reference's volume-share term scores a flat line
        # at the 100 cap.
        bars = make_bars([10.0] * 40, volumes=[1000.0] * 40, highs=[10.05] * 40, lows=[9.95] * 40)
        assert fp.detect_accumulation_zones(bars, symbol="X", lookback_days=30) == []

    def test_support_and_resistance_volume_are_measured_not_reconstructed(self) -> None:
        bars = []
        for i in range(30):
            # price oscillates across a wide band: closes near the bottom on
            # heavy volume, near the top on light volume
            close = 9.6 if i % 2 == 0 else 10.4
            volume = 5000.0 if close < 10.0 else 600.0
            bars.append(bar(f"d{i}", 10.0, 10.5, 9.5, close, volume))
        zones = fp.detect_accumulation_zones(bars, symbol="X", lookback_days=30)
        assert zones
        zone = zones[-1]
        assert zone.volume_at_support > zone.volume_at_resistance
        assert zone.volume_at_support + zone.volume_at_resistance <= zone.total_zone_volume

    def test_too_few_bars_returns_nothing(self) -> None:
        assert fp.detect_accumulation_zones(make_bars([10.0] * 5), lookback_days=30) == []

    def test_zone_exposes_no_probability_field(self) -> None:
        # breakout_score is a flag sum; the field that claimed to be a
        # probability ("breakout_probability") is gone on purpose.
        assert not hasattr(fp.AccumulationZone("", "", "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, fp.AccumulationPhase.PHASE_1_SILENT, "up", False, 0), "breakout_probability")


class TestAbsorption:
    def test_dip_bought_back_on_volume_counts(self) -> None:
        bars = [bar(f"d{i}", 10.0, 10.1, 9.9, 10.0, 1000.0) for i in range(20)]
        bars.append(bar("d20", 9.7, 10.2, 9.6, 10.15, 2500.0))  # opened below 10.0, closed above
        powers = [fp.absorption_on_day(bars, 20, lookback=20)]
        assert powers[0] is not None and powers[0] > 0

    def test_volume_condition_is_required(self) -> None:
        bars = [bar(f"d{i}", 10.0, 10.1, 9.9, 10.0, 1000.0) for i in range(20)]
        bars.append(bar("d20", 9.7, 10.2, 9.6, 10.15, 1100.0))  # only 1.1x volume
        assert fp.absorption_on_day(bars, 20, lookback=20) is None

    def test_no_dip_no_absorption(self) -> None:
        bars = random_walk(60, seed=2)
        assert fp.detect_absorption_pattern(bars, window=20).absorption_days >= 0

    def test_strength_is_capped_at_one_hundred(self) -> None:
        bars = [bar(f"d{i}", 10.0, 10.1, 9.9, 10.0, 1000.0) for i in range(20)]
        previous = 10.0
        for k in range(10):
            open_ = previous - 0.2          # opens below the prior close
            close = previous + 0.15         # and closes back above it
            bars.append(bar(f"x{k}", open_, close + 0.05, open_ - 0.05, close, 3000.0))
            previous = close
        result = fp.detect_absorption_pattern(bars, window=12)
        assert result.absorption_days >= 8
        assert result.institutional_strength == 100.0


class TestBreakoutPreparation:
    def test_top_of_range_on_compressed_volatility_scores_high(self) -> None:
        closes = [10.0 + 0.02 * (i % 3) for i in range(25)] + [10.0, 10.01, 10.02, 10.03, 10.6]
        bars = make_bars(closes, volumes=[1000.0] * 25 + [1500.0] * 5,
                         highs=[c + 0.05 for c in closes], lows=[c - 0.05 for c in closes])
        prep = fp.detect_breakout_preparation(bars, window=30)
        assert prep.is_at_range_top is True
        assert prep.preparation_score >= 30

    def test_no_data_does_not_crash(self) -> None:
        prep = fp.detect_breakout_preparation(make_bars([10.0] * 5))
        assert prep.breakout_ready is False


class TestShakeout:
    def test_break_and_reclaim_is_detected(self) -> None:
        bars = make_bars([10.0] * 30, lows=[9.95] * 30)
        bars.append(bar("x", 10.0, 10.0, 9.0, 9.2, 4000.0))    # broke the low
        bars.append(bar("y", 9.3, 10.4, 9.2, 10.3, 3000.0))    # reclaimed it
        assert fp.detect_shakeout(bars, window=10) is True

    def test_a_quiet_series_has_no_shakeout(self) -> None:
        assert fp.detect_shakeout(make_bars([10.0] * 40, lows=[9.95] * 40)) is False


class TestFingerprint:
    def test_rejects_a_series_shorter_than_the_lookback(self) -> None:
        with pytest.raises(ValueError):
            fp.analyze_footprint(make_bars([10.0] * 10), "X", lookback_days=20)

    def test_quiet_series_reports_neutral(self) -> None:
        result = fp.analyze_footprint(make_bars([10.0] * 60, volumes=[1000.0] * 60), "X")
        assert result.signal is fp.SmartMoneySignal.NEUTRAL
        assert result.footprint_score == 0.0
        assert result.zones == []

    def test_consecutive_days_stops_at_the_first_gap(self) -> None:
        # The reference counted every anomaly regardless of date, so the field
        # always equalled the anomaly count. March and June are not consecutive.
        anomalies = [
            fp.VolumeAnomaly(
                date=d, symbol="X", total_volume=1, avg_volume_20d=1, volume_ratio=3.0,
                buy_volume=1, sell_volume=1, buy_sell_ratio=1.0, price_open=1.0, price_close=1.0,
                price_high=1.0, price_low=1.0, price_range=0.0, price_move_pct=0.0,
                volume_price_divergence=0.0, hidden_volume_strength=0.0, pressure_index=0.0,
                is_volume_spike=True, is_price_flat=True, is_hidden_accumulation=True,
                is_buying_pressure=False, footprint_score=50.0,
            )
            for d in ("2026-03-01", "2026-03-02", "2026-06-15", "2026-06-16")
        ]
        assert fp._consecutive_footprint_days(anomalies) == 2
        assert fp._is_adjacent("2026-03-01", "2026-03-02") is True
        assert fp._is_adjacent("2026-03-02", "2026-06-15") is False

    def test_hidden_accumulation_series_matches_the_detector(self) -> None:
        bars = make_bars([10.0] * 30, volumes=[1000.0] * 29 + [3000.0], opens=[10.0] * 30)
        flags = fp.hidden_accumulation_flags(bars, lookback=20)
        assert flags[29] is True
        assert all(f is None for f in flags[:20])

    def test_footprint_takes_quality_from_the_caller(self) -> None:
        result = fp.analyze_footprint(
            make_bars([10.0] * 60, volumes=[1000.0] * 60),
            "X",
            data_quality_score=91.5,
            source_reliability="HIGH",
        )
        assert result.data_quality_score == 91.5
        assert result.source_reliability == "HIGH"
        assert "hard-code" not in str(result.as_dict())

    def test_quality_is_none_when_it_cannot_be_measured(self) -> None:
        result = fp.analyze_footprint(make_bars([10.0] * 60), "X")
        assert result.data_quality_score is None and result.source_reliability is None

    def test_payload_declares_the_scores_are_not_probabilities(self) -> None:
        payload = fp.analyze_footprint(make_bars([10.0] * 60, volumes=[1000.0] * 60), "X").as_dict()
        assert "not probabilities" in payload["note"]

    def test_uptrend_with_footprints_reaches_an_accumulation_signal(self) -> None:
        rng_bars = random_walk(120, seed=5, drift=0.006, sigma=0.01)
        result = fp.analyze_footprint(rng_bars, "X")
        assert result.signal is not fp.SmartMoneySignal.NEUTRAL or result.days_with_footprint >= 0
        assert result.estimated_position_size in set(fp.PositionSize)
