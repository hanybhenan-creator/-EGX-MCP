"""Event study: forward-return alignment, benchmark, inference, walk-forward."""

from __future__ import annotations

import copy

import pytest

from tradingview_mcp.core.calibration import signals as sig
from tradingview_mcp.core.calibration.study import (
    Measurement,
    Sample,
    call_it,
    forward_return,
    measure,
    measure_all,
    walk_forward,
)
from tests.unit.calibration.conftest import make_bars, random_walk

# A signal that fires on every bar with enough history — the simplest way to
# test the machinery without the signal's own logic in the way.
sig.register(
    sig.SignalSpec(
        name="test_always",
        title="Always fires",
        fn=lambda ctx, i: True,
        hypothesis="test plumbing",
        min_history=30,
    ),
    replace=True,
)
# Fires only above a close threshold, so a planted winner can be selected.
sig.register(
    sig.SignalSpec(
        name="test_expensive_only",
        title="Close above 50",
        fn=lambda ctx, i: (ctx.closes[i] or 0.0) > 50.0,
        hypothesis="test selectivity",
        min_history=5,
    ),
    replace=True,
)


class TestForwardReturn:
    def test_entry_is_the_next_open_and_exit_the_later_close(self) -> None:
        bars = [
            {"date": "d0", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1.0},
            {"date": "d1", "open": 12.0, "high": 13.0, "low": 11.0, "close": 12.5, "volume": 1.0},
            {"date": "d2", "open": 13.0, "high": 15.0, "low": 12.0, "close": 14.0, "volume": 1.0},
        ]
        fwd = forward_return(bars, 0, 2)
        assert fwd is not None
        assert (fwd.entry_index, fwd.exit_index) == (1, 2)
        assert (fwd.entry_date, fwd.exit_date) == ("d1", "d2")
        assert fwd.ret == pytest.approx(14.0 / 12.0 - 1)

    def test_the_signal_bars_own_close_cannot_leak_into_the_return(self) -> None:
        bars = [
            {"date": "d0", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1.0},
            {"date": "d1", "open": 12.0, "high": 13.0, "low": 11.0, "close": 12.5, "volume": 1.0},
            {"date": "d2", "open": 13.0, "high": 15.0, "low": 12.0, "close": 14.0, "volume": 1.0},
        ]
        base = forward_return(bars, 0, 2)
        altered = copy.deepcopy(bars)
        altered[0]["close"] = 999.0
        assert forward_return(altered, 0, 2).ret == base.ret

    def test_falls_back_to_the_entry_close_when_open_is_missing(self) -> None:
        bars = [
            {"date": "d0", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1.0},
            {"date": "d1", "open": None, "high": 13.0, "low": 11.0, "close": 12.0, "volume": 1.0},
            {"date": "d2", "open": 13.0, "high": 15.0, "low": 12.0, "close": 13.2, "volume": 1.0},
        ]
        fwd = forward_return(bars, 0, 2)
        assert fwd.entry_price == 12.0

    def test_returns_none_when_the_future_does_not_exist(self) -> None:
        bars = make_bars([10.0, 11.0, 12.0])
        assert forward_return(bars, 2, 1) is None
        assert forward_return(bars, 0, 5) is None

    def test_absurd_moves_are_treated_as_data_errors(self) -> None:
        bars = [
            {"date": "d0", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            {"date": "d1", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            {"date": "d2", "open": 1.0, "high": 8.0, "low": 1.0, "close": 8.0, "volume": 1.0},
        ]
        assert forward_return(bars, 0, 2) is None

    def test_horizon_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            forward_return(make_bars([10.0, 11.0]), 0, 0)


class TestMeasurement:
    @pytest.fixture
    def sample(self) -> Sample:
        # One winner (A) and two exactly flat symbols for the benchmark.
        a = [
            {
                "date": f"2024-01-{i + 1:02d}",
                "open": 100.0 * 1.05 ** max(0, i - 1),
                "high": 100.0 * 1.05**i * 1.01,
                "low": 100.0 * 1.05**i * 0.99,
                "close": 100.0 * 1.05**i,
                "volume": 1000.0,
            }
            for i in range(60)
        ]
        flat = [
            {
                "date": f"2024-01-{i + 1:02d}",
                "open": 10.0,
                "high": 10.1,
                "low": 9.9,
                "close": 10.0,
                "volume": 1000.0,
            }
            for i in range(60)
        ]
        return Sample(bars_by_symbol={"A": a, "B": flat, "C": flat}, horizon=2)

    def test_event_count_matches_the_signal_positions(self, sample: Sample) -> None:
        m = measure(sample, "test_always", permutations=0)
        # min_history 30 gates i >= 29; a 2-session horizon allows i <= 57
        assert m.events == 3 * 29
        assert m.symbols == 3

    def test_base_rate_is_the_unconditional_hit_rate(self, sample: Sample) -> None:
        m = measure(sample, "test_always", permutations=0)
        # Only A rises; B and C return exactly 0.0, which is not a hit.
        assert m.base_rate == pytest.approx(29 / 87)

    def test_excess_detects_the_planted_winner(self, sample: Sample) -> None:
        m = measure(sample, "test_expensive_only", permutations=0)
        assert m.symbols == 1
        assert m.hit_rate == 1.0
        ev = 1.05**2 - 1
        bench = ev / 3  # A's move averaged with two flat symbols
        assert m.mean_excess == pytest.approx(ev - bench, abs=1e-9)
        assert m.mean_excess > 0

    def test_benchmark_removes_the_market_move(self, sample: Sample) -> None:
        m = measure(sample, "test_always", permutations=0)
        # Averaged across all three symbols the excess must cancel: the
        # benchmark IS the cross-section.
        assert m.mean_excess == pytest.approx(0.0, abs=1e-9)

    def test_confidence_interval_brackets_the_hit_rate(self, sample: Sample) -> None:
        m = measure(sample, "test_expensive_only", permutations=0)
        assert m.hit_ci is not None and m.hit_ci[0] <= m.hit_rate <= m.hit_ci[1]

    def test_window_filter_is_applied_to_events_and_base_rate(self, sample: Sample) -> None:
        m = measure(sample, "test_always", permutations=0, date_start="2024-01-50")
        assert m.first_event >= "2024-01-50"
        assert m.events < 87

    def test_no_events_is_reported_as_such(self, sample: Sample) -> None:
        m = measure(sample, "c3_legacy", permutations=0)
        assert m.events == 0 and m.verdict == "NO_EVENTS"

    def test_p_values_are_skipped_when_permutations_are_off(self, sample: Sample) -> None:
        m = measure(sample, "test_always", permutations=0)
        assert m.p_hit is None and m.p_mean is None

    def test_per_symbol_breakdown_is_included(self, sample: Sample) -> None:
        m = measure(sample, "test_always", permutations=0)
        assert {row["symbol"] for row in m.best_symbols} == {"A", "B", "C"}

    def test_as_dict_serialises_cleanly(self, sample: Sample) -> None:
        payload = measure(sample, "test_always", permutations=0).as_dict()
        assert payload["signal"] == "test_always"
        assert payload["title"] and payload["hypothesis"]
        assert isinstance(payload["hit_rate_ci95"], list)
        assert payload["verdict"] in {"VALIDATED", "SUGGESTIVE", "REJECTED", "NO_EVENTS"}

    def test_contexts_and_forwards_are_computed_once(self, sample: Sample) -> None:
        assert sample.context("A") is sample.context("A")
        assert sample.forwards("A") is sample.forwards("A")


class TestControls:
    """The two controls the whole engine rests on."""

    def test_planted_edge_is_detected_with_a_small_p_value(self, seeded_universe) -> None:
        sample = Sample(seeded_universe(smart=5, seed=1, sessions=320), horizon=5)
        m = measure(sample, "c3", permutations=200, seed=3)
        assert m.events > 100
        assert m.excess_hit is not None and m.excess_hit > 0.03
        assert m.p_hit is not None and m.p_hit < 0.05

    def test_noise_is_not_declared_validated(self, seeded_universe) -> None:
        sample = Sample(seeded_universe(smart=0, seed=2, sessions=320), horizon=5)
        m = measure(sample, "c3", permutations=200, seed=3)
        assert m.verdict != "VALIDATED"
        assert m.p_hit is None or m.p_hit > 0.05


class TestBatch:
    def test_measure_all_corrects_p_values(self, seeded_universe) -> None:
        sample = Sample(seeded_universe(smart=3, seed=1), horizon=5)
        out = measure_all(sample, signals=["c2", "c4", "c3"], permutations=50, seed=1)
        assert len(out) == 3
        for m in out:
            assert m.verdict
            if m.p_hit is not None and m.q_hit is not None:
                assert m.q_hit >= m.p_hit - 1e-12

    def test_measure_all_is_deterministic(self, seeded_universe) -> None:
        bars = seeded_universe(smart=3, seed=1)
        a = measure_all(Sample(bars, horizon=5), signals=["c3"], permutations=50, seed=1)[0]
        b = measure_all(Sample(bars, horizon=5), signals=["c3"], permutations=50, seed=1)[0]
        assert a.p_hit == b.p_hit and a.events == b.events


class TestWalkForward:
    def test_folds_are_contiguous_and_do_not_overlap(self, seeded_universe) -> None:
        sample = Sample(seeded_universe(smart=4, seed=1, sessions=400), horizon=5)
        wf = walk_forward(sample, "c3", folds=3, permutations=0)
        assert len(wf.folds) == 3
        for f in wf.folds:
            assert f.train_end <= f.test_start
            assert f.test.events == 0 or f.test.first_event >= f.test_start
            if f.train.events and f.test.events:
                assert f.train.last_event <= f.test.first_event
        for a, b in zip(wf.folds, wf.folds[1:]):
            assert a.test_end <= b.test_start

    def test_verdict_is_one_of_the_declared_states(self, seeded_universe) -> None:
        sample = Sample(seeded_universe(smart=4, seed=1, sessions=400), horizon=5)
        wf = walk_forward(sample, "c3", folds=3, permutations=0)
        assert wf.verdict in {"ROBUST", "MODERATE", "WEAK", "OVERFITTED", "NO_TRAIN_EDGE", "INSUFFICIENT_HISTORY"}
        assert wf.reason

    def test_short_history_is_refused_rather_than_guessed(self) -> None:
        sample = Sample({"A": make_bars([10.0] * 20)}, horizon=5)
        wf = walk_forward(sample, "c2", folds=4)
        assert wf.verdict == "INSUFFICIENT_HISTORY" and wf.folds == []

    def test_fold_count_must_be_at_least_two(self, seeded_universe) -> None:
        sample = Sample(seeded_universe(sessions=100), horizon=5)
        with pytest.raises(ValueError):
            walk_forward(sample, "c2", folds=1)

    def test_planted_edge_retains_some_out_of_sample_edge(self, seeded_universe) -> None:
        sample = Sample(seeded_universe(smart=5, seed=1, sessions=420), horizon=5)
        wf = walk_forward(sample, "c3", folds=3, permutations=0)
        assert wf.verdict in {"ROBUST", "MODERATE", "WEAK"}
        assert wf.retention is not None and wf.retention > 0

    def test_as_dict_has_the_fold_windows(self, seeded_universe) -> None:
        sample = Sample(seeded_universe(smart=4, seed=1, sessions=400), horizon=5)
        payload = walk_forward(sample, "c3", folds=2, permutations=0).as_dict()
        assert len(payload["folds"]) == 2
        assert all(len(f["test_window"]) == 2 for f in payload["folds"])


class TestVerdicts:
    def test_no_events(self) -> None:
        assert call_it(Measurement("x", 5)) == "NO_EVENTS"

    def test_few_events_in_the_right_direction_is_suggestive(self) -> None:
        m = Measurement("x", 5, events=5, excess_hit=0.1, p_hit=0.01, q_hit=0.01)
        assert call_it(m) == "SUGGESTIVE"

    def test_wrong_direction_is_rejected(self) -> None:
        m = Measurement("x", 5, events=500, excess_hit=-0.02, p_hit=0.9, q_hit=0.9)
        assert call_it(m) == "REJECTED"

    def test_enough_events_with_corrected_significance_is_validated(self) -> None:
        m = Measurement("x", 5, events=200, excess_hit=0.05, p_hit=0.001, q_hit=0.009)
        assert call_it(m) == "VALIDATED"

    def test_uncorrected_significance_is_not_enough(self) -> None:
        m = Measurement("x", 5, events=200, excess_hit=0.05, p_hit=0.01, q_hit=0.4)
        assert call_it(m) == "SUGGESTIVE"

    def test_missing_p_value_keeps_the_claim_at_suggestive(self) -> None:
        m = Measurement("x", 5, events=200, excess_hit=0.05)
        assert call_it(m) == "SUGGESTIVE"
