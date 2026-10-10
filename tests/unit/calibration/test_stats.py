"""Statistical primitives: intervals, permutation p-values, FDR."""

from __future__ import annotations

import random

import pytest

from tradingview_mcp.core.calibration import stats


class TestDescriptive:
    def test_hit_rate_counts_strictly_positive_only(self) -> None:
        assert stats.hit_rate([0.01, -0.01, 0.0, 0.02]) == 0.5

    def test_hit_rate_threshold_is_configurable(self) -> None:
        assert stats.hit_rate([0.01, 0.05, 0.09], threshold=0.05) == pytest.approx(1 / 3)

    def test_hit_rate_of_nothing_is_none_not_zero(self) -> None:
        assert stats.hit_rate([]) is None

    def test_median_even_count_averages_the_middle(self) -> None:
        assert stats.median([1, 2, 3, 4]) == 2.5

    def test_quantile_matches_linear_interpolation(self) -> None:
        assert stats.quantile([0, 10], 0.25) == 2.5
        assert stats.quantile([1, 2, 3, 4, 5], 0.5) == 3

    def test_stdev_is_the_sample_variant(self) -> None:
        assert stats.stdev([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.13809, abs=1e-5)

    def test_describe_carries_the_sample_size(self) -> None:
        out = stats.describe([0.1, -0.2, None, 0.3])
        assert out["n"] == 3 and out["mean"] == pytest.approx(0.0666667, abs=1e-6)


class TestWilsonInterval:
    def test_matches_the_textbook_value(self) -> None:
        lo, hi = stats.wilson_interval(8, 10)
        assert lo == pytest.approx(0.4902, abs=0.001)
        assert hi == pytest.approx(0.9433, abs=0.001)

    def test_brackets_the_point_estimate(self) -> None:
        lo, hi = stats.wilson_interval(9, 12)
        assert lo < 9 / 12 < hi

    def test_stays_inside_the_unit_interval_at_the_boundary(self) -> None:
        lo, hi = stats.wilson_interval(20, 20)
        assert lo < 1.0 and hi == 1.0 and lo > 0.8

    def test_no_observations_means_no_interval(self) -> None:
        assert stats.wilson_interval(0, 0) is None

    def test_narrows_as_the_sample_grows(self) -> None:
        small = stats.wilson_interval(6, 10)
        large = stats.wilson_interval(600, 1000)
        assert (large[1] - large[0]) < (small[1] - small[0])


class TestBlockShuffle:
    def test_preserves_the_value_multiset(self) -> None:
        values = [float(i) for i in range(50)]
        out = stats.block_shuffle(values, block=7, rng=random.Random(3))
        assert sorted(out) == sorted(values)

    def test_same_seed_same_shuffle(self) -> None:
        values = [float(i) for i in range(30)]
        a = stats.block_shuffle(values, block=4, rng=random.Random(11))
        b = stats.block_shuffle(values, block=4, rng=random.Random(11))
        assert a == b

    def test_full_length_block_is_a_rotation_not_a_shuffle(self) -> None:
        values = [float(i) for i in range(20)]
        out = stats.block_shuffle(values, block=20, rng=random.Random(0))
        rotations = [values[k:] + values[:k] for k in range(len(values))]
        assert out in rotations

    def test_carries_local_structure(self) -> None:
        # A strictly increasing series shuffled in blocks of 5 still has 4 of
        # every 5 adjacent pairs increasing — that is the property that makes
        # the permutation honest about momentum.
        values = [float(i) for i in range(200)]
        out = stats.block_shuffle(values, block=5, rng=random.Random(7))
        increasing = sum(1 for a, b in zip(out, out[1:]) if b > a)
        assert increasing / (len(out) - 1) > 0.7

    def test_empty_series(self) -> None:
        assert stats.block_shuffle([], block=5, rng=random.Random(0)) == []


class TestPermutationP:
    def test_planted_advantage_is_detected(self) -> None:
        values = [0.02] * 50 + [-0.01] * 450
        positions = list(range(50))
        p = stats.permutation_p_mean(values, positions, block=1, permutations=200, seed=0)
        assert p is not None and p < 0.02

    def test_arbitrary_timing_is_not_significant(self) -> None:
        rng = random.Random(5)
        values = [rng.gauss(0, 0.02) for _ in range(400)]
        positions = list(range(0, 400, 4))
        p = stats.permutation_p_mean(values, positions, block=1, permutations=200, seed=1)
        assert p is not None and p > 0.05

    def test_p_is_bounded_by_the_permutation_count(self) -> None:
        values = [0.5] * 20 + [0.0] * 20
        p = stats.permutation_p_mean(values, list(range(20)), block=1, permutations=99, seed=0)
        assert p == pytest.approx(1 / 100)

    def test_no_positions_means_no_p_value(self) -> None:
        assert stats.permutation_p_mean([0.1, 0.2], [], block=1, permutations=10) is None

    def test_hit_variant_uses_the_threshold(self) -> None:
        values = [0.0] * 100 + [1.0] * 20
        positions = list(range(100, 120))
        p = stats.permutation_p_hit(values, positions, threshold=0.5, block=1, permutations=100, seed=2)
        assert p is not None and p < 0.05

    def test_rejects_an_unknown_alternative(self) -> None:
        with pytest.raises(ValueError):
            stats.permutation_p(
                values=[1.0], positions=[0], statistic=lambda s, i: 1.0, alternative="both"
            )


class TestBenjaminiHochberg:
    def test_worked_example(self) -> None:
        assert stats.benjamini_hochberg([0.01, 0.04]) == pytest.approx([0.02, 0.04])

    def test_single_p_value_is_its_own_q_value(self) -> None:
        assert stats.benjamini_hochberg([0.03]) == pytest.approx([0.03])

    def test_q_never_falls_below_p(self) -> None:
        ps = [0.8, 0.001, 0.03, 0.2, 0.6]
        for p, q in zip(ps, stats.benjamini_hochberg(ps)):
            assert q >= p - 1e-12 and q <= 1.0

    def test_preserves_input_order_and_monotonicity(self) -> None:
        ps = [0.9, 0.01, 0.4, 0.02]
        qs = stats.benjamini_hochberg(ps)
        assert qs[0] >= qs[2] >= qs[3] >= qs[1]
        assert qs[1] == min(qs)

    def test_none_entries_stay_none(self) -> None:
        qs = stats.benjamini_hochberg([None, 0.04, None, 0.01])
        assert qs[0] is None and qs[2] is None
        assert qs[1] is not None and qs[3] is not None

    def test_all_none(self) -> None:
        assert stats.benjamini_hochberg([None, None]) == [None, None]

    def test_correction_hurts_the_marginal_result(self) -> None:
        # One p=0.04 out of twenty tested is not 0.04 of evidence.
        qs = stats.benjamini_hochberg([0.04] + [0.5] * 19)
        assert qs[0] > 0.04


class TestOrderStatistics:
    def test_z_score_uses_the_null_spread(self) -> None:
        assert stats.z_score(2.0, [0.0, 1.0, -1.0, 0.5, -0.5]) is not None

    def test_z_score_of_empty_or_flat_null_is_none(self) -> None:
        assert stats.z_score(1.0, []) is None
        assert stats.z_score(1.0, [3.0, 3.0, 3.0]) is None

    def test_percentile_rank(self) -> None:
        assert stats.percentile_rank([1, 2, 3, 4], 2) == 0.5
