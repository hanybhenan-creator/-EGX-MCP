"""Money-flow indicators: known answers, missing-data handling, causality."""

from __future__ import annotations

import random

import pytest

from tradingview_mcp.core.analysis import moneyflow as mf


def bar(h: float, l: float, c: float, v: float | None) -> dict:
    return {"high": h, "low": l, "close": c, "volume": v}


# ── Chaikin Money Flow ────────────────────────────────────────────────────────

def test_cmf_known_value_closing_at_high_then_at_low() -> None:
    # bar0 closes at its high: multiplier +1. bar1 closes at its low: multiplier -1.
    # CMF(2) at index 1 = (1*100 + (-1)*300) / (100 + 300) = -0.5
    bars = [bar(10, 8, 10, 100), bar(10, 8, 8, 300)]
    assert mf.chaikin_money_flow(bars, 2)[1] == pytest.approx(-0.5)


def test_cmf_mid_close_is_zero_and_zero_range_bar_is_zero() -> None:
    bars = [bar(10, 8, 9, 100), bar(5, 5, 5, 100)]  # mid close; flat bar
    assert mf.chaikin_money_flow(bars, 2)[1] == pytest.approx(0.0)


def test_cmf_zero_total_volume_is_none_not_zero() -> None:
    bars = [bar(10, 8, 9, 0), bar(10, 8, 9, 0)]
    assert mf.chaikin_money_flow(bars, 2)[1] is None


# ── Money Flow Index ──────────────────────────────────────────────────────────

def test_mfi_known_value() -> None:
    # Typical price equals close here (H = L = C). Volume 100 each.
    # raw flow: 1000, 1200, 1100. Window at index 2 covers changes at bars 1 and 2:
    # bar1 up (10 -> 12): positive 1200; bar2 down (12 -> 11): negative 1100.
    # MFI = 100 - 100 / (1 + 1200/1100) = 52.1739...
    bars = [bar(10, 10, 10, 100), bar(12, 12, 12, 100), bar(11, 11, 11, 100)]
    assert mf.money_flow_index(bars, 2)[2] == pytest.approx(52.173913, rel=1e-6)


def test_mfi_monotone_rising_is_100_and_falling_is_0() -> None:
    up = [bar(p, p, p, 100) for p in (10, 11, 12)]
    down = [bar(p, p, p, 100) for p in (12, 11, 10)]
    assert mf.money_flow_index(up, 2)[2] == 100.0
    assert mf.money_flow_index(down, 2)[2] == 0.0


def test_mfi_no_price_movement_is_neutral_50() -> None:
    flat = [bar(10, 10, 10, 100) for _ in range(3)]
    assert mf.money_flow_index(flat, 2)[2] == 50.0


# ── Volume Price Trend ────────────────────────────────────────────────────────

def test_vpt_known_running_total() -> None:
    # bar1: 200 * (11 - 10) / 10 = 20. bar2: 100 * (9.9 - 11) / 11 = -10. Total 10.
    bars = [bar(10, 10, 10, 100), bar(11, 11, 11, 200), bar(10, 9.9, 9.9, 100)]
    out = mf.volume_price_trend(bars)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(20.0)
    assert out[2] == pytest.approx(10.0)


# ── Relative volume ───────────────────────────────────────────────────────────

def test_relative_volume_baseline_excludes_the_current_bar() -> None:
    # Index 2: baseline [100, 100] -> 300 / 100 = 3.0.
    # Index 3: baseline [100, 300] -> 50 / 200 = 0.25.
    vols = [100, 100, 300, 50]
    bars = [bar(10, 10, 10, v) for v in vols]
    out = mf.relative_volume(bars, 2)
    assert out[0] is None and out[1] is None
    assert out[2] == pytest.approx(3.0)
    assert out[3] == pytest.approx(0.25)


def test_relative_volume_zero_baseline_is_none() -> None:
    bars = [bar(10, 10, 10, 0), bar(10, 10, 10, 0), bar(10, 10, 10, 100)]
    assert mf.relative_volume(bars, 2)[2] is None


# ── Anchored VWAP ─────────────────────────────────────────────────────────────

def test_anchored_vwap_known_value_and_none_before_anchor() -> None:
    # Typical prices 10, 20, 30 with volumes 100, 300, 100. Anchor at index 1.
    # index 1: 20. index 2: (20*300 + 30*100) / 400 = 22.5.
    bars = [bar(10, 10, 10, 100), bar(20, 20, 20, 300), bar(30, 30, 30, 100)]
    out = mf.anchored_vwap(bars, 1)
    assert out[0] is None
    assert out[1] == pytest.approx(20.0)
    assert out[2] == pytest.approx(22.5)


# ── Missing data is never imputed ─────────────────────────────────────────────

def test_missing_volume_propagates_forward_not_repaired() -> None:
    bars = [bar(10 + k, 9 + k, 10 + k, 100) for k in range(6)]
    bars[3] = bar(13, 12, 13, None)  # missing volume on bar 3
    cmf = mf.chaikin_money_flow(bars, 2)
    assert cmf[3] is None and cmf[4] is None  # windows containing bar 3
    assert cmf[2] is not None and cmf[5] is not None  # windows without it
    assert mf.relative_volume(bars, 2)[4] is None  # baseline includes bar 3
    vpt = mf.volume_price_trend(bars)
    assert vpt[2] is not None and vpt[3] is None and vpt[5] is None
    assert mf.anchored_vwap(bars, 0)[3] is None and mf.anchored_vwap(bars, 0)[5] is None


def test_none_close_breaks_mfi_and_vpt() -> None:
    bars = [bar(10, 10, 10, 100), bar(12, 12, 12, 100), bar(11, 11, 11, 100)]
    bars[1] = {"high": 12, "low": 12, "close": None, "volume": 100}
    assert mf.money_flow_index(bars, 2)[2] is None
    assert mf.volume_price_trend(bars)[2] is None


# ── Causality: a value at i never depends on bars after i ─────────────────────

def _random_bars(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    price = 10.0
    out = []
    for _ in range(n):
        o = price
        c = o * (1 + rng.uniform(-0.03, 0.03))
        h = max(o, c) * (1 + rng.uniform(0, 0.01))
        l = min(o, c) * (1 - rng.uniform(0, 0.01))
        out.append(bar(h, l, c, float(rng.randint(50, 500))))
        price = c
    return out


@pytest.mark.parametrize("fn", [
    lambda b: mf.chaikin_money_flow(b, 5),
    lambda b: mf.money_flow_index(b, 5),
    mf.volume_price_trend,
    lambda b: mf.relative_volume(b, 5),
    lambda b: mf.anchored_vwap(b, 3),
])
def test_values_do_not_change_when_future_bars_are_removed(fn) -> None:
    bars = _random_bars(60, seed=11)
    full = fn(bars)
    for cut in (20, 35, 59):
        truncated = fn(bars[: cut + 1])
        for i in range(cut + 1):
            assert truncated[i] == full[i], (cut, i)


def test_output_length_matches_input() -> None:
    bars = _random_bars(12, seed=2)
    for out in (
        mf.chaikin_money_flow(bars), mf.money_flow_index(bars), mf.volume_price_trend(bars),
        mf.relative_volume(bars), mf.anchored_vwap(bars, 0),
    ):
        assert len(out) == len(bars)
