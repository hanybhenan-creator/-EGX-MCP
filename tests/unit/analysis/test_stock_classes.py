"""Stock classes: each synthetic behaviour lands in the class it was built to show."""

from __future__ import annotations

import math
import random

import pytest

from tradingview_mcp.core.analysis import stock_classes as sc


def make(closes, *, opens=None, volumes=None, start_volume=1_000_000.0):
    bars = []
    for k, c in enumerate(closes):
        o = opens[k] if opens is not None else (closes[k - 1] if k else c)
        v = volumes[k] if volumes is not None else start_volume
        bars.append({"date": f"2026-01-{k + 1:02d}", "open": o, "high": max(o, c) * 1.005,
                     "low": min(o, c) * 0.995, "close": c, "volume": v})
    return bars


def trending(n=120, seed=1):
    rng = random.Random(seed)
    price, closes = 10.0, []
    for _ in range(n):
        price *= 1.01 + rng.uniform(-0.002, 0.002)
        closes.append(price)
    return make(closes)


def ranging(n=120):
    closes = [10.0 if k % 2 == 0 else 10.5 for k in range(n)]
    return make(closes)


def gapping(n=120):
    closes = [10.0 + 0.01 * (k % 3) for k in range(n)]
    opens = [closes[k - 1] * (1.03 if k % 2 else 0.97) if k else closes[0] for k in range(n)]
    return make(closes, opens=opens)


def dormant_by_zero_volume(n=120):
    volumes = [0 if k % 3 == 0 else 1_000_000 for k in range(n)]  # a third of sessions no trade
    return make([10.0 + 0.001 * k for k in range(n)], volumes=volumes)


def thin_by_value(n=120):
    return make([10.0 + 0.01 * (k % 5) for k in range(n)], start_volume=10.0)  # 100 EGP a day


def last(history):
    return history[-1]


# ── features and raw classes ──────────────────────────────────────────────────

def test_constant_price_has_zero_efficiency_and_zero_volatility() -> None:
    feats = sc.features_at(make([10.0] * 80), 79, sc.ClassRules())
    assert feats is not None
    assert feats.efficiency_ratio == 0.0
    assert feats.daily_volatility == 0.0


def test_efficiency_ratio_of_a_straight_line_is_one() -> None:
    closes = [10.0 + k for k in range(80)]
    feats = sc.features_at(make(closes), 79, sc.ClassRules())
    assert feats.efficiency_ratio == pytest.approx(1.0)


def test_zigzag_efficiency_is_small() -> None:
    feats = sc.features_at(ranging(), 119, sc.ClassRules())
    assert feats.efficiency_ratio < 0.05


@pytest.mark.parametrize("builder, expected", [
    (trending, sc.TRENDING),
    (ranging, sc.RANGE_BOUND),
    (gapping, sc.GAP_PRONE),
    (dormant_by_zero_volume, sc.DORMANT),
    (thin_by_value, sc.DORMANT),
])
def test_each_synthetic_behaviour_lands_in_its_class(builder, expected) -> None:
    history = sc.classify_history(builder())
    assert last(history).raw_class == expected, last(history).reason


def test_dormant_takes_precedence_over_trending() -> None:
    # Strong trend, but a third of the sessions have no trade: dormant wins.
    closes = [10.0 * (1.01 ** k) for k in range(120)]
    volumes = [0 if k % 3 == 0 else 1_000_000 for k in range(120)]
    history = sc.classify_history(make(closes, volumes=volumes))
    assert last(history).raw_class == sc.DORMANT


def test_short_history_is_unclassified_not_guessed() -> None:
    history = sc.classify_history(trending(n=30))
    assert all(a.raw_class == sc.UNCLASSIFIED for a in history)


def test_a_missing_close_inside_the_window_is_unclassified() -> None:
    bars = trending(n=180)
    bars[100]["close"] = None
    assert sc.features_at(bars, 110, sc.ClassRules()) is None      # window 50..110 contains bar 100
    assert sc.features_at(bars, 170, sc.ClassRules()) is not None  # window 110..170 does not


def test_high_volatility_is_a_flag_not_a_class() -> None:
    rng = random.Random(3)
    closes, price = [], 10.0
    for _ in range(120):
        price *= 1 + rng.choice([-0.08, 0.08])
        closes.append(price)
    history = sc.classify_history(make(closes))
    assert "HIGH_VOLATILITY" in last(history).flags
    assert last(history).official_class in (sc.MIXED, sc.RANGE_BOUND, sc.TRENDING, sc.GAP_PRONE)


# ── confirmation rule: stocks move between classes, but not on one noisy day ─

def test_a_short_blip_does_not_change_the_official_class(monkeypatch) -> None:
    """Deterministic: the daily raw class is scripted, so only the confirmation rule is tested."""
    from types import SimpleNamespace

    T, R = sc.TRENDING, sc.RANGE_BOUND
    # 10 trending, a 2-session blip, 5 trending, 5 ranging, 5 trending.
    script = [T] * 10 + [R] * 2 + [T] * 5 + [R] * 5 + [T] * 5
    counter = {"i": -1}

    def fake_features(bars, i, rules=None):
        counter["i"] += 1
        return SimpleNamespace(daily_volatility=0.0)

    monkeypatch.setattr(sc, "features_at", fake_features)
    monkeypatch.setattr(sc, "raw_class", lambda feats, rules=None: (script[counter["i"]], "scripted"))

    history = sc.classify_history([{"date": str(k), "close": 1.0} for k in range(len(script))])
    official = [a.official_class for a in history]
    assert official[:12] == [T] * 12          # the two-session blip (indices 10-11) is ignored
    assert official[12:17] == [T] * 5
    assert official[17:21] == [T] * 4         # four ranging sessions: not yet confirmed
    assert official[21] == R                  # the fifth consecutive ranging session switches it
    assert official[21:26] == [R] * 5         # four trending sessions (22-25) are not enough...
    assert official[25] == R
    assert official[26] == T                  # ...the fifth confirms the return to trending


def test_a_sustained_change_moves_the_stock_after_confirmation() -> None:
    # The 60-session window lags: after a trend, 40 range sessions are not yet enough
    # to dilute the trend's efficiency. The stock moves once the window has caught up.
    bars = trending(n=100) + ranging(n=120)
    history = sc.classify_history(bars)
    changes = sc.transitions(history, bars)
    moves = [c for c in changes if c["from"] == sc.TRENDING and c["to"] == sc.RANGE_BOUND]
    assert moves, "a sustained range should eventually move the stock"
    first_range = next(i for i, a in enumerate(history) if a.official_class == sc.RANGE_BOUND)
    assert first_range >= 100 + sc.ClassRules().confirm_sessions - 1
    assert history[100 + 30].official_class != sc.RANGE_BOUND  # the window is still mostly trend


# ── causality: the class at session i never depends on later sessions ────────

@pytest.mark.parametrize("builder", [trending, ranging, gapping])
def test_history_is_unchanged_when_later_bars_are_removed(builder) -> None:
    bars = builder()
    full = sc.classify_history(bars)
    for cut in (70, 95, 119):
        truncated = sc.classify_history(bars[: cut + 1])
        for i in range(cut + 1):
            assert truncated[i].official_class == full[i].official_class, (cut, i)
            assert truncated[i].raw_class == full[i].raw_class, (cut, i)


# ── policy: no tool is allowed until it is validated; names must exist ────────

def test_every_class_has_a_policy_and_no_tool_is_validated_yet() -> None:
    assert set(sc.CLASS_POLICY) == set(sc.CLASSES)
    assert all(sc.allowed_tools(c) == () for c in sc.CLASSES)


def test_dormant_excludes_volume_based_tools_by_name() -> None:
    from tradingview_mcp.core.calibration.signals import register_builtins, signal_names

    register_builtins()
    known = set(signal_names())
    excluded = set(sc.CLASS_POLICY[sc.DORMANT]["excluded"])
    assert excluded, "dormant class must exclude something"
    assert excluded <= known, excluded - known
