"""Wyckoff / VSA events: definitions, causality, and the planted-versus-noise controls.

The controls are synthetic. They show the engine can see a planted post-event
drift of the designed size, and that on noise it does not call more than a few
events VALIDATED. The seeds are fixed, so these are regression checks, not
statistical guarantees. They say nothing about whether the events exist in EGX data.
"""

from __future__ import annotations

import pytest

from tests.unit.calibration.wyckoff_fixtures import wyckoff_universe
from tradingview_mcp.core.analysis import wyckoff
from tradingview_mcp.core.calibration import Sample, analyze
from tradingview_mcp.core.calibration.signals import WYCKOFF_CONDITIONS, get_signal
from tradingview_mcp.core.services import egx_wyckoff_state_service as svc


def bar(o, h, l, c, v, day=0):
    return {"date": f"2026-01-{(day % 28) + 1:02d}", "open": o, "high": h, "low": l, "close": c, "volume": v}


def base(n=20):
    """Flat range: lows 9.5, highs 10.5, volume 1000, spread 1.0."""
    return [bar(10.0, 10.5, 9.5, 10.0, 1000.0, day=i) for i in range(n)]


def last_flag(flags):
    return flags[-1]


# ── hand-built definitions ──────────────────────────────────────────────────


def test_selling_climax_fires_on_stopping_volume():
    bars = base() + [bar(10.0, 10.2, 8.8, 9.6, 2500.0, day=20)]
    assert last_flag(wyckoff.selling_climax_flags(bars)) is True
    assert last_flag(wyckoff.spring_flags(bars)) is False  # volume too high for a spring


def test_selling_climax_rejects_close_in_lower_half():
    bars = base() + [bar(10.0, 10.2, 8.8, 9.0, 2500.0, day=20)]  # close location 0.14
    assert last_flag(wyckoff.selling_climax_flags(bars)) is False


def test_spring_fires_on_undercut_and_recovery_on_low_volume():
    bars = base() + [bar(9.8, 9.9, 9.3, 9.7, 800.0, day=20)]
    assert last_flag(wyckoff.spring_flags(bars)) is True
    assert last_flag(wyckoff.selling_climax_flags(bars)) is False


def test_spring_rejects_close_below_support():
    bars = base() + [bar(9.8, 9.9, 9.3, 9.4, 800.0, day=20)]
    assert last_flag(wyckoff.spring_flags(bars)) is False


def test_no_supply_fires_on_narrow_down_bar_on_low_volume():
    bars = base() + [bar(10.0, 10.1, 9.7, 9.9, 600.0, day=20)]
    assert last_flag(wyckoff.no_supply_flags(bars)) is True


def test_no_supply_rejects_wide_spread():
    bars = base() + [bar(10.0, 10.4, 9.6, 9.9, 600.0, day=20)]  # range 0.8 ok, but close loc 0.375
    assert last_flag(wyckoff.no_supply_flags(bars)) is False


def test_secondary_test_follows_a_climax_that_holds():
    bars = base() + [
        bar(10.0, 10.2, 8.8, 9.6, 2500.0, day=20),  # climax, low 8.8
        bar(9.6, 9.7, 8.9, 9.4, 1200.0, day=21),  # retest, low within 3%, volume < 0.8×climax
    ]
    flags = wyckoff.secondary_test_flags(bars)
    assert flags[20] is False
    assert flags[21] is True


def test_secondary_test_is_invalidated_when_the_climax_low_is_undercut():
    bars = base() + [
        bar(10.0, 10.2, 8.8, 9.6, 2500.0, day=20),
        bar(9.6, 9.7, 8.7, 9.4, 1200.0, day=21),  # undercuts 8.8
        bar(9.4, 9.6, 8.9, 9.3, 1100.0, day=22),  # would have matched, but the climax is dead
    ]
    flags = wyckoff.secondary_test_flags(bars)
    assert flags[21] is False
    assert flags[22] is False


def test_secondary_test_expires_after_its_window():
    climax = [bar(10.0, 10.2, 8.8, 9.6, 2500.0, day=20)]
    quiet = [bar(9.6, 9.7, 9.3, 9.5, 1000.0, day=21 + k) for k in range(wyckoff.ST_WINDOW)]
    retest = [bar(9.5, 9.6, 8.9, 9.3, 1000.0, day=60)]
    flags = wyckoff.secondary_test_flags(base() + climax + quiet + retest)
    assert flags[-1] is False


def test_flags_are_none_before_the_lookback():
    flags = wyckoff.selling_climax_flags(base(25))
    assert all(f is None for f in flags[: wyckoff.LOOKBACK])


def test_dead_flat_line_produces_no_events():
    flat = [bar(10.0, 10.0, 10.0, 10.0, 1000.0, day=i) for i in range(80)]
    for name, fn in wyckoff.FLAG_BUILDERS.items():
        assert not any(f for f in fn(flat)), name


# ── causality: a flag at session i must not change when later bars are removed ──


@pytest.mark.parametrize("event", wyckoff.EVENTS)
def test_flags_are_causal(event):
    universe = wyckoff_universe(n=3, sessions=160, plant=False, seed=21)
    fn = wyckoff.FLAG_BUILDERS[event]
    for bars in universe.values():
        full = fn(bars)
        for i in range(wyckoff.LOOKBACK + 1, len(bars), 17):
            assert fn(bars[: i + 1])[i] == full[i], (event, i)


# ── registry ────────────────────────────────────────────────────────────────


def test_wyckoff_conditions_are_registered_outside_the_default_batch():
    from tradingview_mcp.core.services.egx_calibration_service import DEFAULT_SIGNALS

    assert set(WYCKOFF_CONDITIONS) == {"wy_selling_climax", "wy_spring", "wy_no_supply", "wy_secondary_test"}
    for name in WYCKOFF_CONDITIONS:
        spec = get_signal(name)
        assert spec.kind == "wyckoff"
        assert spec.hypothesis
        assert name not in DEFAULT_SIGNALS


# ── planted versus noise controls (fixed seeds) ─────────────────────────────


@pytest.fixture(scope="module")
def planted_verdicts():
    out = {}
    for seed in range(8):
        u = wyckoff_universe(plant=True, seed=seed, drift=0.0025)
        rep = analyze(Sample(u, horizon=5), signals=WYCKOFF_CONDITIONS, permutations=200, seed=seed)
        out[seed] = {m.signal: m.verdict for m in rep.measurements}
    return out


@pytest.fixture(scope="module")
def noise_verdicts():
    out = {}
    for seed in range(8):
        u = wyckoff_universe(plant=False, seed=seed, drift=0.0)
        rep = analyze(Sample(u, horizon=5), signals=WYCKOFF_CONDITIONS, permutations=200, seed=seed)
        out[seed] = {m.signal: m.verdict for m in rep.measurements}
    return out


def test_planted_post_event_drift_is_detected_on_the_climax(planted_verdicts):
    # Observed 5 of 8 seeds at this effect size (20-seed run: 12 of 20). Floor set below that.
    hits = sum(1 for v in planted_verdicts.values() if v["wy_selling_climax"] == "VALIDATED")
    assert hits >= 3


def test_noise_rarely_validates_any_wyckoff_event(noise_verdicts):
    # Observed 2 VALIDATED out of 32 event verdicts on these seeds (about 6%), nominal alpha 0.10.
    total = sum(1 for v in noise_verdicts.values() for verdict in v.values() if verdict == "VALIDATED")
    assert total <= 4


# ── per-symbol state service ────────────────────────────────────────────────


def _fake_calibration(verdict="VALIDATED"):
    return {
        "mode": "offline",
        "measurement": {"symbols_analysed": 14, "sessions": 380},
        "warnings": [],
        "measurements": [
            {
                "signal": name,
                "verdict": verdict,
                "events": 80,
                "hit_rate": 0.6,
                "base_rate": 0.5,
                "excess_hit": 0.1,
                "q_value_hit": 0.02,
            }
            for name in WYCKOFF_CONDITIONS
        ],
        "walk_forward": {"wy_selling_climax": {"verdict": "MODERATE"}},
    }


def _symbol_bars_with_climax(n=60):
    bars = base(n - 1) + [bar(10.0, 10.2, 8.8, 9.6, 2500.0, day=n)]
    for i, b in enumerate(bars):
        b["date"] = f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"
    return bars


def test_state_reports_recent_climax_with_its_measured_verdict(monkeypatch):
    from datetime import date

    bars = _symbol_bars_with_climax()
    today = date.today().isoformat()
    bars[-1]["date"] = today
    monkeypatch.setattr(svc, "egx_event_study", lambda **kw: _fake_calibration())
    monkeypatch.setattr(svc, "_load_bars", lambda symbols, *, lookback_days, max_symbols: ({"COMI": bars}, []))
    out = svc.egx_symbol_events(symbol="comi", lookback_sessions=3)
    assert "error" not in out
    assert out["state"] == "EVENT_FIRED"
    assert out["symbol"] == "COMI"
    assert any(e["event"] == "selling_climax" and e["measured_verdict"] == "VALIDATED" for e in out["recent_events"])
    assert "selling_climax" in out["validated_events_recent"]
    assert "not a trade instruction" in out["disclaimer"]
    assert not any("days old" in w for w in out["warnings"])


def test_state_reports_no_event_when_nothing_fired(monkeypatch):
    bars = base(60)
    monkeypatch.setattr(svc, "egx_event_study", lambda **kw: _fake_calibration("REJECTED"))
    monkeypatch.setattr(svc, "_load_bars", lambda symbols, *, lookback_days, max_symbols: ({"COMI": bars}, []))
    out = svc.egx_symbol_events(symbol="COMI", lookback_sessions=5)
    assert out["state"] == "NO_RECENT_EVENT"
    assert out["recent_events"] == []


def test_state_rejects_bad_symbol_and_window():
    assert "error" in svc.egx_symbol_events(symbol="COMI; DROP")
    assert "error" in svc.egx_symbol_events(symbol="COMI", lookback_sessions=0)


def test_state_reports_no_data(monkeypatch):
    monkeypatch.setattr(svc, "egx_event_study", lambda **kw: _fake_calibration())
    monkeypatch.setattr(svc, "_load_bars", lambda symbols, *, lookback_days, max_symbols: ({}, []))
    out = svc.egx_symbol_events(symbol="COMI")
    assert "error" in out
    assert "daily" in str(out)


def test_secondary_test_rejects_a_retest_far_above_the_climax_low():
    bars = base() + [
        bar(10.0, 10.2, 8.8, 9.6, 2500.0, day=20),  # climax, low 8.8
        bar(9.6, 9.9, 9.3, 9.7, 1200.0, day=21),  # low 9.3 is ~5.7% above 8.8, outside the 3% band
    ]
    assert wyckoff.secondary_test_flags(bars)[21] is False
