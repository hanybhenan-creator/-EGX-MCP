"""egx_price_levels: readings and candidate levels, with no recommendation fields.

The live TradingView call is replaced by a fake analysis so the output shape and
the nearest-level arithmetic can be checked against known values.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tradingview_mcp.core.services import egx_service as svc

FORBIDDEN_KEYS = {
    "recommendation", "trade_setup", "targets", "stop_loss", "risk_reward",
    "entry_points", "tv_recommendation", "stock_score", "grade",
    "trade_quality_score", "trade_quality", "scenarios", "primary_scenario",
}


@pytest.fixture
def fake_market(monkeypatch: pytest.MonkeyPatch) -> None:
    indicators = {"close": 100.0, "ATR": 2.5, "high": 101.0, "low": 99.0}

    def analysis(**_kwargs):
        return {"EGX:COMI": SimpleNamespace(indicators=indicators)}

    monkeypatch.setattr(svc, "_TA_AVAILABLE", True)
    monkeypatch.setattr(svc, "get_multiple_analysis", analysis)
    monkeypatch.setattr(svc, "compute_metrics", lambda ind: {"price": 100.0})
    monkeypatch.setattr(svc, "compute_stock_score", lambda ind, currency: {
        "change_pct": 1.25, "liquidity": {"avg_traded_value": 12_000_000}, "score": 88, "grade": "A",
    })
    monkeypatch.setattr(svc, "compute_trade_setup", lambda ind: {
        "supports": [97.0, 93.0, 90.0],        # nearest below 100 is 97
        "resistances": [104.0, 110.0, 120.0],  # nearest above 100 is 104
        "entry_points": {}, "stop_loss": 95.0, "targets": [110.0], "risk_reward": {"to_target_2": 3.0},
    })
    monkeypatch.setattr(svc, "extract_extended_indicators", lambda ind: {
        "rsi": {"value": 55}, "macd": {"value": 0.1}, "adx": {"value": 20},
        "volume": {"value": 1.0}, "ema": {"ema20": 99}, "bollinger_bands": {"upper": 105},
        "tv_recommendation": "BUY",
    })


def test_output_has_no_recommendation_or_trade_fields(fake_market: None) -> None:
    out = svc.generate_egx_price_levels("COMI", "1D")
    leaked = FORBIDDEN_KEYS & set(out)
    assert not leaked, leaked
    assert "BUY" not in str(out)  # the third-party label must not leak through any field


def test_nearest_levels_and_distances_are_correct(fake_market: None) -> None:
    levels = svc.generate_egx_price_levels("COMI", "1D")["candidate_levels"]
    assert levels["nearest_support"] == 97.0
    assert levels["nearest_resistance"] == 104.0
    assert levels["distance_to_nearest_support_pct"] == pytest.approx(3.0)
    assert levels["distance_to_nearest_resistance_pct"] == pytest.approx(4.0)


def test_atr_percentage_of_price(fake_market: None) -> None:
    readings = svc.generate_egx_price_levels("COMI", "1D")["readings"]
    assert readings["atr_pct_of_price"] == pytest.approx(2.5)


def test_states_that_it_is_not_a_recommendation(fake_market: None) -> None:
    out = svc.generate_egx_price_levels("COMI", "1D")
    assert "does not recommend" in out["not_a_recommendation"]
    assert "has been calibrated against later price moves" in out["how_to_read"]


def test_missing_symbol_returns_an_error_not_empty_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(svc, "_TA_AVAILABLE", True)
    monkeypatch.setattr(svc, "get_multiple_analysis", lambda **_: {})
    out = svc.generate_egx_price_levels("NOPE", "1D")
    assert "error" in out and "candidate_levels" not in out


def test_no_support_or_resistance_gives_none_not_zero(fake_market: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(svc, "compute_trade_setup", lambda ind: None)
    levels = svc.generate_egx_price_levels("COMI", "1D")["candidate_levels"]
    assert levels["nearest_support"] is None and levels["nearest_resistance"] is None
    assert levels["distance_to_nearest_support_pct"] is None
