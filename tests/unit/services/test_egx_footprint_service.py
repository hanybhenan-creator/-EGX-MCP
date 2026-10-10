"""The footprint service: refusals, the measured quality score, payload shape."""

from __future__ import annotations

import pytest

from tradingview_mcp.core.data.calendar import EGXCalendar
from tradingview_mcp.core.data.factory import DataLayer
from tradingview_mcp.core.data.transport import FakeTransport
from tests.unit.calibration.conftest import random_walk


@pytest.fixture
def layer(tmp_path, monkeypatch) -> DataLayer:
    built = DataLayer(
        transport=FakeTransport(),
        db_path=tmp_path / "footprint.db",
        calendar=EGXCalendar(holidays={}),
        include_tradingview=False,
    )
    built.store.write_ohlcv("COMI", random_walk(300, seed=1), source="test")
    built.store.write_ohlcv("FLAT", random_walk(300, seed=2, drift=0.0, sigma=0.0005), source="test")
    monkeypatch.setattr("tradingview_mcp.core.data.factory.get_layer", lambda: built)
    yield built
    built.close()


class TestRefusals:
    def test_unknown_symbol_returns_no_data_with_the_command(self, layer) -> None:
        from tradingview_mcp.core.services.egx_footprint_service import egx_accumulation_fingerprint

        payload = egx_accumulation_fingerprint("NOPE")
        assert payload["error"]["code"] == "NO_DATA"
        assert "egx_data_sync.py" in payload["error"]["message"]

    def test_empty_symbol_is_refused(self, layer) -> None:
        from tradingview_mcp.core.services.egx_footprint_service import egx_accumulation_fingerprint

        assert egx_accumulation_fingerprint("  ")["error"]["code"] == "INVALID_PARAMETER"

    @pytest.mark.parametrize("kwargs", [{"lookback_days": 1}, {"lookback_days": 500},
                                        {"zone_lookback": 5}, {"zone_lookback": 999}])
    def test_out_of_range_parameters(self, layer, kwargs) -> None:
        from tradingview_mcp.core.services.egx_footprint_service import egx_accumulation_fingerprint

        assert egx_accumulation_fingerprint("COMI", **kwargs)["error"]["code"] == "INVALID_PARAMETER"

    def test_short_history_is_refused_rather_than_guessed(self, tmp_path, monkeypatch) -> None:
        from tradingview_mcp.core.services.egx_footprint_service import egx_accumulation_fingerprint

        thin = DataLayer(
            transport=FakeTransport(),
            db_path=tmp_path / "thin.db",
            calendar=EGXCalendar(holidays={}),
            include_tradingview=False,
        )
        thin.store.write_ohlcv("TINY", random_walk(10, seed=3), source="test")
        monkeypatch.setattr("tradingview_mcp.core.data.factory.get_layer", lambda: thin)
        try:
            payload = egx_accumulation_fingerprint("TINY")
        finally:
            thin.close()
        assert payload["error"]["code"] == "NO_DATA"
        assert payload["error"]["stored_bars"] == 10  # extra keys sit on the envelope


class TestPayload:
    def test_reports_the_footprint_vocabulary(self, layer) -> None:
        from tradingview_mcp.core.services.egx_footprint_service import egx_accumulation_fingerprint

        payload = egx_accumulation_fingerprint("COMI", zone_lookback=120)
        assert "error" not in payload
        assert payload["signal"] in {s.value for s in __import__(
            "tradingview_mcp.core.analysis.footprints", fromlist=["SmartMoneySignal"]
        ).SmartMoneySignal}
        assert payload["phase"].startswith("phase_")
        assert payload["bars_used"] == 300
        assert payload["window"]["from"] and payload["window"]["to"]
        for key in ("absorption", "breakout", "targets", "recommendation", "rationale"):
            assert key in payload
        assert set(payload["targets"]) == {"conservative", "breakout", "aggressive", "risk_zone_low", "current_price"}

    def test_every_number_states_what_it_is_not(self, layer) -> None:
        from tradingview_mcp.core.services.egx_footprint_service import egx_accumulation_fingerprint

        payload = egx_accumulation_fingerprint("COMI")
        assert "not probabilities" in payload["note"]
        assert any("order flow" in item for item in payload["unmeasured"])

    def test_quality_score_components_are_all_reported(self, layer) -> None:
        from tradingview_mcp.core.services.egx_footprint_service import egx_accumulation_fingerprint

        quality = egx_accumulation_fingerprint("COMI")["data_quality"]
        assert quality["grade"] in {"A", "B", "C", "D", "F"}
        assert 0 <= quality["score"] <= 100
        assert quality["components"]["bars"] == 300
        assert quality["components"]["newest_settled_session"]
        assert quality["components"]["integrity"] == 1.0  # fixture bars are structurally valid
        assert isinstance(quality["notes"], list)

    def test_source_reliability_is_unmeasured_not_assumed_high(self, layer) -> None:
        from tradingview_mcp.core.services.egx_footprint_service import egx_accumulation_fingerprint

        payload = egx_accumulation_fingerprint("COMI")
        assert payload["source_reliability"] == "UNMEASURED"
        assert any("unmeasured" in note for note in payload["data_quality"]["notes"])

    def test_a_flat_symbol_is_not_called_an_accumulation(self, layer) -> None:
        from tradingview_mcp.core.services.egx_footprint_service import egx_accumulation_fingerprint

        payload = egx_accumulation_fingerprint("FLAT", zone_lookback=120)
        assert payload["signal"] in {"neutral", "hidden_accumulation", "mild_accumulation"}
        assert payload["signal"] != "strong_accumulation"

    def test_egx_prefix_and_lowercase_are_accepted(self, layer) -> None:
        from tradingview_mcp.core.services.egx_footprint_service import egx_accumulation_fingerprint

        assert "error" not in egx_accumulation_fingerprint("egx:comi")


class TestQualityScorer:
    def test_missing_bars_score_zero(self, layer) -> None:
        from tradingview_mcp.core.services.egx_footprint_service import assess_data_quality

        result = assess_data_quality([], symbol="X")
        assert result["score"] == 0.0 and result["grade"] == "F"

    def test_invalid_bars_lower_integrity(self, layer) -> None:
        from tradingview_mcp.core.services.egx_footprint_service import assess_data_quality

        bars = random_walk(60, seed=4)
        clean = assess_data_quality(bars, symbol="X")
        bars[5]["high"] = bars[5]["low"] - 1.0  # high below low: structurally invalid
        dirty = assess_data_quality(bars, symbol="X")
        assert dirty["components"]["integrity"] < clean["components"]["integrity"]
        assert any("structural" in note for note in dirty["notes"])

    def test_stale_history_loses_freshness_but_not_integrity(self, layer) -> None:
        from tradingview_mcp.core.services.egx_footprint_service import assess_data_quality

        result = assess_data_quality(random_walk(120, seed=5), symbol="X")
        assert result["components"]["integrity"] == 1.0
        assert result["components"]["freshness"] is not None
        assert result["components"]["freshness"] < 1.0
