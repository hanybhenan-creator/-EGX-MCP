"""The calibration service: refusals, payload shape, and offline safety.

The service must never invent a measurement. When the store is empty it says
so and names the command that fills it; when a parameter is wrong it says
which one. Every test here runs with the network unreachable by construction.
"""

from __future__ import annotations

import pytest

from tradingview_mcp.core.data.calendar import EGXCalendar
from tradingview_mcp.core.data.factory import DataLayer
from tradingview_mcp.core.data.transport import FakeTransport
from tests.unit.calibration.conftest import random_walk


def _store_bars(layer: DataLayer, symbols: list[str]) -> None:
    for index, symbol in enumerate(symbols):
        layer.store.write_ohlcv(
            symbol,
            random_walk(300, seed=index + 1, boost_after_breakout=0.004 if index < 2 else 0.0),
            source="test",
        )


@pytest.fixture
def layer(tmp_path, monkeypatch) -> DataLayer:
    built = DataLayer(
        transport=FakeTransport(),
        db_path=tmp_path / "calibration.db",
        calendar=EGXCalendar(holidays={}),
        include_tradingview=False,
    )
    _store_bars(built, [f"SYM{i}" for i in range(6)])
    monkeypatch.setattr("tradingview_mcp.core.data.factory.get_layer", lambda: built)
    yield built
    built.close()


PARAMS = {"symbols": "SYM0,SYM1,SYM2,SYM3,SYM4,SYM5", "lookback_days": 3000, "max_symbols": 10}


class TestRefusals:
    def test_empty_store_returns_no_data_with_the_command_to_run(self, tmp_path, monkeypatch) -> None:
        from tradingview_mcp.core.services.egx_calibration_service import egx_event_study

        empty = DataLayer(
            transport=FakeTransport(),
            db_path=tmp_path / "empty.db",
            calendar=EGXCalendar(holidays={}),
            include_tradingview=False,
        )
        monkeypatch.setattr("tradingview_mcp.core.data.factory.get_layer", lambda: empty)
        try:
            payload = egx_event_study(symbols="NOPE1,NOPE2")
        finally:
            empty.close()
        assert payload["error"]["code"] == "NO_DATA"
        assert "egx_data_sync.py" in payload["error"]["message"]

    def test_unknown_signal_is_an_invalid_parameter(self, layer) -> None:
        from tradingview_mcp.core.services.egx_calibration_service import egx_event_study

        payload = egx_event_study(signal="not_a_signal", **PARAMS)
        assert payload["error"]["code"] == "INVALID_PARAMETER"

    def test_unknown_index_is_an_invalid_parameter(self, layer) -> None:
        from tradingview_mcp.core.services.egx_calibration_service import egx_event_study

        payload = egx_event_study(index="EGX999")
        assert payload["error"]["code"] == "INVALID_PARAMETER"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"horizon": 0},
            {"horizon": 999},
            {"permutations": -1},
            {"permutations": 99_999},
            {"max_symbols": 0},
            {"max_symbols": 9999},
        ],
    )
    def test_out_of_range_parameters_are_refused(self, layer, kwargs) -> None:
        from tradingview_mcp.core.services.egx_calibration_service import egx_event_study

        payload = egx_event_study(**{**PARAMS, **kwargs})
        assert payload["error"]["code"] == "INVALID_PARAMETER"


class TestPayload:
    def test_default_batch_measures_every_default_signal(self, layer) -> None:
        from tradingview_mcp.core.services.egx_calibration_service import (
            DEFAULT_SIGNALS,
            egx_event_study,
        )

        payload = egx_event_study(permutations=0, walk_forward_folds=0, **PARAMS)
        assert "error" not in payload
        assert len(payload["measurements"]) == len(DEFAULT_SIGNALS)

    def test_measurement_carries_calibrated_fields(self, layer) -> None:
        from tradingview_mcp.core.services.egx_calibration_service import egx_event_study

        payload = egx_event_study(signal="c2", permutations=20, walk_forward_folds=0, **PARAMS)
        assert "error" not in payload
        row = payload["measurements"][0]
        for key in (
            "signal", "title", "hypothesis", "events", "symbols", "event_dates",
            "hit_rate", "hit_rate_ci95", "base_rate", "excess_hit",
            "mean_return", "mean_excess", "p_value_hit", "q_value_hit", "verdict",
        ):
            assert key in row, f"missing {key}"
        if row["events"]:
            assert row["hit_rate"] is not None and row["base_rate"] is not None
            assert row["hit_rate_ci95"][0] <= row["hit_rate"] <= row["hit_rate_ci95"][1]

    def test_scope_is_reported(self, layer) -> None:
        from tradingview_mcp.core.services.egx_calibration_service import egx_event_study

        payload = egx_event_study(signal="c4", permutations=0, walk_forward_folds=0, **PARAMS)
        scope = payload["measurement"]
        assert scope["symbols_analysed"] == 6
        assert scope["sessions"] >= 250
        assert scope["window"]["from"] and scope["window"]["to"]
        assert "block permutation" in scope["method"]

    def test_walk_forward_is_attached_when_requested(self, layer) -> None:
        from tradingview_mcp.core.services.egx_calibration_service import egx_event_study

        payload = egx_event_study(
            signal="c2", permutations=0, walk_forward_folds=2, **PARAMS
        )
        assert isinstance(payload["walk_forward"], dict)

    def test_validated_signals_list_is_present(self, layer) -> None:
        from tradingview_mcp.core.services.egx_calibration_service import egx_event_study

        payload = egx_event_study(permutations=0, walk_forward_folds=0, **PARAMS)
        assert isinstance(payload["validated_signals"], list)
        assert payload["disclaimer"]

    def test_offline_mode_is_reported_and_does_not_block_calibration(self, layer, monkeypatch) -> None:
        from tradingview_mcp.core.services.egx_calibration_service import egx_event_study

        monkeypatch.setenv("EGX_DATA_OFFLINE", "1")
        payload = egx_event_study(signal="c4", permutations=0, walk_forward_folds=0, **PARAMS)
        assert "error" not in payload
        assert payload["mode"] == "offline"

    def test_short_history_is_flagged_not_hidden(self, tmp_path, monkeypatch) -> None:
        from tradingview_mcp.core.services.egx_calibration_service import egx_event_study

        short = DataLayer(
            transport=FakeTransport(),
            db_path=tmp_path / "short.db",
            calendar=EGXCalendar(holidays={}),
            include_tradingview=False,
        )
        short.store.write_ohlcv("TINY", random_walk(70, seed=1), source="test")
        monkeypatch.setattr("tradingview_mcp.core.data.factory.get_layer", lambda: short)
        try:
            payload = egx_event_study(
                signal="c4", symbols="TINY", lookback_days=3000, permutations=0, walk_forward_folds=0
            )
        finally:
            short.close()
        assert "error" not in payload
        assert any("symbol" in w for w in payload["warnings"])


class TestSignalCatalogue:
    def test_lists_signals_with_hypotheses(self, layer) -> None:
        from tradingview_mcp.core.services.egx_calibration_service import egx_list_events

        payload = egx_list_events()
        assert payload["count"] >= 15
        names = {s["name"] for s in payload["signals"]}
        assert {"c1", "c2", "c3", "c3_legacy", "accumulation_legacy", "distribution_corrected"} <= names
        for entry in payload["signals"]:
            assert entry["title"] and entry["hypothesis"] is not None
        assert [s for s in payload["signals"] if s["default"]]
