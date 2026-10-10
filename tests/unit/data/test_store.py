"""Store: schema migrations, provenance-aware writes, gaps, audit queries."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from tradingview_mcp.core.data.calendar import EGXCalendar
from tradingview_mcp.core.data.provenance import QualityFlag, Sourced, provenance_for
from tradingview_mcp.core.data.store import SCHEMA_VERSION, DataStore


@pytest.fixture
def store(tmp_path) -> DataStore:
    db = DataStore(tmp_path / "egx_test.db")
    yield db
    db.close()


def bars(*days_and_closes) -> list[dict]:
    out = []
    for day, close in days_and_closes:
        out.append(
            {
                "date": day,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 100_000,
            }
        )
    return out


class TestSchema:
    def test_migrates_to_current_version(self, store: DataStore) -> None:
        assert store.schema_version() == SCHEMA_VERSION

    def test_migration_is_idempotent(self, store: DataStore) -> None:
        assert store.migrate() == SCHEMA_VERSION
        assert store.schema_version() == SCHEMA_VERSION

    def test_reopening_an_existing_database_does_not_recreate_it(self, tmp_path) -> None:
        path = tmp_path / "reopen.db"
        first = DataStore(path)
        first.upsert_symbol("COMI", name="CIB")
        first.close()

        second = DataStore(path)
        try:
            assert second.stats().symbols == 1
            assert second.schema_version() == SCHEMA_VERSION
        finally:
            second.close()


class TestSymbols:
    def test_upsert_symbol_records_source(self, store: DataStore) -> None:
        store.upsert_symbol(
            "comi", name="Commercial International Bank", sector="banks", source="egx_official"
        )
        row = store.get_symbols()[0]
        assert row["symbol"] == "COMI"
        assert row["name"] == "Commercial International Bank"
        assert row["sector"] == "banks"
        assert row["last_source"] == "egx_official"
        assert row["is_active"] == 1

    def test_bulk_upsert_counts_rows(self, store: DataStore) -> None:
        n = store.upsert_symbols(
            [{"symbol": "COMI", "name": "CIB"}, {"symbol": "TMGH"}, {"symbol": ""}],
            source="egx_official",
        )
        assert n == 2

    def test_symbols_are_stored_uppercase(self, store: DataStore) -> None:
        store.upsert_symbols([{"symbol": "comi"}])
        assert store.stats().symbols == 1


class TestOHLCV:
    def test_round_trip(self, store: DataStore) -> None:
        written = store.write_ohlcv("COMI", bars(("2026-03-01", 10.0), ("2026-03-02", 11.0)), source="yahoo")
        assert written == 2
        rows = store.get_ohlcv("COMI")
        assert [r["close"] for r in rows] == [10.0, 11.0]
        assert rows[0]["source"] == "yahoo"

    def test_upsert_replaces_values_but_keeps_one_row(self, store: DataStore) -> None:
        store.write_ohlcv("COMI", bars(("2026-03-01", 10.0)), source="yahoo")
        store.write_ohlcv("COMI", bars(("2026-03-01", 10.4)), source="egx_official")
        rows = store.get_ohlcv("COMI")
        assert len(rows) == 1
        assert rows[0]["close"] == 10.4
        assert rows[0]["source"] == "egx_official"

    def test_every_source_observation_is_kept(self, store: DataStore) -> None:
        store.write_ohlcv("COMI", bars(("2026-03-01", 10.0)), source="yahoo")
        store.write_ohlcv("COMI", bars(("2026-03-01", 10.4)), source="egx_official")
        obs = store.observations_for("COMI", "2026-03-01")
        assert {o["source"] for o in obs} == {"yahoo", "egx_official"}
        assert {o["close"] for o in obs} == {10.0, 10.4}

    def test_quality_flags_are_persisted(self, store: DataStore) -> None:
        store.write_ohlcv(
            "COMI", bars(("2026-03-01", 10.0)), source="yahoo", quality_flags=(QualityFlag.OUTLIER,)
        )
        assert store.get_ohlcv("COMI")[0]["quality_flags"] == "OUTLIER"

    def test_limit_returns_the_newest_then_sorts(self, store: DataStore) -> None:
        store.write_ohlcv(
            "COMI",
            bars(("2026-03-01", 10.0), ("2026-03-02", 11.0), ("2026-03-03", 12.0)),
            source="yahoo",
        )
        rows = store.get_ohlcv("COMI", limit=2)
        assert [r["date"] for r in rows] == ["2026-03-02", "2026-03-03"]

    def test_latest_date(self, store: DataStore) -> None:
        store.write_ohlcv("COMI", bars(("2026-03-01", 10.0), ("2026-03-03", 12.0)), source="yahoo")
        assert store.latest_ohlcv_date("COMI") == "2026-03-03"

    def test_rows_without_a_date_are_skipped(self, store: DataStore) -> None:
        assert store.write_ohlcv("COMI", [{"close": 1.0}], source="yahoo") == 0


class TestGaps:
    def test_gaps_are_computed_against_the_egx_calendar(self, store: DataStore) -> None:
        # 2026-03-01 Sun … 03-05 Thu is a full EGX week.
        cal = EGXCalendar(holidays={})
        store.write_ohlcv(
            "COMI",
            bars(("2026-03-01", 10.0), ("2026-03-02", 11.0)),
            source="yahoo",
        )
        gaps = store.ohlcv_gaps("COMI", date(2026, 3, 1), date(2026, 3, 5), calendar=cal)
        assert gaps == ["2026-03-03", "2026-03-04", "2026-03-05"]

    def test_weekend_is_never_reported_as_a_gap(self, store: DataStore) -> None:
        cal = EGXCalendar(holidays={})
        store.write_ohlcv("COMI", bars(("2026-03-05", 10.0)), source="yahoo")
        gaps = store.ohlcv_gaps("COMI", date(2026, 3, 5), date(2026, 3, 7), calendar=cal)
        assert gaps == []

    def test_holidays_are_not_gaps(self, store: DataStore) -> None:
        cal = EGXCalendar(holidays={2026: frozenset({date(2026, 3, 3)})})
        store.write_ohlcv("COMI", bars(("2026-03-01", 10.0), ("2026-03-02", 11.0)), source="yahoo")
        gaps = store.ohlcv_gaps("COMI", date(2026, 3, 1), date(2026, 3, 3), calendar=cal)
        assert gaps == []

    def test_no_data_means_every_trading_day_is_a_gap(self, store: DataStore) -> None:
        cal = EGXCalendar(holidays={})
        gaps = store.ohlcv_gaps("COMI", date(2026, 3, 1), date(2026, 3, 5), calendar=cal)
        assert len(gaps) == 5


class TestQuotes:
    def test_quote_is_appended_not_replaced(self, store: DataStore) -> None:
        for price in (86.0, 86.2, 86.5):
            store.write_quote({"symbol": "COMI", "price": price, "volume": 1000}, source="yahoo")
        assert store.stats().quote_rows == 3
        assert store.latest_quote("COMI")["price"] == 86.5

    def test_quote_without_symbol_is_rejected(self, store: DataStore) -> None:
        assert store.write_quote({"price": 10.0}, source="yahoo") == 0

    def test_prune_keeps_recent_rows(self, store: DataStore) -> None:
        old = datetime.now(timezone.utc) - timedelta(days=120)
        store.write_quote({"symbol": "COMI", "price": 1.0}, source="yahoo", fetched_at=old)
        store.write_quote({"symbol": "COMI", "price": 2.0}, source="yahoo")
        assert store.prune_quotes(keep_days=90) == 1
        assert store.stats().quote_rows == 1


class TestIndexLevels:
    def test_index_row_upserts_per_date(self, store: DataStore) -> None:
        store.write_index("EGX30", "2026-03-01", 32456.78, change_pct=0.85, source="egx_official")
        store.write_index("EGX30", "2026-03-01", 32460.0, change_pct=0.86, source="egx_official")
        rows = store.get_index_history("EGX30")
        assert len(rows) == 1
        assert rows[0]["close"] == 32460.0
        assert rows[0]["change_pct"] == 0.86

    def test_index_name_is_stored_uppercase(self, store: DataStore) -> None:
        store.write_index("egx30", "2026-03-01", 1.0, source="x")
        assert store.get_index_history("egx30")[0]["index_name"] == "EGX30"

    def test_history_is_range_filtered(self, store: DataStore) -> None:
        for day in ("2026-03-01", "2026-03-02", "2026-03-03"):
            store.write_index("EGX30", day, 100.0, source="x")
        assert len(store.get_index_history("EGX30", start="2026-03-02")) == 2


class TestAudit:
    def test_attempts_are_recorded(self, store: DataStore) -> None:
        store.record_attempt(
            source="egx_official", capability="market_snapshot", subject=None,
            ok=False, error="HTTP 503", error_kind="SourceUnavailable",
        )
        store.record_attempt(
            source="yahoo", capability="quote", subject="COMI", ok=True, latency_ms=210.0, rows=1
        )
        rows = store.recent_audit(limit=10)
        assert len(rows) == 2
        assert rows[0]["source"] == "yahoo"
        assert store.recent_audit(failures_only=True)[0]["source"] == "egx_official"

    def test_source_reliability_is_derived_from_history(self, store: DataStore) -> None:
        for ok in (True, True, False, True):
            store.record_attempt(source="yahoo", capability="quote", subject="COMI", ok=ok, latency_ms=100)
        for ok in (False, False):
            store.record_attempt(source="egx_official", capability="quote", subject="COMI", ok=ok, latency_ms=900)

        report = {r["source"]: r for r in store.source_reliability()}
        assert report["yahoo"]["success_rate"] == 0.75
        assert report["egx_official"]["success_rate"] == 0.0
        assert report["egx_official"]["avg_latency_ms"] == 900.0

    def test_prune_audit(self, store: DataStore) -> None:
        store.record_attempt(source="a", capability="quote", subject="X", ok=True)
        assert store.prune_audit(keep_days=180) == 0


class TestQualityEvents:
    def test_flags_are_recorded_with_context(self, store: DataStore) -> None:
        sourced = Sourced(
            value={"symbol": "COMI"},
            provenance=provenance_for("yahoo", endpoint="https://example/x"),
        )
        sourced.add_flag(QualityFlag.STALE, "quote age 900s exceeds 300s")
        sourced.add_flag(QualityFlag.VALID, "structural checks passed")

        written = store.record_flags(sourced, symbol="COMI")
        assert written == 1  # VALID is noise, not an event
        assert store.stats().quality_events == 1

    def test_explicit_event(self, store: DataStore) -> None:
        store.record_quality_event(QualityFlag.CONFLICT, note="EGX vs Yahoo", symbol="COMI", payload={"a": 1})
        assert store.stats().quality_events == 1


class TestRuns:
    def test_run_lifecycle(self, store: DataStore) -> None:
        handle = store.begin_run("ohlcv_refresh", params={"symbols": 292})
        store.finish_run(
            handle, status="ok", rows_written=1234, sources_used=["yahoo", "egx_official"],
            warnings=["holiday list unverified for 2026"],
        )
        # The run row is durable state, not a log line.
        assert store.stats().audit_rows == 0


class TestStatsAndMaintenance:
    def test_stats_reflect_writes(self, store: DataStore) -> None:
        store.upsert_symbols([{"symbol": "COMI"}])
        store.write_ohlcv("COMI", bars(("2026-03-01", 10.0)), source="yahoo")
        store.write_quote({"symbol": "COMI", "price": 10.0}, source="yahoo")
        store.record_attempt(source="yahoo", capability="quote", subject="COMI", ok=True)

        stats = store.stats()
        assert stats.symbols == 1
        assert stats.ohlcv_rows == 1
        assert stats.quote_rows == 1
        assert stats.audit_rows == 1
        assert stats.last_ohlcv_date == "2026-03-01"
        assert stats.as_dict()["schema_version"] == SCHEMA_VERSION

    def test_vacuum_runs(self, store: DataStore) -> None:
        store.vacuum()

    def test_context_manager_closes(self, tmp_path) -> None:
        with DataStore(tmp_path / "ctx.db") as store:
            store.upsert_symbols([{"symbol": "COMI"}])
        with pytest.raises(Exception):
            store.stats()
