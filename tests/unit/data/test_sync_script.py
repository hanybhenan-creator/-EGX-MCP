"""The sync CLI must be runnable and honest even with no network at all."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import egx_data_sync as sync  # noqa: E402


def run(capsys, argv: list[str]) -> tuple[int, dict]:
    code = sync.main(argv)
    out = capsys.readouterr().out
    return code, json.loads(out)


def test_status_runs_offline(capsys, tmp_path) -> None:
    code, payload = run(capsys, ["--db", str(tmp_path / "s.db"), "status"])
    assert code == 0
    assert payload["market_clock"]["phase"]
    assert payload["store"]["schema_version"] >= 1
    assert {row["source"] for row in payload["sources"]} >= {"egx_official", "yahoo"}


def test_status_surfaces_unverified_holidays(capsys, tmp_path) -> None:
    _, payload = run(capsys, ["--db", str(tmp_path / "s.db"), "status"])
    assert payload["calendar_warnings"]


def test_holidays_defaults_to_the_current_year(capsys, tmp_path) -> None:
    code, payload = run(capsys, ["--db", str(tmp_path / "s.db"), "holidays"])
    assert code == 0
    assert payload["year"] >= 2026
    assert payload["verified"] is False
    assert payload["closures_loaded"] == []


def test_holidays_reports_a_specific_year(capsys, tmp_path) -> None:
    _, payload = run(capsys, ["holidays", "--year", "2026"])
    assert payload["year"] == 2026


def test_unknown_index_fails_loudly(capsys, tmp_path) -> None:
    with pytest.raises((ValueError, SystemExit)):
        sync.main(["--db", str(tmp_path / "s.db"), "daily", "--index", "NOPE"])


def test_offline_refresh_reports_failure_per_symbol(capsys, tmp_path, monkeypatch) -> None:
    # No network in this sandbox: every source fails, and the report must say so
    # per symbol instead of pretending the refresh succeeded.
    monkeypatch.setenv("EGX_TRANSPORT_TIMEOUT", "1")
    monkeypatch.setenv("EGX_ENABLE_TRADINGVIEW", "0")
    code, payload = run(
        capsys,
        ["--db", str(tmp_path / "s.db"), "daily", "--symbols", "COMI", "--days", "30"],
    )
    assert payload["requested"] == 1
    assert payload["status"] in {"partial", "failed"}
    assert payload["problems"][0]["symbol"] == "COMI"
    assert payload["problems"][0]["error"]
    assert code in (0, 1)
