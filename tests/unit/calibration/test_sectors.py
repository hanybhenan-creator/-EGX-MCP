"""Sector calibration: planted and noise controls, classification rules, report shape.

The controls are synthetic. They show the method can detect a planted sector drift
of the size it is designed to detect, and that it does not manufacture a
VALIDATED verdict from noise on the fixed seeds below. They say nothing about EGX.
"""

from __future__ import annotations

import json

import pytest

from tests.unit.calibration.sector_fixtures import sector_universe
from tradingview_mcp.core.calibration import sectors as sec
from tradingview_mcp.core.calibration.sectors import analyze_sectors, build_panel


@pytest.fixture(scope="module")
def planted_report():
    bars, sector_of = sector_universe(planted=True, seed=7)
    return analyze_sectors(bars, sector_of=sector_of.get, horizon=5, permutations=200, walk_forward_folds=3)


@pytest.fixture(scope="module")
def noise_reports():
    out = []
    for seed in (1, 2, 3):
        bars, sector_of = sector_universe(planted=False, seed=seed)
        out.append(analyze_sectors(bars, sector_of=sector_of.get, horizon=5, permutations=200, walk_forward_folds=3))
    return out


# ── controls ────────────────────────────────────────────────────────────────


def test_planted_drift_is_detected_by_momentum_rule(planted_report):
    verdicts = {m.signal: m.verdict for m in planted_report.measurements}
    assert verdicts["momentum_20d_top_third"] == "VALIDATED"
    assert "momentum_20d_top_third" in planted_report.as_dict()["validated_rules"]


def test_planted_drift_gives_predictive_rank_quality(planted_report):
    assert planted_report.ic["momentum_20d"]["verdict"] == "PREDICTIVE"
    assert planted_report.ic["momentum_20d"]["mean_ic"] > 0


def test_planted_reversal_rule_is_not_validated(planted_report):
    # Planted drift is trend-like; a 5-day reversal bet should lose or not reach significance.
    verdicts = {m.signal: m.verdict for m in planted_report.measurements}
    assert verdicts["reversal_5d_top_third"] != "VALIDATED"


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_noise_does_not_validate_momentum_or_rank_quality(noise_reports, idx):
    report = noise_reports[idx]
    verdicts = {m.signal: m.verdict for m in report.measurements}
    assert verdicts["momentum_20d_top_third"] != "VALIDATED"
    assert verdicts["momentum_20d_bottom_third"] != "VALIDATED"
    assert report.ic["momentum_20d"]["verdict"] != "PREDICTIVE"


def test_noise_has_no_validated_rule_across_fixed_seeds(noise_reports):
    # Multiple-testing guard: with 8 rules × 3 seeds, a single false VALIDATED is
    # possible (≈0.6% per rule verdict in a 20-seed run). Allow at most one.
    total = sum(1 for r in noise_reports for m in r.measurements if m.verdict == "VALIDATED")
    assert total <= 1


def test_planted_has_more_validated_rules_than_noise(planted_report, noise_reports):
    planted = sum(1 for m in planted_report.measurements if m.verdict == "VALIDATED")
    noise = max(sum(1 for m in r.measurements if m.verdict == "VALIDATED") for r in noise_reports)
    assert planted > noise


# ── classification and membership ───────────────────────────────────────────


def test_unclassified_symbols_are_excluded_from_sectors():
    bars, sector_of = sector_universe(planted=False, sectors=5, members=4, sessions=120, seed=3)
    # Move one whole group to the catch-all bucket.
    for sym, sec_name in list(sector_of.items()):
        if sec_name == "sector_0":
            sector_of[sym] = "other"
    panel = build_panel(bars, sector_of=sector_of.get, horizon=5, min_members=3)
    assert "other" not in panel.sectors
    assert "other" not in panel.members
    assert "sector_0" not in panel.sectors


def test_min_members_is_enforced():
    bars, sector_of = sector_universe(planted=False, sectors=5, members=4, sessions=120, seed=4)
    # Keep only two members of sector_1.
    keep = {"S1_0", "S1_1"}
    for sym in list(bars):
        if sym.startswith("S1_") and sym not in keep:
            del bars[sym]
    strict = build_panel(bars, sector_of=sector_of.get, horizon=5, min_members=3)
    loose = build_panel(bars, sector_of=sector_of.get, horizon=5, min_members=2)
    assert "sector_1" not in strict.sectors
    assert "sector_1" in loose.sectors


def test_analyze_rejects_bad_parameters():
    bars, sector_of = sector_universe(planted=False, sectors=5, members=4, sessions=120, seed=5)
    with pytest.raises(ValueError):
        analyze_sectors(bars, sector_of=sector_of.get, horizon=0, permutations=0, walk_forward_folds=0)
    with pytest.raises(ValueError):
        analyze_sectors(bars, sector_of=sector_of.get, horizon=5, min_members=1, permutations=0, walk_forward_folds=0)
    with pytest.raises(ValueError):
        analyze_sectors(bars, sector_of=sector_of.get, horizon=5, round_trip_cost=0.5, permutations=0, walk_forward_folds=0)


def test_rule_registry_has_eight_rules():
    assert len(sec.RULES) == 8
    assert {"legacy_hot", "legacy_cold"} <= set(sec.RULES)


# ── report shape ────────────────────────────────────────────────────────────


def test_report_dict_has_expected_keys_and_is_json_safe(planted_report):
    payload = planted_report.as_dict()
    for key in (
        "horizon",
        "window",
        "sessions",
        "min_members",
        "round_trip_cost",
        "permutations",
        "method",
        "disclaimer",
        "coverage",
        "measurements",
        "walk_forwards",
        "rank_quality",
        "current",
        "validated_rules",
        "warnings",
    ):
        assert key in payload, key
    json.dumps(payload)  # must not raise


def test_measurement_rows_carry_titles_and_verdicts(planted_report):
    rows = planted_report.as_dict()["measurements"]
    assert len(rows) == 8
    for row in rows:
        assert row["kind"] == "sector_rule"
        assert row["title"]
        assert row["verdict"] in {"VALIDATED", "SUGGESTIVE", "REJECTED", "NO_EVENTS"}


def test_current_rows_use_renamed_fields(planted_report):
    rows = planted_report.as_dict()["current"]
    assert rows, "expected at least one current sector row"
    for row in rows:
        assert "rule_for_label" in row
        assert "rule_verdict_over_history" in row
        assert "calibrated_claim" not in row
        assert "claim_verdict" not in row


def test_method_mentions_approximate_pvalues():
    assert "approximate" in sec.METHOD
    assert "zero-sum" in sec.METHOD


# ── MCP service ─────────────────────────────────────────────────────────────


def test_service_rejects_bad_parameters():
    from tradingview_mcp.core.services.egx_sector_calibration_service import egx_sector_event_study

    out = egx_sector_event_study(horizon=0)
    assert "error" in out
    assert "INVALID_PARAMETER" in json.dumps(out)
    out = egx_sector_event_study(walk_forward_folds=1)
    assert "error" in out


def test_service_end_to_end_with_injected_bars(monkeypatch):
    from tradingview_mcp.core.services import egx_sector_calibration_service as svc

    bars, sector_of = sector_universe(planted=True, sectors=6, members=4, sessions=420, seed=7)
    monkeypatch.setattr(svc, "_load_bars", lambda symbols, *, lookback_days, max_symbols: (bars, []))
    monkeypatch.setattr(svc, "get_sector", sector_of.get)
    out = svc.egx_sector_event_study(permutations=100, walk_forward_folds=0)
    assert "error" not in out
    assert out["mode"] == "offline store"
    assert "momentum_20d_top_third" in out["validated_rules"]
    assert out["symbols"]["with_history"] == len(bars)


def test_service_reports_no_data(monkeypatch):
    from tradingview_mcp.core.services import egx_sector_calibration_service as svc

    monkeypatch.setattr(svc, "_load_bars", lambda symbols, *, lookback_days, max_symbols: ({}, []))
    out = svc.egx_sector_event_study(permutations=0, walk_forward_folds=0)
    assert "error" in out
    assert "egx_data_sync.py" in json.dumps(out)


def test_zero_permutations_computes_no_pvalues_and_never_validates():
    # Regression: an earlier version forced one null draw, so p was fabricated (0.5).
    bars, sector_of = sector_universe(planted=True, seed=7)
    report = analyze_sectors(bars, sector_of=sector_of.get, horizon=5, permutations=0, walk_forward_folds=0)
    for m in report.measurements:
        if m.events:
            assert m.p_mean is None and m.p_hit is None, m.signal
            assert m.q_hit is None and m.q_mean is None, m.signal
            assert m.verdict != "VALIDATED", m.signal
    assert report.as_dict()["validated_rules"] == []
    assert any("permutation test off" in w for w in report.warnings)
