"""egx_classify_stocks: classes, policy, glossary and error paths, on known synthetic stocks."""

from __future__ import annotations

import random

import pytest

from tradingview_mcp.core.analysis import stock_classes as sc
from tradingview_mcp.core.services import egx_stock_classes_service as svc


def make(closes, volumes=None):
    bars = []
    for k, c in enumerate(closes):
        o = closes[k - 1] if k else c
        v = volumes[k] if volumes is not None else 1_000_000.0
        bars.append({"date": f"d{k:04d}", "open": o, "high": max(o, c) * 1.005,
                     "low": min(o, c) * 0.995, "close": c, "volume": v})
    return bars


def trending(n=150, seed=1):
    rng = random.Random(seed)
    price, closes = 10.0, []
    for _ in range(n):
        price *= 1.01 + rng.uniform(-0.002, 0.002)
        closes.append(price)
    return make(closes)


def ranging(n=150):
    return make([10.0 if k % 2 == 0 else 10.5 for k in range(n)])


def dormant_by_zero_volume(n=150):
    volumes = [0 if k % 3 == 0 else 1_000_000 for k in range(n)]
    return make([10.0 + 0.001 * k for k in range(n)], volumes=volumes)


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> dict:
    stocks = {
        "TREND": trending(n=150),
        "RANGE": ranging(n=150),
        "THIN": dormant_by_zero_volume(n=150),
    }
    monkeypatch.setattr(svc, "_resolve_symbols", lambda symbols="", index="", limit=0: list(stocks))
    monkeypatch.setattr(svc, "_load_bars", lambda symbols, *, lookback_days, max_symbols: (dict(stocks), []))
    return stocks


def test_each_stock_gets_its_known_class(fake_store) -> None:
    out = svc.egx_classify_stocks()
    by_symbol = {s["symbol"]: s for s in out["stocks"]}
    assert by_symbol["TREND"]["official_class"] == sc.TRENDING
    assert by_symbol["RANGE"]["official_class"] == sc.RANGE_BOUND
    assert by_symbol["THIN"]["official_class"] == sc.DORMANT


def test_counts_and_groups_agree(fake_store) -> None:
    out = svc.egx_classify_stocks()
    assert sum(out["counts_by_class"].values()) == 3
    for cls, symbols in out["symbols_by_class"].items():
        assert out["counts_by_class"][cls] == len(symbols)


def test_no_tool_is_allowed_for_any_class_yet(fake_store) -> None:
    out = svc.egx_classify_stocks()
    for stock in out["stocks"]:
        assert stock["allowed_tools"] == []
    assert "No tool is allowed" in out["disclaimer"]


def test_dormant_stock_lists_what_it_excludes_and_why(fake_store) -> None:
    thin = next(s for s in svc.egx_classify_stocks()["stocks"] if s["symbol"] == "THIN")
    assert "c4" in thin["excluded_tools"]
    assert "volume-based" in thin["excluded_tools"]["c4"]


def test_response_carries_glossary_and_plain_policy(fake_store) -> None:
    out = svc.egx_classify_stocks()
    assert "glossary" in out and "official_class" in out["glossary"]
    assert set(out["class_policy"]) == set(sc.CLASSES)


def test_unknown_index_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def bad(symbols="", index="", limit=0):
        raise ValueError("unknown index EGX999")

    monkeypatch.setattr(svc, "_resolve_symbols", bad)
    out = svc.egx_classify_stocks(index="EGX999")
    assert "error" in out


def test_empty_store_is_no_data_not_an_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(svc, "_resolve_symbols", lambda symbols="", index="", limit=0: ["NOPE"])
    monkeypatch.setattr(svc, "_load_bars", lambda symbols, *, lookback_days, max_symbols: ({}, []))
    out = svc.egx_classify_stocks()
    assert "error" in out and "stocks" not in out


def test_all_unclassified_adds_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    short = {"SHORT": trending(n=30)}
    monkeypatch.setattr(svc, "_resolve_symbols", lambda symbols="", index="", limit=0: list(short))
    monkeypatch.setattr(svc, "_load_bars", lambda symbols, *, lookback_days, max_symbols: (dict(short), []))
    out = svc.egx_classify_stocks()
    assert out["stocks"][0]["official_class"] == sc.UNCLASSIFIED
    assert any("enough complete history" in w for w in out["warnings"])
