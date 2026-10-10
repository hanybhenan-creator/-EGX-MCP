"""Shared fixtures for the EGX data-layer tests.

Everything here is hermetic: no test in this package touches the network. The
fixtures give the tests a recorded EGX page, a recorded Yahoo payload, and a
``FakeTransport`` to replay them through, so a parser regression fails here
instead of silently producing wrong prices in production.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tradingview_mcp.core.data.calendar import EGXCalendar
from tradingview_mcp.core.data.provenance import Sourced, provenance_for
from tradingview_mcp.core.data.transport import FakeTransport, TransportResponse

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "egx"


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture
def egx_market_watch_html() -> str:
    return (FIXTURE_DIR / "market_watch.html").read_text(encoding="utf-8")


@pytest.fixture
def egx_market_watch_arabic_html() -> str:
    return (FIXTURE_DIR / "market_watch_arabic.html").read_text(encoding="utf-8")


@pytest.fixture
def yahoo_chart_payload() -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / "yahoo_chart_daily.json").read_text(encoding="utf-8"))


@pytest.fixture
def yahoo_error_payload() -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / "yahoo_error_delisted.json").read_text(encoding="utf-8"))


@pytest.fixture
def fake_transport() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def calendar() -> EGXCalendar:
    """Calendar with no holidays, so weekday maths is deterministic."""
    return EGXCalendar(holidays={})


@pytest.fixture
def sourced_factory():
    """Build a :class:`Sourced` around a value with a controllable fetch time."""

    def build(
        value: Any,
        *,
        source: str = "test_source",
        age_s: float = 0.0,
        endpoint: str | None = None,
        locator: dict[str, Any] | None = None,
    ) -> Sourced[Any]:
        fetched = datetime.now(timezone.utc) - timedelta(seconds=age_s)
        return Sourced(
            value=value,
            provenance=provenance_for(
                source, fetched_at=fetched, endpoint=endpoint, locator=locator
            ),
        )

    return build


def make_response(
    text: str, *, status: int = 200, url: str = "", headers: dict[str, str] | None = None
) -> TransportResponse:
    return TransportResponse(url=url, status=status, text=text, headers=headers or {})
