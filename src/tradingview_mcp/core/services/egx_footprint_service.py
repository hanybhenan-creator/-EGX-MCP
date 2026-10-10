"""MCP-facing institutional footprint service.

Turns the stored daily history of one EGX symbol into a described state —
which volume/price divergences are present, whether a consolidation zone is
being held, whether dips are being absorbed, whether the base looks ready —
together with a **measured** data-quality score for that symbol.

Two fabrications in the design document are replaced with real measurements:

* ``data_quality_score`` was the constant ``85.0`` and ``source_reliability``
  was the constant ``"HIGH"``. They are now computed from the data layer: bar
  coverage against the EGX calendar, structural integrity of the OHLCV rows,
  freshness relative to the newest settled session, the symbol's recorded
  quality events, and the per-source success rate the store has actually
  observed on this machine. Every component is returned so the score can be
  argued with.
* ``breakout_probability`` and the composite scores are weighted flag sums, not
  probabilities. The measured probability lives in the calibration engine
  (``egx_signal_calibration`` / ``scripts/egx_calibrate.py``), which reports
  what actually followed these footprints historically.

Nothing here touches the network.
"""

from __future__ import annotations

from typing import Any, Optional

from tradingview_mcp.core.analysis.footprints import analyze_footprint
from tradingview_mcp.core.data.quality import QualityFlag, validate_ohlcv_bars
from tradingview_mcp.core.errors import ErrorCode, make_error

MIN_BARS = 25
QUALITY_WEIGHTS = {
    "coverage": 0.30,
    "integrity": 0.30,
    "freshness": 0.25,
    "source_success": 0.15,
}


def _finite(value: Any) -> bool:
    try:
        return value is not None and float(value) == float(value)
    except (TypeError, ValueError):
        return False


def assess_data_quality(bars: list[dict[str, Any]], *, symbol: str) -> dict[str, Any]:
    """Measure how much this symbol's stored history can be trusted (0-100).

    Components (weights in ``QUALITY_WEIGHTS``):

      coverage         stored sessions / sessions the EGX calendar expects
                       between the first and last stored bar
      integrity        share of bars passing the structural OHLCV checks
                       (high >= low, high >= max(open, close), low <= min(...))
      freshness        how close the newest bar is to the newest *settled*
                       session; ten or more sessions behind scores zero
      source_success   mean measured success rate of every source that has
                       fetched on this machine (from the audit trail)

    Every component is returned next to the score, and a component that cannot
    be measured is excluded from the average and named in ``notes`` — the point
    of this exercise is that a number must be attributable.
    """
    from datetime import date as _date

    from tradingview_mcp.core.data.factory import get_layer
    from tradingview_mcp.core.data.quality import _expected_bar

    layer = get_layer()
    calendar, store = layer.calendar, layer.store

    if not bars:
        return {"score": 0.0, "grade": "F", "components": {}, "notes": ["no stored bars for this symbol"]}

    dates = sorted(str(b.get("date")) for b in bars if b.get("date"))
    first, last = (dates[0], dates[-1]) if dates else (None, None)

    # 1. Coverage against the trading calendar, not against the row count.
    expected_sessions = None
    coverage = None
    if first and last:
        try:
            expected = calendar.trading_days_between(_date.fromisoformat(first[:10]), _date.fromisoformat(last[:10]))
            expected_sessions = len(expected)
            if expected_sessions:
                coverage = min(1.0, len(bars) / expected_sessions)
        except (ValueError, TypeError):
            coverage = None

    # 2. Integrity of the rows themselves.
    verdicts = validate_ohlcv_bars(bars)
    invalid = sum(int((v.detail or {}).get("count", 0)) for v in verdicts if v.flag is not QualityFlag.VALID)
    integrity = max(0.0, 1.0 - invalid / max(1, len(bars)))

    # 3. Freshness against the newest settled session.
    settled = _expected_bar(calendar, None)
    freshness = None
    if last:
        try:
            behind = len(calendar.trading_days_between(_date.fromisoformat(last[:10]), settled))
            freshness = max(0.0, 1.0 - behind / 10.0)
        except (ValueError, TypeError):
            freshness = None

    # 4. Measured source success rate on this machine.
    reliability = store.source_reliability()
    rates = [r["success_rate"] for r in reliability if r.get("success_rate") is not None]
    source_success = sum(rates) / len(rates) if rates else None

    components = {
        "coverage": round(coverage, 4) if coverage is not None else None,
        "integrity": round(integrity, 4),
        "freshness": round(freshness, 4) if freshness is not None else None,
        "source_success": round(source_success, 4) if source_success is not None else None,
        "bars": len(bars),
        "expected_sessions": expected_sessions,
        "newest_bar": last,
        "newest_settled_session": settled.isoformat(),
        "invalid_bars": invalid,
        "recorded_quality_flags": sum(int((v.detail or {}).get("count", 0)) for v in verdicts if v.flag is not QualityFlag.VALID),
        "sources_measured": [
            {"source": r["source"], "attempts": r["attempts"], "success_rate": r["success_rate"]}
            for r in reliability
        ],
    }

    total = weight = 0.0
    for name, value in (
        ("coverage", coverage),
        ("integrity", integrity),
        ("freshness", freshness),
        ("source_success", source_success),
    ):
        if value is None:
            continue
        total += value * QUALITY_WEIGHTS[name]
        weight += QUALITY_WEIGHTS[name]
    score = round(100.0 * total / weight, 1) if weight else None

    notes: list[str] = []
    if coverage is not None and coverage < 0.9:
        notes.append(f"only {len(bars)} of ~{expected_sessions} expected sessions are stored")
    if invalid:
        notes.append(f"{invalid} bar(s) failed structural validation")
    if freshness is not None and freshness < 1.0:
        notes.append(f"newest bar is {last}; the newest settled session is {settled.isoformat()}")
    if source_success is None:
        notes.append("no fetch attempts recorded yet: source reliability is unmeasured, not good")
    grade = (
        "A" if score is not None and score >= 90
        else "B" if score is not None and score >= 75
        else "C" if score is not None and score >= 60
        else "D" if score is not None and score >= 40
        else "F"
    )
    return {"score": score, "grade": grade, "components": components, "notes": notes}


def _reliability_label(quality: dict[str, Any]) -> str:
    """HIGH/MEDIUM/LOW from the *measured* source success rate, not an opinion."""
    rate = quality["components"].get("source_success")
    if rate is None:
        return "UNMEASURED"
    if rate >= 0.9:
        return "HIGH"
    if rate >= 0.7:
        return "MEDIUM"
    return "LOW"


def egx_accumulation_fingerprint(
    symbol: str,
    *,
    lookback_days: int = 20,
    zone_lookback: int = 30,
    exchange: str = "EGX",
) -> dict[str, Any]:
    """Institutional footprint read for one symbol, from the local store only.

    Reports what the bars objectively contain — volume anomalies relative to a
    rolling average, accumulation zones with measured support/resistance
    volume, absorbed dips, breakout preparation, shakeout reclaims and the
    distribution mirror — and states plainly that the composite scores are
    weighted flags whose predictive value is measured separately by
    ``egx_signal_calibration``.

    Args:
      symbol: EGX ticker (``EGX:`` prefix and lowercase are accepted).
      lookback_days: Sessions used for the volume-average baseline (5-120).
      zone_lookback: Sessions scanned for consolidation zones (10-250).
      exchange: Store partition to read (default EGX).
    """
    ticker = (symbol or "").strip().upper().replace("EGX:", "")
    if not ticker:
        return make_error(ErrorCode.INVALID_PARAMETER, "symbol is required", symbol=symbol)
    if not 5 <= lookback_days <= 120:
        return make_error(ErrorCode.INVALID_PARAMETER, "lookback_days must be 5-120", lookback_days=lookback_days)
    if not 10 <= zone_lookback <= 250:
        return make_error(ErrorCode.INVALID_PARAMETER, "zone_lookback must be 10-250", zone_lookback=zone_lookback)

    from tradingview_mcp.core.data.factory import get_layer

    layer = get_layer()
    bars = layer.store.get_ohlcv(ticker, exchange=exchange)
    if len(bars) < MIN_BARS:
        return make_error(
            ErrorCode.NO_DATA,
            f"{ticker}: {len(bars)} stored sessions, {MIN_BARS} needed. Run "
            "`python scripts/egx_data_sync.py daily --symbols "
            f"{ticker} --days 365` first",
            symbol=ticker,
            stored_bars=len(bars),
        )

    quality = assess_data_quality(bars, symbol=ticker)
    try:
        fingerprint = analyze_footprint(
            bars,
            ticker,
            lookback_days=lookback_days,
            zone_lookback=min(zone_lookback, len(bars)),
            data_quality_score=quality["score"],
            source_reliability=_reliability_label(quality),
        )
    except ValueError as exc:
        return make_error(ErrorCode.NO_DATA, str(exc), symbol=ticker)

    payload = fingerprint.as_dict()
    payload["data_quality"] = quality
    payload["bars_used"] = len(bars)
    payload["window"] = {"from": str(bars[0].get("date")), "to": str(bars[-1].get("date"))}
    payload["unmeasured"] = [
        "buy_volume/sell_volume are derived from the candle direction: EGX publishes no "
        "per-side volume, so they are an estimate, not observed order flow",
        "footprint_score/preparation_score/accumulation_strength are weighted flag sums; "
        "their predictive value is what egx_signal_calibration measures",
    ]
    return payload
