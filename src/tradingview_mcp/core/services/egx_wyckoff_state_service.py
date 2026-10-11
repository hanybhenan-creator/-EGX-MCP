"""Per-symbol Wyckoff / VSA state: which events fired recently, and what they measured.

The events themselves are defined in ``core.analysis.wyckoff``. Their historical
verdicts come from the same calibration run that measures every other signal
(``egx_event_study``), so no statistic is computed twice.

The reply is a *measured state*, never a trade instruction. A VALIDATED event
means it survived the calibration on this store's sample; it is not a forecast,
and walk-forward folds did not reliably separate real from noise on the synthetic
controls (see EGX_CALIBRATION.md).
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from tradingview_mcp.core.analysis import wyckoff
from tradingview_mcp.core.calibration.glossary import WYCKOFF_GLOSSARY
from tradingview_mcp.core.calibration.signals import WYCKOFF_CONDITIONS
from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services.egx_calibration_service import (
    DEFAULT_MAX_SYMBOLS,
    _load_bars,
    egx_event_study,
)

_SYMBOL = re.compile(r"^[A-Z0-9.\-]{1,12}$")
STALE_AFTER_DAYS = 10


def _event_flags(bars: list[dict], event: str) -> list:
    return wyckoff.FLAG_BUILDERS[event](bars)


def egx_symbol_events(
    *,
    symbol: str,
    lookback_sessions: int = 5,
    horizon: int = 5,
    index: str = "",
    lookback_days: int = 1500,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    permutations: int = 200,
    walk_forward_folds: int = 3,
) -> dict[str, Any]:
    clean = (symbol or "").upper().replace("EGX:", "").strip()
    if not _SYMBOL.match(clean):
        return make_error(ErrorCode.INVALID_PARAMETER, "symbol must be an EGX ticker such as COMI", symbol=symbol)
    if not 1 <= lookback_sessions <= 20:
        return make_error(ErrorCode.INVALID_PARAMETER, "lookback_sessions must be 1-20", lookback_sessions=lookback_sessions)

    # Calibration over the stored universe: one run, every Wyckoff event.
    calibration = egx_event_study(
        signal=",".join(WYCKOFF_CONDITIONS),
        horizon=horizon,
        index=index,
        lookback_days=lookback_days,
        max_symbols=max_symbols,
        permutations=permutations,
        walk_forward_folds=walk_forward_folds,
    )
    if "error" in calibration:
        return calibration

    bars_by_symbol, skipped = _load_bars([clean], lookback_days=lookback_days, max_symbols=1)
    if clean not in bars_by_symbol:
        return make_error(
            ErrorCode.NO_DATA,
            f"{clean} has fewer than 60 stored sessions. Run `python scripts/egx_data_sync.py daily --index EGX30 --days 365` first",
            symbol=clean,
        )
    bars = bars_by_symbol[clean]
    n = len(bars)
    window = range(max(0, n - lookback_sessions), n)

    by_signal = {m["signal"]: m for m in calibration["measurements"]}
    walk = calibration.get("walk_forward", {})

    recent: list[dict[str, Any]] = []
    for event in wyckoff.EVENTS:
        flags = _event_flags(bars, event)
        for i in window:
            if flags[i]:
                name = f"wy_{event}"
                m = by_signal.get(name, {})
                wf = walk.get(name)
                recent.append(
                    {
                        "date": bars[i]["date"],
                        "event": event,
                        "definition": wyckoff.DEFINITIONS[event],
                        "measured_verdict": m.get("verdict"),
                        "events_in_sample": m.get("events"),
                        "hit_rate": m.get("hit_rate"),
                        "base_rate": m.get("base_rate"),
                        "excess_hit": m.get("excess_hit"),
                        "q_value": m.get("q_value_hit"),
                        "walk_forward": wf.get("verdict") if wf else None,
                    }
                )
    recent.sort(key=lambda r: r["date"])

    last_date = bars[-1]["date"]
    warnings = list(calibration.get("warnings", []))
    try:
        age = (date.today() - date.fromisoformat(str(last_date)[:10])).days
        if age > STALE_AFTER_DAYS:
            warnings.append(f"last stored bar is {age} days old ({last_date}); run the daily sync before reading the state")
    except ValueError:
        warnings.append(f"could not parse last bar date {last_date!r}")

    validated_recent = sorted({r["event"] for r in recent if r["measured_verdict"] == "VALIDATED"})
    return {
        "symbol": clean,
        "as_of": last_date,
        "lookback_sessions": lookback_sessions,
        "horizon": horizon,
        "mode": calibration.get("mode"),
        "recent_events": recent,
        "state": "NO_RECENT_EVENT" if not recent else "EVENT_FIRED",
        "validated_events_recent": validated_recent,
        "calibration_universe": calibration.get("measurement", {}).get("symbols_analysed"),
        "calibration_sessions": calibration.get("measurement", {}).get("sessions"),
        "definitions": wyckoff.DEFINITIONS,
        "glossary": dict(WYCKOFF_GLOSSARY),
        "warnings": warnings,
        "disclaimer": (
            "A measured state, not a trade instruction. Verdicts describe the stored sample's "
            "historical outcomes for each event across the calibration universe, not this symbol's "
            "future. Walk-forward folds did not reliably reject noise on synthetic controls, so "
            "treat VALIDATED as one piece of evidence, not a conclusion."
        ),
    }
