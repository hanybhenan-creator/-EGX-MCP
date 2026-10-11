"""MCP-facing calibration service: what the smart-money conditions actually
predict, measured on the local EGX history.

This is the answer to "the numbers look precise but where do they come from?".
Nothing here touches the network: calibration reads the ``DataStore`` that the
ingestion layer fills. A signal's reputation is therefore reproducible offline
— same database, same seed, same table — and it never claims more than the
stored history can support.

    egx_event_study(signal="accumulation_corrected", horizon=5)

returns, for that signal: events, sample window, hit rate with a 95% interval,
the sample's base rate, the excess, a block-permutation p-value, a q-value
corrected for every signal tested in the batch, and the walk-forward verdict.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

from tradingview_mcp.core.calibration import (
    Sample,
    analyze,
)
from tradingview_mcp.core.calibration.glossary import EVENT_GLOSSARY, attach_glossary
from tradingview_mcp.core.data.factory import offline_mode
from tradingview_mcp.core.errors import ErrorCode, make_error
from tradingview_mcp.core.services.egx_data_service import _resolve_symbols

DEFAULT_MAX_SYMBOLS = 120
MAX_PERMUTATIONS = 2000

# The conditions worth measuring by default: the five legacy/corrected
# conditions, the composites, and the distribution mirror. Measured together so
# that the Benjamini-Hochberg correction sees the true size of the batch.
DEFAULT_SIGNALS = [
    "accumulation_legacy",
    "accumulation_corrected",
    "accumulation_high_conviction",
    "distribution_corrected",
    "hidden_accumulation",
    "volume_footprint",
    "absorption",
    "breakout_preparation",
    "distribution_footprint",
    "shakeout_reclaim",
    "c1", "c2", "c3", "c4", "c5",
]


def _bar_window(lookback_days: int) -> str:
    """Calendar start date for the requested history (EGX: ~21 sessions/month)."""
    return (date.today() - timedelta(days=max(30, int(lookback_days)))).isoformat()


def _load_bars(symbols: list[str], *, lookback_days: int, max_symbols: int) -> tuple[dict[str, list[dict]], list[str]]:
    from tradingview_mcp.core.data.factory import get_layer

    layer = get_layer()
    store = layer.store
    start = _bar_window(lookback_days)
    bars_by_symbol: dict[str, list[dict]] = {}
    skipped: list[str] = []
    for symbol in symbols[:max_symbols]:
        bars = store.get_ohlcv(symbol, start=start)
        if len(bars) >= 60:
            bars_by_symbol[symbol.upper()] = bars
        elif bars:
            skipped.append(symbol.upper())
    return bars_by_symbol, skipped


def _payload_warnings(bars_by_symbol: dict[str, list[dict]], horizon: int, *, folds: int) -> list[str]:
    warnings: list[str] = []
    sessions = max((len(b) for b in bars_by_symbol.values()), default=0)
    if len(bars_by_symbol) < 5:
        warnings.append(
            f"only {len(bars_by_symbol)} symbol(s) have 60+ stored sessions: the cross-sectional "
            "benchmark is thin, so 'excess' is closer to a raw return than it should be"
        )
    if 0 < sessions < 250:
        warnings.append(
            f"longest history is {sessions} sessions (~{sessions // 21} months). That is below the "
            "120-session volume-profile window plus a horizon, so conditions like c1 and the "
            "accumulation composites can report NO_EVENTS for lack of history rather than lack of "
            "signal; raise lookback_days or sync more history."
        )
        warnings.append(
            f"treat every verdict as preliminary below ~250 sessions ({sessions} here), and expect "
            "walk-forward folds to report INSUFFICIENT_HISTORY"
        )
    from tradingview_mcp.core.calibration.study import MIN_SESSIONS_PER_FOLD

    if folds and sessions < folds * MIN_SESSIONS_PER_FOLD:
        warnings.append(
            f"not enough sessions for {folds} walk-forward folds "
            f"(needs {folds * MIN_SESSIONS_PER_FOLD}, longest symbol has {sessions})"
        )
    if horizon > 20:
        warnings.append("horizons beyond 20 sessions overlap heavily across events; the p-value cannot correct that")
    return warnings


def egx_event_study(
    *,
    signal: str = "",
    horizon: int = 5,
    symbols: str = "",
    index: str = "",
    lookback_days: int = 1500,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    permutations: int = 200,
    walk_forward_folds: int = 3,
) -> dict[str, Any]:
    """Measure what each smart-money condition actually predicted, historically.

    Runs entirely on the local store — no network, works in offline mode — and
    returns calibrated numbers instead of point estimates: hit rate with a
    confidence interval, the base rate it must beat, a permutation p-value,
    a multiple-testing-corrected q-value, and a walk-forward verdict.

    Args:
      signal: one name, several comma/semicolon-separated, or empty for the
              default batch. The FDR correction covers whatever is measured.
      horizon: sessions held after the signal (entry at the next session's open).
      symbols: explicit comma/semicolon-separated EGX tickers (optional).
      index: EGX30 / EGX70 / EGX100 / SHARIAH33 / EGX35LV / TAMAYUZ (optional).
      lookback_days: how far back to read the store. Must comfortably exceed
        the longest signal window (the volume profile uses 120 sessions).
      max_symbols: cap on symbols analysed (cost is linear in symbols × sessions).
      permutations: block-permutation rounds per signal (0 disables p-values).
      walk_forward_folds: contiguous out-of-sample folds (0 disables).
    """
    if horizon < 1 or horizon > 250:
        return make_error(
            ErrorCode.INVALID_PARAMETER, "horizon must be between 1 and 250 sessions", horizon=horizon
        )
    if permutations < 0 or permutations > MAX_PERMUTATIONS:
        return make_error(
            ErrorCode.INVALID_PARAMETER,
            f"permutations must be between 0 and {MAX_PERMUTATIONS}",
            permutations=permutations,
        )
    if max_symbols < 1 or max_symbols > 400:
        return make_error(
            ErrorCode.INVALID_PARAMETER, "max_symbols must be between 1 and 400", max_symbols=max_symbols
        )

    try:
        requested = _resolve_symbols(symbols, index) if (symbols or index) else []
    except ValueError as exc:
        return make_error(ErrorCode.INVALID_PARAMETER, str(exc), index=index)

    if not requested:
        requested = _resolve_symbols("", "", limit=max_symbols)

    try:
        requested_signals = [s.strip() for s in (signal or "").replace(";", ",").split(",") if s.strip()]
        chosen = requested_signals or list(DEFAULT_SIGNALS)
        if requested_signals:
            from tradingview_mcp.core.calibration import get_signal

            for name in requested_signals:
                get_signal(name)  # raises KeyError listing the known names
    except KeyError as exc:
        return make_error(ErrorCode.INVALID_PARAMETER, str(exc), signal=signal)

    bars_by_symbol, skipped = _load_bars(requested, lookback_days=lookback_days, max_symbols=max_symbols)
    if not bars_by_symbol:
        return make_error(
            ErrorCode.NO_DATA,
            "no stored daily history for the requested symbols: run "
            "`python scripts/egx_data_sync.py daily --index EGX30 --days 365` (or `universe`) first",
            requested=len(requested),
            lookback_days=lookback_days,
        )

    sample = Sample(bars_by_symbol=bars_by_symbol, horizon=horizon)
    report = analyze(
        sample,
        signals=chosen,
        permutations=permutations,
        walk_forward_folds=walk_forward_folds,
    )
    payload: dict[str, Any] = {
        "mode": "offline" if offline_mode() else "online",
        "horizon": horizon,
        "measurement": {
            "symbols_analysed": len(bars_by_symbol),
            "symbols_requested": len(requested),
            "symbols_skipped_short_history": skipped[:20],
            "sessions": report.sessions,
            "window": {"from": report.window[0], "to": report.window[1]},
            "permutations": permutations,
            "method": (
                "signal at session t (close), entry at t+1 open, exit at t+horizon close; "
                "excess vs the equal-weighted return of every stored symbol over the same dates; "
                "p-values from a block permutation of each symbol's full return series; q-values "
                "from Benjamini-Hochberg across every signal measured in this run"
            ),
        },
        "measurements": [m.as_dict() for m in report.measurements],
        "walk_forward": {name: wf.as_dict() for name, wf in report.walk_forwards.items()},
        "validated_signals": [m.signal for m in report.validated],
        "warnings": _payload_warnings(bars_by_symbol, horizon, folds=walk_forward_folds),
        "disclaimer": (
            "Measured on the local store only; a verdict describes this sample, not the future. "
            "VALIDATED means the edge survived multiple-testing correction and out-of-sample folds "
            "here — not that it will persist."
        ),
    }
    return attach_glossary(payload, EVENT_GLOSSARY)


def egx_list_events() -> dict[str, Any]:
    """List the registered signals with the hypothesis each one is tested against."""
    from tradingview_mcp.core.calibration import all_signals

    return {
        "signals": [
            {
                "name": s.name,
                "title": s.title,
                "kind": s.kind,
                "hypothesis": s.hypothesis,
                "default": s.name in DEFAULT_SIGNALS,
            }
            for s in all_signals()
        ],
        "count": len(all_signals()),
    }
