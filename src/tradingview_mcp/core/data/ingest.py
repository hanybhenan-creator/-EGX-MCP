"""Ingestion: turn router results into a durable, auditable local history.

The router answers questions; this module *fills the store*. It owns the three
things that make a local dataset trustworthy:

1. **A refresh is a transaction with a report.** Every run opens an
   ``ingest_runs`` row, records what it intended to do, what each symbol cost
   (source, latency, rows), what failed and why, and closes with a status. A
   crash leaves the row marked ``running`` — visible, not silent.
2. **Skipping is a decision, not an omission.** A symbol that the calendar says
   has no new session, or that the store already has up to date, is *skipped
   with a reason*; it is never quietly dropped from the results.
3. **Coverage is measured against the EGX calendar.** "How complete is my data?"
   is answered by comparing stored dates to expected trading days (Sun–Thu,
   minus holidays), not to a naive day count.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable, Optional, Sequence

from .calendar import EGXCalendar
from .provenance import QualityFlag, Sourced
from .router import AllSourcesFailed, DataRouter, Resolution
from .sources.base import Capability
from .store import DataStore, RunHandle


@dataclass
class SymbolOutcome:
    """What happened for one symbol in one run."""

    symbol: str
    status: str  # ok | failed | skipped | empty
    rows: int = 0
    source: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    flags: list[str] = field(default_factory=list)
    reason: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "rows": self.rows,
            "source": self.source,
            "latency_ms": round(self.latency_ms, 1) if self.latency_ms else None,
            "error": self.error,
            "reason": self.reason,
            "flags": self.flags,
        }


@dataclass
class RefreshReport:
    kind: str
    run_id: int
    requested: int
    started_at: str
    finished_at: Optional[str] = None
    status: str = "running"
    rows_written: int = 0
    sources_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    outcomes: list[SymbolOutcome] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for o in self.outcomes:
            out[o.status] = out.get(o.status, 0) + 1
        return out

    def as_dict(self, *, include_outcomes: bool = True, limit: int = 50) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "run_id": self.run_id,
            "status": self.status,
            "requested": self.requested,
            "rows_written": self.rows_written,
            "sources_used": self.sources_used,
            "counts": self.counts,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "warnings": self.warnings,
        }
        if include_outcomes:
            failed = [o.as_dict() for o in self.outcomes if o.status in {"failed", "empty"}]
            payload["problems"] = failed[:limit]
            payload["problems_total"] = len(failed)
        return payload


class EGXIngestor:
    """Populates the store from the router, with calendar-aware skipping."""

    def __init__(
        self,
        router: DataRouter,
        store: DataStore,
        *,
        calendar: Optional[EGXCalendar] = None,
    ) -> None:
        self.router = router
        self.store = store
        self.calendar = calendar or EGXCalendar.from_env()
        self._write_lock = threading.Lock()

    # ── daily history ───────────────────────────────────────────────────────

    def refresh_daily(
        self,
        symbols: Sequence[str],
        *,
        days: int = 365,
        cross_check: bool = False,
        max_workers: int = 4,
        force: bool = False,
    ) -> RefreshReport:
        """Fetch settled daily bars and store them with provenance.

        Symbols already current per the calendar are skipped unless ``force`` —
        a nightly refresh over 292 EGX names should cost one request per symbol,
        not one per symbol per run.
        """
        unique = _dedupe(symbols)
        handle = self.store.begin_run(
            "ohlcv_daily",
            params={"symbols": len(unique), "days": days, "cross_check": cross_check, "force": force},
        )
        report = RefreshReport(
            kind="ohlcv_daily",
            run_id=handle.run_id,
            requested=len(unique),
            started_at=handle.started_at.isoformat(),
        )
        report.warnings.extend(self.calendar.sync_warnings(self.calendar.now().year))

        expected_last = _expected_last_bar(self.calendar)

        if not unique:
            return self._finalize(handle, report, status="empty")

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = {
                pool.submit(
                    self._refresh_one_daily, symbol, days, cross_check, force, expected_last
                ): symbol
                for symbol in unique
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    outcome = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    outcome = SymbolOutcome(symbol=symbol, status="failed", error=f"{type(exc).__name__}: {exc}")
                report.outcomes.append(outcome)

        report.outcomes.sort(key=lambda o: o.symbol)
        status = "ok" if not any(o.status == "failed" for o in report.outcomes) else "partial"
        return self._finalize(handle, report, status=status)

    def _refresh_one_daily(
        self,
        symbol: str,
        days: int,
        cross_check: bool,
        force: bool,
        expected_last: Optional[date],
    ) -> SymbolOutcome:
        if not force and expected_last is not None:
            latest = self.store.latest_ohlcv_date(symbol)
            if latest and date.fromisoformat(latest) >= expected_last:
                return SymbolOutcome(
                    symbol=symbol,
                    status="skipped",
                    reason=f"store already current through {latest}",
                )

        try:
            resolution = self.router.fetch(
                Capability.DAILY_OHLCV,
                symbol,
                days=days,
                cross_check=cross_check,
                max_corroborators=1,
            )
        except AllSourcesFailed as exc:
            return SymbolOutcome(symbol=symbol, status="failed", error=str(exc))

        if not resolution.ok:
            first = resolution.attempts[0].error if resolution.attempts else "no sources"
            return SymbolOutcome(symbol=symbol, status="failed", error=first or "no sources")

        result: Sourced[Any] = resolution.result
        bars = [b for b in result.value if b.get("close") is not None]
        if not bars:
            return SymbolOutcome(
                symbol=symbol,
                status="empty",
                source=result.provenance.source,
                flags=[f.value for f in result.flags],
                error="source returned no usable bars",
            )

        with self._write_lock:
            written = self.store.write_ohlcv(
                symbol,
                bars,
                source=result.provenance.source,
                fetched_at=result.provenance.fetched_at,
                quality_flags=result.flags,
            )
            self.store.record_flags(result, symbol=symbol)

        return SymbolOutcome(
            symbol=symbol,
            status="ok",
            rows=written,
            source=result.provenance.source,
            latency_ms=result.provenance.latency_ms,
            flags=[f.value for f in result.flags if f is not QualityFlag.VALID],
        )

    # ── quotes ──────────────────────────────────────────────────────────────

    def refresh_quotes(
        self,
        symbols: Sequence[str],
        *,
        max_workers: int = 4,
        cross_check: bool = False,
    ) -> RefreshReport:
        """Append a quote snapshot per symbol (append-only, never overwritten)."""
        unique = _dedupe(symbols)
        handle = self.store.begin_run("quotes", params={"symbols": len(unique)})
        report = RefreshReport(
            kind="quotes",
            run_id=handle.run_id,
            requested=len(unique),
            started_at=handle.started_at.isoformat(),
        )
        report.warnings.extend(self.calendar.sync_warnings(self.calendar.now().year))

        if not unique:
            return self._finalize(handle, report, status="empty")

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = {
                pool.submit(self._refresh_one_quote, symbol, cross_check): symbol for symbol in unique
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    outcome = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    outcome = SymbolOutcome(symbol=symbol, status="failed", error=f"{type(exc).__name__}: {exc}")
                report.outcomes.append(outcome)

        report.outcomes.sort(key=lambda o: o.symbol)
        status = "ok" if not any(o.status == "failed" for o in report.outcomes) else "partial"
        return self._finalize(handle, report, status=status)

    def _refresh_one_quote(self, symbol: str, cross_check: bool) -> SymbolOutcome:
        try:
            resolution = self.router.fetch(
                Capability.QUOTE, symbol, cross_check=cross_check, max_corroborators=1
            )
        except AllSourcesFailed as exc:
            return SymbolOutcome(symbol=symbol, status="failed", error=str(exc))

        if not resolution.ok:
            first = resolution.attempts[0].error if resolution.attempts else "no sources"
            return SymbolOutcome(symbol=symbol, status="failed", error=first or "no sources")

        result = resolution.result
        quote = dict(result.value)
        quote.setdefault("symbol", symbol)
        with self._write_lock:
            written = self.store.write_quote(
                quote,
                source=result.provenance.source,
                fetched_at=result.provenance.fetched_at,
                quality_flags=result.flags,
                is_stale=QualityFlag.STALE in result.flags,
            )
            self.store.record_flags(result, symbol=symbol)

        return SymbolOutcome(
            symbol=symbol,
            status="ok" if written else "empty",
            rows=written,
            source=result.provenance.source,
            latency_ms=result.provenance.latency_ms,
            flags=[f.value for f in result.flags if f is not QualityFlag.VALID],
            error=None if written else "quote rejected by the store",
        )

    # ── market-wide snapshot ────────────────────────────────────────────────

    def refresh_universe(self, *, cross_check: bool = True) -> RefreshReport:
        """Fetch the whole-market snapshot: symbol master + quote rows.

        This is the cheapest way to learn *what exists* on the EGX, so it also
        refreshes the symbol table — a newly listed company appears here before
        anything else in the system knows about it.
        """
        handle = self.store.begin_run("universe_snapshot", params={"cross_check": cross_check})
        report = RefreshReport(
            kind="universe_snapshot", run_id=handle.run_id, requested=1,
            started_at=handle.started_at.isoformat(),
        )
        report.warnings.extend(self.calendar.sync_warnings(self.calendar.now().year))

        try:
            resolution = self.router.fetch(
                Capability.MARKET_SNAPSHOT, None, cross_check=cross_check, max_corroborators=1
            )
        except AllSourcesFailed as exc:
            report.outcomes.append(SymbolOutcome(symbol="<market>", status="failed", error=str(exc)))
            return self._finalize(handle, report, status="failed")

        if not resolution.ok:
            outcome = SymbolOutcome(
                symbol="<market>",
                status="failed",
                error="; ".join(a.error or "unknown" for a in resolution.attempts) or "no sources",
            )
            report.outcomes.append(outcome)
            return self._finalize(handle, report, status="failed")

        result = resolution.result
        rows = [r for r in result.value if r.get("symbol")]
        with self._write_lock:
            self.store.upsert_symbols(rows, source=result.provenance.source)
            for row in rows:
                self.store.write_quote(
                    row,
                    source=result.provenance.source,
                    fetched_at=result.provenance.fetched_at,
                    quality_flags=result.flags,
                )
            self.store.record_flags(result, symbol=None)

        report.outcomes.append(
            SymbolOutcome(
                symbol="<market>",
                status="ok",
                rows=len(rows),
                source=result.provenance.source,
                latency_ms=result.provenance.latency_ms,
                flags=[f.value for f in result.flags if f is not QualityFlag.VALID],
            )
        )
        report.rows_written = len(rows)
        return self._finalize(handle, report, status="ok")

    # ── reporting ───────────────────────────────────────────────────────────

    def coverage(
        self,
        symbols: Optional[Sequence[str]] = None,
        *,
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        """Answer "how complete is my data?" against the EGX calendar.

        The window ends at the newest **settled** session, not at "today": on a
        trading day before the close, today's bar does not exist yet and must
        not be reported as a gap. Counting it would make every intraday check
        look broken and train the operator to ignore the report.
        """
        today = _expected_last_bar(self.calendar) or self.calendar.now().date()
        start = today - timedelta(days=lookback_days)
        expected = self.calendar.trading_days_between(start, today)

        checked = _dedupe(symbols) if symbols else [s["symbol"] for s in self.store.get_symbols()]
        rows: list[dict[str, Any]] = []
        complete = 0

        for symbol in checked[:500]:
            gaps = self.store.ohlcv_gaps(symbol, start, today, calendar=self.calendar)
            latest = self.store.latest_ohlcv_date(symbol)
            is_current = bool(latest) and self.calendar.data_is_current(date.fromisoformat(latest))
            if is_current and not gaps:
                complete += 1
            rows.append(
                {
                    "symbol": symbol,
                    "latest_bar": latest,
                    "current": is_current,
                    "missing_sessions": len(gaps),
                    "first_missing": gaps[0] if gaps else None,
                }
            )

        stale = [r for r in rows if not r["current"]]
        return {
            "window": {"from": start.isoformat(), "to": today.isoformat(), "expected_sessions": len(expected)},
            "symbols_checked": len(rows),
            "complete": complete,
            "incomplete": len(rows) - complete,
            "current_ratio": round(complete / len(rows), 4) if rows else None,
            "worst_offenders": sorted(stale, key=lambda r: r["missing_sessions"], reverse=True)[:20],
        }

    def _finalize(self, handle: RunHandle, report: RefreshReport, *, status: str) -> RefreshReport:
        report.status = status
        report.rows_written = sum(o.rows for o in report.outcomes if o.status == "ok")
        report.sources_used = sorted({o.source for o in report.outcomes if o.source})
        report.finished_at = None
        self.store.finish_run(
            handle,
            status=status,
            rows_written=report.rows_written,
            sources_used=report.sources_used,
            warnings=report.warnings,
        )
        report.finished_at = _now_iso()
        return report


# ── helpers ─────────────────────────────────────────────────────────────────


def _dedupe(symbols: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for raw in symbols:
        if not raw:
            continue
        symbol = str(raw).strip().upper().replace("EGX:", "")
        if symbol:
            seen.setdefault(symbol, None)
    return list(seen)


def _expected_last_bar(calendar: EGXCalendar) -> Optional[date]:
    """The newest session the store should already hold (see quality._expected_bar)."""
    from .quality import _expected_bar

    try:
        return _expected_bar(calendar, None)
    except Exception:  # pragma: no cover - defensive
        return None


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def symbols_from_coinlist(path: Optional[str] = None) -> list[str]:
    """Bare EGX tickers from ``coinlist/egx.txt`` (``EGX:COMI`` → ``COMI``)."""
    from pathlib import Path

    target = Path(path) if path else Path(__file__).resolve().parents[2] / "coinlist" / "egx.txt"
    if not target.exists():  # pragma: no cover - packaging edge case
        return []
    out: list[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split(":")[-1].upper())
    return _dedupe(out)


def symbols_for_index(index: str, *, limit: Optional[int] = None) -> list[str]:
    """Constituents of an EGX index, from the checked-in constituents module."""
    from .egx_indices import EGX_INDICES

    key = (index or "").strip().upper()
    entry = EGX_INDICES.get(key)
    if not entry:
        raise ValueError(f"unknown EGX index {index!r}; known: {', '.join(EGX_INDICES)}")
    symbols = _dedupe(entry["get_symbols"]())
    return symbols[:limit] if limit else symbols
