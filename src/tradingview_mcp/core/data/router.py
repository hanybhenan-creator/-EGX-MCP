"""Data router — one question, several sources, an auditable answer.

The router is the only place that knows how to turn a *capability request*
("give me settled daily bars for COMI") into a *result with evidence*. It:

1. asks the registry for sources that declare the capability, best-first
   (official → priority → healthy);
2. calls each in turn, running the quality gates appropriate to the
   capability;
3. accepts the first result that is structurally sound and fresh, unless the
   caller asked for corroboration, in which case it keeps going and
   triangulates;
4. records every attempt — including the failures — so the audit table can
   answer "why is this number here and why did source X not contribute?".

The router never averages sources, never fills gaps with a different source
silently, and never returns a value whose blocking quality flag is set. When
everything fails it raises :class:`AllSourcesFailed` carrying the full attempt
list, because "no data" with reasons is actionable and "no data" alone is not.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from .provenance import QualityFlag, Sourced
from .quality import (
    SourceObservation,
    attach_verdicts,
    check_freshness,
    check_price_jump,
    check_volume_anomaly,
    triangulate,
    validate_ohlcv_bars,
    validate_quote,
)
from .transport import TransportError
from .sources.base import (
    Capability,
    MarketDataSource,
    SourceDataError,
    SourceParseError,
    SourceRegistry,
    SourceUnavailable,
)

#: Capability → the method a source must expose to serve it.
METHOD_FOR: dict[Capability, str] = {
    Capability.SYMBOL_MASTER: "symbol_master",
    Capability.QUOTE: "quote",
    Capability.MARKET_SNAPSHOT: "market_snapshot",
    Capability.DAILY_OHLCV: "daily_ohlcv",
    Capability.INTRADAY_OHLCV: "intraday_ohlcv",
    Capability.INDEX_QUOTE: "index_quote",
    Capability.SECTOR_STATS: "sector_stats",
    Capability.COMPANY_PROFILE: "profile",
}

_SIG_CACHE: dict[tuple[type, str], set[str]] = {}


class AllSourcesFailed(RuntimeError):
    """Every candidate source failed; carries the per-source reasons."""

    def __init__(self, capability: Capability, subject: Optional[str], attempts: Sequence["Attempt"]):
        self.capability = capability
        self.subject = subject
        self.attempts = list(attempts)
        reasons = "; ".join(f"{a.source}: {a.error}" for a in self.attempts) or "no sources registered"
        super().__init__(f"{capability.value} for {subject or '-'} unavailable — {reasons}")

    def as_attempts(self) -> list[dict[str, Any]]:
        return [a.as_dict() for a in self.attempts]


@dataclass
class Attempt:
    """One source call, successful or not. The raw material of the audit log."""

    source: str
    capability: Capability
    ok: bool
    latency_ms: Optional[float] = None
    rows: Optional[int] = None
    http_status: Optional[int] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None
    flags: list[str] = field(default_factory=list)
    endpoint: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "capability": self.capability.value,
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 1) if self.latency_ms is not None else None,
            "rows": self.rows,
            "http_status": self.http_status,
            "error": self.error,
            "error_kind": self.error_kind,
            "flags": self.flags,
            "endpoint": self.endpoint,
        }


@dataclass
class Resolution:
    """The router's answer: the value, its provenance, and what else was tried."""

    capability: Capability
    subject: Optional[str]
    result: Optional[Sourced[Any]]
    attempts: list[Attempt]
    #: Other sources consulted for corroboration, when requested.
    corroborating: list[Sourced[Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.result is not None

    @property
    def value(self) -> Any:
        return self.result.value if self.result else None

    @property
    def flags(self) -> list[QualityFlag]:
        return list(self.result.flags) if self.result else []

    @property
    def corroborated(self) -> bool:
        return bool(self.result and self.result.is_cross_checked)

    @property
    def sources_tried(self) -> list[str]:
        return [a.source for a in self.attempts]

    def as_dict(self, *, include_value: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "capability": self.capability.value,
            "subject": self.subject,
            "ok": self.ok,
            "sources_tried": self.sources_tried,
            "attempts": [a.as_dict() for a in self.attempts],
            "corroborating_sources": [c.provenance.source for c in self.corroborating],
        }
        if self.result is not None:
            payload.update(self.result.as_dict())
        if include_value:
            payload["value"] = self.value
        return payload


def _accepted_kwargs(source: MarketDataSource, method_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs to what *source*'s method actually accepts.

    Sources in the chain legitimately have different signatures (Yahoo's
    ``daily_ohlcv`` takes ``days``, the EGX scraper takes none). Rather than
    force a lowest-common-denominator signature on every adapter, the router
    passes only the arguments each one declares — and drops the rest rather
    than raising a TypeError at the worst possible moment.
    """
    key = (type(source), method_name)
    params = _SIG_CACHE.get(key)
    if params is None:
        method = getattr(source, method_name, None)
        if method is None:
            return {}
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):  # pragma: no cover
            return dict(kwargs)
        params = {
            name
            for name, p in sig.parameters.items()
            if name != "self"
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        _SIG_CACHE[key] = params
    return {k: v for k, v in kwargs.items() if k in params}


class DataRouter:
    """Capability-oriented access with fallback, quality gates and auditing."""

    def __init__(
        self,
        registry: SourceRegistry,
        *,
        calendar=None,
        audit_sink: Optional[Callable[[Attempt, Optional[str]], None]] = None,
        max_sources_per_request: int = 4,
    ) -> None:
        self.registry = registry
        self.calendar = calendar
        self.audit_sink = audit_sink
        self.max_sources_per_request = max_sources_per_request

    # ── public API ──────────────────────────────────────────────────────────

    def fetch(
        self,
        capability: Capability,
        subject: Optional[str] = None,
        *,
        cross_check: bool = False,
        max_corroborators: int = 1,
        allow_partial: bool = True,
        sources: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> Resolution:
        """Serve *capability* for *subject*, falling back across sources.

        Args:
            cross_check: keep going after a good result to collect independent
                corroboration (slower, but turns the value into a *verified*
                one).
            allow_partial: accept a result carrying ``PARTIAL`` (missing
                optional fields). ``INVALID``/``STALE``/``CONFLICT`` are never
                accepted.
            sources: restrict to these source names (used by tests and by
                callers that want "EGX only" behaviour).
        """
        wanted = set(sources) if sources else None
        candidates = [
            s
            for s in self.registry.for_capability(capability)
            if wanted is None or s.name in wanted
        ][: self.max_sources_per_request]

        attempts: list[Attempt] = []
        winner: Optional[Sourced[Any]] = None
        corroborating: list[Sourced[Any]] = []

        if not candidates:
            raise AllSourcesFailed(capability, subject, attempts)

        for index, source in enumerate(candidates):
            method_name = METHOD_FOR.get(capability)
            if not method_name:
                raise ValueError(f"no method mapped for capability {capability.value}")

            call_kwargs = _accepted_kwargs(source, method_name, kwargs)
            if "attempt_index" in _accepted_kwargs(source, method_name, {"attempt_index": index}):
                call_kwargs["attempt_index"] = index

            result: Optional[Sourced[Any]] = None
            try:
                method: Callable[..., Sourced[Any]] = getattr(source, method_name)
                result = method(subject, **call_kwargs) if _takes_subject(source, method_name) else method(**call_kwargs)
                if result is None:  # pragma: no cover - defensive
                    raise SourceDataError(source.name, "source returned None")
            except (SourceUnavailable, SourceParseError, SourceDataError, TransportError) as exc:
                # Reachability failures trip the breaker; content failures
                # (parse/data) do not — a renamed page is not an outage.
                if isinstance(exc, TransportError) or (
                    isinstance(exc, SourceUnavailable)
                    and not isinstance(exc, (SourceParseError, SourceDataError))
                ):
                    source.health.record_failure(exc)
                attempt = _failed_attempt(source, capability, exc)
                attempts.append(attempt)
                self._audit(attempt, subject)
                continue
            except Exception as exc:  # unexpected: still not fatal for the chain
                source.health.record_failure(exc)
                attempt = Attempt(
                    source=source.name,
                    capability=capability,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    error_kind="unexpected",
                )
                attempts.append(attempt)
                self._audit(attempt, subject)
                continue

            source.health.record_success()
            self._apply_gates(capability, result)
            attempt = Attempt(
                source=source.name,
                capability=capability,
                ok=True,
                latency_ms=result.provenance.latency_ms,
                rows=_row_count(result.value),
                http_status=result.provenance.http_status,
                flags=[f.value for f in result.flags],
                endpoint=result.provenance.endpoint,
            )
            attempts.append(attempt)
            self._audit(attempt, subject)

            blocking = [f for f in result.flags if f in _BLOCKING]
            if blocking and not allow_partial:
                continue
            if blocking:
                # A value that violates a hard rule is not a fallback: keep
                # looking, and if nothing better exists the failure list will
                # explain why.
                continue

            if winner is None:
                winner = result
                if not cross_check:
                    break
                continue

            if len(corroborating) < max_corroborators:
                corroborating.append(result)

        if winner is not None and corroborating:
            self._cross_validate(capability, winner, corroborating)
            self._audit_cross_check(winner, subject)

        return Resolution(
            capability=capability,
            subject=subject,
            result=winner,
            attempts=attempts,
            corroborating=corroborating,
        )

    def fetch_or_raise(self, capability: Capability, subject: Optional[str] = None, **kwargs: Any) -> Sourced[Any]:
        """Same as :meth:`fetch` but raises :class:`AllSourcesFailed` when empty."""
        res = self.fetch(capability, subject, **kwargs)
        if not res.ok:
            raise AllSourcesFailed(capability, subject, res.attempts)
        return res.result  # type: ignore[return-value]

    def health_report(self) -> list[dict[str, Any]]:
        return self.registry.health_report()

    # ── gates ───────────────────────────────────────────────────────────────

    def _apply_gates(self, capability: Capability, result: Sourced[Any]) -> None:
        """Run the capability-appropriate quality gates in place."""
        value = result.value

        if capability is Capability.QUOTE:
            attach_verdicts(
                result,
                validate_quote(
                    price=value.get("price") or value.get("close"),
                    bid=value.get("bid"),
                    ask=value.get("ask"),
                    volume=value.get("volume"),
                    previous_close=value.get("previous_close"),
                ),
            )
            check_freshness(result, kind="quote", calendar=self.calendar)
            return

        if capability is Capability.INDEX_QUOTE:
            attach_verdicts(
                result,
                validate_quote(
                    price=value.get("close") or value.get("price") or value.get("value"),
                    previous_close=value.get("previous_close"),
                ),
            )
            check_freshness(result, kind="quote", calendar=self.calendar)
            return

        if capability is Capability.DAILY_OHLCV:
            bars = value if isinstance(value, list) else []
            attach_verdicts(result, validate_ohlcv_bars(bars))
            attach_verdicts(result, check_price_jump([b.get("close") for b in bars if b.get("close")]))
            attach_verdicts(
                result, check_volume_anomaly([b.get("volume") or 0 for b in bars])
            )
            check_freshness(result, kind="daily_bar", calendar=self.calendar)
            return

        if capability is Capability.INTRADAY_OHLCV:
            bars = value if isinstance(value, list) else []
            attach_verdicts(result, validate_ohlcv_bars(bars))
            return

        if capability is Capability.MARKET_SNAPSHOT:
            rows = value if isinstance(value, list) else []
            bad = 0
            zero_price = 0
            for row in rows:
                price = row.get("close") or row.get("price")
                if price is None:
                    bad += 1
                elif price <= 0:
                    zero_price += 1
            if rows and (bad + zero_price) / len(rows) > 0.05:
                result.add_flag(
                    QualityFlag.INVALID,
                    f"{bad + zero_price}/{len(rows)} snapshot rows have no usable price",
                )
            elif bad or zero_price:
                result.add_flag(
                    QualityFlag.PARTIAL,
                    f"{bad + zero_price}/{len(rows)} snapshot rows lack a usable price",
                )
            else:
                result.add_flag(QualityFlag.VALID, f"{len(rows)} snapshot rows priced")
            check_freshness(result, kind="quote", calendar=self.calendar)
            return

        # SYMBOL_MASTER / SECTOR_STATS / COMPANY_PROFILE: structural only.
        if isinstance(value, list):
            if not value:
                result.add_flag(QualityFlag.INVALID, "empty payload")
            else:
                result.add_flag(QualityFlag.VALID, f"{len(value)} rows")
        elif isinstance(value, dict):
            if not value:
                result.add_flag(QualityFlag.INVALID, "empty payload")
            else:
                result.add_flag(QualityFlag.VALID)

    def _cross_validate(
        self,
        capability: Capability,
        winner: Sourced[Any],
        others: Sequence[Sourced[Any]],
    ) -> None:
        """Triangulate the winner against corroborating sources."""
        if capability is Capability.QUOTE:
            fields = ["price", "volume", "previous_close"]
            winner_values = {
                f: _as_float(winner.value.get(f) or winner.value.get("close")) for f in fields
            }
            observations = [
                SourceObservation(winner.provenance.source, _drop_none(winner_values), winner.provenance.fetched_at)
            ]
            for other in others:
                vals = {f: _as_float(other.value.get(f) or other.value.get("close")) for f in fields}
                observations.append(
                    SourceObservation(other.provenance.source, _drop_none(vals), other.provenance.fetched_at)
                )
            flag, checks, note = triangulate(observations, fields=[f for f in fields if not _is_none(winner_values.get(f))])
            winner.cross_checks = checks
            if flag is not QualityFlag.UNCROSSCHECKED:
                winner.add_flag(flag, note)
            return

        if capability in {Capability.MARKET_SNAPSHOT, Capability.SYMBOL_MASTER}:
            self._cross_validate_snapshot(winner, others)
            return

        if capability is Capability.DAILY_OHLCV:
            self._cross_validate_bars(winner, others)
            return

        winner.add_flag(
            QualityFlag.UNCROSSCHECKED,
            f"no cross-check implemented for capability {capability.value}",
        )

    def _cross_validate_snapshot(
        self, winner: Sourced[list[dict[str, Any]]], others: Sequence[Sourced[Any]]
    ) -> None:
        winner_map = _by_symbol(winner.value)
        checks: list[dict[str, Any]] = []
        compared = 0
        agreed = 0
        for other in others:
            other_map = _by_symbol(other.value)
            for sym, row in other_map.items():
                mine = winner_map.get(sym)
                if not mine:
                    continue
                a, b = _as_float(mine.get("close") or mine.get("price")), _as_float(
                    row.get("close") or row.get("price")
                )
                if not a or not b:
                    continue
                compared += 1
                rel = abs(a - b) / a
                if rel <= 0.01:
                    agreed += 1
                elif len(checks) < 25:
                    checks.append(
                        {
                            "field": "close",
                            "symbol": sym,
                            "values": {winner.provenance.source: a, other.provenance.source: b},
                            "relative_spread": round(rel, 6),
                            "tolerance": 0.01,
                            "verdict": "CONFLICT" if rel > 0.03 else "DIVERGENCE",
                        }
                    )
        winner.cross_checks = checks
        if compared == 0:
            winner.add_flag(
                QualityFlag.UNCROSSCHECKED,
                f"no overlapping symbols with {', '.join(o.provenance.source for o in others)}",
            )
        elif agreed == compared:
            winner.add_flag(
                QualityFlag.AGREEMENT,
                f"{agreed}/{compared} overlapping symbols agree within 1% with "
                f"{', '.join(o.provenance.source for o in others)}",
            )
        else:
            frac = agreed / compared
            winner.add_flag(
                QualityFlag.CONFLICT if frac < 0.8 else QualityFlag.DIVERGENCE,
                f"{agreed}/{compared} overlapping symbols agree within 1% "
                f"({compared - agreed} disagree)",
            )

    def _cross_validate_bars(
        self, winner: Sourced[list[dict[str, Any]]], others: Sequence[Sourced[Any]]
    ) -> None:
        winner_by_date = {str(b.get("date")): b for b in winner.value}
        checks: list[dict[str, Any]] = []
        compared = 0
        agreed = 0
        for other in others:
            for bar in other.value if isinstance(other.value, list) else []:
                mine = winner_by_date.get(str(bar.get("date")))
                if not mine:
                    continue
                a, b = _as_float(mine.get("close")), _as_float(bar.get("close"))
                if not a or not b:
                    continue
                compared += 1
                rel = abs(a - b) / a
                if rel <= 0.01:
                    agreed += 1
                elif len(checks) < 25:
                    checks.append(
                        {
                            "field": "close",
                            "date": bar.get("date"),
                            "values": {winner.provenance.source: a, other.provenance.source: b},
                            "relative_spread": round(rel, 6),
                            "tolerance": 0.01,
                            "verdict": "CONFLICT" if rel > 0.03 else "DIVERGENCE",
                        }
                    )
        winner.cross_checks = checks
        if compared == 0:
            winner.add_flag(
                QualityFlag.UNCROSSCHECKED,
                "no overlapping bar dates with the other source(s)",
            )
        elif agreed == compared:
            winner.add_flag(
                QualityFlag.AGREEMENT,
                f"all {compared} overlapping daily closes agree within 1%",
            )
        else:
            frac = agreed / compared
            winner.add_flag(
                QualityFlag.CONFLICT if frac < 0.8 else QualityFlag.DIVERGENCE,
                f"{agreed}/{compared} overlapping daily closes agree within 1%",
            )

    # ── auditing ────────────────────────────────────────────────────────────

    def _audit(self, attempt: Attempt, subject: Optional[str]) -> None:
        if self.audit_sink is None:
            return
        try:
            self.audit_sink(attempt, subject)
        except Exception:  # pragma: no cover - auditing must never break fetching
            pass

    def _audit_cross_check(self, winner: Sourced[Any], subject: Optional[str]) -> None:
        if self.audit_sink is None or not winner.cross_checks:
            return
        try:
            self.audit_sink(
                Attempt(
                    source=winner.provenance.source,
                    capability=Capability.QUOTE
                    if isinstance(winner.value, dict)
                    else Capability.DAILY_OHLCV,
                    ok=True,
                    flags=[f.value for f in winner.flags],
                    error=None,
                ),
                subject,
            )
        except Exception:  # pragma: no cover
            pass


_BLOCKING = {QualityFlag.INVALID, QualityFlag.STALE, QualityFlag.CONFLICT}


def _failed_attempt(source: MarketDataSource, capability: Capability, exc: Exception) -> Attempt:
    return Attempt(
        source=source.name,
        capability=capability,
        ok=False,
        error=str(exc),
        error_kind=type(exc).__name__,
    )


def _takes_subject(source: MarketDataSource, method_name: str) -> bool:
    """Does this source's method take a subject argument (symbol/index)?"""
    key = (type(source), method_name + "#subject")
    cached = _SIG_CACHE.get(key)
    if cached is not None:
        return "yes" in cached
    method = getattr(source, method_name, None)
    if method is None:
        return False
    try:
        params = [
            name
            for name, p in inspect.signature(method).parameters.items()
            if name != "self"
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
            and p.default is inspect.Parameter.empty
        ]
    except (TypeError, ValueError):  # pragma: no cover
        return True
    takes = bool(params)
    _SIG_CACHE[key] = {"yes"} if takes else {"no"}
    return takes


def _row_count(value: Any) -> Optional[int]:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return 1
    return None


def _by_symbol(rows: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).strip().upper().replace("EGX:", "")
        if sym:
            out.setdefault(sym, row)
    return out


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_none(value: Any) -> bool:
    return value is None


def _drop_none(values: dict[str, Any]) -> dict[str, float]:
    return {k: v for k, v in values.items() if v is not None}
