"""Quality gates — the rules that decide whether a number may be trusted.

Two independent kinds of check live here, and they answer different questions:

**Structural validity** (:func:`validate_ohlcv_bars`, :func:`validate_quote`)
"Could this bar exist?" — high ≥ low, high ≥ max(open, close), low ≤
min(open, close), non-negative volume, no NaN. These are *hard* rules: a
violation means the parse or the upstream is broken, and the value is refused.

**Plausibility** (:func:`check_price_jump`, :func:`check_volume_anomaly`)
"Does this bar belong to this series?" — a 60% single-day move in a
large-cap EGX name is not impossible, it is *suspicious*, so it is flagged
(``OUTLIER``) and surfaced, never silently dropped. Refusing to flag outliers
is how a bad print from a scraped table silently becomes a signal.

**Agreement** (:func:`triangulate`) compares the same observation across
independent sources within a declared tolerance and reports AGREEMENT /
DIVERGENCE / CONFLICT. This is the mechanism that converts "one number from one
website" into "one number corroborated (or not) by N independent sources" —
the difference between an assertion and a probability.

No gate here ever returns a confidence score. Each returns a verdict plus the
evidence for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Optional, Sequence

import math

try:  # pandas is a hard dependency of the package, but the gates stay usable without it
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from .provenance import QualityFlag, Sourced

# ── thresholds (declared, overridable, documented) ──────────────────────────

#: A daily close-to-close move beyond this fraction is flagged OUTLIER.
#: EGX has no market-wide daily price limit, so this is a *review* threshold,
#: not a rejection rule. Large enough that normal EGX volatility (small caps
#: regularly move 10-20% in a session) does not trip it.
DEFAULT_JUMP_THRESHOLD = 0.35

#: Daily volume beyond this multiple of the trailing median is flagged.
DEFAULT_VOLUME_ANOMALY_MULTIPLE = 12.0

#: Triangulation tolerances: relative differences treated as the same number.
QUOTE_TOLERANCE = 0.01      # 1%  on price
VOLUME_TOLERANCE = 0.05     # 5%  on volume
OHLCV_TOLERANCE = 0.01      # 1%  on OHLC


@dataclass(frozen=True)
class Verdict:
    """Outcome of one gate: a flag, plus the evidence behind it."""

    flag: QualityFlag
    note: str
    detail: dict[str, Any]

    def __bool__(self) -> bool:
        return self.flag is QualityFlag.VALID


# ── structural gates ────────────────────────────────────────────────────────


def _finite(x: Any) -> bool:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def validate_quote(
    *,
    price: Optional[float],
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    volume: Optional[float] = None,
    previous_close: Optional[float] = None,
) -> list[Verdict]:
    """Structural checks on a single quote/print."""
    out: list[Verdict] = []

    if price is None or not _finite(price):
        out.append(Verdict(QualityFlag.INVALID, "price is missing or non-finite", {"price": price}))
    elif price <= 0:
        out.append(Verdict(QualityFlag.INVALID, f"price must be > 0, got {price}", {"price": price}))

    if bid is not None and _finite(bid) and bid < 0:
        out.append(Verdict(QualityFlag.INVALID, f"negative bid {bid}", {"bid": bid}))
    if ask is not None and _finite(ask) and ask < 0:
        out.append(Verdict(QualityFlag.INVALID, f"negative ask {ask}", {"ask": ask}))

    if (
        bid is not None
        and ask is not None
        and _finite(bid)
        and _finite(ask)
        and bid > 0
        and ask > 0
        and bid > ask
    ):
        # A crossed book is not always an error on thin EGX names (stale bid
        # vs fresh ask), so it is flagged as anomalous rather than impossible.
        out.append(
            Verdict(QualityFlag.OUTLIER, f"crossed book: bid {bid} > ask {ask}", {"bid": bid, "ask": ask})
        )

    if volume is not None and _finite(volume) and volume < 0:
        out.append(Verdict(QualityFlag.INVALID, f"negative volume {volume}", {"volume": volume}))

    if (
        price is not None
        and previous_close
        and _finite(price)
        and _finite(previous_close)
        and previous_close > 0
    ):
        move = abs(price / previous_close - 1.0)
        if move > DEFAULT_JUMP_THRESHOLD:
            out.append(
                Verdict(
                    QualityFlag.OUTLIER,
                    f"intraday move {move:.1%} vs previous close {previous_close}",
                    {"move": move, "previous_close": previous_close},
                )
            )

    if not out:
        out.append(Verdict(QualityFlag.VALID, "quote passed structural checks", {}))
    return out


def validate_ohlcv_bars(rows: Sequence[dict[str, Any]]) -> list[Verdict]:
    """Hard validity of OHLCV rows (works on dicts or a DataFrame's records).

    Returns one verdict per kind of problem found (capped in the note) so the
    caller can attach all of them to one :class:`Sourced`.
    """
    out: list[Verdict] = []
    bad_high_low: list[str] = []
    bad_high_body: list[str] = []
    bad_low_body: list[str] = []
    negative: list[str] = []
    nonfinite: list[str] = []
    zero_volume_days = 0

    for r in rows:
        tag = str(r.get("date") or r.get("timestamp") or "?")
        o, h, l, c = (r.get("open"), r.get("high"), r.get("low"), r.get("close"))
        v = r.get("volume")

        if not all(_finite(x) for x in (o, h, l, c)):
            nonfinite.append(tag)
            continue
        o, h, l, c = float(o), float(h), float(l), float(c)

        if min(o, h, l, c) < 0:
            negative.append(tag)
        if h < l:
            bad_high_low.append(tag)
        if h < max(o, c) - 1e-9:
            bad_high_body.append(tag)
        if l > min(o, c) + 1e-9:
            bad_low_body.append(tag)
        if v is not None and _finite(v) and float(v) == 0:
            zero_volume_days += 1

    def _note(label: str, tags: list[str]) -> None:
        if tags:
            sample = ", ".join(tags[:5]) + (" …" if len(tags) > 5 else "")
            out.append(
                Verdict(
                    QualityFlag.INVALID,
                    f"{label} in {len(tags)} bar(s): {sample}",
                    {"count": len(tags), "dates": tags[:20]},
                )
            )

    _note("non-finite OHLC", nonfinite)
    _note("negative price", negative)
    _note("high < low", bad_high_low)
    _note("high < max(open, close)", bad_high_body)
    _note("low > min(open, close)", bad_low_body)

    if zero_volume_days and rows:
        frac = zero_volume_days / len(rows)
        if frac > 0.5:
            out.append(
                Verdict(
                    QualityFlag.PARTIAL,
                    f"{zero_volume_days}/{len(rows)} bars report zero volume "
                    "(suspect: volume column not parsed, or a halted symbol)",
                    {"zero_volume_bars": zero_volume_days, "total": len(rows)},
                )
            )

    if not out:
        out.append(Verdict(QualityFlag.VALID, f"{len(rows)} bars passed structural checks", {}))
    return out


# ── plausibility gates ──────────────────────────────────────────────────────


def check_price_jump(
    closes: Sequence[float],
    *,
    threshold: float = DEFAULT_JUMP_THRESHOLD,
) -> list[Verdict]:
    """Flag single-period close-to-close moves beyond *threshold*."""
    out: list[Verdict] = []
    for i in range(1, len(closes)):
        prev, cur = closes[i - 1], closes[i]
        if not (_finite(prev) and _finite(cur)) or float(prev) <= 0:
            continue
        move = float(cur) / float(prev) - 1.0
        if abs(move) > threshold:
            out.append(
                Verdict(
                    QualityFlag.OUTLIER,
                    f"close-to-close move {move:+.1%} at index {i} "
                    f"({prev} → {cur})",
                    {"index": i, "move": move, "from": float(prev), "to": float(cur)},
                )
            )
    if not out:
        out.append(Verdict(QualityFlag.VALID, "no close-to-close jump above threshold", {}))
    return out


def check_volume_anomaly(
    volumes: Sequence[float],
    *,
    multiple: float = DEFAULT_VOLUME_ANOMALY_MULTIPLE,
    window: int = 20,
) -> list[Verdict]:
    """Flag volume spikes vs the trailing median (median resists prior spikes)."""
    out: list[Verdict] = []
    medians: list[Optional[float]] = [None]
    for i in range(1, len(volumes)):
        trailing = [float(v) for v in volumes[max(0, i - window):i] if _finite(v)]
        medians.append(_median(trailing) if trailing else None)

    for i, (v, med) in enumerate(zip(volumes, medians)):
        if med is None or not _finite(v) or med <= 0:
            continue
        ratio = float(v) / med
        if ratio > multiple:
            out.append(
                Verdict(
                    QualityFlag.OUTLIER,
                    f"volume {float(v):,.0f} is {ratio:.1f}× the trailing "
                    f"{window}-period median",
                    {"index": i, "ratio": ratio, "volume": float(v), "median": med},
                )
            )
    if not out:
        out.append(Verdict(QualityFlag.VALID, "no volume spike above threshold", {}))
    return out


def _median(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


# ── cross-source triangulation ──────────────────────────────────────────────


@dataclass
class SourceObservation:
    """One source's version of the same fact (same symbol, same field, same date)."""

    source: str
    values: dict[str, float]
    fetched_at: datetime


def triangulate(
    observations: Sequence[SourceObservation],
    *,
    fields: Optional[Iterable[str]] = None,
    tolerance: Optional[dict[str, float]] = None,
) -> tuple[QualityFlag, list[dict[str, Any]], str]:
    """Compare concurrent observations of the same fact.

    Returns ``(flag, cross_checks, note)`` where ``cross_checks`` is a
    JSON-serialisable list of per-comparison evidence rows.

    * 0 or 1 observation  → ``UNCROSSCHECKED`` (fact, not failure)
    * all inside tolerance → ``AGREEMENT``; value = median of observations
    * some outside        → ``DIVERGENCE``
    * spread > 3× tolerance → ``CONFLICT``
    """
    tol = {**{"close": OHLCV_TOLERANCE, "price": QUOTE_TOLERANCE, "volume": VOLUME_TOLERANCE}, **(tolerance or {})}

    if len(observations) < 2:
        return (
            QualityFlag.UNCROSSCHECKED,
            [
                {
                    "source": o.source,
                    "checked_against": [],
                    "reason": "only one source available",
                }
                for o in observations
            ],
            "single source — no independent corroboration available",
        )

    baseline = observations[0]
    keys = list(fields) if fields else sorted(baseline.values.keys())
    checks: list[dict[str, Any]] = []
    worst_flag = QualityFlag.AGREEMENT
    max_rel_spread = 0.0

    for field in keys:
        vals = [o.values.get(field) for o in observations]
        pairs = [
            (o.source, v)
            for o, v in zip(observations, vals)
            if v is not None and _finite(v)
        ]
        if len(pairs) < 2:
            continue
        numbers = [float(v) for _, v in pairs]
        reference = _median(numbers)
        field_tol = tol.get(field, QUOTE_TOLERANCE)

        spreads = [
            abs(n - numbers[0]) / abs(numbers[0]) if numbers[0] else 0.0 for n in numbers[1:]
        ]
        rel_spread = max(spreads) if spreads else 0.0
        max_rel_spread = max(max_rel_spread, rel_spread)

        within = rel_spread <= field_tol
        if rel_spread > field_tol * 3:
            verdict = "CONFLICT"
            worst_flag = QualityFlag.CONFLICT
        elif not within:
            verdict = "DIVERGENCE"
            if worst_flag is QualityFlag.AGREEMENT:
                worst_flag = QualityFlag.DIVERGENCE
        else:
            verdict = "AGREEMENT"

        checks.append(
            {
                "field": field,
                "values": {src: v for src, v in pairs},
                "median": reference,
                "relative_spread": round(rel_spread, 6),
                "tolerance": field_tol,
                "verdict": verdict,
            }
        )

    if not checks:
        return (
            QualityFlag.UNCROSSCHECKED,
            [],
            "sources report different fields — nothing comparable to cross-check",
        )

    note = (
        f"{len(observations)} sources compared on {len(checks)} field(s); "
        f"max relative spread {max_rel_spread:.2%}"
    )
    return worst_flag, checks, note


def attach_verdicts(target: "Sourced[Any]", verdicts: Sequence[Verdict]) -> "Sourced[Any]":
    """Apply gate output to a :class:`Sourced` wrapper (skipping VALID markers)."""
    for v in verdicts:
        if v.flag is QualityFlag.VALID:
            continue
        target.add_flag(v.flag, v.note)
    return target


# ── freshness ───────────────────────────────────────────────────────────────


def check_freshness(
    target: "Sourced[Any]",
    *,
    kind: str,
    calendar=None,
    now: Optional[datetime] = None,
) -> "Sourced[Any]":
    """Apply the EGX calendar's staleness contract to a wrapped value.

    ``kind`` is ``"quote"`` (intraday freshness) or ``"daily_bar"`` (the value
    must reach the newest settled session).
    """
    cal = calendar
    if cal is None:
        from .calendar import EGXCalendar

        cal = EGXCalendar.from_env()

    prov = target.provenance
    if kind == "quote":
        if cal.is_stale(prov.effective_fetched_at, now):
            allowed = cal.max_quote_age_s(now)
            target.add_flag(
                QualityFlag.STALE,
                f"quote age {prov.age_seconds:.0f}s exceeds the {allowed}s allowed "
                f"by the market clock ({cal.state(now).phase})",
            )
        else:
            target.add_flag(QualityFlag.VALID)
        return target

    if kind == "daily_bar":
        last = _last_bar_date(target.value)
        if last is None:
            target.add_flag(
                QualityFlag.PARTIAL,
                "could not read a bar date from the payload — freshness un verified",
            )
            return target
        if not cal.data_is_current(last, now):
            target.add_flag(
                QualityFlag.STALE,
                f"newest settled bar is {last.isoformat()} but the market clock "
                f"expects {_expected_bar(cal, now).isoformat()}",
            )
        else:
            target.add_flag(QualityFlag.VALID)
        return target

    raise ValueError(f"unknown freshness kind: {kind!r}")


def _expected_bar(cal, now: Optional[datetime]) -> date:
    ref = cal.to_local(now) if now is not None else cal.now()
    today = ref.date()
    if not cal.is_trading_day(today):
        return cal.previous_trading_day(today)
    _, close_dt = cal._continuous_bounds(today)  # noqa: SLF001 - same package
    from datetime import timedelta as _td

    from .calendar import POST_CLOSE_GRACE_S

    if ref >= close_dt + _td(seconds=POST_CLOSE_GRACE_S):
        return today
    return cal.previous_trading_day(today)


def _last_bar_date(value: Any) -> Optional[date]:
    """Best-effort extraction of the newest bar date from a payload."""
    if value is None:
        return None
    if isinstance(value, list) and value:
        return _last_bar_date(value[-1])
    if isinstance(value, dict):
        for key in ("date", "timestamp", "time", "datetime"):
            if key in value:
                return _coerce_date(value[key])
    if pd is not None and isinstance(value, pd.DataFrame) and not value.empty:
        idx = value.index[-1]
        return _coerce_date(idx)
    return None


def _coerce_date(raw: Any) -> Optional[date]:
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(raw)).date()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(raw, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(raw[: len(fmt) + 2].strip(), fmt).date()
            except ValueError:
                continue
    return None
