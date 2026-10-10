"""Data provenance — every number the system reports must answer for itself.

The premise of this module: an un-sourced price is not data, it is a rumour.
Downstream code (smart-money detection, screener scoring, trade plans) must be
able to ask *"where did this number come from, when was it fetched, what did
the data source disagree with, and what quality gates did it pass?"* without
re-deriving anything.

Nothing here is probabilistic. :class:`Sourced` carries only facts that were
observed: the source id, the fetch timestamp, the fetch latency, the number of
attempts, the validation verdicts, and the tolerance-tested agreement with
other sources.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


class QualityFlag(str, Enum):
    """Objective verdicts from the validation gates.

    A flag is *always* evidence-tagged: it says what was checked and what was
    found, never "we think this is probably fine".
    """

    VALID = "VALID"
    #: A hard rule was violated (negative price, high < low, …).
    INVALID = "INVALID"
    #: Structurally fine but the value is outside historical norms.
    OUTLIER = "OUTLIER"
    #: Fetched data is older than the calendar allows for its kind.
    STALE = "STALE"
    #: The row is a leftover from an earlier session (feed did not update).
    SUPERSEDED = "SUPERSEDED"
    #: Only part of the requested fields could be extracted.
    PARTIAL = "PARTIAL"
    #: Two sources agreed inside tolerance.
    AGREEMENT = "AGREEMENT"
    #: Two sources agreed but outside tolerance.
    DIVERGENCE = "DIVERGENCE"
    #: Two sources disagreed outright.
    CONFLICT = "CONFLICT"
    #: No second source could be consulted (not an error, but not proof).
    UNCROSSCHECKED = "UNCROSSCHECKED"


#: Flags that make a value unfit for downstream use without human review.
BLOCKING_FLAGS: frozenset[QualityFlag] = frozenset(
    {QualityFlag.INVALID, QualityFlag.STALE, QualityFlag.CONFLICT}
)


@dataclass(frozen=True)
class Provenance:
    """Immutable record of *how* one datum entered the system."""

    source: str
    fetched_at: datetime
    #: 0-based order in the fallback chain (0 = primary source).
    attempt_index: int = 0
    latency_ms: Optional[float] = None
    #: HTTP status of the winning request, when applicable.
    http_status: Optional[int] = None
    #: Stable identifier for the exact upstream resource (URL or endpoint id).
    endpoint: Optional[str] = None
    #: Free-form, must stay JSON-serialisable (row index, page, tag…).
    locator: dict[str, Any] = field(default_factory=dict)
    #: True when the value came from the local store rather than the network.
    from_cache: bool = False
    #: When ``from_cache``, the timestamp of the original network fetch.
    origin_fetched_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.fetched_at.tzinfo is None:
            raise ValueError("Provenance.fetched_at must be timezone-aware")
        if self.origin_fetched_at is not None and self.origin_fetched_at.tzinfo is None:
            raise ValueError("Provenance.origin_fetched_at must be timezone-aware")

    @property
    def effective_fetched_at(self) -> datetime:
        """Network-fetch time — the timestamp freshness must be judged against."""
        return self.origin_fetched_at or self.fetched_at

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.effective_fetched_at).total_seconds()

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["fetched_at"] = self.fetched_at.isoformat()
        out["origin_fetched_at"] = (
            self.origin_fetched_at.isoformat() if self.origin_fetched_at else None
        )
        out["age_seconds"] = round(self.age_seconds, 1)
        return out


@dataclass
class Sourced(Generic[T]):
    """A value plus the evidence that backs it.

    ``value`` is the payload; ``provenance`` is the receipt; ``flags`` are the
    verdicts; ``notes`` are human-readable details tied to those verdicts (e.g.
    ``"high < low on 2026-03-04"``).
    """

    value: T
    provenance: Provenance
    flags: list[QualityFlag] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Cross-source comparisons that were actually performed.
    cross_checks: list[dict[str, Any]] = field(default_factory=list)
    #: Source-specific side facts that belong with the value (e.g. Yahoo's
    #: chart ``meta`` block: currency, long name, 52-week range).
    meta: dict[str, Any] = field(default_factory=dict)

    # ── flag handling ───────────────────────────────────────────────────────

    def add_flag(self, flag: QualityFlag, note: Optional[str] = None) -> "Sourced[T]":
        if flag not in self.flags:
            self.flags.append(flag)
        if note:
            self.notes.append(note)
        return self

    def has(self, *flags: QualityFlag) -> bool:
        return any(f in self.flags for f in flags)

    @property
    def is_usable(self) -> bool:
        """False when a blocking flag is present — callers must not trade on it."""
        return not self.has(*BLOCKING_FLAGS)

    @property
    def is_cross_checked(self) -> bool:
        return bool(self.cross_checks)

    # ── serialisation ───────────────────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.as_dict(),
            "quality": {
                "flags": [f.value for f in self.flags],
                "usable": self.is_usable,
                "cross_checked": self.is_cross_checked,
                "notes": list(self.notes),
                "cross_checks": list(self.cross_checks),
            },
            "meta": dict(self.meta),
        }


def now_utc() -> datetime:
    """Single clock entry point so tests can freeze time in one place."""
    return datetime.now(timezone.utc)


def provenance_for(
    source: str,
    *,
    fetched_at: Optional[datetime] = None,
    attempt_index: int = 0,
    latency_ms: Optional[float] = None,
    http_status: Optional[int] = None,
    endpoint: Optional[str] = None,
    locator: Optional[dict[str, Any]] = None,
    from_cache: bool = False,
    origin_fetched_at: Optional[datetime] = None,
) -> Provenance:
    """Convenience constructor with an injectable clock (for tests)."""
    return Provenance(
        source=source,
        fetched_at=fetched_at or now_utc(),
        attempt_index=attempt_index,
        latency_ms=latency_ms,
        http_status=http_status,
        endpoint=endpoint,
        locator=dict(locator or {}),
        from_cache=from_cache,
        origin_fetched_at=origin_fetched_at,
    )
