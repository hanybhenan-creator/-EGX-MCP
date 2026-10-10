"""Versioned local store for EGX data — values, provenance and audit trail.

The sibling module ``egx_data_cache.py`` is a key/value cache: it answers
"what do I have right now?" and nothing else. It cannot answer any of the
questions this layer exists for:

* *Where did this bar come from, and when?* → ``source``, ``fetched_at``.
* *Did another source disagree?* → ``ohlcv_observations`` keeps **every**
  source's version of a bar, not just the winner.
* *Why is today's data missing?* → ``fetch_audit`` records the failed attempts
  with their error kind, and ``ingest_runs`` records what a refresh intended
  to do.
* *Is the series complete?* → :meth:`DataStore.ohlcv_gaps` compares stored
  dates against the EGX trading calendar instead of against "yesterday".

Schema changes are applied through :data:`MIGRATIONS`, keyed by version, so a
deployed database upgrades in place instead of being silently re-created with
``CREATE TABLE IF NOT EXISTS`` (which is what happens when a schema evolves
without versioning: new columns never appear, and inserts start failing on
live boxes while tests on fresh databases stay green).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from .provenance import QualityFlag, Sourced

SCHEMA_VERSION = 1

#: Ordered migrations. ``MIGRATIONS[1]`` takes an empty database to version 1.
MIGRATIONS: dict[int, list[str]] = {
    1: [
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS symbols (
            exchange     TEXT NOT NULL,
            symbol       TEXT NOT NULL,
            name         TEXT,
            sector       TEXT,
            is_active    INTEGER NOT NULL DEFAULT 1,
            first_seen   TEXT NOT NULL,
            last_seen    TEXT NOT NULL,
            last_source  TEXT,
            PRIMARY KEY (exchange, symbol)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ohlcv_daily (
            exchange      TEXT NOT NULL,
            symbol        TEXT NOT NULL,
            date          TEXT NOT NULL,
            open          REAL,
            high          REAL,
            low           REAL,
            close         REAL,
            volume        INTEGER,
            adj_close     REAL,
            source        TEXT NOT NULL,
            fetched_at    TEXT NOT NULL,
            quality_flags TEXT NOT NULL DEFAULT '',
            is_stale      INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (exchange, symbol, date)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_date ON ohlcv_daily (exchange, symbol, date DESC)",
        """
        CREATE TABLE IF NOT EXISTS ohlcv_observations (
            exchange      TEXT NOT NULL,
            symbol        TEXT NOT NULL,
            date          TEXT NOT NULL,
            source        TEXT NOT NULL,
            open          REAL,
            high          REAL,
            low           REAL,
            close         REAL,
            volume        INTEGER,
            fetched_at    TEXT NOT NULL,
            quality_flags TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (exchange, symbol, date, source)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS quotes (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange       TEXT NOT NULL,
            symbol         TEXT NOT NULL,
            observed_at    TEXT NOT NULL,
            fetched_at     TEXT NOT NULL,
            price          REAL,
            previous_close REAL,
            change_pct     REAL,
            day_high       REAL,
            day_low        REAL,
            volume         INTEGER,
            bid            REAL,
            ask            REAL,
            source         TEXT NOT NULL,
            quality_flags  TEXT NOT NULL DEFAULT '',
            is_stale       INTEGER NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_quotes_symbol_time ON quotes (exchange, symbol, observed_at DESC)",
        """
        CREATE TABLE IF NOT EXISTS index_levels (
            index_name   TEXT NOT NULL,
            date         TEXT NOT NULL,
            close        REAL,
            change_pct   REAL,
            source       TEXT NOT NULL,
            fetched_at   TEXT NOT NULL,
            quality_flags TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (index_name, date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS fetch_audit (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            source      TEXT NOT NULL,
            capability  TEXT NOT NULL,
            subject     TEXT,
            ok          INTEGER NOT NULL,
            latency_ms  REAL,
            rows        INTEGER,
            http_status INTEGER,
            error       TEXT,
            error_kind  TEXT,
            flags       TEXT NOT NULL DEFAULT '',
            endpoint    TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_audit_ts ON fetch_audit (ts DESC)",
        "CREATE INDEX IF NOT EXISTS idx_audit_source ON fetch_audit (source, ts DESC)",
        """
        CREATE TABLE IF NOT EXISTS ingest_runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at   TEXT NOT NULL,
            finished_at  TEXT,
            kind         TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'running',
            params       TEXT NOT NULL DEFAULT '{}',
            rows_written INTEGER NOT NULL DEFAULT 0,
            sources_used TEXT NOT NULL DEFAULT '',
            warnings     TEXT NOT NULL DEFAULT '',
            error        TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS quality_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         TEXT NOT NULL,
            exchange   TEXT,
            symbol     TEXT,
            source     TEXT,
            flag       TEXT NOT NULL,
            note       TEXT,
            payload    TEXT
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_quality_symbol ON quality_events (symbol, ts DESC)",
    ]
}


@dataclass
class RunHandle:
    """A row in ``ingest_runs`` — live handle so a crash leaves evidence."""

    run_id: int
    kind: str
    started_at: datetime


@dataclass
class StoreStats:
    symbols: int
    ohlcv_rows: int
    quote_rows: int
    audit_rows: int
    quality_events: int
    last_ohlcv_date: Optional[str]
    last_fetch_at: Optional[str]
    schema_version: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbols": self.symbols,
            "ohlcv_rows": self.ohlcv_rows,
            "quote_rows": self.quote_rows,
            "audit_rows": self.audit_rows,
            "quality_events": self.quality_events,
            "last_ohlcv_date": self.last_ohlcv_date,
            "last_fetch_at": self.last_fetch_at,
            "schema_version": self.schema_version,
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: Optional[datetime]) -> Optional[str]:
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat()


class DataStore:
    """Thread-safe, WAL-mode SQLite store with schema migrations."""

    def __init__(self, db_path: str | Path = "egx_data.db", *, timeout_s: float = 30.0) -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._timeout = timeout_s
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=timeout_s,
            isolation_level=None,  # explicit transactions
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute(f"PRAGMA busy_timeout={int(timeout_s * 1000)}")
        self.migrate()

    # ── lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self) -> "DataStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN")
                yield cur
                cur.execute("COMMIT")
            except Exception:
                try:
                    cur.execute("ROLLBACK")
                except Exception:  # pragma: no cover
                    pass
                raise
            finally:
                cur.close()

    # ── migrations ──────────────────────────────────────────────────────────

    def schema_version(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_meta'"
            )
            if cur.fetchone() is None:
                return 0
            row = self._conn.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()
        return int(row["value"]) if row else 0

    def migrate(self, *, target: Optional[int] = None) -> int:
        """Apply pending migrations. Idempotent; returns the resulting version."""
        target = target or SCHEMA_VERSION
        current = self.schema_version()
        for version in range(current + 1, target + 1):
            statements = MIGRATIONS.get(version)
            if statements is None:
                raise RuntimeError(f"no migration defined for schema version {version}")
            with self._tx() as cur:
                for stmt in statements:
                    cur.execute(stmt)
                cur.execute(
                    "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(version),),
                )
                cur.execute(
                    "INSERT INTO schema_meta (key, value) VALUES ('migrated_at', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (_iso(_utc_now()),),
                )
        return self.schema_version()

    # ── writes ──────────────────────────────────────────────────────────────

    def upsert_symbol(
        self,
        symbol: str,
        *,
        exchange: str = "EGX",
        name: Optional[str] = None,
        sector: Optional[str] = None,
        is_active: bool = True,
        source: Optional[str] = None,
    ) -> None:
        now = _iso(_utc_now())
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO symbols (exchange, symbol, name, sector, is_active,
                                     first_seen, last_seen, last_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(exchange, symbol) DO UPDATE SET
                    name        = COALESCE(excluded.name, symbols.name),
                    sector      = COALESCE(excluded.sector, symbols.sector),
                    is_active   = excluded.is_active,
                    last_seen   = excluded.last_seen,
                    last_source = COALESCE(excluded.last_source, symbols.last_source)
                """,
                (exchange, symbol.upper(), name, sector, 1 if is_active else 0, now, now, source),
            )

    def upsert_symbols(self, rows: Iterable[dict[str, Any]], *, exchange: str = "EGX", source: Optional[str] = None) -> int:
        count = 0
        with self._tx() as cur:
            now = _iso(_utc_now())
            for row in rows:
                symbol = str(row.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                cur.execute(
                    """
                    INSERT INTO symbols (exchange, symbol, name, sector, is_active,
                                         first_seen, last_seen, last_source)
                    VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                    ON CONFLICT(exchange, symbol) DO UPDATE SET
                        name        = COALESCE(excluded.name, symbols.name),
                        sector      = COALESCE(excluded.sector, symbols.sector),
                        last_seen   = excluded.last_seen,
                        last_source = COALESCE(excluded.last_source, symbols.last_source)
                    """,
                    (
                        exchange,
                        symbol,
                        row.get("name"),
                        row.get("sector"),
                        now,
                        now,
                        source,
                    ),
                )
                count += 1
        return count

    def write_ohlcv(
        self,
        symbol: str,
        bars: Sequence[dict[str, Any]],
        *,
        exchange: str = "EGX",
        source: str,
        fetched_at: Optional[datetime] = None,
        quality_flags: Sequence[QualityFlag] = (),
    ) -> int:
        """Upsert daily bars and record this source's observation of each.

        ``ohlcv_daily`` keeps the accepted version; ``ohlcv_observations``
        keeps *this* source's version regardless, so a later disagreement can
        be proven rather than argued about.
        """
        stamps = ",".join(f.value for f in quality_flags)
        when = _iso(fetched_at or _utc_now())
        written = 0
        with self._tx() as cur:
            for bar in bars:
                day = _bar_date(bar)
                if not day:
                    continue
                values = (
                    exchange,
                    symbol.upper(),
                    day,
                    _f(bar.get("open")),
                    _f(bar.get("high")),
                    _f(bar.get("low")),
                    _f(bar.get("close")),
                    _i(bar.get("volume")),
                    _f(bar.get("adj_close")),
                )
                cur.execute(
                    """
                    INSERT INTO ohlcv_daily
                        (exchange, symbol, date, open, high, low, close, volume,
                         adj_close, source, fetched_at, quality_flags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(exchange, symbol, date) DO UPDATE SET
                        open=excluded.open, high=excluded.high, low=excluded.low,
                        close=excluded.close, volume=excluded.volume,
                        adj_close=excluded.adj_close, source=excluded.source,
                        fetched_at=excluded.fetched_at, quality_flags=excluded.quality_flags
                    """,
                    (*values, source, when, stamps),
                )
                cur.execute(
                    """
                    INSERT INTO ohlcv_observations
                        (exchange, symbol, date, source, open, high, low, close,
                         volume, fetched_at, quality_flags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(exchange, symbol, date, source) DO UPDATE SET
                        open=excluded.open, high=excluded.high, low=excluded.low,
                        close=excluded.close, volume=excluded.volume,
                        fetched_at=excluded.fetched_at, quality_flags=excluded.quality_flags
                    """,
                    (
                        exchange,
                        symbol.upper(),
                        day,
                        source,
                        _f(bar.get("open")),
                        _f(bar.get("high")),
                        _f(bar.get("low")),
                        _f(bar.get("close")),
                        _i(bar.get("volume")),
                        when,
                        stamps,
                    ),
                )
                written += 1
        return written

    def write_quote(
        self,
        quote: dict[str, Any],
        *,
        exchange: str = "EGX",
        source: str,
        fetched_at: Optional[datetime] = None,
        quality_flags: Sequence[QualityFlag] = (),
        is_stale: bool = False,
    ) -> int:
        symbol = str(quote.get("symbol") or "").strip().upper()
        if not symbol:
            return 0
        stamps = ",".join(f.value for f in quality_flags)
        when = fetched_at or _utc_now()
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO quotes
                    (exchange, symbol, observed_at, fetched_at, price, previous_close,
                     change_pct, day_high, day_low, volume, bid, ask, source,
                     quality_flags, is_stale)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exchange,
                    symbol,
                    quote.get("source_timestamp") or _iso(when),
                    _iso(when),
                    _f(quote.get("price") or quote.get("close")),
                    _f(quote.get("previous_close")),
                    _f(quote.get("change_percent") or quote.get("change_pct")),
                    _f(quote.get("day_high")),
                    _f(quote.get("day_low")),
                    _i(quote.get("volume")),
                    _f(quote.get("bid")),
                    _f(quote.get("ask")),
                    source,
                    stamps,
                    1 if is_stale else 0,
                ),
            )
        return 1

    def write_index(
        self,
        index_name: str,
        day: str,
        close: Optional[float],
        *,
        change_pct: Optional[float] = None,
        source: str,
        fetched_at: Optional[datetime] = None,
        quality_flags: Sequence[QualityFlag] = (),
    ) -> int:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO index_levels
                    (index_name, date, close, change_pct, source, fetched_at, quality_flags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(index_name, date) DO UPDATE SET
                    close=excluded.close, change_pct=excluded.change_pct,
                    source=excluded.source, fetched_at=excluded.fetched_at,
                    quality_flags=excluded.quality_flags
                """,
                (
                    index_name.upper(),
                    day,
                    close,
                    change_pct,
                    source,
                    _iso(fetched_at or _utc_now()),
                    ",".join(f.value for f in quality_flags),
                ),
            )
        return 1

    # ── audit ───────────────────────────────────────────────────────────────

    def record_attempt(
        self,
        *,
        source: str,
        capability: str,
        subject: Optional[str],
        ok: bool,
        latency_ms: Optional[float] = None,
        rows: Optional[int] = None,
        http_status: Optional[int] = None,
        error: Optional[str] = None,
        error_kind: Optional[str] = None,
        flags: Sequence[str] = (),
        endpoint: Optional[str] = None,
    ) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO fetch_audit
                    (ts, source, capability, subject, ok, latency_ms, rows,
                     http_status, error, error_kind, flags, endpoint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _iso(_utc_now()),
                    source,
                    capability,
                    subject,
                   1 if ok else 0,
                    latency_ms,
                    rows,
                    http_status,
                    error,
                    error_kind,
                    ",".join(flags),
                    endpoint,
                ),
            )

    def record_quality_event(
        self,
        flag: QualityFlag | str,
        *,
        note: Optional[str] = None,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
        source: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        value = flag.value if isinstance(flag, QualityFlag) else str(flag)
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO quality_events (ts, exchange, symbol, source, flag, note, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _iso(_utc_now()),
                    exchange,
                    symbol,
                    source,
                    value,
                    note,
                    json.dumps(payload) if payload else None,
                ),
            )

    def record_flags(
        self,
        sourced: Sourced[Any],
        *,
        exchange: str = "EGX",
        symbol: Optional[str] = None,
        skip_valid: bool = True,
    ) -> int:
        count = 0
        for flag in sourced.flags:
            if skip_valid and flag is QualityFlag.VALID:
                continue
            note = next(
                (n for n in sourced.notes if flag.value.lower() in n.lower()),
                "; ".join(sourced.notes[:1]),
            )
            self.record_quality_event(
                flag,
                note=note or None,
                exchange=exchange,
                symbol=symbol,
                source=sourced.provenance.source,
                payload=dict(sourced.provenance.locator),
            )
            count += 1
        return count

    # ── runs ────────────────────────────────────────────────────────────────

    def begin_run(self, kind: str, *, params: Optional[dict[str, Any]] = None) -> RunHandle:
        started = _utc_now()
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO ingest_runs (started_at, kind, status, params) VALUES (?, ?, 'running', ?)",
                (_iso(started), kind, json.dumps(params or {})),
            )
            run_id = int(cur.lastrowid or 0)
        return RunHandle(run_id=run_id, kind=kind, started_at=started)

    def finish_run(
        self,
        handle: RunHandle,
        *,
        status: str = "ok",
        rows_written: int = 0,
        sources_used: Sequence[str] = (),
        warnings: Sequence[str] = (),
        error: Optional[str] = None,
    ) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                UPDATE ingest_runs
                   SET finished_at=?, status=?, rows_written=?, sources_used=?,
                       warnings=?, error=?
                 WHERE id=?
                """,
                (
                    _iso(_utc_now()),
                    status,
                    rows_written,
                    ",".join(sources_used),
                    json.dumps(list(warnings)),
                    error,
                    handle.run_id,
                ),
            )

    def get_run(self, run_id: int) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM ingest_runs WHERE id=?", (int(run_id),)
            ).fetchone()
        return dict(row) if row else None

    def recent_runs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM ingest_runs ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── reads ───────────────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        *,
        exchange: str = "EGX",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM ohlcv_daily WHERE exchange=? AND symbol=?"
        params: list[Any] = [exchange, symbol.upper()]
        if start:
            sql += " AND date>=?"
            params.append(start)
        if end:
            sql += " AND date<=?"
            params.append(end)
        sql += " ORDER BY date"
        if limit:
            sql = f"SELECT * FROM ({sql} DESC LIMIT {int(limit)}) ORDER BY date"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_symbols(
        self, *, exchange: str = "EGX", active_only: bool = True
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM symbols WHERE exchange=?"
        params: list[Any] = [exchange]
        if active_only:
            sql += " AND is_active=1"
        sql += " ORDER BY symbol"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_index_history(
        self,
        index_name: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM index_levels WHERE index_name=?"
        params: list[Any] = [index_name.upper()]
        if start:
            sql += " AND date>=?"
            params.append(start)
        if end:
            sql += " AND date<=?"
            params.append(end)
        sql += " ORDER BY date"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def latest_ohlcv_date(self, symbol: str, *, exchange: str = "EGX") -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(date) AS d FROM ohlcv_daily WHERE exchange=? AND symbol=?",
                (exchange, symbol.upper()),
            ).fetchone()
        return row["d"] if row and row["d"] else None

    def ohlcv_gaps(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        exchange: str = "EGX",
        calendar=None,
    ) -> list[str]:
        """Trading days in ``[start, end]`` with no stored bar.

        This is a calendar question, not a "yesterday" question: a Friday in
        the output is not a bug, and a Thursday is not automatically expected
        (it may be a holiday).
        """
        if calendar is None:
            from .calendar import EGXCalendar

            calendar = EGXCalendar.from_env()
        expected = {d.isoformat() for d in calendar.trading_days_between(start, end)}
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT date FROM ohlcv_daily
                 WHERE exchange=? AND symbol=? AND date BETWEEN ? AND ?
                """,
                (exchange, symbol.upper(), start.isoformat(), end.isoformat()),
            ).fetchall()
        have = {r["date"] for r in rows}
        return sorted(expected - have)

    def observations_for(
        self, symbol: str, day: str, *, exchange: str = "EGX"
    ) -> list[dict[str, Any]]:
        """Every source's version of one bar — the raw material of triangulation."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM ohlcv_observations
                 WHERE exchange=? AND symbol=? AND date=?
                 ORDER BY source
                """,
                (exchange, symbol.upper(), day),
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_quote(self, symbol: str, *, exchange: str = "EGX") -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM quotes WHERE exchange=? AND symbol=?
                 ORDER BY fetched_at DESC, id DESC LIMIT 1
                """,
                (exchange, symbol.upper()),
            ).fetchone()
        return dict(row) if row else None

    def recent_audit(self, *, limit: int = 50, failures_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM fetch_audit"
        if failures_only:
            sql += " WHERE ok=0"
        sql += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def source_reliability(self, *, since: Optional[str] = None) -> list[dict[str, Any]]:
        """Per-source success rate from the audit trail.

        This is where "which source should be primary?" stops being an opinion:
        the numbers come from what actually happened on this box.
        """
        sql = """
            SELECT source,
                   COUNT(*)                      AS attempts,
                   SUM(ok)                       AS successes,
                   ROUND(AVG(latency_ms), 1)     AS avg_latency_ms,
                   MAX(ts)                       AS last_attempt
              FROM fetch_audit
        """
        params: list[Any] = []
        if since:
            sql += " WHERE ts >= ?"
            params.append(since)
        sql += " GROUP BY source ORDER BY successes DESC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            attempts = r["attempts"] or 0
            successes = r["successes"] or 0
            out.append(
                {
                    "source": r["source"],
                    "attempts": attempts,
                    "successes": successes,
                    "success_rate": round(successes / attempts, 4) if attempts else None,
                    "avg_latency_ms": r["avg_latency_ms"],
                    "last_attempt": r["last_attempt"],
                }
            )
        return out

    def stats(self) -> StoreStats:
        with self._lock:
            def one(sql: str) -> Any:
                row = self._conn.execute(sql).fetchone()
                return row[0] if row else None

            return StoreStats(
                symbols=one("SELECT COUNT(*) FROM symbols") or 0,
                ohlcv_rows=one("SELECT COUNT(*) FROM ohlcv_daily") or 0,
                quote_rows=one("SELECT COUNT(*) FROM quotes") or 0,
                audit_rows=one("SELECT COUNT(*) FROM fetch_audit") or 0,
                quality_events=one("SELECT COUNT(*) FROM quality_events") or 0,
                last_ohlcv_date=one("SELECT MAX(date) FROM ohlcv_daily"),
                last_fetch_at=one("SELECT MAX(fetched_at) FROM ohlcv_daily"),
                schema_version=self.schema_version(),
            )

    # ── maintenance ─────────────────────────────────────────────────────────

    def prune_quotes(self, *, keep_days: int = 90) -> int:
        from datetime import timedelta

        cutoff = (_utc_now() - timedelta(days=keep_days)).isoformat()
        with self._tx() as cur:
            cur.execute("DELETE FROM quotes WHERE fetched_at < ?", (cutoff,))
            deleted = cur.rowcount or 0
        return int(deleted)

    def prune_audit(self, *, keep_days: int = 180) -> int:
        from datetime import timedelta

        cutoff = (_utc_now() - timedelta(days=keep_days)).isoformat()
        with self._tx() as cur:
            cur.execute("DELETE FROM fetch_audit WHERE ts < ?", (cutoff,))
            deleted = cur.rowcount or 0
        return int(deleted)

    def vacuum(self) -> None:
        with self._lock:
            self._conn.execute("VACUUM")


# ── coercion helpers ────────────────────────────────────────────────────────


def _bar_date(bar: dict[str, Any]) -> Optional[str]:
    raw = bar.get("date") or bar.get("datetime") or bar.get("timestamp")
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc).date().isoformat()
    text = str(raw)
    return text[:10] if len(text) >= 10 else None


def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _i(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
