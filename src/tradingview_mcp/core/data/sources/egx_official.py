"""egx.com.eg — the exchange's own feed.

This is the only source in the chain that is *authoritative* about the
Egyptian session: its prints, its symbol universe, its sector aggregates. It is
also the only source with **no stable contract**. There is no published API,
the page layout changes without notice, and an outage is invisible until a
parser returns zero rows.

So this adapter is built defensively in three ways:

1. **Capability honesty.** It declares only what a page-scrape can actually
   deliver (snapshot, quotes, indices, sector stats, symbol master). It does
   *not* claim ``DAILY_OHLCV``: EGX's historical page is an ASP.NET
   postback grid, and pretending otherwise would let the router pick a
   source that cannot deliver. History comes from Yahoo; the router knows
   this because the capability sets are disjoint.

2. **Endpoint candidates, learned at runtime.** Each capability maps to an
   ordered list of candidate paths (English page, Arabic page, alternate
   names). The first one that *parses into rows* wins, and the winner is
   sticky for the process lifetime. When EGX renames a page, the adapter
   moves to the next candidate instead of failing the whole refresh.

3. **Multi-strategy parsing with a report.** HTML tables first, then JSON
   embedded in ``<script>`` blocks, then any table on the page whose header
   row maps to enough known columns. Every parse returns a
   :class:`ParseReport` describing which strategy matched, how many rows came
   out, and which canonical fields were missing — that report is the
   difference between "0 rows because EGX is closed" and "0 rows because our
   selector broke", and it lands in the audit table either way.

Nothing here assumes a specific CSS class or table id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

from ..html_tables import HTable, extract_json_blobs, extract_tables
from ..parsing import (
    clean_text,
    looks_like_symbol,
    normalize_header,
    parse_number,
    parse_percent,
    to_ascii_digits,
)
from ..provenance import QualityFlag, Sourced
from ..transport import Transport, TransportError, TransportResponse
from .base import (
    Capability,
    MarketDataSource,
    SourceDataError,
    SourceDescriptor,
    SourceParseError,
    SourceUnavailable,
)

BASE = "https://www.egx.com.eg"

#: Candidate page paths per capability, best guess first. Kept as data (and
#: overridable from the environment) because EGX renames pages and an operator
#: should be able to fix a live outage without waiting for a release.
ENDPOINT_CANDIDATES: dict[Capability, list[str]] = {
    Capability.MARKET_SNAPSHOT: [
        "/en/StocksData.aspx",
        "/en/MarketWatch.aspx",
        "/en/HomePage.aspx",
        "/ar/StocksData.aspx",
    ],
    Capability.QUOTE: [
        "/en/StocksData.aspx",
        "/en/MarketWatch.aspx",
    ],
    Capability.INDEX_QUOTE: [
        "/en/IndicesData.aspx",
        "/en/HomePage.aspx",
        "/en/MarketWatch.aspx",
    ],
    Capability.SECTOR_STATS: [
        "/en/SectorsData.aspx",
        "/en/HomePage.aspx",
    ],
    Capability.SYMBOL_MASTER: [
        "/en/ListedStocks.aspx",
        "/en/StocksData.aspx",
    ],
}

#: Column synonyms (normalised header → canonical field). Arabic entries are
#: already orthographically folded by ``parsing.normalize_header``, so
#: ``أعلى`` / ``الاعلى`` both land on ``اعلى``.
COLUMN_SYNONYMS: dict[str, str] = {
    # identity
    "symbol": "symbol", "code": "symbol", "ticker": "symbol", "stock": "symbol",
    "الرمز": "symbol", "رمز": "symbol", "كود": "symbol", "الكود": "symbol",
    "اسم_الرمز": "symbol", "symbol_2": "symbol",
    "name": "name", "company": "name", "company_name": "name", "stock_name": "name",
    "الاسم": "name", "الشركه": "name", "اسم_الشركه": "name", "اسم": "name",
    # prices
    "price": "close", "last": "close", "last_price": "close", "close": "close",
    "closing_price": "close", "current_price": "close", "market_price": "close",
    "السعر": "close", "اخر_سعر": "close", "سعر_الاغلاق": "close", "الاغلاق": "close",
    "open": "open", "opening": "open", "opening_price": "open", "open_price": "open",
    "الافتتاح": "open", "سعر_الافتتاح": "open",
    "high": "high", "day_high": "high", "highest": "high",
    "اعلى": "high", "الاعلى": "high", "اعلى_سعر": "high",
    "low": "low", "day_low": "low", "lowest": "low",
    "ادنى": "low", "الادنى": "low", "ادنى_سعر": "low",
    "previous_close": "previous_close", "prev_close": "previous_close",
    "previous": "previous_close", "prev": "previous_close", "last_close": "previous_close",
    "الاغلاق_السابق": "previous_close", "اغلاق_سابق": "previous_close", "السابق": "previous_close",
    # activity
    "change": "change", "net_change": "change", "price_change": "change", "chg": "change",
    "التغيير": "change", "تغير": "change", "صافي_التغيير": "change",
    "change_pct": "change_pct", "percentage_change": "change_pct", "pct_change": "change_pct",
    "change_pct_": "change_pct", "chg_pct": "change_pct", "net_change_pct": "change_pct",
    "نسبه_التغيير": "change_pct", "التغيير_pct": "change_pct", "النسبه_المئويه": "change_pct",
    "volume": "volume", "traded_volume": "volume", "quantity": "volume", "qty": "volume",
    "الكميه": "volume", "كميه": "volume", "حجم": "volume", "حجم_التداول": "volume",
    "الكميه_المتداوله": "volume",
    "value": "value_traded", "value_traded": "value_traded", "turnover": "value_traded",
    "traded_value": "value_traded",
    "القيمه": "value_traded", "قيمه_التداول": "value_traded", "القيمه_المتداوله": "value_traded",
    "trades": "trades", "no_of_trades": "trades", "deals": "trades", "transactions": "trades",
    "عدد_العمليات": "trades", "العمليات": "trades", "التعاملات": "trades",
    "market_cap": "market_cap", "capitalisation": "market_cap", "mkt_cap": "market_cap",
    "القيمه_السوقيه": "market_cap", "قيمه_سوقيه": "market_cap",
    "sector": "sector", "activity": "sector", "القطاع": "sector", "النشاط": "sector",
    # indices
    "index": "index_name", "index_name": "index_name", "المؤشر": "index_name",
    "مؤشر": "index_name", "اسم_المؤشر": "index_name",
}

#: A row is unusable without these.
REQUIRED_FIELDS: tuple[str, ...] = ("symbol",)
#: Fields a price grid is *expected* to carry. Missing ones do not invalidate
#: the parse, they downgrade it to PARTIAL so the gap is visible downstream
#: instead of being discovered later as a silent hole in the series.
EXPECTED_FIELDS: tuple[str, ...] = ("symbol", "close", "volume")
MIN_FIELDS_PER_ROW = 2


def _column_key(header: str) -> str:
    """Normalise a scraped header into the synonym-table key form."""
    key = normalize_header(header).replace(" ", "_")
    key = key.replace("%", "pct").replace("(pct)", "pct")
    return "_".join(filter(None, key.split("_")))


def _build_synonyms() -> dict[str, str]:
    """Index the synonym table under *normalised* keys.

    Arabic entries are orthographically folded here rather than hand-written in
    folded form, so ``أعلى`` in the table matches ``الاعلى`` on a page — the
    difference between a parser that survives an EGX page edit and one that
    silently returns zero rows.
    """
    out: dict[str, str] = {}
    for raw, target in COLUMN_SYNONYMS.items():
        key = _column_key(raw)
        if key:
            out.setdefault(key, target)
    return out


_SYNONYMS: dict[str, str] = _build_synonyms()


@dataclass
class ParseReport:
    """What actually happened during a parse — first-class output, not logging."""

    strategy: str
    rows: int
    fields_present: list[str] = field(default_factory=list)
    fields_missing: list[str] = field(default_factory=list)
    tables_seen: int = 0
    json_blobs_seen: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.rows > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "rows": self.rows,
            "fields_present": sorted(self.fields_present),
            "fields_missing": sorted(self.fields_missing),
            "tables_seen": self.tables_seen,
            "json_blobs_seen": self.json_blobs_seen,
            "notes": self.notes,
        }


@dataclass
class ParsedRows:
    rows: list[dict[str, Any]]
    report: ParseReport


class EGXOfficialSource(MarketDataSource):
    """Scraper with endpoint candidates and multi-strategy parsing."""

    descriptor = SourceDescriptor(
        name="egx_official",
        capabilities=frozenset(
            {
                Capability.MARKET_SNAPSHOT,
                Capability.QUOTE,
                Capability.INDEX_QUOTE,
                Capability.SECTOR_STATS,
                Capability.SYMBOL_MASTER,
            }
        ),
        priority=10,
        official=True,
        usage_note=(
            "Public pages of egx.com.eg. Scraped politely (paced, identified UA, "
            "cached); no API contract is offered by the exchange. Consult the site's "
            "terms before redistributing the data."
        ),
        typical_latency_ms=1500,
        symbol_style="bare",
    )

    def __init__(
        self,
        transport: Transport,
        *,
        base: str = BASE,
        candidates: Optional[dict[Capability, list[str]]] = None,
        health=None,
        clock=None,
        max_candidates_per_call: int = 3,
    ) -> None:
        super().__init__(transport, health=health, clock=clock)
        self.base = base.rstrip("/")
        self.candidates = candidates or ENDPOINT_CANDIDATES
        self.max_candidates_per_call = max_candidates_per_call
        #: capability → endpoint that last produced rows (learned, process-local)
        self._sticky: dict[Capability, str] = {}

    # ── configuration ───────────────────────────────────────────────────────

    def available(self) -> bool:
        return super().available() and self.transport is not None

    def _ordered_candidates(self, capability: Capability) -> list[str]:
        paths = list(self.candidates.get(capability, []))
        sticky = self._sticky.get(capability)
        if sticky and sticky in paths:
            paths.remove(sticky)
            paths.insert(0, sticky)
        elif sticky:  # pragma: no cover - sticky path no longer configured
            paths.insert(0, sticky)
        return paths[: self.max_candidates_per_call]

    def _url(self, path: str) -> str:
        return f"{self.base}{path}" if path.startswith("/") else f"{self.base}/{path}"

    # ── fetching ────────────────────────────────────────────────────────────

    def _fetch_candidates(
        self,
        capability: Capability,
        parser,
        *,
        attempt_index: int = 0,
    ) -> tuple[ParsedRows, TransportResponse, str]:
        """Try candidates until one yields rows; report the last failure otherwise."""
        errors: list[str] = []
        last_report: Optional[ParseReport] = None
        last_resp: Optional[TransportResponse] = None
        saw_content = False

        for path in self._ordered_candidates(capability):
            url = self._url(path)
            try:
                resp = self._get(url, headers={"Referer": f"{self.base}/en/HomePage.aspx"})
            except (SourceUnavailable, TransportError) as exc:
                errors.append(f"{path}: {getattr(exc, 'reason', exc)}")
                continue

            try:
                parsed = parser(resp.text)
            except SourceParseError as exc:
                # The page arrived but our mapping failed: that is a parser
                # regression, and it must not be reported as an outage.
                errors.append(f"{path}: parse error {exc.reason}")
                last_report = ParseReport(
                    strategy="failed",
                    rows=0,
                    notes=[str(exc.reason)],
                    tables_seen=1 if resp.text.strip() else 0,
                )
                last_resp = resp
                saw_content = saw_content or bool(resp.text.strip())
                continue

            last_report, last_resp = parsed.report, resp
            if parsed.report.ok:
                self._sticky[capability] = path
                return parsed, resp, path
            errors.append(
                f"{path}: {parsed.report.rows} rows "
                f"(strategy={parsed.report.strategy}, tables={parsed.report.tables_seen})"
            )
            saw_content = saw_content or bool(resp.text.strip())

        detail = "; ".join(errors) if errors else "no candidates configured"
        if saw_content:
            # The pages reached us and had content, but nothing mapped to rows.
            # That is a parser regression, not an outage — say so, loudly, and
            # do not let the circuit breaker hide it behind a generic failure.
            raise SourceParseError(
                self.name,
                f"page(s) reached but no rows mapped ({detail})",
                sample=(last_resp.text[:400] if last_resp is not None else ""),
            )
        raise SourceUnavailable(self.name, f"no candidate produced rows ({detail})")

    # ── parsing ─────────────────────────────────────────────────────────────

    @staticmethod
    def _map_columns(headers: Iterable[str]) -> dict[int, str]:
        """Column index → canonical field.

        Resolution order per header, most reliable first:

        1. **Whole-key exact match** — ``"change %"`` → ``change_pct``.
        2. **Token match** — split on separators and resolve each token
           independently, then prefer the last (most specific) token that maps
           to a field. This is what makes a two-level header work: the merged
           label ``"Prices (EGP) Open"`` resolves on ``open``, not on the
           ``price`` substring, so a colspan'd price group cannot hijack the
           open column into ``close``.
        3. **Substring** — last resort for decorated labels.

        A field is assigned to at most one column (first writer wins), which
        keeps a duplicated header from silently overwriting a good mapping.
        """
        mapping: dict[int, str] = {}
        for idx, header in enumerate(headers):
            key = _column_key(header)
            canonical = _SYNONYMS.get(key)

            if canonical is None:
                tokens = [t for t in key.split("_") if t]
                token_hits = [t for t in tokens if t in _SYNONYMS]
                if token_hits:
                    candidates = {_SYNONYMS[t] for t in token_hits}
                    # "Change %" style: an explicit percent token plus a
                    # change-ish token resolves to the percent field.
                    if "pct" in tokens or "percent" in tokens:
                        for cand in candidates:
                            if cand == "change":
                                canonical = "change_pct"
                                break
                    if canonical is None:
                        canonical = _SYNONYMS[token_hits[-1]]

            if canonical is None:
                for token, target in _SYNONYMS.items():
                    if len(token) > 2 and token in key:
                        canonical = target
                        break

            if canonical and canonical not in mapping.values():
                mapping[idx] = canonical
        return mapping

    @classmethod
    def rows_from_table(cls, table: HTable) -> ParsedRows:
        """Map one HTML table into canonical rows."""
        headers = table.normalised_headers()
        mapping = cls._map_columns(headers)
        rows: list[dict[str, Any]] = []

        if not mapping:
            return ParsedRows(
                [],
                ParseReport(
                    strategy="html_table",
                    rows=0,
                    fields_missing=list(REQUIRED_FIELDS),
                    notes=[f"no header matched a known column (headers={headers[:8]})"],
                ),
            )

        for raw in table.rows:
            record: dict[str, Any] = {}
            for idx, field_name in mapping.items():
                if idx >= len(raw):
                    continue
                value = raw[idx]
                if value is None or str(value).strip() == "":
                    continue
                record[field_name] = cls._coerce(field_name, value)
            if not cls._is_row_usable(record):
                continue
            record["_raw"] = {headers[i]: raw[i] for i in range(min(len(raw), len(headers)))}
            rows.append(record)

        present = sorted({k for r in rows for k in r if k != "_raw"})
        return ParsedRows(
            rows,
            ParseReport(
                strategy="html_table",
                rows=len(rows),
                fields_present=present,
                fields_missing=[f for f in EXPECTED_FIELDS if f not in present],
                tables_seen=1,
            ),
        )

    @staticmethod
    def _coerce(field_name: str, raw: str) -> Any:
        if field_name in {"symbol", "name", "sector", "index_name"}:
            text = clean_text(raw)
            if field_name == "symbol":
                upper = to_ascii_digits(text).upper().replace("EGX:", "").strip()
                return upper
            return text
        if field_name in {"change_pct"}:
            return parse_percent(raw)
        if field_name in {
            "close", "open", "high", "low", "previous_close",
            "change", "value_traded", "market_cap",
        }:
            return parse_number(raw)
        if field_name in {"volume", "trades"}:
            value = parse_number(raw)
            return int(value) if value is not None else None
        return clean_text(raw)

    @classmethod
    def _is_row_usable(cls, record: dict[str, Any]) -> bool:
        symbol = record.get("symbol")
        if not isinstance(symbol, str) or not looks_like_symbol(symbol):
            return False
        fields = [k for k, v in record.items() if v is not None and k != "symbol"]
        return len(fields) >= MIN_FIELDS_PER_ROW

    @classmethod
    def parse_snapshot(cls, html_text: str) -> ParsedRows:
        """Whole-market snapshot: first table that yields the most usable rows."""
        tables = extract_tables(html_text)
        best: Optional[ParsedRows] = None
        for table in tables:
            candidate = cls.rows_from_table(table)
            if best is None or len(candidate.rows) > len(best.rows):
                best = candidate

        if best and best.rows:
            best.report.tables_seen = len(tables)
            return best

        # Fallback: JSON embedded in the page (some EGX views hydrate tables
        # client-side but still ship the payload in a <script> block).
        blobs = extract_json_blobs(html_text)
        rows = cls.rows_from_json(blobs)
        if rows:
            return ParsedRows(
                rows,
                ParseReport(
                    strategy="embedded_json",
                    rows=len(rows),
                    fields_present=sorted({k for r in rows for k in r if k != "_raw"}),
                    fields_missing=[],
                    tables_seen=len(tables),
                    json_blobs_seen=len(blobs),
                ),
            )

        return ParsedRows(
            [],
            ParseReport(
                strategy="none",
                rows=0,
                fields_missing=list(REQUIRED_FIELDS),
                tables_seen=len(tables),
                json_blobs_seen=len(blobs),
                notes=["no table or embedded payload yielded symbol rows"],
            ),
        )

    @classmethod
    def rows_from_json(cls, blobs: Iterable[Any]) -> list[dict[str, Any]]:
        """Pull record lists out of arbitrary embedded JSON."""
        out: list[dict[str, Any]] = []
        for blob in blobs:
            for obj in _walk_dicts(blob):
                record = cls._json_record(obj)
                if record is not None and cls._is_row_usable(record):
                    out.append(record)
        return out

    @classmethod
    def _json_record(cls, obj: dict[str, Any]) -> Optional[dict[str, Any]]:
        record: dict[str, Any] = {}
        for key, value in obj.items():
            canonical = COLUMN_SYNONYMS.get(
                "_".join(normalize_header(str(key)).split())
            ) or COLUMN_SYNONYMS.get(normalize_header(str(key)).replace(" ", "_"))
            if not canonical or canonical in record:
                continue
            if isinstance(value, (dict, list)):
                continue
            coerced = cls._coerce(canonical, str(value))
            if coerced is not None:
                record[canonical] = coerced
        return record or None

    # ── capabilities ────────────────────────────────────────────────────────

    def market_snapshot(self, *, attempt_index: int = 0) -> Sourced[list[dict[str, Any]]]:
        parsed, resp, path = self._fetch_candidates(
            Capability.MARKET_SNAPSHOT, self.parse_snapshot, attempt_index=attempt_index
        )
        out = self.wrap(
            parsed.rows,
            response=resp,
            endpoint=self._url(path),
            locator={"endpoint_id": path, "parse": parsed.report.as_dict()},
            attempt_index=attempt_index,
        )
        self._annotate_report(out, parsed.report)
        return out

    def symbol_master(self, *, attempt_index: int = 0) -> Sourced[list[dict[str, Any]]]:
        parsed, resp, path = self._fetch_candidates(
            Capability.SYMBOL_MASTER, self.parse_snapshot, attempt_index=attempt_index
        )
        out = self.wrap(
            parsed.rows,
            response=resp,
            endpoint=self._url(path),
            locator={"endpoint_id": path, "parse": parsed.report.as_dict()},
            attempt_index=attempt_index,
        )
        self._annotate_report(out, parsed.report)
        return out

    def quote(
        self,
        symbol: str,
        *,
        attempt_index: int = 0,
        snapshot: Optional[list[dict[str, Any]]] = None,
    ) -> Sourced[dict[str, Any]]:
        """Single quote, served from the market snapshot (EGX has no per-symbol page)."""
        wanted = to_ascii_digits(symbol.strip().upper()).replace("EGX:", "")
        rows = snapshot
        if rows is None:
            rows = self.market_snapshot(attempt_index=attempt_index).value

        for row in rows:
            if str(row.get("symbol", "")).upper() == wanted:
                out = self.wrap(
                    row,
                    endpoint=self._url(self._sticky.get(Capability.MARKET_SNAPSHOT, "")),
                    locator={"resolved_from": "market_snapshot", "symbol": wanted},
                    attempt_index=attempt_index,
                )
                return out
        raise SourceDataError(self.name, f"symbol {wanted} not present in the EGX snapshot")

    def index_quote(self, index: str, *, attempt_index: int = 0) -> Sourced[dict[str, Any]]:
        wanted = index.strip().upper()

        def parser(html_text: str) -> ParsedRows:
            return self._parse_indexes(html_text, wanted)

        parsed, resp, path = self._fetch_candidates(
            Capability.INDEX_QUOTE, parser, attempt_index=attempt_index
        )
        row = parsed.rows[0]
        out = self.wrap(
            row,
            response=resp,
            endpoint=self._url(path),
            locator={"endpoint_id": path, "index": wanted, "parse": parsed.report.as_dict()},
            attempt_index=attempt_index,
        )
        self._annotate_report(out, parsed.report)
        return out

    @classmethod
    def _parse_indexes(cls, html_text: str, wanted: str) -> ParsedRows:
        tables = extract_tables(html_text)
        matches: list[dict[str, Any]] = []
        for table in tables:
            parsed = cls.rows_from_table(table)
            for row in parsed.rows:
                label = str(row.get("index_name") or row.get("name") or "").upper()
                if wanted in label:
                    row["index"] = wanted
                    matches.append(row)

        if not matches:
            # Indices pages often print "EGX 30   12,345.67   +1.2%  10:30"
            for table in tables:
                for raw in table.rows:
                    joined = " ".join(clean_text(c) for c in raw)
                    if wanted in joined.replace(" ", "") or wanted in joined.upper():
                        numbers = [parse_number(c) for c in raw]
                        numbers = [n for n in numbers if n is not None]
                        name_match = _index_level(numbers)
                        if name_match is not None:
                            matches.append(
                                {
                                    "index": wanted,
                                    "index_name": wanted,
                                    "close": name_match[0],
                                    "change_pct": name_match[1],
                                    "_raw": {str(i): v for i, v in enumerate(raw)},
                                }
                            )
                            break

        if not matches:
            raise SourceParseError(
                cls.descriptor.name,
                f"index {wanted} not found among {len(tables)} table(s)",
            )
        return ParsedRows(
            matches[:1],
            ParseReport(
                strategy="html_table_index",
                rows=1,
                fields_present=sorted({k for k in matches[0] if k != "_raw"}),
                tables_seen=len(tables),
            ),
        )

    def sector_stats(self, *, attempt_index: int = 0) -> Sourced[list[dict[str, Any]]]:
        parsed, resp, path = self._fetch_candidates(
            Capability.SECTOR_STATS, self.parse_snapshot, attempt_index=attempt_index
        )
        rows = [r for r in parsed.rows if r.get("sector") or r.get("name")]
        if not rows:
            raise SourceDataError(self.name, "sector page produced no sector rows")
        out = self.wrap(
            rows,
            response=resp,
            endpoint=self._url(path),
            locator={"endpoint_id": path, "parse": parsed.report.as_dict()},
            attempt_index=attempt_index,
        )
        self._annotate_report(out, parsed.report)
        return out

    # ── reporting ───────────────────────────────────────────────────────────

    @staticmethod
    def _annotate_report(out: Sourced[Any], report: ParseReport) -> None:
        if report.fields_missing:
            out.add_flag(
                QualityFlag.PARTIAL,
                f"EGX page parsed but missing field(s): {', '.join(report.fields_missing)}",
            )
        if report.rows and report.strategy in {"embedded_json"}:
            out.add_flag(
                QualityFlag.VALID,
                "rows recovered from an embedded JSON payload after the HTML "
                "tables failed to map — verify the page layout has not shifted",
            )


def _index_level(numbers: list[float]) -> Optional[tuple[float, Optional[float]]]:
    """Pick (level, change) from a row of numbers: level is the big one."""
    if not numbers:
        return None
    # A level is usually >= 100 and larger than any percent on the same row.
    big = max(numbers)
    if big < 100:
        return None
    change = next(
        (n for n in numbers if n is not big and abs(n) <= 25),
        None,
    )
    return big, change


def _walk_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    """Yield every dict in a nested JSON structure."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_dicts(item)
