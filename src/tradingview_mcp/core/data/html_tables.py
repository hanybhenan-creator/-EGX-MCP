"""HTML table extraction using the standard library only.

No BeautifulSoup, no lxml. The parent project ships a small dependency set on
purpose (see ``pyproject.toml``), and a scraper that adds a parser dependency
for one page is a scraper that breaks other people's installs.

What this module guarantees:

* **Deterministic row/column geometry.** ``colspan``/``rowspan`` are expanded
  so ``rows[i][j]`` always refers to the *same column*, which is what a column
  mapper needs. Naive ``find_all("td")`` parsers silently shift every column to
  the left of a merged cell — the classic way scraped price tables end up
  reporting yesterday's volume as today's price.
* **Header discovery.** A header row is any row whose cells are ``<th>``, or
  the first row that has no parseable number in its numeric-looking columns.
* **Escaping.** Entities are unescaped once, after tag stripping (double
  unescaping turns ``&amp;lt;`` into a tag).
"""

from __future__ import annotations

import html as _html
import json as _json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Iterable, Optional

_CELL_TAGS = {"td", "th"}
_SKIP_TAGS = {"script", "style", "noscript", "svg"}
_WS_RE = re.compile(r"[\s\u00a0\u200f\u200e]+")


@dataclass
class HTable:
    """One ``<table>``: headers (if any) plus a rectangular body."""

    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: str = ""
    #: DOM order within the document, useful for provenance locators.
    index: int = 0
    #: Raw ``<th>`` count per header phase, kept for debugging odd layouts.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return max((len(r) for r in self.rows), default=len(self.headers))

    def normalised_headers(self) -> list[str]:
        """Headers padded/trimmed to the table width with ``col_N`` fallbacks."""
        width = self.width
        heads = list(self.headers[:width])
        heads += [f"col_{i}" for i in range(len(heads), width)]
        return [_normalise(h) for h in heads]

    def dict_rows(self) -> list[dict[str, str]]:
        """Row dicts keyed by :meth:`normalised_headers`."""
        heads = self.normalised_headers()
        out: list[dict[str, str]] = []
        for row in self.rows:
            if len(row) < len(heads):
                row = [*row, *[ "" ] * (len(heads) - len(row))]
            out.append({h: row[i] for i, h in enumerate(heads)})
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "caption": self.caption,
            "headers": self.headers,
            "row_count": len(self.rows),
            "sample": self.rows[:3],
        }


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.tables: list[HTable] = []
        self.json_blobs: list[str] = []

        self._depth_skip = 0
        self._table: Optional[HTable] = None
        self._row: Optional[list[str]] = None
        self._cell_chunks: Optional[list[str]] = None
        self._cell_is_header = False
        self._pending_rowspan: dict[int, int] = {}
        self._caption_chunks: Optional[list[str]] = None
        self._script_chunks: Optional[list[str]] = None
        self._table_index = 0
        self._header_rows: list[list[str]] = []

    # ── tag handling ────────────────────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            if tag == "script":
                self._script_chunks = []
            self._depth_skip += 1
            return
        if self._depth_skip:
            return

        if tag == "table":
            self._table = HTable(index=self._table_index)
            self._table_index += 1
            self._header_rows = []
            self._pending_rowspan = {}
        elif tag == "caption" and self._table is not None:
            self._caption_chunks = []
        elif tag == "tr" and self._table is not None:
            self._row = []
            self._row_had_th = False
            self._rowspan_created = set()
        elif tag in _CELL_TAGS and self._table is not None:
            self._cell_chunks = []
            self._cell_is_header = tag == "th"
            if self._cell_is_header:
                self._row_had_th = True
                self._table.meta["seen_th"] = True
            colspan = _int_attr(attrs, "colspan", 1)
            rowspan = _int_attr(attrs, "rowspan", 1)
            self._cell_colspan = max(1, colspan)
            self._cell_rowspan = max(1, rowspan)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            if tag == "script" and self._script_chunks is not None:
                blob = "".join(self._script_chunks).strip()
                if blob:
                    self.json_blobs.append(blob)
                self._script_chunks = None
            self._depth_skip = max(0, self._depth_skip - 1)
            return
        if self._depth_skip:
            return

        if tag == "caption" and self._caption_chunks is not None and self._table is not None:
            self._table.caption = _clean("".join(self._caption_chunks))
            self._caption_chunks = None
        elif tag in _CELL_TAGS and self._cell_chunks is not None and self._table is not None:
            value = _clean("".join(self._cell_chunks))
            self._cell_chunks = None
            if self._row is None:
                return
            colspan = getattr(self, "_cell_colspan", 1)
            rowspan = getattr(self, "_cell_rowspan", 1)

            # Advance past every column that a rowspan from an earlier row still
            # owns. A single-slot pad (the naive version) misaligns the whole row
            # whenever two adjacent cells are row-spanned — the classic cause of
            # a price column being read as volume.
            guard = 0
            while True:
                pending = self._pending_rowspan.get(len(self._row), 0)
                if pending <= 0:
                    break
                self._pending_rowspan[len(self._row)] = pending - 1
                self._row.append("")
                guard += 1
                if guard > 512:  # pragma: no cover - malformed rowspan
                    break

            col = len(self._row)
            self._row.append(value)
            for extra in range(1, colspan):
                self._row.append("")
            if rowspan > 1:
                for extra in range(colspan):
                    slot = col + extra
                    self._pending_rowspan[slot] = rowspan - 1
                    self._rowspan_created.add(slot)

            if self._cell_is_header:
                idx = len(self._table.rows)
                marks = self._table.meta.setdefault("header_row_indices", [])
                if not marks or marks[-1] != idx:
                    marks.append(idx)
        elif tag == "tr" and self._row is not None and self._table is not None:
            row = self._row
            self._row = None
            row = self._close_rowspan_gaps(row)
            if not any(c.strip() for c in row):
                return
            if self._looks_like_header_row(row):
                if not self._table.headers:
                    self._table.headers = row
                    self._header_rows.append(row)
                    return
                self._header_rows.append(row)
                # Multi-row headers (EGX index pages use them): merge by position.
                self._table.headers = [
                    _clean(f"{a} {b}") for a, b in zip(self._table.headers, row)
                ] if len(row) == len(self._table.headers) else self._table.headers
                return
            # Rows without <th> are DATA, even when they are all text. A
            # symbol-master grid (ticker + company name, no numbers) would
            # otherwise lose every row to a "looks like a header" heuristic.
            self._table.rows.append(row)
        elif tag == "table" and self._table is not None:
            if self._row is not None:
                self._table.rows.append(self._row)
                self._row = None
            self.tables.append(self._table)
            self._table = None

    def handle_data(self, data: str) -> None:
        if self._script_chunks is not None:
            self._script_chunks.append(data)
            return
        if self._depth_skip:
            return
        if self._cell_chunks is not None:
            self._cell_chunks.append(data)
        elif self._caption_chunks is not None:
            self._caption_chunks.append(data)

    def handle_entityref(self, name: str) -> None:
        self._entity(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._entity(f"&#{name};")

    def _entity(self, raw: str) -> None:
        if self._script_chunks is not None or self._depth_skip:
            return
        if self._cell_chunks is not None:
            self._cell_chunks.append(raw)
        elif self._caption_chunks is not None:
            self._caption_chunks.append(raw)

    # ── heuristics ──────────────────────────────────────────────────────────

    def _close_rowspan_gaps(self, row: list[str]) -> list[str]:
        """Expire this row's rowspan obligations, padding the slots they held.

        A ``rowspan=2`` cell from the previous row occupies its column in this
        row too (that is the whole point of rowspan), so a row that simply has
        no cell there must still be padded to that column. Left unhandled, the
        leftover obligation leaks into the *next* row and shifts every column
        right by one — exactly the failure this method exists to prevent.
        """
        if not self._pending_rowspan:
            return row
        created = getattr(self, "_rowspan_created", set())
        for col in sorted(self._pending_rowspan):
            if col >= len(row):
                row = _pad_to_width(row, col + 1)
            if col in created:
                # Placed by THIS row: it covers the following row, so it must
                # survive this row's cleanup untouched.
                continue
            self._pending_rowspan[col] -= 1
        self._pending_rowspan = {
            k: v for k, v in self._pending_rowspan.items() if v > 0
        }
        return row

    def _looks_like_header_row(self, row: list[str]) -> bool:
        """Is this row a header?

        Two independent signals, because EGX pages come in both flavours:

        * **Markup** — a row containing ``<th>`` cells is a header, and once a
          table has shown a ``<th>``, *only* ``<th>`` rows are headers. This is
          what stops a two-cell data row (``COMI | 86.20``) from being eaten as
          a header, which is exactly what a pure content heuristic does.
        * **Content** — for grids built entirely from ``<td>`` cells: the first
          row is a header when it contains no numbers at all and at least two
          labels.

        After data rows have started, nothing is a header.
        """
        if self._table is None:
            return False
        if self._table.rows:
            return False
        if getattr(self, "_row_had_th", False):
            return True
        if self._table.meta.get("seen_th"):
            return False
        numeric = sum(1 for c in row if _looks_numeric(c))
        if numeric:
            return False
        labelled = sum(1 for c in row if c)
        return labelled >= 2


def _looks_like_continued_header(row: list[str]) -> bool:
    if len([c for c in row if c.strip()]) == 0:
        return True
    numeric = sum(1 for c in row if _looks_numeric(c))
    return numeric == 0 and all(len(c) <= 24 for c in row)


def _pad_to_width(row: list[str], width: int) -> list[str]:
    if len(row) >= width:
        return row
    return [*row, *[""] * (width - len(row))]


def _int_attr(attrs: Iterable[tuple[str, Optional[str]]], name: str, default: int) -> int:
    for key, value in attrs:
        if key.lower() == name and value:
            try:
                return int(str(value).strip())
            except ValueError:
                return default
    return default


_NUM_RE = re.compile(r"^-?[\d,]+(\.\d+)?%?$")


def _looks_numeric(text: str) -> bool:
    t = text.strip().replace("\u066b", ".").replace("\u060c", "")
    if not t:
        return False
    return bool(_NUM_RE.match(t.replace(" ", "")))


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", _html.unescape(text)).strip()


def _normalise(header: str) -> str:
    return _WS_RE.sub("_", header.strip().lower()).strip("_")


# ── public API ──────────────────────────────────────────────────────────────


def extract_tables(html_text: str) -> list[HTable]:
    """All tables in document order, with merged cells expanded."""
    parser = _TableParser()
    parser.feed(html_text)
    parser.close()
    return parser.tables


def tables_with_headers(html_text: str) -> list[HTable]:
    """Only tables that yielded at least one header and one data row."""
    return [t for t in extract_tables(html_text) if t.headers and t.rows]


def extract_json_blobs(html_text: str, *, min_length: int = 2) -> list[Any]:
    """JSON payloads embedded in ``<script>`` blocks.

    Handles the three shapes seen in market sites: a bare JSON literal, a
    ``var x = {...}`` assignment, and JSON wrapped in a JS call such as
    ``__INITIAL_STATE__ = {...};``. Anything that does not parse is skipped —
    a page with 30 ad scripts must not raise.
    """
    parser = _TableParser()
    parser.feed(html_text)
    parser.close()

    out: list[Any] = []
    for blob in parser.json_blobs:
        for candidate in _json_candidates(blob):
            try:
                parsed = _json.loads(candidate)
            except (ValueError, TypeError):
                continue
            if parsed is not None:
                out.append(parsed)
    return out


def _json_candidates(blob: str) -> list[str]:
    text = blob.strip().rstrip(";").strip()
    candidates: list[str] = []

    if text[:1] in "{[":
        candidates.append(text)

    for marker in ("=", "(", ":"):
        idx = text.find(marker)
        if idx == -1:
            continue
        rest = text[idx + 1 :].strip()
        start = _first_of(rest, "{[\"")
        if start == -1:
            continue
        sliced = _balanced_slice(rest, start)
        if sliced:
            candidates.append(sliced)

    # Last resort: the largest balanced {...} / [...] run anywhere in the blob.
    for opener, closer in (("{", "}"), ("[", "]")):
        idx = text.find(opener)
        if idx != -1:
            sliced = _balanced_slice(text, idx)
            if sliced and len(sliced) > 2:
                candidates.append(sliced)

    seen: set[str] = set()
    uniq: list[str] = []
    for c in candidates:
        if len(c) < 2 or c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def _first_of(text: str, chars: str) -> int:
    positions = [text.find(c) for c in chars]
    positions = [p for p in positions if p != -1]
    return min(positions) if positions else -1


def _balanced_slice(text: str, start: int) -> Optional[str]:
    """Substring from *start* to its matching closer, respecting strings."""
    opener = text[start]
    closer = {"{": "}", "[": "]", '"': '"'}.get(opener)
    if closer is None:
        return None
    if opener == '"':
        end = text.find('"', start + 1)
        return text[start : end + 1] if end != -1 else None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
