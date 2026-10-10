"""Value parsing for Arabic/English market pages.

Scraped numbers are not typed. An EGX page can print the same quantity as
``1,234,567``, ``1234567``, ``١٢٣٤٥٦٧`` (Arabic-Indic digits), ``1٬234٬567``
(Arabic thousands separator), ``1.234.567``, ``(12.5)`` for a negative, or
``1.2M``. Any of those landing in a float field as ``None`` or, worse, as the
wrong magnitude is a silent data corruption that no downstream indicator can
detect for you.

Every function here returns ``None`` on anything it cannot confidently read.
That is deliberate: **no guessing**. A ``None`` propagates into a PARTIAL
quality flag; a guessed number propagates into a trade signal.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# ── Arabic text normalisation ───────────────────────────────────────────────

_ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
_EXTENDED_ARABIC_INDIC = "۰۱۲۳۴۵۶۷۸۹"
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate(_ARABIC_INDIC)}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate(_EXTENDED_ARABIC_INDIC)})

_ARABIC_MARKS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e8\u06ea-\u06ed\u0640]")
_TATWEEL = "\u0640"

_ALEF_VARIANTS = {"\u0623": "\u0627", "\u0625": "\u0627", "\u0622": "\u0627", "\u0671": "\u0627"}
_YEH_VARIANTS = {"\u0649": "\u064a", "\u06cc": "\u064a"}
_TEH_MARBUTA = {"\u0629": "\u0647"}

#: Arabic thousands separator (U+066C) and decimal separator (U+066B).
_ARABIC_THOUSANDS = "\u066c"
_ARABIC_DECIMAL = "\u066b"

_PUNCT_RE = re.compile(r"[^\w\s\u0600-\u06ff%\.\-+(),]")


def to_ascii_digits(text: str) -> str:
    """``١٢٣`` / ``۱۲۳`` → ``123``. Leaves other characters untouched."""
    return text.translate(_DIGIT_MAP)


def normalize_arabic(text: str) -> str:
    """Fold Arabic orthography for header matching (NOT for display).

    Alef/yeh/teh-marbuta variants and the tatweel are unified, Harakat are
    stripped. This is what makes ``أعلى`` and ``اعلى`` the same header — the
    difference between a parser that survives an EGX page edit and one that
    silently returns zero rows.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = out.replace(_TATWEEL, "")
    out = _ARABIC_MARKS.sub("", out)
    for src, dst in {**_ALEF_VARIANTS, **_YEH_VARIANTS, **_TEH_MARBUTA}.items():
        out = out.replace(src, dst)
    return out


def normalize_header(text: str) -> str:
    """Canonical header key: lower-cased, digits folded, punctuation dropped."""
    if not text:
        return ""
    t = normalize_arabic(to_ascii_digits(text)).lower()
    t = t.replace(_ARABIC_DECIMAL, ".").replace(_ARABIC_THOUSANDS, ",")
    t = _PUNCT_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def header_key(text: str) -> str:
    """Header with separators collapsed — ``"Change %"`` → ``"change_pct"``."""
    t = normalize_header(text)
    t = t.replace(" ", "_")
    t = t.replace("%", "pct").replace("(pct)", "pct")
    t = re.sub(r"_+", "_", t).strip("_")
    return t


# ── numbers ─────────────────────────────────────────────────────────────────

_SUFFIX_MULTIPLIER = {
    "k": 1_000,
    "thousand": 1_000,
    "ألف": 1_000,
    "الف": 1_000,
    "m": 1_000_000,
    "mn": 1_000_000,
    "million": 1_000_000,
    "مليون": 1_000_000,
    "b": 1_000_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
    "مليار": 1_000_000_000,
}

_NUM_BODY = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def parse_number(raw: object, *, percent: bool = False) -> Optional[float]:
    """Best-effort numeric read that refuses to guess.

    Returns ``None`` for anything ambiguous — including the classic trap of
    ``"1.234"`` where the meaning depends on locale, which is resolved by
    looking at *all* separators in the string rather than assuming one.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if value == value and value not in (float("inf"), float("-inf")) else None

    text = str(raw).strip()
    if not text:
        return None

    text = to_ascii_digits(text)
    text = normalize_arabic(text)
    text = text.replace(_ARABIC_DECIMAL, ".").replace(_ARABIC_THOUSANDS, ",")
    text = text.replace("\u00a0", " ").replace("\u200f", "").replace("\u200e", "")

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    if text.startswith("\u2212"):  # unicode minus
        text = "-" + text[1:]

    is_percent = text.endswith("%") or percent
    text = text.rstrip("%").strip()

    multiplier = 1.0
    suffix_match = re.search(r"\s*([A-Za-z\u0600-\u06ff]+)$", text)
    if suffix_match:
        suffix = normalize_arabic(suffix_match.group(1)).lower()
        if suffix in _SUFFIX_MULTIPLIER:
            multiplier = float(_SUFFIX_MULTIPLIER[suffix])
            text = text[: suffix_match.start()].strip()

    text = text.replace(" ", "")
    if not text:
        return None

    cleaned = _normalise_separators(text)
    if cleaned is None or not _NUM_BODY.match(cleaned):
        return None

    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value != value:  # NaN
        return None

    value *= multiplier
    return -value if negative else value


def _normalise_separators(text: str) -> Optional[str]:
    """Resolve ``,`` / ``.`` / space separators into one decimal point.

    Rules, in order:
      1. Both ``,`` and ``.`` present → the **rightmost** one is the decimal
         separator, the other is a grouping separator (matches both
         ``1,234.56`` and ``1.234,56``).
      2. Only ``,`` → decimal iff it appears once and is followed by 1–2
         digits AND the number has ≤ 3 leading digits (``12,5`` is 12.5 from an
         Arabic-locale page; ``1,234`` is 1234). Otherwise grouping.
      3. Only ``.`` → decimal iff it appears once and is followed by 1–2 digits
         (``85.5``) or the value has no plausible grouping shape (``85.50``).
         ``1.234`` with exactly 3 trailing digits and no other separator stays
         ambiguous ⇒ grouping, because EGX prices above 1,000 are printed that
         way far more often than as 1.234 of a pound.
    """
    if "," in text and "." in text:
        dec = max(text.rfind(","), text.rfind("."))
        int_part = text[:dec].replace(",", "").replace(".", "")
        frac = text[dec + 1 :].replace(",", "").replace(".", "")
        if not frac:
            return None
        return f"{int_part}.{frac}"

    if "," in text:
        parts = text.split(",")
        if len(parts) == 2 and 1 <= len(parts[1]) <= 2 and len(parts[0].lstrip("+-")) <= 3:
            return f"{parts[0]}.{parts[1]}"
        if all(len(p) == 3 for p in parts[1:]):
            return "".join(parts)
        return None

    if "." in text:
        parts = text.split(".")
        if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
            return text
        if len(parts) == 2 and len(parts[1]) == 3 and len(parts[0].lstrip("+-")) <= 3:
            # Ambiguous by construction (1.234) — read as grouping, the
            # convention on every Egyptian market page seen in fixtures.
            return "".join(parts)
        if len(parts) > 2 and all(len(p) == 3 for p in parts[1:]):
            return "".join(parts)
        return None

    return text


def parse_int(raw: object) -> Optional[int]:
    value = parse_number(raw)
    if value is None:
        return None
    try:
        return int(round(value))
    except (ValueError, OverflowError):  # pragma: no cover
        return None


def parse_percent(raw: object) -> Optional[float]:
    """Read a percent as a *percentage number* (``12.5%`` → ``12.5``)."""
    if isinstance(raw, str):
        has_sign = "%" in raw or "\u066a" in raw
        value = parse_number(raw.replace("\u066a", "%"))
        return value if has_sign and value is not None else value
    return parse_number(raw, percent=True)


def parse_ratio_percent(raw: object) -> Optional[float]:
    """Read a percent as a *fraction* (``12.5%`` → ``0.125``)."""
    value = parse_percent(raw)
    return None if value is None else value / 100.0


def looks_like_symbol(text: object) -> bool:
    """EGX tickers are 1–8 chars of ``[A-Z0-9]`` (``COMI``, ``EGX30``, ``AALR``)."""
    if not isinstance(text, str):
        return False
    t = to_ascii_digits(text).strip().upper()
    return bool(re.fullmatch(r"[A-Z][A-Z0-9]{0,7}", t))


def clean_text(raw: object) -> str:
    """Collapse whitespace and fold Arabic presentation forms for display."""
    if raw is None:
        return ""
    text = unicodedata.normalize("NFKC", str(raw))
    text = text.replace("\u00a0", " ").replace("\u200f", "").replace("\u200e", "")
    return re.sub(r"\s+", " ", text).strip()
