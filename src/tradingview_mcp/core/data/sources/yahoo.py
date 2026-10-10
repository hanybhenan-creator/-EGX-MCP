"""Yahoo Finance source adapter (public chart endpoint, no API key).

Why Yahoo is in the chain at all: the EGX's own site is authoritative for
*what happened in the Egyptian session*, but it is a scraping target behind a
page layout we do not control. Yahoo gives an independent second opinion on
settled daily OHLCV, with deep history, from a completely separate
infrastructure — which is exactly what :func:`quality.triangulate` needs to
turn "one website said so" into "two independent sources agree".

Egyptian tickers live on Yahoo under the ``.CA`` suffix (Cairo — note this is
Yahoo's own quirk, unrelated to Canada's ``.CA``). Index levels are published
under caret symbols (``^CASE30`` for EGX30).

Everything in this module is a pure function of a transport response, so the
whole parser is testable offline against recorded payloads.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from ..provenance import QualityFlag, Sourced
from ..transport import Transport
from .base import (
    Capability,
    MarketDataSource,
    SourceDataError,
    SourceDescriptor,
    SourceParseError,
    SourceUnavailable,
)

BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

#: Yahoo's own limits on (interval → max range) before it rejects the request.
#: Asking anyway wastes a round trip and pollutes the audit log with 422s.
INTERVAL_MAX_RANGE_DAYS: dict[str, int] = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "90m": 60,
    "1h": 730,
    "1d": 365 * 60,
    "5d": 365 * 60,
    "1wk": 365 * 60,
    "1mo": 365 * 60,
    "3mo": 365 * 60,
}

#: Index names → Yahoo symbols. ``verified`` records whether the mapping was
#: confirmed against a live response; only verified rows are eligible for a
#: cross-check that *proves* agreement rather than assuming it.
INDEX_SYMBOLS: dict[str, dict[str, Any]] = {
    "EGX30": {"yahoo": "^CASE30", "verified": False, "note": "Cairo 30 — confirm against egx.com.eg EGX30 print"},
    "EGX70": {"yahoo": "^CASE70", "verified": False, "note": "not all index families are published by Yahoo"},
    "EGX100": {"yahoo": "^CASE100", "verified": False, "note": "spotty coverage"},
}

DEFAULT_RANGE = "1y"


def yahoo_symbol(symbol: str, exchange: str = "EGX") -> str:
    """Map a bare ticker to its Yahoo notation.

    ``COMI`` + EGX → ``COMI.CA``. Symbols that already carry a suffix (or a
    caret) are passed through untouched, so callers can hand in a fully
    qualified Yahoo ticker without this function mangling it.
    """
    s = (symbol or "").strip().upper()
    if not s:
        raise ValueError("empty symbol")
    if s.startswith("^") or "." in s or "=" in s or "-" in s:
        return s
    suffix = {"EGX": ".CA", "CAIRO": ".CA", "BIST": ".IS", "TADAWUL": ".SR"}.get(
        (exchange or "").strip().upper()
    )
    return f"{s}{suffix}" if suffix else s


def bare_symbol(yahoo: str) -> str:
    """``COMI.CA`` → ``COMI`` (the ticker the MCP tools speak)."""
    s = (yahoo or "").strip().upper()
    if s.endswith(".CA"):
        return s[: -len(".CA")]
    return s


class YahooFinanceSource(MarketDataSource):
    """Chart-endpoint reader: quotes, daily and intraday OHLCV."""

    descriptor = SourceDescriptor(
        name="yahoo",
        capabilities=frozenset(
            {
                Capability.QUOTE,
                Capability.DAILY_OHLCV,
                Capability.INTRADAY_OHLCV,
                Capability.INDEX_QUOTE,
                Capability.COMPANY_PROFILE,
            }
        ),
        priority=50,
        official=False,
        usage_note=(
            "Public Yahoo Finance chart endpoint. Unofficial and undocumented: "
            "treat it as a corroborating source, never as the sole authority."
        ),
        typical_latency_ms=700,
        symbol_style="yahoo",
    )

    def __init__(
        self,
        transport: Transport,
        *,
        base: str = BASE,
        health=None,
        clock=None,
    ) -> None:
        super().__init__(transport, health=health, clock=clock)
        self.base = base.rstrip("/")

    # ── request construction ────────────────────────────────────────────────

    def _chart_url(self, yahoo_sym: str) -> str:
        return f"{self.base}/{yahoo_sym}"

    @staticmethod
    def _params(
        interval: str,
        *,
        range_: Optional[str] = None,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> dict[str, Any]:
        if start or end:
            p: dict[str, Any] = {"interval": interval}
            if end is not None:
                p["period2"] = int(
                    datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc).timestamp()
                ) + 86_400
            if start is not None:
                p["period1"] = int(
                    datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp()
                )
            return p
        return {"interval": interval, "range": range_ or DEFAULT_RANGE}

    @staticmethod
    def _check_window(interval: str, params: dict[str, Any]) -> None:
        limit = INTERVAL_MAX_RANGE_DAYS.get(interval)
        if not limit:
            return
        if "period1" in params and "period2" in params:
            span = (params["period2"] - params["period1"]) / 86_400
            if span > limit:
                raise SourceUnavailable(
                    "yahoo",
                    f"interval {interval} supports at most {limit}d of history "
                    f"(requested {span:.0f}d) — Yahoo will reject this",
                    retryable=False,
                )

    # ── pure parsing ────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise SourceParseError("yahoo", "response is not a JSON object")
        chart = payload.get("chart")
        if not isinstance(chart, dict):
            raise SourceParseError("yahoo", "missing 'chart' key", sample=str(payload)[:200])
        err = chart.get("error")
        if err:
            code = err.get("code") if isinstance(err, dict) else err
            desc = err.get("description") if isinstance(err, dict) else ""
            raise SourceDataError("yahoo", f"upstream error {code}: {desc}")
        result = chart.get("result")
        if not result:
            raise SourceDataError("yahoo", "empty chart result")
        first = result[0]
        if not isinstance(first, dict):
            raise SourceParseError("yahoo", "chart result[0] is not an object")
        return first

    @staticmethod
    def parse_bars(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Chart result → OHLCV rows, dropping null candles.

        Yahoo emits ``null`` for every field of a session with no trades
        (halts, holidays it did not model, suspended names). Emitting those as
        zeros would poison every downstream indicator — a 0 close is a -100%
        move to any moving average — so they are dropped and counted by the
        caller instead.
        """
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quote_block = (indicators.get("quote") or [{}])[0] or {}
        adj_block = (indicators.get("adjclose") or [{}])[0] or {}

        opens = quote_block.get("open") or []
        highs = quote_block.get("high") or []
        lows = quote_block.get("low") or []
        closes = quote_block.get("close") or []
        volumes = quote_block.get("volume") or []
        adj = adj_block.get("adjclose") or []

        tz_name = ((result.get("meta") or {}).get("exchangeTimezoneName")) or "UTC"
        try:
            from zoneinfo import ZoneInfo

            tzinfo = ZoneInfo(tz_name)
        except Exception:  # pragma: no cover
            tzinfo = timezone.utc

        n = min(len(timestamps), len(opens), len(highs), len(lows), len(closes))
        bars: list[dict[str, Any]] = []
        for i in range(n):
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            if None in (o, h, l, c):
                continue
            ts = timestamps[i]
            moment = datetime.fromtimestamp(float(ts), tz=tzinfo)
            vol = volumes[i] if i < len(volumes) else None
            row: dict[str, Any] = {
                "timestamp": int(ts),
                "datetime": moment.isoformat(),
                "date": moment.date().isoformat(),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": int(vol) if vol is not None else 0,
            }
            if i < len(adj) and adj[i] is not None:
                row["adj_close"] = float(adj[i])
            bars.append(row)
        return bars

    @staticmethod
    def parse_meta(result: dict[str, Any]) -> dict[str, Any]:
        meta = dict(result.get("meta") or {})
        out: dict[str, Any] = {
            "symbol": meta.get("symbol"),
            "currency": meta.get("currency"),
            "exchange": meta.get("exchangeName") or meta.get("fullExchangeName"),
            "instrument_type": meta.get("instrumentType"),
            "timezone": meta.get("exchangeTimezoneName"),
            "name": meta.get("longName") or meta.get("shortName"),
        }
        for key in (
            "regularMarketPrice",
            "chartPreviousClose",
            "previousClose",
            "regularMarketDayHigh",
            "regularMarketDayLow",
            "regularMarketVolume",
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow",
        ):
            if meta.get(key) is not None:
                out[_snake(key)] = meta[key]
        if meta.get("regularMarketTime"):
            out["regular_market_time"] = datetime.fromtimestamp(
                float(meta["regularMarketTime"]), tz=timezone.utc
            ).isoformat()
        if out.get("regular_market_price") is None and result.get("timestamp"):
            bars = YahooFinanceSource.parse_bars(result)
            if bars:
                out["regular_market_price"] = bars[-1]["close"]
        return out

    # ── public capabilities ─────────────────────────────────────────────────

    def daily_ohlcv(
        self,
        symbol: str,
        *,
        exchange: str = "EGX",
        days: int = 365,
        range_: Optional[str] = None,
        attempt_index: int = 0,
    ) -> Sourced[list[dict[str, Any]]]:
        """Settled daily bars, oldest→newest."""
        sym = yahoo_symbol(symbol, exchange)
        if not range_:
            if days <= 8:
                range_ = "10d"
            elif days <= 32:
                range_ = "1mo"
            elif days <= 95:
                range_ = "3mo"
            elif days <= 190:
                range_ = "6mo"
            elif days <= 370:
                range_ = "1y"
            elif days <= 740:
                range_ = "2y"
            elif days <= 1850:
                range_ = "5y"
            else:
                range_ = "10y"

        params = self._params("1d", range_=range_)
        resp = self._get(self._chart_url(sym), params=params)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise SourceParseError("yahoo", f"invalid JSON: {exc}", sample=resp.text[:200]) from exc

        result = self._unwrap(payload)
        bars = self.parse_bars(result)
        if not bars:
            raise SourceDataError("yahoo", f"no usable bars returned for {sym}")

        out = self.wrap(
            bars,
            response=resp,
            locator={"yahoo_symbol": sym, "interval": "1d", "range": range_},
            attempt_index=attempt_index,
        )
        out.meta = self.parse_meta(result)
        return out

    def intraday_ohlcv(
        self,
        symbol: str,
        *,
        exchange: str = "EGX",
        interval: str = "15m",
        days: int = 30,
        attempt_index: int = 0,
    ) -> Sourced[list[dict[str, Any]]]:
        sym = yahoo_symbol(symbol, exchange)
        if interval not in INTERVAL_MAX_RANGE_DAYS:
            raise SourceUnavailable("yahoo", f"unsupported interval {interval!r}", retryable=False)
        limit = INTERVAL_MAX_RANGE_DAYS[interval]
        use_days = min(days, limit)
        end = datetime.now(timezone.utc).date()
        # ``end`` is padded to the next midnight, so the window is exactly
        # ``use_days`` long — not one day longer, which the upstream rejects.
        start = end - timedelta(days=max(0, use_days - 1))
        params = self._params(interval, start=start, end=end)
        self._check_window(interval, params)

        resp = self._get(self._chart_url(sym), params=params)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise SourceParseError("yahoo", f"invalid JSON: {exc}", sample=resp.text[:200]) from exc

        result = self._unwrap(payload)
        bars = self.parse_bars(result)
        if not bars:
            raise SourceDataError("yahoo", f"no usable {interval} bars for {sym}")
        return self.wrap(
            bars,
            response=resp,
            locator={"yahoo_symbol": sym, "interval": interval, "days": use_days},
            attempt_index=attempt_index,
        )

    def quote(
        self,
        symbol: str,
        *,
        exchange: str = "EGX",
        attempt_index: int = 0,
    ) -> Sourced[dict[str, Any]]:
        """Latest print, taken from chart metadata (no crumb/cookie dance)."""
        sym = yahoo_symbol(symbol, exchange)
        resp = self._get(self._chart_url(sym), params={"interval": "1d", "range": "5d"})
        try:
            payload = resp.json()
        except ValueError as exc:
            raise SourceParseError("yahoo", f"invalid JSON: {exc}", sample=resp.text[:200]) from exc

        result = self._unwrap(payload)
        meta = self.parse_meta(result)
        price = meta.get("regular_market_price")
        if price is None:
            raise SourceDataError("yahoo", f"no market price in metadata for {sym}")

        prev = meta.get("chart_previous_close") or meta.get("previous_close")
        quote = {
            "symbol": bare_symbol(str(meta.get("symbol") or sym)),
            "yahoo_symbol": sym,
            "name": meta.get("name"),
            "currency": meta.get("currency"),
            "price": float(price),
            "previous_close": float(prev) if prev is not None else None,
            "day_high": meta.get("regular_market_day_high"),
            "day_low": meta.get("regular_market_day_low"),
            "volume": meta.get("regular_market_volume"),
            "fifty_two_week_high": meta.get("fifty_two_week_high"),
            "fifty_two_week_low": meta.get("fifty_two_week_low"),
            "source_timestamp": meta.get("regular_market_time"),
        }
        if quote["previous_close"]:
            quote["change_percent"] = round(
                (quote["price"] / quote["previous_close"] - 1.0) * 100.0, 4
            )
        else:
            quote["change_percent"] = None

        out = self.wrap(quote, response=resp, locator={"yahoo_symbol": sym}, attempt_index=attempt_index)
        if quote["source_timestamp"] is None:
            out.add_flag(
                QualityFlag.PARTIAL,
                "Yahoo returned no regularMarketTime — quote age cannot be verified",
            )
        return out

    def index_quote(self, index: str, *, attempt_index: int = 0) -> Sourced[dict[str, Any]]:
        entry = INDEX_SYMBOLS.get(index.strip().upper())
        if not entry:
            raise SourceUnavailable(
                "yahoo", f"no Yahoo mapping for index {index!r}", retryable=False
            )
        out = self.quote(entry["yahoo"], exchange="YAHOO", attempt_index=attempt_index)
        out.value["index"] = index.strip().upper()
        out.provenance.locator["mapping_verified"] = entry["verified"]
        if not entry["verified"]:
            out.add_flag(
                QualityFlag.UNCROSSCHECKED,
                f"Yahoo symbol {entry['yahoo']} for {index} is unverified — "
                f"confirm before trusting this cross-check ({entry['note']})",
            )
        return out

    def profile(self, symbol: str, *, exchange: str = "EGX", attempt_index: int = 0) -> Sourced[dict[str, Any]]:
        """Company facts derivable from the chart metadata."""
        sym = yahoo_symbol(symbol, exchange)
        resp = self._get(self._chart_url(sym), params={"interval": "1d", "range": "1mo"})
        try:
            payload = resp.json()
        except ValueError as exc:
            raise SourceParseError("yahoo", f"invalid JSON: {exc}", sample=resp.text[:200]) from exc
        meta = self.parse_meta(self._unwrap(payload))
        if meta.get("name") is None and meta.get("fifty_two_week_high") is None:
            raise SourceDataError("yahoo", f"no profile facts for {sym}")
        meta["symbol"] = bare_symbol(str(meta.get("symbol") or sym))
        meta["yahoo_symbol"] = sym
        return self.wrap(meta, response=resp, locator={"yahoo_symbol": sym}, attempt_index=attempt_index)


def _snake(name: str) -> str:
    out = []
    for ch in name:
        if ch.isupper() and out:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
