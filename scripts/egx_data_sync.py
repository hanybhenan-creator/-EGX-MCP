#!/usr/bin/env python3
"""EGX data sync & live source probe.

Refreshes happen here rather than behind an MCP tool because every tool in
this server declares ``readOnlyHint=True`` (a marketplace-directory
requirement enforced by ``tests/unit/test_tool_annotations.py``). This is the
same split the project already uses for ``scripts/refresh_coinlists.py``.

    # What is the state of the data system right now?
    python scripts/egx_data_sync.py status

    # Which EGX endpoints actually answer, and what do we get out of them?
    python scripts/egx_data_sync.py probe

    # Pull a year of daily history for the EGX30 constituents
    python scripts/egx_data_sync.py daily --index EGX30 --days 365

    # Whole-market snapshot: symbol master + latest quotes
    python scripts/egx_data_sync.py universe

    # What does the trading calendar believe about a year?
    python scripts/egx_data_sync.py holidays --year 2026

Every command is safe to re-run: refreshes skip symbols already current for
the EGX session, and the store is append-only where it matters (quotes) and
idempotent where it should be (daily bars).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradingview_mcp.core.data.calendar import EGXCalendar, holiday_metadata  # noqa: E402
from tradingview_mcp.core.data.factory import DataLayer  # noqa: E402
from tradingview_mcp.core.data.ingest import (  # noqa: E402
    EGXIngestor,
    symbols_for_index,
    symbols_from_coinlist,
)
from tradingview_mcp.core.data.sources.base import Capability  # noqa: E402
from tradingview_mcp.core.data.sources.egx_official import ENDPOINT_CANDIDATES  # noqa: E402
from tradingview_mcp.core.data.sources.yahoo import INDEX_SYMBOLS, YahooFinanceSource  # noqa: E402


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


# ── status ──────────────────────────────────────────────────────────────────


def cmd_status(args: argparse.Namespace) -> int:
    layer = DataLayer(db_path=args.db)
    try:
        state = layer.calendar.state()
        _print(
            {
                "market_clock": {
                    "now_local": state.now_local.isoformat(),
                    "phase": state.phase,
                    "is_trading_day": state.is_trading_day,
                    "expect_live_data": state.expect_live_data,
                    "seconds_to_open": state.seconds_to_open,
                },
                "calendar_warnings": layer.calendar.sync_warnings(state.now_local.year),
                "sources": layer.registry.health_report(),
                "store": layer.store.stats().as_dict(),
                "source_reliability": layer.store.source_reliability(),
                "recent_runs": layer.store.recent_runs(limit=5),
                "recent_failures": layer.store.recent_audit(limit=15, failures_only=True),
            }
        )
        return 0
    finally:
        layer.close()


# ── live probe ──────────────────────────────────────────────────────────────


def cmd_probe(args: argparse.Namespace) -> int:
    """Ask every candidate endpoint what it can actually give us.

    This is the command that turns "the scraper should work" into evidence on
    *this* network: per URL, the HTTP status, byte count, how many tables were
    found, and how many rows mapped to known columns. Run it after any EGX
    redesign, and after changing the endpoint list.
    """
    from tradingview_mcp.core.data.transport import HttpxTransport, TransportError

    cal = EGXCalendar.from_env()
    transport = HttpxTransport(default_timeout_s=args.timeout)
    report: dict[str, object] = {"checked_at": cal.now().isoformat(), "endpoints": {}}

    for capability, paths in ENDPOINT_CANDIDATES.items():
        entries = []
        for path in paths:
            url = f"https://www.egx.com.eg{path}"
            entry: dict[str, object] = {"url": url}
            try:
                resp = transport.request("GET", url)
                entry["status"] = resp.status
                entry["bytes"] = resp.bytes_received
                if capability in (Capability.MARKET_SNAPSHOT, Capability.QUOTE, Capability.SYMBOL_MASTER):
                    from tradingview_mcp.core.data.sources.egx_official import EGXOfficialSource

                    parsed = EGXOfficialSource.parse_snapshot(resp.text)
                    entry["tables_seen"] = parsed.report.tables_seen
                    entry["rows_parsed"] = parsed.report.rows
                    entry["fields_present"] = parsed.report.fields_present
                    entry["fields_missing"] = parsed.report.fields_missing
                    entry["strategy"] = parsed.report.strategy
                    if parsed.rows:
                        entry["sample"] = {
                            k: v for k, v in parsed.rows[0].items() if k != "_raw"
                        }
            except TransportError as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            entries.append(entry)
        report["endpoints"][capability.value] = entries  # type: ignore[index]

    # Independent corroboration: can Yahoo serve each EGX index symbol?
    yahoo = YahooFinanceSource(transport)
    index_results = {}
    for name, mapping in INDEX_SYMBOLS.items():
        try:
            result = yahoo.quote(mapping["yahoo"], exchange="YAHOO")
            index_results[name] = {
                "yahoo_symbol": mapping["yahoo"],
                "price": result.value.get("price"),
                "mapping_verified": mapping["verified"],
            }
        except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
            index_results[name] = {"yahoo_symbol": mapping["yahoo"], "error": str(exc)}
    report["yahoo_index_mapping"] = index_results

    transport.close()
    _print(report)
    return 0


# ── refreshes ───────────────────────────────────────────────────────────────


def _resolve(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [s.strip().upper().replace("EGX:", "") for s in args.symbols.split(",") if s.strip()]
    if args.index:
        return symbols_for_index(args.index)
    return symbols_from_coinlist()


def _refresh(args: argparse.Namespace, kind: str) -> int:
    layer = DataLayer(db_path=args.db)
    try:
        ingestor = EGXIngestor(layer.router, layer.store, calendar=layer.calendar)
        if kind == "universe":
            report = ingestor.refresh_universe(cross_check=not args.no_cross_check)
        elif kind == "quotes":
            report = ingestor.refresh_quotes(
                _resolve(args)[: args.limit], cross_check=not args.no_cross_check
            )
        else:
            report = ingestor.refresh_daily(
                _resolve(args)[: args.limit],
                days=args.days,
                cross_check=not args.no_cross_check,
                force=args.force,
                max_workers=args.workers,
            )
        _print(report.as_dict(limit=args.show_problems))
        if args.coverage:
            _print(ingestor.coverage(_resolve(args)[: args.limit]))
        return 0 if report.status in {"ok", "partial", "empty"} else 1
    finally:
        layer.close()


# ── calendar ────────────────────────────────────────────────────────────────


def cmd_holidays(args: argparse.Namespace) -> int:
    cal = EGXCalendar.from_env()
    meta = holiday_metadata().get("years", {})
    days = sorted(d.isoformat() for d in cal.holidays_for(args.year))
    _print(
        {
            "year": args.year,
            "closures_loaded": days,
            "closures_count": len(days),
            "verified": bool(meta.get(str(args.year), {}).get("verified", False)),
            "warnings": cal.sync_warnings(args.year),
            "note": (
                "Moveable Islamic holidays (Eid al-Fitr, Eid al-Adha, Hijri New Year, "
                "Mawlid) and one-off closures must be copied from the EGX announcement "
                "page into src/tradingview_mcp/core/data/egx_holidays.json, then set "
                "verified=true for the year."
            ),
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=os.environ.get("EGX_DATA_DB", "egx_data.db"))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show source health, store size and calendar state")

    probe = sub.add_parser("probe", help="check every EGX endpoint candidate live")
    probe.add_argument("--timeout", type=float, default=20.0)

    for name, help_text in (
        ("daily", "fetch and store daily OHLCV history"),
        ("quotes", "append latest quotes for a symbol set"),
        ("universe", "whole-market snapshot: symbols + quotes"),
    ):
        p = sub.add_parser(name, help=help_text)
        if name != "universe":
            p.add_argument("--symbols", default="", help="comma-separated tickers")
            p.add_argument("--index", default="", help="EGX30 | EGX70 | EGX100 | SHARIAH33 | EGX35LV | TAMAYUZ")
            p.add_argument("--limit", type=int, default=60)
            p.add_argument("--no-cross-check", action="store_true")
        if name == "daily":
            p.add_argument("--days", type=int, default=365)
            p.add_argument("--force", action="store_true")
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--coverage", action="store_true")
        p.add_argument("--show-problems", type=int, default=25)

    holidays = sub.add_parser("holidays", help="show what the trading calendar believes")
    holidays.add_argument("--year", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "probe":
        return cmd_probe(args)
    if args.command == "holidays":
        if args.year is None:
            args.year = EGXCalendar.from_env().now().year
        return cmd_holidays(args)
    return _refresh(args, args.command)


if __name__ == "__main__":
    raise SystemExit(main())
