#!/usr/bin/env python3
"""Calibrate the EGX smart-money conditions against the stored history.

    # what do the conditions actually predict on the data I have?
    python scripts/egx_calibrate.py

    # one condition, 10-session horizon, with out-of-sample folds
    python scripts/egx_calibrate.py --signals accumulation_corrected --horizon 10 --folds 4

    # full report to disk for a repo/issue attachment
    python scripts/egx_calibrate.py --json reports/calibration.json --markdown reports/calibration.md

    # which signals exist, and what is each one testing?
    python scripts/egx_calibrate.py --list

Signals are measured on the local ``DataStore`` only — nothing is fetched. If
the store is empty the command tells you which sync to run; use
``EGX_DATA_DB`` or ``--db`` to point at a specific database.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VERDICT_ORDER = {"VALIDATED": 0, "SUGGESTIVE": 1, "REJECTED": 2, "NO_EVENTS": 3, "UNMEASURED": 4}


def _load():
    from tradingview_mcp.core.services.egx_calibration_service import (
        DEFAULT_SIGNALS,
        egx_calibration_signals,
        egx_signal_calibration,
    )

    return DEFAULT_SIGNALS, egx_calibration_signals, egx_signal_calibration


def _fmt(value, *, digits: int = 3, percent: bool = False) -> str:
    if value is None:
        return "—"
    if percent:
        return f"{value * 100:.1f}%"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render_table(payload: dict) -> str:
    rows = sorted(
        payload["measurements"],
        key=lambda m: (VERDICT_ORDER.get(m["verdict"], 9), -(m["excess_hit"] or -9)),
    )
    header = (
        f"{'signal':30.30s} {'n':>6s} {'dates':>6s} {'hit':>7s} {'base':>7s} "
        f"{'excess':>7s} {'p':>7s} {'q':>7s}  verdict"
    )
    lines = [header, "─" * len(header)]
    for m in rows:
        lines.append(
            f"{m['signal']:30.30s} {m['events']:>6d} {m['event_dates']:>6d} "
            f"{_fmt(m['hit_rate'], percent=True):>7s} {_fmt(m['base_rate'], percent=True):>7s} "
            f"{_fmt(m['excess_hit'], percent=True):>7s} {_fmt(m['p_value_hit']):>7s} "
            f"{_fmt(m['q_value_hit']):>7s}  {m['verdict']}"
        )
        if m["events"] == 0:
            lines[-1] += "   ← never fires: check the condition's arithmetic"
    return "\n".join(lines)


def render_markdown(payload: dict) -> str:
    scope = payload["measurement"]
    out = [
        "# EGX signal calibration",
        "",
        f"- horizon: **{payload['horizon']} sessions** (entry next session's open, exit at close)",
        f"- symbols analysed: **{scope['symbols_analysed']}** of {scope['symbols_requested']} requested",
        f"- sessions: **{scope['sessions']}** ({scope['window']['from']} → {scope['window']['to']})",
        f"- permutations per signal: **{scope['permutations']}**",
        f"- method: {scope['method']}",
        "",
        "| signal | events | dates | hit rate | base rate | excess | p | q | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in sorted(payload["measurements"], key=lambda m: (VERDICT_ORDER.get(m["verdict"], 9), -(m["excess_hit"] or -9))):
        out.append(
            f"| `{m['signal']}` | {m['events']} | {m['event_dates']} | "
            f"{_fmt(m['hit_rate'], percent=True)} | {_fmt(m['base_rate'], percent=True)} | "
            f"{_fmt(m['excess_hit'], percent=True)} | {_fmt(m['p_value_hit'])} | "
            f"{_fmt(m['q_value_hit'])} | **{m['verdict']}** |"
        )
    if payload.get("walk_forward"):
        out += ["", "## Walk-forward (out-of-sample folds)", "", "| signal | verdict | retention | folds | reason |", "|---|---|---:|---:|---|"]
        for name, wf in payload["walk_forward"].items():
            out.append(
                f"| `{name}` | **{wf['verdict']}** | {_fmt(wf['retention'])} | {len(wf['folds'])} | {wf['reason']} |"
            )
    out += ["", "## Why these numbers are calibrated", ""]
    out += [
        "- **Base rate**: the hit rate of buying a random stored symbol on a random session for the same holding period. An excess only counts next to it.",
        "- **Excess vs benchmark**: every event's return is reduced by the equal-weighted return of all stored symbols over the same entry/exit dates, so a market-wide move cannot masquerade as a signal.",
        "- **p-value**: a block permutation of each symbol's whole return series, recomputed at the same event positions — the null is that the signal's *timing* carries no information.",
        "- **q-value**: Benjamini–Hochberg across every signal in this run. Testing a dozen conditions on one history always produces one that looks significant.",
        "- **Walk-forward**: contiguous out-of-sample folds; a fold only votes when its in-sample edge and event count are big enough to be worth retaining.",
        "",
        f"> {payload['disclaimer']}",
    ]
    if payload.get("warnings"):
        out += ["", "## Warnings", ""] + [f"- {w}" for w in payload["warnings"]]
    return "\n".join(out)


def cmd_list() -> int:
    _, catalogue, _ = _load()
    payload = catalogue()
    print(f"{payload['count']} signals registered:\n")
    for entry in payload["signals"]:
        marker = "*" if entry["default"] else " "
        print(f" {marker} {entry['name']:32.32s} {entry['kind']:13s} {entry['title']}")
        if entry["hypothesis"]:
            print(f"   {'':32.32s} tests: {entry['hypothesis']}")
    print("\n(* = included in the default batch)")
    return 0


def _fmt_pct(value) -> str:
    return "—" if value is None else f"{value * 100:+.2f}%"


def _fmt_p(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_sector_table(payload: dict) -> str:
    lines = [
        f"{'rule':30s} {'n':>5s} {'dates':>5s} {'hit':>7s} {'base':>7s} {'excess':>8s} "
        f"{'mean exc':>9s} {'p':>7s} {'q':>7s}  verdict",
        "-" * 112,
    ]
    for m in payload["measurements"]:
        hit = "—" if m["hit_rate"] is None else f"{m['hit_rate'] * 100:.1f}%"
        base = "—" if m["base_rate"] is None else f"{m['base_rate'] * 100:.1f}%"
        exc = "—" if m["excess_hit"] is None else f"{m['excess_hit'] * 100:+.1f}"
        lines.append(
            f"{m['signal']:30.30s} {m['events']:>5d} {m['event_dates']:>5d} {hit:>7s} {base:>7s} "
            f"{exc:>8s} {_fmt_pct(m['mean_excess']):>9s} {_fmt_p(m['p_value_mean']):>7s} "
            f"{_fmt_p(m['q_value_mean']):>7s}  {m['verdict']}"
        )
    lines += ["", "rank quality (cross-sectional IC per date):"]
    for name, ic in payload["rank_quality"].items():
        mean = "—" if ic["mean_ic"] is None else f"{ic['mean_ic']:+.3f}"
        t = "—" if ic["t_stat"] is None else f"{ic['t_stat']:+.2f}"
        lines.append(f"  {name:20s} mean IC {mean}  t {t}  dates {ic['dates']:>4d}  {ic['verdict']}")
    lines += ["", "current sector labels (latest session):"]
    for row in payload["current"][:8]:
        claim = row["rule_for_label"] or "—"
        lines.append(
            f"  {row['sector']:32.32s} {row['status'] or '—':9s} score {row['momentum_score']:>5} "
            f"20d {row['ret_20d_pct'] if row['ret_20d_pct'] is not None else '—'}%  "
            f"rule {claim} → {row['rule_verdict_over_history'] or '—'} (full history)"
        )
    return "\n".join(lines)


def cmd_sectors(args: argparse.Namespace) -> int:
    from tradingview_mcp.core.data.factory import reset_layer
    from tradingview_mcp.core.services.egx_sector_calibration_service import egx_sector_calibration

    reset_layer()
    payload = egx_sector_calibration(
        horizon=args.horizon,
        permutations=args.permutations,
        walk_forward_folds=args.folds,
        round_trip_cost=args.cost,
        lookback_days=args.lookback_days,
    )
    if "error" in payload:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    print(render_sector_table(payload))
    print(f"\n{payload['sessions']} sessions ({payload['window'][0]} -> {payload['window'][1]}) · "
          f"horizon {payload['horizon']} · round-trip cost {payload['round_trip_cost']:.2%} · "
          f"{len(payload['coverage'])} sectors")
    if payload.get("walk_forwards"):
        print("\nwalk-forward (rules with a measurable in-sample edge):")
        for name, wf in payload["walk_forwards"].items():
            print(f"  {name:32.32s} {wf['verdict']:20s} retention={_fmt(wf['retention'])}")
    for warning in payload["warnings"]:
        print(f"warning: {warning}")
    print(f"\n{payload['disclaimer']}")
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=os.environ.get("EGX_DATA_DB", "egx_data.db"))
    parser.add_argument("--signals", default="", help="comma-separated signal names (default: the standard batch)")
    parser.add_argument("--horizon", type=int, default=5, help="sessions held after the signal")
    parser.add_argument("--symbols", default="", help="explicit EGX tickers")
    parser.add_argument("--index", default="", help="EGX30 | EGX70 | EGX100 | SHARIAH33 | EGX35LV | TAMAYUZ")
    parser.add_argument("--lookback-days", type=int, default=1500)
    parser.add_argument("--max-symbols", type=int, default=120)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--json", dest="json_out", default="", help="write the raw payload here")
    parser.add_argument("--markdown", dest="markdown_out", default="", help="write a readable report here")
    parser.add_argument("--list", action="store_true", help="list the registered signals and exit")
    parser.add_argument("--sectors", action="store_true",
                        help="calibrate the sector scanner's Hot/Cold labels and ranking instead")
    parser.add_argument("--cost", type=float, default=0.0,
                        help="round-trip cost as a fraction, applied per sector event (default 0 = gross)")
    args = parser.parse_args(argv)

    if args.list:
        return cmd_list()

    os.environ["EGX_DATA_DB"] = args.db
    if args.sectors:
        return cmd_sectors(args)
    _, _, calibrate = _load()

    payload = calibrate(
        signal=args.signals,
        horizon=args.horizon,
        symbols=args.symbols,
        index=args.index,
        lookback_days=args.lookback_days,
        max_symbols=args.max_symbols,
        permutations=args.permutations,
        walk_forward_folds=args.folds,
    )

    if "error" in payload:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    print(render_table(payload))
    scope = payload["measurement"]
    print(
        f"\n{scope['symbols_analysed']} symbols · {scope['sessions']} sessions "
        f"({scope['window']['from']} → {scope['window']['to']}) · horizon {payload['horizon']}"
    )
    if payload.get("warnings"):
        for warning in payload["warnings"]:
            print(f"warning: {warning}")
    if payload.get("walk_forward"):
        print("\nwalk-forward:")
        for name, wf in payload["walk_forward"].items():
            print(f"  {name:30.30s} {wf['verdict']:20s} retention={_fmt(wf['retention'])}")
    print(f"\n{payload['disclaimer']}")

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {path}")
    if args.markdown_out:
        path = Path(args.markdown_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(payload), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
