# AGENTS.md — rules for working in this repository

This is a fork of `atilaahmettaner/tradingview-mcp`, extended with an EGX (Egyptian Exchange)
analysis layer. It is an MCP server: its consumers are language models (for example inside
OpenCode) that have no project context. Read this file before changing anything.

## Scope rules (agreed with the maintainer — do not change without asking)

- **EGX only** for new analysis tools. Upstream tools (crypto, futures, US stocks) are left alone.
- **Free sources only:** EGX official website, Yahoo Finance, public Mubasher / TradingView pages.
  No paid or broker APIs. Excluded: egxapi.com, egxpy, any order-routing or broker API.
- **OHLCV is the base data.** Order book, depth, tick footprints, block trades, insider and
  institutional holdings are deferred until a verified free source exists. Do not fake them.
- **Output is a measured state, not a trade instruction.** No BUY/SELL verdicts, no
  position sizing, no share counts, no "STRONG BUY". Do not label a score as a probability
  unless it has been calibrated on out-of-sample data.
- **Calibration is required** for any signal that is presented as predictive: hit rate with
  base rate, p-value (block permutation), q-value (Benjamini–Hochberg), walk-forward check.
- **Licenses: permissive only (MIT, BSD, Apache-2.0).** No GPL/AGPL. Verify a license on the
  project's own page before adding a dependency; "Other" or missing licenses are rejected.
- **Plain English names.** Tool names, parameters and descriptions must be understandable
  without project context (e.g. `egx_event_study`, not internal jargon). Every statistical
  response carries a `glossary` field.
- **Arabic documentation** is kept for the maintainer (`EGX_*.md` files, README). Tool
  descriptions in code are English.

## Architecture (what exists)

```
src/tradingview_mcp/
  server.py                 MCP tool registry (FastMCP). Thin wrappers only.
  core/data/                EGX data layer: sources, router, store (SQLite), quality, provenance.
  core/analysis/            Pure analysis: sector_rotation.py, footprints.py, wyckoff.py,
                            moneyflow.py (CMF, MFI, VPT, RVOL, anchored VWAP), stock_classes.py
  core/calibration/         Event study engine: signals.py (registry), study.py (hit rate,
                            permutation p, walk-forward, BH), sectors.py, glossary.py,
                            stats.py (Wilson interval and BH via statsmodels/scipy)
  core/services/            Service functions behind each tool (egx_*_service.py)
scripts/                    egx_data_sync.py (fetch/store), egx_calibrate.py (CLI calibration)
tests/unit/, tests/stress/  pytest suites
```

Rules for the layers:
- Analysis code is pure: no network, no database access inside functions that compute numbers.
- Services may read the local store; they must not fetch live data silently.
- Signals are causal: a condition at session t may use bars up to and including t only, and
  the entry is the next session's open.
- A failed or unmeasured component is reported as `UNMEASURED` or `NO_DATA`, never as zero or
  as a healthy value.

## Commands

```bash
# install (dev)
uv sync                       # dev deps (pytest, pytest-asyncio) live in [tool.uv] dev-dependencies

# tests — the full offline suite must pass before any commit
PYTHONPATH=src .venv/bin/python -m pytest tests/unit tests/stress -q

# data (needs network; EGX official endpoints may be unreachable from some networks)
python scripts/egx_data_sync.py probe
python scripts/egx_data_sync.py daily --index EGX30 --days 365

# calibration
python scripts/egx_calibrate.py --list
python scripts/egx_calibrate.py --index EGX30 --horizon 5 --folds 3 --markdown reports/calibration.md
python scripts/egx_calibrate.py --sectors --db <path> --permutations 200 --folds 3

# run the MCP server
tradingview-mcp            # console script (server:main)
```

Always run Python from the repository root with `PYTHONPATH=src` (or inside the venv
after `pip install -e .`). Do not install packages into the system Python.

## Engineering standards

- Do not claim a feature works without running it. Report test results as passed, failed, or
  not tested — never blend them.
- Every new statistic needs a test on synthetic data with a known answer, and a control
  showing it does **not** fire on pure noise.
- Do not tune thresholds on the same data used to report the result. Keep a held-out seed or
  period and say which one was used.
- Do not add dependencies for a single helper. Record the license in the PR description.
- Never commit secrets. `.env` is ignored; `.env.example` must contain no real keys.
- Never delete or overwrite the user's own documentation (`EGX_DATA_STRATEGY.md` is the
  maintainer's original). Add new documents instead.
- Do not add write-capable MCP tools (orders, account changes). Tools are read-only.

## Known limitations to keep visible

- Walk-forward did not reliably reject noise on synthetic event-signal controls. Treat a
  VALIDATED verdict as one piece of evidence.
- Sector membership is a manual classification of EGX stocks.
- Live EGX and Yahoo access was unreachable from the development sandbox; real-data
  validation must be run on a machine with network access.
