# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Sector calibration** (`tradingview_mcp.core.calibration.sectors`, MCP tool
  `egx_sector_calibration`, `python scripts/egx_calibrate.py --sectors`). Replays
  the sector scanner's Hot/Cold labels and its three continuous scores (momentum,
  20-session return, 5-session reversal) over stored history. Excess is measured
  against the same-date average sector, with block-permutation p-values, a
  Benjamini–Hochberg correction across the eight rules, walk-forward folds for
  rules that reach SUGGESTIVE or VALIDATED, and a cross-sectional information
  coefficient per score. The scanner's rules now live in
  `core.analysis.sector_rotation`, shared by the live scanner and the calibration,
  so the two cannot drift apart. The Wilder RSI used by the scanner has a series
  form (`wilder_rsi_series`); it matches a pandas `ewm` reference exactly by about
  session 200, and the earlier sessions differ only through the SMA seed. Round-trip cost is user-supplied (`--cost`, default 0 = gross);
  the tool does not assume a commission or stamp-duty figure. MCP tool count: 44.

  **Known limits (stated, not hidden).** Per-sector shuffling does not preserve
  the cross-sectional zero-sum of excess returns, so p-values are approximate.
  Sector membership is a manual, non-point-in-time classification. The IC t-test
  is parametric and not corrected across scores or dates. Two of 60 IC verdicts
  on noise were PREDICTIVE (3.3%), and walk-forward gave MODERATE or ROBUST on 4
  of 75 noise runs (about 5%), so a walk-forward pass is not a verdict on its own.

  **Power and false positives (synthetic controls, 6 sectors × 4 members, 420
  sessions; not EGX data).** Innovation scale `drift_scale` 0.0005 (latent
  sector drift s.d. ≈ 0.2% per session) is VALIDATED in 3 of 4 seeds. `drift_scale`
  0.00025 (≈ 0.1%) reaches SUGGESTIVE in 2 of 4 seeds and is never VALIDATED.
  `drift_scale` 0.0001 or less is not detected. On noise over 20 seeds, 1 of 160 rule
  verdicts was VALIDATED (0.6%, `momentum_20d_top_third`, seed 116). Test fixtures
  use seeds 1–3 for the noise checks.

  **Fixed in this round:** with `permutations=0` the sector test had been drawing
  one null sample, which fabricated a p-value of 0.5. Now no null is drawn, p and q
  are reported as `—`, and the report warns that no rule can be VALIDATED.
- **Institutional footprint layer** (`tradingview_mcp.core.analysis.footprints`).
  The "volume rises while price stays flat" design from the architecture guide,
  implemented over stored bars with no indicators: volume anomalies against a
  trailing average, accumulation zones (active zones included) with measured
  support/resistance volume, absorbed dips, breakout preparation, shakeout
  reclaims, and the distribution mirror. The composite scores are documented
  as weighted flag sums, not probabilities; their predictive value is what the
  calibration engine measures, and six footprint conditions are registered for
  exactly that purpose.
- **`egx_accumulation_fingerprint`** MCP tool (read-only, offline) and
  `python scripts/egx_data_sync.py fingerprint SYMBOL`. Each reply includes a
  **measured** `data_quality` score (0-100, grade A-F) built from calendar
  coverage, OHLCV integrity, freshness against the newest settled session, and
  the per-source success rate observed on this machine. Every component is
  returned with the score, and an unmeasurable component is excluded and named
  rather than assumed. 43 tools total.

  **Controls (synthetic, seeded — not live EGX data; this sandbox has no
  outbound access to EGX or Yahoo, so no real-market hit rate is claimed).**
  24 symbols × 430 sessions, horizon 5, 150 permutations, 3 folds:

  | run | signal | events | hit vs base | excess | q | verdict | walk-forward |
  |---|---|---|---|---|---|---|---|
  | footprint planted | hidden_accumulation | 467 | 68.1% vs 63.8% | +4.3 pts | 0.050 | VALIDATED | MODERATE (0.65) |
  | footprint planted | volume_footprint | 865 | 67.9% vs 63.8% | +4.0 pts | 0.040 | VALIDATED | MODERATE (0.68) |
  | footprint planted | absorption | 27 | 66.7% vs 63.8% | +2.8 pts | 0.437 | SUGGESTIVE | NO_TRAIN_EDGE |
  | pure noise | hidden_accumulation | 154 | 50.6% vs 53.2% | −2.6 pts | 0.821 | REJECTED | not run |
  | pure noise | volume_footprint | 604 | 52.6% vs 53.2% | −0.6 pts | 0.821 | REJECTED | not run |
  | pure noise | absorption | 23 | 52.2% vs 53.2% | −1.0 pts | 0.821 | REJECTED | not run |

  Walk-forward runs only for signals that come back VALIDATED or SUGGESTIVE, by
  design, so the noise rows have no fold analysis.
- **Signal calibration engine** (`tradingview_mcp.core.calibration`) — the
  answer to "confidence: 87, according to what?". Every smart-money condition is
  re-expressed as a function over time and measured against the stored EGX
  history, then reported with the numbers that make a rate falsifiable:

  - event study with no look-ahead (signal on the close of *t*, entry at the
    open of *t+1*, exit at the close of *t+horizon*);
  - a base rate — what buying a random stored symbol for the same holding
    period would have produced — next to every hit rate;
  - an equal-weighted benchmark over the identical entry/exit dates, so a
    market-wide move cannot masquerade as a signal;
  - p-values from a **block permutation** of each symbol's whole return series
    (blocks preserve momentum and volatility clustering; the statistic is
    recomputed at the same event positions, so the null is about the signal's
    *timing*);
  - **Benjamini–Hochberg** q-values across every signal measured in the run;
  - **walk-forward** folds with a ROBUST / MODERATE / WEAK / OVERFITTED
    verdict, where a fold only votes when its in-sample edge and event counts
    are big enough to be worth retaining;
  - an explicit verdict per signal: VALIDATED (enough events, positive excess,
    q ≤ 0.10) / SUGGESTIVE / REJECTED / NO_EVENTS.

  Stdlib only (no numpy/pandas), so it runs directly over the data layer's
  stored bars; third-party conditions join by registering a `SignalSpec`.
- **New MCP tools** `egx_signal_calibration` (measure conditions against the
  local store — read-only, works offline) and `egx_calibration_signals`
  (catalogue with each signal's hypothesis).
- **`scripts/egx_calibrate.py`** — the same engine from the shell, with
  `--list`, per-signal selection, configurable horizon/permutations/folds, and
  JSON/Markdown report output.
- **Distribution mirrors** of all five conditions (the guide asks for a
  DistributionDetector and only the accumulation side ever shipped), registered
  so they can be measured like everything else.

### Changed
- **Walk-forward retention is now the pooled ratio of mean excess** (out-of-
  sample ÷ in-sample), and a fold votes only when its in-sample mean excess
  clears 0.2% *and* one standard error. The previous per-fold ratio of hit-rate
  excess swung between 0.19 and 0.83 on the same planted-edge data, and a fold
  with a near-zero in-sample edge could dominate the verdict. Walk-forward
  payloads now include `train_mean_excess`, `test_mean_excess` and
  `retention_basis`.
- **Calibration default lookback is 1,500 sessions** (was 700). The volume
  profile needs 120 sessions; a 700-session default on short stored history made
  long-memory conditions report NO_EVENTS for lack of history, not lack of
  signal. Short histories now produce an explicit warning.
- Default calibration batch: 15 signals (9 legacy + 6 footprint), 200
  permutations.

### Fixed
- **Footprint port: four defects found by the test suite before release.**
  Absorption measured volume against the scanned tail instead of the full
  series (so it never fired); the zone strength formula scaled a share by 1000
  and scored a dead flat line at 30/100 (now gated on volume activity); the
  consecutive-days counter reported the anomaly count (now stops at the first
  calendar gap); and fixture generators produced candles violating high ≥
  max(open, close), which the quality scorer correctly flagged as 119 invalid
  bars.
- **The accumulation signal's breakout condition was unsatisfiable.** The
  legacy `is_above_20d_high` compared the current close against the highest
  high of a window that *included* the current bar — and a close can never
  exceed its own session's high. Measured over 4,800 synthetic sessions it was
  true exactly **zero** times, which made the "5/5 conditions" branch of
  `AccumulationSignal` (and every `is_high_conviction()` claim) dead code: the
  maximum reachable condition count was 4. Price-structure levels now come from
  the window *before* the current bar (a genuine breakout), which makes the
  condition fire 405 times over the same 4,800 sessions and lets 5/5 signals
  exist. The original formulation is preserved in the calibration catalogue as
  `c3_legacy` so the difference stays visible. `tests/unit/services/test_smart_money_core.py`
  pins both behaviours.
- **EGX data acquisition & verification layer** (`tradingview_mcp.core.data`).
  The previous EGX data stack was scaffolding: abstract source interfaces with
  no implementation, hard-coded sector/index tables, a key/value SQLite cache,
  and no way to answer "where did this number come from?". This replaces it
  with a layer whose whole purpose is that a number can be defended:

  - `calendar.py` — EGX trading calendar. Sunday→Thursday week, session phases
    (pre-open 09:30, continuous 10:00, closing auction 14:20, post-close 14:30)
    computed in `Africa/Cairo` via `zoneinfo` so Egypt's reinstated DST is
    handled by tzdata rather than by hand-rolled offsets. Holidays are data
    (`egx_holidays.json`) with a per-year `verified` flag: an un-synced year is
    *reported*, never silently treated as "no holidays".
  - `provenance.py` — `Provenance`, `Sourced[T]` and `QualityFlag`. Every value
    carries source, fetch time, latency, attempt index and validation verdicts.
  - `transport.py` — injectable `Transport` protocol; `HttpxTransport` with
    explicit timeouts, exponential backoff + jitter, `Retry-After` support and
    per-host token-bucket pacing (default 0.5 rps on `egx.com.eg`); plus
    `FakeTransport` so every parser test runs without a network.
  - `html_tables.py` — stdlib-only HTML table reader. Correctly expands
    `rowspan`/`colspan` so a two-level header cannot shift columns (the failure
    mode where volume is read as price), and recovers JSON embedded in
    `<script>` blocks as a fallback.
  - `parsing.py` — refuses to guess: Arabic-Indic digits, ٬/٫ separators,
    `(12.5)` negatives, `1.2M` suffixes, and the genuine `1.234` ambiguity.
  - `quality.py` — three layers of checks: hard structural validity, plausibility
    flags (`OUTLIER`) and cross-source triangulation
    (`AGREEMENT`/`DIVERGENCE`/`CONFLICT`).
  - `sources/` — `EGXOfficialSource` (multi-strategy parsing, ordered endpoint
    candidates with the working one remembered, EN/AR column synonyms, and a
    `ParseReport` that distinguishes "market closed" from "our selector broke");
    `YahooFinanceSource` (`.CA` chart API, null candles dropped rather than
    zeroed); `TradingViewSource` (delegates to the existing screener path
    instead of building a second upstream client).
  - `router.py` — capability-oriented fallback chain with official-first
    ordering, quality gating, optional corroboration and a per-source circuit
    breaker; `AllSourcesFailed` carries every attempt and its reason.
  - `store.py` — schema-versioned SQLite (migrations, not `CREATE TABLE IF NOT
    EXISTS` drift) with `ohlcv_observations` keeping *every* source's version of
    a bar, `fetch_audit`, `ingest_runs` and `quality_events`; `ohlcv_gaps()`
    measures completeness against the trading calendar.
  - `ingest.py` — refreshes as transactions with reports: rows written, symbols
    skipped (with the reason), per-symbol failures, and coverage measured to the
    newest *settled* session.
- **New MCP tool `egx_data_status`** — read-only health view: market clock,
  source reachability + breaker state, store contents, calendar coverage,
  measured per-source success rates on this machine, and holiday-sync warnings.
- **`scripts/egx_data_sync.py`** — `status`, `probe` (live endpoint calibration
  for `egx.com.eg`), `daily`, `quotes`, `universe`, `holidays`. Refreshes live
  here rather than behind a tool because every tool in this server declares
  `readOnlyHint=True`, an invariant enforced by `tests/unit/test_tool_annotations.py`.
- 266 hermetic tests for the above (`tests/unit/data/`, `tests/unit/services/`).

### Fixed
- **TradingView interval codes rejected as timeframes**: `"60"`, `"240"`,
  `"15"`, `"5"`, `"1440"`, `"D"` and `"W"` are now accepted and mapped to
  `1h`, `4h`, `15m`, `5m` and `1D` / `1W`. Integrations sending TradingView's
  native codes used to silently get the tool's default timeframe instead
  (a `"60"` request returned 15m data), and since 0.9.0's strict validation
  they got `INVALID_TIMEFRAME`. Unsupported intervals (`30m`, `120`, ...)
  still fail loudly.
- **Futures watchlist returned nothing for the main US index futures** (#98):
  `FUTURES_WATCHLIST` used exchange prefixes TradingView's scanner doesn't
  index (`CME:ES1!` instead of `CME_MINI:ES1!`, likewise NQ, RTY, YM and EMD,
  plus livestock under `CBOT` instead of `CME`). A wrong prefix returns 0
  rows without an error, so `futures_watchlist`, `futures_category_snapshot`
  and `futures_market_overview` were silently empty for those contracts.
- **Stale US stock coinlists** (#96): `nasdaq.txt` and `nyse.txt` hadn't been
  refreshed since 2025. About 1,500 entries were no longer listed on that
  venue, about 1,600 newer listings were missing, and 18 tickers had moved
  venue (WMT, AZN, KHC, ETSY, QBTS, QS and others), which broke error
  suggestions, venue fallback and exchange-level scan coverage. Regenerated
  from the scanner.

### Added
- **India and Indonesia**: NSE, BSE and IDX route to their TradingView markets
  and ship scanner-built coinlists (stocks and depositary receipts) for
  exchange-level scans. The benchmark indices resolve by name whatever exchange
  the caller passes: `NIFTY` / `NIFTY50` / `^NSEI`, `BANKNIFTY` / `^NSEBANK`,
  `SENSEX` / `^BSESN`, and `JKSE` / `^JKSE` / `IHSG` for the Jakarta Composite.
- **European and Canadian stock exchanges** (#95, thanks @Zulimon): Euronext
  Paris, Amsterdam, Brussels and Lisbon (`EPA`, `AMS`, `BRU`, `LIS`), Borsa
  Italiana (`MIL`), London (`LSE`), SIX (`SIX`), Madrid (`BME`), Xetra
  (`XETRA`), Frankfurt (`FWB`), Toronto (`TSX`) and TSX Venture (`TSXV`), plus
  common aliases. Each routes to its TradingView market and symbol prefix and
  ships a scanner-generated coinlist (stocks and depositary receipts) for
  exchange-level scans. Euronext's four markets share one `EURONEXT` prefix,
  so the market follows the venue the caller named. Venue fallback now stays
  in the requested venue's market first, so a ticker that exists on both
  sides of the Atlantic (MC is Moelis on NYSE, LVMH in Paris) doesn't jump
  continents.
- `scripts/refresh_coinlists.py` rebuilds the stock coinlists (US, European
  and Canadian venues) from the scanner. `--check` reports drift without
  writing (exit 1 if stale), and a guard refuses to write a list that shrank
  suspiciously.

## [0.9.0] - 2026-08-26

### Changed (behavior)
- **Strict input validation**: tools now return `INVALID_EXCHANGE` /
  `INVALID_TIMEFRAME` error envelopes (listing valid values) instead of
  silently substituting the default — `exchange="KRAKEN"` used to return
  KUCOIN data with no warning. Aliases (`1d`→`1D`) and omitted parameters
  still resolve silently.
- **`PARTIAL_DATA` envelopes**: batched scans that abort mid-flight
  (wall-clock budget / consecutive-failure bail) return a `PARTIAL_DATA`
  envelope that still carries the collected rows, instead of a plain list
  indistinguishable from a complete scan.

### Fixed
- **`top_losers` returned gainers**: the service sorted descending and
  truncated to `limit` before the tool re-sorted ascending, so it returned
  the smallest of the top gainers. Sorting now happens before truncation.
- **Volume breakout scanner fabricated breakouts**: symbols missing a
  `volume.SMA20` baseline got a fallback ratio that was always exactly 2.0
  and passed the default gate on price change alone. They are now skipped.
- **Walk-forward robustness inverted for losing strategies**: a strategy
  losing MORE out-of-sample scored a capped 2.0 ("maximally robust") in the
  both-negative branch. Warmup-starved test folds are flagged
  `insufficient_data` and excluded instead of reading as OVERFITTED.
- **Multi-timeframe alignment misattribution**: a failed timeframe shifted
  every later score onto the wrong timeframe key in `scores_by_tf`.
- **`fetch_multi_timeframe_patterns` ignored its `symbols` argument**
  (whole-exchange scan) while caching on it.
- **Backtest metrics**: drawdown/Calmar/Sharpe now come from a per-bar
  mark-to-market equity series (intra-trade dips count); open positions at
  data end are force-closed and flagged `forced_exit`; the buy-and-hold
  benchmark pays the same round-trip costs as the strategy;
  `compare_strategies` validates `period`.
- **Thread safety**: the Yahoo options session handshake is lock-guarded;
  the screener cache is bounded (256 entries); cached indicator dicts are
  copied before ATR backfill; the resilience layer's stale-while-error
  cache is actually populated on success for `cache_key` callers.
- **Timeouts everywhere**: all `get_scanner_data` calls pass explicit
  timeouts (`requests` has no default).
- **Docker HEALTHCHECK**: the server now serves `GET /health` under the
  streamable-http transport, so containers stop cycling to `unhealthy`.
- Marketaux sentiment: whole-word keyword scoring and exact entity-symbol
  matching; multi-agent analysis no longer crashes on explicit-null
  indicators; proxy credentials are URL-encoded.

### Removed
- Dead code (~800 lines): the unwired paper-trading `portfolio.py`, the
  RSS `news_service.py` and Reddit `sentiment_service.py` (both replaced by
  Marketaux in 0.6), unused fetchers in `screener_provider.py` /
  `screener_service.py`, and the stray `PR_BODY.md`.

### Infrastructure
- **CI now runs the test suite** (Python 3.10–3.13 matrix) on every
  push/PR, and the Docker image publish is gated on it.
- `pandas` declared as a direct dependency (server.py imports it at module
  top; it previously arrived only via the tradingview-screener pin).
- `docker-compose.yml` points at the GHCR image CI actually builds.
- Real `SECURITY.md` (GitHub Security Advisories; 0.9.x supported).

## [0.8.1] - 2026-08-02

### Fixed
- Raised the `mcp` lower bound to `>=1.14.0` (still `<2`): with
  `from __future__ import annotations`, `Tool.from_function` in <=1.13.x
  dies at import time on string annotations, so the server never started
  on those SDK versions.

### Added
- Official MCP Registry manifest (`server.json`) + OIDC publish workflow.

## [0.8.0] - 2026-07-29

### Fixed
- **CRITICAL — pinned `mcp[cli]>=1.12.0,<2`**: `mcp` 2.0.0 (released 2026-07-28
  alongside the MCP 2026-07-28 stateless spec) removes the `mcp.server.fastmcp`
  module this server imports. 0.7.1's published metadata carries the open
  `>=1.12.0` range, so every fresh `pip install` / `uvx` since 2026-07-28
  resolved mcp 2.0.0 and died on startup with
  `ModuleNotFoundError: No module named 'mcp.server.fastmcp'`. Upgrade to
  0.8.0 to get a working install. Migrating to the 2.x SDK
  (FastMCP → MCPServer) will be a deliberate follow-up release.
- **`volume_confirmation_analysis` bare-symbol failures**: bare crypto symbols
  (no exchange prefix) failed ~99% of the time; symbols are now resolved
  before analysis.
- **Donchian breakout off-by-one**: the breakout strategy read a window that
  included the current bar, inflating backtest results; window is now strictly
  historical (regression-tested).
- **Backtest input validation**: initial capital and cost parameters are now
  validated instead of silently producing nonsense results.
- **`coin_analysis` ATR null bug**: `tradingview_ta` omits the `ATR` column from
  its analysis payload, which left `indicators["ATR"]` (and every downstream
  consumer — stop-loss sizing, trade-quality score, volatility metrics) at
  `None` on every call. `analyze_coin` now falls back to a direct
  `scanner.tradingview.com/<market>/scan` request via the new
  `fetch_atr_for_ticker()` helper in `screener_provider.py`. The lookup is
  best-effort (silent `None` on any network / parse failure) and only fires
  when `tradingview_ta` returned no ATR, so it does not regress healthy
  payloads. Timeframe → resolution mapping is shared with the existing
  screener column logic (`5m→5`, `15m→15`, `1h→60`, `4h→240`, `1D/1W/1M`
  unchanged); unknown timeframes degrade to the unsuffixed `ATR` column.

### Added
- **Markets**: Taiwan (TWSE, TPEX), Saudi Arabia (TADAWUL), US futures across
  CME/COMEX/NYMEX/CBOT (equity index, energy, metals, agriculture, rates, FX,
  crypto futures), AMEX/NYSEARCA aliases, auto-venue fallback for unlisted
  symbols, precious-metal futures resolution via TVC.
- **Tools**: `stock_screener` + `stock_prices` (bulk fetches up to 1,000 rows,
  `exclude_otc`, server-side sorting, daily OHLC), `stock_options_chain` +
  `stock_options_unusual_activity`, `stock_extended_hours`,
  `bitcoin_market_pulse`.
- **Directory-grade tool metadata**: every tool now ships `title`,
  `readOnlyHint`, and an explicit `destructiveHint=False` annotation.
- **Multi-arch Docker image CI**: GHCR images built for linux/amd64 +
  linux/arm64 on every push to main and on version tags.

### Changed
- **Every tool now runs off the event loop** (blanket async offload) — a slow
  upstream call can no longer block the MCP server's event loop.
- **Structured error envelopes everywhere**: screener/scanner failures return
  machine-readable envelopes with retryability signals and symbol
  suggestions instead of bare strings; note that tool return types are now
  `list[dict] | dict` (consumers that assumed a bare list should handle the
  envelope shape).
- **News pipeline**: RSS/Reddit scraping replaced with the licensed Marketaux
  API. Set `MARKETAUX_API_TOKEN` (see `.env.example`) to enable
  `financial_news` and news-driven sentiment; without a token the news tools
  return empty results while everything else works normally.
- **Reliability**: bounded HTTP timeouts with stale-while-error fallback,
  retry + 60s TTL response cache, `tradingview_ta` request throttling,
  fast-fail on upstream outages.
- Python support capped below 3.14 (`requires-python >=3.10,<3.14`) until the
  dependency stack publishes 3.14 wheels.
- `requests` is now an explicit dependency in `pyproject.toml`. It was already
  pulled in transitively by `tradingview-screener` / `tradingview-ta`, but the
  new ATR injection path uses it directly, so it is no longer safe to rely on
  the transitive resolution.

## [0.7.1] - 2026-04-14

### Added
- **MEXC Exchange Support** (`exchange="MEXC"`):
  - 420 MEXC trading pairs added — including many unique pairs not available on Binance, KuCoin, or Bybit
  - Fully supported in all tools: `top_gainers`, `top_losers`, `bollinger_scan`, `coin_analysis`, `multi_agent_analysis`, `volume_breakout_scanner`, `consecutive_candles_scan`, `advanced_candle_pattern`, and more
  - MEXC correctly categorized as a crypto screener market

---

## [0.7.0] - 2026-03-29

### Added
- **Walk-Forward Backtesting** (`walk_forward_backtest_strategy`):
  - Splits data into N folds (train/test) to validate strategy on unseen forward data
  - Per-fold in-sample vs out-of-sample return comparison
  - **Robustness score** (test/train ratio): ROBUST ≥ 0.8 | MODERATE ≥ 0.5 | WEAK ≥ 0.2 | OVERFITTED < 0.2
  - Aggregate out-of-sample metrics: Sharpe, win rate, max drawdown, total return
  - Supports 2–10 splits, configurable train ratio, both 1d and 1h intervals
- **Full Trade Log** (`include_trade_log=True`):
  - Per-trade breakdown: entry/exit date & price, holding days, gross/net return %, cost %
  - Running capital and cumulative return at each trade
- **Equity Curve** (`include_equity_curve=True`):
  - Capital value + drawdown % at each trade exit — ready for charting
- **1h (Hourly) Timeframe** (`interval="1h"`):
  - All strategies and compare now support intraday hourly data
  - Sharpe ratio annualization corrected for 1h bars (252 × 6 trading hours)
  - Works on `backtest_strategy`, `compare_strategies`, and `walk_forward_backtest_strategy`

### Changed
- `backtest_strategy` tool: added `interval`, `include_trade_log`, `include_equity_curve` params
- `compare_strategies` tool: added `interval` param; now documents all 6 strategies (was 4)
- `run_backtest()` now returns last 5 trades always (`recent_trades`) for quick inspection
- Sharpe ratio calculation now uses interval-aware annualization factor

---

## [0.6.0] - 2026-03-29

### Added
- **Backtesting Engine v2** (`backtest_strategy`, `compare_strategies`):
  - 6 trading strategies: RSI, Bollinger Band, MACD, EMA Cross, **Supertrend** (🔥 trending 2025), **Donchian Channel** (Turtle Trader classic)
  - Institutional-grade metrics: Sharpe Ratio, Calmar Ratio, Expectancy, Profit Factor, Max Drawdown
  - Transaction cost simulation: per-trade commission + slippage
  - Buy-and-hold benchmark comparison
  - Single OHLCV fetch for `compare_strategies` (all 6 strategies in ~0.3s)
- **Yahoo Finance Integration** (`yahoo_price`, `market_snapshot`):
  - Real-time quotes for stocks, crypto, ETFs, indices (S&P500, NASDAQ, VIX), FX
  - Global market snapshot with 14 instruments across 4 asset classes
  - Turkish stocks supported (THYAO.IS, SASA.IS...)
- **Webshare Rotating Proxy Manager**:
  - 250 sticky sessions for rate-limit bypass
  - Direct-first + proxy-fallback architecture for reliability
  - Zero-config for users (optional env-based configuration)
- **Technical Indicators (pure Python, zero deps)**:
  - ATR (Average True Range)
  - Supertrend
  - Donchian Channel

### Changed
- `compare_strategies` now fetches OHLCV once and runs all strategies on cached data (5x faster)
- Yahoo Finance data fetching uses direct connection first, proxy fallback only on failure

## [0.5.0] - 2026-03-29

### Added
- **Real-Time Market Sentiment (Agent-Reach Integration)**: Integrated Reddit JSON API to track symbol sentiment across finance communities (`market_sentiment`).
- **Live Financial News RSS**: Added `fetch_news` service via `feedparser` to track real-time headlines across Reuters, CoinDesk, and CoinTelegraph (`financial_news`).
- **Combined Analysis Power Tool**: The new `combined_analysis` tool merges TradingView technicals, Reddit sentiment, and live news into a single confluence analysis (signals agree/conflict, confidence score, full recommendation).
- Added `feedparser` dependency to `pyproject.toml`.

## [0.4.0] - 2026-03-29

### Added
- **EGX (Egyptian Exchange) Full Support**: Complete trading infrastructure for the Egyptian Stock Market.
  - `egx_market_overview`: Top gainers, losers, most active stocks, and market breadth stats (advancing/declining/unchanged).
  - `egx_sector_scan`: Scan across 18 EGX sectors (banks, real_estate, healthcare_and_pharma, technology, etc.).
  - `egx_stock_screener`: Cross-sectional ranking with auto trade plan generation.
  - `egx_trade_plan`: Single-stock detailed trade setup (entry, stop-loss, targets, R:R).
  - `egx_fibonacci_retracement`: Fibonacci retracement + extension levels with golden pocket detection.
  - 6 EGX indices: EGX30, EGX70, EGX100, SHARIAH33, EGX35LV, TAMAYUZ (200+ symbols).
- **3-Layer Stock Decision Engine**:
  - Layer A: 100-point stock ranking model (trend, momentum, risk, fundamentals).
  - Layer B: Trade setup engine (entry points, stop-loss, targets, support/resistance, R:R).
  - Layer C: Trade quality scoring (structure, risk/reward, volume, liquidity).
- **Liquidity-Aware Scoring**: Hard grade caps prevent illiquid stocks from ever receiving "Strong" or "Elite" grades.
- **23 Technical Indicators**: Expanded from 5 → 23: CCI, Williams %R, Awesome Oscillator, Momentum, Parabolic SAR, Ichimoku, Stoch RSI, ADX +DI/-DI, Hull MA, VWMA, Ultimate Oscillator, full EMA/SMA suites.
- **Multi-Timeframe Alignment**: Weekly→Daily→4H→1H→15m bias analysis with timeframe-specific advice.

### Fixed
- Hardcoded `"crypto"` market type in `_fetch_multi_changes`, `_fetch_multi_timeframe_patterns`, and `screener_provider` — now dynamically resolved per exchange (egypt, turkey, america, etc.).
- `volume_confirmation_analysis` was appending `"USDT"` to stock symbols — now exchange-aware with proper `EGX:SYMBOL` formatting.

### Changed
- MCP server name updated to "TradingView Multi-Market Screener".
- All tool descriptions updated to reference both crypto and stock exchanges.
- Added `is_stock_exchange()`, `get_market_type()`, and `STOCK_EXCHANGES` helpers to `validators.py`.

## [0.3.0] - 2026-03-24

### Added
- **Docker Support**: Official Dockerfile and docker-compose.yml for easy 1-click self-hosting.
- **PyPI Release**: Added proper metadata and structuring for PyPI distribution (`pip install tradingview-mcp`).

## [0.2.0] - 2026-03-24

### Added
- **Multi-Agent Trading Framework**: Introduced `multi_agent_analysis` MCP tool.
  - **Technical Analyst Agent**: Analyzes RSI, MACD, and Bollinger Bands.
  - **Sentiment Analyst Agent**: Calculates momentum and produces a sentiment score.
  - **Risk Manager Agent**: Evaluates volatility (BBW) and mean reversion risk.
- **Debate System**: Agents combine their scores to provide a single, logical Framework Decision (Strong Buy, Buy, Hold, Sell, Strong Sell) with confidence levels.

### Changed
- Repositioned the project from a "screener" to an "AI Trading Intelligence Framework".
- Updated `README.md` to reflect the new architecture.

## [0.1.0] - Initial Release
- Basic MCP Server setup.
- Bollinger Band squeeze detection.
- Consecutive candle pattern detection.
- Real-time market screening (gainers, losers).
