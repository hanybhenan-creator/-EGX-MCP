"""Sector calibration: does the sector scanner's ranking predict anything?

The live sector scanner labels every sector Hot / Warming / Cooling / Cold and
writes "Money rotating INTO <sector>" for the Hot ones. That sentence asserts a
causal fact about money flow from a single day's snapshot. This module replaces
the assertion with a measurement over stored history.

Design, in the same spirit as the stock-level engine:

1. **Panel.** Every session, each sector's members (≥ ``min_members`` with data
   that day) are averaged into the scanner's own features: average intraday
   change, breadth, net volume flow against each member's 20-session average,
   and mean Wilder RSI. The momentum score and status come from
   ``core.analysis.sector_rotation`` — the same functions the live scanner runs.

2. **Outcome.** A sector's forward return enters at the next session's open and
   exits at the close ``horizon`` sessions later, the same convention as the
   stock engine. Its *excess* is that return minus the mean sector return on the
   same dates. Excess removes the market move, so a rising tide cannot make a
   sector look like a rotation.

3. **Events.** Each measurement is a rule that selects (session, sector) pairs:
   the scanner's Hot and Cold labels, and the top and bottom third of three
   continuous scores on each date. Top-third events are expected to outperform,
   bottom-third events to underperform.

4. **Inference.** The null shuffles each sector's excess series in blocks,
   keeps the selected events in place, and recomputes the statistic. Because the
   excess is market-neutral per date, reordering it within a sector tests timing.
   Caveat: shuffling each sector separately breaks the cross-sectional zero-sum
   of excess returns that holds on every date, so the null is only approximate
   and the p-values should be read as approximate. Multiple tests are corrected
   with Benjamini–Hochberg across the batch.

5. **Stability.** Walk-forward folds across the session axis, with the same
   gating and retention verdicts as the stock engine.

6. **Rank quality.** For each continuous score, the information coefficient (IC)
   — the cross-sectional Spearman correlation with forward return, per date —
   summarised by its mean, its sign consistency and a t-statistic that accounts
   for overlapping forward windows.

Deliberate choices, stated because each one changes a number:

* ``min_members`` (default 3). A sector of one or two stocks is one stock's
  noise. The live scanner accepts any count; the calibration does not.
* Excess is computed only on dates with at least ``MIN_SECTORS_PER_DATE``
  sectors observed, otherwise the "mean sector" is not a market.
* ``round_trip_cost`` is a parameter (default 0). Net results subtract it per
  event. It is not guessed here: EGX commission and stamp-duty figures should
  come from the user's broker, not from this file.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from tradingview_mcp.core.analysis.sector_rotation import (
    change_rank_percentiles,
    classify_status,
    momentum_score,
    wilder_rsi_series,
)
from tradingview_mcp.core.calibration import stats
from tradingview_mcp.core.calibration.study import (
    MIN_SESSIONS_PER_FOLD,
    Fold,
    Measurement,
    WalkForward,
    apply_fdr,
    call_it,
    retention_verdict,
)
from tradingview_mcp.core.data.egx_sectors import get_sector

MIN_MEMBERS = 3
MIN_SECTORS_PER_DATE = 5
MIN_EVENTS = 30
MIN_TEST_EVENTS = 10
MIN_TRAIN_EXCESS = 0.002
IC_MIN_DATES = 30
SCORE_NAMES = ("legacy_momentum", "momentum_20d", "reversal_5d")
UNCLASSIFIED = {"", "other"}


# ── panel ───────────────────────────────────────────────────────────────────


@dataclass
class _SymbolSeries:
    close: list[Optional[float]]
    open: list[Optional[float]]
    volume: list[Optional[float]]
    rsi: list[Optional[float]]
    volume_sma20: list[Optional[float]]


def _symbol_series(bars: Sequence[Mapping[str, Any]], position: Mapping[str, int], n: int) -> _SymbolSeries:
    close: list[Optional[float]] = [None] * n
    open_: list[Optional[float]] = [None] * n
    volume: list[Optional[float]] = [None] * n
    present: list[tuple[int, Mapping[str, Any]]] = []
    for bar in bars:
        idx = position.get(str(bar.get("date")))
        if idx is None:
            continue
        c, o, v = bar.get("close"), bar.get("open"), bar.get("volume")
        close[idx] = float(c) if c is not None else None
        open_[idx] = float(o) if o is not None else None
        volume[idx] = float(v) if v is not None else None
        present.append((idx, bar))

    # RSI and the 20-session volume average are computed on the symbol's own
    # sessions in order, then placed back on the shared axis.
    rsi: list[Optional[float]] = [None] * n
    vol_sma: list[Optional[float]] = [None] * n
    seq_idx = [i for i in range(n) if close[i] is not None]
    rsi_seq = wilder_rsi_series([close[i] for i in seq_idx])  # type: ignore[misc]
    for i, value in zip(seq_idx, rsi_seq):
        rsi[i] = value
    seq_vol = [(i, volume[i]) for i in range(n) if volume[i] is not None]
    for k in range(19, len(seq_vol)):
        window = [v for _, v in seq_vol[k - 19 : k + 1]]
        vol_sma[seq_vol[k][0]] = sum(window) / 20.0  # type: ignore[arg-type]
    return _SymbolSeries(close, open_, volume, rsi, vol_sma)


@dataclass
class SectorPanel:
    axis: list[str]
    horizon: int
    members: dict[str, list[str]]
    eligible_dates: dict[str, int]
    scores: dict[str, dict[str, list[Optional[float]]]]  # score -> sector -> series
    status: dict[str, list[Optional[str]]]
    fwd: dict[str, list[Optional[float]]]
    excess: dict[str, list[Optional[float]]]

    @property
    def sectors(self) -> list[str]:
        return sorted(s for s, series in self.excess.items() if any(v is not None for v in series))


def build_panel(
    bars_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    sector_of: Callable[[str], str] = get_sector,
    horizon: int = 5,
    min_members: int = MIN_MEMBERS,
) -> SectorPanel:
    if horizon < 1:
        raise ValueError("horizon must be at least 1 session")
    axis = sorted({str(b["date"]) for bars in bars_by_symbol.values() for b in bars if b.get("date")})
    n = len(axis)
    position = {d: i for i, d in enumerate(axis)}

    members: dict[str, list[str]] = {}
    series: dict[str, _SymbolSeries] = {}
    for sym, bars in bars_by_symbol.items():
        sector = sector_of(sym)
        if sector in UNCLASSIFIED:
            continue
        members.setdefault(sector, []).append(sym)
        series[sym] = _symbol_series(bars, position, n)

    sectors = sorted(members)
    scores: dict[str, dict[str, list[Optional[float]]]] = {
        name: {s: [None] * n for s in sectors} for name in SCORE_NAMES
    }
    status: dict[str, list[Optional[str]]] = {s: [None] * n for s in sectors}
    fwd: dict[str, list[Optional[float]]] = {s: [None] * n for s in sectors}
    eligible = {s: 0 for s in sectors}

    for t in range(n):
        # ── the scanner's daily snapshot, per sector ──
        day: dict[str, dict[str, float]] = {}
        for sector in sectors:
            rows = []
            for sym in members[sector]:
                sr = series[sym]
                c, o = sr.close[t], sr.open[t]
                if c is None or o is None or o == 0:
                    continue
                rows.append((sym, ((c - o) / o) * 100.0))
            if len(rows) < min_members:
                continue
            eligible[sector] += 1
            changes = [chg for _, chg in rows]
            avg_change = round(sum(changes) / len(changes), 2)
            advancing = sum(1 for chg in changes if chg > 0)
            breadth = round(advancing / len(rows) * 100, 1)
            rsis = [series[sym].rsi[t] for sym, _ in rows if series[sym].rsi[t] is not None]
            avg_rsi = round(sum(rsis) / len(rsis), 2) if rsis else 50.0  # type: ignore[arg-type]
            flow = 0.0
            for sym, _ in rows:
                v = series[sym].volume[t]
                if v is None:
                    continue
                flow += v - (series[sym].volume_sma20[t] or 0.0)
            day[sector] = {
                "avg_change": avg_change,
                "breadth": breadth,
                "avg_rsi": avg_rsi,
                "flow_positive": flow > 0,
            }

        if day:
            ranks = change_rank_percentiles({s: f["avg_change"] for s, f in day.items()})
            for sector, f in day.items():
                momentum = momentum_score(
                    f["avg_change"], f["avg_rsi"], f["breadth"], f["flow_positive"], ranks[sector]
                )
                scores["legacy_momentum"][sector][t] = float(momentum)
                status[sector][t] = classify_status(momentum, f["avg_change"], f["breadth"], f["flow_positive"])

        # ── trailing and forward returns (sector mean over members with data) ──
        for sector in sectors:
            if t >= 20:
                scores["momentum_20d"][sector][t] = _mean_member_return(sector, members, series, t, 20, min_members)
            if t >= 5:
                ret5 = _mean_member_return(sector, members, series, t, 5, min_members)
                scores["reversal_5d"][sector][t] = None if ret5 is None else -ret5
            if t + 1 < n and t + horizon < n:
                fwd[sector][t] = _forward_return(sector, members, series, t, horizon, min_members)

    # ── excess: subtract the same-date mean across sectors ──
    excess: dict[str, list[Optional[float]]] = {s: [None] * n for s in sectors}
    for t in range(n):
        values = {s: fwd[s][t] for s in sectors if fwd[s][t] is not None}
        if len(values) < MIN_SECTORS_PER_DATE:
            continue
        mean_fwd = sum(values.values()) / len(values)
        for s, v in values.items():
            excess[s][t] = v - mean_fwd

    return SectorPanel(
        axis=axis,
        horizon=horizon,
        members=members,
        eligible_dates=eligible,
        scores=scores,
        status=status,
        fwd=fwd,
        excess=excess,
    )


def _mean_member_return(
    sector: str,
    members: Mapping[str, list[str]],
    series: Mapping[str, _SymbolSeries],
    t: int,
    lookback: int,
    min_members: int,
) -> Optional[float]:
    rets = []
    for sym in members[sector]:
        close = series[sym].close
        if t - lookback < 0 or close[t] is None or close[t - lookback] in (None, 0):
            continue
        rets.append(close[t] / close[t - lookback] - 1.0)  # type: ignore[operator]
    return sum(rets) / len(rets) if len(rets) >= min_members else None


def _forward_return(
    sector: str,
    members: Mapping[str, list[str]],
    series: Mapping[str, _SymbolSeries],
    t: int,
    horizon: int,
    min_members: int,
) -> Optional[float]:
    rets = []
    for sym in members[sector]:
        sr = series[sym]
        entry, exit_ = sr.open[t + 1], sr.close[t + horizon]
        if entry in (None, 0) or exit_ is None:
            continue
        rets.append(exit_ / entry - 1.0)  # type: ignore[operator]
    return sum(rets) / len(rets) if len(rets) >= min_members else None


# ── events and measurement ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Event:
    t: int
    sector: str
    sign: int  # +1: expected to outperform, -1: expected to underperform


@dataclass(frozen=True)
class Rule:
    name: str
    title: str
    hypothesis: str
    sign: int


def _rules() -> list[Rule]:
    rules = [
        Rule(
            "legacy_hot",
            "Scanner 'Hot' sectors outperform",
            "Sectors the live scanner labels Hot (momentum ≥ 65 with net volume inflow) beat the "
            "average sector over the next horizon.",
            +1,
        ),
        Rule(
            "legacy_cold",
            "Scanner 'Cold' sectors underperform",
            "Sectors the live scanner labels Cold trail the average sector over the next horizon.",
            -1,
        ),
    ]
    descriptions = {
        "legacy_momentum": "the scanner's 0-100 momentum score",
        "momentum_20d": "the trailing 20-session sector return",
        "reversal_5d": "the negated trailing 5-session sector return (short-term reversal)",
    }
    for score, text in descriptions.items():
        rules.append(
            Rule(
                f"{score}_top_third",
                f"Top third by {score} outperforms",
                f"On each date, the third of sectors ranked highest by {text} beat the average sector.",
                +1,
            )
        )
        rules.append(
            Rule(
                f"{score}_bottom_third",
                f"Bottom third by {score} underperforms",
                f"On each date, the third of sectors ranked lowest by {text} trail the average sector.",
                -1,
            )
        )
    return rules


RULES = {rule.name: rule for rule in _rules()}


def select_events(panel: SectorPanel, rule: Rule) -> list[Event]:
    """The (session, sector) pairs a rule selects, with the sign it is judged by."""
    out: list[Event] = []
    if rule.name in ("legacy_hot", "legacy_cold"):
        label = "Hot" if rule.name == "legacy_hot" else "Cold"
        for t in range(len(panel.axis)):
            for s in panel.sectors:
                if panel.status[s][t] == label and panel.excess[s][t] is not None:
                    out.append(Event(t, s, rule.sign))
        return out

    score_name = rule.name.removesuffix("_top_third").removesuffix("_bottom_third")
    for t in range(len(panel.axis)):
        candidates = [
            (panel.scores[score_name][s][t], s)
            for s in panel.sectors
            if panel.scores[score_name][s][t] is not None and panel.excess[s][t] is not None
        ]
        if len(candidates) < 3:
            continue
        k = max(1, len(candidates) // 3)
        ordered = sorted(candidates, key=lambda c: (c[0], c[1]))
        chosen = ordered[-k:] if rule.sign > 0 else ordered[:k]
        out.extend(Event(t, s, rule.sign) for _, s in chosen)
    return out


def _net(panel: SectorPanel, e: Event, cost: float) -> Optional[float]:
    """The event's signed excess net of cost: positive means the call was right."""
    value = panel.excess[e.sector][e.t]
    return None if value is None else e.sign * value - cost


def _base_rate(panel: SectorPanel, sign: int, cost: float) -> Optional[float]:
    hits = total = 0
    for s in panel.sectors:
        for value in panel.excess[s]:
            if value is None:
                continue
            total += 1
            hits += sign * value - cost > 0
    return hits / total if total else None


def _shuffled_excess(
    panel: SectorPanel, *, block: int, permutations: int, seed: int
) -> list[dict[str, list[Optional[float]]]]:
    """One block-shuffled copy of every sector's excess series per permutation."""
    rng = random.Random(seed)
    null_panels = []
    for _ in range(permutations):  # permutations=0 → no null, so no p-values
        shuffled: dict[str, list[Optional[float]]] = {}
        for s in panel.sectors:
            series = panel.excess[s]
            defined = [v for v in series if v is not None]
            permuted = iter(stats.block_shuffle(defined, block=block, rng=rng))
            shuffled[s] = [next(permuted) if v is not None else None for v in series]
        null_panels.append(shuffled)
    return null_panels


def measure_rule(
    panel: SectorPanel,
    rule: Rule,
    *,
    cost: float,
    null_panels: Sequence[Mapping[str, list[Optional[float]]]],
) -> Measurement:
    events = select_events(panel, rule)
    m = Measurement(signal=rule.name, horizon=panel.horizon)
    if not events:
        m.verdict = "NO_EVENTS"
        return m

    nets = [_net(panel, e, cost) for e in events]
    nets = [x for x in nets if x is not None]
    if not nets:
        m.verdict = "NO_EVENTS"
        return m
    m.events = len(nets)
    m.symbols = len({e.sector for e in events})
    m.event_dates = len({e.t for e in events})
    hits = sum(1 for x in nets if x > 0)
    m.hit_rate = hits / len(nets)
    m.hit_ci = stats.wilson_interval(hits, len(nets))
    m.base_rate = _base_rate(panel, rule.sign, cost)
    m.excess_hit = None if m.base_rate is None else m.hit_rate - m.base_rate
    raw = [panel.fwd[e.sector][e.t] for e in events if panel.fwd[e.sector][e.t] is not None]
    m.mean_return = stats.mean(raw)
    m.mean_excess = stats.mean(nets)
    m.median_excess = stats.median(nets)
    m.excess_stdev = stats.stdev(nets)
    m.first_event = panel.axis[min(e.t for e in events)]
    m.last_event = panel.axis[max(e.t for e in events)]

    m.p_mean, m.p_hit = _event_pvalues(panel, rule, events, cost, null_panels, m.mean_excess, m.hit_rate)
    m.best_symbols, m.worst_symbols = _sector_breakdown(panel, events, cost)
    return m


def _event_pvalues(
    panel: SectorPanel,
    rule: Rule,
    events: Sequence[Event],
    cost: float,
    null_panels: Sequence[Mapping[str, list[Optional[float]]]],
    obs_mean: Optional[float],
    obs_hit: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    if obs_mean is None or obs_hit is None or not null_panels:
        return None, None
    at_least_mean = at_least_hit = 0
    for shuffled in null_panels:
        xs = []
        for e in events:
            value = shuffled[e.sector][e.t]
            if value is not None:
                xs.append(e.sign * value - cost)
        if not xs:
            continue
        if sum(xs) / len(xs) >= obs_mean:
            at_least_mean += 1
        if sum(1 for x in xs if x > 0) / len(xs) >= obs_hit:
            at_least_hit += 1
    total = len(null_panels)
    return (1 + at_least_mean) / (1 + total), (1 + at_least_hit) / (1 + total)


def _sector_breakdown(panel: SectorPanel, events: Sequence[Event], cost: float) -> tuple[list[dict], list[dict]]:
    by_sector: dict[str, list[float]] = {}
    for e in events:
        x = _net(panel, e, cost)
        if x is not None:
            by_sector.setdefault(e.sector, []).append(x)
    rows = [
        {"symbol": s, "events": len(xs), "mean_excess": round(sum(xs) / len(xs), 4)}
        for s, xs in by_sector.items()
    ]
    rows.sort(key=lambda r: (-r["mean_excess"], r["symbol"]))
    return rows[:3], list(reversed(rows[-3:]))


# ── walk-forward ────────────────────────────────────────────────────────────


def _fold_measurement(panel: SectorPanel, events: Sequence[Event], cost: float, name: str) -> Measurement:
    m = Measurement(signal=name, horizon=panel.horizon)
    xs = [x for x in (_net(panel, e, cost) for e in events) if x is not None]
    if not xs:
        return m
    m.events = len(xs)
    m.mean_excess = stats.mean(xs)
    m.excess_stdev = stats.stdev(xs)
    m.hit_rate = sum(1 for x in xs if x > 0) / len(xs)
    return m


def sector_walk_forward(
    panel: SectorPanel,
    rule: Rule,
    *,
    cost: float,
    folds: int = 3,
    min_train_events: int = 30,
    min_test_events: int = 10,
) -> WalkForward:
    if folds < 2:
        raise ValueError("walk-forward needs at least 2 folds")
    n = len(panel.axis)
    if n < folds * MIN_SESSIONS_PER_FOLD:
        return WalkForward(rule.name, panel.horizon, [], None, "INSUFFICIENT_HISTORY",
                           f"only {n} sessions; {folds} folds need at least {folds * MIN_SESSIONS_PER_FOLD}")
    events = select_events(panel, rule)
    bounds = [min(n - 1, int(n * k / (folds + 1))) for k in range(folds + 2)]
    bounds[-1] = n - 1
    out: list[Fold] = []
    usable_train: list[float] = []
    usable_test: list[float] = []
    for k in range(1, folds + 1):
        lo, hi = bounds[k], bounds[k + 1]
        train_ev = [e for e in events if e.t + panel.horizon < lo]
        test_ev = [e for e in events if lo <= e.t and e.t + panel.horizon < hi]
        train = _fold_measurement(panel, train_ev, cost, rule.name)
        test = _fold_measurement(panel, test_ev, cost, rule.name)
        out.append(
            Fold(
                index=k,
                train_start=panel.axis[0],
                train_end=panel.axis[lo],
                test_start=panel.axis[lo],
                test_end=panel.axis[hi],
                train=train,
                test=test,
            )
        )
        if train.events < min_train_events or test.events < min_test_events or train.mean_excess is None:
            continue
        se = (train.excess_stdev or 0.0) / math.sqrt(max(1, train.events))
        if train.mean_excess >= MIN_TRAIN_EXCESS and train.mean_excess >= se:
            usable_train.append(train.mean_excess)
            usable_test.append(test.mean_excess or 0.0)

    if not usable_train:
        return WalkForward(rule.name, panel.horizon, out, None, "NO_TRAIN_EDGE",
                           "no fold had enough in-sample events with a mean excess above both "
                           f"{MIN_TRAIN_EXCESS:.1%} and one standard error")
    retention = sum(usable_test) / sum(usable_train)
    verdict, reason = retention_verdict(retention)
    return WalkForward(rule.name, panel.horizon, out, retention, verdict, reason)


# ── rank quality ────────────────────────────────────────────────────────────


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy)


def information_coefficient(panel: SectorPanel, score: str) -> dict[str, Any]:
    ics: list[float] = []
    for t in range(len(panel.axis)):
        pairs = [
            (panel.scores[score][s][t], panel.fwd[s][t])
            for s in panel.sectors
            if panel.scores[score][s][t] is not None and panel.fwd[s][t] is not None
        ]
        if len(pairs) < MIN_SECTORS_PER_DATE:
            continue
        ic = spearman([p[0] for p in pairs], [p[1] for p in pairs])  # type: ignore[misc]
        if ic is not None:
            ics.append(ic)
    n = len(ics)
    result: dict[str, Any] = {"dates": n, "mean_ic": None, "t_stat": None, "sign_consistency": None, "verdict": "INSUFFICIENT_DATA"}
    if n < IC_MIN_DATES:
        return result
    mean = sum(ics) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in ics) / (n - 1))
    # Forward windows of length `horizon` overlap, so consecutive IC values are
    # correlated; the effective number of independent observations is about n / horizon.
    effective = max(1.0, n / panel.horizon)
    t_stat = mean / (sd / math.sqrt(effective)) if sd > 0 else None
    result.update(
        {
            "mean_ic": round(mean, 4),
            "t_stat": None if t_stat is None else round(t_stat, 3),
            "sign_consistency": round(sum(1 for x in ics if x > 0) / n, 3),
            "verdict": (
                "PREDICTIVE" if t_stat is not None and t_stat >= 2
                else "INVERTED" if t_stat is not None and t_stat <= -2
                else "NOT_DISTINGUISHABLE_FROM_ZERO"
            ),
        }
    )
    return result


# ── report ──────────────────────────────────────────────────────────────────

METHOD = (
    "Sector panel from the stored daily history: each sector's members are averaged into the live "
    "scanner's own features; the momentum score and Hot/Warming/Cooling/Cold labels are computed by "
    "the same functions the scanner runs. Outcomes are forward sector returns (entry at the next open, "
    "exit at the close horizon sessions later) minus the same-date mean across sectors. Measurements "
    "are rule-selected events; significance comes from block-shuffling each sector's excess series with "
    "events held in place, corrected across the batch by Benjamini–Hochberg. Walk-forward folds follow "
    "the stock engine's gating and verdicts. Caveat: each sector is shuffled independently, so the "
    "cross-sectional zero-sum of excess returns is not preserved in the null; p-values are approximate, "
    "and a walk-forward pass on its own is not a verdict."
)
DISCLAIMER = (
    "Measured on stored history. A sector that outperformed in the sample is a finding about the sample, "
    "not a forecast. Sector membership is a manual classification and is not point-in-time."
)


@dataclass
class SectorReport:
    horizon: int
    axis_window: tuple[Optional[str], Optional[str]]
    sessions: int
    min_members: int
    round_trip_cost: float
    permutations: int
    coverage: dict[str, dict[str, int]]
    measurements: list[Measurement]
    walk_forwards: dict[str, WalkForward]
    ic: dict[str, dict[str, Any]]
    current: list[dict[str, Any]]
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        verdicts = {m.signal: m.verdict for m in self.measurements}
        rows = []
        for m in self.measurements:
            d = m.as_dict()
            d["title"] = RULES[m.signal].title if m.signal in RULES else m.signal
            d["hypothesis"] = RULES[m.signal].hypothesis if m.signal in RULES else ""
            d["kind"] = "sector_rule"
            d["sign"] = "outperform" if RULES[m.signal].sign > 0 else "underperform"
            rows.append(d)
        return {
            "horizon": self.horizon,
            "window": list(self.axis_window),
            "sessions": self.sessions,
            "min_members": self.min_members,
            "round_trip_cost": self.round_trip_cost,
            "permutations": self.permutations,
            "method": METHOD,
            "disclaimer": DISCLAIMER,
            "coverage": self.coverage,
            "measurements": rows,
            "walk_forwards": {k: v.as_dict() for k, v in self.walk_forwards.items()},
            "rank_quality": self.ic,
            "current": self.current,
            "validated_rules": sorted(k for k, v in verdicts.items() if v == "VALIDATED"),
            "warnings": self.warnings,
        }


def analyze_sectors(
    bars_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    sector_of: Callable[[str], str] = get_sector,
    horizon: int = 5,
    min_members: int = MIN_MEMBERS,
    permutations: int = 300,
    walk_forward_folds: int = 3,
    round_trip_cost: float = 0.0,
    block: int = 10,
    seed: int = 0,
) -> SectorReport:
    if not 1 <= horizon <= 60:
        raise ValueError("horizon must be between 1 and 60 sessions")
    if not 0.0 <= round_trip_cost < 0.2:
        raise ValueError("round_trip_cost must be a fraction in [0, 0.2)")
    if min_members < 2:
        raise ValueError("min_members must be at least 2")

    panel = build_panel(bars_by_symbol, sector_of=sector_of, horizon=horizon, min_members=min_members)
    n = len(panel.axis)
    warnings: list[str] = []
    if not panel.sectors:
        warnings.append(
            f"no sector had {min_members}+ members with data on any session; "
            "run the daily sync for the sector members first"
        )
    excluded = sorted(s for s in panel.members if panel.eligible_dates.get(s, 0) == 0)
    if excluded:
        warnings.append(f"sectors never reaching {min_members} members with data: {', '.join(excluded)}")
    thin = sorted(s for s, c in panel.eligible_dates.items() if 0 < c < max(1, n // 2))
    if thin:
        warnings.append(f"sectors eligible on fewer than half the sessions (membership gaps): {', '.join(thin)}")
    if n < 250:
        warnings.append(f"only {n} sessions: treat every verdict as preliminary")

    coverage = {
        s: {"members": len(panel.members[s]), "eligible_dates": panel.eligible_dates.get(s, 0)}
        for s in sorted(panel.members)
    }

    null_panels = _shuffled_excess(panel, block=block, permutations=permutations, seed=seed)
    if permutations == 0:
        warnings.append(
            "permutation test off (permutations=0): p- and q-values are not computed, so no rule can be VALIDATED"
        )
    measurements = [
        measure_rule(panel, rule, cost=round_trip_cost, null_panels=null_panels) for rule in RULES.values()
    ]
    apply_fdr(measurements, on="hit")
    apply_fdr(measurements, on="mean")
    for m in measurements:
        m.verdict = call_it(m)

    walk_forwards: dict[str, WalkForward] = {}
    if walk_forward_folds:
        for m in measurements:
            if m.verdict in {"VALIDATED", "SUGGESTIVE"}:
                walk_forwards[m.signal] = sector_walk_forward(
                    panel, RULES[m.signal], cost=round_trip_cost, folds=walk_forward_folds
                )

    ic = {score: information_coefficient(panel, score) for score in SCORE_NAMES}

    current = _current_snapshot(panel, measurements)
    return SectorReport(
        horizon=horizon,
        axis_window=(panel.axis[0] if panel.axis else None, panel.axis[-1] if panel.axis else None),
        sessions=n,
        min_members=min_members,
        round_trip_cost=round_trip_cost,
        permutations=permutations,
        coverage=coverage,
        measurements=measurements,
        walk_forwards=walk_forwards,
        ic=ic,
        current=current,
        warnings=warnings,
    )


def _current_snapshot(panel: SectorPanel, measurements: Sequence[Measurement]) -> list[dict[str, Any]]:
    """Latest session: each sector's label, score, and the calibrated verdict of that label."""
    if not panel.axis:
        return []
    t = len(panel.axis) - 1
    verdicts = {m.signal: m.verdict for m in measurements}
    rows = []
    for s in panel.sectors:
        if panel.scores["legacy_momentum"][s][t] is None:
            continue
        label = panel.status[s][t]
        claim = {"Hot": "legacy_hot", "Cold": "legacy_cold"}.get(label or "")
        rows.append(
            {
                "sector": s,
                "as_of": panel.axis[t],
                "status": label,
                "momentum_score": panel.scores["legacy_momentum"][s][t],
                "ret_20d_pct": None
                if panel.scores["momentum_20d"][s][t] is None
                else round(panel.scores["momentum_20d"][s][t] * 100, 2),  # type: ignore[operator]
                "members_today": len(panel.members[s]),
                "rule_for_label": claim,
                "rule_verdict_over_history": verdicts.get(claim) if claim else None,
            }
        )
    rows.sort(key=lambda r: -(r["momentum_score"] or 0))
    return rows
