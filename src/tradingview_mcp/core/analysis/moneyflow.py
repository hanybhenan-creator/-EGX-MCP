"""Volume and money-flow indicators computed from daily OHLCV only.

Each function takes the bars in time order and returns one value per bar. A value
at index ``i`` uses bars ``0..i`` only (causal), so truncating the series after
``i`` never changes it. Missing inputs are never imputed: a value that depends on
a missing bar is ``None``. A zero denominator is ``None`` unless the formula
defines a value for it (stated per function).

These are measurements of past trading. They are not buy or sell signals; their
predictive value must be measured by the calibration engine before any use.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

Bar = Mapping[str, Any]


def _num(bar: Bar, key: str) -> Optional[float]:
    value = bar.get(key)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # reject NaN


def _hlcv(bar: Bar) -> Optional[tuple[float, float, float, float]]:
    h, l, c, v = (_num(bar, k) for k in ("high", "low", "close", "volume"))
    if None in (h, l, c, v):
        return None
    return h, l, c, v


def _typical(h: float, l: float, c: float) -> float:
    return (h + l + c) / 3.0


def chaikin_money_flow(bars: Sequence[Bar], period: int = 20) -> list[Optional[float]]:
    """Chaikin Money Flow over ``period`` bars, in [-1, 1].

    Money-flow multiplier per bar: ((close - low) - (high - close)) / (high - low).
    CMF = sum(multiplier * volume) / sum(volume) over the window.
    A bar with high == low has multiplier 0 (the standard convention: no range,
    no direction). A window whose total volume is 0 gives None.
    """
    rows = [_hlcv(b) for b in bars]
    out: list[Optional[float]] = []
    for i in range(len(rows)):
        if i + 1 < period:
            out.append(None)
            continue
        window = rows[i + 1 - period : i + 1]
        if any(r is None for r in window):
            out.append(None)
            continue
        num = 0.0
        den = 0.0
        for h, l, c, v in window:  # type: ignore[misc]
            mult = 0.0 if h == l else ((c - l) - (h - c)) / (h - l)
            num += mult * v
            den += v
        out.append(None if den == 0 else num / den)
    return out


def money_flow_index(bars: Sequence[Bar], period: int = 14) -> list[Optional[float]]:
    """Money Flow Index over ``period`` bars, in [0, 100].

    Typical price = (high + low + close) / 3. Raw money flow = typical price × volume.
    A bar counts as positive flow when its typical price is above the previous bar's,
    negative when below, and neither when equal. MFI = 100 − 100 / (1 + pos / neg).
    If both sums are 0 (no price movement in the window) the value is 50 (neutral).
    If only the negative sum is 0 the value is 100.
    """
    rows = [_hlcv(b) for b in bars]
    typical: list[Optional[float]] = [None if r is None else _typical(r[0], r[1], r[2]) for r in rows]
    raw: list[Optional[float]] = [None if r is None else typical[k] * r[3] for k, r in enumerate(rows)]  # type: ignore[operator]
    out: list[Optional[float]] = []
    for i in range(len(rows)):
        if i < period:  # needs `period` price changes, i.e. period + 1 bars
            out.append(None)
            continue
        pos = 0.0
        neg = 0.0
        ok = True
        for j in range(i + 1 - period, i + 1):
            if typical[j] is None or typical[j - 1] is None or raw[j] is None:
                ok = False
                break
            if typical[j] > typical[j - 1]:  # type: ignore[operator]
                pos += raw[j]  # type: ignore[operator]
            elif typical[j] < typical[j - 1]:  # type: ignore[operator]
                neg += raw[j]  # type: ignore[operator]
        if not ok:
            out.append(None)
        elif pos == 0 and neg == 0:
            out.append(50.0)
        elif neg == 0:
            out.append(100.0)
        else:
            out.append(100.0 - 100.0 / (1.0 + pos / neg))
    return out


def volume_price_trend(bars: Sequence[Bar]) -> list[Optional[float]]:
    """Cumulative Volume Price Trend: running sum of volume × (close change / previous close).

    The first bar contributes 0. A bar with a missing value, or with a previous close of
    0, makes the running total None from that bar onward: the total is not repaired.
    """
    out: list[Optional[float]] = []
    total: Optional[float] = 0.0
    prev_close: Optional[float] = None
    for k, b in enumerate(bars):
        c = _num(b, "close")
        v = _num(b, "volume")
        if total is None or c is None or v is None:
            total = None
        elif k == 0:
            total = 0.0
        elif prev_close in (None, 0.0):
            total = None
        else:
            total = total + v * (c - prev_close) / prev_close  # type: ignore[operator]
        out.append(total)
        prev_close = c
    return out


def relative_volume(bars: Sequence[Bar], period: int = 20) -> list[Optional[float]]:
    """Volume divided by the average volume of the previous ``period`` bars.

    The baseline excludes the current bar, so a spike cannot inflate its own average.
    A zero baseline gives None (no reference to compare against).
    """
    vols = [_num(b, "volume") for b in bars]
    out: list[Optional[float]] = []
    for i in range(len(vols)):
        if i < period or vols[i] is None:
            out.append(None)
            continue
        base = vols[i - period : i]
        if any(v is None for v in base):
            out.append(None)
            continue
        mean = sum(base) / period  # type: ignore[arg-type]
        out.append(None if mean == 0 else vols[i] / mean)  # type: ignore[operator]
    return out


def anchored_vwap(bars: Sequence[Bar], anchor: int) -> list[Optional[float]]:
    """Volume-weighted average typical price from bar ``anchor`` to each later bar.

    Values before ``anchor`` are None. The anchor is an index chosen by the caller
    before measurement (for example a prior swing low), never picked after seeing
    the outcome. A zero cumulative volume gives None.
    """
    rows = [_hlcv(b) for b in bars]
    out: list[Optional[float]] = []
    num = 0.0
    den = 0.0
    broken = False
    for i, r in enumerate(rows):
        if i < anchor:
            out.append(None)
            continue
        if r is None:
            broken = True
        if broken:
            out.append(None)
            continue
        h, l, c, v = r  # type: ignore[misc]
        num += _typical(h, l, c) * v
        den += v
        out.append(None if den == 0 else num / den)
    return out
