"""Calibrated signal measurement for the EGX smart-money model.

    from tradingview_mcp.core.calibration import Sample, analyze

    sample = Sample(bars_by_symbol=store_bars, horizon=5)
    report = analyze(sample, signals=["accumulation_corrected", "c2"])

``report`` holds, for every signal: how often it fired, what happened next,
how that compares with the base rate of the same sample, a p-value from a
block permutation, a q-value corrected for the number of signals tested, and
whether the edge survived out-of-sample. That is the difference between
"confidence: 87" and a number somebody can check.

The engine is deliberately dependency-free (no numpy/pandas) and reads plain
bar dictionaries, so it runs directly on the data layer's ``DataStore``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from tradingview_mcp.core.calibration.signals import (  # noqa: F401
    CORRECTED_CONDITIONS,
    DEFAULT_PARAMS,
    DISTRIBUTION_CONDITIONS,
    LEGACY_CONDITIONS,
    Bar,
    SignalContext,
    SignalSpec,
    all_signals,
    get_signal,
    register,
    signal_names,
)
from tradingview_mcp.core.calibration.stats import (  # noqa: F401
    benjamini_hochberg,
    block_shuffle,
    describe,
    hit_rate,
    mean,
    median,
    permutation_p,
    permutation_p_hit,
    permutation_p_mean,
    quantile,
    stdev,
    wilson_interval,
)
from tradingview_mcp.core.calibration.study import (  # noqa: F401
    Fold,
    Forward,
    Measurement,
    Sample,
    WalkForward,
    apply_fdr,
    call_it,
    forward_return,
    forward_returns,
    measure,
    measure_all,
    session_axis,
    walk_forward,
)

__all__ = [
    "Bar",
    "SignalContext",
    "SignalSpec",
    "Sample",
    "Measurement",
    "Fold",
    "WalkForward",
    "Forward",
    "analyze",
    "report_to_dict",
    "register",
    "all_signals",
    "signal_names",
    "get_signal",
    "DEFAULT_PARAMS",
    "LEGACY_CONDITIONS",
    "CORRECTED_CONDITIONS",
    "DISTRIBUTION_CONDITIONS",
]


@dataclass
class CalibrationReport:
    horizon: int
    measurements: list[Measurement] = field(default_factory=list)
    walk_forwards: dict[str, WalkForward] = field(default_factory=dict)
    sessions: int = 0
    symbols: int = 0
    window: tuple[Optional[str], Optional[str]] = (None, None)
    permutations: int = 0

    @property
    def validated(self) -> list[Measurement]:
        return [m for m in self.measurements if m.verdict == "VALIDATED"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "sessions": self.sessions,
            "symbols": self.symbols,
            "window": {"from": self.window[0], "to": self.window[1]},
            "permutations": self.permutations,
            "measurements": [m.as_dict() for m in self.measurements],
            "walk_forward": {name: wf.as_dict() for name, wf in self.walk_forwards.items()},
            "validated_signals": [m.signal for m in self.validated],
        }


def analyze(
    sample: Sample,
    *,
    signals: Optional[Iterable[str]] = None,
    permutations: int = 500,
    walk_forward_folds: int = 0,
    block: int = 10,
    seed: int = 0,
    min_events: int = 30,
    alpha: float = 0.10,
) -> CalibrationReport:
    """Measure, correct, and (optionally) walk the signals forward.

    Signals are measured first (each with its own permutation), then FDR-corrected
    as a batch, then classified; ``walk_forward_folds`` > 0 additionally measures
    every *verified* signal in contiguous out-of-sample folds.
    """
    measurements = measure_all(
        sample, signals=signals, permutations=permutations, block=block, seed=seed
    )
    for m in measurements:
        m.verdict = call_it(m, min_events=min_events, alpha=alpha)
    axis = session_axis(sample)
    report = CalibrationReport(
        horizon=sample.horizon,
        measurements=measurements,
        sessions=len(axis),
        symbols=len(sample.bars_by_symbol),
        window=(axis[0], axis[-1]) if axis else (None, None),
        permutations=permutations,
    )
    if walk_forward_folds:
        for m in measurements:
            if m.events and m.verdict in {"VALIDATED", "SUGGESTIVE"}:
                report.walk_forwards[m.signal] = walk_forward(
                    sample, m.signal, folds=walk_forward_folds, seed=seed
                )
    return report


def report_to_dict(report: CalibrationReport) -> dict[str, Any]:
    return report.as_dict()
