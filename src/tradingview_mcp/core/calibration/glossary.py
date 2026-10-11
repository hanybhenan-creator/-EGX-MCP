"""Plain-English definitions attached to every calibration response.

Tool outputs are read by language models that have no project context. Each
response that reports statistics carries a ``glossary`` with the meaning of
every verdict and number it contains, so the output can be read on its own.

The definitions describe what the code computes. If a rule changes, change the
definition in the same commit.
"""

from __future__ import annotations

from typing import Any

EVENT_GLOSSARY: dict[str, str] = {
    "event": "A date on which a stated condition was true for a stock (or sector), decided only from prices and volume up to that session's close.",
    "horizon": "Number of trading sessions held after the event. The entry is the next session's open and the exit is the close at the end of the horizon.",
    "hit_rate": "Share of events whose return over the horizon was strictly positive.",
    "base_rate": "Share of all symbol-sessions in the same sample whose return over the horizon was strictly positive. This is what a random pick would have achieved. Compare hit_rate against it, not against 50%.",
    "excess_hit": "hit_rate minus base_rate, in percentage points. Positive means the condition beat a random pick over the same dates.",
    "wilson_interval_95": "95% range for the true hit rate given the number of events (Wilson score interval). A narrow range needs many events.",
    "mean_return": "Average return over the horizon across the events.",
    "p_value": "How often a random reshuffling of the return series, with the same block structure, produces an excess at least as large as the observed one. Small means the timing of the event matters. It is not the probability that the condition is true.",
    "q_value": "The p-value adjusted for testing many conditions at once (Benjamini-Hochberg false discovery rate). A q_value of 0.10 means that, among conditions called significant at this level, about 10% are expected to be false discoveries.",
    "walk_forward": "The sample is split into consecutive time folds. The edge is measured on earlier data and then on later, unseen data. Retention is the later excess divided by the earlier excess.",
    "walk_forward_verdict": "ROBUST: out-of-sample excess keeps at least 80% of the in-sample edge. MODERATE: keeps 50-80%. WEAK: most of the edge disappears out of sample. OVERFITTED: under 20% survives out of sample.",
    "verdict": "VALIDATED: at least 30 events, positive excess hit rate, and q_value at or below 0.10. SUGGESTIVE: the direction is positive but evidence is not yet enough (few events, or q_value above 0.10). REJECTED: the excess is zero or negative. NO_EVENTS: the condition never fired in the sample.",
    "measured_sample": "All results describe the stored local sample only. A verdict is a statement about past data, not a forecast.",
}

SECTOR_GLOSSARY: dict[str, str] = {
    **EVENT_GLOSSARY,
    "excess_return": "Sector return over the horizon minus the average return of all sectors on the same date. This removes market-wide moves.",
    "top_third / bottom_third": "On each date, the sectors ranked in the top or bottom third by a score. The rule is tested by comparing those two groups.",
    "information_coefficient": "Spearman rank correlation, per date, between a score and the next-horizon excess return across sectors. +1 means the ranking perfectly predicted the order; 0 means no relation; -1 means perfectly reversed.",
    "mean_ic": "Average information coefficient across dates. Positive means the score ranks sectors in the right order on average.",
    "t_stat": "mean_ic divided by its standard error across dates. Values of at least 2 in absolute size are called PREDICTIVE or INVERTED. This threshold is not corrected for testing several scores, so treat those calls as possible false positives.",
    "sign_consistency": "Share of dates on which the information coefficient had the same sign as its average.",
    "PREDICTIVE": "t_stat of at least +2: the score ranked sectors in the right order more often than chance.",
    "INVERTED": "t_stat of at most -2: the score ranked sectors in the wrong order.",
    "NOT_DISTINGUISHABLE_FROM_ZERO": "The information coefficient is too small relative to its noise to tell apart from zero.",
    "INSUFFICIENT_DATA": "Too few dates to compute the information coefficient.",
    "hot_cold_labels": "The live sector scanner labels sectors Hot, Warming, Cooling or Cold from a momentum score. These are the labels that get replayed here.",
}

WYCKOFF_GLOSSARY: dict[str, str] = {
    **EVENT_GLOSSARY,
    "selling_climax": "A high-volume down day that closes near its high after a new 20-session low. Practitioners read it as selling being absorbed.",
    "spring": "A new 20-session low that closes back above that low, on below-average volume.",
    "no_supply": "A narrow-range down day on low volume, closing in the upper half of its range. Practitioners read it as sellers being absent.",
    "secondary_test": "A later low near a selling-climax low, on lower volume than the climax, that does not undercut the climax low.",
    "state": "NO_RECENT_EVENT: no event in the lookback window. EVENT_FIRED: at least one event occurred. The state is descriptive, not an instruction.",
}


def attach_glossary(payload: dict[str, Any], glossary: dict[str, str]) -> dict[str, Any]:
    """Return ``payload`` with a ``glossary`` key (copied, not mutated)."""
    out = dict(payload)
    out["glossary"] = dict(glossary)
    return out
