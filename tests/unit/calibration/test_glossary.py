"""Every verdict the engine can emit must be defined in the response glossary."""

from __future__ import annotations

from tradingview_mcp.core.calibration.glossary import (
    EVENT_GLOSSARY,
    SECTOR_GLOSSARY,
    WYCKOFF_GLOSSARY,
    attach_glossary,
)

VERDICTS = ["VALIDATED", "SUGGESTIVE", "REJECTED", "NO_EVENTS"]
WALK_FORWARD = ["ROBUST", "MODERATE", "WEAK", "OVERFITTED"]
IC_VERDICTS = ["PREDICTIVE", "INVERTED", "NOT_DISTINGUISHABLE_FROM_ZERO", "INSUFFICIENT_DATA"]


def test_event_glossary_defines_every_verdict_and_walk_forward_label() -> None:
    text = EVENT_GLOSSARY["verdict"] + EVENT_GLOSSARY["walk_forward_verdict"]
    for label in VERDICTS + WALK_FORWARD:
        assert label in text, label


def test_sector_glossary_defines_information_coefficient_verdicts() -> None:
    for label in IC_VERDICTS:
        assert label in SECTOR_GLOSSARY, label


def test_glossaries_are_plain_english_and_non_empty() -> None:
    for glossary in (EVENT_GLOSSARY, SECTOR_GLOSSARY, WYCKOFF_GLOSSARY):
        assert glossary
        for key, value in glossary.items():
            assert isinstance(value, str) and len(value) > 20, key


def test_attach_glossary_copies_instead_of_mutating() -> None:
    payload = {"a": 1}
    out = attach_glossary(payload, EVENT_GLOSSARY)
    assert "glossary" in out and "glossary" not in payload
    out["glossary"]["hit_rate"] = "changed"
    assert EVENT_GLOSSARY["hit_rate"] != "changed"
