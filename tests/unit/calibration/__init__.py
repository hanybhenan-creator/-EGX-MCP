"""Calibration engine tests — the statistics have to be right, or the whole
"calibrated instead of assumed" claim is just a louder assumption.

The load-bearing tests here are the two controls in ``test_stats`` and
``test_study``: a planted edge must be found, pure noise must not be. Everything
else pins the arithmetic (OBV/ADL against hand-computed values, forward returns
against explicit indices, FDR against a worked example).
"""
