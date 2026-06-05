"""Failure label definition for NN-1.

A "failure" is a BRAIN call that did not produce a usable simulation result.
That is: BRAIN returned an error status (FAIL), the submit timed out (TIMEOUT),
or the submit itself errored before reaching the simulator (SUBMIT_ERROR).

A status of OK means BRAIN ran the expression to completion, regardless of
whether the resulting Sharpe / fitness / turnover cleared submission gates.
NN-1 does NOT care about submission-eligibility — that's NN-2/NN-3/NN-4's job.
This definition is intentionally narrow so NN-1 has enough positive examples
to learn from (you have ~40 failures in 250 entries; you have ~6 truly
submission-eligible alphas, which is far too few for a classifier).
"""
from __future__ import annotations

LABEL_FAIL = 0  # BRAIN rejected the expression
LABEL_OK = 1    # BRAIN ran it successfully (any Sharpe/fitness)

_FAIL_STATUSES = frozenset({"FAIL", "SUBMIT_ERROR", "TIMEOUT"})
_OK_STATUSES = frozenset({"OK"})


def classify_outcome(status: str | None) -> int | None:
    """Map a feedback-entry status to the NN-1 label, or None if not labelable."""
    if status is None:
        return None
    s = status.strip().upper()
    if s in _FAIL_STATUSES:
        return LABEL_FAIL
    if s in _OK_STATUSES:
        return LABEL_OK
    return None
