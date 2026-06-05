"""Surrogate model v2 — LightGBM/GBR regression for OS Sharpe + fitness.

Predicts three quantities from a raw expression + optional settings dict:
  - predict_os_sharpe     → float | None
  - predict_fitness       → float | None
  - predict_pass_probability → float | None  (proxy for IS Sharpe >= 1.25)

Uses the 41-dim feature extractor from judge/features.py verbatim.
Model artifacts are stored at surrogate_v1.joblib next to this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import joblib

from judge.features import extract_features, to_vector, FEATURE_NAMES  # noqa: E402

MODEL_PATH = Path(__file__).parent / "surrogate_v1.joblib"


class AlphaSurrogate:
    """LightGBM/GBR surrogate for OS Sharpe + fitness prediction.

    Lazy-loads the model bundle on first prediction call so importing this
    module has no I/O cost.
    """

    def __init__(self) -> None:
        self._bundle: dict | None = None

    def _load(self) -> None:
        if self._bundle is None and MODEL_PATH.exists():
            self._bundle = joblib.load(MODEL_PATH)

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict_os_sharpe(self, expression: str, settings=None) -> float | None:
        """Predict out-of-sample Sharpe ratio for *expression*."""
        self._load()
        if self._bundle is None:
            return None
        feat = extract_features(expression, settings)
        x = np.array([to_vector(feat)])
        return float(self._bundle["sharpe_model"].predict(x)[0])

    def predict_fitness(self, expression: str, settings=None) -> float | None:
        """Predict Brain fitness score for *expression*."""
        self._load()
        if self._bundle is None:
            return None
        feat = extract_features(expression, settings)
        x = np.array([to_vector(feat)])
        return float(self._bundle["fitness_model"].predict(x)[0])

    def predict_pass_probability(
        self, expression: str, settings=None
    ) -> float | None:
        """Estimate P(IS Sharpe >= 1.25) via sigmoid on predicted OS Sharpe."""
        sh = self.predict_os_sharpe(expression, settings)
        if sh is None:
            return None
        # Sigmoid centred at 1.25 with scale 2.0 — empirically reasonable for
        # Brain Sharpe distributions.
        return float(1.0 / (1.0 + np.exp(-2.0 * (sh - 1.25))))

    # ------------------------------------------------------------------
    # Ranking helper
    # ------------------------------------------------------------------

    def rank_candidates(
        self,
        candidates: list[dict],
        book_expressions: list[str] | None = None,
    ) -> list[dict]:
        """Rank a list of alpha dicts by predicted EV.

        Each dict must have an ``expression`` key and optionally a
        ``settings`` key.  Adds three scratch keys prefixed with
        ``_surrogate_`` and returns candidates sorted descending by EV.
        *book_expressions* is accepted for API compatibility but currently
        unused (reserved for future correlation-based diversity scoring).
        """
        for c in candidates:
            expr = c.get("expression", "")
            settings = c.get("settings")
            sh = self.predict_os_sharpe(expr, settings) or 0.0
            fit = self.predict_fitness(expr, settings) or 0.0
            c["_surrogate_sharpe"] = sh
            c["_surrogate_fitness"] = fit
            c["_surrogate_ev"] = max(0.0, sh) * max(0.0, fit)
        candidates.sort(key=lambda x: x.get("_surrogate_ev", 0.0), reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str | None:
        """Name of the ML backend used (e.g. 'LightGBM' or 'sklearn GBR')."""
        self._load()
        if self._bundle is None:
            return None
        return self._bundle.get("backend")

    @property
    def n_train(self) -> int | None:
        """Number of training samples used to fit the loaded model."""
        self._load()
        if self._bundle is None:
            return None
        return self._bundle.get("n_train")

    @property
    def is_loaded(self) -> bool:
        self._load()
        return self._bundle is not None


# ---------------------------------------------------------------------------
# Module-level singleton + convenience functions
# ---------------------------------------------------------------------------

_SURROGATE = AlphaSurrogate()


def predict_os_sharpe(expression: str, settings=None) -> float | None:
    """Module-level convenience: predict OS Sharpe for *expression*."""
    return _SURROGATE.predict_os_sharpe(expression, settings)


def predict_fitness(expression: str, settings=None) -> float | None:
    """Module-level convenience: predict fitness for *expression*."""
    return _SURROGATE.predict_fitness(expression, settings)


def predict_pass_probability(expression: str, settings=None) -> float | None:
    """Module-level convenience: P(IS Sharpe >= 1.25) for *expression*."""
    return _SURROGATE.predict_pass_probability(expression, settings)


def rank_candidates(
    candidates: list[dict],
    book_expressions: list[str] | None = None,
) -> list[dict]:
    """Module-level convenience: rank alpha dicts by predicted EV."""
    return _SURROGATE.rank_candidates(candidates, book_expressions)
