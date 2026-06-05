"""
aq2.validation.deflated
=======================
Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR) gate.

References
----------
Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Chapter 15.
Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
    Journal of Portfolio Management, 40(5), 94-107.
"""

import sys
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Path setup — makes the module importable from any working directory.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent.parent  # .../alphaquant/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Data paths (used by the regression test)
# ---------------------------------------------------------------------------
_DATA_DIR = _REPO / "data"
_FEEDBACK_PATH = _DATA_DIR / "feedback.json"
_MEMORY_PATH = _DATA_DIR / "memory.json"

# Euler–Mascheroni constant
_EULER: float = 0.5772156649015328


# ---------------------------------------------------------------------------
# 1. compute_psr
# ---------------------------------------------------------------------------

def compute_psr(pnl_series: list, sr_benchmark: float) -> float:
    """
    Probabilistic Sharpe Ratio (per-period / daily units).

    Parameters
    ----------
    pnl_series : list of (date_iso_str, float)
        Cumulative PnL series, sorted by date.  First-differenced internally
        to produce daily returns.
    sr_benchmark : float
        Benchmark Sharpe ratio in *per-period* (daily) units.

    Returns
    -------
    float
        PSR = P(SR > sr_benchmark | observed data), a probability in [0, 1].

    Formula (Lopez de Prado 2018, eq 15.2)
    ----------------------------------------
    SR_hat  = mean(r) / std(r, ddof=1)          # per-period, NOT annualized
    skew    = skewness of returns
    kurt    = kurtosis (NOT excess, i.e. Fisher=False)
    denom   = sqrt( (1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2) / (T-1) )
    z       = (SR_hat - sr_benchmark) / denom
    PSR     = Phi(z)
    """
    if len(pnl_series) < 3:
        raise ValueError(
            f"pnl_series must have at least 3 observations, got {len(pnl_series)}"
        )

    # First-difference cumulative PnL -> daily returns
    cum_pnl = np.array([v for _, v in pnl_series], dtype=float)
    returns = np.diff(cum_pnl)

    T = len(returns)
    if T < 2:
        raise ValueError("Need at least 2 daily returns to compute PSR.")

    sr_hat = float(np.mean(returns)) / float(np.std(returns, ddof=1))
    skew = float(sp_stats.skew(returns))
    # fisher=False => regular kurtosis (normal = 3), NOT excess kurtosis
    kurt = float(sp_stats.kurtosis(returns, fisher=False))

    denom_inner = (
        1.0
        - skew * sr_hat
        + (kurt - 1.0) / 4.0 * sr_hat ** 2
    )
    # Guard against numerical zero / negative values
    denom_inner = max(denom_inner, 1e-10)

    z = (sr_hat - sr_benchmark) * math.sqrt(T - 1) / math.sqrt(denom_inner)
    psr = float(sp_stats.norm.cdf(z))
    return psr


# ---------------------------------------------------------------------------
# 2. expected_max_sr
# ---------------------------------------------------------------------------

def expected_max_sr(n_trials: int, sigma_sr: float) -> float:
    """
    Expected maximum Sharpe ratio under the null (all SRs are noise).

    Parameters
    ----------
    n_trials : int
        Number of independent strategies tested.
    sigma_sr : float
        Standard deviation of the SR distribution across trials.
        Must be in the *same* units as the desired benchmark:
          - annualized units  -> pass sigma_annualized
          - per-period units  -> pass sigma_annualized / sqrt(252)

    Returns
    -------
    float
        E[max SR | null] in the same units as sigma_sr.

    Formula (Bailey & Lopez de Prado 2014)
    ----------------------------------------
    EULER = 0.5772156649015328
    t1 = (1 - EULER) * Phi^{-1}(1 - 1/n)
    t2 = EULER         * Phi^{-1}(1 - 1/(n * sqrt(e) * sqrt(ln(n))))
    E[max SR] = sigma_sr * (t1 + t2)
    """
    if n_trials < 2:
        raise ValueError("n_trials must be >= 2.")
    if sigma_sr <= 0:
        raise ValueError("sigma_sr must be positive.")

    n = float(n_trials)
    t1 = (1.0 - _EULER) * sp_stats.norm.ppf(1.0 - 1.0 / n)
    t2 = _EULER * sp_stats.norm.ppf(
        1.0 - 1.0 / (n * math.sqrt(math.e) * math.sqrt(math.log(n)))
    )
    return sigma_sr * (t1 + t2)


# ---------------------------------------------------------------------------
# 3. dsr_regression_test
# ---------------------------------------------------------------------------

def _load_is_sharpes() -> np.ndarray:
    """
    Load all IS Sharpe ratios from feedback.json and memory.json.
    Returns a 1-D float array of valid numeric values.
    """
    sharpes: list[float] = []

    for path in (_FEEDBACK_PATH, _MEMORY_PATH):
        with open(path, "r") as fh:
            data = json.load(fh)

        # Both files use the schema {"_comment": ..., "entries": [...]}
        entries = data.get("entries", data) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ValueError(f"Unexpected structure in {path}: 'entries' is not a list.")

        for entry in entries:
            raw = entry.get("sharpe")
            if raw is None:
                continue
            try:
                sharpes.append(float(raw))
            except (ValueError, TypeError):
                continue

    return np.array(sharpes, dtype=float)


def dsr_regression_test() -> None:
    """
    Reproduce E[max SR | null] ≈ 2.386 from the project's historical alpha data.

    Loads all IS Sharpe ratios from feedback.json + memory.json, computes
    E[max SR | null] in annualized units, prints the result, and asserts it
    falls in [2.3, 2.5].

    Expected values (as of 2026-06-01):
        N       ≈ 1921
        sigma   ≈ 0.6794
        E[max]  ≈ 2.386
    """
    sharpes = _load_is_sharpes()
    n = len(sharpes)
    sigma = float(np.std(sharpes, ddof=1))

    emax = expected_max_sr(n_trials=n, sigma_sr=sigma)

    print(
        f"DSR regression test\n"
        f"  N trials      : {n}\n"
        f"  sigma(SR)     : {sigma:.4f}  (annualized units)\n"
        f"  E[max SR|null]: {emax:.4f}"
    )

    assert 2.3 <= emax <= 2.5, (
        f"E[max SR|null] = {emax:.4f} is outside the expected range [2.3, 2.5]. "
        "Check the data files or formula."
    )
    print("  PASSED: E[max SR|null] is within [2.3, 2.5]")


# ---------------------------------------------------------------------------
# 4. alpha_clears_dsr_bar
# ---------------------------------------------------------------------------

def alpha_clears_dsr_bar(
    pnl_series: list,
    n_trials: int,
    sigma_sr: float,
    psr_threshold: float = 0.95,
) -> bool:
    """
    DSR gate check: returns True iff PSR(benchmark=E[max SR|null]) > psr_threshold.

    This is the core production guard against selection bias.  It asks:
    "Given that we tested n_trials strategies with cross-sectional SR dispersion
    sigma_sr (annualized), is *this* strategy's Sharpe ratio significantly
    better than the best one we'd expect from pure noise?"

    Parameters
    ----------
    pnl_series : list of (date_iso_str, float)
        Cumulative PnL, sorted by date (first-differenced internally).
    n_trials : int
        Total number of strategies evaluated (including this one).
    sigma_sr : float
        Annualized standard deviation of IS Sharpe ratios across all trials.
    psr_threshold : float, optional
        Minimum PSR required to pass (default 0.95).

    Returns
    -------
    bool
        True  -> strategy clears the DSR bar (passes the gate).
        False -> strategy does not clear the DSR bar (reject / do not deploy).

    Notes
    -----
    All internal PSR arithmetic is done in *per-period* (daily) units.
    sigma_sr (annualized) is converted to daily via / sqrt(252) before being
    passed to expected_max_sr(), so the benchmark returned is also in daily
    units and directly comparable to the per-period SR_hat computed inside
    compute_psr().
    """
    # Convert annualized sigma to per-period (daily) units for the benchmark
    sigma_sr_daily = sigma_sr / math.sqrt(252.0)

    # E[max SR | null] in per-period units
    benchmark_daily = expected_max_sr(n_trials=n_trials, sigma_sr=sigma_sr_daily)

    # PSR against that per-period benchmark
    psr = compute_psr(pnl_series=pnl_series, sr_benchmark=benchmark_daily)

    return psr > psr_threshold


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dsr_regression_test()
