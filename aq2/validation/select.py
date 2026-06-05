"""
aq2.validation.select
=====================
OS-deflated correlation-marginal greedy selection.

Main selection layer for the alpha submission pipeline:
  1. Fetches PnL for memory.json alphas (reusing judge/self_correlation.py)
  2. Computes PSR/DSR gate using deflated.py
  3. Runs greedy uncorrelated subset selection in OS-deflated fitness order
  4. Prints the submittable set

Public API:
    compute_os_stats_for_book(session, use_cache=True)
        -> dict[alpha_id -> {expression, sharpe_is, sharpe_os, fitness_is,
                              fitness_os, pnl_series, passes_dsr}]

    passes_dsr_gate(pnl_series, n_trials, sigma_sr, psr_threshold=0.95)
        -> bool

    select_submittable_book(session, threshold=0.70, psr_threshold=0.95,
                             score_by="fitness_is")
        -> (submittable_ids, all_stats, corr_matrix)

    print_book_audit(session)
        -> None  (prints table + summary to stdout)

CLI:
    python -m aq2.validation.select
    python aq2/validation/select.py
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — makes the module importable from any working directory.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent.parent  # .../alphaquant/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from brain_api import fetch_alpha_pnl, fetch_alpha_stats, load_credentials
from judge.self_correlation import (
    fetch_pnl_for_memory,
    correlation_matrix,
    greedy_uncorrelated_subset,
)
from aq2.validation.deflated import (
    alpha_clears_dsr_bar,
    expected_max_sr,
    dsr_regression_test,
)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
_DATA_DIR = _REPO / "data"
_FEEDBACK_PATH = _DATA_DIR / "feedback.json"
_MEMORY_PATH = _DATA_DIR / "memory.json"
_CORR_CACHE_PATH = _DATA_DIR / "self_correlation_cache.json"

# The OS-stats cache stores per-alpha OS sharpe/fitness keyed by alpha_id so
# repeat calls do not re-hit the BRAIN API.  OS stats are stable once an alpha
# has been simulated, so the cache is safe to keep indefinitely.
_OS_CACHE_PATH = _DATA_DIR / "os_stats_cache.json"


# ---------------------------------------------------------------------------
# Helpers — data loading
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def _load_memory_entries() -> list[dict]:
    doc = _load_json(_MEMORY_PATH, {})
    entries = doc.get("entries", doc) if isinstance(doc, dict) else doc
    return entries if isinstance(entries, list) else []


def _load_feedback_entries() -> list[dict]:
    doc = _load_json(_FEEDBACK_PATH, {})
    entries = doc.get("entries", doc) if isinstance(doc, dict) else doc
    return entries if isinstance(entries, list) else []


# ---------------------------------------------------------------------------
# 1. passes_dsr_gate
# ---------------------------------------------------------------------------

def passes_dsr_gate(
    pnl_series: list,
    n_trials: int,
    sigma_sr: float,
    psr_threshold: float = 0.95,
) -> bool:
    """Thin wrapper around alpha_clears_dsr_bar from aq2.validation.deflated.

    Parameters
    ----------
    pnl_series : list of (date_iso_str, float)
        Cumulative PnL series, sorted by date.
    n_trials : int
        Total number of strategies evaluated (including this one).
    sigma_sr : float
        Annualized standard deviation of IS Sharpe ratios across all trials.
    psr_threshold : float
        Minimum PSR required to pass (default 0.95).

    Returns
    -------
    bool
        True if the alpha clears the DSR bar, False otherwise.
    """
    try:
        return alpha_clears_dsr_bar(
            pnl_series=pnl_series,
            n_trials=n_trials,
            sigma_sr=sigma_sr,
            psr_threshold=psr_threshold,
        )
    except Exception as exc:
        warnings.warn(
            f"[DSR gate] PSR computation failed ({type(exc).__name__}: {exc}); "
            "treating alpha as NOT clearing the gate.",
            stacklevel=2,
        )
        return False


# ---------------------------------------------------------------------------
# 2. compute_os_stats_for_book
# ---------------------------------------------------------------------------

def compute_os_stats_for_book(
    session,
    use_cache: bool = True,
) -> dict[str, dict]:
    """Fetch PnL and OS stats for all memory.json alphas with an alpha_id.

    For each alpha:
      - Fetches daily PnL via brain_api.fetch_alpha_pnl (uses
        data/self_correlation_cache.json so repeated calls are fast).
      - Fetches IS + OS stats via brain_api.fetch_alpha_stats (uses
        data/os_stats_cache.json).

    When OS stats are unavailable (BRAIN returns an empty/None os block) the
    function falls back to IS sharpe/fitness with a WARNING note and sets
    ``os_data_available=False`` on that entry.

    Parameters
    ----------
    session : BrainSession
        An authenticated BRAIN session.
    use_cache : bool
        Whether to read from (and write to) the local caches.

    Returns
    -------
    dict[alpha_id, stats_dict]
        Keys per entry:
          alpha_id, expression, archetype,
          sharpe_is, fitness_is, sharpe_os, fitness_os,
          pnl_series,          # list[(date_iso, float)]
          passes_dsr,          # bool  (set provisionally; final gate in
                               #  select_submittable_book after N/sigma are known)
          os_data_available,   # bool
          dsr_used_is_fallback # bool — True when IS data was used as fallback
    """
    memory_entries = _load_memory_entries()
    memory_entries = [e for e in memory_entries if e.get("alpha_id")]

    # ------------------------------------------------------------------ caches
    pnl_cache: dict[str, list] = {}
    if use_cache and _CORR_CACHE_PATH.exists():
        try:
            pnl_cache = json.loads(_CORR_CACHE_PATH.read_text())
        except json.JSONDecodeError:
            pnl_cache = {}

    os_cache: dict[str, dict] = {}
    if use_cache and _OS_CACHE_PATH.exists():
        try:
            os_cache = json.loads(_OS_CACHE_PATH.read_text())
        except json.JSONDecodeError:
            os_cache = {}

    pnl_new = 0
    os_new = 0
    pnl_cache_hits = 0
    os_cache_hits = 0
    pnl_misses = 0
    os_errors = 0

    out: dict[str, dict] = {}

    for entry in memory_entries:
        aid = entry["alpha_id"]

        # ------------------------------------------------ PnL series
        if aid in pnl_cache:
            pnl_pairs = [(row[0], row[1]) for row in pnl_cache[aid]]
            pnl_cache_hits += 1
        else:
            try:
                pnl_pairs = fetch_alpha_pnl(session, aid)
            except Exception as exc:
                print(f"  [select/pnl err] {aid}: {type(exc).__name__}: {exc}")
                pnl_pairs = None

            if pnl_pairs is None:
                pnl_misses += 1
                if pnl_misses <= 3:
                    print(
                        f"  [select/pnl miss] {aid}: BRAIN returned no PnL "
                        f"(404 / 5xx / non-JSON). Will retry next run."
                    )
                continue

            pnl_cache[aid] = [list(p) for p in pnl_pairs]
            pnl_new += 1

        if not pnl_pairs:
            continue

        # ------------------------------------------------ OS stats
        os_data_available = False
        sharpe_os: float | None = None
        fitness_os: float | None = None

        if aid in os_cache:
            cached_os = os_cache[aid]
            sharpe_is_api = cached_os.get("sharpe_is")
            fitness_is_api = cached_os.get("fitness_is")
            sharpe_os = cached_os.get("sharpe_os")
            fitness_os = cached_os.get("fitness_os")
            os_data_available = cached_os.get("os_data_available", False)
            os_cache_hits += 1
        else:
            try:
                stats = fetch_alpha_stats(session, aid)
                sharpe_is_api = stats.get("sharpe")
                fitness_is_api = stats.get("fitness")
                raw_os = stats.get("_raw_os") or {}
                sharpe_os = raw_os.get("sharpe") if raw_os else stats.get("os_sharpe")
                fitness_os = raw_os.get("fitness") if raw_os else stats.get("os_fitness")
                os_data_available = bool(
                    isinstance(sharpe_os, (int, float)) and
                    isinstance(fitness_os, (int, float))
                )
                os_cache[aid] = {
                    "sharpe_is": sharpe_is_api,
                    "fitness_is": fitness_is_api,
                    "sharpe_os": sharpe_os,
                    "fitness_os": fitness_os,
                    "os_data_available": os_data_available,
                }
                os_new += 1
            except Exception as exc:
                print(f"  [select/os err] {aid}: {type(exc).__name__}: {exc}")
                sharpe_is_api = None
                fitness_is_api = None
                os_errors += 1

        # Prefer the memory.json IS values (already validated when logged) over
        # the API re-fetch, which may diverge slightly on old sims.  Fall back
        # to API values only when memory.json lacks them.
        sharpe_is = entry.get("sharpe") if isinstance(entry.get("sharpe"), (int, float)) else sharpe_is_api
        fitness_is = entry.get("fitness") if isinstance(entry.get("fitness"), (int, float)) else fitness_is_api

        dsr_used_is_fallback = not os_data_available

        out[aid] = {
            "alpha_id": aid,
            "expression": entry.get("expression", ""),
            "archetype": entry.get("archetype", "?"),
            "name": entry.get("name", ""),
            "sharpe_is": sharpe_is,
            "fitness_is": fitness_is,
            "sharpe_os": sharpe_os,
            "fitness_os": fitness_os,
            "pnl_series": pnl_pairs,
            # passes_dsr is computed later in select_submittable_book once
            # N and sigma_sr are known globally.
            "passes_dsr": None,
            "os_data_available": os_data_available,
            "dsr_used_is_fallback": dsr_used_is_fallback,
        }

    # Write updated caches if anything new was fetched
    if pnl_new > 0:
        _CORR_CACHE_PATH.write_text(json.dumps(pnl_cache))
    if os_new > 0:
        _OS_CACHE_PATH.write_text(json.dumps(os_cache, default=str))

    n_is_fallback = sum(1 for v in out.values() if v["dsr_used_is_fallback"])
    n_os_real = len(out) - n_is_fallback
    print(
        f"  [select/fetch] pnl: cache_hits={pnl_cache_hits}  new={pnl_new}  "
        f"misses={pnl_misses} | os: {n_os_real} with OS stats, {n_is_fallback} IS-only fallback "
        f"| book_size={len(out)}"
    )
    if n_is_fallback > 0 and n_os_real == 0:
        print(f"  [select/note] No OS stats available from BRAIN for any alpha — using IS fallback gate (sh≥1.25 + PSR0≥0.90) for all")
    return out


# ---------------------------------------------------------------------------
# 3. _compute_dsr_benchmark_params
# ---------------------------------------------------------------------------

def _compute_dsr_benchmark_params() -> tuple[int, float]:
    """Compute (n_trials, sigma_sr) from feedback.json + memory.json.

    n_trials: total number of alpha evaluations (all entries in both files
              that have a numeric sharpe).
    sigma_sr: annualized std-dev of IS Sharpe ratios across all trials.

    Returns
    -------
    (n_trials, sigma_sr)
    """
    sharpes: list[float] = []

    for path in (_FEEDBACK_PATH, _MEMORY_PATH):
        doc = _load_json(path, {})
        entries = doc.get("entries", doc) if isinstance(doc, dict) else doc
        if not isinstance(entries, list):
            continue
        for entry in entries:
            raw = entry.get("sharpe")
            if raw is None:
                continue
            try:
                sharpes.append(float(raw))
            except (ValueError, TypeError):
                continue

    if len(sharpes) < 2:
        raise ValueError(
            f"Need at least 2 Sharpe observations for DSR benchmark, "
            f"got {len(sharpes)}."
        )

    arr = np.array(sharpes, dtype=float)
    n_trials = len(arr)
    sigma_sr = float(np.std(arr, ddof=1))
    return n_trials, sigma_sr


# ---------------------------------------------------------------------------
# 4. select_submittable_book
# ---------------------------------------------------------------------------

def select_submittable_book(
    session,
    threshold: float = 0.70,
    psr_threshold: float = 0.95,
    score_by: str = "fitness_is",
) -> tuple[list[str], dict[str, dict], np.ndarray]:
    """Full selection pipeline: DSR gate then greedy uncorrelated subset.

    Parameters
    ----------
    session : BrainSession
        Authenticated BRAIN session.
    threshold : float
        Pairwise correlation threshold (default 0.70, matching BRAIN's gate).
    psr_threshold : float
        Minimum PSR required to pass DSR gate (default 0.95).
    score_by : str
        Field used to rank alphas in greedy selection.  One of:
          "fitness_os"  — preferred: OS fitness (most predictive of live perf)
          "fitness_is"  — IS fitness (fallback when OS unavailable)
          "sharpe_os"   — OS Sharpe
          "sharpe_is"   — IS Sharpe

    Returns
    -------
    (submittable_ids, all_stats, corr_matrix)
        submittable_ids : list[str]   alpha_ids in the submittable book
        all_stats       : dict        alpha_id -> stats dict (from compute_os_stats_for_book,
                                      enriched with passes_dsr and psr_value)
        corr_matrix     : np.ndarray  n×n Pearson correlation matrix (NaN where insufficient data)
    """
    # Step 1: fetch PnL + OS stats
    all_stats = compute_os_stats_for_book(session, use_cache=True)
    if not all_stats:
        print("[select] No alphas with PnL data found — returning empty book.")
        return [], {}, np.zeros((0, 0))

    # Step 2: DSR benchmark params (N and sigma_sr from the full history)
    n_trials, sigma_sr = _compute_dsr_benchmark_params()
    emax_sr_ann = expected_max_sr(n_trials=n_trials, sigma_sr=sigma_sr)
    emax_sr_daily = emax_sr_ann / math.sqrt(252.0)

    print(
        f"  [select/dsr] n_trials={n_trials}  sigma_sr={sigma_sr:.4f}  "
        f"E[max SR|null]={emax_sr_ann:.4f} (ann) / {emax_sr_daily:.5f} (daily)"
    )

    # Step 3: Apply DSR gate per-alpha and record results
    survivors: list[str] = []

    for aid, stats in all_stats.items():
        pnl = stats["pnl_series"]
        sharpe_is = stats["sharpe_is"]
        os_available = stats.get("os_data_available", False)

        if os_available:
            # Full DSR gate: PSR against E[max SR|null] using the OS PnL series.
            gate_passed = passes_dsr_gate(
                pnl_series=pnl,
                n_trials=n_trials,
                sigma_sr=sigma_sr,
                psr_threshold=psr_threshold,
            )
            stats["gate_mode"] = "dsr_os"
        else:
            # OS stats unavailable — fall back to two-tier IS check:
            # Tier 1: PSR > 0.90 against SR*=0 (is this Sharpe significantly positive
            #         from the PnL series alone, ignoring multiple-testing correction?).
            # Tier 2: IS Sharpe > 1.25 (BRAIN's own submission gate, already cleared
            #         once so this is a minimal sanity check).
            try:
                from aq2.validation.deflated import compute_psr
                psr_vs_zero = compute_psr(pnl, sr_benchmark=0.0)
            except Exception:
                psr_vs_zero = 0.0

            sharpe_ok = isinstance(sharpe_is, (int, float)) and sharpe_is >= 1.25
            gate_passed = sharpe_ok and psr_vs_zero >= 0.90
            stats["gate_mode"] = "is_fallback"
            stats["psr_vs_zero"] = psr_vs_zero

        stats["passes_dsr"] = gate_passed
        stats["dsr_is_conservative_admit"] = (not os_available and gate_passed)

        if gate_passed:
            survivors.append(aid)

    print(
        f"  [select/dsr_gate] {len(survivors)}/{len(all_stats)} alphas pass "
        f"DSR gate (PSR > {psr_threshold})"
    )

    if not survivors:
        # No alpha clears the bar — return empty book
        alpha_ids_all = list(all_stats.keys())
        pnl_data_all = {
            aid: {"pnl": all_stats[aid]["pnl_series"],
                  "expression": all_stats[aid]["expression"],
                  "archetype": all_stats[aid]["archetype"]}
            for aid in alpha_ids_all
        }
        _, corr = correlation_matrix(pnl_data_all)
        return [], all_stats, corr

    # Step 4: Build correlation matrix over survivors only
    pnl_data_survivors = {
        aid: {
            "pnl": all_stats[aid]["pnl_series"],
            "expression": all_stats[aid]["expression"],
            "archetype": all_stats[aid]["archetype"],
        }
        for aid in survivors
    }
    alpha_ids_corr, corr = correlation_matrix(pnl_data_survivors)

    # Step 5: Greedy uncorrelated subset, scored by score_by
    # Build score vector for greedy selection.  When the requested field is
    # unavailable for an alpha, fall back gracefully so the greedy order is
    # still meaningful.
    def _get_score(aid: str) -> float:
        s = all_stats[aid]
        # Prefer the explicitly requested field.
        v = s.get(score_by)
        if isinstance(v, (int, float)):
            return float(v)
        # Automatic fallback chain: os -> is for fitness/sharpe
        if score_by in ("fitness_os", "fitness_is"):
            for fld in ("fitness_os", "fitness_is"):
                v = s.get(fld)
                if isinstance(v, (int, float)):
                    return float(v)
        if score_by in ("sharpe_os", "sharpe_is"):
            for fld in ("sharpe_os", "sharpe_is"):
                v = s.get(fld)
                if isinstance(v, (int, float)):
                    return float(v)
        return 0.0

    scores = [_get_score(aid) for aid in alpha_ids_corr]
    submittable_ids = greedy_uncorrelated_subset(
        alpha_ids=alpha_ids_corr,
        scores=scores,
        corr_matrix=corr,
        threshold=threshold,
    )

    print(
        f"  [select/greedy] {len(submittable_ids)}/{len(survivors)} survivors "
        f"pass correlation gate (|r| < {threshold})"
    )

    # Annotate the full correlation matrix if it was computed only over survivors;
    # build a full one for the audit table.
    _, corr_full = correlation_matrix(
        {
            aid: {
                "pnl": all_stats[aid]["pnl_series"],
                "expression": all_stats[aid]["expression"],
                "archetype": all_stats[aid]["archetype"],
            }
            for aid in all_stats
        }
    )

    return submittable_ids, all_stats, corr_full


# ---------------------------------------------------------------------------
# 5. print_book_audit
# ---------------------------------------------------------------------------

def print_book_audit(session) -> None:
    """Run the full selection pipeline and print an audit table + summary.

    Columns printed:
        alpha_id | name | IS sharpe | OS sharpe | IS fitness | OS fitness |
        os_avail | passes_gate | in_submittable

    Summary:
        N total graduates, N passing DSR gate, N in uncorrelated submittable set
    """
    print("\n" + "=" * 90)
    print("OS-DEFLATED CORRELATION-MARGINAL BOOK AUDIT")
    print("=" * 90)

    submittable_ids, all_stats, corr_full = select_submittable_book(
        session,
        threshold=0.70,
        psr_threshold=0.95,
        score_by="fitness_os",
    )

    if not all_stats:
        print("[audit] No data to display.")
        return

    submittable_set = set(submittable_ids)

    # Determine DSR benchmark for display
    try:
        n_trials, sigma_sr = _compute_dsr_benchmark_params()
        emax_ann = expected_max_sr(n_trials=n_trials, sigma_sr=sigma_sr)
    except Exception:
        emax_ann = float("nan")

    # Header
    col_w = {
        "alpha_id": 14,
        "name": 30,
        "sh_is": 7,
        "sh_os": 7,
        "psr0": 6,
        "gate": 12,
        "submit": 7,
    }
    hdr = (
        f"  {'alpha_id':<{col_w['alpha_id']}} "
        f"{'name':<{col_w['name']}} "
        f"{'sh_IS':>{col_w['sh_is']}} "
        f"{'sh_OS':>{col_w['sh_os']}} "
        f"{'PSR0':>{col_w['psr0']}} "
        f"{'gate':<{col_w['gate']}} "
        f"{'submit':>{col_w['submit']}}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    n_gate_pass = 0
    for aid, s in all_stats.items():
        sh_is = s.get("sharpe_is")
        sh_os = s.get("sharpe_os")
        os_av = s.get("os_data_available", False)
        gate = s.get("passes_dsr", False)
        gate_mode = s.get("gate_mode", "?")
        psr0 = s.get("psr_vs_zero")
        in_sub = aid in submittable_set

        if gate:
            n_gate_pass += 1

        def _fmt(v, width=7, prec=2):
            if isinstance(v, (int, float)):
                return f"{v:.{prec}f}".rjust(width)
            return "?".rjust(width)

        if gate_mode == "is_fallback":
            gate_str = ("PASS[IS]" if gate else "fail[IS]")
        else:
            gate_str = ("PASS[OS]" if gate else "fail[OS]")

        row = (
            f"  {aid:<{col_w['alpha_id']}} "
            f"{(s.get('name') or '')[:col_w['name']]:<{col_w['name']}} "
            f"{_fmt(sh_is, col_w['sh_is'])} "
            f"{_fmt(sh_os, col_w['sh_os'])} "
            f"{_fmt(psr0, col_w['psr0'], prec=3)} "
            f"{gate_str:<{col_w['gate']}} "
            f"{'YES' if in_sub else 'no':>{col_w['submit']}}"
        )
        print(row)

    print()
    print("  SUMMARY")
    print(f"    Total graduates in memory.json (with alpha_id):  {len(all_stats)}")
    print(f"    E[max SR | null, N={n_trials}]:                   {emax_ann:.4f}  (annualized)")
    n_os = sum(1 for s in all_stats.values() if s.get("os_data_available"))
    print(f"    With OS stats / IS-only fallback:                {n_os} / {len(all_stats) - n_os}")
    print(f"    Passing gate (OS: PSR>0.95 | IS: sh≥1.25+PSR0>0.90): {n_gate_pass}")
    print(f"    Uncorrelated submittable book (|r| < 0.70):      {len(submittable_ids)}")

    if submittable_ids:
        print()
        print("  SUBMITTABLE BOOK:")
        print(f"  {'#':>3}  {'alpha_id':<14} {'name':<30} {'sh_IS':>7} {'sh_OS':>7} {'fit_IS':>7} {'fit_OS':>7}  expression")
        print("  " + "-" * 110)
        for i, aid in enumerate(submittable_ids, 1):
            s = all_stats[aid]
            sh_is = s.get("sharpe_is")
            sh_os = s.get("sharpe_os")
            fit_is = s.get("fitness_is")
            fit_os = s.get("fitness_os")
            expr = (s.get("expression") or "")[:55]

            def _f(v):
                return f"{v:.2f}" if isinstance(v, (int, float)) else "?"

            print(
                f"  {i:>3}  {aid[:14]:<14} {(s.get('name') or '')[:30]:<30} "
                f"{_f(sh_is):>7} {_f(sh_os):>7} {_f(fit_is):>7} {_f(fit_os):>7}  {expr}"
            )
    else:
        print()
        print("  [!] No alphas cleared both the DSR gate and the correlation filter.")
        print("      Check that memory.json has alpha_id entries and BRAIN is reachable.")

    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from brain_api import BrainSession, load_credentials

    try:
        creds = load_credentials()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[auth] connecting to BRAIN as {creds['email']}...")
    session = BrainSession(creds["email"], creds["password"])
    print("[auth] authenticated.")

    print_book_audit(session)
