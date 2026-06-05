"""Local self-correlation checker — BRAIN-style pairwise correlation.

BRAIN's own SELF_CORRELATION check (returned in each simulation's `checks`
block) only tells you the MAX correlation of an alpha vs your existing pool.
It does NOT give you the full pairwise matrix, which is what you need to
answer: "of my 23 graduated alphas, which subset is actually independent
enough to submit?"

This module fetches each alpha's daily PnL series via the
/alphas/{id}/recordsets/pnl endpoint (already exposed in
brain_api.fetch_alpha_pnl), aligns them by date, and computes the full
Pearson correlation matrix locally. Submission-eligible subsets are then
found by greedy selection: walk top-fitness alphas in order, keep each one
that has correlation < threshold against everyone already kept.

Public API:
    fetch_pnl_for_memory(session, max_alphas=None)
        -> dict[alpha_id, dict] with {expression, archetype, sharpe, dates, pnl_array}

    correlation_matrix(pnl_data)
        -> (alpha_ids, np.ndarray)  full Pearson correlation matrix

    greedy_uncorrelated_subset(alpha_ids, scores, corr_matrix, threshold=0.70)
        -> list[str]  alpha_ids that are pairwise < threshold and rank highest by score

Usage from CLI: python -m judge.self_correlation [--threshold 0.70] [--max N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MEMORY_PATH = REPO_ROOT / "data" / "memory.json"
CORR_CACHE_PATH = REPO_ROOT / "data" / "self_correlation_cache.json"

# BRAIN's submission-correlation gate. An alpha rejected by SELF_CORRELATION
# at the platform side has Pearson correlation >= this value against at least
# one existing alpha in your pool. Single point of configuration.
DEFAULT_THRESHOLD = 0.70


# ---------------------------------------------------------- fetch

def _load_memory_entries() -> list[dict[str, Any]]:
    if not MEMORY_PATH.exists():
        return []
    try:
        doc = json.loads(MEMORY_PATH.read_text())
    except json.JSONDecodeError:
        return []
    return doc.get("entries", []) if isinstance(doc, dict) else []


def fetch_pnl_for_memory(
    session, max_alphas: int | None = None, use_cache: bool = True
) -> dict[str, dict[str, Any]]:
    """Pull daily PnL for every alpha in memory.json that has an alpha_id.

    Caches results to data/self_correlation_cache.json (keyed by alpha_id)
    so repeat invocations don't re-hit the API. The PnL series is immutable
    once an alpha is simulated, so the cache is safe to keep forever.
    """
    from brain_api import fetch_alpha_pnl

    entries = _load_memory_entries()
    entries = [e for e in entries if e.get("alpha_id")]
    if max_alphas:
        entries = entries[:max_alphas]

    cache: dict[str, list[list]] = {}
    if use_cache and CORR_CACHE_PATH.exists():
        try:
            cache = json.loads(CORR_CACHE_PATH.read_text())
        except json.JSONDecodeError:
            cache = {}

    out: dict[str, dict[str, Any]] = {}
    misses = 0
    none_count = 0
    error_count = 0
    cache_hits = 0
    for i, e in enumerate(entries, 1):
        aid = e["alpha_id"]
        if aid in cache:
            pnl_pairs = cache[aid]
            cache_hits += 1
        else:
            try:
                pnl_pairs = fetch_alpha_pnl(session, aid)
            except Exception as ex:
                error_count += 1
                print(f"  [fetch err] {i}/{len(entries)} {aid}: {type(ex).__name__}: {ex}")
                continue
            if pnl_pairs is None:
                none_count += 1
                # Print early-failure detail so transient BRAIN issues don't
                # silently produce a "got 0 alphas" mystery later.
                if none_count <= 3:
                    print(f"  [fetch miss] {i}/{len(entries)} {aid}: BRAIN returned no PnL "
                          f"(404 / 5xx / non-JSON). Will retry next run.")
                continue
            cache[aid] = [list(p) for p in pnl_pairs]
            misses += 1
        if not pnl_pairs:
            continue
        out[aid] = {
            "alpha_id": aid,
            "expression": e.get("expression", ""),
            "archetype": e.get("archetype", "?"),
            "sharpe": e.get("sharpe"),
            "fitness": e.get("fitness"),
            "turnover_pct": e.get("turnover_pct"),
            "pnl": pnl_pairs,
        }

    # Write cache even on partial success so retries don't redo successful fetches.
    if misses > 0:
        CORR_CACHE_PATH.write_text(json.dumps(cache))

    print(f"  [fetch] cache_hits={cache_hits}  new={misses}  "
          f"BRAIN_misses={none_count}  errors={error_count}")
    return out


# ---------------------------------------------------------- correlation

def _align_returns(
    pnl_data: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str], np.ndarray]:
    """Align all PnL series on a common date index, take first-differences to
    get DAILY RETURNS (PnL change is what BRAIN's correlation check uses, not
    the raw cumulative PnL), and return (alpha_ids, dates, returns_matrix).

    returns_matrix has shape (n_alphas, n_dates). Rows aligned to alpha_ids;
    NaN-filled for missing observations.
    """
    if not pnl_data:
        return [], [], np.zeros((0, 0))

    # Find the intersection of all date sets — only keep dates where every
    # alpha has an observation. Conservative; could be relaxed to a union
    # with pairwise masking but intersection gives the cleanest correlation.
    date_sets = [set(d for d, _ in v["pnl"]) for v in pnl_data.values()]
    common = sorted(set.intersection(*date_sets)) if date_sets else []
    if len(common) < 30:
        # If we have fewer than 30 common observations, correlations are
        # noise. Fall back to a more lenient "any alpha that has at least
        # 30 dates" by using the longest single series as the anchor.
        common = sorted(max(date_sets, key=len))

    alpha_ids = list(pnl_data.keys())
    cum_matrix = np.full((len(alpha_ids), len(common)), np.nan, dtype=float)
    date_index = {d: i for i, d in enumerate(common)}

    for i, aid in enumerate(alpha_ids):
        for d, p in pnl_data[aid]["pnl"]:
            j = date_index.get(d)
            if j is not None:
                cum_matrix[i, j] = p

    # Daily returns = first difference of cumulative PnL.
    # BRAIN's correlation check operates on PnL CHANGES, not cumulative levels.
    returns_matrix = np.diff(cum_matrix, axis=1)

    return alpha_ids, common, returns_matrix


def correlation_matrix(
    pnl_data: dict[str, dict[str, Any]]
) -> tuple[list[str], np.ndarray]:
    """Full pairwise Pearson correlation matrix on daily returns.

    Returns (alpha_ids, n×n matrix). Self-correlations are 1.0 on the
    diagonal. NaN entries indicate insufficient overlapping data — those
    pairs should be treated as unknown, not as uncorrelated.
    """
    alpha_ids, _dates, returns = _align_returns(pnl_data)
    n = len(alpha_ids)
    if n == 0:
        return [], np.zeros((0, 0))

    corr = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(i, n):
            ri = returns[i]
            rj = returns[j]
            mask = np.isfinite(ri) & np.isfinite(rj)
            if mask.sum() < 30:
                continue
            x = ri[mask]
            y = rj[mask]
            if x.std() < 1e-10 or y.std() < 1e-10:
                continue
            c = float(np.corrcoef(x, y)[0, 1])
            corr[i, j] = c
            corr[j, i] = c
    np.fill_diagonal(corr, 1.0)
    return alpha_ids, corr


# ---------------------------------------------------------- selection

def greedy_uncorrelated_subset(
    alpha_ids: list[str],
    scores: list[float],
    corr_matrix: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[str]:
    """Greedy: rank alphas by score descending, walk the list, keep each one
    whose pairwise correlation with every already-kept alpha is below
    threshold. NaN correlations are conservatively treated as "above
    threshold" (skip the pair) — better to defer than to admit a possibly
    correlated pair.

    Returns the kept alpha_ids in keep order (descending score within the
    chosen subset).
    """
    if not alpha_ids:
        return []
    idx_order = sorted(range(len(alpha_ids)), key=lambda i: -scores[i])
    kept: list[int] = []
    for i in idx_order:
        ok = True
        for k in kept:
            c = corr_matrix[i, k]
            if np.isnan(c) or abs(c) >= threshold:
                ok = False
                break
        if ok:
            kept.append(i)
    return [alpha_ids[k] for k in kept]


# ---------------------------------------------------------- pretty-print

def _print_matrix(alpha_ids: list[str], corr: np.ndarray, threshold: float) -> None:
    print(f"\nPairwise correlation matrix (★ = |r| >= {threshold}):")
    short = [aid[:6] for aid in alpha_ids]
    print("       " + "".join(f"{s:>8}" for s in short))
    for i, aid_i in enumerate(alpha_ids):
        row_cells = []
        for j, _ in enumerate(alpha_ids):
            c = corr[i, j]
            if np.isnan(c):
                cell = "   --   "
            elif i == j:
                cell = "  1.00  "
            else:
                mark = "★" if abs(c) >= threshold else " "
                cell = f"{c:+6.2f}{mark} "
            row_cells.append(cell)
        print(f"  {short[i]:>5}" + "".join(row_cells))


def _print_high_corr_pairs(alpha_ids: list[str], corr: np.ndarray, threshold: float,
                           pnl_data: dict[str, dict[str, Any]]) -> None:
    pairs = []
    for i in range(len(alpha_ids)):
        for j in range(i + 1, len(alpha_ids)):
            c = corr[i, j]
            if not np.isnan(c) and abs(c) >= threshold:
                pairs.append((abs(c), c, i, j))
    pairs.sort(reverse=True)
    if not pairs:
        print(f"\nNo pairs above |r| = {threshold}. Pool is already uncorrelated.")
        return
    print(f"\nHIGHLY CORRELATED PAIRS (would be rejected by BRAIN SELF_CORRELATION):")
    print(f"  {'|r|':>5}  {'r':>6}  {'alpha A':<20}  {'alpha B':<20}")
    print("  " + "-" * 70)
    for _, c, i, j in pairs[:20]:
        ai = alpha_ids[i]; aj = alpha_ids[j]
        sh_i = pnl_data[ai].get("sharpe")
        sh_j = pnl_data[aj].get("sharpe")
        sh_i_s = f"sh={sh_i:.2f}" if isinstance(sh_i, (int, float)) else ""
        sh_j_s = f"sh={sh_j:.2f}" if isinstance(sh_j, (int, float)) else ""
        print(f"  {abs(c):>5.3f}  {c:>+6.3f}  {ai[:14]} {sh_i_s:<5}  {aj[:14]} {sh_j_s:<5}")


# ---------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description="Local self-correlation check on memory.json alphas.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Correlation threshold for SELF_CORRELATION-style filtering (default {DEFAULT_THRESHOLD}).")
    ap.add_argument("--max", type=int, default=None,
                    help="Limit to N most recent alphas in memory.json (default all).")
    ap.add_argument("--no-cache", action="store_true",
                    help="Bypass the PnL cache and re-fetch from BRAIN.")
    ap.add_argument("--score-by", choices=("fitness", "sharpe"), default="fitness",
                    help="Field used to rank alphas in greedy uncorrelated selection (default fitness).")
    args = ap.parse_args()

    # Lazy import so the module can be used programmatically without a session.
    from brain_api import BrainSession, load_credentials
    creds = load_credentials()
    print(f"[auth] connecting to BRAIN as {creds['email']}...")
    session = BrainSession(creds["email"], creds["password"])

    t0 = time.time()
    pnl_data = fetch_pnl_for_memory(session, max_alphas=args.max, use_cache=not args.no_cache)
    print(f"[fetch] got PnL for {len(pnl_data)} alphas in {time.time()-t0:.1f}s")
    if not pnl_data:
        print("[error] no PnL fetched — check that memory.json entries have alpha_id and BRAIN is reachable.")
        return 1

    alpha_ids, corr = correlation_matrix(pnl_data)
    print(f"[corr] {len(alpha_ids)} alphas, matrix shape {corr.shape}")

    _print_matrix(alpha_ids, corr, args.threshold)
    _print_high_corr_pairs(alpha_ids, corr, args.threshold, pnl_data)

    # Greedy uncorrelated subset, sorted by fitness (or sharpe)
    scores = [pnl_data[aid].get(args.score_by) or 0 for aid in alpha_ids]
    kept = greedy_uncorrelated_subset(alpha_ids, scores, corr, threshold=args.threshold)
    print(f"\nMAXIMAL UNCORRELATED SUBMISSION SET (greedy by {args.score_by}):")
    print(f"  {len(kept)} alphas, all pairwise |r| < {args.threshold}")
    print(f"  {'#':>3}  {'alpha_id':<14} {'arch':<18} {'sharpe':>6} {'fitness':>7}  expression")
    print("  " + "-" * 100)
    for i, aid in enumerate(kept, 1):
        d = pnl_data[aid]
        sh = d.get("sharpe"); ft = d.get("fitness"); arch = d.get("archetype", "?")
        expr = d.get("expression", "")[:55]
        sh_s = f"{sh:.2f}" if isinstance(sh, (int, float)) else "?"
        ft_s = f"{ft:.2f}" if isinstance(ft, (int, float)) else "?"
        print(f"  {i:>3}  {aid[:14]:<14} {arch:<18} {sh_s:>6} {ft_s:>7}  {expr}")
    print(f"\nDropped {len(alpha_ids) - len(kept)} alphas due to correlation with a higher-ranked alpha.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
