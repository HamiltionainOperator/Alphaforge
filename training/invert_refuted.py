"""Resubmit `refuted_inverted` alphas with their sign flipped.

When reflect.py classifies a simulation as 'refuted_inverted', the mechanism
was right but the sign was wrong (Sharpe in roughly -1.5 to -0.5). Multiplying
the expression by -1 should flip Sharpe into +0.5 to +1.5 territory — often
above the 1.25 passing threshold.

This script:
  1. Scans data/hypotheses/*.json for entries with reflection.verdict == 'refuted_inverted'
  2. Wraps each expression as `-1 * (original)` (or strips an existing leading
     `-1 *` if double-negation would cancel)
  3. Submits the inverted variants to BRAIN with the same settings
  4. Lets brain_api.submit_batch auto-log results to memory.json / feedback.json
  5. Updates the source hypothesis file with an `inversion_attempt` block so
     you can see the before/after

Usage:
    python training/invert_refuted.py                # all refuted_inverted
    python training/invert_refuted.py --limit 5      # first 5 only
    python training/invert_refuted.py --dry-run      # show what would be done
    python training/invert_refuted.py --max-concurrent 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HYPOTHESES_DIR = ROOT / "data" / "hypotheses"


def _invert_expression(expr: str) -> str:
    """Wrap expression in -1 * (...), or strip an existing leading -1 *."""
    stripped = expr.strip()
    # Cancel double-negation if already starts with `-1 *` or `-1*`
    m = re.match(r"^-\s*1\s*\*\s*\(\s*(.*)\s*\)\s*$", stripped, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.match(r"^-\s*1\s*\*\s*(.+)$", stripped, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # Wrap. Add explicit parens to be safe even if original starts with `rank(...)`.
    return f"-1 * ({stripped})"


def _build_inverted_alpha(source: dict) -> dict:
    """Build an alpha dict suitable for brain_api.submit_batch from a source hypothesis file."""
    original_expr = source["expression"]
    inverted_expr = _invert_expression(original_expr)
    original_name = source.get("alpha_name") or "alpha"
    original_hyp = source.get("hypothesis") or {}

    new_hyp = dict(original_hyp)
    # Flip the sign claim — mechanism stays, direction inverts
    if new_hyp.get("claim"):
        new_hyp["claim"] = (
            "[INVERTED] Original sign was wrong; same mechanism, opposite direction. "
            + new_hyp["claim"]
        )
    new_hyp["inverted_from"] = original_name
    if new_hyp.get("prediction"):
        # The refuted_inverted classification implies the original prediction direction
        # was backwards; the new prediction is the inverse.
        new_hyp["prediction"] = {
            **new_hyp.get("prediction", {}),
            "regime_works": (new_hyp["prediction"].get("regime_fails") or "(inverted regime)"),
            "regime_fails": (new_hyp["prediction"].get("regime_works") or "(inverted regime)"),
        }
    if new_hyp.get("refutation_criterion"):
        new_hyp["refutation_criterion"] = (
            f"Inverted sign should now produce Sharpe > 0.5 with same fitness profile. "
            f"Refute again if: {new_hyp['refutation_criterion']}"
        )

    return {
        "name": f"inv_{original_name}"[:64],
        "expression": inverted_expr,
        "archetype": source.get("archetype", "novel"),
        "logic": "[INVERTED] " + (source.get("logic") or "")[:280],
        "settings": source.get("settings", {}),
        "expected_sharpe": "0.8-1.3",  # rough inversion estimate
        "paper_source": source.get("paper_source"),
        "hypothesis": new_hyp,
    }


def _collect_refuted_inverted(limit: int | None) -> list[tuple[Path, dict]]:
    files = sorted(HYPOTHESES_DIR.glob("*.json"))
    picks: list[tuple[Path, dict]] = []
    for fp in files:
        try:
            data = json.loads(fp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        ref = data.get("reflection") or {}
        if ref.get("verdict") != "refuted_inverted":
            continue
        # Skip if we already inverted this one
        if data.get("inversion_attempt"):
            continue
        picks.append((fp, data))
        if limit is not None and len(picks) >= limit:
            break
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N refuted_inverted entries.")
    ap.add_argument("--dry-run", action="store_true", help="Print planned inversions without submitting.")
    ap.add_argument("--max-concurrent", type=int, default=3)
    args = ap.parse_args()

    picks = _collect_refuted_inverted(args.limit)
    if not picks:
        print("No refuted_inverted entries to process. Run a generation+sim session first.")
        return 0

    print(f"Found {len(picks)} refuted_inverted alpha(s):\n")
    alphas: list[dict] = []
    plan: list[tuple[Path, dict, dict]] = []
    for fp, src in picks:
        inv = _build_inverted_alpha(src)
        sharpe_was = (src.get("reflection") or {}).get("sharpe_actual") or (src.get("result") or {}).get("sharpe")
        print(f"  [{src.get('archetype', '?'):<11}] {src.get('alpha_name')}")
        print(f"      old sharpe = {sharpe_was}")
        print(f"      old expr   = {src.get('expression')[:100]}")
        print(f"      new expr   = {inv['expression'][:100]}")
        print()
        alphas.append(inv)
        plan.append((fp, src, inv))

    if args.dry_run:
        print(f"[dry-run] would submit {len(alphas)} inverted alphas.")
        return 0

    print(f"Submitting {len(alphas)} inverted alphas to BRAIN (max_concurrent={args.max_concurrent})...")
    from brain_api import submit_batch
    records = submit_batch(alphas, max_concurrent=args.max_concurrent)

    # Stitch each result back into the source hypothesis file as `inversion_attempt`.
    by_name = {r["alpha"]["name"]: r for r in records}
    confirmed_count = 0
    near_count = 0
    for fp, src, inv in plan:
        rec = by_name.get(inv["name"])
        if not rec:
            continue
        result = rec.get("result", {})
        sharpe = result.get("sharpe")
        fitness = result.get("fitness")
        passing = result.get("submission_pass")
        if passing:
            confirmed_count += 1
        elif isinstance(sharpe, (int, float)) and sharpe >= 1.0:
            near_count += 1

        src["inversion_attempt"] = {
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "inverted_expression": inv["expression"],
            "result": {
                "sharpe": sharpe,
                "fitness": fitness,
                "turnover_pct": result.get("turnover_pct"),
                "submission_pass": passing,
            },
        }
        try:
            fp.write_text(json.dumps(src, indent=2) + "\n")
        except OSError as ex:
            print(f"  [warn] failed to write back to {fp.name}: {ex}")

    print(f"\n=== Inversion Sweep Summary ===")
    print(f"  Submitted:           {len(alphas)}")
    print(f"  Newly passing:       {confirmed_count}")
    print(f"  Near (sharpe>=1.0):  {near_count}")
    print(f"  Already in memory.json (passing) / feedback.json (otherwise).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
