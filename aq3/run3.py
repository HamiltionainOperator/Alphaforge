"""
aq3.run3 — the AlphaQuant v3 active-learning loop.

    evolve locally (thousands of candidates against the surrogate)
        → take the top-K structurally-distinct elite
        → BRAIN-simulate ONLY the elite (the only true fitness signal)
        → results route to memory.json / feedback.json
        → RETRAIN the surrogate on the enlarged dataset
        → repeat

This is the whole point of v3: BRAIN slots are spent only on candidates that
survived a full local evolution, and every real result sharpens the surrogate
that guides the next evolution (active learning). The surrogate is a NOISY guide
— BRAIN is ground truth — so each outer iteration both spends evidence and earns
better evidence.

Usage:
  python -m aq3.run3 --no-simulate                 # local only: evolve + print elite
  python -m aq3.run3 --generations 30 --elite 5    # one iteration, simulate the elite
  python -m aq3.run3 --iterations 3 --elite 5      # full active-learning loop (needs BRAIN)
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from aq3.evolve import Evolver, EvalResult
from math_engine import MathEngine


# --------------------------------------------------------------------------- #
# Hypothesis pre-generation stage                                              #
# --------------------------------------------------------------------------- #

def run_hypothesis_stage(args) -> list[dict]:
    """Run HypothesisEngine before the evolution loop and return the top-N
    hypotheses by novelty score.  Logs each step so the caller can confirm
    this stage fired before expression generation.

    Returns an empty list if ANTHROPIC_API_KEY is absent or --no-hypotheses
    was passed — the caller falls back to seeding without intent context.
    """
    if getattr(args, "no_hypotheses", False):
        print("[hypothesis] stage skipped (--no-hypotheses)")
        return []

    try:
        from hypothesis_generator import HypothesisEngine
    except ImportError as exc:
        print(f"[hypothesis] HypothesisEngine import failed ({exc}); skipping stage")
        return []

    try:
        engine = HypothesisEngine()
    except EnvironmentError as exc:
        print(f"[hypothesis] {exc}; skipping stage")
        return []

    archetype = getattr(args, "archetype", None) or None
    theme = getattr(args, "theme", None) or None
    hyp_count = getattr(args, "hyp_count", 10)

    print(f"\n{'='*64}")
    print(f"[hypothesis] PRE-GENERATION STAGE  count={hyp_count}  archetype={archetype or 'any'}  theme={theme or 'any'}")
    print(f"{'='*64}")

    raw = engine.generate(archetype=archetype, theme=theme, count=hyp_count)
    novel = engine.filter_novel(raw)

    top_n = getattr(args, "hyp_top", 3)
    top = sorted(novel, key=lambda h: h["novelty_score"], reverse=True)[:top_n]

    print(f"\n[hypothesis] Top-{len(top)} hypotheses passed to generation loop:")
    for i, h in enumerate(top, 1):
        print(f"  {i}. [{h['archetype']}] {h['title']}  (novelty={h['novelty_score']}/10)")
        print(f"     claim:   {h['claim']}")
        print(f"     fields:  {h['fields_suggested']}")
        print(f"     turnover:{h['expected_turnover_range']}  sharpe:{h['expected_sharpe_range']}")

    return top


def _settings_for(expr: str, me: MathEngine) -> dict:
    """Per-alpha tuned settings (reuses v1 auto_settings → defaults to Subindustry)."""
    try:
        s = me.auto_settings(expr)
        return s if isinstance(s, dict) else {}
    except Exception:
        return {"neutralization": "Subindustry", "decay": 6, "truncation": 0.05}


def evolve_elite(rng: random.Random, pop_size: int, generations: int,
                 elite_k: int, verbose: bool = True) -> list[EvalResult]:
    ev = Evolver(rng, pop_size=pop_size)
    final = ev.run(generations=generations, verbose=verbose)
    return ev.select_elite(final, k=elite_k)


def _print_elite(elite: list[EvalResult], me: MathEngine) -> None:
    print(f"\n[v3] Top-{len(elite)} elite for BRAIN validation:")
    for i, r in enumerate(elite, 1):
        s = _settings_for(r.expr, me)
        print(f"  {i}. pred_sharpe={r.pred_sharpe:.2f}  pred_fitness={r.pred_fitness:.2f}  "
              f"size={r.genome.inner.size()}")
        print(f"     neut={s.get('neutralization')} decay={s.get('decay')} trunc={s.get('truncation')}")
        print(f"     {r.expr}")


def run_iteration(rng: random.Random, args, me: MathEngine,
                  session=None, hypotheses: list[dict] | None = None) -> dict:
    # Log which hypotheses are being used as intent for this iteration.
    if hypotheses:
        print(f"\n[v3] Using {len(hypotheses)} hypothesis intents for this generation:")
        for h in hypotheses:
            print(f"     [{h['archetype']}] {h['title']} — {h['claim'][:80]}")

    print(f"\n{'='*64}\n[v3] EVOLVE → {args.pop} genomes × {args.generations} generations\n{'='*64}")
    elite = evolve_elite(rng, args.pop, args.generations, args.elite, verbose=True)
    _print_elite(elite, me)

    if args.no_simulate:
        return {"elite": elite, "records": []}

    # Build alpha dicts for BRAIN
    alphas = []
    for i, r in enumerate(elite):
        alphas.append({
            "name": f"aq3_g{args.generations}_e{i+1}",
            "expression": r.expr,
            "settings": _settings_for(r.expr, me),
            "archetype": "aq3_evolved",
            "_pred_sharpe": r.pred_sharpe,
            "_pred_fitness": r.pred_fitness,
        })

    from brain_api import submit_batch
    print(f"\n[v3] Simulating {len(alphas)} elite on BRAIN "
          f"(max_concurrent={args.max_concurrent}, sweep={args.sweep} → strong-negatives auto-invert)...")
    # sweep>=1 makes submit_batch run simulate_alpha_adaptive, which auto-inverts
    # any strong-negative result (sharpe <= -0.3) — a wrong SIGN is a lead, not a
    # failure. max_concurrent is capped to BRAIN's per-account slot limit so no
    # elite is dropped with CONCURRENT_SIMULATION_LIMIT_EXCEEDED.
    records = submit_batch(alphas, session=session,
                           max_concurrent=args.max_concurrent, sweep=args.sweep)
    n_pass = sum(1 for rec in records if rec.get("result", {}).get("submission_pass"))
    # Surface inversion leads explicitly so they aren't lost in the noise.
    leads = [rec for rec in records
             if isinstance(rec.get("result", {}).get("sharpe"), (int, float))
             and rec["result"]["sharpe"] <= -0.5
             and not rec["result"].get("submission_pass")]
    print(f"[v3] BRAIN results: {n_pass}/{len(records)} passed submission gate")
    if leads:
        print(f"[v3] {len(leads)} strong-negative INVERSION lead(s) "
              f"(mechanism has edge, sign backwards — invert to flip):")
        for rec in leads:
            r = rec["result"]
            print(f"     sh={r.get('sharpe')}  {rec['alpha'].get('name')}  "
                  f"(inverted ≈ +{abs(r.get('sharpe', 0)):.2f})")
    return {"elite": elite, "records": records}


def retrain_surrogate() -> None:
    print("\n[v3] Retraining surrogate on enlarged dataset (active learning)...")
    try:
        from aq2.surrogate.train import train
        train()
    except Exception as e:
        print(f"[v3] surrogate retrain skipped: {e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aq3.run3",
                                 description="AlphaQuant v3 — evolutionary alpha search")
    ap.add_argument("--pop", type=int, default=200, help="population size")
    ap.add_argument("--generations", type=int, default=25, help="GP generations per iteration")
    ap.add_argument("--elite", type=int, default=5, help="elite genomes to BRAIN-validate")
    ap.add_argument("--iterations", type=int, default=1, help="active-learning outer iterations")
    ap.add_argument("--sweep", type=int, default=1,
                    help="per-alpha BRAIN sweep budget; >=1 auto-inverts strong-negatives (default 1)")
    ap.add_argument("--max-concurrent", type=int, default=3,
                    help="parallel BRAIN sims; cap at your account's slot limit (default 3)")
    ap.add_argument("--no-simulate", action="store_true", help="local evolve only, no BRAIN")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (default: time-free fixed 0)")
    # Hypothesis pre-generation stage flags.
    ap.add_argument("--no-hypotheses", action="store_true",
                    help="skip the HypothesisEngine pre-generation stage")
    ap.add_argument("--archetype", type=str, default=None,
                    help="constrain hypothesis generation to this archetype")
    ap.add_argument("--theme", type=str, default=None,
                    help="free-text research theme passed to hypothesis generation")
    ap.add_argument("--hyp-count", type=int, default=10, dest="hyp_count",
                    help="number of hypotheses to generate (before novelty filter)")
    ap.add_argument("--hyp-top", type=int, default=3, dest="hyp_top",
                    help="top-N hypotheses (by novelty) passed as intent to the generation loop")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed if args.seed is not None else 0)
    me = MathEngine()

    # ---- Hypothesis pre-generation stage (runs BEFORE the evolution loop) ----
    hypotheses = run_hypothesis_stage(args)

    session = None
    if not args.no_simulate:
        try:
            from brain_api import BrainSession, load_credentials
            creds = load_credentials()
            session = BrainSession(creds["email"], creds["password"])
        except Exception as e:
            print(f"[v3] BRAIN unavailable ({e}). Falling back to --no-simulate.")
            args.no_simulate = True

    for it in range(args.iterations):
        if args.iterations > 1:
            print(f"\n########## ACTIVE-LEARNING ITERATION {it+1}/{args.iterations} ##########")
        run_iteration(rng, args, me, session=session, hypotheses=hypotheses)
        if not args.no_simulate and args.iterations > 1:
            retrain_surrogate()

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
