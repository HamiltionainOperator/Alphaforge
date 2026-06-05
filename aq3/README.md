# AlphaQuant v3 — Evolutionary Alpha Search

v3 is a ground-up redesign of the *generation* layer. v1/v2 had an LLM free-write
FastExpr strings, which hallucinated fields, broke operator arity, and produced
structural clones — every weak candidate burning a 2–5 minute BRAIN slot. v3
removes the LLM from the inner loop entirely.

## The core bet

An alpha is **not a string** — it is a **typed expression tree** built only from
validated catalog primitives, wrapped in a protected, always-neutralized envelope:

```
[-1 *] group_neutralize( bound_op( <inner signal tree> ), group )
```

Because every alpha is constructed from typed primitives inside this envelope,
the entire class of v1/v2 failures is **structurally impossible to represent**:

- hallucinated field → can't; leaves are drawn from a curated 86-field pool
- wrong operator arity → can't; arg shapes are parsed from `operators.json`
- ranking a VECTOR field → can't; only MATRIX fields are in the pool
- missing window arg → can't; `ts_*` ops always carry a typed WINDOW child
- raw field outside neutralization → can't; the envelope is invariant under evolution

Self-test result: **998/1000 random genomes pass `math_engine.critique`** (the
~0.2% are data-driven `learned_bad_pair` hits, not grammar gaps).

## The loop

```
evolve locally (hundreds of genomes × tens of generations, ~20s on the M4)
   scored by the LOCAL SURROGATE  (no BRAIN calls)
        → top-K structurally-DISTINCT elite
        → BRAIN-simulate ONLY the elite        (the only true fitness signal)
        → results route to memory.json / feedback.json
        → RETRAIN the surrogate on the enlarged data   (active learning)
        → repeat
```

BRAIN slots are spent only on candidates that survived a full local evolution.
Every real result sharpens the surrogate that guides the next round.

## Modules

| File | Role |
|---|---|
| `genome.py` | Typed expression tree. `random()`, `mutate()`, `crossover()`, `from_fastexpr()` (seed from winners), `to_fastexpr()`, `signature()`. Protected envelope enforced. |
| `primitives.json` | Curated, quality-filtered pools: 86 fields (MATRIX, real coverage, proven by `alpha_count`), 21 operators with parsed arg-shapes, groups, windows, consts. |
| `seed.py` | Builds the starting population from **real evidence**: near-misses (feedback.json sharpe ≥ 1.0), graduates (memory.json), and paper hypotheses. 236 evidence-grounded seeds, zero random fill at pop 200. |
| `evolve.py` | The GP loop. Surrogate fitness + structural-novelty bonus − parsimony penalty, tournament selection, elitism, hard size cap (turnover guard), coarse-structural elite diversity. |
| `run3.py` | The active-learning runner: evolve → BRAIN-validate elite → retrain surrogate → repeat. |

## Reused from v1/v2 (unchanged)

BRAIN client (`brain_api.py`), the DSR/PSR/correlation selection layer
(`aq2/validation/`), the 41-dim feature extractor (`judge/features.py`), and the
surrogate model (`aq2/surrogate/`). v3 replaces only generation + search.

## Usage

```bash
# Prove the grammar: 998/1000 random genomes are valid FastExpr
python aq3/genome.py

# Inspect the evidence-grounded seed population
python -m aq3.seed

# Local evolution only (no BRAIN) — evolve + print the elite
python -m aq3.run3 --no-simulate --generations 25 --pop 250 --elite 5

# One iteration: evolve locally, BRAIN-validate the 5 elite (needs credentials)
python -m aq3.run3 --generations 25 --elite 5 --sweep 2

# Full active-learning loop: evolve → validate → retrain → repeat
python -m aq3.run3 --iterations 3 --generations 25 --elite 5
```

## Honest limitations

- **The surrogate is a noisy guide.** CV RMSE ≈ 0.71 on Sharpe, 0.40 on fitness.
  Evolving against it partly overfits its errors, and the elite vary run-to-run
  by RNG seed. This is *expected* — BRAIN is ground truth, the surrogate only
  decides what's worth a slot, and the active-learning loop calibrates it as real
  results arrive. Do not read `pred_fitness` as truth; read it as a ranking.
- **No local backtest.** BRAIN does not expose raw field data (by design), so
  there is no way to compute true IC locally. The surrogate is the only local
  signal available. This is the binding constraint the whole architecture is
  shaped around.
- **Search ≠ edge.** Faster, cleaner search finds robust orthogonal alphas more
  efficiently; it does not manufacture edge that isn't there. The job is to spend
  BRAIN slots well and notice what holds up out-of-sample (the v2 DSR gate).
