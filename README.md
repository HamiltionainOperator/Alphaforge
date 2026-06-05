# AlphaQuant v3 — Evolutionary Alpha Search (standalone)

Self-contained alpha-generation pipeline for WorldQuant BRAIN. An alpha is a
**typed expression tree** inside a protected, always-neutralized envelope, so
hallucinated fields / broken arity / malformed alphas are structurally
impossible. A genetic-programming loop evolves candidates against a local
surrogate; BRAIN simulation is reserved for the elite. See `aq3/README.md` for
the full architecture.

## Setup
```bash
pip install -r requirements.txt
cp credentials.json.example credentials.json   # then add your BRAIN login
```

## Run
```bash
python aq3/genome.py                                                    # prove the grammar
python -m aq3.run3 --no-simulate --generations 25 --pop 250 --elite 5   # local evolve + elite
python -m aq3.run3 --iterations 3 --generations 25 --elite 5            # full loop (needs BRAIN)
```

## Layout
- `aq3/` — v3 engine: genome (typed tree), seed, evolve (GP), run3 (loop)
- `math_engine.py`, `brain_api.py` — reused validator + BRAIN client
- `judge/features.py`, `aq2/surrogate/`, `aq2/validation/` — features, surrogate, DSR/PSR selection
- `data/` — field catalog, memory/feedback, paper hypotheses
# Alphaforge
