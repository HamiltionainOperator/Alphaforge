"""
aq3.seed — build a starting population grounded in real evidence.

The single highest-EV thing aq3 can do is NOT start cold. We seed the
population from:
  1. Near-misses    — feedback.json entries with sharpe >= NEAR_SHARPE. These are
                      structures already close to passing; one mutation may finish them.
  2. Graduates      — memory.json winners (proven structures to recombine).
  3. Paper anchors  — expressions from papers_kb/ hypotheses (economic priors).
Everything that parses into the typed grammar becomes a seed genome; the rest of
the population is filled with random genomes for exploration.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from aq3.genome import Genome

_REPO = Path(__file__).resolve().parent.parent
_FEEDBACK = _REPO / "data" / "feedback.json"
_MEMORY = _REPO / "data" / "memory.json"
_PAPERS_KB = _REPO / "data" / "papers_kb"

NEAR_SHARPE = 1.0   # feedback entries at/above this are "near-misses" worth seeding


def _entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        doc = json.loads(path.read_text())
        return doc.get("entries", []) if isinstance(doc, dict) else doc
    except Exception:
        return []


def _genomes_from_expressions(exprs: list[str]) -> list[Genome]:
    out = []
    for e in exprs:
        g = Genome.from_fastexpr(e)
        if g is not None:
            out.append(g)
    return out


def collect_seed_expressions() -> dict[str, list[str]]:
    """Return seed expressions grouped by source (for reporting)."""
    near = [e["expression"] for e in _entries(_FEEDBACK)
            if isinstance(e.get("sharpe"), (int, float)) and e["sharpe"] >= NEAR_SHARPE
            and e.get("expression")]
    grads = [e["expression"] for e in _entries(_MEMORY) if e.get("expression")]
    papers = []
    if _PAPERS_KB.exists():
        for p in sorted(_PAPERS_KB.glob("*.json")):
            try:
                doc = json.loads(p.read_text())
                for h in doc.get("hypotheses", []):
                    if h.get("expression"):
                        papers.append(h["expression"])
            except Exception:
                continue
    return {"near_miss": near, "graduate": grads, "paper": papers}


def build_population(rng: random.Random, size: int = 200,
                     max_depth: int = 4) -> tuple[list[Genome], dict]:
    """Build the initial population. Returns (population, seed_report)."""
    sources = collect_seed_expressions()
    seeds: list[Genome] = []
    report = {}
    for name, exprs in sources.items():
        gs = _genomes_from_expressions(exprs)
        # dedup by signature within source
        seen, uniq = set(), []
        for g in gs:
            s = g.signature()
            if s not in seen:
                seen.add(s)
                uniq.append(g)
        report[name] = {"raw": len(exprs), "parsed": len(gs), "unique": len(uniq)}
        seeds.extend(uniq)

    # global dedup across seeds
    seen, uniq_seeds = set(), []
    for g in seeds:
        s = g.signature()
        if s not in seen:
            seen.add(s)
            uniq_seeds.append(g)

    population = list(uniq_seeds[:size])
    # fill remainder with random exploration genomes
    while len(population) < size:
        population.append(Genome.random(rng, max_depth=rng.choice([3, max_depth, max_depth + 1])))

    report["total_seeds"] = len(uniq_seeds)
    report["random_fill"] = max(0, size - len(uniq_seeds))
    report["population"] = len(population)
    return population, report


if __name__ == "__main__":
    rng = random.Random(0)
    pop, rep = build_population(rng, size=200)
    print("[seed] population report:")
    for k, v in rep.items():
        print(f"    {k}: {v}")
    print("\n[seed] sample seed genomes:")
    for g in pop[:8]:
        print(f"    {g.to_fastexpr()[:100]}")
