"""
aq3.evolve — genetic-programming loop over typed genomes, scored by the local
surrogate so thousands of candidates are evaluated per second WITHOUT touching
BRAIN. BRAIN simulation is reserved for the elite that survive local evolution.

Fitness
-------
The surrogate (aq2) predicts OS Sharpe and fitness from the 41-dim feature
vector. The BRAIN submission gate is fitness >= 1.0 AND sharpe >= 1.25, so we
optimize a combined score that rewards clearing both, plus a small structural
novelty bonus so the population doesn't collapse onto one surrogate-favored
shape (the surrogate is a NOISY proxy — diversity hedges its errors).

    score = w_fit * pred_fitness + w_sh * pred_sharpe + novelty_bonus

Honest caveat: optimizing a noisy surrogate will partly overfit its errors.
That is *why* the elite still go to BRAIN and the surrogate retrains on every
real result (active learning, in run3.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field as dc_field
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from aq3.genome import Genome, Node
from aq3.seed import build_population
from aq2.surrogate.model import AlphaSurrogate
from math_engine import MathEngine

_DEFAULT_SETTINGS = {"neutralization": "Subindustry", "decay": 6, "truncation": 0.05}


@dataclass
class EvalResult:
    genome: Genome
    expr: str
    pred_sharpe: float
    pred_fitness: float
    score: float


@dataclass
class Evolver:
    rng: random.Random
    pop_size: int = 200
    elite_frac: float = 0.15
    crossover_rate: float = 0.5
    mutation_rate: float = 0.4
    tournament_k: int = 4
    w_fit: float = 1.0
    w_sh: float = 0.5
    novelty_w: float = 0.15
    size_budget: int = 14         # inner-tree nodes allowed before parsimony penalty
    size_penalty: float = 0.05    # score docked per node over budget (anti-bloat)
    hard_size_cap: int = 20       # offspring exceeding this are rejected (turnover guard)

    def __post_init__(self):
        self._surr = AlphaSurrogate()
        self._me = MathEngine()
        self._fit_cache: dict[str, tuple[float, float]] = {}
        self._seen_signatures: set[str] = set()

    # ----------------------------------------------------- fitness
    def _surrogate_scores(self, expr: str) -> tuple[float, float]:
        if expr in self._fit_cache:
            return self._fit_cache[expr]
        sh = self._surr.predict_os_sharpe(expr, _DEFAULT_SETTINGS)
        ft = self._surr.predict_fitness(expr, _DEFAULT_SETTINGS)
        sh = float(sh) if sh is not None else 0.0
        ft = float(ft) if ft is not None else 0.0
        self._fit_cache[expr] = (sh, ft)
        return sh, ft

    def _valid(self, expr: str) -> bool:
        try:
            return self._me.critique(expr, _DEFAULT_SETTINGS).get("verdict") != "FAIL"
        except Exception:
            return False

    def evaluate(self, genome: Genome, novelty_pool: set[str]) -> EvalResult:
        expr = genome.to_fastexpr()
        if not self._valid(expr):
            return EvalResult(genome, expr, -9.0, -9.0, -9.0)
        sh, ft = self._surrogate_scores(expr)
        # structural novelty: reward signatures not already common in the pool
        sig = genome.signature()
        novelty = self.novelty_w if sig not in novelty_pool else 0.0
        # parsimony: dock bloated trees (anti-overfit-to-surrogate + lower turnover)
        bloat = self.size_penalty * max(0, genome.inner.size() - self.size_budget)
        score = self.w_fit * ft + self.w_sh * sh + novelty - bloat
        return EvalResult(genome, expr, sh, ft, score)

    @staticmethod
    def _coarse_sig(genome: Genome) -> str:
        """Operator-skeleton signature with ALL leaves abstracted — two alphas
        with the same shape but different fields collapse together. Used to keep
        the BRAIN-bound elite genuinely structurally distinct."""
        def sk(n: Node) -> str:
            if n.kind in ("field", "window", "const", "group"):
                return "."
            return f"{n.value}(" + ",".join(sk(c) for c in n.children) + ")"
        return f"{genome.bound_op}:{sk(genome.inner)}"

    # ----------------------------------------------------- selection
    def _tournament(self, scored: list[EvalResult]) -> Genome:
        contenders = self.rng.sample(scored, min(self.tournament_k, len(scored)))
        return max(contenders, key=lambda r: r.score).genome

    # ----------------------------------------------------- one generation
    def _next_generation(self, scored: list[EvalResult]) -> list[Genome]:
        scored.sort(key=lambda r: r.score, reverse=True)
        n_elite = max(1, int(self.pop_size * self.elite_frac))
        # elitism: carry the best, structurally-distinct genomes forward unchanged
        elite: list[Genome] = []
        elite_sigs: set[str] = set()
        for r in scored:
            s = r.genome.signature()
            if s not in elite_sigs:
                elite.append(r.genome)
                elite_sigs.add(s)
            if len(elite) >= n_elite:
                break

        new_pop: list[Genome] = list(elite)
        guard = 0
        while len(new_pop) < self.pop_size and guard < self.pop_size * 30:
            guard += 1
            roll = self.rng.random()
            if roll < self.crossover_rate:
                a = self._tournament(scored)
                b = self._tournament(scored)
                child = Genome.crossover(a, b, self.rng)
            elif roll < self.crossover_rate + self.mutation_rate:
                child = self._tournament(scored).mutate(self.rng)
            else:
                child = Genome.random(self.rng, max_depth=self.rng.choice([3, 4, 5]))
            # hard size cap: reject bloated offspring outright (BRAIN turnover guard)
            if child.inner.size() > self.hard_size_cap:
                continue
            new_pop.append(child)
        # backfill if the cap rejected too many
        while len(new_pop) < self.pop_size:
            new_pop.append(Genome.random(self.rng, max_depth=self.rng.choice([3, 4])))
        return new_pop

    # ----------------------------------------------------- run
    def run(self, generations: int = 30, verbose: bool = True,
            population: list[Genome] | None = None) -> list[EvalResult]:
        if population is None:
            population, _ = build_population(self.rng, size=self.pop_size)
        history = []
        for gen in range(generations):
            # build a novelty pool of signatures seen so far (common = no bonus)
            novelty_pool = set(self._seen_signatures)
            scored = [self.evaluate(g, novelty_pool) for g in population]
            for r in scored:
                self._seen_signatures.add(r.genome.signature())
            scored.sort(key=lambda r: r.score, reverse=True)
            best = scored[0]
            n_gate = sum(1 for r in scored if r.pred_fitness >= 1.0 and r.pred_sharpe >= 1.25)
            mean_score = sum(r.score for r in scored) / len(scored)
            history.append((gen, best.score, mean_score, n_gate))
            if verbose:
                print(f"  gen {gen:2d}  best_score={best.score:.3f}  "
                      f"mean={mean_score:.3f}  pred_gate_passers={n_gate:3d}  "
                      f"best: {best.expr[:62]}")
            population = self._next_generation(scored)

        # final evaluation of the last population
        novelty_pool = set(self._seen_signatures)
        final = [self.evaluate(g, novelty_pool) for g in population]
        final.sort(key=lambda r: r.score, reverse=True)
        return final

    # ----------------------------------------------------- elite selection
    def select_elite(self, scored: list[EvalResult], k: int = 5) -> list[EvalResult]:
        """Top-k COARSELY-distinct genomes — different operator skeletons, not just
        different fields. This is what BRAIN should validate: spending 5 slots on 5
        field-variants of one structure is waste; 5 distinct shapes is information."""
        out, shapes = [], set()
        for r in scored:
            shape = self._coarse_sig(r.genome)
            if shape in shapes:
                continue
            shapes.add(shape)
            out.append(r)
            if len(out) >= k:
                break
        return out


def _demo() -> None:
    rng = random.Random(7)
    ev = Evolver(rng, pop_size=200)
    print("[evolve] seeding population from real evidence + evolving 25 generations...")
    final = ev.run(generations=25, verbose=True)
    print("\n[evolve] top-5 elite (structurally distinct) for BRAIN validation:")
    for i, r in enumerate(ev.select_elite(final, k=5), 1):
        print(f"  {i}. pred_sh={r.pred_sharpe:.2f} pred_fit={r.pred_fitness:.2f} "
              f"score={r.score:.2f}")
        print(f"     {r.expr}")


if __name__ == "__main__":
    _demo()
