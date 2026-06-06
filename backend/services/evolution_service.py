"""
Gen2 evolution engine integrated into AlphaForge.

What this does:
  1. breed(alphas)        — after generation, crossover + mutate the LLM-generated
                            expressions to produce extra variant candidates
  2. record_result(...)   — after each simulation, feed sharpe/fitness back into
                            the genetic population so the engine learns what works
  3. get_params()         — returns SelfOptimizer hints (mutation_rate, temperature)
                            that can be passed back to the LLM generation step

Singleton: one shared engine instance lives for the server lifetime so the
population accumulates across forge requests.
"""

from __future__ import annotations

import json
import logging
import sys
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── locate gen2 ────────────────────────────────────────────────────────────────
_GEN2_ROOT = Path(__file__).resolve().parents[3] / "worldquant-miner"
if _GEN2_ROOT.exists():
    sys.path.insert(0, str(_GEN2_ROOT))

try:
    from generation_two.evolution.alpha_evolution_engine import (
        AlphaEvolutionEngine,
        AlphaResult,
    )
    from generation_two.evolution.self_optimizer import SelfOptimizer
    from generation_two.evolution.alpha_quality_monitor import AlphaQualityMonitor
    from generation_two.evolution.advanced_bandits import (
        AdvancedBanditSystem,
        BanditContext,
    )

    _GEN2_AVAILABLE = True
    logger.info("Gen2 evolution engine loaded (full — evolution + quality + bandits).")
except ImportError as e:
    _GEN2_AVAILABLE = False
    logger.warning(f"Gen2 not found — evolution disabled. ({e})")


# ── persistence ────────────────────────────────────────────────────────────────
_STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "evolution_state.json"


def _save_state(population: list[dict], params: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_STATE_PATH, "w") as f:
            json.dump({"population": population, "params": params}, f, indent=2)
    except Exception as exc:
        logger.warning(f"Could not save evolution state: {exc}")


def _load_state() -> tuple[list[dict], dict]:
    if not _STATE_PATH.exists():
        return [], {}
    try:
        with open(_STATE_PATH) as f:
            data = json.load(f)
        return data.get("population", []), data.get("params", {})
    except Exception:
        return [], {}


# ── service ────────────────────────────────────────────────────────────────────
class EvolutionService:
    """
    Thin wrapper around gen2's AlphaEvolutionEngine + SelfOptimizer.
    Safe to use even when gen2 is not installed (all methods become no-ops).
    """

    def __init__(self) -> None:
        self._enabled = _GEN2_AVAILABLE
        if not self._enabled:
            return

        self._engine = AlphaEvolutionEngine(mutation_rate=0.15, crossover_rate=0.7)
        self._optimizer = SelfOptimizer(optimization_interval=20)

        # Restore previous population if any
        saved_pop, saved_params = _load_state()
        if saved_pop:
            alpha_results = [
                AlphaResult(
                    template=p["expression"],
                    sharpe=p.get("sharpe", 1.0),
                    fitness=p.get("fitness", 1.0),
                    turnover=p.get("turnover", 20.0),
                )
                for p in saved_pop
                if p.get("expression")
            ]
            if alpha_results:
                self._engine.initialize_population(alpha_results)
                logger.info(f"Restored {len(alpha_results)} alphas from evolution state.")

        if saved_params:
            self._optimizer.current_params.update(saved_params)

        self._result_buffer: list[dict] = []  # accumulate before optimizer update

        # Quality monitor — tracks per-alpha health over time
        self._quality_monitor = AlphaQualityMonitor(monitoring_window=30)

        # Bandit system — learns which operators/strategies win
        self._bandit = AdvancedBanditSystem()
        self._sim_count = 0
        self._success_count = 0

    # ── public API ─────────────────────────────────────────────────────────────

    def breed(
        self,
        alphas: list[dict[str, Any]],
        variants_per_alpha: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Take LLM-generated alpha dicts, produce crossover+mutation variants.
        Returns only the NEW variants (originals are not duplicated).
        Falls back to [] if gen2 is unavailable or population is too small.
        """
        if not self._enabled:
            return []

        expressions = [a.get("expression", "") for a in alphas if a.get("expression")]
        if not expressions:
            return []

        variants: list[dict[str, Any]] = []
        params = self._optimizer.current_params

        for i, expr in enumerate(expressions):
            for _ in range(variants_per_alpha):
                try:
                    # Crossover: if we have a population, breed with a top performer
                    if len(self._engine.population) >= 2:
                        parents = self._engine.select_parents(2)
                        if parents and len(parents) >= 2:
                            child_expr = self._engine.crossover(expr, parents[0])
                        else:
                            child_expr = self._engine.mutate(expr)
                    else:
                        child_expr = self._engine.mutate(expr)

                    if not child_expr or child_expr == expr:
                        continue

                    # Skip if too similar to parent (SequenceMatcher ratio > 0.92)
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, expr, child_expr).ratio()
                    if similarity > 0.92:
                        continue

                    # Clone the parent alpha dict, swap expression
                    variant = deepcopy(alphas[i])
                    variant["expression"] = child_expr
                    variant["name"] = variant.get("name", "") + " [evolved]"
                    variant["_evolved"] = True
                    variant["_parent_expr"] = expr
                    variants.append(variant)
                except Exception as exc:
                    logger.debug(f"Evolution step skipped: {exc}")

        logger.info(
            f"Evolution: {len(expressions)} originals → "
            f"{len(variants)} variants (mutation_rate={params.get('mutation_rate', 0.15):.2f})"
        )
        return variants

    def record_result(
        self,
        expression: str,
        sharpe: float | None,
        fitness: float | None,
        turnover: float | None,
        success: bool = False,
    ) -> None:
        """Feed a simulation result back into the genetic population."""
        if not self._enabled:
            return
        if not expression:
            return

        sharpe = sharpe or 0.0
        fitness = fitness or 0.0
        turnover = turnover or 50.0

        self._sim_count += 1
        if success:
            self._success_count += 1

        result = AlphaResult(
            template=expression,
            sharpe=sharpe,
            fitness=fitness,
            turnover=turnover,
            success=success,
        )

        # Update fitness in the engine
        self._engine.update_fitness(expression, result)

        # Buffer for optimizer
        self._result_buffer.append({
            "expression": expression,
            "sharpe": sharpe,
            "fitness": fitness,
            "turnover": turnover,
            "success": success,
        })

        # Run optimizer every 20 results
        if len(self._result_buffer) >= 20:
            self._run_optimizer()
            self._result_buffer = []

        # Persist if this was a winner
        if success and sharpe >= 1.25 and fitness >= 1.0:
            self._persist_winner(expression, sharpe, fitness, turnover)

    def get_params(self) -> dict[str, float]:
        """Return current optimized parameters for use in generation."""
        if not self._enabled:
            return {}
        return dict(self._optimizer.current_params)

    def get_bandit_hints(
        self,
        region: str = "USA",
        operator_success_rates: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Ask the bandit system which operators/strategy to prefer.
        Returns hints dict to inject into the LLM generation prompt.
        """
        if not self._enabled:
            return {}
        try:
            import datetime
            hour = datetime.datetime.now().hour
            if hour < 9:
                tod = "morning"
            elif hour < 12:
                tod = "afternoon"
            elif hour < 17:
                tod = "evening"
            else:
                tod = "night"

            success_rate = self._success_count / max(self._sim_count, 1)
            explore_phase = (
                "early" if self._sim_count < 50
                else "mid" if self._sim_count < 200
                else "late"
            )

            ctx = BanditContext(
                region=region,
                time_of_day=tod,
                market_volatility=0.5,
                recent_performance=success_rate,
                exploration_phase=explore_phase,
                total_simulations=self._sim_count,
                successful_simulations=self._success_count,
                persona_diversity=0.5,
                operator_usage_distribution=operator_success_rates or {},
            )
            action = self._bandit.select_action(ctx)
            stats = self._bandit.get_statistics()
            return {
                "action": action,
                "exploration_rate": stats.get("current_exploration", 0.3),
                "best_strategy": stats.get("best_strategy", ""),
                "persona": action.get("persona", {}),
                "path": action.get("path", {}),
            }
        except Exception as exc:
            logger.debug(f"get_bandit_hints error: {exc}")
            return {}

    def update_bandit(self, action: dict[str, Any], reward: float, region: str = "USA") -> None:
        """Feed simulation outcome back into the bandit system."""
        if not self._enabled or not action:
            return
        try:
            import datetime
            hour = datetime.datetime.now().hour
            tod = "morning" if hour < 9 else "afternoon" if hour < 12 else "evening" if hour < 17 else "night"
            ctx = BanditContext(
                region=region,
                time_of_day=tod,
                market_volatility=0.5,
                recent_performance=reward,
                exploration_phase="mid",
                total_simulations=self._sim_count,
                successful_simulations=self._success_count,
                persona_diversity=0.5,
                operator_usage_distribution={},
            )
            self._bandit.update(action, reward, ctx)
        except Exception as exc:
            logger.debug(f"update_bandit error: {exc}")

    def track_quality(
        self,
        alpha_id: str,
        sharpe: float,
        fitness: float,
        returns: float,
        turnover: float | None = None,
    ) -> dict[str, Any]:
        """
        Track alpha performance over time. Returns health info.
        Call this every time an alpha_id is simulated (across forge sessions).
        """
        if not self._enabled or not alpha_id:
            return {}
        try:
            self._quality_monitor.track_alpha(alpha_id, {
                "sharpe": sharpe,
                "fitness": fitness,
                "returns": returns,
                "turnover": turnover,
            })
            stats = self._quality_monitor.get_alpha_statistics(alpha_id)
            if stats:
                stats["is_degrading"] = self._quality_monitor.detect_degradation(alpha_id)
                stats["health_score"] = self._quality_monitor.get_alpha_health_score(alpha_id)
            return stats or {}
        except Exception as exc:
            logger.debug(f"track_quality error: {exc}")
            return {}

    def get_quality_report(self) -> list[dict[str, Any]]:
        """Return health report for all tracked alphas."""
        if not self._enabled:
            return []
        report = []
        for alpha_id in self._quality_monitor.get_all_alpha_ids():
            stats = self._quality_monitor.get_alpha_statistics(alpha_id)
            if stats:
                stats["alpha_id"] = alpha_id
                stats["is_degrading"] = self._quality_monitor.detect_degradation(alpha_id)
                report.append(stats)
        return sorted(report, key=lambda x: x.get("health_score", 0))

    # ── private ────────────────────────────────────────────────────────────────

    def _run_optimizer(self) -> None:
        successful = [r for r in self._result_buffer if r["success"]]
        success_rate = len(successful) / max(len(self._result_buffer), 1)
        avg_sharpe = (
            sum(r["sharpe"] for r in successful) / len(successful)
            if successful else 0.0
        )
        params = self._optimizer.optimize_parameters({
            "success_rate": success_rate,
            "avg_sharpe": avg_sharpe,
            **self._optimizer.current_params,
        })
        if params:
            logger.info(
                f"SelfOptimizer updated: "
                f"mutation={params.get('mutation_rate', 0):.3f} "
                f"temp={params.get('temperature', 0):.3f} "
                f"explore={params.get('exploration_rate', 0):.3f} "
                f"(success_rate={success_rate:.1%})"
            )
            self._engine.mutation_rate = params.get("mutation_rate", self._engine.mutation_rate)
            _save_state(
                [{"expression": r.template, "sharpe": r.sharpe, "fitness": r.fitness, "turnover": r.turnover}
                 for r in self._engine.population],
                params,
            )

    def _persist_winner(
        self, expression: str, sharpe: float, fitness: float, turnover: float
    ) -> None:
        saved_pop, saved_params = _load_state()
        # Avoid duplicates
        if not any(p.get("expression") == expression for p in saved_pop):
            saved_pop.append({"expression": expression, "sharpe": sharpe, "fitness": fitness, "turnover": turnover})
            _save_state(saved_pop, saved_params)


# ── singleton ──────────────────────────────────────────────────────────────────
_service: EvolutionService | None = None


def get_evolution_service() -> EvolutionService:
    global _service
    if _service is None:
        _service = EvolutionService()
    return _service
