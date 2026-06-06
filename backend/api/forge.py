from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.services.brain_service import simulate_alpha
from backend.services.openrouter_service import OpenRouterServiceError, generate_alphas, repair_alpha, research
from backend.services.validator_service import normalize_settings, reconcile_neutralization, validate_alpha
from backend.api.repair import _build_sim_feedback, _both_negative, _negate_expression
from backend.services.evolution_service import get_evolution_service
from backend.services.storage_service import get_storage_service


router = APIRouter(tags=["forge"])

SIM_CONCURRENCY = 2
# Max post-sim repair attempts per alpha when fitness/sharpe gates fail.
FITNESS_REPAIR_ATTEMPTS = int(__import__("os").getenv("ALPHAFORGE_FITNESS_REPAIR_ATTEMPTS", "2"))


class ForgeRequest(BaseModel):
    intent: str = ""
    archetype: str = "any"
    count: int = Field(default=3, ge=1, le=100)
    web_research: bool = True
    gen_provider: str = "openrouter"
    research_provider: str = "openrouter"
    hypothesis_data: dict | None = None
    think_mode: str = "adaptive"
    think_provider: str = ""  # overrides gen_provider for deep think critics/refine; empty = use gen_provider


def _fitness_failed(simulation: dict[str, Any]) -> bool:
    """True when the alpha simulated OK but missed the fitness or sharpe gate."""
    if simulation.get("status") != "OK":
        return False
    sharpe = simulation.get("sharpe")
    fitness = simulation.get("fitness")
    sharpe_ok = isinstance(sharpe, (int, float)) and sharpe >= 1.25
    fitness_ok = isinstance(fitness, (int, float)) and fitness >= 1.0
    return not (sharpe_ok and fitness_ok)


def _corr_failed(simulation: dict[str, Any]) -> bool:
    """True when simulation completed OK (good Sharpe/Fitness) but SELF_CORRELATION failed."""
    if simulation.get("status") != "OK":
        return False
    sc = simulation.get("self_correlation") or {}
    return isinstance(sc, dict) and sc.get("result") == "FAIL"


def _fitness_repair_feedback(simulation: dict[str, Any]) -> str:
    """Produce a specific, actionable repair brief focused on the fitness gate."""
    sharpe = simulation.get("sharpe")
    fitness = simulation.get("fitness")
    turnover = simulation.get("turnover_pct")
    lines = [_build_sim_feedback(simulation, simulation.get("failures", []))]
    # Diagnose the likely cause and prescribe a fix.
    if isinstance(fitness, (int, float)) and fitness < 1.0:
        if isinstance(turnover, (int, float)) and turnover > 35:
            lines.append(
                f"PRIMARY CAUSE: turnover is too high ({turnover:.1f}%). "
                "Reduce it to 5-35%: (a) extend all time-series windows by 2-3x, "
                "(b) switch to a fundamentals-based leg (ebit, sales, equity ratios), "
                "(c) increase decay to 8-10, or (d) wrap the signal in ts_decay_linear(x, 10)."
            )
        elif isinstance(turnover, (int, float)) and turnover < 5:
            lines.append(
                f"PRIMARY CAUSE: turnover is too low ({turnover:.1f}%). "
                "The signal barely moves day-to-day — add a daily-varying leg such as "
                "rank(ts_delta(close,1)/ts_mean(close,5)) or ts_zscore(returns, 5)."
            )
        elif isinstance(sharpe, (int, float)) and sharpe < 1.25:
            lines.append(
                f"PRIMARY CAUSE: Sharpe {sharpe:.2f} is below 1.25. "
                "Strengthen the signal: (a) add group_neutralize to remove sector beta, "
                "(b) combine with a negatively correlated second leg, "
                "(c) try -ts_mean(returns*returns*returns, 21) or rank(ts_corr(returns, volume, 10))."
            )
    return "\n".join(lines)


async def _simulate_with_fitness_repair(
    item: dict[str, Any], provider: str
) -> None:
    """Simulate an alpha, then repair and re-simulate up to FITNESS_REPAIR_ATTEMPTS
    times when the fitness or sharpe gate is not met. Updates item in-place."""
    expression = item["expression"]
    settings = item.get("settings", {})

    simulation = await run_in_threadpool(simulate_alpha, expression, settings, item)
    item["simulation"] = simulation

    for attempt in range(FITNESS_REPAIR_ATTEMPTS):
        if not _fitness_failed(simulation):
            break

        # Fast path: both metrics negative → just negate.
        if _both_negative(simulation):
            expression = _negate_expression(expression)
        else:
            feedback = _fitness_repair_feedback(simulation)
            try:
                fix = await repair_alpha(
                    expression,
                    issues=item.get("validation", {}).get("issues", []),
                    intent=item.get("hypothesis", ""),
                    archetype=item.get("archetype", ""),
                    sim_feedback=feedback,
                    provider=provider,
                )
                expression = fix["expression"]
                item["repair_notes"] = item.get("repair_notes", []) + [
                    {"attempt": attempt + 1, "change_note": fix.get("change_note", "")}
                ]
            except OpenRouterServiceError:
                break  # Repair failed — keep current best.

        # Re-validate and re-simulate the repaired expression.
        item["expression"] = expression
        item["settings"] = reconcile_neutralization(expression, normalize_settings(item.get("settings")))
        item["validation"] = validate_alpha({"expression": expression, "settings": item["settings"]}).get("validation", {})
        simulation = await run_in_threadpool(simulate_alpha, expression, item["settings"], item)
        item["simulation"] = simulation

    # If fitness/sharpe pass but SELF_CORRELATION fails, do one diversity-focused
    # regeneration with a modified intent that forces use of underexplored fields.
    if _corr_failed(simulation) and not item.get("_diversity_regen_done"):
        sc_val = (simulation.get("self_correlation") or {}).get("value")
        sc_str = f"{sc_val:.3f}" if isinstance(sc_val, float) else "high"
        diversity_intent = (
            f"[DIVERSITY — self-correlation {sc_str} too high, MUST use non-Price-Volume fields] "
            + (item.get("hypothesis") or item.get("name") or "novel alpha")
        )
        try:
            new_alphas = await generate_alphas(
                diversity_intent,
                item.get("archetype", "any"),
                1,
                brief="",
                provider=provider,
            )
            if new_alphas:
                new_item = validate_alpha(new_alphas[0])
                new_item["_diversity_regen_done"] = True
                new_sim = await run_in_threadpool(
                    simulate_alpha, new_item["expression"], new_item.get("settings", {}), new_item
                )
                new_item["simulation"] = new_sim
                new_item["simulation_state"] = "completed" if new_sim.get("status") == "OK" else "failed"
                new_item["change_note"] = f"diversity regen (orig self-corr={sc_str})"
                # Replace the item's data with the diversity regen result.
                item.update(new_item)
                return
        except Exception:  # noqa: BLE001
            pass  # keep original if regen fails

    status = simulation.get("status")
    item["simulation_state"] = "completed" if status == "OK" else "failed"

    succeeded = status == "OK" and not _fitness_failed(simulation)
    evo = get_evolution_service()
    storage = get_storage_service()

    # Genetic population + self-optimizer feedback
    evo.record_result(
        expression=item.get("expression", ""),
        sharpe=simulation.get("sharpe"),
        fitness=simulation.get("fitness"),
        turnover=simulation.get("turnover_pct"),
        success=succeeded,
    )

    # Bandit update — reward = normalised sharpe (0-1 scale, clamped)
    bandit_action = item.pop("_bandit_action", None)
    region = item.get("settings", {}).get("region", "USA")
    reward = min(max((simulation.get("sharpe") or 0.0) / 3.0, 0.0), 1.0)
    evo.update_bandit(bandit_action, reward, region)

    # Quality monitor — track per alpha_id over time
    alpha_id = simulation.get("alpha_id", "")
    if alpha_id:
        health = evo.track_quality(
            alpha_id=alpha_id,
            sharpe=simulation.get("sharpe") or 0.0,
            fitness=simulation.get("fitness") or 0.0,
            returns=simulation.get("returns") or 0.0,
            turnover=simulation.get("turnover_pct"),
        )
        if health:
            item["_quality"] = health

    # Persist to SQLite
    storage.store(item, simulation)


@router.post("/forge")
async def forge(payload: ForgeRequest) -> dict[str, Any]:
    brief = ""
    research_payload: dict[str, Any] | None = None
    if payload.web_research:
        try:
            result = await research(payload.intent, payload.archetype, provider=payload.research_provider)
            brief = result.get("brief", "")
            research_payload = {"brief": brief, "sources": result.get("sources", [])}
        except Exception:  # noqa: BLE001
            research_payload = None

    evo = get_evolution_service()
    storage = get_storage_service()

    # Get bandit hints — only meaningful after 200+ simulations
    stats = storage.get_stats()
    bandit_hints = (
        evo.get_bandit_hints(
            region="USA",
            operator_success_rates=storage.get_operator_success_rates(),
        )
        if stats.get("total_simulations", 0) >= 200
        else {}
    )

    try:
        generated = await generate_alphas(
            payload.intent,
            payload.archetype,
            payload.count,
            brief=brief,
            provider=payload.gen_provider,
            hypothesis_data=payload.hypothesis_data or None,
            think_mode=payload.think_mode,
            think_provider=payload.think_provider or payload.gen_provider,
            bandit_hints=bandit_hints or None,
        )
    except OpenRouterServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Tag each alpha with the bandit action so record_result can update it
    bandit_action = bandit_hints.get("action") if bandit_hints else None
    for alpha in generated:
        alpha["_bandit_action"] = bandit_action

    # Only breed once we have enough history to make mutations meaningful.
    # Before 100 stored results, random mutation mostly wastes simulation slots.
    stats = storage.get_stats()
    evolution_variants = (
        evo.breed(generated, variants_per_alpha=2)
        if stats.get("total_simulations", 0) >= 100
        else []
    )
    all_candidates = generated + evolution_variants

    # ── Uniqueness filter — drop exact duplicates and near-duplicates ──────────
    # Saves simulation slots and avoids WQ self-correlation failures.
    unique_candidates: list[dict[str, Any]] = []
    seen_exprs: set[str] = set()
    for alpha in all_candidates:
        expr = (alpha.get("expression") or "").strip()
        if not expr or expr in seen_exprs:
            continue
        region = alpha.get("settings", {}).get("region", "USA")
        # Exact match in DB
        if storage.has_been_simulated(expr, region):
            continue
        # Near-duplicate check (similarity > 0.85 against recent DB entries)
        try:
            similar = storage._db.check_template_similarity(expr, region, similarity_threshold=0.85, max_check=100) if storage._enabled else []
            if similar:
                continue
        except Exception:
            pass
        seen_exprs.add(expr)
        unique_candidates.append(alpha)

    forged: list[dict[str, Any]] = [validate_alpha(alpha) for alpha in unique_candidates]

    semaphore = asyncio.Semaphore(SIM_CONCURRENCY)

    async def simulate_one(item: dict[str, Any]) -> None:
        async with semaphore:
            await _simulate_with_fitness_repair(item, payload.gen_provider)

    await asyncio.gather(*(simulate_one(item) for item in forged))
    return {"alphas": forged, "research": research_payload}
