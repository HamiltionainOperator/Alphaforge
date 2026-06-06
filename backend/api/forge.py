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

    status = simulation.get("status")
    item["simulation_state"] = "completed" if status == "OK" else "failed"


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
        )
    except OpenRouterServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    forged: list[dict[str, Any]] = [validate_alpha(alpha) for alpha in generated]

    semaphore = asyncio.Semaphore(SIM_CONCURRENCY)

    async def simulate_one(item: dict[str, Any]) -> None:
        async with semaphore:
            await _simulate_with_fitness_repair(item, payload.gen_provider)

    await asyncio.gather(*(simulate_one(item) for item in forged))
    return {"alphas": forged, "research": research_payload}
