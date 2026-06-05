from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.services.brain_service import simulate_alpha
from backend.services.openrouter_service import OpenRouterServiceError, repair_alpha
from backend.services.validator_service import validate_alpha


router = APIRouter(tags=["repair"])


def _negate_expression(expression: str) -> str:
    """Flip the sign of the whole signal. Toggles a "-1 * ( ... )" wrapper we
    added on a previous flip so repeated negations don't nest forever."""
    expr = (expression or "").strip()
    prefix = "-1 * ("
    if expr.startswith(prefix) and expr.endswith(")"):
        inner = expr[len(prefix):-1]
        depth = 0
        balanced = True
        for ch in inner:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    balanced = False
                    break
        if balanced and depth == 0:
            return inner.strip()
    return f"-1 * ({expr})"


def _both_negative(simulation: dict[str, Any]) -> bool:
    """True when BOTH Sharpe and Fitness came back negative — the signal has the
    right structure but is pointing the wrong way, so negating it is the fix."""
    sharpe = simulation.get("sharpe")
    fitness = simulation.get("fitness")
    return (
        isinstance(sharpe, (int, float)) and sharpe < 0
        and isinstance(fitness, (int, float)) and fitness < 0
    )


class RepairRequest(BaseModel):
    expression: str
    issues: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[dict[str, Any]] = Field(default_factory=list)
    simulation: dict[str, Any] = Field(default_factory=dict)
    name: str = ""
    archetype: str = "any"
    hypothesis: str = ""
    intent: str = ""
    settings: dict[str, Any] = Field(default_factory=dict)
    expected_sharpe: str = ""
    regime_note: str = ""
    turnover_note: str = ""
    simulate: bool = True
    provider: str = "openrouter"


def _build_sim_feedback(simulation: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    metrics = []
    for key, label in (("sharpe", "Sharpe"), ("fitness", "Fitness"), ("turnover_pct", "Turnover%")):
        val = simulation.get(key)
        if isinstance(val, (int, float)):
            metrics.append(f"{label}={val}")
    if metrics:
        lines.append("Measured: " + ", ".join(metrics))
    status = simulation.get("status")
    if status and status != "OK":
        err = simulation.get("error")
        lines.append(f"BRAIN returned status {status}" + (f": {str(err)[:200]}" if err else ""))
    for f in failures or []:
        if isinstance(f, dict) and (f.get("message") or f.get("name")):
            lines.append(f"- {f.get('message') or f.get('name')}")
    return "\n".join(lines)


@router.post("/repair")
async def repair(payload: RepairRequest) -> dict[str, Any]:
    if not payload.expression.strip():
        raise HTTPException(status_code=422, detail="expression is required")

    # Shortcut: if Sharpe AND Fitness are both negative, the edge is simply
    # inverted — multiply the whole expression by -1 (no LLM call needed).
    if _both_negative(payload.simulation):
        fix = {
            "expression": _negate_expression(payload.expression),
            "change_note": (
                f"Multiplied the whole signal by -1 — Sharpe "
                f"({payload.simulation.get('sharpe')}) and Fitness "
                f"({payload.simulation.get('fitness')}) were both negative, so the edge was inverted."
            ),
        }
    else:
        sim_feedback = _build_sim_feedback(payload.simulation, payload.failures)
        try:
            fix = await repair_alpha(
                payload.expression,
                payload.issues,
                intent=payload.intent,
                archetype=payload.archetype,
                sim_feedback=sim_feedback,
                provider=payload.provider,
            )
        except OpenRouterServiceError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    alpha = {
        "name": payload.name or "repaired alpha",
        "archetype": payload.archetype if payload.archetype != "any" else "novel",
        "hypothesis": payload.hypothesis,
        "expression": fix["expression"],
        "settings": payload.settings,
        "expected_sharpe": payload.expected_sharpe,
        "regime_note": payload.regime_note,
        "turnover_note": payload.turnover_note,
    }
    item = validate_alpha(alpha)
    item["repaired"] = True
    item["change_note"] = fix.get("change_note", "")

    simulation: dict[str, Any] = {"status": "queued"}
    if payload.simulate:
        try:
            simulation = await run_in_threadpool(
                simulate_alpha, item["expression"], item.get("settings", {}), item
            )
        except Exception as exc:  # noqa: BLE001
            simulation = {"status": "EXCEPTION", "error": f"{type(exc).__name__}: {exc}"}

    item["simulation"] = simulation
    status = simulation.get("status")
    item["simulation_state"] = "completed" if status == "OK" else ("queued" if status == "queued" else "failed")
    return {"alpha": item}
