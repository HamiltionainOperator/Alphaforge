from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.services.brain_service import simulate_alpha
from backend.services.validator_service import normalize_settings


router = APIRouter(tags=["simulate"])


class SimulateRequest(BaseModel):
    expression: str
    settings: dict[str, Any] = Field(default_factory=dict)


@router.post("/simulate")
async def simulate(payload: SimulateRequest) -> dict[str, Any]:
    expression = payload.expression.strip()
    if not expression:
        raise HTTPException(status_code=422, detail="expression is required")
    settings = normalize_settings(payload.settings)
    return await run_in_threadpool(simulate_alpha, expression, settings, {"expression": expression, "settings": settings})

