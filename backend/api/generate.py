from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.openrouter_service import OpenRouterServiceError, generate_alphas
from backend.services.validator_service import validate_alpha


router = APIRouter(tags=["generate"])


class GenerateRequest(BaseModel):
    intent: str = ""
    archetype: str = "any"
    count: int = Field(default=3, ge=1, le=100)
    provider: str = "openrouter"


@router.post("/generate")
async def generate(payload: GenerateRequest) -> dict[str, list[dict[str, Any]]]:
    try:
        alphas = await generate_alphas(payload.intent, payload.archetype, payload.count, provider=payload.provider)
    except OpenRouterServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"alphas": [validate_alpha(alpha) for alpha in alphas]}

