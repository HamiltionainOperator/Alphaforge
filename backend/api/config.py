from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.services.openrouter_service import available_providers, get_token_usage, reset_token_usage


router = APIRouter(tags=["config"])


@router.get("/providers")
async def providers() -> dict[str, Any]:
    """Report which LLM engines are usable so the UI can enable/disable options."""
    return {"providers": available_providers()}


@router.get("/usage")
async def usage() -> dict[str, Any]:
    """Cumulative token usage since the backend started (or last reset)."""
    u = get_token_usage()
    return {
        "input_tokens": u["input_tokens"],
        "output_tokens": u["output_tokens"],
        "total_tokens": u["input_tokens"] + u["output_tokens"],
        "calls": u["calls"],
    }


@router.post("/usage/reset")
async def usage_reset() -> dict[str, str]:
    reset_token_usage()
    return {"status": "reset"}
