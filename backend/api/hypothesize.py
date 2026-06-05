from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["hypothesize"])


class HypothesizeRequest(BaseModel):
    archetype: str = "any"
    theme: str = ""
    count: int = Field(default=5, ge=1, le=20)
    provider: str = "claude_code"


@router.post("/hypothesize")
async def hypothesize(payload: HypothesizeRequest) -> dict[str, Any]:
    try:
        from hypothesis_generator import HypothesisEngine
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"HypothesisEngine unavailable: {exc}") from exc

    try:
        from starlette.concurrency import run_in_threadpool

        archetype = payload.archetype if payload.archetype != "any" else None
        theme = payload.theme.strip() or None

        def _run():
            engine = HypothesisEngine(provider=payload.provider)
            raw = engine.generate(archetype=archetype, theme=theme, count=payload.count)
            novel = engine.filter_novel(raw)
            return sorted(novel, key=lambda h: h["novelty_score"], reverse=True)

        hypotheses = await run_in_threadpool(_run)
        return {"hypotheses": hypotheses}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
