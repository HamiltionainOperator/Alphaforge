from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.services.brain_service import fetch_details, submit_alpha_to_pool


router = APIRouter(tags=["alpha"])


class AlphaIdRequest(BaseModel):
    alpha_id: str


@router.post("/alpha/details")
async def alpha_details(payload: AlphaIdRequest) -> dict[str, Any]:
    """Fetch everything about a simulated alpha: stats, checks, correlations,
    the PnL curve, and a submittable/failures verdict. Returns 200 with the
    payload even on a soft error so the UI can surface exactly what happened."""
    if not payload.alpha_id.strip():
        raise HTTPException(status_code=422, detail="alpha_id is required")
    return await run_in_threadpool(fetch_details, payload.alpha_id.strip())


@router.post("/alpha/submit")
async def alpha_submit(payload: AlphaIdRequest) -> dict[str, Any]:
    """Submit a simulated alpha to the BRAIN alpha pool. This is a real,
    account-level action — the frontend confirms before calling it. Always
    returns 200 with a status (SUBMITTED / REJECTED / SUBMIT_ERROR / ...)."""
    if not payload.alpha_id.strip():
        raise HTTPException(status_code=422, detail="alpha_id is required")
    return await run_in_threadpool(submit_alpha_to_pool, payload.alpha_id.strip())
