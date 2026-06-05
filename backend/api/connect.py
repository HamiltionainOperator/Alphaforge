from __future__ import annotations

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from backend.services import brain_service


router = APIRouter(tags=["connect"])


@router.post("/connect")
async def connect_brain() -> dict:
    try:
        return await run_in_threadpool(brain_service.connect)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/connect")
async def brain_status() -> dict:
    return brain_service.get_status()

