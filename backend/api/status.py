"""
status.py — Server-side forge status tracker.

Any device that opens the UI polls GET /api/status every 2 seconds.
The state is updated by websocket.py, forge.py, and jobs.py at each phase boundary.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["status"])

_lock = threading.Lock()
_started_ts: float | None = None

_state: dict[str, Any] = {
    "phase": "idle",
    "job_id": None,
    "label": "",
    "intent": "",
    "archetype": "",
    "count": 0,
    "current": 0,
    "elapsed_s": 0.0,
    "provider": "",
    "think_mode": "adaptive",
    "last_updated": None,
}


def set_running(
    phase: str,
    *,
    intent: str = "",
    archetype: str = "",
    count: int = 0,
    current: int = 0,
    provider: str = "",
    think_mode: str = "adaptive",
    job_id: str | None = None,
    label: str = "",
) -> None:
    global _started_ts
    with _lock:
        was_idle = _state["phase"] == "idle"
        if was_idle:
            _started_ts = time.time()
        _state.update({
            "phase": phase,
            "job_id": job_id,
            "label": label or intent[:60],
            "intent": intent,
            "archetype": archetype,
            "count": count,
            "current": current,
            "provider": provider,
            "think_mode": think_mode,
            "elapsed_s": round(time.time() - _started_ts, 1) if _started_ts else 0.0,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })


def tick(current: int) -> None:
    """Update the current alpha index without changing anything else."""
    with _lock:
        _state["current"] = current
        if _started_ts:
            _state["elapsed_s"] = round(time.time() - _started_ts, 1)
        _state["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def set_idle() -> None:
    global _started_ts
    with _lock:
        _started_ts = None
        _state.update({
            "phase": "idle",
            "job_id": None,
            "label": "",
            "intent": "",
            "archetype": "",
            "count": 0,
            "current": 0,
            "elapsed_s": 0.0,
            "provider": "",
            "think_mode": "adaptive",
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })


def get() -> dict[str, Any]:
    with _lock:
        s = dict(_state)
    if s["phase"] != "idle" and _started_ts:
        s["elapsed_s"] = round(time.time() - _started_ts, 1)
    return s


@router.get("/status")
async def forge_status() -> dict[str, Any]:
    return get()
