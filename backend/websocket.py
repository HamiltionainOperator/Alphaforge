from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from backend.services.brain_service import simulate_alpha
from backend.services.openrouter_service import OpenRouterServiceError, generate_alphas, research
from backend.services.validator_service import validate_alpha
from backend.api import status as srv_status


router = APIRouter()

# How many BRAIN simulations to run at once.
SIM_CONCURRENCY = 2


@router.websocket("/ws/forge")
async def forge_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    started = time.time()
    # Sends can race once simulations run concurrently — serialize them.
    send_lock = asyncio.Lock()

    async def send(event: str, data: dict[str, Any]) -> None:
        async with send_lock:
            await _send(websocket, event, data)

    try:
        payload = await websocket.receive_json()
        intent = str(payload.get("intent") or "")
        archetype = str(payload.get("archetype") or "any")
        count = max(1, min(int(payload.get("count") or 3), 100))
        web_research = bool(payload.get("web_research", True))
        gen_provider = str(payload.get("gen_provider") or "openrouter")
        research_provider = str(payload.get("research_provider") or "openrouter")
        hypothesis_data = payload.get("hypothesis_data") or None
        if not isinstance(hypothesis_data, dict):
            hypothesis_data = None
        think_mode = str(payload.get("think_mode") or "adaptive")
        think_provider = str(payload.get("think_provider") or "")

        # 1 — web research (optional, best-effort).
        brief = ""
        if web_research:
            srv_status.set_running(
                "researching",
                intent=intent, archetype=archetype, count=count,
                provider=research_provider, think_mode=think_mode,
            )
            await send("research_started", {"provider": research_provider})
            try:
                result = await research(intent, archetype, provider=research_provider)
                brief = result.get("brief", "")
                await send("research_completed", {
                    "brief": brief,
                    "sources": result.get("sources", []),
                    "provider": research_provider,
                })
            except Exception as exc:  # noqa: BLE001 — research is non-fatal
                await send("research_skipped", {"reason": f"{type(exc).__name__}: {exc}"})

        # 2 — generation.
        srv_status.set_running(
            "generating",
            intent=intent, archetype=archetype, count=count,
            provider=gen_provider, think_mode=think_mode,
        )
        await send("generation_started", {"count": count, "provider": gen_provider})
        try:
            generated = await generate_alphas(
                intent, archetype, count, brief=brief, provider=gen_provider,
                hypothesis_data=hypothesis_data,
                think_mode=think_mode,
                think_provider=think_provider or gen_provider,
            )
        except OpenRouterServiceError as exc:
            await send("forge_failed", {"error": str(exc)})
            srv_status.set_idle()
            return

        alphas: list[dict[str, Any]] = []
        for idx, alpha in enumerate(generated):
            item = validate_alpha(alpha)
            item["id"] = item.get("id") or f"a{idx}"
            item["simulation"] = {"status": "queued"}
            item["simulation_state"] = "queued"
            alphas.append(item)
            srv_status.tick(idx + 1)
            await send("alpha_generated", {"index": idx, "alpha": item})

        # 3 — simulation, SIM_CONCURRENCY at a time.
        srv_status.set_running(
            "simulating",
            intent=intent, archetype=archetype, count=count,
            provider=gen_provider, think_mode=think_mode,
        )
        semaphore = asyncio.Semaphore(SIM_CONCURRENCY)
        sims_done = 0

        async def simulate_one(idx: int, item: dict[str, Any]) -> None:
            nonlocal sims_done
            async with semaphore:
                await send("simulation_started", {"index": idx, "alpha_id": item["id"]})
                simulation = await run_in_threadpool(
                    simulate_alpha,
                    item["expression"],
                    item.get("settings", {}),
                    item,
                )
                item["simulation"] = simulation
                item["simulation_state"] = "completed" if simulation.get("status") == "OK" else "failed"
                sims_done += 1
                srv_status.tick(sims_done)
                await send("simulation_completed", {
                    "index": idx,
                    "alpha_id": item["id"],
                    "simulation": simulation,
                    "alpha": item,
                })

        await asyncio.gather(*(simulate_one(idx, item) for idx, item in enumerate(alphas)))

        srv_status.set_idle()
        await send("forge_completed", {
            "elapsed_s": round(time.time() - started, 1),
            "alphas": alphas,
        })
    except WebSocketDisconnect:
        srv_status.set_idle()
        return
    except Exception as exc:  # noqa: BLE001
        srv_status.set_idle()
        try:
            await send("forge_failed", {"error": f"{type(exc).__name__}: {exc}"})
        except Exception:  # noqa: BLE001
            return


async def _send(websocket: WebSocket, event: str, payload: dict[str, Any]) -> None:
    await websocket.send_json({
        "event": event,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    })
