from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import alpha, config, connect, forge, generate, hypothesize, repair, simulate, status
from backend.api import jobs as jobs_module
from backend.api import intelligence as intelligence_module
from backend.api import analytics as analytics_module
from backend.websocket import router as websocket_router


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    jobs_module.startup_resume_jobs()
    # Pre-initialize gen2 services so population + DB load from disk at startup
    from backend.services.evolution_service import get_evolution_service
    from backend.services.storage_service import get_storage_service
    get_evolution_service()
    get_storage_service()
    yield


app = FastAPI(title="AlphaForge Brain API", version="1.0.0", lifespan=lifespan)

_raw_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# When FRONTEND_ORIGINS contains "*" (LAN mode), allow all origins.
# This is intentional for a local-network-only dev tool.
_allow_all = "*" in origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else origins,
    allow_credentials=False if _allow_all else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(connect.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(simulate.router, prefix="/api")
app.include_router(forge.router, prefix="/api")
app.include_router(repair.router, prefix="/api")
app.include_router(hypothesize.router, prefix="/api")
app.include_router(alpha.router, prefix="/api")
app.include_router(config.router, prefix="/api")
app.include_router(jobs_module.router, prefix="/api")
app.include_router(intelligence_module.router, prefix="/api")
app.include_router(status.router, prefix="/api")
app.include_router(analytics_module.router, prefix="/api")
app.include_router(websocket_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

