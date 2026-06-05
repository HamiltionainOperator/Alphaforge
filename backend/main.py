from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import alpha, config, connect, forge, generate, hypothesize, repair, simulate
from backend.websocket import router as websocket_router


load_dotenv()

app = FastAPI(title="AlphaForge Brain API", version="1.0.0")

origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
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
app.include_router(websocket_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

