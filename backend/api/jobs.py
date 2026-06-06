"""
jobs.py — Browser-independent background forge job queue.

A POST /api/jobs request schedules a full forge run as a background asyncio task.
The job persists to data/forge_jobs.json so it survives server restarts
(incomplete jobs are resumed on startup).

Endpoints:
  POST /api/jobs          — queue a new forge job, returns {job_id}
  GET  /api/jobs          — list all jobs (most recent first)
  GET  /api/jobs/{id}     — full details + results for one job
  DELETE /api/jobs/{id}   — remove a completed/failed job

Optional email notification: set NOTIFY_EMAIL and SMTP_HOST/SMTP_USER/SMTP_PASS in .env.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import smtplib
import uuid
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from backend.services.openrouter_service import (
    OpenRouterServiceError,
    generate_alphas,
    research,
)
from backend.services.brain_service import simulate_alpha
from backend.services.validator_service import normalize_settings, reconcile_neutralization, validate_alpha
from backend.api.forge import _simulate_with_fitness_repair
from backend.api import status as srv_status

logger = logging.getLogger(__name__)

router = APIRouter(tags=["jobs"])

_JOBS_PATH = Path(__file__).resolve().parents[2] / "data" / "forge_jobs.json"
_JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)

# In-memory store (loaded from disk on startup).
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()

# ──────────────────────────────────────────────────────────────── persistence

def _load_jobs() -> None:
    """Load jobs from disk into memory. Called once at startup."""
    global _jobs
    try:
        data = json.loads(_JOBS_PATH.read_text())
        if isinstance(data, dict):
            _jobs = data
    except (OSError, json.JSONDecodeError):
        _jobs = {}


def _save_jobs_sync() -> None:
    try:
        _JOBS_PATH.write_text(json.dumps(_jobs, indent=2, default=str))
    except OSError as exc:
        logger.warning("[jobs] failed to save jobs: %s", exc)


# ──────────────────────────────────────────────────────────────── email

def _send_email(subject: str, body: str) -> None:
    recipient = os.getenv("NOTIFY_EMAIL", "").strip()
    if not recipient:
        return
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASS", "").strip()
    if not smtp_user or not smtp_pass:
        logger.info("[jobs] email skipped — SMTP_USER/SMTP_PASS not configured (to: %s)", recipient)
        return
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = recipient
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as srv:
            srv.starttls()
            srv.login(smtp_user, smtp_pass)
            srv.sendmail(smtp_user, [recipient], msg.as_string())
        logger.info("[jobs] email sent to %s", recipient)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[jobs] email failed: %s", exc)


# ──────────────────────────────────────────────────────────────── job runner

async def _run_job(job_id: str) -> None:
    """Execute a queued forge job fully in the background."""
    async with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        _save_jobs_sync()

    request = job.get("request", {})
    intent = str(request.get("intent") or "")
    archetype = str(request.get("archetype") or "any")
    count = max(1, min(int(request.get("count") or 3), 100))
    web_research = bool(request.get("web_research", True))
    gen_provider = str(request.get("gen_provider") or "openrouter")
    research_provider = str(request.get("research_provider") or "openrouter")
    hypothesis_data = request.get("hypothesis_data") or None
    think_mode = str(request.get("think_mode") or "adaptive")

    log_lines: list[str] = []

    def log(msg: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        log_lines.append(f"[{stamp}] {msg}")
        logger.info("[job %s] %s", job_id, msg)

    try:
        # 1 — research
        brief = ""
        if web_research:
            srv_status.set_running(
                "researching", intent=intent, archetype=archetype,
                count=count, provider=research_provider,
                think_mode=think_mode, job_id=job_id,
                label=job.get("label", ""),
            )
            log("Web research started.")
            try:
                result = await research(intent, archetype, provider=research_provider)
                brief = result.get("brief", "")
                sources = result.get("sources", [])
                log(f"Research complete ({len(sources)} sources).")
            except Exception as exc:  # noqa: BLE001
                log(f"Research skipped: {exc}")

        # 2 — generation (with optional deep think)
        srv_status.set_running(
            "generating", intent=intent, archetype=archetype,
            count=count, provider=gen_provider,
            think_mode=think_mode, job_id=job_id,
            label=job.get("label", ""),
        )
        log(f"Generation started ({count} alphas, mode={think_mode}).")
        generated = await generate_alphas(
            intent, archetype, count,
            brief=brief,
            provider=gen_provider,
            hypothesis_data=hypothesis_data,
            think_mode=think_mode,
        )
        log(f"Generated {len(generated)} alpha candidates.")

        # 3 — validate + simulate
        forged: list[dict[str, Any]] = [validate_alpha(alpha) for alpha in generated]
        sim_sem = asyncio.Semaphore(2)
        sims_done = 0

        srv_status.set_running(
            "simulating", intent=intent, archetype=archetype,
            count=len(forged), provider=gen_provider,
            think_mode=think_mode, job_id=job_id,
            label=job.get("label", ""),
        )

        async def simulate_one(item: dict[str, Any]) -> None:
            nonlocal sims_done
            async with sim_sem:
                await _simulate_with_fitness_repair(item, gen_provider)
                sims_done += 1
                srv_status.tick(sims_done)
                sim_status = item.get("simulation", {}).get("status", "?")
                log(f"Simulated '{item.get('name', '?')}': {sim_status}")

        await asyncio.gather(*(simulate_one(item) for item in forged))

        passed = sum(1 for a in forged if a.get("simulation", {}).get("status") == "OK")
        log(f"Forge complete — {passed}/{len(forged)} alphas simulated OK.")
        srv_status.set_idle()

        async with _jobs_lock:
            _jobs[job_id].update({
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result": {
                    "alphas": forged,
                    "brief": brief,
                    "count": len(forged),
                    "passed": passed,
                },
                "logs": log_lines,
            })
            _save_jobs_sync()

        _send_email(
            subject=f"AlphaForge job {job_id} complete — {passed}/{len(forged)} alphas passed",
            body="\n".join(log_lines) + f"\n\nIntent: {intent or '(none)'}\nArchetype: {archetype}",
        )

    except Exception as exc:  # noqa: BLE001
        srv_status.set_idle()
        log(f"Job failed: {type(exc).__name__}: {exc}")
        async with _jobs_lock:
            _jobs[job_id].update({
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
                "logs": log_lines,
            })
            _save_jobs_sync()

        _send_email(
            subject=f"AlphaForge job {job_id} FAILED",
            body="\n".join(log_lines) + f"\n\nError: {exc}",
        )


# ──────────────────────────────────────────────────────────────── startup

def startup_resume_jobs() -> None:
    """Called once at server startup: reload disk state and re-queue running/queued jobs."""
    _load_jobs()
    incomplete = [jid for jid, j in _jobs.items() if j.get("status") in ("queued", "running")]
    if incomplete:
        logger.info("[jobs] re-queuing %d incomplete jobs from disk", len(incomplete))
        # asyncio loop may not be running yet — jobs will be picked up by FastAPI lifespan.
        for jid in incomplete:
            _jobs[jid]["status"] = "queued"
            _jobs[jid].pop("started_at", None)


# ──────────────────────────────────────────────────────────────── request model

class JobRequest(BaseModel):
    intent: str = ""
    archetype: str = "any"
    count: int = Field(default=3, ge=1, le=100)
    web_research: bool = True
    gen_provider: str = "openrouter"
    research_provider: str = "openrouter"
    hypothesis_data: dict | None = None
    think_mode: str = "adaptive"
    label: str = ""


# ──────────────────────────────────────────────────────────────── endpoints

@router.post("/jobs")
async def create_job(payload: JobRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:10]
    job: dict[str, Any] = {
        "id": job_id,
        "label": payload.label or payload.intent[:60] or f"job-{job_id}",
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "finished_at": None,
        "request": payload.model_dump(),
        "result": None,
        "error": None,
        "logs": [],
    }
    async with _jobs_lock:
        _jobs[job_id] = job
        _save_jobs_sync()

    background_tasks.add_task(_run_job, job_id)
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs")
async def list_jobs() -> dict[str, Any]:
    async with _jobs_lock:
        jobs = list(_jobs.values())
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    # Strip heavy result payload from list view.
    slim = []
    for j in jobs:
        s = {k: v for k, v in j.items() if k not in ("result",)}
        if j.get("result"):
            s["result_summary"] = {
                "count": j["result"].get("count", 0),
                "passed": j["result"].get("passed", 0),
            }
        slim.append(s)
    return {"jobs": slim}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    async with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str) -> dict[str, str]:
    async with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="job not found")
        if _jobs[job_id].get("status") == "running":
            raise HTTPException(status_code=409, detail="job is currently running")
        del _jobs[job_id]
        _save_jobs_sync()
    return {"status": "deleted"}
