"""
intelligence.py ��� Field intelligence and simulation insights API.

Endpoints:
  GET  /api/intelligence/fields    — ranked field list with alpha_count, conversion_rate
  GET  /api/intelligence/insights  — simulation history patterns (hot fields, settings, archetypes)
  GET  /api/intelligence/mechanics — Brain simulation mechanics documentation
  POST /api/intelligence/refresh   — refresh field catalog + personal alphas from Brain API
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["intelligence"])

_ROOT = Path(__file__).resolve().parents[2]
_MECHANICS_PATH = _ROOT / "data" / "brain_docs" / "simulation_mechanics.json"
_CREDS_PATH = _ROOT / "credentials.json"


# ─────────────────────────────────────────────────────────────── GET /fields

@router.get("/intelligence/fields")
async def get_fields(
    archetype: str = Query(default="any", description="Filter/weight by archetype"),
    category:  str = Query(default="",    description="Filter by category: Technical, Fundamental, Model"),
    n:         int = Query(default=50,    ge=1, le=500),
) -> dict[str, Any]:
    """Return top N fields ranked by weighted alpha_count for the given archetype."""
    from backend.services.field_intelligence import get_top_fields, get_field_stats
    try:
        fields = get_top_fields(
            n=n,
            category=category or None,
            archetype=archetype if archetype != "any" else None,
        )
        stats = get_field_stats()
        return {
            "fields": fields,
            "total_in_catalog": stats["total"],
            "fetched_at": stats["fetched_at"],
            "by_category": stats["by_category"],
            "query": {"archetype": archetype, "category": category or None, "n": n},
        }
    except Exception as exc:
        logger.exception("get_fields failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ──────────────────────────────────────────────── GET /intelligence/underexplored

@router.get("/intelligence/underexplored")
async def get_underexplored(
    n: int = Query(default=20, ge=1, le=100),
    min_coverage: float = Query(default=0.75, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """Return hidden-gem fields: high coverage, low user_count, decent conversion_rate."""
    from backend.services.field_intelligence import get_underexplored_fields
    try:
        fields = get_underexplored_fields(n=n, min_coverage=min_coverage)
        return {"fields": fields, "n": len(fields)}
    except Exception as exc:
        logger.exception("get_underexplored failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────── GET /insights

@router.get("/intelligence/insights")
async def get_insights() -> dict[str, Any]:
    """Return simulation history patterns: hot fields, archetype success, winning settings."""
    from backend.services.simulation_insights import get_simulation_insights
    try:
        return get_simulation_insights()
    except Exception as exc:
        logger.exception("get_insights failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─────────────────────────────────────────────────────────────── GET /mechanics

@router.get("/intelligence/mechanics")
async def get_mechanics() -> dict[str, Any]:
    """Return the Brain simulation mechanics documentation."""
    import json
    try:
        return json.loads(_MECHANICS_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail=f"simulation_mechanics.json not found: {exc}") from exc


# ──────────────────────────────────────────────────────────────── POST /refresh

def _run_refresh(region: str, universe: str, delay: int) -> None:
    """Background task: re-fetch field catalog + personal alphas from Brain API."""
    try:
        import json as _json
        from brain_api import BrainSession
        creds = _json.loads(_CREDS_PATH.read_text())
        session = BrainSession(creds["email"], creds["password"])

        logger.info("[intelligence] fetching data fields from Brain...")
        result = session.fetch_data_fields(region=region, universe=universe, delay=delay)
        if result.get("saved"):
            logger.info("[intelligence] fields updated: %d total", result["total"])
        else:
            logger.warning("[intelligence] field fetch failed: %s", result.get("error"))

        logger.info("[intelligence] fetching personal alphas from Brain...")
        alphas = session.list_my_alphas(limit=500)
        logger.info("[intelligence] fetched %d personal alphas", len(alphas))

        # Clear field intelligence cache so next request gets fresh data.
        from backend.services.field_intelligence import reload as _reload_fields
        _reload_fields()
        logger.info("[intelligence] field intelligence cache cleared")

    except Exception as exc:
        logger.exception("[intelligence] refresh failed: %s", exc)


@router.post("/intelligence/refresh")
async def refresh_brain_data(
    background_tasks: BackgroundTasks,
    region:   str = Query(default="USA",    description="Brain region"),
    universe: str = Query(default="TOP3000", description="Universe"),
    delay:    int = Query(default=1,         description="Delay (0 or 1)"),
) -> dict[str, str]:
    """Queue a background refresh of the Brain field catalog and personal alpha history.

    Requires credentials.json in the project root. The refresh runs asynchronously;
    poll GET /intelligence/fields to confirm the fetched_at timestamp has updated.
    """
    if not _CREDS_PATH.exists():
        raise HTTPException(
            status_code=422,
            detail="credentials.json not found. Add Brain credentials to enable refresh.",
        )
    background_tasks.add_task(_run_refresh, region, universe, delay)
    return {"status": "queued", "message": "Brain data refresh started in background."}
