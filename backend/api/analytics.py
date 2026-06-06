"""
Analytics endpoints — exposes gen2 pattern analysis, alpha health, and bandit stats.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.services.evolution_service import get_evolution_service
from backend.services.storage_service import get_storage_service

router = APIRouter(tags=["analytics"])


@router.get("/analytics/patterns")
async def get_patterns() -> dict[str, Any]:
    """Pattern analysis across all stored simulations (by region, sharpe tier, operator)."""
    return get_storage_service().get_patterns()


@router.get("/analytics/storage")
async def get_storage_stats() -> dict[str, Any]:
    """Storage stats: total simulations, success rate, avg sharpe."""
    return get_storage_service().get_stats()


@router.get("/analytics/quality")
async def get_quality_report() -> list[dict[str, Any]]:
    """Per-alpha health report from AlphaQualityMonitor (sorted worst health first)."""
    return get_evolution_service().get_quality_report()


@router.get("/analytics/bandit")
async def get_bandit_stats() -> dict[str, Any]:
    """AdvancedBanditSystem stats: exploration rate, best strategy, persona generation."""
    evo = get_evolution_service()
    if not evo._enabled:
        return {"enabled": False}
    try:
        return evo._bandit.get_statistics()
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/analytics/evolution")
async def get_evolution_params() -> dict[str, Any]:
    """Current SelfOptimizer parameters."""
    evo = get_evolution_service()
    params = evo.get_params()
    return {
        "enabled": evo._enabled,
        "params": params,
        "sim_count": getattr(evo, "_sim_count", 0),
        "success_count": getattr(evo, "_success_count", 0),
        "population_size": len(evo._engine.population) if evo._enabled else 0,
    }
