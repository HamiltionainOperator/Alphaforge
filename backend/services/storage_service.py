"""
Gen2 BacktestStorage + AlphaRegrouper integrated into AlphaForge.

Replaces the flat memory.json / feedback.json approach with a proper
SQLite database that tracks every simulation result, enables pattern
analysis, and feeds operator usage stats back to the bandit system.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GEN2_ROOT = Path(__file__).resolve().parents[3] / "worldquant-miner"
if _GEN2_ROOT.exists():
    sys.path.insert(0, str(_GEN2_ROOT))

try:
    from generation_two.storage.backtest_storage import BacktestStorage, BacktestRecord
    from generation_two.storage.regroup import AlphaRegrouper
    _STORAGE_AVAILABLE = True
    logger.info("Gen2 storage engine loaded.")
except ImportError as e:
    _STORAGE_AVAILABLE = False
    logger.warning(f"Gen2 storage not available — using memory.json fallback. ({e})")

_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "alphaquant_backtests.db"


class StorageService:
    """
    Persists every simulation result to SQLite and exposes pattern analysis.
    Falls back gracefully when gen2 is not available.
    """

    def __init__(self) -> None:
        self._enabled = _STORAGE_AVAILABLE
        if not self._enabled:
            return
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._db = BacktestStorage(str(_DB_PATH))
        self._regrouper = AlphaRegrouper()
        self._results_cache: list = []  # in-memory for fast regrouping
        logger.info(f"BacktestStorage initialised at {_DB_PATH}")

    # ── write ──────────────────────────────────────────────────────────────────

    def store(self, item: dict[str, Any], simulation: dict[str, Any]) -> None:
        """Persist a forge result (item + simulation) to SQLite."""
        if not self._enabled:
            return
        try:
            record = BacktestRecord(
                template=item.get("expression", ""),
                region=item.get("settings", {}).get("region", "USA"),
                sharpe=simulation.get("sharpe") or 0.0,
                fitness=simulation.get("fitness") or 0.0,
                turnover=simulation.get("turnover_pct") or 0.0,
                returns=simulation.get("returns") or 0.0,
                drawdown=simulation.get("drawdown") or 0.0,
                margin=simulation.get("margin") or 0.0,
                longCount=simulation.get("long_count") or 0,
                shortCount=simulation.get("short_count") or 0,
                success=simulation.get("status") == "OK",
                alpha_id=simulation.get("alpha_id") or "",
                raw_data=json.dumps(simulation),
            )
            self._db.store_result(record)
            self._results_cache.append(record)
            if len(self._results_cache) > 500:
                self._results_cache = self._results_cache[-500:]
        except Exception as exc:
            logger.warning(f"StorageService.store error: {exc}")

    # ── read / analytics ───────────────────────────────────────────────────────

    def get_patterns(self) -> dict[str, Any]:
        """Return pattern analysis across all stored results."""
        if not self._enabled or not self._results_cache:
            return {}
        try:
            by_region = self._regrouper.get_regroup_summary(
                self._regrouper.regroup_by_region(self._results_cache)
            )
            by_tier = self._regrouper.get_regroup_summary(
                self._regrouper.regroup_by_sharpe_tier(self._results_cache)
            )
            by_operator = self._regrouper.get_regroup_summary(
                self._regrouper.regroup_by_operator(self._results_cache)
            )
            return {
                "total": len(self._results_cache),
                "by_region": by_region,
                "by_sharpe_tier": by_tier,
                "by_operator": by_operator,
            }
        except Exception as exc:
            logger.warning(f"get_patterns error: {exc}")
            return {}

    def get_operator_success_rates(self) -> dict[str, float]:
        """
        Returns {operator: success_rate} for use by the bandit system.
        Built from in-memory cache for speed.
        """
        if not self._enabled or not self._results_cache:
            return {}
        try:
            grouped = self._regrouper.regroup_by_operator(self._results_cache)
            rates: dict[str, float] = {}
            for op, records in grouped.items():
                if not records:
                    continue
                successes = sum(1 for r in records if getattr(r, "success", False))
                rates[op] = successes / len(records)
            return rates
        except Exception as exc:
            logger.warning(f"get_operator_success_rates error: {exc}")
            return {}

    def get_stats(self) -> dict[str, Any]:
        """Quick summary stats."""
        if not self._enabled:
            return {"enabled": False}
        total = len(self._results_cache)
        successes = sum(1 for r in self._results_cache if getattr(r, "success", False))
        sharpes = [getattr(r, "sharpe", 0) for r in self._results_cache if getattr(r, "success", False)]
        return {
            "enabled": True,
            "total_simulations": total,
            "successful": successes,
            "success_rate": round(successes / total, 3) if total else 0,
            "avg_sharpe": round(sum(sharpes) / len(sharpes), 3) if sharpes else 0,
            "db_path": str(_DB_PATH),
        }

    def has_been_simulated(self, expression: str, region: str = "USA") -> bool:
        if not self._enabled:
            return False
        try:
            return self._db.has_been_simulated(expression, region)
        except Exception:
            return False


_service: StorageService | None = None


def get_storage_service() -> StorageService:
    global _service
    if _service is None:
        _service = StorageService()
    return _service
