"""
Duplicate Detection System (ported from generation_two/ollama/duplicate_detector.py).

Prevents generating duplicate or near-duplicate alpha expressions by:
  - Normalizing expressions and hashing both the literal form and the structure
  - Tracking every generated expression in an ``expression_history`` SQLite table
  - Feeding the literal recent expressions back into the LLM prompt (avoidance context)
  - Rejecting anything above a character-similarity threshold

The history table lives in the SAME database AlphaForge already uses for backtest
results (``data/alphaquant_backtests.db``), so it sits alongside the gen2 storage
engine wired up in ``storage_service.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set

logger = logging.getLogger(__name__)

# Make generation_two importable regardless of import order (mirrors storage_service.py),
# so we can prefer the real gen2 DuplicateDetector over the local port below.
_GEN2_ROOT = Path(__file__).resolve().parents[3] / "worldquant-miner"
if _GEN2_ROOT.exists() and str(_GEN2_ROOT) not in sys.path:
    sys.path.insert(0, str(_GEN2_ROOT))

# Same database file as StorageService (backend/services -> parents[2] == repo root).
_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "alphaquant_backtests.db"


@dataclass
class ExpressionSignature:
    """Signature of an alpha expression for duplicate detection."""

    template: str
    normalized: str  # Normalized version for comparison
    hash: str  # Hash for quick lookup
    operators: Set[str]  # Set of operators used
    structure_hash: str  # Hash of expression structure


class DuplicateDetector:
    """
    Detects and prevents duplicate alpha expressions.

    Features:
    - Normalizes expressions for comparison
    - Tracks all generated expressions
    - Provides context to the LLM to avoid duplicates
    - Supports a configurable similarity threshold
    """

    def __init__(self, db_path: str | None = None, similarity_threshold: float = 0.85):
        self.db_path = str(db_path or _DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._memory_cache: Dict[str, ExpressionSignature] = {}
        # Expressions with >= this similarity are considered duplicates.
        self.similarity_threshold = similarity_threshold

    def _init_database(self) -> None:
        """Initialize database for expression tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS expression_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                template TEXT NOT NULL,
                normalized TEXT NOT NULL,
                template_hash TEXT NOT NULL UNIQUE,
                structure_hash TEXT NOT NULL,
                operators TEXT,
                region TEXT,
                timestamp REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_template_hash ON expression_history(template_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_structure_hash ON expression_history(structure_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_normalized ON expression_history(normalized)")

        conn.commit()
        conn.close()

    def normalize_expression(self, expression: str) -> str:
        """Normalize expression for comparison (strip whitespace, lowercase, drop comments)."""
        normalized = re.sub(r"\s+", "", expression)
        normalized = normalized.lower()
        normalized = re.sub(r"//.*", "", normalized)
        return normalized

    def extract_structure(self, expression: str) -> str:
        """
        Extract structure of expression (operators and nesting, not values).

        Example: ts_rank(close, 20) -> ts_rank(_, _)
        """
        structure = re.sub(r"\d+\.?\d*", "_", expression)
        common_fields = ["close", "open", "high", "low", "volume", "vwap", "returns", "volatility"]
        for field in common_fields:
            structure = re.sub(rf"\b{field}\b", "_", structure, flags=re.IGNORECASE)
        return structure

    def create_signature(self, expression: str) -> ExpressionSignature:
        """Create a signature for an expression."""
        normalized = self.normalize_expression(expression)
        structure = self.extract_structure(expression)
        operators = set(re.findall(r"([a-z_]+)\s*\(", expression.lower()))
        template_hash = hashlib.md5(normalized.encode()).hexdigest()
        structure_hash = hashlib.md5(structure.encode()).hexdigest()
        return ExpressionSignature(
            template=expression,
            normalized=normalized,
            hash=template_hash,
            operators=operators,
            structure_hash=structure_hash,
        )

    def is_duplicate(self, expression: str) -> bool:
        """Return True if the expression is an exact or near-duplicate of a known one."""
        if not expression or not expression.strip():
            return False

        signature = self.create_signature(expression)

        # Check memory cache first.
        if signature.hash in self._memory_cache:
            return True

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Exact match.
        cursor.execute("SELECT id FROM expression_history WHERE template_hash = ?", (signature.hash,))
        if cursor.fetchone():
            conn.close()
            self._memory_cache[signature.hash] = signature
            return True

        # Structure-similarity match.
        cursor.execute(
            "SELECT template, normalized FROM expression_history WHERE structure_hash = ?",
            (signature.structure_hash,),
        )
        similar = cursor.fetchall()
        conn.close()

        for _existing_template, existing_normalized in similar:
            similarity = self._calculate_similarity(signature.normalized, existing_normalized)
            if similarity >= self.similarity_threshold:
                logger.debug("Found similar expression: %.2f%% similarity", similarity * 100)
                return True

        return False

    def register_expression(self, expression: str, region: str = "") -> None:
        """Register a new expression (mark as used)."""
        if not expression or not expression.strip():
            return

        signature = self.create_signature(expression)
        self._memory_cache[signature.hash] = signature

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO expression_history
                (template, normalized, template_hash, structure_hash, operators, region, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    expression,
                    signature.normalized,
                    signature.hash,
                    signature.structure_hash,
                    json.dumps(list(signature.operators)),
                    region,
                    time.time(),
                ),
            )
            conn.commit()
            conn.close()
            logger.debug("Registered expression: %s...", expression[:50])
        except sqlite3.IntegrityError:
            pass  # Already exists.
        except Exception as exc:  # noqa: BLE001
            logger.error("Error registering expression: %s", exc)

    def get_avoidance_context(self, limit: int = 10) -> str:
        """Return a prompt block listing recent expressions for the LLM to avoid."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT template FROM expression_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            recent = cursor.fetchall()
            conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not build avoidance context: %s", exc)
            return ""

        expressions = [row[0] for row in recent if row and row[0]]
        if not expressions:
            return ""

        context = "Avoid generating expressions similar to these recent ones:\n"
        for i, expr in enumerate(expressions, 1):
            context += f"{i}. {expr}\n"
        return context

    def _calculate_similarity(self, expr1: str, expr2: str) -> float:
        """Character-based similarity between two normalized expressions."""
        if expr1 == expr2:
            return 1.0

        len1, len2 = len(expr1), len(expr2)
        max_len = max(len1, len2)
        if max_len == 0:
            return 1.0

        min_len = min(len1, len2)
        common = sum(1 for i in range(min_len) if expr1[i] == expr2[i])

        common_substrings = 0
        for i in range(min_len - 2):
            if expr1[i : i + 3] in expr2:
                common_substrings += 1

        return (common / max_len) * 0.7 + (common_substrings / max(min_len - 2, 1)) * 0.3

    def get_statistics(self) -> Dict:
        """Return duplicate-detection statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM expression_history")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT structure_hash) FROM expression_history")
        unique_structures = cursor.fetchone()[0]
        conn.close()
        return {
            "total_expressions": total,
            "unique_structures": unique_structures,
            "duplicate_rate": 1.0 - (unique_structures / total) if total > 0 else 0.0,
            "cached_signatures": len(self._memory_cache),
        }


# Prefer the REAL gen2 DuplicateDetector (single source of truth, same as the rest
# of AlphaForge's gen2 integration). Fall back to the local port above only when
# generation_two is not importable.
try:
    from generation_two.ollama.duplicate_detector import DuplicateDetector as _Gen2DuplicateDetector

    _GEN2_DETECTOR_AVAILABLE = True
    logger.info("Using gen2 DuplicateDetector (generation_two.ollama.duplicate_detector).")
except Exception as _exc:  # noqa: BLE001
    _GEN2_DETECTOR_AVAILABLE = False
    logger.warning("gen2 DuplicateDetector unavailable — using local port. (%s)", _exc)


_detector = None


def get_duplicate_detector():
    """Return the process-wide DuplicateDetector singleton.

    Uses generation_two's DuplicateDetector when available (so AlphaForge and gen2
    share identical dedup logic); otherwise the local port in this module. Either
    way it is pointed at AlphaForge's own backtest DB.
    """
    global _detector
    if _detector is None:
        if _GEN2_DETECTOR_AVAILABLE:
            _detector = _Gen2DuplicateDetector(db_path=str(_DB_PATH))
            logger.info("gen2 DuplicateDetector initialised at %s", _DB_PATH)
        else:
            _detector = DuplicateDetector()
            logger.info("Local DuplicateDetector initialised at %s", _DB_PATH)
    return _detector
