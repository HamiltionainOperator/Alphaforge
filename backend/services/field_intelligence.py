"""
field_intelligence.py — Field performance intelligence from the Brain field catalog.

Parses data/brain_docs/fields_full.json to rank fields by proven alpha-generation
potential (alpha_count, conversion_rate) and exposes context strings for LLM injection.

Key exports:
    get_top_fields(n, category, archetype)       → sorted field dicts
    get_field_context_for_prompt(archetype, n)   → formatted LLM injection string
    get_underexplored_fields(n, min_coverage)    → hidden-gem fields
    get_field_stats()                             → catalog metadata
"""
from __future__ import annotations

import json
import logging
import re as _re
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_FIELDS_PATH = _ROOT / "data" / "brain_docs" / "fields_full.json"
_KB_PATH = _ROOT / "brain_kb.json"
_MEMORY_PATH = _ROOT / "data" / "memory.json"

# Core price/volume tokens that are always saturated — guaranteed to cause
# high SELF_CORRELATION when used as the sole signal.
_ALWAYS_SATURATED = {"close", "open", "high", "low", "vwap", "volume", "returns"}

_OP_RE = _re.compile(
    r"^(ts_mean|ts_sum|ts_std_dev|ts_delta|ts_delay|ts_corr|ts_covariance|"
    r"ts_zscore|ts_rank|ts_max|ts_min|ts_arg_max|ts_arg_min|ts_decay_linear|"
    r"ts_product|ts_returns|ts_skewness|ts_kurtosis|ts_count_nans|"
    r"rank|zscore|scale|normalize|quantile|signed_power|group_neutralize|"
    r"group_rank|group_mean|group_zscore|group_sum|group_max|group_min|"
    r"if_else|abs|log|sign|exp|sqrt|pow|ceil|floor|round|less|greater|"
    r"equal|and|or|not|min|max|add|subtract|multiply|divide|"
    r"sector|industry|subindustry|market|int|float|true|false)$"
)


def _extract_expr_fields(expression: str) -> set[str]:
    tokens = _re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", expression or "")
    return {t for t in tokens if not _OP_RE.match(t)}

# Archetype → category preference weights.
# Category names match the actual Brain field catalog (fields_full.json).
# "Price Volume" = price/volume/microstructure fields
# "Fundamental"  = balance sheet, income statement
# "Model"        = model-derived composite signals
# "Analyst"      = analyst estimates and revisions
# "Option"       = options implied volatility / skew
# "News"         = news sentiment
_AFFINITY: dict[str, dict[str, float]] = {
    "reversal":         {"Price Volume": 3.0, "Fundamental": 0.4, "Model": 0.8, "Analyst": 0.5, "Option": 0.5, "News": 0.3, "Sentiment": 0.3, "Social Media": 0.2},
    "microstructure":   {"Price Volume": 3.0, "Fundamental": 0.2, "Model": 0.6, "Analyst": 0.3, "Option": 0.8, "News": 0.3, "Sentiment": 0.3, "Social Media": 0.2},
    "volatility":       {"Price Volume": 2.5, "Fundamental": 0.4, "Model": 1.5, "Analyst": 0.5, "Option": 2.0, "News": 0.4, "Sentiment": 0.4, "Social Media": 0.2},
    "fundamental":      {"Price Volume": 0.3, "Fundamental": 3.0, "Model": 0.8, "Analyst": 1.5, "Option": 0.3, "News": 0.3, "Sentiment": 0.2, "Social Media": 0.1},
    "analyst_revision": {"Price Volume": 0.4, "Fundamental": 2.0, "Model": 2.5, "Analyst": 3.0, "Option": 0.4, "News": 1.0, "Sentiment": 0.5, "Social Media": 0.2},
    "earnings_event":   {"Price Volume": 0.8, "Fundamental": 2.5, "Model": 2.0, "Analyst": 2.5, "Option": 0.8, "News": 1.0, "Sentiment": 0.5, "Social Media": 0.2},
    "options_implied":  {"Price Volume": 1.5, "Fundamental": 0.3, "Model": 0.5, "Analyst": 0.3, "Option": 3.0, "News": 0.3, "Sentiment": 0.3, "Social Media": 0.1},
    "factor_residual":  {"Price Volume": 0.8, "Fundamental": 2.0, "Model": 2.5, "Analyst": 1.5, "Option": 0.4, "News": 0.4, "Sentiment": 0.3, "Social Media": 0.1},
    "dispersion":       {"Price Volume": 0.8, "Fundamental": 1.5, "Model": 2.5, "Analyst": 3.0, "Option": 0.8, "News": 0.5, "Sentiment": 0.3, "Social Media": 0.2},
    "novel":            {"Price Volume": 1.0, "Fundamental": 1.0, "Model": 1.0, "Analyst": 1.0, "Option": 1.0, "News": 1.0, "Sentiment": 1.0, "Social Media": 1.0},
    "any":              {"Price Volume": 1.0, "Fundamental": 1.0, "Model": 1.0, "Analyst": 1.0, "Option": 1.0, "News": 1.0, "Sentiment": 1.0, "Social Media": 1.0},
}


@lru_cache(maxsize=1)
def _load_forbidden() -> tuple[set[str], tuple[str, ...]]:
    """Load forbidden fields from brain_kb.json. Returns (exact_set, prefix_tuple)."""
    try:
        kb = json.loads(_KB_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return set(), ()
    forbidden_raw = kb.get("forbidden_fields", [])
    exact = {f for f in forbidden_raw if not f.endswith("*")}
    prefixes = tuple(f[:-1] for f in forbidden_raw if f.endswith("*"))
    return exact, prefixes


# Group tokens used in group_neutralize(x, sector) — not signal fields.
_GROUP_TOKENS = {"sector", "industry", "subindustry", "market", "country", "currency"}


def _is_forbidden(field_id: str) -> bool:
    if field_id in _GROUP_TOKENS:
        return True
    exact, prefixes = _load_forbidden()
    if field_id in exact:
        return True
    return any(field_id.startswith(p) for p in prefixes)


@lru_cache(maxsize=1)
def _load_catalog() -> tuple[dict[str, dict], str]:
    """Load and score fields_full.json. Returns (fields_dict, fetched_at)."""
    try:
        raw = json.loads(_FIELDS_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[field_intel] cannot load fields_full.json: %s", exc)
        return {}, ""

    fields_raw: dict[str, Any] = raw.get("fields", {})
    fetched_at: str = raw.get("fetched_at", "")

    # Normalize max values for percentage-based scoring.
    all_alpha_counts = [int(f.get("alpha_count") or 0) for f in fields_raw.values()]
    max_alpha = max(all_alpha_counts) if all_alpha_counts else 1

    scored: dict[str, dict] = {}
    for fid, meta in fields_raw.items():
        if _is_forbidden(fid):
            continue  # never suggest forbidden fields to the LLM
        alpha_count = int(meta.get("alpha_count") or 0)
        user_count = int(meta.get("user_count") or 0)
        coverage = float(meta.get("coverage") or 0.0)
        category = str(meta.get("category") or "").strip() or "Price Volume"
        # alpha_count can exceed user_count (one user may submit many alphas using this field).
        # conversion_rate = alphas submitted per user who explored this field.
        conversion_rate = alpha_count / max(user_count, 1)

        scored[fid] = {
            "id": fid,
            "description": str(meta.get("description") or "").strip(),
            "dataset_id": str(meta.get("dataset_id") or ""),
            "dataset_name": str(meta.get("dataset_name") or ""),
            "category": category,
            "subcategory": str(meta.get("subcategory") or ""),
            "coverage": coverage,
            "user_count": user_count,
            "alpha_count": alpha_count,
            "alpha_score": alpha_count,
            "alpha_score_pct": round(alpha_count / max_alpha, 4),
            "conversion_rate": round(conversion_rate, 4),
            "themes": meta.get("themes") or [],
        }

    return scored, fetched_at


def _fields() -> dict[str, dict]:
    catalog, _ = _load_catalog()
    return catalog


def _affinity_for(archetype: str | None) -> dict[str, float]:
    arch = (archetype or "any").strip().lower()
    return _AFFINITY.get(arch, _AFFINITY["any"])


def get_top_fields(
    n: int = 20,
    category: str | None = None,
    archetype: str | None = None,
) -> list[dict]:
    """Return top n fields sorted by weighted alpha_score.

    Args:
        n: Number of fields to return.
        category: If set, filter to only this category (Technical/Fundamental/Model).
        archetype: If set, weight scores by archetype affinity before sorting.
    """
    catalog = _fields()
    if not catalog:
        return []

    affinity = _affinity_for(archetype)
    result = []
    for field in catalog.values():
        if category and field["category"].lower() != category.lower():
            continue
        cat_weight = affinity.get(field["category"], 1.0)
        field = dict(field)  # copy so we don't mutate the cache
        field["weighted_score"] = field["alpha_score"] * cat_weight
        result.append(field)

    result.sort(key=lambda f: f["weighted_score"], reverse=True)
    return result[:n]


def get_field_context_for_prompt(archetype: str | None = None, n: int = 25) -> str:
    """Return a formatted string listing top fields for injection into LLM prompts.

    For archetype-specific requests, shows the top category-specific fields first
    (characteristic fields for the archetype) followed by universally useful fields.
    This avoids flood-ranking of generic fields (close, returns, volume) drowning
    out signal-specific fields (implied_volatility_*, ebit, analyst estimates).
    """
    arch = (archetype or "any").strip().lower()
    affinity = _affinity_for(arch)
    catalog = _fields()
    if not catalog:
        return ""

    # Sort categories by their affinity weight for this archetype.
    sorted_cats = sorted(affinity.items(), key=lambda kv: kv[1], reverse=True)

    # Allocate slots proportionally among top categories.
    # Highest-affinity category gets ~40% of slots, others share the rest.
    seen_ids: set[str] = set()
    selected: list[dict] = []
    total_weight = sum(w for _, w in sorted_cats[:4])

    for cat, weight in sorted_cats[:4]:
        if weight < 0.5:
            continue
        cat_n = max(3, round(n * weight / max(total_weight, 1)))
        cat_fields = [
            f for f in catalog.values()
            if f["category"] == cat and f["id"] not in seen_ids
        ]
        cat_fields.sort(key=lambda f: f["alpha_score"], reverse=True)
        for f in cat_fields[:cat_n]:
            if f["id"] not in seen_ids:
                seen_ids.add(f["id"])
                selected.append(f)
        if len(selected) >= n:
            break

    # Backfill with global top fields if we didn't fill all n slots.
    if len(selected) < n:
        global_top = sorted(catalog.values(), key=lambda f: f["alpha_score"], reverse=True)
        for f in global_top:
            if f["id"] not in seen_ids and len(selected) < n:
                seen_ids.add(f["id"])
                selected.append(f)

    arch_label = arch if arch != "any" else "general"
    lines = [f"High-signal fields for {arch_label} archetype (ranked by Brain alpha_count):"]
    for f in selected:
        desc = (f["description"] or "")[:60] or "(no description)"
        lines.append(
            f"  - {f['id']:<45} [alpha_count={f['alpha_count']}, cat={f['category']}] — {desc}"
        )
    return "\n".join(lines)


def get_underexplored_fields(n: int = 10, min_coverage: float = 0.75) -> list[dict]:
    """Return fields that are high-coverage but have relatively few users trying them,
    yet decent alpha_count — indicating untapped potential ('hidden gems').

    A field qualifies if:
      - coverage >= min_coverage (data quality gate)
      - user_count < 30 (not yet saturated)
      - alpha_count >= 3 (at least a few successes — not just flukes)
      - conversion_rate >= 0.20 (at least 20% of triers succeeded)
    """
    catalog = _fields()
    candidates = [
        f for f in catalog.values()
        if f["coverage"] >= min_coverage
        and f["user_count"] < 30
        and f["alpha_count"] >= 3
        and f["conversion_rate"] >= 0.20
    ]
    candidates.sort(key=lambda f: f["conversion_rate"], reverse=True)
    return candidates[:n]


def get_field_stats() -> dict:
    """Return catalog metadata: total field count, last-fetched timestamp, category breakdown."""
    catalog, fetched_at = _load_catalog()
    if not catalog:
        return {"total": 0, "fetched_at": None, "by_category": {}}

    by_category: dict[str, int] = {}
    for f in catalog.values():
        cat = f["category"]
        by_category[cat] = by_category.get(cat, 0) + 1

    return {
        "total": len(catalog),
        "fetched_at": fetched_at,
        "by_category": by_category,
    }


def get_field(field_id: str) -> dict | None:
    """Return metadata for a single field by ID, or None if not found."""
    return _fields().get(field_id)


def get_saturated_fields() -> set[str]:
    """Return fields already used in confirmed_working alphas + user's passing alphas.

    These are the fields that cause high SELF_CORRELATION — any new alpha whose
    primary mechanism relies only on these will likely fail Brain's correlation check.
    """
    saturated = set(_ALWAYS_SATURATED)

    # Add fields from confirmed_working exemplars in brain_kb.json.
    try:
        kb = json.loads(_KB_PATH.read_text())
        for entry in kb.get("confirmed_working", []):
            saturated |= _extract_expr_fields(entry.get("expression", ""))
            saturated |= set(entry.get("fields", []))
    except (OSError, json.JSONDecodeError):
        pass

    # Add fields from the user's passing simulation history.
    try:
        mem = json.loads(_MEMORY_PATH.read_text())
        entries = mem.get("entries", mem) if isinstance(mem, dict) else mem
        if isinstance(entries, list):
            for entry in entries:
                expr = (entry.get("alpha") or entry).get("expression", "")
                saturated |= _extract_expr_fields(expr)
    except (OSError, json.JSONDecodeError):
        pass

    # Never include group tokens or forbidden fields in the "saturated" output.
    saturated -= _GROUP_TOKENS
    return saturated


def get_diversity_context_for_prompt(archetype: str | None = None, n: int = 15) -> str:
    """Return a two-part string for LLM injection:
      1. SATURATED fields — already in the user's pool, avoid as primary signal
      2. UNDEREXPLORED fields — proven but low-saturation, prefer for low self-correlation

    This is the key tool for reducing Brain's SELF_CORRELATION check failures.
    """
    saturated = get_saturated_fields()
    catalog = _fields()
    if not catalog:
        return ""

    # --- Part 1: saturated fields list ---
    saturated_known = sorted(f for f in saturated if f in catalog or f in _ALWAYS_SATURATED)
    saturated_line = ", ".join(saturated_known[:20]) if saturated_known else "close, open, volume, returns, vwap"

    # --- Part 2: underexplored candidates ---
    # Primary: fields with low user_count, decent alpha_count, high coverage.
    underexplored = [
        f for f in get_underexplored_fields(n=50, min_coverage=0.70)
        if f["id"] not in saturated
    ]

    # Supplement with mid-tier fields (alpha_count 200–15000) not yet saturated,
    # weighted by archetype affinity.
    affinity = _affinity_for(archetype)
    mid_tier = [
        f for f in catalog.values()
        if 200 <= f["alpha_count"] <= 15000
        and f["id"] not in saturated
        and f["id"] not in _GROUP_TOKENS
    ]
    mid_tier.sort(
        key=lambda f: f["alpha_score"] * affinity.get(f["category"], 0.5),
        reverse=True,
    )

    # Merge: underexplored first, then mid-tier to fill up to n.
    seen: set[str] = set()
    candidates: list[dict] = []
    for f in underexplored + mid_tier:
        if f["id"] not in seen and len(candidates) < n:
            seen.add(f["id"])
            candidates.append(f)

    if not candidates:
        return ""

    arch_label = (archetype or "any").strip()
    lines = [
        f"=== SELF-CORRELATION AVOIDANCE ({arch_label}) ===",
        f"SATURATED fields (already in your pool — AVOID as the sole/primary signal): {saturated_line}",
        f"Brain's SELF_CORRELATION check WILL FAIL if your alpha uses only these.",
        "",
        "UNDEREXPLORED fields (low saturation, proven — PREFER at least one of these):",
    ]
    for f in candidates:
        desc = (f["description"] or "")[:55] or "(no description)"
        lines.append(
            f"  - {f['id']:<45} [alpha_count={f['alpha_count']}, cat={f['category']}] — {desc}"
        )
    return "\n".join(lines)


def reload() -> None:
    """Clear the LRU cache so the next call re-reads fields_full.json from disk."""
    _load_catalog.cache_clear()
