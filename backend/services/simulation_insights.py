"""
simulation_insights.py — Mine simulation history for generation-guiding patterns.

Reads data/memory.json (passing alphas), data/feedback.json (failing), and
data/lessons.jsonl to extract: hot fields, winning settings, archetype success
rates, and common failure modes.

All functions degrade gracefully when history files are empty or missing.

Key exports:
    get_simulation_insights()     → full insights dict
    format_insights_for_prompt()  → LLM-injection string (empty str when no data)
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_MEMORY_PATH    = _ROOT / "data" / "memory.json"
_FEEDBACK_PATH  = _ROOT / "data" / "feedback.json"
_LESSONS_PATH   = _ROOT / "data" / "lessons.jsonl"
_MY_ALPHAS_PATH = _ROOT / "data" / "brain_docs" / "my_alphas.json"

# Operator/keyword tokens to skip when extracting fields from expressions.
_OP_TOKENS = re.compile(
    r"^(ts_mean|ts_sum|ts_std_dev|ts_delta|ts_delay|ts_corr|ts_covariance|"
    r"ts_zscore|ts_rank|ts_max|ts_min|ts_arg_max|ts_arg_min|ts_decay_linear|"
    r"ts_product|ts_returns|ts_skewness|ts_kurtosis|ts_count_nans|"
    r"rank|zscore|scale|normalize|quantile|signed_power|group_neutralize|"
    r"group_rank|group_mean|group_zscore|group_sum|group_max|group_min|"
    r"if_else|abs|log|sign|exp|sqrt|pow|ceil|floor|round|less|greater|"
    r"equal|and|or|not|min|max|add|subtract|multiply|divide|"
    r"sector|industry|subindustry|market|int|float|true|false)$"
)


def _extract_fields(expression: str) -> list[str]:
    tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", expression or "")
    return [t for t in tokens if not _OP_TOKENS.match(t)]


def _load_entries(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        entries = data.get("entries") or data.get("alphas") or []
        return entries if isinstance(entries, list) else []
    return []


def _load_lessons() -> list[dict]:
    try:
        lines = _LESSONS_PATH.read_text().splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def get_simulation_insights() -> dict[str, Any]:
    """Compute simulation insights from all available history.

    Returns a dict with keys:
        total_pass, total_fail, total_simulations
        hot_fields          — {field_id: {"pass": n, "fail": n, "pass_rate": f}}
        archetype_success   — {archetype: {"pass": n, "fail": n, "pass_rate": f}}
        winning_settings    — {setting: {value: count}} for decay/neutralization/truncation
        failure_reasons     — [(reason_text, count), ...]
        has_data            — bool, False if no history exists yet
    """
    passing = _load_entries(_MEMORY_PATH)
    failing = _load_entries(_FEEDBACK_PATH)
    lessons = _load_lessons()

    # Try to supplement with my_alphas.json from Brain API
    my_alphas = _load_entries(_MY_ALPHAS_PATH)

    total_pass = len(passing)
    total_fail = len(failing)
    total_sim  = total_pass + total_fail

    if total_sim == 0 and not lessons:
        return {
            "total_pass": 0, "total_fail": 0, "total_simulations": 0,
            "hot_fields": {}, "archetype_success": {}, "winning_settings": {},
            "failure_reasons": [], "has_data": False,
        }

    # ── field frequency ────────────────────────────────────────────────────
    field_pass: Counter = Counter()
    field_fail: Counter = Counter()

    for entry in passing:
        for f in _extract_fields(entry.get("expression", "")):
            field_pass[f] += 1
    for entry in failing:
        for f in _extract_fields(entry.get("expression", "")):
            field_fail[f] += 1

    all_fields = set(field_pass) | set(field_fail)
    hot_fields: dict[str, dict] = {}
    for fid in all_fields:
        p, ff = field_pass[fid], field_fail[fid]
        total = p + ff
        hot_fields[fid] = {
            "pass": p, "fail": ff,
            "pass_rate": round(p / total, 3) if total else 0.0,
        }

    # ── archetype success ──────────────────────────────────────────────────
    arch_pass: Counter = Counter()
    arch_fail: Counter = Counter()
    for entry in passing:
        a = str(entry.get("archetype") or "unknown").lower()
        arch_pass[a] += 1
    for entry in failing:
        a = str(entry.get("archetype") or "unknown").lower()
        arch_fail[a] += 1

    all_archs = set(arch_pass) | set(arch_fail)
    archetype_success: dict[str, dict] = {}
    for a in all_archs:
        p, ff = arch_pass[a], arch_fail[a]
        total = p + ff
        archetype_success[a] = {
            "pass": p, "fail": ff,
            "pass_rate": round(p / total, 3) if total else 0.0,
        }

    # ── winning settings ───────────────────────────────────────────────────
    decay_pass: Counter = Counter()
    neut_pass:  Counter = Counter()
    trunc_pass: Counter = Counter()
    for entry in passing:
        s = entry.get("settings") or {}
        if s.get("decay") is not None:
            decay_pass[str(s["decay"])] += 1
        if s.get("neutralization"):
            neut_pass[str(s["neutralization"])] += 1
        if s.get("truncation") is not None:
            trunc_pass[str(s["truncation"])] += 1

    winning_settings = {
        "decay": dict(decay_pass.most_common()),
        "neutralization": dict(neut_pass.most_common()),
        "truncation": dict(trunc_pass.most_common()),
    }

    # ── failure reasons ────────────────────────────────────────────────────
    reason_counter: Counter = Counter()
    for entry in failing:
        reasons = entry.get("failures") or []
        for r in reasons:
            msg = str(r.get("message") or r.get("kind") or "").strip()
            if msg:
                reason_counter[msg] += 1
    # Also check lessons for failure context
    for lesson in lessons:
        text = str(lesson.get("lesson") or "").lower()
        if "turnover" in text and "high" in text:
            reason_counter["turnover_too_high"] += 1
        elif "fitness" in text and "low" in text:
            reason_counter["low_fitness"] += 1
        elif "units" in text or "mismatch" in text:
            reason_counter["units_mismatch"] += 1

    failure_reasons = reason_counter.most_common(10)

    return {
        "total_pass": total_pass,
        "total_fail": total_fail,
        "total_simulations": total_sim,
        "hot_fields": hot_fields,
        "archetype_success": archetype_success,
        "winning_settings": winning_settings,
        "failure_reasons": failure_reasons,
        "has_data": True,
    }


def format_insights_for_prompt(top_fields: int = 8) -> str:
    """Return a compact LLM-injection string. Returns empty string when no data."""
    insights = get_simulation_insights()
    if not insights.get("has_data"):
        return ""

    lines = [f"=== SIMULATION HISTORY ({insights['total_simulations']} runs, "
             f"{insights['total_pass']} passed) ==="]

    # Hot fields — exclude always-saturated price/volume tokens to avoid reinforcing
    # the common-field bias that causes high self-correlation.
    _always_sat = {"close", "open", "high", "low", "vwap", "volume", "returns"}
    try:
        from backend.services.field_intelligence import get_saturated_fields
        _always_sat |= get_saturated_fields()
    except Exception:
        pass

    hot = sorted(
        [(fid, v) for fid, v in insights["hot_fields"].items() if fid not in _always_sat],
        key=lambda kv: (kv[1]["pass"], kv[1]["pass_rate"]),
        reverse=True,
    )[:top_fields]
    if hot:
        field_parts = [f"{fid}({v['pass']}✓/{v['pass']+v['fail']})" for fid, v in hot]
        lines.append("Non-saturated fields in passing alphas: " + ", ".join(field_parts))

    # Archetype success rates
    arch = sorted(
        insights["archetype_success"].items(),
        key=lambda kv: kv[1]["pass_rate"],
        reverse=True,
    )
    if arch:
        arch_parts = [f"{a}({int(v['pass_rate']*100)}%)" for a, v in arch if v["pass"] + v["fail"] >= 2]
        if arch_parts:
            lines.append("Archetype pass rates: " + ", ".join(arch_parts))

    # Winning settings
    ws = insights["winning_settings"]
    if ws.get("neutralization"):
        top_neut = list(ws["neutralization"].keys())[0]
        lines.append(f"Most common neutralization in winners: {top_neut}")
    if ws.get("decay"):
        top_decay = list(ws["decay"].keys())[0]
        lines.append(f"Most common decay in winners: {top_decay}")

    # Failure reasons
    if insights["failure_reasons"]:
        top3 = [r[0] for r in insights["failure_reasons"][:3]]
        lines.append("Most common failure reasons: " + ", ".join(top3))

    return "\n".join(lines)
