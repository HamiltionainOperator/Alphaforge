"""Multi-provider LLM service for AlphaForge.

Despite the historical filename, this module dispatches the three LLM jobs
(generation, research synthesis, repair) across selectable providers:
  - "openrouter"  -> OpenRouter chat completions (free owl-alpha by default)
  - "anthropic"   -> Anthropic Messages API (needs ANTHROPIC_API_KEY)
  - "claude_code" -> local Claude Code CLI (`claude -p`), uses the user's auth
``OpenRouterServiceError`` is the shared error type for all providers.
"""
from __future__ import annotations

import asyncio
import functools
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import httpx

from backend.services.search_service import search_provider, web_search
from backend.services.validator_service import normalize_settings, validate_expression


_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPERATORS_PATH = _REPO_ROOT / "data" / "brain_docs" / "operators.json"
_BRAIN_KB_PATH = _REPO_ROOT / "brain_kb.json"
# Operator categories rendered in this order in the catalog the model sees.
_CATEGORY_ORDER = ("arithmetic", "logical", "cross-sectional", "time-series", "group")


def _get_field_context(archetype: str | None, n: int = 20) -> str:
    """Return field intelligence context string for LLM injection. Empty on failure."""
    try:
        from backend.services.field_intelligence import get_field_context_for_prompt
        return get_field_context_for_prompt(archetype=archetype, n=n)
    except Exception:
        return ""


def _get_diversity_context(archetype: str | None) -> str:
    """Return self-correlation avoidance context for LLM injection. Empty on failure."""
    try:
        from backend.services.field_intelligence import get_diversity_context_for_prompt
        return get_diversity_context_for_prompt(archetype=archetype, n=15)
    except Exception:
        return ""


def _get_sim_insights() -> str:
    """Return simulation history insight string for LLM injection. Empty on failure."""
    try:
        from backend.services.simulation_insights import format_insights_for_prompt
        return format_insights_for_prompt()
    except Exception:
        return ""


@functools.lru_cache(maxsize=1)
def _brain_catalog() -> dict[str, Any]:
    """Authoritative Brain operator menu, derived from the SAME data the validator
    uses: data/brain_docs/operators.json minus brain_kb forbidden_operators.

    Returns a dict with a formatted ``text`` catalog (grouped by category, with
    signatures), the ``valid`` operator-name set, the ``forbidden`` set, the
    ``signatures`` name->signature map, and the ``forbidden_fields`` list. This is
    the single source of truth so the prompt can never drift from the validator.
    """
    try:
        ops = json.loads(_OPERATORS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        ops = {}
    try:
        kb = json.loads(_BRAIN_KB_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        kb = {}

    forbidden = {str(o) for o in kb.get("forbidden_operators", [])}
    forbidden_fields = [str(f) for f in kb.get("forbidden_fields", [])]

    valid: set[str] = set()
    signatures: dict[str, str] = {}
    by_cat: dict[str, list[tuple[str, str, str]]] = {}
    for name, meta in ops.items():
        if name.startswith("_") or not isinstance(meta, dict):
            continue
        if name in forbidden:
            continue
        signature = str(meta.get("signature") or name)
        valid.add(name)
        signatures[name] = signature
        category = str(meta.get("category") or "other")
        by_cat.setdefault(category, []).append((name, signature, str(meta.get("description") or "")))

    lines: list[str] = []
    ordered = list(_CATEGORY_ORDER) + [c for c in by_cat if c not in _CATEGORY_ORDER]
    for category in ordered:
        if category not in by_cat:
            continue
        lines.append(f"[{category}]")
        for _name, signature, description in sorted(by_cat[category]):
            lines.append(f"- {signature} — {description}" if description else f"- {signature}")
    return {
        "text": "\n".join(lines),
        "valid": valid,
        "forbidden": forbidden,
        "signatures": signatures,
        "forbidden_fields": forbidden_fields,
    }


def _grammar() -> str:
    """Build the FastExpr grammar from the live catalog so the operator menu shown
    to the model is always exactly the set the validator accepts."""
    catalog = _brain_catalog()
    forbidden_ops = ", ".join(sorted(catalog["forbidden"])) or "—"
    forbidden_fields = ", ".join(catalog["forbidden_fields"]) or "—"
    return f"""FastExpr is a pure daily cross-sectional expression: positive is long, negative is short, magnitude is weight.

You may ONLY use operators from the catalog below. Any operator not in this catalog will FAIL simulation as an unknown operator — never invent or guess operator names.

VALID BRAIN OPERATORS (authoritative — generated from the live Brain catalog):
{catalog['text']}

Core fields: close, open, high, low, vwap, volume, returns.
Fundamentals: assets, sales, equity, debt, liabilities, ebit, ebitda, capex, sharesout.
Group tokens: sector, industry, subindustry.

FORBIDDEN operators (do NOT use — not in Brain): {forbidden_ops}.
Forbidden fields: {forbidden_fields}.

BANNED OPERATOR SUBSTITUTIONS — these names will FAIL simulation with "unknown operator":
- ts_max(x,d)      → ts_rank(x,d)                            [high rank ≈ near-period-max]
- ts_min(x,d)      → ts_rank(-x,d)                           [high rank where x is near-min]
- ts_skewness(x,d) → -ts_mean(x*x*x, d)                     [cubic-moment skewness proxy]
- ts_kurtosis(x,d) → ts_std_dev(x*x, d) / (ts_mean(x*x,d)+1e-8)
- ts_returns(x,d)  → ts_delta(close,1)/ts_delay(close,1)
- decay_linear(…)  → ts_decay_linear(…)                      [ts_ prefix is required]
- winsorize(…)     → remove it; Brain applies truncation automatically

Hard rules:
- Time-series operators ALWAYS take an integer day-window argument: ts_mean(x, d), never ts_mean(x).
- Use ts_decay_linear, never ts_decay or decay_linear.
- Use arithmetic symbols + - * /, never add(), subtract(), multiply(), divide().
- zscore(x) is unary; use ts_zscore(x, d) for the windowed version.
- group_zscore(x, group), group_rank(x, group), group_neutralize(x, group) take EXACTLY 2 args; group_mean(x, weight, group) takes EXACTLY 3.
- Do not add scalar epsilons such as +1 to unitful denominators under unitHandling=VERIFY.
- Prefer continuous gates such as rank(x)-0.5 over binary sign() gates.

FITNESS FORMULA (the gating metric): Fitness = Sharpe * sqrt(|Returns|/max(Turnover, 0.125)).
Target: Fitness >= 1.0, Sharpe >= 1.25.
- HIGH TURNOVER KILLS FITNESS: a 50% daily turnover alpha needs 16x the returns of a 2% turnover alpha to match fitness. Keep turnover in the 5-35% range.
- Use decay >= 6 to smooth signals and reduce churn. Slower signals (fundamentals, 60-day windows) naturally have lower turnover.
- DO NOT use ts_std_dev(returns, 200+) alone — near-zero turnover, never passes.
- DO NOT use raw volume as primary signal — mega-cap bias, low fitness.
- DO NOT use raw ts_delta(close, N) without dividing by close or ts_mean(close, M) — price-level bias.
- Cubic skewness works: -ts_mean(returns*returns*returns, 21) (negative = buy positive skew). Always negate.
- PROVEN high-fitness patterns: zscore(-ts_mean(returns*returns*returns,21)), rank(ts_corr(returns,volume,10)),
  ts_zscore(ebit/assets, 63), zscore(vwap/close)*(1-rank(high/low)), -rank(ebit/capex).
"""


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "openrouter/owl-alpha"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

# --------------------------------------------------------------------------- #
# Global token usage tracker                                                   #
# --------------------------------------------------------------------------- #
import threading as _threading

_usage_lock = _threading.Lock()
_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def get_token_usage() -> dict[str, int]:
    with _usage_lock:
        return dict(_usage)


def reset_token_usage() -> None:
    with _usage_lock:
        _usage["input_tokens"] = 0
        _usage["output_tokens"] = 0
        _usage["calls"] = 0


def _record_usage(input_tokens: int, output_tokens: int) -> None:
    with _usage_lock:
        _usage["input_tokens"] += input_tokens
        _usage["output_tokens"] += output_tokens
        _usage["calls"] += 1

# Max plans generated per single planner call (keeps prompt size reasonable).
_PLAN_BATCH_SIZE = int(os.getenv("ALPHAFORGE_PLAN_BATCH_SIZE", "15"))
# Max concurrent LLM build calls (hypothesis builder + regrounder).
_BUILD_CONCURRENCY = int(os.getenv("ALPHAFORGE_BUILD_CONCURRENCY", "15"))

SYSTEM_PROMPT = """You are a senior quantitative researcher generating WorldQuant Brain FastExpr alphas.
Return only strict JSON. No markdown, no commentary, no code fences."""


class OpenRouterServiceError(RuntimeError):
    pass


def _adaptive_enabled() -> bool:
    return os.getenv("ALPHAFORGE_ADAPTIVE", "1").strip().lower() in ("1", "true", "yes", "on")


async def generate_alphas(
    intent: str,
    archetype: str,
    count: int,
    brief: str = "",
    provider: str = "openrouter",
    hypothesis_data: dict[str, Any] | None = None,
    think_mode: str = "adaptive",
    think_provider: str = "",
    bandit_hints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate alphas with configurable thinking depth.

    think_mode:
      "standard"  — single-shot prompt (fast, less sophisticated)
      "adaptive"  — plan → build (existing default)
      "deep"      — plan → [4 critics → synthesis → refine] × N → build → verify (best quality)

    think_provider: the LLM used for deep-think critics/refine/verify.
      Defaults to provider (gen engine) when empty.
      Set to "claude_code" to think with Claude while generating with owl-alpha.
    """
    safe_count = max(1, min(int(count or 3), 100))
    mode = (think_mode or "adaptive").strip().lower()
    effective_think_provider = (think_provider or provider or "openrouter").strip()

    # Inject bandit hints into brief so the LLM is guided toward high-performing operators
    if bandit_hints and isinstance(bandit_hints, dict):
        persona = bandit_hints.get("persona") or {}
        path = bandit_hints.get("path") or {}
        hint_lines = []
        if persona.get("preferred_operators"):
            hint_lines.append(f"BANDIT_HINT: Prefer operators: {', '.join(str(o) for o in persona['preferred_operators'][:5])}")
        if persona.get("preferred_fields"):
            hint_lines.append(f"BANDIT_HINT: Prefer fields: {', '.join(str(f) for f in persona['preferred_fields'][:5])}")
        if path.get("strategy"):
            hint_lines.append(f"BANDIT_HINT: Winning strategy recently: {path['strategy']}")
        if hint_lines:
            brief = (brief + "\n\n" + "\n".join(hint_lines)).strip()

    # Inject the avoidance context (gen2 DuplicateDetector port): feed the literal
    # recent expressions back into the prompt so the LLM produces unique alphas.
    try:
        from backend.services.duplicate_detector import get_duplicate_detector

        avoidance = get_duplicate_detector().get_avoidance_context(limit=15)
        if avoidance:
            brief = (brief + "\n\n" + avoidance).strip()
    except Exception:  # noqa: BLE001
        pass

    # Hypothesis-guided path: use structured hypothesis metadata directly.
    if hypothesis_data and isinstance(hypothesis_data, dict) and hypothesis_data.get("fields_suggested"):
        try:
            return await _generate_alphas_from_hypothesis(
                intent, archetype, safe_count, brief, provider, hypothesis_data
            )
        except OpenRouterServiceError:
            pass  # Fall through to adaptive/deep/single-shot

    if mode == "deep" and _adaptive_enabled():
        try:
            return await _generate_alphas_deep(
                intent, archetype, safe_count, brief,
                gen_provider=provider,
                think_provider=effective_think_provider,
            )
        except OpenRouterServiceError:
            pass  # Fall through to adaptive

    if mode != "standard" and _adaptive_enabled():
        try:
            return await _generate_alphas_adaptive(intent, archetype, safe_count, brief, provider)
        except OpenRouterServiceError:
            pass
    return await _generate_alphas_single_shot(intent, archetype, safe_count, brief, provider)


async def _generate_alphas_single_shot(
    intent: str, archetype: str, safe_count: int, brief: str, provider: str
) -> list[dict[str, Any]]:
    prompt = _build_generation_prompt(intent=intent, archetype=archetype, count=safe_count, brief=brief)
    text = await _complete(
        provider,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "0.65")),
    )
    parsed = _extract_json(text)
    items = _extract_alpha_items(parsed)
    if not items:
        preview = (text or "")[:200].replace("\n", " ")
        raise OpenRouterServiceError(
            f"the model returned no parseable alpha expressions. "
            f"Raw response preview: {preview!r}"
        )

    alphas: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        alpha = _coerce_alpha(item, index=len(alphas), requested_archetype=archetype)
        expr_key = re.sub(r"\s+", "", alpha["expression"]).lower()
        if not expr_key or expr_key in seen:
            continue
        seen.add(expr_key)
        alphas.append(alpha)
        if len(alphas) >= safe_count:
            break

    if not alphas:
        raise OpenRouterServiceError("the model returned no parseable alpha expressions.")
    return alphas


# --------------------------------------------------------------- adaptive flow

PLAN_SYSTEM = (
    "You are a senior quantitative researcher who reasons step by step. You FIRST understand "
    "the economic mechanism of a hypothesis, THEN decompose it into concrete computational steps, "
    "THEN map each step to specific WorldQuant Brain operators chosen from a fixed catalog. "
    "You never invent operators. Return only strict JSON."
)


def _plan_prompt(
    intent: str,
    archetype: str,
    count: int,
    brief: str,
    exclude_archetypes: list[str] | None = None,
) -> str:
    clean_intent = (intent or "").strip() or "Find a robust, tractable cross-sectional USA equity alpha."
    clean_arch = (archetype or "any").strip()
    arch_line = "researcher's choice" if clean_arch == "any" else clean_arch
    brief_block = (
        f"\nResearch brief (ground your reasoning in this):\n{brief.strip()}\n"
        if (brief or "").strip()
        else ""
    )
    arch_exclusion_block = ""
    if exclude_archetypes:
        used = ", ".join(sorted(set(str(a) for a in exclude_archetypes if a)))
        arch_exclusion_block = (
            f"\n\nBATCH DIVERSITY — the following archetypes have ALREADY been planned. "
            f"Do NOT repeat them:\nAlready used: {used}\n"
        )
    field_ctx = _get_field_context(clean_arch if clean_arch != "any" else None, n=20)
    field_block = (
        f"\n\n=== FIELD CATALOG (reference — what fields exist in Brain) ===\n"
        f"{field_ctx}\n"
        "NOTE: High alpha_count fields like close/open/volume/returns are proven BUT saturated — "
        "see the SELF-CORRELATION AVOIDANCE section below before selecting fields."
    ) if field_ctx else ""
    diversity_ctx = _get_diversity_context(clean_arch if clean_arch != "any" else None)
    diversity_block = f"\n\n{diversity_ctx}" if diversity_ctx else ""
    sim_insights = _get_sim_insights()
    insights_block = f"\n\n{sim_insights}" if sim_insights else ""
    return f"""{_grammar()}{field_block}{diversity_block}{insights_block}

Research intent:
{clean_intent}
{brief_block}
Target archetype: {arch_line}
Number of alphas to plan: {count}
{arch_exclusion_block}
Think ADAPTIVELY, one alpha at a time. Do NOT write the final expression yet. For EACH alpha:
1. hypothesis — why this signal predicts forward cross-sectional returns (1-2 sentences).
2. mechanism_steps — break the computation into 2-4 ordered, concrete steps in plain words. CRITICAL: each step must target LOW TURNOVER (5-35% daily). Prefer fundamental ratios, 20-63 day windows, and rank/zscore normalization. Avoid short windows (<5 days) as the SOLE signal.
3. operators — for those steps, list the EXACT operator names you will use. Every name MUST appear verbatim in the catalog above. If a step needs an operator that is not in the catalog, redesign the step to use one that is. Include ts_zscore or rank for normalization (essential for good Sharpe).
4. fields — CRITICAL SELF-CORRELATION RULE: Do NOT build the primary signal using ONLY {{close, open, volume, returns, vwap, high, low}}. Brain's SELF_CORRELATION check WILL FAIL. At least one key input MUST come from the UNDEREXPLORED list in the SELF-CORRELATION AVOIDANCE section above (Analyst, Option, Model, or Fundamental fields not already in your pool).
5. decay_hint — integer 6-10 for slow fundamental signals, 4-6 for medium-speed signals. Never below 4.

Return exactly this JSON shape, nothing else:
{{
  "plans": [
    {{
      "name": "<five words or fewer>",
      "archetype": "reversal|microstructure|volatility|fundamental|analyst_revision|earnings_event|options_implied|factor_residual|dispersion|novel",
      "hypothesis": "<why it works>",
      "mechanism_steps": ["step 1", "step 2"],
      "operators": ["ts_mean", "rank"],
      "fields": ["close", "volume"],
      "decay_hint": 6
    }}
  ]
}}"""


def _build_prompt(plan: dict[str, Any], intent: str) -> str:
    catalog = _brain_catalog()
    chosen = [o for o in plan.get("operators", []) if o in catalog["valid"]]
    sig_lines = "\n".join(f"- {catalog['signatures'][o]}" for o in chosen) or "- (none selected — pick from the catalog)"
    steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan.get("mechanism_steps", []))) or "(none)"
    fields = ", ".join(str(f) for f in plan.get("fields", [])) or "researcher's choice"
    decay = max(4, min(10, int(plan.get("decay_hint") or 6)))
    return f"""{_grammar()}

Now WRITE the FastExpr expression for this planned alpha. Implement the mechanism faithfully and use ONLY the selected operators (their exact signatures are given). You may use fewer of them, but introduce no operator outside this list.

Name: {plan.get('name', 'alpha')}
Hypothesis: {plan.get('hypothesis', '')}
Mechanism steps:
{steps}
Selected operators (use only these):
{sig_lines}
Input fields: {fields}
Original intent: {(intent or '').strip()}

Return ONLY this JSON object, no markdown:
{{
  "name": "{plan.get('name', 'alpha')}",
  "archetype": "{plan.get('archetype', 'novel')}",
  "hypothesis": "{plan.get('hypothesis', '')}",
  "expression": "<valid FastExpr using only the selected operators>",
  "settings": {{"universe": "TOP3000", "neutralization": "Subindustry", "decay": {decay}, "truncation": 0.05, "delay": 1}},
  "expected_sharpe": "x.x-y.y",
  "expected_fitness": "x.x",
  "regime_note": "<one sentence>",
  "turnover_note": "<target turnover %: 5-35% for good fitness>"
}}"""


async def _plan_alphas(
    intent: str,
    archetype: str,
    count: int,
    brief: str,
    provider: str,
    exclude_archetypes: list[str] | None = None,
    thinking_budget: int | None = None,
) -> list[dict[str, Any]]:
    # Scale token budget with batch size; cap at 6000 to avoid over-spending.
    plan_tokens = max(2000, min(6000, 300 * count))
    text = await _complete(
        provider,
        [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": _plan_prompt(intent, archetype, count, brief, exclude_archetypes=exclude_archetypes)},
        ],
        temperature=float(os.getenv("ALPHAFORGE_PLAN_TEMPERATURE", "0.5")),
        max_tokens=plan_tokens,
        thinking_budget=thinking_budget,
    )
    parsed = _extract_json(text)
    plans = parsed.get("plans") if isinstance(parsed, dict) else parsed
    if not isinstance(plans, list):
        raise OpenRouterServiceError("the planner returned JSON without a plans array.")
    catalog = _brain_catalog()
    clean: list[dict[str, Any]] = []
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        ops = [str(o).strip() for o in plan.get("operators", []) if str(o).strip()]
        plan["operators"] = [o for o in ops if o in catalog["valid"]]
        # Surface operators the planner picked that are NOT in Brain — used only for
        # observability; they are already dropped from the builder's toolset.
        plan["_rejected_operators"] = [o for o in ops if o not in catalog["valid"]]
        clean.append(plan)
    if not clean:
        raise OpenRouterServiceError("the planner produced no usable alpha plans.")
    return clean


async def _build_alpha(plan: dict[str, Any], index: int, intent: str, archetype: str, provider: str) -> dict[str, Any] | None:
    text = await _complete(
        provider,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(plan, intent)},
        ],
        temperature=float(os.getenv("ALPHAFORGE_BUILD_TEMPERATURE", "0.45")),
        max_tokens=800,
    )
    parsed = _extract_json(text)
    # If the model wrapped the single alpha in an array or another key, unwrap it.
    if not isinstance(parsed, dict) or not str(parsed.get("expression") or "").strip():
        items = _extract_alpha_items(parsed)
        if not items or not isinstance(items[0], dict) or not str(items[0].get("expression") or "").strip():
            return None
        parsed = items[0]
    parsed.setdefault("name", plan.get("name"))
    parsed.setdefault("hypothesis", plan.get("hypothesis"))
    parsed.setdefault("archetype", plan.get("archetype"))
    alpha = _coerce_alpha(parsed, index=index, requested_archetype=archetype)
    alpha["plan"] = {
        "mechanism_steps": plan.get("mechanism_steps", []),
        "operators": plan.get("operators", []),
    }
    return alpha


# Operators that are in brain_kb.json's forbidden list but may not be caught by
# the MathEngine validator (it only knows its own rule set).  Map op → fix hint.
_BANNED_OP_FIXES: dict[str, str] = {
    "ts_max":      "Replace ts_max(x,d) with ts_rank(x,d) — high rank indicates near-period-max.",
    "ts_min":      "Replace ts_min(x,d) with ts_rank(-x,d) — high rank where x is near its minimum.",
    "ts_skewness": "Replace ts_skewness(x,d) with -ts_mean(x*x*x, d) for the cubic-moment skewness proxy.",
    "ts_kurtosis": "Replace ts_kurtosis(x,d) with ts_std_dev(x*x,d)/(ts_mean(x*x,d)+1e-8).",
    "ts_returns":  "Replace ts_returns with ts_delta(close,1)/ts_delay(close,1).",
    "decay_linear":"Use ts_decay_linear(x,d) — the ts_ prefix is required.",
    "winsorize":   "Remove winsorize(); Brain applies truncation automatically via settings.truncation.",
    "ts_ir":       "ts_ir is not available; use ts_corr or a ratio of ts_mean/ts_std_dev instead.",
}


def _non_brain_operator_issues(expression: str, settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Local validator issues that specifically mean 'this used a non-Brain operator'.

    Combines the MathEngine's rule-based issues with a regex pass over the
    expression for operators that are in brain_kb.json's forbidden list but
    may not be flagged by the MathEngine (e.g. ts_max, ts_min).
    """
    report = validate_expression(expression, settings)
    issues = [
        i for i in report.get("issues", [])
        if i.get("rule") in ("unknown_operator", "forbidden_operator_talib", "invalid_delay_operator")
    ]
    expr_lower = (expression or "").lower()
    for op, fix in _BANNED_OP_FIXES.items():
        if re.search(rf"\b{re.escape(op)}\s*\(", expr_lower):
            issues.append({
                "verdict": "FAIL",
                "rule": "unknown_operator",
                "message": f"'{op}' is not available in Brain FastExpr and will fail simulation.",
                "fix": fix,
            })
    return issues


async def _reground_operators(alpha: dict[str, Any], provider: str) -> dict[str, Any]:
    """If a built alpha still references operators outside the Brain catalog, repair it
    by re-grounding on the catalog. Runs up to 2 passes before giving up."""
    for _ in range(2):
        issues = _non_brain_operator_issues(alpha["expression"], alpha.get("settings", {}))
        if not issues:
            return alpha
        try:
            fixed = await repair_alpha(
                alpha["expression"],
                issues=issues,
                intent=alpha.get("hypothesis", ""),
                archetype=alpha.get("archetype", ""),
                provider=provider,
            )
        except OpenRouterServiceError:
            return alpha
        alpha["expression"] = fixed["expression"]
    return alpha


async def _generate_alphas_adaptive(
    intent: str, archetype: str, safe_count: int, brief: str, provider: str
) -> list[dict[str, Any]]:
    # For large counts batch the planner so each call stays within a manageable token budget
    # and we can request diversity hints from the second batch onward.
    if safe_count <= _PLAN_BATCH_SIZE:
        plans = await _plan_alphas(intent, archetype, safe_count, brief, provider)
    else:
        all_plans: list[dict[str, Any]] = []
        remaining = safe_count
        while remaining > 0:
            batch_n = min(_PLAN_BATCH_SIZE, remaining)
            used_archetypes = [p.get("archetype", "") for p in all_plans] if all_plans else None
            try:
                batch = await _plan_alphas(
                    intent, archetype, batch_n, brief, provider,
                    exclude_archetypes=used_archetypes,
                )
            except OpenRouterServiceError:
                break
            all_plans.extend(batch)
            remaining -= batch_n
        if not all_plans:
            raise OpenRouterServiceError("adaptive generation produced no plans.")
        plans = all_plans

    # Concurrency-limited building.
    build_sem = asyncio.Semaphore(_BUILD_CONCURRENCY)

    async def _build_one(plan: dict[str, Any], idx: int) -> dict[str, Any] | None:
        async with build_sem:
            return await _build_alpha(plan, idx, intent, archetype, provider)

    built = await asyncio.gather(
        *(_build_one(plan, idx) for idx, plan in enumerate(plans)),
        return_exceptions=True,
    )
    candidates = [a for a in built if isinstance(a, dict)]

    # Catch any non-Brain operator that slipped past the planner/builder before sim.
    async def _reground_one(a: dict[str, Any]) -> dict[str, Any]:
        async with build_sem:
            return await _reground_operators(a, provider)

    regrounded = await asyncio.gather(
        *(_reground_one(a) for a in candidates),
        return_exceptions=True,
    )

    alphas: list[dict[str, Any]] = []
    seen: set[str] = set()
    for alpha in regrounded:
        if not isinstance(alpha, dict):
            continue
        expr_key = re.sub(r"\s+", "", alpha.get("expression", "")).lower()
        if not expr_key or expr_key in seen:
            continue
        seen.add(expr_key)
        alpha["id"] = f"a{len(alphas)}"
        alphas.append(alpha)
        if len(alphas) >= safe_count:
            break

    if not alphas:
        raise OpenRouterServiceError("adaptive generation produced no parseable alpha expressions.")
    return alphas


async def _generate_alphas_deep(
    intent: str,
    archetype: str,
    safe_count: int,
    brief: str,
    gen_provider: str,
    think_provider: str = "",
) -> list[dict[str, Any]]:
    """Deep think: plan → [4 specialists critique → synthesis → refine] × N → build → verify → reground.

    gen_provider   — used for planning and expression building.
    think_provider — used for critics, synthesis, refine, verify (defaults to gen_provider).
                     Set to "claude_code" to think with Claude while building with owl-alpha.
    """
    from backend.services.thinking_service import deep_think_plans, verify_built_alphas

    effective_think = (think_provider or gen_provider).strip() or gen_provider

    # 1 — plan (gen_provider writes the initial plan; think_provider will critique it)
    is_think_anthropic = _normalize_provider(effective_think) == "anthropic"
    plan_thinking_budget = int(os.getenv("ALPHAFORGE_THINK_BUDGET", "10000")) if is_think_anthropic else None

    if safe_count <= _PLAN_BATCH_SIZE:
        raw_plans = await _plan_alphas(
            intent, archetype, safe_count, brief, gen_provider,
            thinking_budget=plan_thinking_budget,
        )
    else:
        all_plans: list[dict[str, Any]] = []
        remaining = safe_count
        while remaining > 0:
            batch_n = min(_PLAN_BATCH_SIZE, remaining)
            used_archetypes = [p.get("archetype", "") for p in all_plans] if all_plans else None
            try:
                batch = await _plan_alphas(
                    intent, archetype, batch_n, brief, gen_provider,
                    exclude_archetypes=used_archetypes,
                    thinking_budget=plan_thinking_budget,
                )
            except OpenRouterServiceError:
                break
            all_plans.extend(batch)
            remaining -= batch_n
        if not all_plans:
            raise OpenRouterServiceError("deep generation produced no plans.")
        raw_plans = all_plans

    # 2 — multi-specialist critique → synthesis → iterative refinement (think_provider)
    plans = await deep_think_plans(raw_plans, effective_think)

    # 3 — build expressions from refined plans (gen_provider writes the expression)
    build_sem = asyncio.Semaphore(_BUILD_CONCURRENCY)

    async def _build_one(plan: dict[str, Any], idx: int) -> dict[str, Any] | None:
        async with build_sem:
            return await _build_alpha(plan, idx, intent, archetype, gen_provider)

    built = await asyncio.gather(
        *(_build_one(plan, idx) for idx, plan in enumerate(plans)),
        return_exceptions=True,
    )
    candidates = [a for a in built if isinstance(a, dict)]

    # 4 — LLM expression verifier (think_provider checks syntax before simulation)
    candidates = await verify_built_alphas(candidates, effective_think)

    # 5 — reground (deterministic safety net)
    async def _reground_one(a: dict[str, Any]) -> dict[str, Any]:
        async with build_sem:
            return await _reground_operators(a, gen_provider)

    regrounded = await asyncio.gather(
        *(_reground_one(a) for a in candidates),
        return_exceptions=True,
    )

    alphas: list[dict[str, Any]] = []
    seen: set[str] = set()
    for alpha in regrounded:
        if not isinstance(alpha, dict):
            continue
        expr_key = re.sub(r"\s+", "", alpha.get("expression", "")).lower()
        if not expr_key or expr_key in seen:
            continue
        seen.add(expr_key)
        alpha["id"] = f"a{len(alphas)}"
        alpha["deep_think"] = True
        alphas.append(alpha)
        if len(alphas) >= safe_count:
            break

    if not alphas:
        raise OpenRouterServiceError("deep generation produced no parseable alpha expressions.")
    return alphas


# --------------------------------------------------------------------------- #
# Hypothesis-guided generation                                                 #
# --------------------------------------------------------------------------- #

# Archetype → preferred operators (validated against the live catalog at call time).
# These are SUGGESTIONS shown first to the LLM, not hard restrictions — the prompt
# now says "preferred" so the model can pull in other catalog operators as needed.
_ARCHETYPE_OPERATORS: dict[str, list[str]] = {
    "reversal":         ["ts_delta", "ts_mean", "ts_rank", "ts_sum", "rank", "zscore", "ts_corr", "ts_zscore"],
    "microstructure":   ["ts_corr", "zscore", "rank", "ts_mean", "ts_sum", "ts_std_dev", "ts_zscore"],
    "volatility":       ["ts_std_dev", "abs", "rank", "zscore", "ts_mean", "signed_power", "ts_zscore", "ts_rank"],
    "fundamental":      ["ts_zscore", "rank", "zscore", "group_zscore", "ts_mean", "ts_delta", "group_neutralize"],
    "analyst_revision": ["ts_zscore", "rank", "zscore", "ts_delta", "ts_mean", "ts_corr"],
    "earnings_event":   ["zscore", "rank", "ts_zscore", "ts_mean", "ts_delta", "ts_std_dev"],
    "options_implied":  ["zscore", "rank", "ts_zscore", "ts_std_dev", "ts_mean", "ts_corr"],
    "factor_residual":  ["group_zscore", "group_neutralize", "group_rank", "rank", "zscore", "ts_zscore"],
    "dispersion":       ["rank", "zscore", "ts_zscore", "ts_std_dev", "ts_mean", "ts_corr"],
    "novel":            ["rank", "zscore", "ts_mean", "ts_std_dev", "ts_corr", "signed_power", "group_neutralize", "ts_zscore"],
}

# Variation instruction indices that reference hardcoded fundamental fields (ebit, assets,
# debt, sales, capex). These should only be applied to hypotheses that actually involve
# fundamental data — applying them to volatility/microstructure hypotheses forces all
# variations into the same fundamental-quality structural pattern regardless of the mechanism.
_FUNDAMENTAL_VARIATION_INDICES: frozenset[int] = frozenset([2, 10, 12, 18, 21, 23])
# Indices that are meaningful only when the hypothesis uses volume as a primary field.
_VOLUME_VARIATION_INDICES: frozenset[int] = frozenset([22])
_FUNDAMENTAL_ARCHETYPES: frozenset[str] = frozenset(
    ["fundamental", "analyst_revision", "earnings_event", "factor_residual"]
)
_FUNDAMENTAL_FIELDS: frozenset[str] = frozenset(
    ["ebit", "ebitda", "assets", "equity", "debt", "liabilities", "sales", "capex", "sharesout"]
)


def _applicable_variation_indices(archetype: str, fields: list[str]) -> list[int]:
    """Return the subset of _VARIATION_INSTRUCTIONS indices appropriate for this hypothesis.

    Fundamental-specific instructions (quality gates, leverage filters, sales screens)
    are excluded unless the hypothesis archetype or suggested fields are fundamentals-based.
    Volume-specific instructions are excluded unless volume is in the suggested fields.
    """
    field_set = {f.lower() for f in fields}
    is_fundamental = (
        archetype in _FUNDAMENTAL_ARCHETYPES
        or bool(field_set & _FUNDAMENTAL_FIELDS)
    )
    has_volume = "volume" in field_set or archetype in ("microstructure", "dispersion")

    excluded: set[int] = set()
    if not is_fundamental:
        excluded |= _FUNDAMENTAL_VARIATION_INDICES
    if not has_volume:
        excluded |= _VOLUME_VARIATION_INDICES

    available = [i for i in range(len(_VARIATION_INSTRUCTIONS)) if i not in excluded]
    return available if available else list(range(len(_VARIATION_INSTRUCTIONS)))

# Per-variation extra instructions so N alphas from one hypothesis are genuinely distinct.
# 25 instructions cover every major alpha-engineering dimension; for counts > 25 the list
# cycles with an increasing temperature and an explicit "distinctly different" directive.
_VARIATION_INSTRUCTIONS: list[str] = [
    # 0 — base: faithful implementation of the mechanism
    "",
    # 1 — simplified
    "SIMPLIFIED form: use at most 2 of the required fields. Capture only the primary signal; omit secondary interactions.",
    # 2 — quality gate
    "QUALITY GATE: multiply the primary signal by rank(ebit/assets) or rank(equity/assets) to favor high-quality companies.",
    # 3 — long lookback
    "LONGER LOOKBACK (2–3× the implied windows) for the slow-decay, low-turnover version. Increase decay to 8–10.",
    # 4 — group neutralization sector
    "SECTOR GROUP NEUTRALIZATION: apply group_zscore(signal, sector) to isolate the idiosyncratic component.",
    # 5 — volume confirmation
    "VOLUME CONFIRMATION: multiply by rank(ts_sum(volume,5)/ts_mean(volume,60)) to confirm with recent liquidity.",
    # 6 — short-window fast
    "SHORT-WINDOW (3–7 day) version. High-frequency rebalancing; keep decay=4 to avoid over-churn.",
    # 7 — interaction product
    "INTERACTION: form a multiplicative interaction — rank(signal_A) * rank(signal_B) — from two sub-components of the mechanism.",
    # 8 — correlation-based
    "CORRELATION-BASED: replace direct signal with ts_corr() between two hypothesis fields over 10–20 days.",
    # 9 — ratio form
    "RATIO form: express as rank(field_A / field_B) (e.g., ebit/sales, close/vwap). Normalize with rank() or zscore().",
    # 10 — leverage filter
    "LEVERAGE FILTER: multiply by (1 - rank(debt/assets)) to overweight low-leverage stocks.",
    # 11 — skewness gate
    "SKEWNESS GATE: incorporate -rank(ts_mean(returns*returns*returns, 21)) to favor positive-skew stocks.",
    # 12 — sales growth
    "SALES GROWTH SCREEN: multiply by rank(ts_delta(sales, 252) / ts_mean(sales, 63)) for revenue-momentum quality.",
    # 13 — industry neutralization
    "INDUSTRY NEUTRALIZATION: use group_zscore(signal, industry) for tight within-industry signal comparison.",
    # 14 — volatility-adjusted
    "VOLATILITY-ADJUSTED: divide the signal by ts_std_dev(returns, 21) to normalize per unit of realized risk.",
    # 15 — composite weighted
    "COMPOSITE: weighted sum of two distinct sub-signals — 0.6*rank(signal_A) + 0.4*rank(signal_B), each derived from the mechanism.",
    # 16 — market residual
    "MARKET RESIDUAL: apply group_neutralize(signal, market) first, then rank, to strip out market-direction bias.",
    # 17 — annual lookback
    "ANNUAL LOOKBACK (252-day windows) for the ultra-low-turnover, fundamentals-dominant variant. decay=10.",
    # 18 — capex efficiency
    "CAPEX EFFICIENCY gate: multiply by rank(-capex / (assets + 1e-6)) to favor asset-light, high-ROIC companies.",
    # 19 — momentum combination
    "MOMENTUM LEG: add ts_zscore(returns, 21) as a secondary signal leg for trend-confirmation weighting.",
    # 20 — simplified + subindustry neutral
    "SIMPLIFIED + SUBINDUSTRY NEUTRAL: at most 2 fields AND group_zscore(signal, subindustry).",
    # 21 — quality + long lookback
    "QUALITY + LONG LOOKBACK: ebit/assets quality gate combined with 3× extended time windows.",
    # 22 — volume + correlation
    "VOLUME-WEIGHTED CORRELATION: ts_corr(primary_field, volume, 20) normalized by rank().",
    # 23 — debt filter + industry neutral
    "DEBT FILTER + INDUSTRY NEUTRAL: (1 - rank(debt/assets)) leverage screen AND group_zscore(signal, industry).",
    # 24 — skewness + high decay
    "SKEWNESS + HIGH DECAY: cubic-returns filter combined with decay=10 for the absolute minimum-turnover version.",
]


def _operators_for_archetype(archetype: str, catalog: dict[str, Any]) -> list[str]:
    ops = _ARCHETYPE_OPERATORS.get(str(archetype).lower(), _ARCHETYPE_OPERATORS["novel"])
    return [op for op in ops if op in catalog["valid"]]


def _turnover_range_to_decay(turnover_range: str) -> int:
    """'15-30%' → decay hint. Low turnover target → higher decay to smooth signal."""
    nums = re.findall(r"\d+(?:\.\d+)?", str(turnover_range or ""))
    if not nums:
        return 6
    values = [float(n) for n in nums[:2]]
    midpoint = sum(values) / len(values)
    if midpoint < 12:
        return 10
    elif midpoint < 20:
        return 8
    elif midpoint < 30:
        return 6
    return 4


def _hypothesis_to_plan(hyp: dict[str, Any], variation_index: int = 0, brief: str = "") -> dict[str, Any]:
    """Convert a HypothesisEngine dict into a planner-compatible plan dict.

    The hypothesis engine already did the hard work of identifying fields, mechanism,
    archetype, and expected performance. This extracts that structure rather than
    having the LLM planner re-derive it from free-form text.
    """
    catalog = _brain_catalog()
    archetype = str(hyp.get("archetype") or "novel").strip().lower()
    decay_hint = _turnover_range_to_decay(str(hyp.get("expected_turnover_range") or "15-35%"))

    # Decompose mechanism text into 2-3 ordered steps.
    mechanism_text = str(hyp.get("mechanism") or hyp.get("claim") or "")
    raw_steps = [s.strip() for s in re.split(r"[.;]", mechanism_text) if len(s.strip()) > 10]
    steps = raw_steps[:3] if raw_steps else [mechanism_text or str(hyp.get("claim", ""))]

    fields = [str(f).strip() for f in (hyp.get("fields_suggested") or []) if f]
    operators = _operators_for_archetype(archetype, catalog)

    # Filter the variation pool to only instructions that make sense for this hypothesis's
    # archetype and fields.  Cycling through the full unfiltered list causes fundamental-
    # specific instructions (quality gates, leverage filters, sales screens) to appear for
    # volatility/microstructure hypotheses, making all archetypes produce the same
    # structural alpha patterns.
    available_slots = _applicable_variation_indices(archetype, fields)
    num_available = len(available_slots)
    slot_position = variation_index % num_available
    variation_cycle = variation_index // num_available
    variation_slot = available_slots[slot_position]

    # Give each variation a distinct, descriptive name so they're identifiable in the UI.
    base_title = str(hyp.get("title") or "alpha").strip()[:40]
    plan_name = f"{base_title} v{variation_index + 1}" if variation_index > 0 else base_title

    plan: dict[str, Any] = {
        "name": plan_name[:50],
        "archetype": archetype,
        "hypothesis": str(hyp.get("claim") or ""),
        "mechanism": str(hyp.get("mechanism") or ""),
        "mechanism_steps": steps,
        "operators": operators,
        "fields": fields,
        "decay_hint": decay_hint,
        "regime_guard": str(hyp.get("regime_guard") or ""),
        "novelty_reason": str(hyp.get("novelty_reason") or ""),
        "expected_sharpe_range": str(hyp.get("expected_sharpe_range") or "1.0-2.0"),
        "expected_turnover_range": str(hyp.get("expected_turnover_range") or "15-35%"),
        "_from_hypothesis": True,
        "_variation_index": variation_index,
        "_variation_cycle": variation_cycle,
        "_brief": brief.strip() if brief else "",
    }

    base_instruction = _VARIATION_INSTRUCTIONS[variation_slot]
    if variation_cycle > 0:
        # Second+ cycle of the same variation slot — demand a structurally distinct expression.
        cycle_tag = (
            f" [Cycle {variation_cycle + 1}: produce a STRUCTURALLY DIFFERENT expression — "
            "use different operators, different field interactions, or a different normalization "
            "approach from earlier alphas of this variation.]"
        )
        if base_instruction:
            plan["_extra_instruction"] = base_instruction + cycle_tag
        else:
            plan["_extra_instruction"] = (
                f"Cycle {variation_cycle + 1}: generate a completely fresh expression. "
                "Choose different operators, different field combinations, and a different "
                "signal architecture from all previous alphas of this hypothesis."
            )
    else:
        plan["_extra_instruction"] = base_instruction
    return plan


def _build_prompt_from_hypothesis(plan: dict[str, Any], intent: str) -> str:
    """Build the expression-generation prompt with hypothesis metadata as explicit constraints.

    Unlike the generic _build_prompt, this:
    - Presents fields as REQUIRED inputs (not 'researcher's choice')
    - Explains the full mechanism rather than a bare hypothesis sentence
    - Embeds the turnover/sharpe targets derived from the hypothesis engine
    - Injects a per-variation instruction for diversity across count>1 calls
    """
    catalog = _brain_catalog()
    chosen_ops = [o for o in plan.get("operators", []) if o in catalog["valid"]]
    sig_lines = "\n".join(f"- {catalog['signatures'][o]}" for o in chosen_ops) or "- (see catalog above)"
    steps = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(plan.get("mechanism_steps", []))) or "  (see mechanism below)"

    fields = plan.get("fields") or []
    if fields:
        field_block = (
            "REQUIRED fields (pre-validated against the Brain catalog by the hypothesis engine — "
            "use AT LEAST the primary ones; do NOT substitute simpler names):\n"
            + "\n".join(f"  - {f}" for f in fields)
        )
    else:
        field_block = "Input fields: researcher's choice — pick from Core fields and Fundamentals in the grammar above."

    decay = max(4, min(10, int(plan.get("decay_hint") or 6)))
    turnover_range = plan.get("expected_turnover_range", "15-35%")
    sharpe_range = plan.get("expected_sharpe_range", "1.0-2.0")

    regime_line = f"\nRegime guard (when signal fails): {plan['regime_guard']}" if plan.get("regime_guard") else ""
    novelty_line = f"\nNovelty context: {plan['novelty_reason']}" if plan.get("novelty_reason") else ""
    extra = plan.get("_extra_instruction", "")
    extra_block = f"\n\nVARIATION — apply this specifically:\n{extra}" if extra else ""

    brief_text = (plan.get("_brief") or "").strip()
    brief_block = f"\n\n=== RESEARCH CONTEXT ===\n{brief_text}" if brief_text else ""

    # Field intelligence: show top fields for this archetype to guide implementation.
    field_ctx = _get_field_context(plan.get("archetype"), n=15)
    field_intel_block = (
        f"\n\n=== FIELD CATALOG (reference) ===\n{field_ctx}"
    ) if field_ctx else ""
    # Diversity: show which fields are already saturated and what to use instead.
    div_ctx = _get_diversity_context(plan.get("archetype"))
    field_intel_block += f"\n\n{div_ctx}" if div_ctx else ""

    # Sanitize strings embedded in the JSON template so stray quotes don't break it.
    safe_name = plan.get("name", "alpha").replace('"', "'")
    safe_arch = plan.get("archetype", "novel").replace('"', "'")
    safe_hyp = plan.get("hypothesis", "").replace('"', "'")
    safe_regime = plan.get("regime_guard", "").replace('"', "'")
    safe_intent = (intent or "").strip().replace('"', "'") or "(implement the hypothesis above)"

    return f"""{_grammar()}

You are implementing a pre-specified economic hypothesis as a WorldQuant Brain FastExpr alpha.
The hypothesis was generated by a research engine — faithfully implement the stated mechanism.{brief_block}

=== HYPOTHESIS ===
Name: {safe_name}
Archetype: {safe_arch}
Economic claim: {safe_hyp}
Full mechanism: {plan.get('mechanism', '').replace('"', "'")}

=== IMPLEMENTATION STEPS ===
{steps}

=== INPUT DATA ===
{field_block}{field_intel_block}

=== PREFERRED OPERATORS (start with these; may supplement from the full catalog above when the mechanism specifically requires it) ===
{sig_lines}

=== TARGETS ===
Expected turnover: {turnover_range} → settings.decay = {decay} (required).
Expected Sharpe: {sharpe_range}.{regime_line}{novelty_line}{extra_block}

Original user intent: {safe_intent}

Implement the mechanism faithfully. Use rank() or zscore() for cross-sectional normalization.
Ensure unit consistency in if_else/max/min branches.

Return ONLY this JSON, no markdown:
{{
  "name": "{safe_name}",
  "archetype": "{safe_arch}",
  "hypothesis": "{safe_hyp}",
  "expression": "<valid FastExpr>",
  "settings": {{"universe": "TOP3000", "neutralization": "Subindustry", "decay": {decay}, "truncation": 0.05, "delay": 1}},
  "expected_sharpe": "{sharpe_range}",
  "expected_fitness": "1.0",
  "regime_note": "{safe_regime}",
  "turnover_note": "target {turnover_range} turnover"
}}"""


async def _build_alpha_from_hypothesis(
    plan: dict[str, Any], index: int, intent: str, archetype: str, provider: str
) -> dict[str, Any] | None:
    base_temp = float(os.getenv("ALPHAFORGE_BUILD_TEMPERATURE", "0.45"))
    # Each extra cycle bumps temperature slightly so the model generates increasingly
    # creative (yet still grounded) expressions rather than repeating itself.
    cycle = int(plan.get("_variation_cycle") or 0)
    temperature = min(0.80, base_temp + cycle * 0.08)
    text = await _complete(
        provider,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt_from_hypothesis(plan, intent)},
        ],
        temperature=temperature,
        max_tokens=800,
    )
    parsed = _extract_json(text)
    # Unwrap if the model returned an array or non-standard wrapper instead of a bare object.
    if not isinstance(parsed, dict) or not str(parsed.get("expression") or "").strip():
        items = _extract_alpha_items(parsed)
        if not items or not isinstance(items[0], dict) or not str(items[0].get("expression") or "").strip():
            return None
        parsed = items[0]
    parsed.setdefault("name", plan.get("name"))
    parsed.setdefault("hypothesis", plan.get("hypothesis"))
    parsed.setdefault("archetype", plan.get("archetype", archetype))
    alpha = _coerce_alpha(parsed, index=index, requested_archetype=archetype)
    alpha["plan"] = {
        "mechanism_steps": plan.get("mechanism_steps", []),
        "operators": plan.get("operators", []),
        "fields": plan.get("fields", []),
        "_from_hypothesis": True,
        "_variation": plan.get("_extra_instruction", ""),
    }
    return alpha


def _check_hypothesis_alignment(alpha: dict[str, Any], hypothesis: dict[str, Any]) -> dict[str, Any]:
    """Heuristic alignment check: does the expression use any of the suggested fields?

    Non-fatal — just flags misaligned alphas so the user/repair step can notice.
    An LLM-evaluated check would be more thorough but adds a full round-trip per alpha;
    the field-presence heuristic catches the worst failures (wrong field family) cheaply.
    """
    expression = alpha.get("expression", "")
    suggested = [str(f) for f in (hypothesis.get("fields_suggested") or []) if f]
    if not suggested or not expression:
        return alpha

    used = [f for f in suggested if re.search(re.escape(f), expression)]
    if used:
        alpha["hypothesis_aligned"] = True
        alpha["alignment_note"] = f"Uses {len(used)}/{len(suggested)} suggested field(s): {', '.join(used[:3])}."
    else:
        alpha["hypothesis_aligned"] = False
        alpha["alignment_note"] = (
            f"Expression uses none of the suggested fields ({', '.join(suggested[:3])}). "
            "Economic mechanism may not be faithfully implemented — consider repair."
        )
    return alpha


async def _generate_alphas_from_hypothesis(
    intent: str,
    archetype: str,
    safe_count: int,
    brief: str,
    provider: str,
    hypothesis_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Hypothesis-guided generation path.

    Bypasses the free-form planner entirely. The structured hypothesis metadata —
    fields, mechanism, archetype, turnover expectation — becomes the plan directly.
    Each of `safe_count` variations emphasizes a different aspect of the mechanism
    (base / simplified / with-filter / longer-windows / group-neutralized).
    """
    hyp_archetype = str(hypothesis_data.get("archetype") or "novel").lower()
    effective_archetype = (
        hyp_archetype if hyp_archetype != "novel"
        else (archetype if archetype not in ("any", "") else "novel")
    )

    plans = [_hypothesis_to_plan(hypothesis_data, i, brief) for i in range(safe_count)]
    for plan in plans:
        plan["archetype"] = effective_archetype

    effective_intent = (
        (intent or "").strip()
        or str(hypothesis_data.get("claim") or "")
        or "implement the hypothesis"
    )

    build_sem = asyncio.Semaphore(_BUILD_CONCURRENCY)

    async def _build_hyp_one(plan: dict[str, Any], idx: int) -> dict[str, Any] | None:
        async with build_sem:
            return await _build_alpha_from_hypothesis(plan, idx, effective_intent, effective_archetype, provider)

    built = await asyncio.gather(
        *(_build_hyp_one(plan, idx) for idx, plan in enumerate(plans)),
        return_exceptions=True,
    )
    candidates = [a for a in built if isinstance(a, dict)]
    if not candidates:
        raise OpenRouterServiceError("hypothesis-guided builder produced no parseable expressions.")

    async def _reground_hyp_one(a: dict[str, Any]) -> dict[str, Any]:
        async with build_sem:
            return await _reground_operators(a, provider)

    regrounded = await asyncio.gather(
        *(_reground_hyp_one(a) for a in candidates),
        return_exceptions=True,
    )

    alphas: list[dict[str, Any]] = []
    seen: set[str] = set()
    for alpha in regrounded:
        if not isinstance(alpha, dict):
            continue
        expr_key = re.sub(r"\s+", "", alpha.get("expression", "")).lower()
        if not expr_key or expr_key in seen:
            continue
        seen.add(expr_key)
        alpha["id"] = f"a{len(alphas)}"
        alpha = _check_hypothesis_alignment(alpha, hypothesis_data)
        alphas.append(alpha)
        if len(alphas) >= safe_count:
            break

    if not alphas:
        raise OpenRouterServiceError("hypothesis-guided generation produced no parseable alpha expressions.")
    return alphas


RESEARCH_SYSTEM = (
    "You are a senior quantitative equity researcher. Ground yourself in current academic and "
    "practitioner sources using web search, then write a tight, source-named research brief. "
    "Do not write code."
)


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:5173"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "AlphaForge"),
    }


async def _post_chat(
    messages: list[dict[str, Any]],
    api_key: str,
    *,
    model: str | None = None,
    plugins: list[dict[str, Any]] | None = None,
    temperature: float = 0.65,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """POST a chat completion and return the assistant message dict.

    The returned message carries ``content`` and, when the web plugin ran,
    an ``annotations`` array of url citations.
    """
    chosen = (model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)).strip() or DEFAULT_MODEL
    payload: dict[str, Any] = {
        "model": chosen,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if plugins:
        payload["plugins"] = plugins
    timeout = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(OPENROUTER_URL, headers=_headers(api_key), json=payload)
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise OpenRouterServiceError(f"OpenRouter returned non-JSON response ({response.status_code}).") from exc
    if response.status_code >= 400:
        err = data.get("error")
        message = err.get("message") if isinstance(err, dict) else err
        raise OpenRouterServiceError(str(message or f"OpenRouter API error ({response.status_code})"))
    choices = data.get("choices") or []
    if not choices:
        raise OpenRouterServiceError("OpenRouter returned an empty choices array.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise OpenRouterServiceError("OpenRouter returned a malformed message.")
    usage = data.get("usage") or {}
    _record_usage(int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))
    return message


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [block["text"] for block in content if isinstance(block, dict) and isinstance(block.get("text"), str)]
        return "\n".join(parts).strip()
    return ""


def _annotations_to_sources(message: dict[str, Any]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for ann in message.get("annotations") or []:
        if not isinstance(ann, dict):
            continue
        citation = ann.get("url_citation") if isinstance(ann.get("url_citation"), dict) else {}
        url = str(citation.get("url") or ann.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = str(citation.get("title") or url).strip()
        sources.append({"title": title[:160], "url": url})
    return sources[:10]


# ----------------------------------------------------------------- providers

PROVIDERS = ("openrouter", "claude_code", "anthropic")


def _claude_bin() -> str:
    return os.getenv("CLAUDE_CODE_BIN", "claude").strip() or "claude"


def available_providers() -> list[dict[str, Any]]:
    """Report which engines are usable so the UI can enable/disable options."""
    claude_bin = _claude_bin()
    return [
        {
            "id": "openrouter",
            "label": "OpenRouter (owl-alpha)",
            "available": bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
            "reason": "set OPENROUTER_API_KEY in the backend .env",
        },
        {
            "id": "claude_code",
            "label": "Claude Code",
            "available": bool(shutil.which(claude_bin)),
            "reason": f"the '{claude_bin}' CLI is not on the backend's PATH",
        },
        {
            "id": "anthropic",
            "label": "Anthropic API",
            "available": bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
            "reason": "set ANTHROPIC_API_KEY in the backend .env",
        },
    ]


def _normalize_provider(provider: str | None) -> str:
    p = (provider or "openrouter").strip().lower().replace("-", "_")
    return p if p in PROVIDERS else "openrouter"


async def _complete(
    provider: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.65,
    max_tokens: int | None = None,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    thinking_budget: int | None = None,
) -> str:
    """Provider-agnostic chat completion. Returns the assistant text.

    thinking_budget: when set and provider=="anthropic", enables Claude's
    extended thinking with that many budget tokens. Requires temperature=1.
    """
    chosen = _normalize_provider(provider)
    if chosen == "anthropic":
        return await _anthropic_complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            thinking_budget=thinking_budget,
        )
    if chosen == "claude_code":
        return await _claude_code_complete(messages, allowed_tools=allowed_tools, model=model)
    # default: openrouter
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise OpenRouterServiceError("OPENROUTER_API_KEY is not configured in the backend environment.")
    message = await _post_chat(messages, api_key, model=model, temperature=temperature, max_tokens=max_tokens)
    text = _message_text(message)
    if not text:
        raise OpenRouterServiceError("OpenRouter returned an empty message.")
    return text


async def _anthropic_complete(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.65,
    max_tokens: int | None = None,
    model: str | None = None,
    thinking_budget: int | None = None,
) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise OpenRouterServiceError("ANTHROPIC_API_KEY is not configured in the backend environment.")
    system = "\n\n".join(m.get("content", "") for m in messages if m.get("role") == "system").strip()
    convo = [
        {"role": m["role"], "content": m.get("content", "")}
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]
    chosen = (model or os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)).strip() or DEFAULT_ANTHROPIC_MODEL

    # Extended thinking requires temperature=1 and a budget_tokens parameter.
    use_thinking = bool(thinking_budget and thinking_budget > 0)
    effective_temp = 1 if use_thinking else temperature
    effective_max = max(max_tokens or 1500, (thinking_budget or 0) + 1500) if use_thinking else (max_tokens or 1500)

    body: dict[str, Any] = {
        "model": chosen,
        "max_tokens": effective_max,
        "temperature": effective_temp,
        "messages": convo or [{"role": "user", "content": ""}],
    }
    if system:
        body["system"] = system
    if use_thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

    headers: dict[str, str] = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if use_thinking:
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

    timeout = httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(ANTHROPIC_URL, headers=headers, json=body)
    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        raise OpenRouterServiceError(f"Anthropic returned non-JSON response ({response.status_code}).") from exc
    if response.status_code >= 400:
        err = data.get("error")
        message = err.get("message") if isinstance(err, dict) else err
        raise OpenRouterServiceError(str(message or f"Anthropic API error ({response.status_code})"))
    blocks = data.get("content") or []
    # Extended thinking returns both "thinking" blocks and "text" blocks — we
    # only want the text blocks for JSON parsing downstream.
    text = "\n".join(
        b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
    ).strip()
    if not text:
        raise OpenRouterServiceError("Anthropic returned an empty message.")
    usage = data.get("usage") or {}
    _record_usage(int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))
    return text


async def _claude_code_complete(
    messages: list[dict[str, Any]],
    *,
    allowed_tools: list[str] | None = None,
    model: str | None = None,
) -> str:
    """Run the local Claude Code CLI headless: `claude -p --output-format json`.

    Uses the user's existing Claude Code auth (no API key). The prompt is fed via
    stdin. For research we pass --allowedTools WebSearch so Claude searches the
    web itself; generation/repair pass no tools (pure text).
    """
    binary = _claude_bin()
    if not shutil.which(binary):
        raise OpenRouterServiceError(
            f"Claude Code CLI '{binary}' was not found on PATH. Install/login to Claude Code or pick another engine."
        )
    args = [binary, "-p", "--output-format", "json"]
    if allowed_tools:
        args += ["--allowedTools", ",".join(allowed_tools)]
    chosen_model = (model or os.getenv("CLAUDE_CODE_MODEL", "")).strip()
    if chosen_model:
        args += ["--model", chosen_model]
    try:
        timeout = float(os.getenv("CLAUDE_CODE_TIMEOUT", "180"))
    except ValueError:
        timeout = 180.0

    prompt = _messages_to_prompt(messages)
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise OpenRouterServiceError(f"Claude Code CLI '{binary}' could not be launched: {exc}") from exc
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=timeout)
    except asyncio.TimeoutError as exc:
        proc.kill()
        raise OpenRouterServiceError(f"Claude Code CLI timed out after {timeout:.0f}s.") from exc
    if proc.returncode != 0:
        detail = (stderr.decode(errors="replace") or "").strip()[:300]
        raise OpenRouterServiceError(f"Claude Code CLI failed (exit {proc.returncode}): {detail}")

    raw = (stdout.decode(errors="replace") or "").strip()
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        envelope = None
    if isinstance(envelope, dict):
        if envelope.get("is_error"):
            raise OpenRouterServiceError(f"Claude Code error: {str(envelope.get('result'))[:300]}")
        # Claude Code JSON envelope carries usage stats.
        usage = envelope.get("usage") or envelope.get("token_usage") or {}
        _record_usage(
            int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
            int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
        )
        result = envelope.get("result")
        if isinstance(result, str) and result.strip():
            return result.strip()
    if raw:
        return raw
    raise OpenRouterServiceError("Claude Code returned empty output.")


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        content = m.get("content", "")
        if m.get("role") == "system":
            parts.append(f"[System instructions]\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(p for p in parts if p)


_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def _urls_from_text(text: str) -> list[dict[str, str]]:
    """Best-effort source extraction from a model-written brief (markdown links + bare URLs)."""
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for title, url in _MD_LINK_RE.findall(text or ""):
        clean = url.rstrip(".,);")
        if clean not in seen:
            seen.add(clean)
            sources.append({"title": title.strip()[:160] or clean, "url": clean})
    for url in _URL_RE.findall(text or ""):
        clean = url.rstrip(".,);")
        if clean not in seen:
            seen.add(clean)
            sources.append({"title": clean[:160], "url": clean})
    return sources[:10]


async def research(intent: str, archetype: str, provider: str = "openrouter") -> dict[str, Any]:
    """Run a web-grounded research pass and return ``{"brief", "sources"}``.

    - provider == "claude_code": Claude Code does the web search natively
      (reliable, free) and writes the brief in one shot.
    - otherwise: a FREE search provider (Tavily / Brave / DuckDuckGo) fetches
      results and the chosen model synthesizes the brief — no paid web plugin.

    Best-effort: callers treat failures as non-fatal and fall back to model knowledge.
    """
    chosen = _normalize_provider(provider)
    raw_intent = (intent or "").strip()
    clean_intent = raw_intent or (
        "a specific, tractable cross-sectional US equity signal currently discussed in the literature"
    )
    clean_arch = (archetype or "any").strip()
    arch_line = "researcher's choice" if clean_arch in ("", "any") else clean_arch

    if chosen == "claude_code":
        return await _claude_code_research(clean_intent, arch_line)

    if raw_intent:
        query = f"{raw_intent} cross-sectional equity alpha factor"
    elif arch_line != "researcher's choice":
        query = f"{arch_line} cross-sectional equity alpha factor stock returns"
    else:
        query = "cross-sectional equity alpha factor predicting stock returns"

    try:
        max_results = int(os.getenv("OPENROUTER_WEB_MAX_RESULTS", "5"))
    except ValueError:
        max_results = 5

    results = await web_search(query, max_results)
    if not results:
        raise OpenRouterServiceError(
            f"web search ({search_provider()}) returned no results — add a free TAVILY_API_KEY/BRAVE_API_KEY "
            "or set the research engine to Claude Code"
        )

    sources = [{"title": r["title"], "url": r["url"]} for r in results]
    context = "\n\n".join(
        f"[{i + 1}] {r['title']}\n{r['url']}\n{r.get('content', '')}".strip()
        for i, r in enumerate(results)
    )
    prompt = f"""Using ONLY the web search results below, write a SHORT research brief (under 180 words) for building a cross-sectional equity alpha.

Topic: {clean_intent}
Target archetype: {arch_line}

Web search results:
{context}

Cover: (1) the economic mechanism — what predicts forward returns and why; (2) which observable daily per-stock quantities express it; (3) regime sensitivities and known failure modes; (4) turnover character — does the data update fast (daily) or slow (quarterly / options surface)? Cite sources inline as [1], [2], etc. Do not write code. Be specific and concrete."""

    research_model = os.getenv("OPENROUTER_RESEARCH_MODEL", "").strip() or None
    brief = await _complete(
        chosen,
        [
            {"role": "system", "content": RESEARCH_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        model=research_model,
        temperature=0.4,
        max_tokens=900,
    )
    if not brief:
        raise OpenRouterServiceError("model returned an empty research brief.")
    return {"brief": brief, "sources": sources}


async def _claude_code_research(clean_intent: str, arch_line: str) -> dict[str, Any]:
    """Research via Claude Code's native WebSearch tool (reliable + free)."""
    prompt = f"""Search the web for current academic and practitioner thinking, then write a SHORT research brief (under 180 words) for building a cross-sectional equity alpha.

Topic: {clean_intent}
Target archetype: {arch_line}

Cover: (1) the economic mechanism — what predicts forward returns and why; (2) which observable daily per-stock quantities express it; (3) regime sensitivities and known failure modes; (4) turnover character — does the data update fast (daily) or slow (quarterly / options surface)? Cite the sources you used as inline markdown links. Do not write code. Be specific and concrete."""
    brief = await _complete(
        "claude_code",
        [
            {"role": "system", "content": RESEARCH_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        allowed_tools=["WebSearch"],
        temperature=0.4,
        max_tokens=1200,
    )
    if not brief:
        raise OpenRouterServiceError("Claude Code returned an empty research brief.")
    return {"brief": brief, "sources": _urls_from_text(brief)}


async def repair_alpha(
    expression: str,
    issues: list[dict[str, Any]] | None = None,
    intent: str = "",
    archetype: str = "",
    sim_feedback: str = "",
    provider: str = "openrouter",
) -> dict[str, str]:
    """Ask the chosen engine to repair a FastExpr expression.

    Repairs from the deterministic validator issues and, when provided, from
    REAL BRAIN simulation feedback (sim_feedback) — the measured metrics and the
    specific submission gates / checks the alpha missed.
    """
    base_expr = (expression or "").strip()
    if not base_expr:
        raise OpenRouterServiceError("Cannot repair an empty expression.")

    issue_lines = "\n".join(
        f"- [{i.get('verdict', 'WARN')}] {i.get('message', '')}"
        + (f" -> {i.get('fix')}" if i.get("fix") else "")
        for i in (issues or [])
        if isinstance(i, dict)
    ) or "- (no machine-readable issues)"

    context = ""
    if (intent or "").strip():
        context += f"\nOriginal research intent: {intent.strip()}"
    if (archetype or "").strip() and archetype.strip() != "any":
        context += f"\nTarget archetype: {archetype.strip()}"

    feedback_block = ""
    if (sim_feedback or "").strip():
        feedback_block = (
            "\n\nThe alpha DID simulate on WorldQuant BRAIN but is NOT submittable. "
            "Real simulation feedback (fix these specifically):\n" + sim_feedback.strip()
        )

    prompt = f"""{_grammar()}

You previously wrote this FastExpr alpha:
{base_expr}

A deterministic pre-screen flagged these issues:
{issue_lines}{feedback_block}{context}

Fix ALL the issues while preserving the economic idea. FITNESS = Sharpe * sqrt(|Returns|/max(Turnover,0.125)) must reach >= 1.0. If fitness is low, the most likely cause is HIGH TURNOVER — add smoothing (increase ts_mean window, use ts_decay_linear), switch to a slower field (fundamentals, 60-day windows), or add rank/zscore normalization. If turnover, fitness, sharpe or correlation are the issue: reduce turnover to 5-35% range first. Return ONLY this JSON object, no markdown and no prose:
{{"expression": "<corrected FastExpr>", "change_note": "<one sentence on what you changed>"}}"""

    text = await _complete(
        provider,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    parsed = _extract_json(text)
    if not isinstance(parsed, dict) or not str(parsed.get("expression") or "").strip():
        raise OpenRouterServiceError("the model did not return a corrected expression.")
    return {
        "expression": str(parsed["expression"]).strip(),
        "change_note": str(parsed.get("change_note") or "").strip(),
    }


def _build_generation_prompt(intent: str, archetype: str, count: int, brief: str = "") -> str:
    clean_intent = (intent or "").strip() or "Find a robust, tractable cross-sectional USA equity alpha."
    clean_arch = (archetype or "any").strip()
    arch_line = "researcher's choice" if clean_arch == "any" else clean_arch
    brief_block = (
        f"\nResearch brief (ground your alphas in this):\n{brief.strip()}\n"
        if (brief or "").strip()
        else ""
    )
    return f"""{_grammar()}

Research intent:
{clean_intent}
{brief_block}
Target archetype: {arch_line}
Candidate count: {count}

Generate {count} mechanistically distinct alphas. Favor expressions that are likely to pass deterministic syntax validation and survive real Brain simulation.

Return exactly this JSON shape:
{{
  "alphas": [
    {{
      "name": "<five words or fewer>",
      "archetype": "reversal|microstructure|volatility|fundamental|analyst_revision|earnings_event|options_implied|factor_residual|dispersion|novel",
      "hypothesis": "<one or two sentences>",
      "expression": "<valid FastExpr>",
      "settings": {{
        "universe": "TOP3000",
        "neutralization": "Market|Sector|Industry|Subindustry",
        "decay": 6,
        "truncation": 0.05,
        "delay": 1
      }},
      "expected_sharpe": "x.x-y.y",
      "expected_fitness": "x.x",
      "regime_note": "<one sentence>",
      "turnover_note": "<target turnover %: 5-35% for good fitness>"
    }}
  ]
}}"""


_CTRL_ESCAPE: dict[str, str] = {
    '\n': '\\n', '\r': '\\r', '\t': '\\t', '\b': '\\b', '\f': '\\f',
}


def _sanitize_control_chars(text: str) -> str:
    """Escape raw ASCII control characters that appear inside JSON string literals.

    LLMs sometimes emit literal newlines/tabs inside quoted values; json.loads
    rejects them (RFC 8259 §7). This pass replaces them with their proper
    JSON escape sequences without touching already-escaped sequences.
    """
    buf: list[str] = []
    in_str = False
    esc = False
    for ch in text:
        if esc:
            esc = False
            buf.append(ch)
            continue
        if ch == '\\':
            esc = True
            buf.append(ch)
            continue
        if ch == '"':
            in_str = not in_str
            buf.append(ch)
            continue
        if in_str and ord(ch) < 0x20:
            buf.append(_CTRL_ESCAPE.get(ch, f'\\u{ord(ch):04x}'))
            continue
        buf.append(ch)
    return ''.join(buf)


def _try_parse_json(candidate: str) -> Any:
    """Parse a JSON string with three escalating repair strategies.

    Returns the parsed value, or None if all attempts fail (never raises).
    """
    # Attempt 1: direct parse.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Attempt 2: strip trailing commas (e.g. `[1,2,]`).
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
    except json.JSONDecodeError:
        pass
    # Attempt 3: sanitize raw control characters inside strings, then strip trailing commas.
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", _sanitize_control_chars(candidate)))
    except json.JSONDecodeError:
        pass
    return None


def _extract_balanced(text: str, start: int, opener: str, closer: str) -> str | None:
    """Return the substring of `text` from `start` to the matching `closer`.

    Returns None when the string is truncated (never finds depth==0).
    """
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _extract_json(text: str) -> Any:
    """Extract and parse the first valid JSON object/array from model output.

    Strips markdown code fences, then scans every `{` and `[` position in
    order until one produces parseable JSON.  This handles:
      - preamble text containing stray `{` before the real JSON
      - raw control characters inside string values
      - trailing commas
      - missing/mismatched code-fence markers
    """
    cleaned = (
        text
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )
    # Scan every potential JSON start character.  Objects are tried before
    # arrays at each position so `{"alphas":[…]}` beats a stray `[` that
    # appears earlier in preamble text.
    for i, ch in enumerate(cleaned):
        if ch not in ('{', '['):
            continue
        closer = '}' if ch == '{' else ']'
        candidate = _extract_balanced(cleaned, i, ch, closer)
        if candidate is None:
            continue  # Truncated — skip this position.
        result = _try_parse_json(candidate)
        if result is not None:
            return result
    return None


def _extract_alpha_items(parsed: Any) -> list[Any]:
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return []
    # Standard wrapper key
    if isinstance(parsed.get("alphas"), list):
        return parsed["alphas"]
    # Single alpha returned directly
    if parsed.get("expression"):
        return [parsed]
    # Fallback: any list-valued key whose items carry "expression" (e.g. the model used
    # "plans", "data", "results", or some other wrapper instead of "alphas").
    for v in parsed.values():
        if isinstance(v, list) and v and any(isinstance(i, dict) and i.get("expression") for i in v):
            return [i for i in v if isinstance(i, dict)]
    return []


def _coerce_alpha(item: Any, index: int, requested_archetype: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        item = {}
    expression = str(item.get("expression") or "").strip()
    settings = normalize_settings(item.get("settings") if isinstance(item.get("settings"), dict) else {})
    archetype = str(item.get("archetype") or requested_archetype or "novel").strip() or "novel"
    if archetype == "any":
        archetype = "novel"
    return {
        "id": f"a{index}",
        "name": str(item.get("name") or f"alpha {index + 1}").strip()[:80],
        "archetype": archetype,
        "hypothesis": str(item.get("hypothesis") or "").strip(),
        "expression": expression,
        "settings": settings,
        "expected_sharpe": str(item.get("expected_sharpe") or "").strip(),
        "regime_note": str(item.get("regime_note") or "").strip(),
        "turnover_note": str(item.get("turnover_note") or "").strip(),
    }

