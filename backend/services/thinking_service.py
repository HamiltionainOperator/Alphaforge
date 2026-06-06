"""
thinking_service.py — Multi-agent, iterative deep-think pipeline for AlphaForge.

Architecture (Deep Think mode):

  [Plan]
     │
     ├─► [Critic: economic]    ─┐
     ├─► [Critic: operators]    ├─► [Synthesis] ─► [Refine] ─► score < threshold?
     ├─► [Critic: fitness]      │                              └─► loop back (max N rounds)
     └─► [Critic: novelty]    ──┘
                                                  [Build expression]
                                                         │
                                              [Expression Verifier]
                                                         │
                                              (fix if needed, max 2 passes)

Each plan goes through all stages independently — true pipeline parallelism across plans.

Configuration (all optional .env overrides):
  ALPHAFORGE_THINK_ROUNDS   — max critique/refine iterations per plan (default 2)
  ALPHAFORGE_THINK_BUDGET   — Anthropic extended thinking budget tokens (default 10000)
  ALPHAFORGE_THINK_ENSEMBLE — generate N plans per slot, keep the best-scored one (default 1)
  ALPHAFORGE_THINK_CONCURRENCY — max parallel LLM calls in the think pipeline (default 12)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_THINK_ROUNDS = int(os.getenv("ALPHAFORGE_THINK_ROUNDS", "2"))
_THINK_BUDGET = int(os.getenv("ALPHAFORGE_THINK_BUDGET", "10000"))
_THINK_ENSEMBLE = max(1, int(os.getenv("ALPHAFORGE_THINK_ENSEMBLE", "1")))


def _think_concurrency() -> int:
    """Read concurrency limit at call time so .env changes take effect without restart.

    Claude Code spawns one subprocess per call — keep this at 4 to avoid hammering
    the CLI. OpenRouter/Anthropic are HTTP and can safely go to 12+.
    """
    return int(os.getenv("ALPHAFORGE_THINK_CONCURRENCY", "4"))

# ─────────────────────────────────────────────────── specialist critic specs

_CRITIC_SPECS: list[dict[str, str]] = [
    {
        "name": "economic",
        "role": "You are a macro economist and behavioral finance expert reviewing a quantitative signal plan.",
        "mission": (
            "Evaluate the ECONOMIC MECHANISM only. Ask: Is there a genuine, academically-grounded "
            "risk premium or behavioral inefficiency that justifies this signal predicting forward "
            "cross-sectional returns? Is there a plausible decay timeline (how long before arbitrage "
            "erodes the edge)? Is there hidden lookahead bias in any step? Could this be a regime-specific "
            "artefact rather than a persistent factor? Rate 1-5 (5=strong, persistent mechanism)."
        ),
    },
    {
        "name": "operators",
        "role": "You are a WorldQuant Brain platform engineer who knows every operator and its exact signature.",
        "mission": (
            "Evaluate OPERATOR CORRECTNESS only. Check: Are every operator name in the catalog? "
            "Do the signatures match (e.g., group functions take exactly 2-3 args, ts_ functions need "
            "an integer window)? Are any banned operators referenced (ts_max, ts_min, ts_skewness, "
            "winsorize, decay_linear without ts_ prefix)? Are arithmetic symbols used instead of "
            "add()/subtract()? Flag every specific issue with the exact fix. Rate 1-5 (5=all operators valid)."
        ),
    },
    {
        "name": "fitness",
        "role": "You are a portfolio construction specialist who optimises for the Brain fitness formula.",
        "mission": (
            "Evaluate TURNOVER AND FITNESS only. Fitness = Sharpe * sqrt(|Returns|/max(Turnover, 0.125)). "
            "Target: Fitness >= 1.0, Sharpe >= 1.25, Turnover 5-35%. "
            "Given the mechanism steps and fields: Will daily turnover be in range? Short windows (<5 days) "
            "as the sole signal, raw price levels, or raw volume all kill fitness. What is your "
            "turnover estimate and what specific change (extend windows, add decay, use fundamentals, "
            "add rank/zscore) would most improve it? Rate 1-5 (5=confident turnover will be 5-35%)."
        ),
    },
    {
        "name": "novelty",
        "role": "You are a systematic hedge fund PM who knows every known alpha factor family.",
        "mission": (
            "Evaluate NOVELTY AND CROWDING RISK only. Is this signal genuinely differentiated from "
            "the most common factors (momentum, value, quality, low-vol, reversal)? Does it combine "
            "fields in an unusual way? Will it be correlated with ubiquitous factors the judge already "
            "holds? The highest-novelty alphas exploit underused data fields or combine standard fields "
            "through unusual economic logic. Rate 1-5 (5=highly novel, low crowding risk)."
        ),
    },
]

_SYNTHESIS_ROLE = (
    "You are a senior quantitative researcher who has just read four specialist critiques "
    "of a signal plan. Synthesise their findings into a single, prioritised, actionable fix list. "
    "Return only strict JSON."
)

_REFINE_ROLE = (
    "You are a senior quantitative researcher who has read a full critique synthesis of your plan. "
    "Incorporate every actionable fix to produce a stronger plan. Return only strict JSON."
)

_VERIFY_ROLE = (
    "You are a WorldQuant Brain FastExpr syntax checker. Verify the expression strictly and flag "
    "every violation. Return only strict JSON."
)


# ─────────────────────────────────────────────────── prompt builders

def _critic_prompt(spec: dict[str, str], plan: dict[str, Any], grammar: str) -> str:
    steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan.get("mechanism_steps", [])))
    ops = ", ".join(plan.get("operators", [])) or "(none listed)"
    fields = ", ".join(plan.get("fields", [])) or "(not specified)"
    return f"""{grammar}

You are reviewing this alpha plan. Your specialisation: {spec['mission']}

Plan under review:
  Name: {plan.get('name', '?')}
  Archetype: {plan.get('archetype', '?')}
  Hypothesis: {plan.get('hypothesis', '')}
  Mechanism steps:
{steps}
  Planned operators: {ops}
  Input fields: {fields}
  Decay hint: {plan.get('decay_hint', 6)}

Return ONLY this JSON (no markdown):
{{
  "critic": "{spec['name']}",
  "score": <integer 1-5>,
  "verdict": "<one sentence summary>",
  "issues": ["<specific issue 1>", "<specific issue 2>"],
  "fixes": ["<actionable fix 1>", "<actionable fix 2>"],
  "redesign_steps": ["<improved step>"],
  "redesign_operators": ["<valid Brain operator>"],
  "redesign_decay": <integer 4-12>
}}"""


def _synthesis_prompt(plan: dict[str, Any], critiques: list[dict[str, Any]]) -> str:
    critic_blocks = "\n\n".join(
        f"[{c.get('critic', '?').upper()} — score {c.get('score', '?')}/5]\n"
        f"Verdict: {c.get('verdict', '')}\n"
        f"Issues: {json.dumps(c.get('issues', []))}\n"
        f"Fixes: {json.dumps(c.get('fixes', []))}"
        for c in critiques
    )
    all_steps = [s for c in critiques for s in (c.get("redesign_steps") or [])]
    all_ops = [o for c in critiques for o in (c.get("redesign_operators") or [])]
    max_decay = max((c.get("redesign_decay") or 6 for c in critiques), default=6)
    min_score = min((c.get("score") or 3 for c in critiques), default=3)

    return f"""Four specialist critics have reviewed the plan '{plan.get('name', '?')}'. Synthesise their findings.

{critic_blocks}

Your job: produce a unified, non-redundant fix list ordered by impact. Combine overlapping suggestions.
Compute overall_score as the average of the four critic scores (to 1 decimal).
Compute fitness_forecast: your honest estimate of the likely Fitness score (0.0 - 3.0) IF the plan is implemented as-is.
Compute fitness_forecast_after_fix: your estimate of Fitness IF all fixes are applied.

Return ONLY this JSON (no markdown):
{{
  "overall_score": <float>,
  "fitness_forecast": <float>,
  "fitness_forecast_after_fix": <float>,
  "top_issue": "<the single most critical problem>",
  "unified_fixes": ["<highest-impact fix>", "<second fix>", ...],
  "final_steps": {json.dumps(list(dict.fromkeys(all_steps)) or plan.get("mechanism_steps", []))},
  "final_operators": {json.dumps(list(dict.fromkeys(all_ops)) or plan.get("operators", []))},
  "final_decay": {max_decay},
  "worth_refining": <true if fitness_forecast < 1.2 or min_score < 3, else false>
}}"""


def _refine_prompt(plan: dict[str, Any], synthesis: dict[str, Any], grammar: str, round_n: int) -> str:
    return f"""{grammar}

This is refinement round {round_n}. You have a critique synthesis of your alpha plan.
Apply EVERY fix and produce an improved plan that will achieve Fitness >= 1.2.

Original plan: {plan.get('name', '?')} [{plan.get('archetype', '?')}]
Original hypothesis: {plan.get('hypothesis', '')}

Critique synthesis:
  Overall score: {synthesis.get('overall_score', '?')}/5
  Fitness forecast (current): {synthesis.get('fitness_forecast', '?')}
  Fitness forecast (after fix): {synthesis.get('fitness_forecast_after_fix', '?')}
  Top issue: {synthesis.get('top_issue', '')}
  Unified fixes to apply:
{chr(10).join('  - ' + f for f in (synthesis.get('unified_fixes') or []))}

Suggested steps: {json.dumps(synthesis.get('final_steps') or [])}
Suggested operators: {json.dumps(synthesis.get('final_operators') or [])}
Suggested decay: {synthesis.get('final_decay', 6)}

Produce the refined plan. Use ONLY operators from the catalog above.
Ensure turnover target is 5-35%. Keep the core economic hypothesis but fix all technical issues.

Return ONLY this JSON (no markdown):
{{
  "name": "{plan.get('name', 'alpha').replace('"', "'")}",
  "archetype": "{plan.get('archetype', 'novel')}",
  "hypothesis": "<keep or sharpen>",
  "mechanism_steps": ["<refined step 1>", "<refined step 2>", ...],
  "operators": ["<Brain operators only>"],
  "fields": ["<input fields>"],
  "decay_hint": <integer 4-12>
}}"""


def _verify_prompt(expression: str, grammar: str) -> str:
    return f"""{grammar}

Verify this FastExpr expression for WorldQuant Brain. Check EVERY rule:
1. Every operator name is in the catalog (no ts_max, ts_min, ts_skewness, winsorize, decay_linear, ts_ir).
2. Windowed ts_ operators have an integer literal as the second argument.
3. group_ functions have the correct number of arguments.
4. No scalar epsilons added to unitful denominators under VERIFY mode.
5. Arithmetic symbols used (+, -, *, /) not add()/subtract().
6. If if_else/max/min branches: both branches have the same units.
7. Overall: would this expression pass Brain syntax validation?

Expression to verify:
{expression}

Return ONLY this JSON (no markdown):
{{
  "valid": <true|false>,
  "issues": ["<issue 1 with exact fix>", ...],
  "corrected_expression": "<fixed expression, or same as input if valid>"
}}"""


# ─────────────────────────────────────────────────── core think passes

async def _run_critic(
    spec: dict[str, str], plan: dict[str, Any], provider: str, grammar: str, sem: asyncio.Semaphore
) -> dict[str, Any]:
    from backend.services.openrouter_service import _complete, _extract_json, OpenRouterServiceError

    async with sem:
        try:
            text = await _complete(
                provider,
                [
                    {"role": "system", "content": spec["role"]},
                    {"role": "user", "content": _critic_prompt(spec, plan, grammar)},
                ],
                temperature=0.25,
                max_tokens=700,
            )
            parsed = _extract_json(text)
            if isinstance(parsed, dict):
                return parsed
        except (OpenRouterServiceError, Exception) as exc:
            logger.warning("[think] critic '%s' failed: %s", spec["name"], exc)
    return {
        "critic": spec["name"], "score": 3, "verdict": "critique unavailable",
        "issues": [], "fixes": [], "redesign_steps": [], "redesign_operators": [], "redesign_decay": 6,
    }


async def _synthesise_critiques(
    plan: dict[str, Any], critiques: list[dict[str, Any]], provider: str, sem: asyncio.Semaphore
) -> dict[str, Any]:
    from backend.services.openrouter_service import _complete, _extract_json, OpenRouterServiceError

    async with sem:
        try:
            text = await _complete(
                provider,
                [
                    {"role": "system", "content": _SYNTHESIS_ROLE},
                    {"role": "user", "content": _synthesis_prompt(plan, critiques)},
                ],
                temperature=0.2,
                max_tokens=800,
            )
            parsed = _extract_json(text)
            if isinstance(parsed, dict):
                return parsed
        except (OpenRouterServiceError, Exception) as exc:
            logger.warning("[think] synthesis failed: %s", exc)
    return {"overall_score": 3.0, "fitness_forecast": 1.0, "worth_refining": False,
            "unified_fixes": [], "final_steps": plan.get("mechanism_steps", []),
            "final_operators": plan.get("operators", []), "final_decay": plan.get("decay_hint", 6)}


async def _refine_plan_once(
    plan: dict[str, Any], synthesis: dict[str, Any], provider: str, grammar: str,
    round_n: int, sem: asyncio.Semaphore
) -> dict[str, Any]:
    from backend.services.openrouter_service import _complete, _extract_json, _brain_catalog, OpenRouterServiceError

    async with sem:
        try:
            # Use extended thinking for Anthropic provider — this is where the real reasoning happens
            is_anthropic = provider.strip().lower().replace("-", "_") == "anthropic"
            thinking_budget = _THINK_BUDGET if is_anthropic else None

            text = await _complete(
                provider,
                [
                    {"role": "system", "content": _REFINE_ROLE},
                    {"role": "user", "content": _refine_prompt(plan, synthesis, grammar, round_n)},
                ],
                temperature=0.3,
                max_tokens=900,
                thinking_budget=thinking_budget,
            )
            parsed = _extract_json(text)
            if not isinstance(parsed, dict):
                return plan

            catalog = _brain_catalog()
            refined = dict(plan)
            if isinstance(parsed.get("mechanism_steps"), list) and parsed["mechanism_steps"]:
                refined["mechanism_steps"] = parsed["mechanism_steps"]
            if isinstance(parsed.get("operators"), list):
                raw_ops = [str(o).strip() for o in parsed["operators"]]
                refined["operators"] = [o for o in raw_ops if o in catalog["valid"]]
                refined["_rejected_operators"] = [o for o in raw_ops if o not in catalog["valid"]]
            if isinstance(parsed.get("fields"), list) and parsed["fields"]:
                refined["fields"] = parsed["fields"]
            if isinstance(parsed.get("decay_hint"), (int, float)):
                refined["decay_hint"] = max(4, min(12, int(parsed["decay_hint"])))
            if isinstance(parsed.get("hypothesis"), str) and parsed["hypothesis"].strip():
                refined["hypothesis"] = parsed["hypothesis"].strip()

            refined["_think_meta"] = refined.get("_think_meta", {})
            refined["_think_meta"][f"round_{round_n}"] = {
                "fitness_forecast_before": synthesis.get("fitness_forecast"),
                "fitness_forecast_after": synthesis.get("fitness_forecast_after_fix"),
                "top_issue": synthesis.get("top_issue", ""),
                "fixes_applied": synthesis.get("unified_fixes", []),
            }
            return refined
        except (OpenRouterServiceError, Exception) as exc:
            logger.warning("[think] refine round %d failed: %s", round_n, exc)
            return plan


async def _verify_expression(
    expression: str, provider: str, grammar: str, sem: asyncio.Semaphore
) -> dict[str, Any]:
    from backend.services.openrouter_service import _complete, _extract_json, OpenRouterServiceError

    async with sem:
        try:
            text = await _complete(
                provider,
                [
                    {"role": "system", "content": _VERIFY_ROLE},
                    {"role": "user", "content": _verify_prompt(expression, grammar)},
                ],
                temperature=0.1,
                max_tokens=600,
            )
            parsed = _extract_json(text)
            if isinstance(parsed, dict):
                return parsed
        except (OpenRouterServiceError, Exception) as exc:
            logger.warning("[think] expression verify failed: %s", exc)
    return {"valid": True, "issues": [], "corrected_expression": expression}


# ─────────────────────────────────────────────────── main pipeline

async def _deep_think_one_plan(
    plan: dict[str, Any],
    provider: str,
    grammar: str,
    sem: asyncio.Semaphore,
    max_rounds: int,
) -> dict[str, Any]:
    """Full multi-round, multi-specialist think pipeline for a single plan."""
    current_plan = plan

    for round_n in range(1, max_rounds + 1):
        # 1. Run all 4 specialist critics in parallel
        critiques = await asyncio.gather(
            *(_run_critic(spec, current_plan, provider, grammar, sem) for spec in _CRITIC_SPECS)
        )
        critiques = [c for c in critiques if isinstance(c, dict)]

        # 2. Synthesise critiques
        synthesis = await _synthesise_critiques(current_plan, critiques, provider, sem)

        # Store the first-round forecast for observability
        if round_n == 1:
            current_plan["_initial_fitness_forecast"] = synthesis.get("fitness_forecast")
            current_plan["_all_critiques"] = critiques

        # 3. Skip refinement if the plan is already good enough
        fitness_forecast = synthesis.get("fitness_forecast", 1.0)
        worth_refining = synthesis.get("worth_refining", fitness_forecast < 1.2)
        if not worth_refining:
            logger.info("[think] plan '%s' passed at round %d (forecast=%.2f)",
                        current_plan.get("name"), round_n, fitness_forecast)
            break

        # 4. Refine the plan
        current_plan = await _refine_plan_once(
            current_plan, synthesis, provider, grammar, round_n, sem
        )
        logger.info("[think] plan '%s' refined (round %d, was forecast=%.2f)",
                    current_plan.get("name"), round_n, fitness_forecast)

    current_plan["_deep_think"] = True
    return current_plan


async def deep_think_plans(
    plans: list[dict[str, Any]],
    provider: str,
) -> list[dict[str, Any]]:
    """Run the full deep-think pipeline over all plans concurrently.

    Each plan is processed independently (plan → 4 critics → synthesis → refine → repeat).
    Returns refined plans in the same order.
    """
    from backend.services.openrouter_service import _grammar as get_grammar

    grammar_text = get_grammar()
    sem = asyncio.Semaphore(_think_concurrency())
    max_rounds = _THINK_ROUNDS

    results = await asyncio.gather(
        *(_deep_think_one_plan(plan, provider, grammar_text, sem, max_rounds) for plan in plans),
        return_exceptions=True,
    )
    return [
        r if isinstance(r, dict) else plans[i]
        for i, r in enumerate(results)
    ]


async def verify_built_alphas(
    alphas: list[dict[str, Any]],
    provider: str,
) -> list[dict[str, Any]]:
    """Post-build expression verification pass.

    For each alpha, an LLM checks the expression against the Brain grammar.
    If issues are found, the corrected expression replaces the original.
    Runs concurrently across all alphas.
    """
    from backend.services.openrouter_service import _grammar as get_grammar

    grammar_text = get_grammar()
    sem = asyncio.Semaphore(_think_concurrency())

    async def verify_one(alpha: dict[str, Any]) -> dict[str, Any]:
        expr = alpha.get("expression", "")
        if not expr:
            return alpha
        result = await _verify_expression(expr, provider, grammar_text, sem)
        if not result.get("valid") and result.get("corrected_expression"):
            corrected = result["corrected_expression"].strip()
            if corrected and corrected != expr:
                alpha["expression"] = corrected
                alpha["_verify_fixed"] = True
                alpha["_verify_issues"] = result.get("issues", [])
                logger.info("[think] expression corrected for '%s': %d issues",
                            alpha.get("name", "?"), len(result.get("issues", [])))
        return alpha

    results = await asyncio.gather(
        *(verify_one(alpha) for alpha in alphas),
        return_exceptions=True,
    )
    return [r if isinstance(r, dict) else alphas[i] for i, r in enumerate(results)]


# ─────────────────────────────────────────────────── backward compat shim

async def critique_and_refine_plans(
    plans: list[dict[str, Any]],
    provider: str,
    concurrency: int = 12,
) -> list[dict[str, Any]]:
    """Legacy shim — now delegates to the full multi-specialist pipeline."""
    return await deep_think_plans(plans, provider)
