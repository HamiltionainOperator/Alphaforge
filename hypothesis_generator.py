"""
hypothesis_generator.py — standalone hypothesis ideation stage for AlphaQuant v3.

Runs BEFORE expression generation. Calls an LLM (via the same provider stack as
the rest of AlphaForge) to generate structured hypotheses grounded in Brain KB
rules. No Anthropic API key needed — defaults to the Claude Code CLI provider
which uses your existing Claude subscription.

Provider priority (override with ALPHAFORGE_HYPOTHESIS_PROVIDER env var):
  1. claude_code  — Claude Code CLI (no extra key, uses your subscription)
  2. openrouter   — needs OPENROUTER_API_KEY in .env
  3. anthropic    — needs ANTHROPIC_API_KEY in .env

Usage:
    engine = HypothesisEngine()
    hyps   = engine.generate(archetype="fundamental", count=10)
    novel  = engine.filter_novel(hyps)
    top3   = sorted(novel, key=lambda h: h["novelty_score"], reverse=True)[:3]
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent
_KB_PATH = _REPO / "brain_kb.json"
_MEMORY_PATH = _REPO / "data" / "memory.json"

# Keys that every returned hypothesis dict must contain.
HYPOTHESIS_KEYS = (
    "title",
    "archetype",
    "claim",
    "mechanism",
    "regime_guard",
    "fields_suggested",
    "novelty_score",
    "novelty_reason",
    "expected_turnover_range",
    "expected_sharpe_range",
)


# --------------------------------------------------------------------------- #
# Provider detection                                                           #
# --------------------------------------------------------------------------- #

def _detect_provider() -> str:
    """Pick the best available provider automatically, or use the env override."""
    override = os.getenv("ALPHAFORGE_HYPOTHESIS_PROVIDER", "").strip().lower()
    if override:
        return override
    claude_bin = os.getenv("CLAUDE_CODE_BIN", "claude").strip() or "claude"
    if shutil.which(claude_bin):
        return "claude_code"
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        return "openrouter"
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    # Fall back to claude_code and let _complete raise a clear error.
    return "claude_code"


# --------------------------------------------------------------------------- #
# KB loading helpers — no hardcoded field/archetype names                     #
# --------------------------------------------------------------------------- #

def _load_kb() -> dict[str, Any]:
    try:
        return json.loads(_KB_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("brain_kb.json unreadable (%s) — using empty KB", exc)
        return {}


def _extract_fields_from_expression(expr: str) -> set[str]:
    """Best-effort field token extraction from a FastExpr string."""
    tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", expr or "")
    return {t for t in tokens if not re.match(
        r"^(ts_|group_|rank|zscore|scale|normalize|quantile|signed_power|if_else|"
        r"abs|log|sign|exp|sqrt|pow|ceil|floor|round|less|greater|equal|and|or|not|min|max|"
        r"add|subtract|multiply|divide|sector|industry|subindustry)$", t
    )}


def _confirmed_winner_fields(kb: dict[str, Any]) -> list[set[str]]:
    result: list[set[str]] = []
    for entry in kb.get("confirmed_working", []):
        expr = entry.get("expression", "")
        fields = set(entry.get("fields", [])) | _extract_fields_from_expression(expr)
        if fields:
            result.append(fields)
    return result


def _build_system_prompt(kb: dict[str, Any]) -> str:
    """Build the hypothesis ideation system prompt fully from brain_kb.json."""
    archetypes = list(kb.get("archetypes", {}).keys())
    forbidden_ops = kb.get("forbidden_operators", [])
    forbidden_fields = kb.get("forbidden_fields", [])
    language_primer = kb.get("language_primer", {})
    fitness_formula = kb.get("fitness_formula", {})

    eval_model = language_primer.get("evaluation_model", "")
    core_fields_block = (
        eval_model[:600] + "..."
        if len(eval_model) > 600
        else eval_model
    ) or (
        "Core: close, open, high, low, vwap, volume, returns. "
        "Fundamentals: assets, sales, equity, debt, liabilities, ebit, ebitda, capex, sharesout."
    )

    winners = kb.get("confirmed_working", [])
    winner_lines = "\n".join(
        f"  - [{w.get('archetype', '?')}] {w.get('expression', '')[:80]}"
        for w in winners if w.get("expression")
    )

    fitness_note = ""
    if fitness_formula:
        fitness_note = (
            f"\nFitness formula: {fitness_formula.get('expression', '')}\n"
            f"Targets: {json.dumps(fitness_formula.get('targets', {}))}\n"
            "Low turnover (5-35%) is critical for high fitness."
        )

    archetype_lines = []
    for name, meta in kb.get("archetypes", {}).items():
        if not isinstance(meta, dict):
            continue
        sigs = meta.get("signature", [])
        expected_sh = meta.get("expected_sharpe", [])
        to_rng = meta.get("expected_turnover_pct", [])
        line = f"  - {name}"
        if sigs:
            line += f": typical signals [{', '.join(str(s) for s in sigs[:4])}]"
        if expected_sh:
            line += f"  sharpe≈{expected_sh}"
        if to_rng:
            line += f"  turnover≈{to_rng}%"
        archetype_lines.append(line)

    return f"""You are a senior quantitative researcher specialising in WorldQuant BRAIN alpha ideation.

Your job is HYPOTHESIS GENERATION ONLY — do NOT write FastExpr code or expressions.
Generate cross-sectional US equity signal hypotheses that are:
  • Economically motivated with a clear mechanism for predicting forward returns.
  • Grounded in observable BRAIN data fields (no invented fields).
  • Genuinely novel — not recombinations of the confirmed winners listed below.
  • Cognisant of the fitness formula so the signal design targets low turnover.

=== BRAIN LANGUAGE CONTEXT ===
{core_fields_block}

=== VALID ARCHETYPES ===
{chr(10).join(archetype_lines) or ', '.join(archetypes)}

=== FORBIDDEN OPERATORS (NEVER reference these) ===
{', '.join(str(o) for o in forbidden_ops) or '(none)'}

=== FORBIDDEN FIELDS (NEVER suggest these) ===
{', '.join(str(f) for f in forbidden_fields) or '(none)'}

=== CONFIRMED WINNERS (avoid similar field combinations — push for novelty) ===
{winner_lines or '(none yet)'}
{fitness_note}

=== OUTPUT RULES ===
Return ONLY a JSON object. No markdown, no commentary outside the JSON.
Schema:
{{
  "hypotheses": [
    {{
      "title": "<5 words max>",
      "archetype": "<one of the valid archetypes>",
      "claim": "<one sentence: what predicts what, and direction>",
      "mechanism": "<2-3 sentences: WHY does this predict returns? economic or behavioural reasoning>",
      "regime_guard": "<one sentence: when does this fail or reverse?>",
      "fields_suggested": ["<exact Brain field name>", ...],
      "novelty_score": <integer 1-10, 10=completely novel>,
      "novelty_reason": "<one sentence: what makes this different from the confirmed winners>",
      "expected_turnover_range": "<e.g. '10-25%'>",
      "expected_sharpe_range": "<e.g. '1.2-1.8'>"
    }}
  ]
}}"""


# --------------------------------------------------------------------------- #
# HypothesisEngine                                                             #
# --------------------------------------------------------------------------- #

class HypothesisEngine:
    """Generate and filter signal hypotheses using the project's LLM provider stack.

    No Anthropic API key required — defaults to the Claude Code CLI provider
    which uses your existing Claude subscription.
    """

    def __init__(self, kb_path: Path = _KB_PATH, provider: str | None = None) -> None:
        self._kb = _load_kb()
        self._system_prompt = _build_system_prompt(self._kb)
        self._archetypes = list(self._kb.get("archetypes", {}).keys())
        self._provider = provider or _detect_provider()
        logger.info("[hypothesis] using provider: %s", self._provider)

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def generate(
        self,
        archetype: str | None = None,
        theme: str | None = None,
        count: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate `count` signal hypotheses via the configured LLM provider."""
        safe_count = max(1, min(int(count), 20))
        user_prompt = self._build_user_prompt(archetype, theme, safe_count)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        print(f"[hypothesis] generating {safe_count} hypotheses via {self._provider} "
              f"| archetype={archetype or 'any'} theme={theme or 'any'}")

        # _complete is async; run it synchronously so HypothesisEngine stays
        # usable from plain (non-async) scripts like aq3/run3.py.
        raw = self._run_complete(messages)
        hypotheses = self._parse_response(raw)

        print(f"[hypothesis] received {len(hypotheses)} hypotheses")
        for h in hypotheses:
            print(f"  [hyp] {h.get('title','?'):<35} archetype={h.get('archetype','?'):<20} "
                  f"novelty={h.get('novelty_score',0)}/10  fields={h.get('fields_suggested',[])}")
        return hypotheses

    def filter_novel(
        self,
        hypotheses: list[dict[str, Any]],
        memory_path: Path | str = _MEMORY_PATH,
        overlap_threshold: float = 0.60,
    ) -> list[dict[str, Any]]:
        """Drop hypotheses that overlap too heavily with confirmed winners or use
        forbidden fields."""
        winner_field_sets = _confirmed_winner_fields(self._kb)

        try:
            mem = json.loads(Path(memory_path).read_text())
            mem_entries = (
                mem.get("entries", []) if isinstance(mem, dict)
                else (mem if isinstance(mem, list) else [])
            )
            for entry in mem_entries:
                fs = _extract_fields_from_expression(entry.get("expression", ""))
                if fs:
                    winner_field_sets.append(fs)
        except (OSError, json.JSONDecodeError):
            pass

        forbidden_fields = set(self._kb.get("forbidden_fields", []))
        forbidden_prefixes = tuple(f[:-1] for f in forbidden_fields if f.endswith("*"))
        forbidden_exact = {f for f in forbidden_fields if not f.endswith("*")}

        kept: list[dict[str, Any]] = []
        for h in hypotheses:
            fields = set(h.get("fields_suggested") or [])

            bad = {
                f for f in fields
                if f in forbidden_exact or any(f.startswith(p) for p in forbidden_prefixes)
            }
            if bad:
                print(f"  [filter] DROP '{h.get('title','?')}' — forbidden fields: {bad}")
                continue

            if winner_field_sets and fields:
                max_overlap = max(
                    len(fields & w) / len(fields | w)
                    for w in winner_field_sets if fields | w
                )
                if max_overlap > overlap_threshold:
                    print(f"  [filter] DROP '{h.get('title','?')}' — "
                          f"{max_overlap:.0%} field overlap with a winner")
                    continue

            kept.append(h)

        print(f"[hypothesis] filter_novel: {len(kept)}/{len(hypotheses)} kept "
              f"(threshold={overlap_threshold:.0%})")
        return kept

    # ---------------------------------------------------------------------- #
    # Internals                                                                #
    # ---------------------------------------------------------------------- #

    def _run_complete(self, messages: list[dict[str, Any]]) -> str:
        """Run the async _complete call synchronously.

        Always called from a worker thread (via run_in_threadpool), so there is
        no running event loop in scope — asyncio.run() creates a fresh one.
        """
        from backend.services.openrouter_service import _complete, OpenRouterServiceError
        try:
            return asyncio.run(
                _complete(self._provider, messages, temperature=0.6, max_tokens=4096)
            )
        except OpenRouterServiceError as exc:
            raise RuntimeError(f"[hypothesis] LLM call failed ({self._provider}): {exc}") from exc

    def _build_user_prompt(self, archetype: str | None, theme: str | None, count: int) -> str:
        arch_line = (
            f"Focus on the '{archetype}' archetype."
            if archetype and archetype in self._archetypes
            else "Span a diverse mix of archetypes."
        )
        theme_line = (
            f"Theme / angle to explore: {theme.strip()}"
            if (theme or "").strip()
            else "Choose whichever themes show the most unexploited alpha potential."
        )
        return (
            f"Generate exactly {count} signal hypotheses.\n"
            f"{arch_line}\n"
            f"{theme_line}\n\n"
            "Push for genuine novelty — avoid recombining fields already used in confirmed winners. "
            "Each hypothesis must have a distinct economic mechanism. "
            "Estimate expected_turnover_range honestly; low turnover (5-35%) is key to fitness >= 1.0."
        )

    def _parse_response(self, raw: str) -> list[dict[str, Any]]:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        start = cleaned.find("{")
        if start < 0:
            logger.warning("[hypothesis] no JSON object found in model response")
            return []
        depth, in_str, esc = 0, False, False
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if esc:
                esc = False; continue
            if ch == "\\":
                esc = True; continue
            if ch == '"':
                in_str = not in_str; continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        try:
                            parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
                        except json.JSONDecodeError:
                            logger.warning("[hypothesis] JSON parse failed")
                            return []
                    items = parsed.get("hypotheses") if isinstance(parsed, dict) else parsed
                    if not isinstance(items, list):
                        return []
                    return [self._coerce(h) for h in items if isinstance(h, dict)]
        return []

    def _coerce(self, raw: dict[str, Any]) -> dict[str, Any]:
        archetypes = self._archetypes or ["novel"]
        arch = str(raw.get("archetype") or "novel").strip().lower()
        if arch not in archetypes:
            arch = "novel"
        novelty = raw.get("novelty_score")
        try:
            novelty = max(1, min(10, int(novelty)))
        except (TypeError, ValueError):
            novelty = 5
        fields = raw.get("fields_suggested")
        if not isinstance(fields, list):
            fields = []
        return {
            "title": str(raw.get("title") or "").strip()[:80],
            "archetype": arch,
            "claim": str(raw.get("claim") or "").strip(),
            "mechanism": str(raw.get("mechanism") or "").strip(),
            "regime_guard": str(raw.get("regime_guard") or "").strip(),
            "fields_suggested": [str(f).strip() for f in fields if f],
            "novelty_score": novelty,
            "novelty_reason": str(raw.get("novelty_reason") or "").strip(),
            "expected_turnover_range": str(raw.get("expected_turnover_range") or "").strip(),
            "expected_sharpe_range": str(raw.get("expected_sharpe_range") or "").strip(),
        }
