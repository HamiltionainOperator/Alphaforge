"""Deterministic mutation of near-miss alphas — no LLM in the loop.

A near-miss is an alpha that already has positive edge (Sharpe ~0.8–1.2, or a
strong Sharpe held back by one failing gate) but did not pass. Historically these
were shown to the LLM as text hints and then abandoned. They are too valuable for
that: a Sharpe-1.8 / fitness-0.96 alpha needs a nudge, not a fresh idea.

`NearMissMutator` takes one near-miss (the analyzer's record dict, which carries
the expression, the simulation result, and the settings) and produces up to 5
concrete, rule-based mutations. The rules map directly onto the observed failure
modes:

  turnover > 70%  →  double every window arg AND wrap the fastest leg in
                     ts_decay_linear(..., 8)        (kill churn hard)
  turnover > 60%  →  wrap the fastest leg in ts_decay_linear(..., 6)
  fitness < 1.0 and sharpe > 0  →  tighten neutralization to subindustry (settings-level
                     neutralization is auto-forced to 'None' whenever the expression
                     already self-neutralizes via group_neutralize — no double demean)
  sharpe < 0 (sign wrong)       →  prepend  -1 *  to the whole expression
  single-leg expression         →  multiply in  rank(ebit/assets)  as a quality gate
  contains a 5-day window        →  also try the same expression at 10 and 21

Every candidate is validated through MathEngine().critique() before it is
returned, and duplicates / no-ops are dropped. The surviving alpha dicts are
shaped exactly like generate_batch output, so run.py can drop them straight into
submit_batch — they bypass the LLM and the prompt entirely.
"""
from __future__ import annotations

import re
from typing import Any

from math_engine import FAIL, MathEngine

_GROUP_OPS = ("group_neutralize", "group_rank", "group_zscore")
# Transforms that must never wrap a neutralized signal (doing so re-ranks globally
# and destroys within-group dollar-neutrality). Mirrors the critique() check.
_TRANSFORM_WRAPPERS = (
    "rank", "zscore", "scale", "normalize", "quantile",
    "ts_rank", "ts_zscore", "signed_power", "group_rank", "group_zscore",
)
_MAX_WINDOW = 252  # don't let window-doubling run off to absurd horizons


def _norm(expr: str) -> str:
    return re.sub(r"\s+", "", expr or "")


def _outer_call(expr: str) -> tuple[str, str] | None:
    """If `expr` is a single call wrapping everything — op(inner) — return
    (op, inner). Returns None when there is trailing content after the matching
    close paren (e.g. `rank(a)*rank(b)` is not a single outer call)."""
    e = expr.strip()
    m = re.match(r"^([A-Za-z_][A-Za-z_0-9]*)\s*\(", e)
    if not m:
        return None
    op = m.group(1)
    start = m.end()  # first char inside the '('
    depth, i = 1, start
    while i < len(e) and depth > 0:
        if e[i] == "(":
            depth += 1
        elif e[i] == ")":
            depth -= 1
        i += 1
    if depth != 0 or i != len(e):
        return None  # unbalanced, or stuff after the close paren
    return op, e[start:i - 1]


def _safe_neutralization(expression: str, preferred: str) -> str:
    """Return 'None' if the expression self-neutralizes via group_neutralize,
    else return `preferred`. Prevents double-neutralization at the settings level."""
    if "group_neutralize(" in _norm(expression):
        return "None"
    return preferred


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split `s` on `sep` only at parenthesis depth 0."""
    parts, depth, last = [], 0, 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(s[last:i])
            last = i + 1
    parts.append(s[last:])
    return [p.strip() for p in parts]


def _decompose(expr: str) -> tuple[str | None, str, str | None]:
    """Peel a group_* wrapper off so we can operate on the inner signal.

    Returns (group_op, core_signal, group_name). When there is no group wrapper,
    group_op / group_name are None and core_signal is the whole expression."""
    call = _outer_call(expr)
    if call and call[0] in _GROUP_OPS:
        args = _split_top_level(call[1], ",")
        if len(args) == 2:
            return call[0], args[0].strip(), args[1].strip()
    return None, expr.strip(), None


def _recompose(group_op: str | None, core: str, group_name: str | None) -> str:
    if group_op and group_name:
        return f"{group_op}({core}, {group_name})"
    return core


def _leg_speed(leg: str) -> int:
    """Lower = faster (shorter horizon). Smallest window arg in the leg, with raw
    intraday gaps (close-vwap, close-open, high-low) treated as the fastest (1)."""
    has_intraday = re.search(
        r"(close\s*-\s*vwap|vwap\s*-\s*close|close\s*-\s*open|open\s*-\s*close|high\s*-\s*low)",
        leg,
    )
    # A raw intraday gap churns every day no matter what window smooths its
    # denominator, so it is the fastest leg even when it also carries a window.
    if has_intraday:
        return 1
    windows = [int(x) for x in re.findall(r",\s*(\d+)\s*\)", leg)]
    if windows:
        return min(windows)
    return 9999


def _wrap_fastest_leg(expr: str, decay: int) -> str | None:
    """Wrap the fastest top-level multiplicative leg of the core signal in
    ts_decay_linear(..., decay). Returns None if there is nothing useful to wrap
    (already decayed, or no identifiable fast leg)."""
    group_op, core, group_name = _decompose(expr)
    legs = _split_top_level(core, "*")
    legs = [leg for leg in legs if leg]
    if not legs:
        return None
    fastest_idx = min(range(len(legs)), key=lambda i: _leg_speed(legs[i]))
    target = legs[fastest_idx]
    if target.strip().startswith("ts_decay_linear("):
        return None  # already smoothed — don't double-wrap
    legs[fastest_idx] = f"ts_decay_linear({target}, {decay})"
    return _recompose(group_op, " * ".join(legs), group_name)


def _double_windows(expr: str) -> str:
    """Double every integer that sits as the last argument of a call, capped at
    _MAX_WINDOW. Catches ts_mean(x,5)->ts_mean(x,10), ts_delta(close,2)->(close,4)."""
    return re.sub(
        r",\s*(\d+)\s*\)",
        lambda m: f", {min(_MAX_WINDOW, int(m.group(1)) * 2)})",
        expr,
    )


def _hoist_neutralize(expr: str) -> str | None:
    """If a transform wraps a group_neutralize (e.g. rank(group_neutralize(...))),
    hoist the neutralization to the outside, which is where it belongs. Returns None
    when the expression is already well-formed."""
    call = _outer_call(expr)
    if not call or call[0] not in _TRANSFORM_WRAPPERS:
        return None
    inner_first = _split_top_level(call[1], ",")[0].strip()
    inner_call = _outer_call(inner_first)
    if inner_call and inner_call[0] == "group_neutralize":
        return inner_first  # the group_neutralize(...) becomes the whole expression
    return None


def _add_quality_gate(expr: str) -> str | None:
    """Multiply rank(ebit/assets) into a single-leg expression. For a group-wrapped
    signal the gate goes inside the wrapper so neutralization still applies."""
    gate = "rank(ebit/assets)"
    group_op, core, group_name = _decompose(expr)
    if len(_split_top_level(core, "*")) > 1:
        return None  # not a single leg
    new_core = f"{core} * {gate}"
    return _recompose(group_op, new_core, group_name)


class NearMissMutator:
    """Turns one near-miss into a handful of validated, queue-ready mutations."""

    def __init__(self, engine: MathEngine | None = None) -> None:
        self.engine = engine or MathEngine()

    # ----------------------------------------------------------- validation
    def _structurally_valid(self, expr: str, settings: dict | None) -> bool:
        """A mutation is kept unless critique() hard-FAILs it. All remaining FAIL
        rules are real BRAIN-platform rejections (unknown field/op, arity, unit
        mismatch, double-neutralization, learned bad pairs), so any FAIL means the
        mutation would be wasted at submission. (The old 'saturated_mechanism_family'
        FAIL we used to tolerate here has been removed — it was editorial steering,
        not a platform rule.)"""
        crit = self.engine.critique(expr, settings=settings)
        return crit["verdict"] != FAIL

    # ----------------------------------------------------------- mutation
    def mutate(self, near_miss: dict, max_mutations: int = 5) -> list[dict]:
        """Produce up to `max_mutations` validated mutation alpha-dicts for one
        near-miss record (as emitted by FeedbackAnalyzer)."""
        base_expr = (near_miss.get("expression") or "").strip()
        if not base_expr:
            return []
        sharpe = near_miss.get("sharpe")
        fitness = near_miss.get("fitness")
        turnover = near_miss.get("turnover_pct")
        base_settings = dict(near_miss.get("settings") or {})

        # (mutated_expr, settings_overrides, tag, human_reason)
        candidates: list[tuple[str, dict, str, str]] = []

        def add(expr: str | None, tag: str, reason: str, settings_over: dict | None = None):
            if expr:
                candidates.append((expr, settings_over or {}, tag, reason))

        # --- structural fix: neutralization must be outermost --------------
        # If the base wraps a transform around group_neutralize, correct it FIRST
        # and build every other mutation on the corrected form, so we never emit a
        # new hierarchy violation. The recorded _base_expression still shows the
        # original near-miss.
        fixed_base = _hoist_neutralize(base_expr)
        work_expr = fixed_base or base_expr
        if fixed_base:
            add(
                fixed_base,
                "hoist_neutralize",
                "neutralization was not outermost: hoisted group_neutralize to the outside "
                "to restore subindustry dollar-neutrality",
            )

        # --- turnover band -------------------------------------------------
        if isinstance(turnover, (int, float)) and turnover > 70:
            # Try the GENTLE decay-6 smoothing FIRST — a high-Sharpe near-miss usually
            # just needs its fastest leg slowed, not its whole window structure doubled
            # (which can over-smooth and kill the edge). The aggressive doubled-window +
            # decay-8 variant follows as a fallback for when decay-6 alone won't pull
            # turnover back under the 70% gate.
            add(
                _wrap_fastest_leg(work_expr, 6),
                "decay6",
                f"turnover {turnover:.0f}% >70: ts_decay_linear(...,6) on fastest leg to pull turnover toward 30-40%",
                {"decay": 6},
            )
            add(
                _wrap_fastest_leg(_double_windows(work_expr), 8),
                "decay8_doublewin",
                f"turnover {turnover:.0f}% >70: doubled windows + ts_decay_linear(...,8) on fastest leg (aggressive churn cut)",
                {"decay": 8},
            )
        elif isinstance(turnover, (int, float)) and turnover > 60:
            add(
                _wrap_fastest_leg(work_expr, 6),
                "decay6",
                f"turnover {turnover:.0f}% >60: ts_decay_linear(...,6) on fastest leg",
                {"decay": 6},
            )

        # --- fitness low but Sharpe positive: tighten neutralization -------
        if (isinstance(fitness, (int, float)) and fitness < 1.0
                and isinstance(sharpe, (int, float)) and sharpe > 0):
            if re.search(r",\s*sector\s*\)", work_expr):
                add(
                    re.sub(r",\s*sector\s*\)", ", subindustry)", work_expr),
                    "subindustry",
                    "fitness<1.0, Sharpe>0: neutralize by subindustry to strip finer factor clustering",
                    {"neutralization": "Subindustry"},
                )
            elif str(base_settings.get("neutralization", "")).lower() == "sector":
                # No in-expression group op to flip; move the settings knob instead.
                add(
                    work_expr,
                    "subindustry_setting",
                    "fitness<1.0, Sharpe>0: switch settings neutralization Sector→Subindustry",
                    {"neutralization": "Subindustry"},
                )

        # --- sign wrong ----------------------------------------------------
        if isinstance(sharpe, (int, float)) and sharpe < 0:
            add(
                f"-1 * ({work_expr})",
                "sign_flip",
                f"Sharpe {sharpe:+.2f} < 0: invert the whole expression",
            )

        # --- single leg → quality gate ------------------------------------
        add(
            _add_quality_gate(work_expr),
            "quality_gate",
            "single-leg signal: multiply in rank(ebit/assets) as a quality gate",
        )

        # --- 5-day window sweep -------------------------------------------
        if re.search(r",\s*5\s*\)", work_expr):
            for win in (10, 21):
                add(
                    re.sub(r",\s*5\s*\)", f", {win})", work_expr),
                    f"win{win}",
                    f"5-day window present: lengthen 5→{win} to slow the signal",
                )

        # ---- validate, de-duplicate, materialize -------------------------
        # Seed dedup with only the ORIGINAL base, so a pure hierarchy correction
        # (work_expr != base_expr) is still emitted when no other rule fires.
        out: list[dict] = []
        seen = {_norm(base_expr)}
        for expr, settings_over, tag, reason in candidates:
            if len(out) >= max_mutations:
                break
            key = _norm(expr)
            if key in seen:
                continue
            settings = {**base_settings, **settings_over}
            # Guard against double-neutralization: if the candidate expression
            # self-neutralizes via group_neutralize(), the settings-level
            # neutralization must be 'None' regardless of what any rule requested.
            if "neutralization" in settings:
                settings["neutralization"] = _safe_neutralization(
                    expr, str(settings["neutralization"])
                )
            if not self._structurally_valid(expr, settings):
                continue
            seen.add(key)
            out.append(self._build_alpha(near_miss, expr, settings, tag, reason))
        return out

    # ----------------------------------------------------------- packaging
    @staticmethod
    def _build_alpha(near_miss: dict, expr: str, settings: dict, tag: str, reason: str) -> dict[str, Any]:
        base_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(near_miss.get("name") or "near_miss"))[:48]
        sharpe = near_miss.get("sharpe")
        return {
            "name": f"{base_name}_mut_{tag}"[:64],
            "expression": expr,
            "archetype": near_miss.get("archetype") or "novel",
            "delay": int(settings.get("delay", 1)),
            "logic": f"deterministic near-miss mutation ({tag}): {reason}",
            "paper_source": None,
            "settings": settings,
            "expected_sharpe": str(sharpe) if isinstance(sharpe, (int, float)) else "?",
            "hypothesis": {
                "claim": f"hardened variant of near-miss '{near_miss.get('name')}' "
                         f"(was Sharpe={sharpe}, fitness={near_miss.get('fitness')}, "
                         f"turnover={near_miss.get('turnover_pct')}%)",
                "mechanism": "near_miss_mutation",
                "fields_used": sorted(near_miss.get("fields") or []),
                "field_rationale": reason,
                "refutation_criterion": "the mutation fails to close the gate that the base alpha missed",
            },
            "_mutation": tag,
            "_mutation_reason": reason,
            "_base_expression": near_miss.get("expression"),
            "_base_result": {
                "sharpe": sharpe,
                "fitness": near_miss.get("fitness"),
                "turnover_pct": near_miss.get("turnover_pct"),
            },
        }


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    from feedback_analyzer import FeedbackAnalyzer

    fa = FeedbackAnalyzer()
    mut = NearMissMutator()
    produced = 0
    for nm in fa.top_near_misses(6, exclude_modes=("other",)):
        muts = mut.mutate(nm)
        produced += len(muts)
        print(f"\nBASE [{nm['failure_mode']}] sh={nm['sharpe']} fit={nm['fitness']} "
              f"tov={nm['turnover_pct']}\n  {nm['expression'][:120]}")
        for m in muts:
            print(f"  → [{m['_mutation']}] {m['expression'][:120]}")
    print(f"\nTOTAL mutations produced: {produced}")
