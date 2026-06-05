from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from math_engine import MathEngine


# Operators that perform their own cross-sectional neutralization inside the
# expression. When one is present, the simulation settings must carry a matching
# neutralization rather than "None"/missing.
NEUTRALIZATION_OPS = ("group_neutralize", "vector_neut", "regression_neut")
# Ordered longest-first so "subindustry" wins over the "industry" substring.
_GROUP_LEVELS: tuple[tuple[str, str], ...] = (
    ("subindustry", "Subindustry"),
    ("industry", "Industry"),
    ("sector", "Sector"),
    ("market", "Market"),
)


DEFAULT_SETTINGS: dict[str, Any] = {
    "universe": "TOP3000",
    "neutralization": "Subindustry",
    "decay": 6,
    "truncation": 0.05,
    "delay": 1,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "language": "FASTEXPR",
}


@lru_cache(maxsize=1)
def _engine() -> MathEngine:
    engine = MathEngine()
    # Free-creation mode: judge each expression on its own merits and never reject
    # it for having failed in a previous run (drops the learned_bad_pair memory).
    engine.bad_operator_field_pairs = set()
    return engine


def normalize_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = settings or {}
    out = dict(DEFAULT_SETTINGS)
    out.update({k: v for k, v in raw.items() if v is not None})

    out["universe"] = str(out.get("universe") or "TOP3000").upper()
    if out["universe"] != "TOP3000":
        out["universe"] = "TOP3000"

    neutralization = str(out.get("neutralization") or "Subindustry").strip()
    neut_map = {
        "market": "Market",
        "sector": "Sector",
        "industry": "Industry",
        "subindustry": "Subindustry",
        "none": "None",
    }
    out["neutralization"] = neut_map.get(neutralization.lower(), "Subindustry")

    out["decay"] = _bounded_int(out.get("decay"), default=4, low=0, high=30)
    out["delay"] = _bounded_int(out.get("delay"), default=1, low=0, high=2)
    out["truncation"] = _bounded_float(out.get("truncation"), default=0.05, low=0.0, high=0.2)
    out["pasteurization"] = str(out.get("pasteurization") or "ON").upper()
    out["unitHandling"] = str(out.get("unitHandling") or "VERIFY").upper()
    out["nanHandling"] = str(out.get("nanHandling") or "OFF").upper()
    out["language"] = str(out.get("language") or "FASTEXPR").upper()
    return out


def expression_neutralizes(expression: str) -> bool:
    """True when the expression self-neutralizes via a neutralization operator."""
    expr = re.sub(r"\s+", "", (expression or "").lower())
    return any(f"{op}(" in expr for op in NEUTRALIZATION_OPS)


def _match_paren(text: str, open_idx: int) -> int:
    """Index of the ')' that matches the '(' at open_idx, or -1."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_top_args(inner: str) -> list[str]:
    """Split a call's argument string on top-level commas only."""
    args: list[str] = []
    depth = 0
    cur = ""
    for ch in inner:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


def _level_in(text: str) -> str | None:
    low = text.lower()
    for token, level in _GROUP_LEVELS:
        if re.search(rf"\b{token}\b", low):
            return level
    return None


def expression_neutralization_level(expression: str) -> str | None:
    """The group level a self-neutralizing expression operates at, if detectable.

    Reads the *last argument* of the OUTERMOST group_neutralize() (the
    portfolio-relevant grouping) rather than the first group token anywhere — so
    nested calls like group_neutralize(group_rank(x, subindustry), sector) resolve
    to Sector, not Subindustry.
    """
    expr = expression or ""
    idx = expr.lower().find("group_neutralize(")
    if idx >= 0:
        open_idx = idx + len("group_neutralize")
        close_idx = _match_paren(expr, open_idx)
        if close_idx > open_idx:
            args = _split_top_args(expr[open_idx + 1:close_idx])
            if args:
                level = _level_in(args[-1])
                if level:
                    return level
    # Fallback: any group token anywhere (covers odd phrasings / non-group_neutralize ops).
    return _level_in(expr)


def reconcile_neutralization(expression: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Mirror neutralization that lives in the *expression* into the *settings*.

    When the expression self-neutralizes (group_neutralize / vector_neut /
    regression_neut), the simulation settings must carry a matching neutralization
    instead of "None"/missing. Mirroring the same group level is idempotent at the
    portfolio level, so it is safe and keeps the displayed and simulated settings
    consistent with the code.
    """
    out = dict(settings or {})
    if not expression_neutralizes(expression):
        return out
    level = expression_neutralization_level(expression)
    current = str(out.get("neutralization") or "").strip()
    if level:
        out["neutralization"] = level
    elif current.lower() in ("", "none"):
        out["neutralization"] = "Subindustry"
    return out


def validate_expression(expression: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    expr = (expression or "").strip()
    if not expr:
        return {
            "verdict": "FAIL",
            "turnover": 0.0,
            "predicted_turnover": 0.0,
            "predicted_sharpe_range": "0.0-0.0",
            "archetype": "unknown",
            "issues": [
                {
                    "verdict": "FAIL",
                    "rule": "empty_expression",
                    "message": "Expression is empty.",
                    "fix": "Generate a non-empty FastExpr expression.",
                }
            ],
            "settings_overrides": {},
        }

    clean_settings = normalize_settings(settings)
    try:
        critique = _engine().critique(expr, clean_settings)
    except Exception as exc:  # noqa: BLE001
        return {
            "verdict": "FAIL",
            "turnover": 0.0,
            "predicted_turnover": 0.0,
            "predicted_sharpe_range": "unknown",
            "archetype": "unknown",
            "issues": [
                {
                    "verdict": "FAIL",
                    "rule": "validator_exception",
                    "message": f"Validator failed: {type(exc).__name__}: {exc}",
                    "fix": "Inspect the expression syntax and local Brain catalog files.",
                }
            ],
            "settings_overrides": {},
        }

    turnover = float(critique.get("predicted_turnover") or 0.0)

    # We intentionally mirror the expression's neutralization level into the
    # settings (reconcile_neutralization). Re-neutralizing at the SAME group level
    # is idempotent, so the engine's "set neutralization=None" double-neutralization
    # warning would now be self-contradictory — drop it when the levels already match.
    expr_level = expression_neutralization_level(expr)
    settings_level = str(clean_settings.get("neutralization") or "")
    suppress_double = (
        expression_neutralizes(expr)
        and expr_level is not None
        and settings_level == expr_level
    )

    issues = []
    for issue in critique.get("issues", []):
        rule = issue.get("rule_id") or issue.get("rule") or "validator"
        if suppress_double and rule == "double_neutralization":
            continue
        issues.append({
            "verdict": issue.get("verdict", "WARN"),
            "rule": rule,
            "message": issue.get("message", ""),
            "fix": issue.get("fix_hint") or issue.get("fix"),
        })

    verdict = critique.get("verdict", "FAIL")
    if suppress_double:
        # Recompute from the surviving issues so the badge matches what is shown.
        if any(i["verdict"] == "FAIL" for i in issues):
            verdict = "FAIL"
        elif any(i["verdict"] == "WARN" for i in issues):
            verdict = "WARN"
        else:
            verdict = "PASS"

    return {
        "verdict": verdict,
        "turnover": round(turnover, 1),
        "predicted_turnover": round(turnover, 1),
        "predicted_sharpe_range": critique.get("predicted_sharpe_range"),
        "archetype": critique.get("archetype"),
        "issues": issues,
        "settings_overrides": critique.get("settings_overrides", {}),
    }


def validate_alpha(alpha: dict[str, Any]) -> dict[str, Any]:
    clean = dict(alpha)
    expression = clean.get("expression", "")
    # Mirror any neutralization in the expression into the settings before the
    # settings are used for both display and simulation.
    clean["settings"] = reconcile_neutralization(expression, normalize_settings(clean.get("settings")))
    clean["validation"] = validate_expression(expression, clean["settings"])
    return clean


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _bounded_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return round(max(low, min(high, parsed)), 4)

