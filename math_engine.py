"""
Rule-based Math Engine for WorldQuant Brain alphas.

Deterministic critique layer. Treats Brain as a formal expression language:
  - Operator validation is data-driven from data/brain_docs/operators.json
  - The only hardcoded rejections are genuine syntax errors and TA-Lib operators
    (SMA, EMA, RSI, MACD, ...) that simply don't exist in Brain.
  - Empirical risk patterns (quadratic returns alone, raw volume as primary
    signal, static long-window std, etc.) are WARN — they flag historically
    losing structures without preventing the model from implementing research
    papers that legitimately use them inside composites.

Public API:
    MathEngine.critique(expression, settings=None) -> dict
    MathEngine.predict_turnover(expression) -> float
    MathEngine.predict_archetype(expression) -> str
    MathEngine.auto_settings(expression, memory=None) -> dict
    MathEngine.signature(expression) -> dict
    MathEngine.similarity(sig_a, sig_b) -> float

    validate_formula_structure(formula_str) -> list[str]   # local AST pre-screen
    validate_against_brain_docs() -> list[str]
"""
from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "brain_kb.json"
BRAIN_DOCS = ROOT / "data" / "brain_docs"
OPS_PATH = BRAIN_DOCS / "operators.json"
BAD_PAIRS_PATH = ROOT / "data" / "bad_operator_field_pairs.json"
FEEDBACK_PATH = ROOT / "data" / "feedback.json"
MEMORY_PATH = ROOT / "data" / "memory.json"

# ----------------------------------------------------------------- mechanism families
# A "mechanism family" is the OUTER structural signature of an alpha — the
# economic recipe, independent of which cross-leg or normalization wraps it.
# Two alphas with the same family are highly likely to be return-correlated
# above BRAIN's 0.70 SELF_CORRELATION gate, regardless of LLM-supplied
# archetype tags. We use this for two things:
#   1. Saturated-recipe banning at the prompt layer (avoid clones)
#   2. Hard FAIL in critique() when memory.json already has >=3 of a family
# Add a new family by appending a (name, regex) pair. Match order matters —
# more specific families first.
_MECHANISM_FAMILIES: list[tuple[str, re.Pattern]] = [
    # "ts_std_dev(returns,A) - ts_std_dev(returns,B)" — vol expansion/contraction ratio
    # Also catches the historical_volatility_* form, which is the same economic
    # mechanism with different field names. LLM workaround found in the 2026-05-19
    # 100-round run: it switched from ts_std_dev(returns, N) to historical_volatility_N
    # to bypass the original regex. The new variants here close that loophole.
    ("vol_expansion_ratio",
     re.compile(
         r"ts_std_dev\(returns,\s*\d+\)\s*-\s*ts_std_dev\(returns,\s*\d+\)"
         r"|"
         r"historical_volatility_\d+\s*-\s*historical_volatility_\d+"
         r"|"
         r"parkinson_volatility_\d+\s*-\s*parkinson_volatility_\d+"
         r"|"
         r"realized_volatility_\d+\s*-\s*realized_volatility_\d+"
     )),
    # "(open - ts_delay(close,1)) / ts_delay(close,1)" — overnight gap reversal
    ("overnight_gap",
     re.compile(r"\(?open\s*-\s*ts_delay\(close,\s*1\)\)?\s*/\s*ts_delay\(close,\s*1\)")),
    # "(close - vwap) / vwap" — intraday VWAP-deviation
    ("vwap_deviation",
     re.compile(r"\(?-?\(?close\s*-\s*vwap\)?\s*/\s*vwap")),
    # "ts_delta(debt, N) / ts_delay(debt, N)" — debt-change momentum/reversal
    ("debt_change_ratio",
     re.compile(r"ts_delta\(debt,\s*\d+\)\s*/\s*ts_delay\(debt,\s*\d+\)")),
    # "ts_decay_linear(rank(anl4_*), N)" — analyst-revision decayed signal
    ("analyst_revision_decay",
     re.compile(r"ts_decay_linear\(rank\(anl4_[A-Za-z_]+\),\s*\d+\)")),
    # "abnormal_return_earnings_release" or eps_surprise — true PEAD signal
    ("pead_event",
     re.compile(r"abnormal_return_earnings|eps_surprise|earnings_release_event")),
    # "ts_corr(returns, volume, N)" with short N — return-volume comovement
    ("returns_volume_corr",
     re.compile(r"ts_corr\(returns,\s*volume,\s*(?:[3-9]|1[0-9])\)")),
    # "ts_delta(close, 1) / vwap" — short-window reversal
    ("close_delta_1_vwap",
     re.compile(r"-?ts_delta\(close,\s*1\)\s*/\s*vwap")),
    # "ts_mean(returns, N)" for small N as a primary leg — short reversal
    ("short_returns_mean",
     re.compile(r"-?rank\(\s*-?ts_(?:mean|sum)\(returns,\s*[1-9]\b")),
    # "rank(ts_decay_linear(rank(sales/assets ... )))" — quality-decay
    ("quality_decay",
     re.compile(r"ts_decay_linear\(rank\((?:sales/assets|ebit/assets|gross_profit_to_assets[A-Za-z_]*)")),
]

# The family names above are CATEGORY LABELS for a regex taxonomy — NOT Brain
# fields. They are used ONLY for post-hoc analytics tagging now (classify_mechanism_family
# for brain_api result logging, feedback_analyzer stats, run.py reporting); they are never
# injected into the generation prompt, so the LLM cannot mistake a label for a field.


def classify_mechanism_family(expression: str) -> str:
    """Return a mechanism-family label, or 'other' if nothing matches.

    Pure regex on the normalized expression. Match order matters — first match
    wins, so more specific families are checked first.
    """
    expr = _norm(expression)
    for name, pat in _MECHANISM_FAMILIES:
        if pat.search(expr):
            return name
    return "other"


def memory_family_counts(memory_path: Path | None = None) -> dict[str, int]:
    """Count graduated passes by mechanism family. Used by both the prompt
    saturated-recipes block and the critique() hard-FAIL rule."""
    path = memory_path or MEMORY_PATH
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    entries = doc.get("entries", []) if isinstance(doc, dict) else []
    counts: dict[str, int] = {}
    for e in entries:
        expr = (e.get("expression") or "").strip()
        if not expr:
            continue
        fam = classify_mechanism_family(expr)
        counts[fam] = counts.get(fam, 0) + 1
    return counts


# ----------------------------------------------------------------------------
# Operators that BRAIN rejects on event-typed (sparse / News+Social VECTOR) inputs.
# Confirmed empirically from feedback logs: rank, ts_decay_linear on nws18_*, scl12_*.
# zscore is included by symmetry (same cross-sectional family as rank).
_EVENT_UNSAFE_OPS = {"rank", "zscore", "ts_decay_linear", "normalize", "quantile", "winsorize"}

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

# Reserved tokens that look like calls but aren't operators
RESERVED_CALLABLES = {"if", "and", "or", "not"}

# Unit classification for branch-consistency checks (if_else / max / min / signed_power)
# PRICE-like (CSPrice): dollar-denominated cross-sectional fields.
# UNITLESS: returns, ranks, zscores, correlations, day-counts, booleans.
# VOLUME: share counts and dollar-volume aggregates.
_PRICE_FIELDS = {"close", "open", "high", "low", "vwap", "cap",
                 "enterprise_value", "ebitda"}
_VOLUME_FIELDS = {"volume"}
_UNITLESS_FIELDS = {"returns", "sector", "industry", "subindustry"}
# Operators whose output is always unitless regardless of input units
_UNITLESS_OPS = {"rank", "zscore", "normalize", "quantile", "winsorize",
                 "sign", "ts_rank", "ts_zscore", "ts_corr", "ts_arg_max",
                 "ts_arg_min", "ts_count_nans", "ts_returns", "ts_skewness",
                 "ts_kurtosis", "group_rank", "group_zscore",
                 "less", "greater", "equal", "and", "or", "not", "if_else"}


@dataclass
class Issue:
    rule_id: str
    verdict: str
    message: str
    fix_hint: str | None = None


@dataclass
class Critique:
    verdict: str
    predicted_turnover: float
    predicted_sharpe_range: str
    archetype: str
    issues: list[Issue] = field(default_factory=list)
    fix: str | None = None
    settings_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "predicted_turnover": round(self.predicted_turnover, 1),
            "predicted_sharpe_range": self.predicted_sharpe_range,
            "archetype": self.archetype,
            "issues": [
                {
                    "rule_id": i.rule_id,
                    "verdict": i.verdict,
                    "message": i.message,
                    "fix_hint": i.fix_hint,
                }
                for i in self.issues
            ],
            "fix": self.fix,
            "settings_overrides": self.settings_overrides,
        }


def _load_kb() -> dict:
    return json.loads(KB_PATH.read_text())


def _load_known_operators() -> set[str]:
    """Read operator whitelist from data/brain_docs/operators.json (data-driven)."""
    if not OPS_PATH.exists():
        return set()
    try:
        doc = json.loads(OPS_PATH.read_text())
    except json.JSONDecodeError:
        return set()
    return {k for k in doc.keys() if not k.startswith("_")}


def _load_operator_arity() -> dict[str, tuple[int, int]]:
    """Derive each operator's (min_args, max_args) from its `signature` string in
    operators.json, e.g. 'ts_corr(x, y, d)' -> (3, 3), 'scale(x, scale=1)' -> (1, 2).

    Data-driven so it auto-covers every documented operator (ts_corr, ts_covariance,
    quantile, group_mean, ...) instead of a hardcoded list — closing the arg-count
    hole that let `ts_corr(returns, volume)` (2 args, needs 3) reach BRAIN and burn
    a simulation slot on a guaranteed 'Invalid number of inputs' rejection.
    """
    if not OPS_PATH.exists():
        return {}
    try:
        doc = json.loads(OPS_PATH.read_text())
    except json.JSONDecodeError:
        return {}
    arity: dict[str, tuple[int, int]] = {}
    for name, meta in doc.items():
        if name.startswith("_") or not isinstance(meta, dict):
            continue
        sig = meta.get("signature")
        if not isinstance(sig, str):
            continue
        m = re.match(r"[A-Za-z_][A-Za-z_0-9]*\((.*)\)\s*$", sig.strip())
        if not m:
            continue
        inner = m.group(1).strip()
        if not inner:
            arity[name] = (0, 0)
            continue
        args = [a.strip() for a in inner.split(",") if a.strip()]
        required = sum(1 for a in args if "=" not in a)
        arity[name] = (required, len(args))
    return arity


_OPERATOR_ARITY_CACHE: dict[str, tuple[int, int]] | None = None


def _operator_arity_cached() -> dict[str, tuple[int, int]]:
    """Process-lifetime cache of the signature-derived operator arity map."""
    global _OPERATOR_ARITY_CACHE
    if _OPERATOR_ARITY_CACHE is None:
        _OPERATOR_ARITY_CACHE = _load_operator_arity()
    return _OPERATOR_ARITY_CACHE


_FIELD_IDS_CACHE: set[str] | None = None


def _field_ids_cached() -> set[str]:
    """Process-lifetime cache of every field id in the BRAIN catalog
    (fields_full.json). Used to reject hallucinated field names pre-flight so
    they never burn a BRAIN simulation slot on an 'unknown variable' rejection.
    Empty set when the catalog hasn't been fetched — callers must treat empty as
    'checking disabled' to avoid false positives."""
    global _FIELD_IDS_CACHE
    if _FIELD_IDS_CACHE is None:
        _FIELD_IDS_CACHE = set(_load_field_catalog_metadata().keys())
    return _FIELD_IDS_CACHE


# Catalog-driven field metadata. Loaded lazily from data/brain_docs/fields_full.json
# (produced by training/fetch_fields.py). Lets the validator know the unit bucket of
# any field Brain exposes, not just the canonical seven.
_FIELDS_FULL_PATH = ROOT / "data" / "brain_docs" / "fields_full.json"


def _load_field_catalog_metadata() -> dict[str, dict]:
    """Return {field_id: {dataset_id, category, type, ...}} from fields_full.json.
    Empty dict if the catalog hasn't been fetched yet."""
    if not _FIELDS_FULL_PATH.exists():
        return {}
    try:
        doc = json.loads(_FIELDS_FULL_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return doc.get("fields") or {}


# How Brain categories map to the unit buckets the validator already uses for
# branch-consistency checks. PRICE = dollar-denominated; UNITLESS = ratios / counts /
# scores; UNKNOWN = type that doesn't cleanly fit (treated as 'unknown' = no check fires).
_CATEGORY_TO_BUCKET = {
    "Price Volume": "price",       # close, cap, adv20, vwap, etc. (dollar/share)
    "Fundamental":  "fundamental",  # assets, eps, sales — dollar-ish, own bucket
    "Analyst":      "fundamental",  # consensus EPS / sales estimates — same bucket
    "Option":       "unitless",     # implied vol, breakevens etc. (ratios/dollars mixed)
    "Sentiment":    "unitless",
    "News":         "unitless",
    "Model":        "unitless",     # composite scores
}


def _load_empirical_sharpe_by_archetype(min_samples: int = 5) -> dict[str, float]:
    """Median sharpe per archetype from past simulations (feedback.json + memory.json).
    Returns {} when there's no data — callers fall back to the KB midpoint.

    The KB's `expected_sharpe` ranges are aspirational (1.0-2.5 etc.) and decouple
    the pre-submission fitness predictor from reality. After 400+ sims, the median
    sharpe per archetype is closer to 0.0-0.3 — using that lets the fitness gate
    actually discriminate between candidate alphas instead of waving them all through.
    """
    by_arch: dict[str, list[float]] = {}
    for path in (FEEDBACK_PATH, MEMORY_PATH):
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for e in doc.get("entries", []):
            arch = e.get("archetype")
            sh = e.get("sharpe")
            if isinstance(arch, str) and isinstance(sh, (int, float)):
                by_arch.setdefault(arch, []).append(float(sh))
    out: dict[str, float] = {}
    for arch, samples in by_arch.items():
        if len(samples) >= min_samples:
            samples_sorted = sorted(samples)
            out[arch] = samples_sorted[len(samples_sorted) // 2]  # median
    return out


def _load_bad_pairs() -> set[tuple[str, str]]:
    """Read learned (operator, field) pairs that BRAIN has rejected. File is
    appended to by brain_api._log_result on 'does not support ... inputs' errors."""
    if not BAD_PAIRS_PATH.exists():
        return set()
    try:
        doc = json.loads(BAD_PAIRS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    pairs = doc.get("pairs", []) if isinstance(doc, dict) else []
    out: set[tuple[str, str]] = set()
    for p in pairs:
        if isinstance(p, dict) and p.get("operator") and p.get("field"):
            out.add((p["operator"], p["field"]))
        elif isinstance(p, (list, tuple)) and len(p) == 2:
            out.add((str(p[0]), str(p[1])))
    return out


def _classify_catalog_field(field_id: str, meta: dict) -> str:
    """Map a catalog field to the validator's unit-bucket vocabulary.
    Some Price-Volume fields are actually share counts or returns — patch them by name."""
    if meta.get("type") in ("GROUP", "SYMBOL"):
        return "categorical"
    name = field_id.lower()
    # Volume-flavored fields keep the volume bucket
    if name == "volume" or name.startswith("volume_") or "_volume" in name:
        return "volume"
    if re.fullmatch(r"adv\d+", name):
        return "price"   # dollar-volume
    if name in ("returns", "ret", "ret_d"):
        return "unitless"
    return _CATEGORY_TO_BUCKET.get(meta.get("category"), "unknown")


def _norm(expr: str) -> str:
    return re.sub(r"\s+", "", expr)


def _extract_call_names(expression: str) -> list[str]:
    """All identifiers that appear immediately before '('. Case-sensitive."""
    return [m.group(1) for m in re.finditer(r"([A-Za-z_][A-Za-z_0-9]*)\s*\(", expression)]


class MathEngine:
    def __init__(self, kb: dict | None = None) -> None:
        self.kb = kb if kb is not None else _load_kb()
        self.archetypes = self.kb.get("archetypes", {})
        self.forbidden = set(self.kb.get("forbidden_operators", []))
        self.forbidden_fields = set(
            f for f in self.kb.get("forbidden_fields", []) if not f.endswith("*")
        )
        self.forbidden_field_prefixes = tuple(
            f[:-1] for f in self.kb.get("forbidden_fields", []) if f.endswith("*")
        )
        self.defaults = self.kb.get("settings_defaults", {})
        self.known_ops = _load_known_operators()
        # Catalog-driven field-unit map. Fields not in the catalog fall back to the
        # canonical _PRICE_FIELDS/_VOLUME_FIELDS/_UNITLESS_FIELDS sets, so this is
        # purely additive: extends coverage without changing existing behavior.
        self.field_catalog: dict[str, dict] = _load_field_catalog_metadata()
        self.catalog_units: dict[str, str] = {
            fid: _classify_catalog_field(fid, meta)
            for fid, meta in self.field_catalog.items()
        }
        # GROUP-type fields are valid group args for group_neutralize/group_rank/group_zscore.
        self.group_fields: set[str] = {
            fid for fid, meta in self.field_catalog.items() if meta.get("type") == "GROUP"
        }
        # Event-typed fields: BRAIN rejects rank/zscore/ts_decay_linear over these.
        # VECTOR + (News|Social Media) is the empirical pattern (Ravenpack nws18_*, scl12_*).
        self.event_fields: set[str] = {
            fid for fid, meta in self.field_catalog.items()
            if meta.get("type") == "VECTOR"
            and meta.get("category") in ("News", "Social Media")
        }
        # Persisted bad pairs learned from prior BRAIN errors (operator, field).
        self.bad_operator_field_pairs: set[tuple[str, str]] = _load_bad_pairs()
        # Empirical median sharpe per archetype, from feedback.json + memory.json.
        # Replaces the aspirational KB midpoints in _predict_fitness so the
        # pre-submission gate actually discriminates between candidates.
        self.empirical_sharpe: dict[str, float] = _load_empirical_sharpe_by_archetype()

    # ------------------------------------------------------------------ critique
    def critique(self, expression: str, settings: dict | None = None) -> dict:
        expr = _norm(expression)
        issues: list[Issue] = []
        overrides: dict[str, Any] = {}

        # ---- SYNTAX-LEVEL (FAIL) -----------------------------------------
        # TA-Lib forbid list (genuine non-Brain operators)
        for op in self.forbidden:
            if re.search(rf"\b{re.escape(op)}\s*\(", expression):
                issues.append(Issue(
                    "forbidden_operator_talib", FAIL,
                    f"'{op}' is a TA-Lib operator, not a Brain operator.",
                    fix_hint="Use the equivalent Brain operator (ts_mean, ts_std_dev, ts_delta, ts_decay_linear, ...).",
                ))

        # (Removed: the editorial "saturated_mechanism_family" FAIL. It blocked the
        # LLM from generating in any mechanism family that already had a few passes —
        # steering, not a platform rule. The real BRAIN self-correlation gate is still
        # enforced at submission via data/self_correlation_cache.json. The regex
        # taxonomy (classify_mechanism_family) is retained ONLY as a post-hoc analytics
        # tag, never to constrain generation.)

        # Time-series operators must have a day argument
        if re.search(r"ts_(mean|sum|std_dev|delta|corr|zscore|covariance|product|rank|arg_max|arg_min|decay_linear|skewness|kurtosis|returns|count_nans)\(([^,()]+)\)", expression):
            issues.append(Issue(
                "missing_day_arg", FAIL,
                "ts_* operator missing day-window argument (e.g. ts_mean(x) → ts_mean(x, d)).",
            ))

        # Python/pandas-style method syntax is invalid in FastExpr (e.g. volume.shift(10))
        if re.search(r"\b[A-Za-z_][A-Za-z_0-9]*\.(shift|mean|std|rolling|ewm)\s*\(", expression):
            issues.append(Issue(
                "python_method_syntax", FAIL,
                "Python/pandas method syntax detected (e.g. x.shift(...), x.mean(...)); FastExpr does not support dot-method calls.",
                fix_hint="Use Brain operators: ts_delay(x, d), ts_mean(x, d), ts_std_dev(x, d), etc.",
            ))

        # Common non-Brain alias seen in papers/code snippets
        if re.search(r"\bdelay\s*\(", expression):
            issues.append(Issue(
                "invalid_delay_operator", FAIL,
                "'delay(...)' is not a Brain operator in this project. Use ts_delay(x, d).",
                fix_hint="Replace delay(x, d) with ts_delay(x, d).",
            ))

        # Convenience placeholders like mean_volume20 are not Brain fields.
        if re.search(r"\bmean_volume\d+\b", expression):
            issues.append(Issue(
                "invalid_convenience_variable", FAIL,
                "Convenience variable like mean_volumeN detected; Brain expects explicit operators.",
                fix_hint="Replace mean_volumeN with ts_mean(volume, N).",
            ))

        # zscore is unary
        i = 0
        while True:
            m = re.search(r"\bzscore\(", expression[i:])
            if not m:
                break
            start = i + m.end()   # index of char after the opening '('
            depth = 1
            j = start
            while j < len(expression) and depth > 0:
                if expression[j] == "(":
                    depth += 1
                elif expression[j] == ")":
                    depth -= 1
                j += 1
            inner = expression[start:j - 1]
            # Count top-level commas in inner
            d2, top_commas = 0, 0
            for c in inner:
                if c == "(":
                    d2 += 1
                elif c == ")":
                    d2 -= 1
                elif c == "," and d2 == 0:
                    top_commas += 1
            if top_commas > 0:
                issues.append(Issue(
                    "zscore_multi_arg", FAIL,
                    "zscore() takes exactly one argument — use ts_zscore(x, d) for windowed version.",
                ))
                break
            i = j  # advance past this zscore(...)

        # Parenthesis balance — cheap sanity check
        if expression.count("(") != expression.count(")"):
            issues.append(Issue(
                "paren_imbalance", FAIL,
                f"Parenthesis imbalance: {expression.count('(')} '(' vs {expression.count(')')} ')'.",
            ))

        # Forbidden field detection (e.g. enterprise_value, cap, adv20, fnd6_*).
        # A token is a "field" if it appears NOT immediately followed by '(' (which would be a call).
        flagged_fields: set[str] = set()
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z_0-9]*)\b", expression):
            tok = m.group(1)
            if tok in flagged_fields:
                continue
            next_char = expression[m.end():m.end() + 1]
            if next_char == "(":
                continue
            if tok in self.forbidden_fields or any(tok.startswith(p) for p in self.forbidden_field_prefixes):
                flagged_fields.add(tok)
                issues.append(Issue(
                    "forbidden_field", FAIL,
                    f"'{tok}' is not a Brain data field — submission will fail at parse time.",
                    fix_hint="Use sharesout*close for market cap proxy, or one of the valid fundamentals in data/brain_docs/fields.json (ebitda, ebit, sales, assets, equity, liabilities, debt, sharesout, dividends).",
                ))

        # Hard-FAIL heuristic: positive cubic moment buys lottery stocks (empirically negative Sharpe)
        if re.search(r"zscore\(\+?ts_mean\(returns\*returns\*returns", expr) and \
           not re.search(r"zscore\(-ts_mean\(returns\*returns\*returns", expr):
            issues.append(Issue(
                "positive_cubic_moment", FAIL,
                "Positive cubic moment buys lottery-like stocks (empirically negative Sharpe). Negate it.",
                fix_hint="zscore(-ts_mean(returns*returns*returns, 21))",
            ))

        # Event-typed fields (News / Social Media VECTOR) are rejected by BRAIN
        # when wrapped in cross-sectional / decay ops. Block at pre-flight.
        if self.event_fields:
            for op_name, inner in self._direct_wrapped_args(expression):
                inner_token = inner.lstrip("+-").strip()
                # only fail when the WHOLE wrapped argument is a bare event field
                if (op_name in _EVENT_UNSAFE_OPS
                        and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", inner_token)
                        and inner_token in self.event_fields):
                    issues.append(Issue(
                        "event_input_unsupported", FAIL,
                        f"'{op_name}({inner_token})' wraps an event-typed VECTOR field — BRAIN rejects this at parse time.",
                        fix_hint=f"Combine {inner_token} via group_neutralize/group_rank, or feed it into ts_corr(..., d) as the second arg.",
                    ))
                    break

        # Learned bad pairs from past BRAIN errors.
        if self.bad_operator_field_pairs:
            for op_name, inner in self._direct_wrapped_args(expression):
                inner_token = inner.lstrip("+-").strip()
                if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", inner_token):
                    continue
                if (op_name, inner_token) in self.bad_operator_field_pairs:
                    issues.append(Issue(
                        "learned_bad_pair", FAIL,
                        f"'{op_name}({inner_token})' was rejected by BRAIN previously (see data/bad_operator_field_pairs.json).",
                        fix_hint="Pick a different operator/field combination.",
                    ))
                    break

        # ---- DATA-DRIVEN OP VALIDATION (WARN) -----------------------------
        if self.known_ops:
            seen = set()
            for name in _extract_call_names(expression):
                if name in seen or name in RESERVED_CALLABLES:
                    continue
                seen.add(name)
                if name in self.forbidden:
                    continue  # already handled above
                if name not in self.known_ops:
                    issues.append(Issue(
                        "unknown_operator", FAIL,
                        f"'{name}' is not in data/brain_docs/operators.json. May be a typo or an unseeded operator.",
                        fix_hint="Correct the operator name, or seed it in data/brain_docs/operators.json before generating.",
                    ))

        # ---- EMPIRICAL RISK HEURISTICS (WARN) -----------------------------
        # Quadratic returns as primary signal
        if "returns*returns" in expr and "returns*returns*returns" not in expr \
                and not self._inside_correlation_op(expression, "returns*returns"):
            issues.append(Issue(
                "quadratic_returns_alone", WARN,
                "returns*returns as primary signal creates a directionless variance proxy. Empirically weak as standalone.",
                fix_hint="Either embed inside ts_corr/ts_covariance, or use cubic: -ts_mean(returns*returns*returns, 21).",
            ))

        # Raw volume as primary signal (volume not normalized, and not inside ts_corr/ts_covariance)
        if self._has_raw_volume_primary(expr):
            issues.append(Issue(
                "raw_volume_primary", WARN,
                "Raw volume as primary signal biases toward mega-caps.",
                fix_hint="Normalize: ts_sum(volume, 5) / ts_mean(volume, 60), or use volume only inside ts_corr.",
            ))

        # ts_delta(close, N) not normalized
        if self._has_raw_close_delta(expr):
            issues.append(Issue(
                "raw_ts_delta_close", WARN,
                "ts_delta(close, N) is dollar-denominated. Consider /close or /ts_mean(close, M).",
                fix_hint="ts_delta(close, 1) / ts_mean(close, 252)",
            ))

        # Static long-window std as sole signal
        if self._is_static_long_std(expr):
            issues.append(Issue(
                "static_long_window_std", WARN,
                "ts_std_dev(returns, d>=200) alone produces near-zero daily turnover. Combine with a daily-varying signal or shorter window.",
                fix_hint="Multiply by a daily-varying signal (e.g. rank(ts_sum(volume,5)/ts_mean(volume,60)) ) or shorten window to 20-60d.",
            ))

        # Double-neutralization guard
        # When the expression already contains group_neutralize(), the portfolio-level
        # neutralization setting must be "None" — not "Market". Setting it to "Market"
        # applies a second demean pass that collapses the within-group return spread.
        if "group_neutralize(" in expr:
            if settings and settings.get("neutralization") not in (None, "None"):
                issues.append(Issue(
                    "double_neutralization", WARN,
                    "group_neutralize() inside expression — neutralization setting must be 'None' "
                    "to avoid a second portfolio-level demean that collapses returns.",
                    fix_hint="Set neutralization=None",
                ))
            overrides["neutralization"] = "None"

        # Min decay heuristic — WARN only, do NOT force.
        # Microstructure alphas are intentionally short-horizon (decay 0..2 OK).
        # All other archetypes do better with decay >= 4 by default.
        cand_archetype = self.predict_archetype(expression)
        if (settings and settings.get("decay", 4) < 4
                and cand_archetype not in ("microstructure",)):
            issues.append(Issue(
                "min_decay", WARN,
                f"decay={settings.get('decay')} may produce high turnover for a {cand_archetype} alpha; "
                f"decay>=4 is typical unless this is intentionally a fast signal.",
                fix_hint="Consider decay >= 4, or override via auto_settings archetype defaults.",
            ))

        # Unguarded division: a / b where b is a potentially-zero quantity (field or ts_* call)
        # Brain's unit verifier rejects additive scalar guards on unitful denominators
        # (e.g. vwap + 0.001, ts_mean(volume,60) + 1), so this is a light WARN only.
        # Examples:
        #   close / vwap                       → WARN
        #   (close - vwap) / vwap             → unit-consistent
        #   ts_sum(volume,5) / ts_mean(volume,60) → unit-consistent
        for div_expr in self._extract_division_denominators(expression):
            if self._looks_like_unitful_denominator(div_expr):
                issues.append(Issue(
                    "unguarded_division", WARN,
                    f"Division by `{div_expr.strip()}` relies on Brain's NaN handling; do not add scalar epsilons to unitful denominators.",
                    fix_hint="Keep unitful ratios unit-consistent: x / denom, then wrap the final ratio in rank(), zscore(), or ts_zscore().",
                ))
                break  # one warning per expression is enough

        # Brain VERIFY mode rejects adding dimensionless constants to unitful fields
        # or unitful time-series operators. This is the source of errors like:
        # expected Unit[TSPrice:1,TSShare:1], found Unit[].
        bad_add = self._find_unitful_numeric_add(expression)
        if bad_add:
            issues.append(Issue(
                "unitful_numeric_add", FAIL,
                f"Unit-incompatible numeric add `{bad_add}`. Brain VERIFY rejects adding scalar constants to price/share/fundamental units.",
                fix_hint="Remove the + constant from unitful quantities; use x / denom and normalize the ratio with rank(), zscore(), or ts_zscore().",
            ))

        # Unit-mismatch in if_else / max / min branches
        for op_name in ("if_else", "max", "min"):
            for call in self._find_calls(expression, op_name):
                args = self._split_args(call)
                if op_name == "if_else" and len(args) >= 3:
                    branches = args[1:3]
                elif op_name in ("max", "min") and len(args) >= 2:
                    branches = args[:2]
                else:
                    continue
                units = [self._classify_unit(b) for b in branches]
                known = [u for u in units if u != "unknown"]
                if len(set(known)) >= 2:
                    issues.append(Issue(
                        "units_mismatch", WARN,
                        f"{op_name}() branches have incompatible units: {' vs '.join(units)}. "
                        f"Brain will emit a UNITS warning and the alpha cannot be submitted.",
                        fix_hint=f"Normalize both branches to the same unit (e.g. wrap price in rank() or divide by ts_mean(close,N))",
                    ))
                    break  # one per op_name is enough

        archetype = self.predict_archetype(expression)
        predicted_turnover = self.predict_turnover(expression)
        sharpe_range = self._sharpe_range(archetype)
        predicted_fitness = self._predict_fitness(archetype, predicted_turnover)

        # Predicted turnover out of band
        if not any(i.verdict == FAIL for i in issues):
            if predicted_turnover < 1 or predicted_turnover > 70:
                issues.append(Issue(
                    "predicted_turnover_out_of_band", WARN,
                    f"Predicted turnover {predicted_turnover:.1f}% outside 1-70% target band.",
                ))

        # Predicted fitness clearly below Brain's submission floor (conservative threshold).
        # Only fires when the structure has a known weak pattern: high turnover, OR
        # quadratic returns / raw volume / broken if_else already flagged elsewhere.
        # We skip novel archetypes with modest turnover — too easy to false-positive there.
        has_smoothing = "ts_decay_linear" in expr or "group_neutralize" in expr
        weak_structure = (
            predicted_turnover >= 40.0
            or any(i.rule_id in ("quadratic_returns_alone", "raw_volume_primary",
                                  "units_mismatch", "discrete_signal_no_smoothing")
                   for i in issues)
        )
        if (predicted_fitness < 0.70
                and weak_structure
                and not has_smoothing
                and not any(i.verdict == FAIL for i in issues)):
            hint = "Smooth with ts_decay_linear, lengthen ts_arg_max/ts_rank windows, or raise decay setting."
            issues.append(Issue(
                "predicted_low_fitness", WARN,
                f"Predicted fitness {predicted_fitness:.2f} < 1.0 floor "
                f"(sharpe_mid~{self._sharpe_mid(archetype):.2f}, turnover~{predicted_turnover:.0f}%).",
                fix_hint=hint,
            ))

        # Discrete signal (ts_arg_max/ts_arg_min) over a tiny window with no smoothing
        if self._is_discrete_low_resolution(expression, settings):
            issues.append(Issue(
                "discrete_signal_no_smoothing", WARN,
                "ts_arg_max/ts_arg_min over a short window produces only a handful of distinct values; "
                "without ts_decay_linear or high decay, fitness will be low.",
                fix_hint="Wrap in ts_decay_linear(..., 4-8) or extend the window to >=10.",
            ))

        # Structural hierarchy: group_neutralize must be the OUTERMOST operator.
        # Wrapping a cross-sectional transform (rank/zscore/scale/...) around a
        # neutralized signal re-ranks it globally and destroys the within-group
        # dollar-neutrality the neutralization just imposed — a silent factor-
        # exposure leak. (A leading sign flip / scalar is fine and not matched here.)
        if re.search(
            r"\b(rank|zscore|scale|normalize|quantile|ts_rank|ts_zscore|"
            r"signed_power|group_rank|group_zscore)\(\s*group_neutralize\(",
            expression,
        ):
            issues.append(Issue(
                "neutralize_not_outermost", WARN,
                "group_neutralize is wrapped by a transform; it should be the OUTERMOST operator. "
                "Wrapping rank/zscore/scale around a neutralized signal re-introduces factor "
                "exposure and breaks subindustry dollar-neutrality.",
                fix_hint="Move neutralization outside: group_neutralize(rank(A) * sign(rank(B)-0.5), subindustry).",
            ))

        verdict = PASS
        if any(i.verdict == FAIL for i in issues):
            verdict = FAIL
        elif any(i.verdict == WARN for i in issues):
            verdict = WARN

        fix = self._propose_fix(expression, issues)
        return Critique(
            verdict=verdict,
            predicted_turnover=predicted_turnover,
            predicted_sharpe_range=sharpe_range,
            archetype=archetype,
            issues=issues,
            fix=fix,
            settings_overrides=overrides,
        ).to_dict()

    # ------------------------------------------------------------ unit inference
    def _classify_unit(self, sub: str) -> str:
        """Coarse unit bucket for a sub-expression: 'price', 'volume', 'unitless', or 'unknown'.
        Heuristic — not a full type system. Used only to flag obvious mismatches in
        if_else / max / min / signed_power branches."""
        s = sub.strip()
        if not s:
            return "unknown"
        # Numeric literal → unitless
        if re.fullmatch(r"-?\d+(?:\.\d*)?", s):
            return "unitless"
        # Outermost call: if its operator is always unitless, the whole branch is unitless.
        m = re.match(r"([A-Za-z_][A-Za-z_0-9]*)\s*\(", s)
        if m and m.group(1) in _UNITLESS_OPS:
            return "unitless"
        # ts_std_dev / ts_mean / ts_max / ts_min / ts_decay_linear / ts_sum / ts_delta
        # inherit the unit of their first argument.
        if m and m.group(1) in {"ts_std_dev", "ts_mean", "ts_max", "ts_min",
                                "ts_decay_linear", "ts_sum", "ts_delta",
                                "ts_product", "abs", "log", "exp", "sqrt"}:
            inner = self._first_arg(s)
            return self._classify_unit(inner) if inner else "unknown"
        # signed_power / pow inherit unit of base argument (sign of e doesn't matter here)
        if m and m.group(1) in {"signed_power", "pow"}:
            inner = self._first_arg(s)
            return self._classify_unit(inner) if inner else "unknown"
        # Bare field tokens
        if s in _PRICE_FIELDS or re.fullmatch(r"adv\d+", s):
            return "price"
        if s in _VOLUME_FIELDS:
            return "volume"
        if s in _UNITLESS_FIELDS or s.startswith("fnd6_"):
            return "unitless"
        # Catalog-driven fallback for fields beyond the canonical seven
        if s in self.catalog_units:
            bucket = self.catalog_units[s]
            if bucket != "unknown":
                return bucket
        # Division of same-unit cancels to unitless; rough check
        if "/" in s and "(" not in s:
            return "unitless"
        return "unknown"

    @staticmethod
    def _first_arg(call_expr: str) -> str | None:
        """Return the first argument of a function call expression, respecting nested parens."""
        open_idx = call_expr.find("(")
        if open_idx == -1:
            return None
        depth = 0
        for i in range(open_idx, len(call_expr)):
            c = call_expr[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    inner = call_expr[open_idx + 1:i]
                    break
        else:
            return None
        # Split on commas at depth 0
        depth = 0
        start = 0
        for i, c in enumerate(inner):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == "," and depth == 0:
                return inner[start:i].strip()
        return inner.strip()

    def _split_args(self, call_expr: str) -> list[str]:
        """Split a function call's argument list at top-level commas."""
        open_idx = call_expr.find("(")
        if open_idx == -1:
            return []
        depth = 0
        for i in range(open_idx, len(call_expr)):
            c = call_expr[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    inner = call_expr[open_idx + 1:i]
                    break
        else:
            return []
        args: list[str] = []
        depth = 0
        start = 0
        for i, c in enumerate(inner):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == "," and depth == 0:
                args.append(inner[start:i].strip())
                start = i + 1
        args.append(inner[start:].strip())
        return args

    def _find_calls(self, expression: str, op_name: str) -> list[str]:
        """Return each top-level-or-nested `op_name(...)` substring."""
        out: list[str] = []
        pattern = re.compile(rf"\b{re.escape(op_name)}\s*\(")
        for m in pattern.finditer(expression):
            start = m.start()
            depth = 0
            i = m.end() - 1  # at '('
            while i < len(expression):
                c = expression[i]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        out.append(expression[start:i + 1])
                        break
                i += 1
        return out

    def _wraps_pattern(self, expression: str, outer_op: str, inner_patterns: list[str]) -> bool:
        """True if any `outer_op(...)` call contains a match for one of inner_patterns."""
        for call in self._find_calls(expression, outer_op):
            for pat in inner_patterns:
                if re.search(pat, call):
                    return True
        return False

    def _direct_wrapped_args(self, expression: str) -> list[tuple[str, str]]:
        """Yield (op_name, first_arg_text) for every call in the expression.
        Used to spot patterns like rank(nws18_qcm) where an unsafe op directly
        wraps a single field token."""
        out: list[tuple[str, str]] = []
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\(", expression):
            op_name = m.group(1)
            if op_name in RESERVED_CALLABLES:
                continue
            depth = 0
            i = m.end() - 1  # at '('
            start = i + 1
            while i < len(expression):
                c = expression[i]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        inner = expression[start:i]
                        # first top-level arg only
                        d = 0
                        first_end = len(inner)
                        for k, ch in enumerate(inner):
                            if ch == "(":
                                d += 1
                            elif ch == ")":
                                d -= 1
                            elif ch == "," and d == 0:
                                first_end = k
                                break
                        out.append((op_name, inner[:first_end].strip()))
                        break
                i += 1
        return out

    # ------------------------------------------------------------ division guard
    @staticmethod
    def _extract_division_denominators(expression: str) -> list[str]:
        """Return each immediate right-operand of '/' in the expression.
        Handles `/ field`, `/ ts_call(...)`, `/ (parenthesized)`."""
        out: list[str] = []
        i = 0
        while i < len(expression):
            if expression[i] != "/":
                i += 1
                continue
            j = i + 1
            while j < len(expression) and expression[j] == " ":
                j += 1
            if j >= len(expression):
                break
            if expression[j] == "(":
                # Parenthesized denominator — capture matching pair
                depth = 0
                start = j
                while j < len(expression):
                    c = expression[j]
                    if c == "(":
                        depth += 1
                    elif c == ")":
                        depth -= 1
                        if depth == 0:
                            out.append(expression[start:j + 1])
                            break
                    j += 1
                i = j + 1
                continue
            # Identifier (possibly a call): grab identifier + optional (...)
            m = re.match(r"([A-Za-z_][A-Za-z_0-9]*)", expression[j:])
            if not m:
                i = j + 1
                continue
            name = m.group(1)
            after = j + len(name)
            if after < len(expression) and expression[after] == "(":
                # function call — find its closing paren
                depth = 0
                k = after
                while k < len(expression):
                    c = expression[k]
                    if c == "(":
                        depth += 1
                    elif c == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1
                out.append(expression[j:k + 1])
                i = k + 1
            else:
                out.append(name)
                i = after
        return out

    @staticmethod
    def _looks_like_unitful_denominator(denom: str) -> bool:
        d = denom.strip()
        if d.startswith("(") and d.endswith(")"):
            d = d[1:-1]
        unitful_tokens = (
            "close", "open", "high", "low", "vwap", "volume",
            "ebitda", "ebit", "sales", "assets", "equity",
            "liabilities", "debt", "sharesout",
        )
        return any(re.search(rf"\b{tok}\b", d) for tok in unitful_tokens)

    @classmethod
    def _find_unitful_numeric_add(cls, expression: str) -> str | None:
        """Find common Brain unit errors like `(vwap + 0.001)` or `(sales+1)`.

        Adding a dimensionless scalar to a unitful series fails when unitHandling=VERIFY.
        Cross-sectional ops (rank / zscore / normalize / quantile / scale / group_*)
        return unitless values, so a unitful token wrapped inside one of those ops
        does NOT make the surrounding subtraction unit-incompatible — e.g.
        `1 - rank(high/low)` is fine even though `high` and `low` are unitful.
        Strip cross-sectional wraps before checking for residual unitful tokens.
        """
        unitful = (
            "close", "open", "high", "low", "vwap", "volume",
            "ebitda", "ebit", "sales", "assets", "equity",
            "liabilities", "debt", "sharesout",
        )
        _CROSS_SECTIONAL_OPS_RE = re.compile(
            r"\b(?:rank|zscore|normalize|quantile|scale|group_rank|group_zscore|"
            r"group_neutralize|group_mean|winsorize)\s*\("
        )

        def _strip_xsection_wraps(s: str) -> str:
            """Repeatedly remove the contents of any rank()/zscore()/etc call.
            What's left is the expression with all unitless sub-expressions
            replaced by an empty token, so unitful-token detection only fires
            on bare unitful references."""
            out = s
            for _ in range(8):  # bounded loop — practical expressions nest < 8 deep
                m = _CROSS_SECTIONAL_OPS_RE.search(out)
                if not m:
                    break
                # Find the matching closing paren for this call
                start = m.end()
                depth = 1
                i = start
                while i < len(out) and depth > 0:
                    if out[i] == "(":
                        depth += 1
                    elif out[i] == ")":
                        depth -= 1
                    i += 1
                if depth != 0:
                    break  # malformed; bail out
                out = out[:m.start()] + " UNITLESS " + out[i:]
            return out

        def has_unitful_token(s: str) -> bool:
            scrubbed = _strip_xsection_wraps(s)
            return any(re.search(rf"\b{tok}\b", scrubbed) for tok in unitful)

        # Parenthesized top-level additions/subtractions with a scalar side.
        for m in re.finditer(r"\(([^()]*?(?:\([^()]*\)[^()]*)*)\)", expression):
            inner = m.group(1)
            depth = 0
            for idx, c in enumerate(inner):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif c in "+-" and depth == 0:
                    left = inner[:idx].strip()
                    right = inner[idx + 1:].strip()
                    if has_unitful_token(left) and re.fullmatch(r"\d+(?:\.\d+)?", right):
                        return m.group(0)
                    if has_unitful_token(right) and re.fullmatch(r"\d+(?:\.\d+)?", left):
                        return m.group(0)

        # Also catch non-parenthesized field+number forms.
        for tok in unitful:
            m = re.search(rf"\b{tok}\b\s*[+-]\s*\d+(?:\.\d+)?", expression)
            if m:
                return m.group(0)
            m = re.search(rf"\d+(?:\.\d+)?\s*[+-]\s*\b{tok}\b", expression)
            if m:
                return m.group(0)
        return None

    # ------------------------------------------------------------ raw heuristics
    def _inside_correlation_op(self, expression: str, needle: str) -> bool:
        """True if `needle` appears only inside ts_corr() or ts_covariance() args."""
        # Strip ts_corr/ts_covariance subtrees; if needle disappears it was inside.
        scrubbed = re.sub(r"ts_(corr|covariance)\([^()]*(?:\([^()]*\)[^()]*)*\)", "", expression)
        return needle not in _norm(scrubbed)

    def _has_raw_volume_primary(self, expr: str) -> bool:
        if "volume" not in expr:
            return False
        # Volume inside ts_corr/ts_covariance is fine
        scrubbed = re.sub(r"ts_(corr|covariance)\([^)]*\)", "", expr)
        if "volume" not in scrubbed:
            return False
        # Volume is "normalized" if it appears divided by a longer-window volume aggregate.
        # Do not add scalar epsilons to unitful volume denominators under unitHandling=VERIFY.
        has_norm = (
            bool(re.search(r"/(ts_mean|ts_sum|adv\d+)\(volume", scrubbed))
            or bool(re.search(r"volume/(ts_mean|ts_sum)\(volume", scrubbed))
            or bool(re.search(r"ts_sum\(volume,\d+\)/ts_(mean|sum)\(volume,\d+\)", scrubbed))
        )
        return not has_norm

    def _has_raw_close_delta(self, expr: str) -> bool:
        if "ts_delta(close" not in expr:
            return False
        # Accept raw unit-consistent `/close` / `/ts_mean(close,N)`.
        normalized = (
            bool(re.search(r"ts_delta\(close,\d+\)/(close|ts_mean\(close|ts_sum\(close)", expr))
        )
        return not normalized

    def _is_static_long_std(self, expr: str) -> bool:
        m = re.search(r"ts_std_dev\(returns,(\d+)\)", expr)
        if not m:
            return False
        window = int(m.group(1))
        if window < 200:
            return False
        other_ts = [t for t in re.findall(r"ts_[a-z_]+\(", expr) if t != "ts_std_dev("]
        return len(other_ts) == 0

    # ----------------------------------------------------------- turnover model
    def predict_turnover(self, expression: str) -> float:
        expr = _norm(expression)
        if self._is_static_long_std(expr):
            return 2.0
        # ts_arg_max / ts_arg_min are heavy on turnover — the position of the max
        # within a short window swings day-to-day. Check BEFORE other heuristics.
        m_arg = re.search(r"ts_arg_(?:max|min)\([^,]+,(\d+)\)", expr)
        # Hyper-fast intraday legs: (close-open), (high-low), (close-vwap) without
        # a normalising window are the empirical >70% turnover killers (validated
        # from feedback: intraday_gap_reversal_quality at 81.87%, sales_efficiency_
        # gap_reversal at 113.63%). Catch them BEFORE the gentler vwap base.
        short_delta_match = re.search(r"ts_delta\(close,([1-5])\)", expr)
        if m_arg:
            w = int(m_arg.group(1))
            # window 5 → ~65%, window 10 → ~50%, window 20 → ~35%
            base = max(25.0, 75.0 - 2.0 * w)
        elif re.search(r"\(close-open\)|\(open-close\)", expr):
            base = 85.0
        elif re.search(r"\(high-low\)|\(low-high\)", expr):
            base = 80.0
        elif re.search(r"\(close-vwap\)|\(vwap-close\)", expr):
            base = 75.0
        elif "ts_delta(close,1)" in expr:
            base = 80.0
        elif short_delta_match:
            # ts_delta(close,2..5): still very fast, 60-75% before smoothing
            w = int(short_delta_match.group(1))
            base = 80.0 - 4.0 * (w - 1)  # 80, 76, 72, 68, 64
        elif re.search(r"ts_corr\(returns,volume,(\d+)\)", expr):
            base = 50.0
        elif "returns*returns*returns" in expr:
            base = 25.0
        elif "vwap" in expr:
            base = 40.0
        elif re.search(r"ts_zscore\([^,]+,(\d+)\)", expr):
            base = 15.0
        elif "ts_std_dev" in expr:
            base = 25.0
        elif "ts_decay_linear" in expr:
            base = 20.0  # decay smooths
        else:
            base = 30.0

        # ts_decay_linear wrapping the OUTER signal cuts turnover materially.
        # Window matters: longer = more smoothing. Apply per-window discount.
        # CAVEAT: if the high-turnover leg ((close-open), short ts_delta, etc.)
        # is NOT itself inside ts_decay_linear, the decay smooths a different
        # leg and the discount doesn't apply to the dominant churn driver.
        decay_windows = [int(m.group(1)) for m in re.finditer(r"ts_decay_linear\([^,]+,(\d+)\)", expr)]
        if decay_windows:
            d = max(decay_windows)
            # d=2 → 0.75x, d=4 → 0.55x, d=6 → 0.45x, d=10 → 0.35x
            discount = max(0.30, 1.0 - 0.075 * d)
            # Check whether the dominant high-turnover pattern is actually smoothed.
            hot_patterns = [
                r"\(close-open\)", r"\(open-close\)",
                r"\(high-low\)", r"\(low-high\)",
                r"\(close-vwap\)", r"\(vwap-close\)",
                r"ts_delta\(close,[1-5]\)",
            ]
            if any(re.search(p, expr) for p in hot_patterns):
                if not self._wraps_pattern(expr, "ts_decay_linear", hot_patterns):
                    # decay smooths a side-leg; the churn leg is still churning.
                    discount = min(1.0, max(discount, 0.90))
            base *= discount
        if re.search(r"ts_sum\(volume,\d+\)/\(?ts_mean\(volume,\d+\)", expr):
            base *= 0.9
        windows = [int(m.group(1)) for m in re.finditer(r"ts_mean\([^,]+,(\d+)\)", expr)]
        if windows and max(windows) >= 60:
            base *= 0.85
        return float(base)

    # ----------------------------------------------------------- fitness model
    def _sharpe_mid(self, archetype: str) -> float:
        meta = self.archetypes.get(archetype, {})
        rng = meta.get("expected_sharpe")
        if not rng or len(rng) < 2:
            return 1.0
        return (float(rng[0]) + float(rng[1])) / 2.0

    def _predict_fitness(self, archetype: str, predicted_turnover_pct: float) -> float:
        """Rough fitness estimate. Fitness = Sharpe * sqrt(|Returns| / max(Turnover, 0.125)).
        Returns is unobservable pre-simulation; we approximate by ~10%/sharpe-unit scaling.

        Sharpe estimate uses empirical median per archetype if we have >=5 samples
        in feedback/memory (the realistic estimate), and a 70/30 blend with the KB
        midpoint (the aspirational ceiling) — this keeps the gate honest while
        still rewarding archetypes that have been outperforming their KB band.
        """
        emp = self.empirical_sharpe.get(archetype)
        kb_mid = self._sharpe_mid(archetype)
        if emp is not None:
            sharpe = 0.70 * emp + 0.30 * kb_mid
        else:
            sharpe = kb_mid
        # |Returns| scales with sharpe but with a floor — even a sharpe=0 alpha
        # has some return magnitude. Cap to keep the prediction from blowing up
        # on the rare high-sharpe outlier.
        approx_returns = max(0.05, min(0.25, 0.08 * max(sharpe, 0.5)))
        turnover_frac = max(predicted_turnover_pct / 100.0, 0.125)
        return sharpe * (approx_returns / turnover_frac) ** 0.5

    def _is_discrete_low_resolution(self, expression: str, settings: dict | None) -> bool:
        m = re.search(r"ts_arg_(?:max|min)\([^,]+,\s*(\d+)\s*\)", expression)
        if not m:
            return False
        window = int(m.group(1))
        if window >= 10:
            return False
        if "ts_decay_linear" in expression:
            return False
        if settings and int(settings.get("decay", 0)) >= 6:
            return False
        return True

    # ----------------------------------------------------------- archetype
    def predict_archetype(self, expression: str) -> str:
        expr = _norm(expression)
        best = "novel"
        best_hits = 0
        for name, meta in self.archetypes.items():
            hits = sum(1 for sig in meta.get("signature", []) if sig in expr)
            if hits > best_hits:
                best_hits = hits
                best = name
        return best

    def _sharpe_range(self, archetype: str) -> str:
        meta = self.archetypes.get(archetype, {})
        rng = meta.get("expected_sharpe")
        if not rng:
            return "unknown"
        return f"{rng[0]:.1f}-{rng[1]:.1f}"

    # ----------------------------------------------------------- settings
    def auto_settings(
        self,
        expression: str,
        memory: list[dict] | None = None,
        feedback: list[dict] | None = None,
        hypothesis: dict | None = None,
    ) -> dict:
        """Per-alpha settings tuner. Wraps the archetype-level baseline with three
        per-expression refinements:

          1. **Predicted-turnover → decay.** If predict_turnover(expr) lands above
             the [10%, 70%] band, pre-emptively bump decay so the actual simulation
             lands in band. If it lands too low, reduce decay.
          2. **Universe is pinned to TOP3000.** Never switched to a narrower
             universe, regardless of field-category mix.
          3. **Hypothesis mechanism → neutralization.** Fundamental claims default
             to Subindustry, behavioral to Sector, microstructure/risk_premium to
             Market — unless group_neutralize() is inside the expression, in which
             case _enforce() forces None (the expression self-neutralizes; a second
             portfolio-level pass would collapse the within-group return spread).
        """
        archetype = self.predict_archetype(expression)
        # Phase 1: archetype baseline + memory + feedback tuning (the old auto_settings logic)
        base = dict(self.defaults)
        if memory:
            same = [m for m in memory if m.get("archetype") == archetype and m.get("sharpe", 0) > 1.5]
            if len(same) >= 3:
                best = max(same, key=lambda m: m.get("sharpe", 0))
                if "settings" in best:
                    base.update(best["settings"])
        if "decay" not in base or base.get("universe") is None:
            arch_meta = self.archetypes.get(archetype, {})
            base.update(arch_meta.get("settings", {}))
        base = self._apply_feedback_tuning(base, archetype, feedback)

        # Phase 2: per-alpha smart tuning based on the expression itself
        base = self._smart_tune_per_alpha(expression, base, hypothesis)
        return self._enforce(expression, base)

    def settings_variants(
        self,
        expression: str,
        base_settings: dict,
        max_variants: int = 4,
    ) -> list[tuple[str, dict]]:
        """Return a small grid of settings configs to sweep for ONE alpha, ordered
        by expected fitness lift. The pipeline simulates each and keeps the best.

        WHY: fitness = sqrt(|returns|/max(turnover,0.125)) * sharpe is the binding
        submission gate (empirically only 12/1539 sims cleared fitness>=1.0 while
        223 cleared sharpe>=1.0). The biggest documented fitness levers are:
          - neutralization: Market preserves more return than Subindustry → higher
            fitness (slightly lower sharpe). Only meaningful when the expression
            does NOT already self-neutralize via group_neutralize() (which _enforce
            pins to None — letting the expression neutralize itself without a
            redundant portfolio-level pass on top).
          - truncation: a higher per-name cap concentrates weight → higher returns
            → higher fitness (still under BRAIN's 10% weight-concentration gate).
          - decay: more smoothing lowers turnover → raises the fitness denominator.

        (Universe is pinned to TOP3000 — never swept as a lever.)
        The first element is always the unchanged base config.
        """
        base = self._enforce(expression, dict(base_settings))
        variants: list[tuple[str, dict]] = [("base", base)]
        has_gn = "group_neutralize(" in _norm(expression)

        # Lever 1 — higher truncation (more concentration → more returns → fitness)
        v = dict(base)
        v["truncation"] = 0.10
        variants.append(("hi_trunc", self._enforce(expression, v)))

        # Lever 2 — more decay (lower turnover → higher fitness denominator)
        v = dict(base)
        v["decay"] = min(20, int(base.get("decay", 4)) + 4)
        variants.append(("hi_decay", self._enforce(expression, v)))

        # Lever 3 — neutralization variant (always Subindustry as the alternative)
        if not has_gn:
            v = dict(base)
            v["neutralization"] = "Subindustry"
            variants.append(("neut_subindustry", self._enforce(expression, v)))

        # De-duplicate identical configs while preserving order.
        seen: set[str] = set()
        uniq: list[tuple[str, dict]] = []
        for label, s in variants:
            key = json.dumps(s, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((label, s))
        return uniq[:max_variants]

    def _smart_tune_per_alpha(
        self, expression: str, settings: dict, hypothesis: dict | None = None
    ) -> dict:
        """The new per-alpha layer: read the expression's structure + the LLM's
        hypothesis (if any) and adapt the settings accordingly."""

        # ----- 1. Predicted turnover → decay -----
        # Targets: turnover must land in [1%, 70%]. The fitness gate (>=1.0) is
        # MUCH harder to clear than the sharpe gate (>=1.25): of 246 recent OK
        # sims, 49 cleared sharpe>=1.0 but failed fitness<1.0, with median
        # turnover 31%. Required turnover to clear fitness is typically 10-20%
        # for those sharpe-likely alphas. So this layer is aggressively biased
        # toward MORE smoothing — even in the "moderate" 20-35% predicted band.
        # The old `< 12% → -2 decay` rule was retired: predict_turnover() is
        # consistently optimistic for some patterns (predicted 11%, actual 30%+),
        # so decreasing decay based on a low prediction frequently hurt the
        # alphas it was trying to help. The new low band is < 5% only — there
        # the alpha really is daily-static and needs more variation.
        pred_turnover = self.predict_turnover(expression)
        decay = int(settings.get("decay", 4))
        if pred_turnover > 80:
            settings["decay"] = max(decay, 12)
            settings["_smart_tune_reason"] = f"predicted_turnover={pred_turnover:.0f}% > 80 → decay=12 (cap)"
        elif pred_turnover > 65:
            settings["decay"] = max(decay, 10)
            settings["_smart_tune_reason"] = f"predicted_turnover={pred_turnover:.0f}% > 65 → decay>=10"
        elif pred_turnover > 50:
            settings["decay"] = max(decay, 8)
            settings["_smart_tune_reason"] = f"predicted_turnover={pred_turnover:.0f}% > 50 → decay>=8"
        elif pred_turnover > 35:
            # Reverted to +2 from the over-aggressive decay>=8. The aggressive
            # tune did not produce a measurable lift in the empirical data
            # (median decay stayed at 4 because the LLM responded to the
            # parallel prompt change by writing lower-turnover expressions —
            # so this band fired less often). Keeping the conservative bump.
            settings["decay"] = min(10, max(decay, decay + 2))
            settings["_smart_tune_reason"] = f"predicted_turnover={pred_turnover:.0f}% > 35 → +2 decay"
        elif pred_turnover < 5:
            # Narrower than the old `<12` band. Only fires for genuinely static
            # signals (e.g. quarterly fundamentals with no time component).
            settings["decay"] = max(0, decay - 2)
            settings["_smart_tune_reason"] = f"predicted_turnover={pred_turnover:.0f}% < 5 → -2 decay"

        # ----- 2. Universe is pinned to TOP3000 -----
        # We used to flip pure-fundamental alphas to TOP1000 for denser quarterly
        # coverage, but the universe is now fixed at TOP3000 everywhere, so we
        # leave it untouched here.
        # NOTE: we also used to force delay=1 for pure-fundamental alphas under the
        # assumption that "fundamentals only go T+1." Verified against real D0
        # submissions (e.g. -rank(ebit/capex) @ delay=0 hit Sharpe 2.02 / Fitness
        # 2.30 / Turnover 2.36%), that assumption is wrong: fundamental ratios
        # CAN run at delay=0 and benefit from immediate execution. We now leave
        # the LLM's chosen delay alone — the prompt teaches the decision rules.

        # ----- 2b. D0-aware decay floor -----
        # If the LLM chose delay=0, the signal is being read at today's close.
        # Real D0 passing alphas observed in the wild use decay across the full
        # 0-15 range (e.g. ebit/capex @ decay=0, vwap-reversion @ decay=15).
        # We DON'T lift decay for D0 here — the predicted-turnover branch above
        # already handles that. We only ensure D0 alphas with a hypothesis that
        # explicitly says "intraday" or "same-day" don't get over-smoothed by
        # the >35% turnover branch above (which would dampen their edge).
        # The smoothing branches already cap at decay=12, so no change needed.
        # Keeping this block as a documentation marker for the D0 invariant.

        # ----- 3. Mechanism / field-mix → neutralization -----
        # Pick the level that strips the factor the alpha is NOT trying to bet on,
        # so the residual is pure alpha. Uses the hypothesis mechanism when given,
        # else infers from the expression's own field mix — so we make a SMART
        # choice even when the LLM gave no mechanism (instead of defaulting to the
        # KB's blanket "Market"). (Skipped when the expression self-neutralizes via
        # group_neutralize — _enforce() pins None there.)
        if "group_neutralize(" not in _norm(expression):
            settings["neutralization"] = self._smart_neutralization(expression, hypothesis)

        # ----- 4. Concentration-prone structures → tighter truncation -----
        if self._has_concentration_risk(expression):
            current_trunc = float(settings.get("truncation", 0.05))
            settings["truncation"] = min(current_trunc, 0.03)

        return settings

    def _smart_neutralization(self, expression: str, hypothesis: dict | None = None) -> str:
        """Choose the cross-sectional neutralization level intelligently.

        Principle: neutralize at the granularity where the *unwanted* common factor
        lives, so what's left is the alpha. Fundamental ratios cluster by
        sub-industry (peers with similar margins/leverage), behavioral & alt-data
        signals cluster by sector (sentiment/flow moves sector-wide), broad
        price/microstructure effects are market-wide.

          1. Trust an explicit hypothesis mechanism when the LLM gave one.
          2. Otherwise INFER from the expression's field mix (so we still make a
             smart choice instead of falling back to the KB's blanket 'Market').
          3. Default to 'Market' for mixed/unclassifiable signals — it preserves
             the most return, which is the binding fitness lever.
        """
        if isinstance(hypothesis, dict):
            mech = (hypothesis.get("mechanism") or "").lower()
            if "fundamental" in mech:
                return "Subindustry"
            if "behavioral" in mech:
                return "Subindustry"
            if any(k in mech for k in ("microstructure", "risk_premium", "informational")):
                return "Subindustry"
        # No usable mechanism — infer from what the expression actually trades on.
        fundamental, price, alt = self._field_category_hits(expression)
        if fundamental and fundamental >= price and fundamental >= alt:
            return "Subindustry"
        if alt and alt >= price and alt >= fundamental:
            return "Subindustry"
        return "Subindustry"

    def _field_category_hits(self, expression: str) -> tuple[int, int, int]:
        """Count how many fields used in expression fall into:
        (fundamental, price/volume, alt-data) buckets, using the catalog metadata."""
        if not getattr(self, "catalog_units", None):
            return (0, 0, 0)
        fundamental = price = alt = 0
        seen: set[str] = set()
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z_0-9]*)\b", expression):
            tok = m.group(1)
            if tok in seen or expression[m.end():m.end() + 1] == "(":
                continue
            seen.add(tok)
            bucket = self.catalog_units.get(tok)
            if not bucket or bucket == "unknown":
                # Also try the canonical sets — close/vwap/returns aren't in catalog_units
                if tok in _PRICE_FIELDS or tok in _VOLUME_FIELDS:
                    price += 1
                continue
            if bucket == "fundamental":
                fundamental += 1
            elif bucket in ("price", "volume"):
                price += 1
            elif bucket == "unitless":
                alt += 1  # options vol, sentiment, news, model scores all surface here
        return fundamental, price, alt

    @staticmethod
    def _has_concentration_risk(expression: str) -> bool:
        """Single-field reliance on extreme tails — likely to concentrate weight."""
        expr = _norm(expression)
        # ts_arg_max/ts_arg_min on short windows produce coarse signals that
        # concentrate weight. zscore with no group neutralization on a single
        # field with heavy-tail distribution likewise concentrates.
        if re.search(r"ts_arg_(?:max|min)\([^,]+,\s*[0-9]\s*\)", expr):
            return True
        if re.search(r"signed_power\([^)]+,\s*[3-9]", expr):
            return True
        return False

    @staticmethod
    def _apply_feedback_tuning(settings: dict, archetype: str, feedback: list[dict] | None) -> dict:
        """Small adaptive settings nudges from recent failures.

        Goal: improve pass-rate by reacting to systematic misses (mainly turnover band).
        """
        if not feedback:
            return settings

        same = [f for f in feedback if f.get("archetype") == archetype]
        if not same:
            return settings

        # Focus on recent, simulated outcomes where turnover is known.
        recent = [
            f for f in same[-30:]
            if f.get("status") == "OK" and isinstance(f.get("turnover_pct"), (int, float))
        ]
        if not recent:
            return settings

        high_turn = sum(1 for f in recent if float(f["turnover_pct"]) > 70.0)
        low_turn = sum(1 for f in recent if float(f["turnover_pct"]) < 1.0)
        ratio_high = high_turn / len(recent)
        ratio_low = low_turn / len(recent)

        decay = int(settings.get("decay", 4))
        if ratio_high >= 0.40:
            settings["decay"] = min(12, decay + 2)
        elif ratio_low >= 0.40:
            settings["decay"] = max(0, decay - 1)

        # Concentration fails: lower truncation (more diversification).
        conc_fails = sum(
            1 for f in same[-30:]
            if isinstance(f.get("reason"), str) and "concentrated_weight" in f["reason"].lower()
        )
        if conc_fails >= 2:
            t = float(settings.get("truncation", 0.05))
            settings["truncation"] = max(0.01, min(t, 0.05))
        return settings

    def _enforce(self, expression: str, settings: dict) -> dict:
        expr = _norm(expression)
        if "group_neutralize(" in expr:
            # The expression already performs its own cross-sectional neutralization
            # inside group_neutralize(). A portfolio-level neutralization setting of
            # "Market" would apply a SECOND pass on top, collapsing the within-group
            # spread and destroying returns. Set "None" so Brain honours the expression's
            # own neutralization and does not re-neutralize at the portfolio level.
            settings["neutralization"] = "None"

        # Keep settings inside legal/sane ranges used by Brain.
        try:
            settings["decay"] = max(0, int(settings.get("decay", 4)))
        except (TypeError, ValueError):
            settings["decay"] = 4
        try:
            settings["delay"] = max(0, min(1, int(settings.get("delay", 1))))
        except (TypeError, ValueError):
            settings["delay"] = 1
        try:
            settings["truncation"] = max(0.0, min(1.0, float(settings.get("truncation", 0.05))))
        except (TypeError, ValueError):
            settings["truncation"] = 0.05

        neut = str(settings.get("neutralization", "Subindustry"))
        valid_neut = {"Market", "None", "Sector", "Industry", "Subindustry"}
        settings["neutralization"] = neut if neut in valid_neut else "Subindustry"

        # Universe is pinned to TOP3000 everywhere — never use a narrower universe.
        settings["universe"] = "TOP3000"
        return settings

    # ----------------------------------------------------------- signature & dedup
    def signature(self, expression: str) -> dict:
        expr = _norm(expression)
        call_names = set(_extract_call_names(expression))
        ops = sorted(call_names)
        # Capture ALL bare identifiers (not followed by '('), so the dedup sees
        # catalog fields like gross_profit_to_assets_ratio, historical_volatility_60,
        # trailing_twelve_month_accruals — not just the canonical seven. Skip
        # reserved keywords / numbers / group names that aren't real fields.
        _STOPWORDS = {"if", "else", "and", "or", "not", "true", "false", "None"}
        fields_raw: set[str] = set()
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z_0-9]*)\b", expression):
            tok = m.group(1)
            if tok in call_names or tok in _STOPWORDS:
                continue
            # Not a function call (no '(' immediately after) → treat as field/identifier
            if expression[m.end():m.end() + 1] == "(":
                continue
            fields_raw.add(tok)
        fields = sorted(fields_raw)
        windows = sorted(set(int(m.group(1)) for m in re.finditer(r",(\d+)\)", expr)))
        return {
            "ops": ops,
            "fields": fields,
            "windows": windows,
            "archetype": self.predict_archetype(expression),
            "normalized": expr,  # for exact-string-match pre-filter
        }

    def similarity(self, sig_a: dict, sig_b: dict) -> float:
        # Exact-match fast path — catches literal duplicates regardless of how
        # other components score. Compares normalized expressions (whitespace stripped).
        norm_a, norm_b = sig_a.get("normalized"), sig_b.get("normalized")
        if norm_a and norm_b and norm_a == norm_b:
            return 1.0

        def jac(a: Iterable, b: Iterable) -> float:
            sa, sb = set(a), set(b)
            if not sa and not sb:
                # Empty-set match should NOT count as full similarity — that's how
                # catalog-field alphas used to slip through (sig had no canonical fields).
                return 0.0
            return len(sa & sb) / max(1, len(sa | sb))
        ops_j = jac(sig_a.get("ops", []), sig_b.get("ops", []))
        fields_j = jac(sig_a.get("fields", []), sig_b.get("fields", []))
        windows_j = jac(sig_a.get("windows", []), sig_b.get("windows", []))
        arch_match = 1.0 if sig_a.get("archetype") == sig_b.get("archetype") else 0.0
        # Field overlap weighted higher than ops — same-ops different-fields is a
        # genuinely different alpha; same-fields different-ops is more often a near-dupe.
        return 0.30 * ops_j + 0.45 * fields_j + 0.15 * windows_j + 0.10 * arch_match

    # ----------------------------------------------------------- fix synthesis
    def _propose_fix(self, expression: str, issues: list[Issue]) -> str | None:
        for i in issues:
            if i.verdict == FAIL and i.fix_hint:
                return i.fix_hint
        for i in issues:
            if i.verdict == WARN and i.fix_hint:
                return i.fix_hint
        return None


# ------------------------------------------------------------------ validation
# ════════════════════════════════════════════════════════════════════════════
# LOCAL AST VALIDATOR — WorldQuant Brain grammar pre-screen
# ════════════════════════════════════════════════════════════════════════════
# A fast, dependency-free structural gate that runs BEFORE the heavier critique()
# layer and before any BRAIN API call. It parses the candidate expression with
# Python's native `ast` module and walks the tree, rejecting expressions that
# violate Brain's operator grammar so the generation loop can retry locally
# instead of burning an API simulation slot on a guaranteed-invalid formula.
#
# Operator policy is whitelist-driven: an operator is legal iff it appears in
# data/brain_docs/operators.json (the catalog fetched from Brain). On top of that
# whitelist we enforce the argument SHAPE of the dictionary operators below.
# NOTE on zscore/normalize: both ARE valid Brain operators (present in
# operators.json) and are used by passing alphas, so they are deliberately NOT
# banned here. Only genuinely-absent hallucinations (standardize, winsorize via
# the whitelist, etc.) are rejected.
logger = logging.getLogger("alphaquant.math_engine")

# Time-series operators that REQUIRE an explicit positive-integer lookback as
# their last argument: ts_op(x, d). Exactly 2 args; d must be an int literal > 0.
_TS_LOOKBACK_OPS = {
    "ts_decay_linear", "ts_mean", "ts_std_dev", "ts_zscore",
    "ts_rank", "ts_delta", "ts_delay", "ts_skewness",
}
# Cross-sectional group operators: group_op(x, g). Exactly 2 args; g must be one
# of the literal grouping tokens below. The spec lists subindustry/industry/sector;
# `market` is added because Brain exposes it and passing alphas in memory.json use
# it (e.g. a Sharpe-1.67 composite) — restricting to the spec's three would refuse
# valid expressions.
_GROUP_OPS = {"group_neutralize", "group_zscore", "group_rank"}
_GROUP_LITERALS = {"subindustry", "industry", "sector", "market"}
# Structural math functions with a fixed arity.
_FIXED_ARITY_OPS = {"rank": 1, "scale": 1, "signed_power": 2, "if_else": 3}
# Max rank() factors allowed in a single multiplicative chain before it's treated
# as overfitting. The spec's intent is to stop runaway rank-stacking; empirically
# the 3-rank group_neutralize composite is this codebase's dominant WINNING
# structure (Sharpe up to 2.11 at 17-40% turnover, never >55%), so the block fires
# only at 4+ — the genuinely over-engineered chains (0/25 winners use them).
_MAX_RANK_FACTORS = 3
# Operators the LLM hallucinates that do NOT exist on the current Brain profile,
# regardless of whitelist state — flagged with a tailored fix hint. zscore and
# normalize are intentionally absent (Brain exposes them).
_HALLUCINATED_OPS = {
    "standardize": "Brain has no standardize(); use ts_zscore(x, d) or group_zscore(x, g)",
}


def _ast_is_positive_int(node: ast.AST) -> bool:
    """True only for an explicit positive integer literal (e.g. 21 — not -5, 5.0, or `d`)."""
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and node.value > 0
    )


def _ast_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _ast_call_name(node: ast.AST) -> str | None:
    """Resolve the callable name of an ast.Call's func, or None if it isn't a simple call."""
    func = getattr(node, "func", None)
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):   # e.g. close.shift(1) -> "shift"
        return func.attr
    return None


def _ast_mult_factors(node: ast.AST) -> list[ast.AST]:
    """Flatten a nested chain of `*` into its leaf factors."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _ast_mult_factors(node.left) + _ast_mult_factors(node.right)
    return [node]


class _BrainGrammarValidator(ast.NodeVisitor):
    """Walks the AST once, accumulating structural error strings in `self.errors`."""

    def __init__(
        self,
        whitelist: set[str],
        arity: dict[str, tuple[int, int]] | None = None,
        known_fields: set[str] | None = None,
    ) -> None:
        self.whitelist = whitelist
        self.arity = arity or {}
        # Field-existence checking is enabled only when the catalog is substantial
        # (>1000 fields). A small/empty set means the catalog wasn't fetched, so we
        # disable the check rather than flag every field as unknown.
        self.known_fields = known_fields if (known_fields and len(known_fields) > 1000) else set()
        self.errors: list[str] = []
        self._parents: dict[int, ast.AST] = {}

    # -- bare identifiers (field references) ----------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        if not self.known_fields:
            return  # catalog not loaded — checking disabled
        parent = self._parents.get(id(node))
        # Skip operator names (the func position of a call, e.g. `rank` in rank(x)).
        if isinstance(parent, ast.Call) and parent.func is node:
            return
        nm = node.id
        if nm in _GROUP_LITERALS or nm in RESERVED_CALLABLES or nm in self.whitelist:
            return
        if nm not in self.known_fields:
            self.errors.append(
                f"unknown field '{nm}' — not in the BRAIN field catalog (fields_full.json); "
                f"likely hallucinated. Use a catalog field id."
            )

    # -- function calls -------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        name = _ast_call_name(node)
        if name is None:
            self.errors.append(
                f"call on a non-identifier ('{_ast_unparse(node.func)}') is not valid Brain syntax"
            )
            self.generic_visit(node)
            return

        # 1. Known hallucinations (standardize, ...) — tailored message.
        if name in _HALLUCINATED_OPS:
            self.errors.append(f"banned operator '{name}()': {_HALLUCINATED_OPS[name]}")
        # 2. Whitelist membership (data-driven from operators.json). Only enforced
        #    when the whitelist actually loaded; an empty set means the docs are
        #    missing, so we skip the check rather than reject every expression.
        elif self.whitelist and name not in self.whitelist and name not in RESERVED_CALLABLES:
            self.errors.append(
                f"unknown operator '{name}()' — not in Brain operators.json whitelist "
                f"(likely hallucinated; remove it or use a documented operator)"
            )

        # 3. Keyword arguments never appear in Brain expressions.
        if node.keywords:
            kw = node.keywords[0].arg or "**kwargs"
            self.errors.append(
                f"operator '{name}()' uses keyword argument '{kw}=' — "
                f"Brain operators take positional arguments only"
            )

        # 4. Argument-shape grammar for the dictionary operators.
        self._check_arity(name, node)

        self.generic_visit(node)

    def _check_arity(self, name: str, node: ast.Call) -> None:
        nargs = len(node.args)
        if name in _TS_LOOKBACK_OPS:
            if nargs != 2:
                self.errors.append(
                    f"time-series operator '{name}(x, d)' needs exactly 2 arguments; got {nargs}"
                )
                return
            d = node.args[1]
            if not _ast_is_positive_int(d):
                self.errors.append(
                    f"time-series operator '{name}' is missing its integer lookback: argument "
                    f"d='{_ast_unparse(d)}' must be an explicit positive integer (e.g. 21)"
                )
        elif name in _GROUP_OPS:
            if nargs != 2:
                self.errors.append(
                    f"group operator '{name}(x, g)' needs exactly 2 arguments; got {nargs}"
                )
                return
            g = node.args[1]
            token = g.id if isinstance(g, ast.Name) else None
            if token not in _GROUP_LITERALS:
                self.errors.append(
                    f"group operator '{name}' second argument must be one of "
                    f"{sorted(_GROUP_LITERALS)}; got '{_ast_unparse(g)}'"
                )
        elif name in _FIXED_ARITY_OPS:
            need = _FIXED_ARITY_OPS[name]
            if nargs != need:
                self.errors.append(
                    f"operator '{name}()' takes exactly {need} argument(s); got {nargs}"
                )
        elif name in self.arity:
            # General signature-derived arity check (covers ts_corr, ts_covariance,
            # quantile, group_mean, ... — every documented operator not specially
            # handled above). BRAIN rejects wrong arg counts at sim start, so catch
            # them here for free.
            lo, hi = self.arity[name]
            if not (lo <= nargs <= hi):
                want = f"{lo}" if lo == hi else f"{lo}-{hi}"
                self.errors.append(
                    f"operator '{name}()' takes {want} argument(s); got {nargs}"
                )

    # -- binary operators -----------------------------------------------------
    def visit_BinOp(self, node: ast.BinOp) -> None:
        # Overfitting cross-product: more than two rank() factors multiplied.
        if isinstance(node.op, ast.Mult):
            parent = self._parents.get(id(node))
            outermost = not (isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Mult))
            if outermost:
                factors = _ast_mult_factors(node)
                n_rank = sum(
                    1 for f in factors
                    if isinstance(f, ast.Call) and _ast_call_name(f) == "rank"
                )
                if n_rank > _MAX_RANK_FACTORS:
                    self.errors.append(
                        f"overfitting/turnover risk: {n_rank} rank() factors multiplied together "
                        f"over-constrain the signal distribution and risk turnover blowups — "
                        f"combine at most {_MAX_RANK_FACTORS} rank legs "
                        f"(use add()/a single composite instead of chaining rank*rank*rank*rank)"
                    )
        # Unprotected division: denominator that can hit 0/NaN.
        if isinstance(node.op, ast.Div):
            risk = self._unstable_denominator(node.right)
            if risk:
                self.errors.append(
                    f"liquidity risk: division denominator '{_ast_unparse(node.right)}' {risk} — "
                    f"it can evaluate to 0/NaN and ruin the daily alpha vector; divide by a stable "
                    f"baseline (e.g. ts_mean(...) / ts_std_dev(...) / abs(...)) rather than a raw difference"
                )
        self.generic_visit(node)

    @staticmethod
    def _unstable_denominator(denom: ast.AST) -> str | None:
        """Return a reason string if `denom` is a zero-crossing difference, else None.

        Scope is deliberately narrow: only genuine NaN/zero risks (a subtraction or
        a ts_delta(), both of which routinely cross zero). Raw single price/volume
        fields and plain ts_delay(field, d) are NOT flagged — they are effectively
        non-zero for tradable instruments, AND Brain's unitHandling=VERIFY forbids
        the '+epsilon' stabilizer the naive guard would add (see
        MathEngine._find_unitful_numeric_add), so flagging them would create an
        unwinnable local-retry loop."""
        node = denom
        # peel a leading unary +/- so -(a - b) is still seen as a difference
        while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            node = node.operand
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            return "is a raw difference (a - b)"
        if isinstance(node, ast.Call) and _ast_call_name(node) == "ts_delta":
            return "is a ts_delta() difference"
        return None


def validate_formula_structure(formula_str: str) -> list[str]:
    """Local AST pre-screen of a Brain alpha expression against the BRAIN grammar.

    Returns a list of structural error strings; an EMPTY list means the formula
    passes local verification and is safe to hand to critique()/BRAIN. This is the
    cheap first gate in alpha_generator's quota-guarantee loop — it catches
    hallucinated operators, missing lookback windows, illegal group tokens,
    rank-chain overfitting, and unguarded divisions WITHOUT spending an API call.

    Detailed exception logging wraps the parser so an unparseable expression
    degrades into a single, readable error string instead of crashing the loop.
    """
    if not isinstance(formula_str, str) or not formula_str.strip():
        return ["empty or non-string expression"]

    # Brain uses '^' for exponentiation; Python's ast reads bare '^' as bitwise-xor.
    # Translate to '**' so the intent parses as a Pow node rather than a BitXor.
    prepared = formula_str.replace("^", "**").strip().rstrip(";").strip()

    try:
        tree = ast.parse(prepared, mode="exec")
    except SyntaxError as ex:
        logger.warning(
            "validate_formula_structure: SyntaxError on %r: %s (line %s, col %s)",
            prepared, ex.msg, ex.lineno, ex.offset,
        )
        detail = ex.msg + (f" near col {ex.offset}" if ex.offset else "")
        return [f"unparseable expression (SyntaxError: {detail})"]
    except (ValueError, MemoryError, RecursionError) as ex:
        logger.error(
            "validate_formula_structure: %s on %r: %s", type(ex).__name__, prepared, ex
        )
        return [f"unparseable expression ({type(ex).__name__}: {ex})"]
    except Exception as ex:  # never let the validator take down the generation loop
        logger.exception("validate_formula_structure: unexpected parse error on %r", prepared)
        return [f"unparseable expression ({type(ex).__name__}: {ex})"]

    validator = _BrainGrammarValidator(
        _load_known_operators(), _operator_arity_cached(), _field_ids_cached()
    )
    # Map parents first so the Mult-chain check fires only on the outermost node.
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            validator._parents[id(child)] = parent
    try:
        validator.visit(tree)
    except Exception as ex:  # defensive — a malformed node shouldn't crash the loop
        logger.exception("validate_formula_structure: traversal error on %r", prepared)
        return [f"validation traversal error ({type(ex).__name__}: {ex})"]

    # De-duplicate while preserving order (a rule can fire on repeated subtrees).
    seen: set[str] = set()
    unique: list[str] = []
    for e in validator.errors:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique


def validate_against_brain_docs() -> list[str]:
    """Cross-check KB operators against operators.json. Returns list of issues."""
    issues: list[str] = []
    kb = _load_kb()
    if not OPS_PATH.exists():
        return ["data/brain_docs/operators.json missing"]
    ops_doc = json.loads(OPS_PATH.read_text())
    known = {k for k in ops_doc.keys() if not k.startswith("_")}

    referenced: set[str] = set()
    for alpha in kb.get("confirmed_working", []) + kb.get("confirmed_failing", []):
        expr = alpha.get("expression", "")
        for name in _extract_call_names(expr):
            if name not in RESERVED_CALLABLES:
                referenced.add(name)

    forbidden = set(kb.get("forbidden_operators", []))
    for op in sorted(referenced):
        if op in forbidden:
            continue
        if op not in known:
            issues.append(f"operator '{op}' referenced in brain_kb.json but missing from operators.json")
    return issues


if __name__ == "__main__":
    me = MathEngine()
    # Original 5 spec cases: heuristic FAILs are now WARN (still flagged, not blocked).
    # Genuine syntax errors stay FAIL.
    cases = [
        # (expression, expected_verdict, note)
        ("scale(-1*rank(ts_std_dev(returns,252)))", WARN, "static long-window std (heuristic, now WARN)"),
        ("scale(zscore(-ts_mean(returns*returns,21)))", WARN, "quadratic returns (heuristic, now WARN)"),
        ("scale(rank(ts_sum(volume,5)))", WARN, "raw volume primary (heuristic, now WARN)"),
        # Lottery+volume with unit-consistent division
        ("scale(group_neutralize(zscore(-ts_mean(returns*returns*returns,21))*rank(ts_sum(volume,5)/ts_mean(volume,60)),sector))", WARN, "Sharpe-2.99 lottery+volume — unit-consistent ratio"),
        # Additive scalar guards on unitful denominators are invalid under Brain VERIFY
        ("scale(group_neutralize(zscore(-ts_mean(returns*returns*returns,21))*rank(ts_sum(volume,5)/(ts_mean(volume,60)+1)),sector))", FAIL, "lottery+volume with invalid scalar guard"),
        # VWAP premium with unit-consistent division
        ("scale(zscore(-(close-vwap)/vwap))", WARN, "VWAP premium unit-consistent"),
        # 101-Formulaic-Alphas style
        ("-1*rank(ts_corr(returns,volume,10))", PASS, "no scale() needed — confirmed Sharpe 2.29 alpha"),
        ("rank(ts_arg_max(signed_power(if_else(less(returns,0),ts_std_dev(returns,20),close),2),5))-0.5", WARN, "broken alpha101_001 — UNITS mismatch + discrete signal"),
        ("rank(ts_decay_linear(ts_arg_max(signed_power(if_else(returns<0,ts_std_dev(returns,20),abs(returns)),2),10),5))-0.5", PASS, "alpha101_001 fixed — units OK, smoothed"),
        ("rank(ts_decay_linear(-ts_delta(close,1)/ts_mean(close,252),7))", WARN, "reversal with unit-consistent division"),
        ("ts_max(close,5)/ts_mean(close,60)-1", FAIL, "ts_max is forbidden in current Brain profile"),
        # Forbidden field — enterprise_value, cap, adv20
        ("-ts_zscore(enterprise_value/ebitda,63)", FAIL, "enterprise_value is not a Brain field"),
        ("rank(cap)", FAIL, "cap is not a Brain field"),
        ("rank(ts_sum(volume,5)/adv20)", FAIL, "adv20 is not a Brain field"),
        # Forbidden operator — winsorize
        ("winsorize(returns)", FAIL, "winsorize() does not exist on Brain"),
        # Unguarded division WARN
        ("close/vwap", WARN, "unguarded division by vwap"),
        # Genuine syntax/semantic errors stay FAIL
        ("SMA(3,close)-SMA(7,close)", FAIL, "TA-Lib SMA (real syntax error)"),
        ("scale(ts_mean(returns))", FAIL, "missing day arg (real syntax error)"),
        ("scale(zscore(returns, 21))", FAIL, "zscore multi-arg (real syntax error)"),
        ("scale(zscore(+ts_mean(returns*returns*returns,21)))", FAIL, "positive cubic (empirical, kept FAIL)"),
    ]
    print(f"{'verdict':>7s}  expected  case")
    print("-" * 100)
    fails = 0
    for expr, expected, note in cases:
        result = me.critique(expr)
        ok = result["verdict"] == expected
        if not ok:
            fails += 1
        flag = "OK " if ok else "NO "
        print(f"[{flag}] {result['verdict']:>5s} | {expected:>5s} | {note}")
        print(f"        {expr}")
        for issue in result["issues"]:
            print(f"        └─ [{issue['verdict']}] {issue['rule_id']}: {issue['message']}")
        print()
    print(f"\n{'PASS' if fails == 0 else f'FAIL — {fails} mismatch'}")
