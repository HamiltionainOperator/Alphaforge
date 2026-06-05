"""Feature extractor v1 for the Judge NN cascade.

Produces a 28-dim vector from a raw expression string (and optional settings).
The same vector is consumed by NN-1 today and by NN-2/3/4 once they come online.

Versioning: FEATURE_VERSION is stored alongside every trained model and every
dataset row. A model trained on v1 features must not be served v2 features —
the trainer enforces this. Bump the version any time you add, remove, or
change the semantics of a feature.

Reuses MathEngine for operator extraction, archetype prediction, and turnover
prediction so we have one source of truth for expression parsing.
"""
from __future__ import annotations

import re
from typing import Iterable

from math_engine import (
    MathEngine,
    _extract_call_names,
    _norm,
)

FEATURE_VERSION = "v3"

# Field-category sets, kept in sync with data/brain_docs/fields.json.
_PRICE_FIELDS = frozenset({"close", "open", "high", "low", "vwap", "returns"})
_VOLUME_FIELDS = frozenset({"volume"})
_FUNDAMENTAL_FIELDS = frozenset({
    "ebitda", "ebit", "sales", "assets", "equity",
    "liabilities", "debt", "sharesout", "dividends",
})
_GROUP_FIELDS = frozenset({"sector", "subindustry", "market", "industry"})
_ALL_KNOWN_FIELDS = (
    _PRICE_FIELDS | _VOLUME_FIELDS | _FUNDAMENTAL_FIELDS | _GROUP_FIELDS
)

# Reserved keywords that are not fields or operators.
_RESERVED_TOKENS = frozenset({
    "if", "else", "and", "or", "not", "true", "false", "None",
})

# Operator aliases that look like valid functions but aren't in operators.json.
# These came from real failure logs: bare `delay(`, TA-Lib names, ts_ir typo.
_INVALID_OP_ALIASES = frozenset({
    "delay",       # users mean ts_delay; bare `delay` is rejected
    "winsorize",   # not a BRAIN operator
    "SMA", "EMA", "RSI", "MACD", "ATR", "BBANDS", "STOCH",
    "WMA", "TEMA", "DEMA",
    "ts_ir",       # common typo for ts_returns
})

# Full BRAIN field catalog (7,642 fields). Loaded once at module init so
# every extract_features() call can do O(1) lookups.
# - _CATALOG_FIELDS: union of canonical (fields.json) and extended (fields_full.json).
# - _VECTOR_FIELDS: subset typed as VECTOR — these are event-stream fields that
#   ALL fail when passed to standard operators (rank, zscore, ts_*, arithmetic).
#   In our 269-entry dataset every reference to a VECTOR field results in a
#   FAIL or TIMEOUT, with zero OK cases. That makes it the single most
#   predictive new feature.
# - _EXTENDED_CATALOG_FIELDS: in the full catalog but not in the canonical 21.
#   Tracked separately so the model can distinguish "obscure but valid" from
#   "typo" (without this, both look like 'unknown identifier').
def _load_catalog_metadata() -> tuple[frozenset, frozenset, frozenset]:
    """Read data/brain_docs/fields_full.json. Returns (all, vector, extended)."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data" / "brain_docs" / "fields_full.json"
    if not path.exists():
        # Fallback: project might be running without the full catalog fetched.
        return frozenset(_ALL_KNOWN_FIELDS), frozenset(), frozenset()
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return frozenset(_ALL_KNOWN_FIELDS), frozenset(), frozenset()
    fields = doc.get("fields", {}) if isinstance(doc, dict) else {}
    all_ids = set(fields.keys()) | set(_ALL_KNOWN_FIELDS)
    vector_ids = {fid for fid, meta in fields.items() if meta.get("type") == "VECTOR"}
    extended_ids = set(fields.keys()) - _ALL_KNOWN_FIELDS
    return frozenset(all_ids), frozenset(vector_ids), frozenset(extended_ids)


_CATALOG_FIELDS, _VECTOR_FIELDS, _EXTENDED_CATALOG_FIELDS = _load_catalog_metadata()


# Operator groupings used as binary features. Names must match
# data/brain_docs/operators.json.
_TS_CORR_LIKE = frozenset({"ts_corr", "ts_covariance"})
_TS_ARG_EXTREMA = frozenset({"ts_arg_max", "ts_arg_min"})
_TS_EXTREMA = frozenset({"ts_max", "ts_min"})
_GROUP_RANK_LIKE = frozenset({"group_rank", "group_zscore"})
_LOGICAL_OPS = frozenset({"if_else", "less", "greater", "equal", "and", "or", "not"})
_TS_OPS_ALL = frozenset({
    "ts_mean", "ts_sum", "ts_product", "ts_std_dev", "ts_delta", "ts_delay",
    "ts_corr", "ts_covariance", "ts_zscore", "ts_rank", "ts_max", "ts_min",
    "ts_arg_max", "ts_arg_min", "ts_decay_linear", "ts_skewness", "ts_kurtosis",
    "ts_count_nans", "ts_returns",
})
_CS_OPS = frozenset({"rank", "zscore", "normalize", "quantile", "scale"})
_GROUP_OPS = frozenset({
    "group_neutralize", "group_rank", "group_mean",
    "group_zscore", "group_sum", "group_max", "group_min",
})


# Feature names in stable order. The integer position is the column index in
# the feature vector. NEVER reorder — append-only across versions.
FEATURE_NAMES: tuple[str, ...] = (
    # Block A — operator presence (10)
    "has_rank",
    "has_zscore",
    "has_ts_decay_linear",
    "has_ts_corr_like",          # ts_corr OR ts_covariance
    "has_ts_zscore",
    "has_ts_rank",
    "has_ts_arg_extrema",        # ts_arg_max OR ts_arg_min
    "has_ts_extrema",            # ts_max OR ts_min
    "has_if_else",
    "has_group_neutralize",
    "has_group_rank_like",       # group_rank OR group_zscore
    "arithmetic_only",           # no TS, no CS, no group operators
    # Block B — outer-op identity (4, one-hot, "other" implied)
    "outer_is_rank",
    "outer_is_zscore",
    "outer_is_group_neutralize",
    "outer_is_ts_op",
    # Block C — field composition (5)
    "n_price_fields",
    "n_fundamental_fields",
    "has_vwap",
    "has_returns",
    "n_distinct_fields",
    # Block D — complexity (3)
    "n_total_operators",
    "n_unique_operators",
    "max_nesting_depth",
    # Block E — windows (3)
    "min_window",                # 0 if no ts_* window args present
    "max_window",                # 0 if no ts_* window args present
    "has_window_under_5",
    # Block F — derived signals (3)
    "predicted_turnover",                  # from MathEngine.predict_turnover
    "mathengine_critique_severity",        # 0=OK, 1=WARN, 2=FAIL (worst issue)
    "expr_token_count",                    # rough complexity proxy
    # Block G — v2 catalog-aware features (5 new)
    "has_vector_field",                    # ANY VECTOR-type field referenced (deterministic fail signal)
    "n_unknown_identifiers",               # bare idents in no catalog (typos, wrong names)
    "n_extended_catalog_fields",           # fields in fields_full but not canonical (obscure-valid)
    "has_pandas_syntax",                   # x.shift(...), x.rolling(...) — invalid in FastExpr
    "has_invalid_op_alias",                # bare delay(...) / SMA(...) etc. — TA-Lib aliases
    # Block H — v3 settings features (6 new). These live in the alpha's `settings`
    # dict, NOT in the expression. Previously invisible to the model — fixing a
    # measurable blind spot. Subindustry-neutralization in particular has ~2x the
    # fail rate of Market/Sector in our data.
    "universe_top1000",                    # 1 if TOP1000, 0 if TOP3000 / other
    "neutralization_sector",               # 1 if Sector
    "neutralization_subindustry",          # 1 if Subindustry (highest-fail-rate setting)
    "decay_setting",                       # numerical 0..50, the actual decay in settings
    "truncation_setting",                  # numerical 0..1
    "delay_setting",                       # numerical 0/1; almost always 1 but kept for future
)


_FAIL_SEVERITY = {"OK": 0, "WARN": 1, "FAIL": 2}


# Module-level shared MathEngine. critique() is pure given the KB; safe to share.
_ENGINE: MathEngine | None = None


def _engine() -> MathEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = MathEngine()
    return _ENGINE


def _outer_op(expression: str) -> str | None:
    """Identify the outermost operator name, ignoring leading unary minus / whitespace."""
    s = expression.strip()
    if s.startswith("-"):
        s = s[1:].lstrip()
    m = re.match(r"([A-Za-z_][A-Za-z_0-9]*)\s*\(", s)
    if not m:
        return None
    name = m.group(1)
    # Verify the matched call wraps the entire remaining expression.
    open_idx = s.find("(", m.end() - 1)
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                # If the closing paren is the final non-whitespace char,
                # this op wraps the whole expression.
                return name if s[i + 1 :].strip() == "" else None
    return None


def _max_nesting_depth(expression: str) -> int:
    depth = 0
    max_depth = 0
    for ch in expression:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
    return max_depth


def _extract_windows(expression: str) -> list[int]:
    """All integer day-window arguments. Captures `,<digits>)` pattern used by ts_* ops."""
    expr = _norm(expression)
    return [int(m.group(1)) for m in re.finditer(r",(\d+)\)", expr)]


def _extract_fields(expression: str) -> set[str]:
    """Bare identifiers (not followed by '(') that are known catalog fields."""
    call_names = set(_extract_call_names(expression))
    out: set[str] = set()
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z_0-9]*)\b", expression):
        tok = m.group(1)
        if tok in call_names:
            continue
        if expression[m.end() : m.end() + 1] == "(":
            continue
        if tok in _ALL_KNOWN_FIELDS:
            out.add(tok)
    return out


def _extract_all_bare_identifiers(expression: str) -> set[str]:
    """All bare identifiers (not call names, not reserved words). Used by the
    v2 features that need to see identifiers BEYOND the canonical 21 — VECTOR
    fields, extended-catalog fields, and unknown/typo identifiers."""
    call_names = set(_extract_call_names(expression))
    out: set[str] = set()
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z_0-9]*)\b", expression):
        tok = m.group(1)
        if tok in call_names or tok in _RESERVED_TOKENS:
            continue
        if expression[m.end() : m.end() + 1] == "(":
            continue
        out.add(tok)
    return out


_PANDAS_METHOD_RE = re.compile(r"\b[A-Za-z_]\w*\.\w+\s*\(")


def _critique_severity(expression: str, settings: dict | None) -> int:
    """Worst verdict from MathEngine.critique(): 0=OK, 1=WARN, 2=FAIL."""
    try:
        crit = _engine().critique(expression, settings or {})
    except Exception:
        return 2  # treat parse failures as FAIL
    verdict = crit.get("verdict", "OK")
    return _FAIL_SEVERITY.get(verdict, 0)


def extract_features(
    expression: str, settings: dict | None = None
) -> dict[str, float]:
    """Compute the v1 feature vector. Returns a dict; the column order is FEATURE_NAMES.

    Pure function: no I/O, no global state mutation beyond the cached engine.
    Robust to malformed expressions — never raises.
    """
    expr_norm = _norm(expression)
    call_names = _extract_call_names(expression)
    call_set = set(call_names)
    windows = _extract_windows(expression)
    fields = _extract_fields(expression)
    outer = _outer_op(expression)

    n_price = sum(1 for f in fields if f in _PRICE_FIELDS)
    n_fund = sum(1 for f in fields if f in _FUNDAMENTAL_FIELDS)

    has_ts_any = bool(call_set & _TS_OPS_ALL)
    has_cs_any = bool(call_set & _CS_OPS)
    has_group_any = bool(call_set & _GROUP_OPS)

    try:
        predicted_turnover = float(_engine().predict_turnover(expression))
    except Exception:
        predicted_turnover = 50.0  # neutral fallback

    # v2 catalog-aware identifier analysis. One scan over all bare identifiers,
    # bucketed against the loaded catalog sets.
    all_bare = _extract_all_bare_identifiers(expression)
    vector_hits = all_bare & _VECTOR_FIELDS
    extended_hits = all_bare & _EXTENDED_CATALOG_FIELDS
    unknown_idents = all_bare - _CATALOG_FIELDS  # neither canonical nor extended catalog
    pandas_syntax = bool(_PANDAS_METHOD_RE.search(expression))
    invalid_aliases = call_set & _INVALID_OP_ALIASES

    # v3 settings extraction. Defaults match brain_kb.json archetype defaults so
    # a missing settings dict doesn't bias the model toward an arbitrary value.
    s = settings or {}
    universe = str(s.get("universe", "TOP3000"))
    neutralization = str(s.get("neutralization", "Market"))
    try:
        decay_val = float(s.get("decay", 4))
    except (TypeError, ValueError):
        decay_val = 4.0
    try:
        trunc_val = float(s.get("truncation", 0.05))
    except (TypeError, ValueError):
        trunc_val = 0.05
    try:
        delay_val = float(s.get("delay", 1))
    except (TypeError, ValueError):
        delay_val = 1.0

    feat: dict[str, float] = {
        # Block A
        "has_rank": float("rank" in call_set),
        "has_zscore": float("zscore" in call_set),
        "has_ts_decay_linear": float("ts_decay_linear" in call_set),
        "has_ts_corr_like": float(bool(call_set & _TS_CORR_LIKE)),
        "has_ts_zscore": float("ts_zscore" in call_set),
        "has_ts_rank": float("ts_rank" in call_set),
        "has_ts_arg_extrema": float(bool(call_set & _TS_ARG_EXTREMA)),
        "has_ts_extrema": float(bool(call_set & _TS_EXTREMA)),
        "has_if_else": float("if_else" in call_set),
        "has_group_neutralize": float("group_neutralize" in call_set),
        "has_group_rank_like": float(bool(call_set & _GROUP_RANK_LIKE)),
        "arithmetic_only": float(not (has_ts_any or has_cs_any or has_group_any)),
        # Block B
        "outer_is_rank": float(outer == "rank"),
        "outer_is_zscore": float(outer == "zscore"),
        "outer_is_group_neutralize": float(outer == "group_neutralize"),
        "outer_is_ts_op": float(outer in _TS_OPS_ALL if outer else False),
        # Block C
        "n_price_fields": float(n_price),
        "n_fundamental_fields": float(n_fund),
        "has_vwap": float("vwap" in fields),
        "has_returns": float("returns" in fields),
        "n_distinct_fields": float(len(fields)),
        # Block D
        "n_total_operators": float(len(call_names)),
        "n_unique_operators": float(len(call_set)),
        "max_nesting_depth": float(_max_nesting_depth(expression)),
        # Block E
        "min_window": float(min(windows) if windows else 0),
        "max_window": float(max(windows) if windows else 0),
        "has_window_under_5": float(any(w < 5 for w in windows)),
        # Block F
        "predicted_turnover": predicted_turnover,
        "mathengine_critique_severity": float(_critique_severity(expression, settings)),
        "expr_token_count": float(len(re.findall(r"[A-Za-z_]\w*|\d+|[(),+\-*/]", expression))),
        # Block G — v2 catalog-aware features
        "has_vector_field": float(bool(vector_hits)),
        "n_unknown_identifiers": float(len(unknown_idents)),
        "n_extended_catalog_fields": float(len(extended_hits)),
        "has_pandas_syntax": float(pandas_syntax),
        "has_invalid_op_alias": float(bool(invalid_aliases)),
        # Block H — v3 settings features
        "universe_top1000": float(universe == "TOP1000"),
        "neutralization_sector": float(neutralization == "Sector"),
        "neutralization_subindustry": float(neutralization == "Subindustry"),
        "decay_setting": decay_val,
        "truncation_setting": trunc_val,
        "delay_setting": delay_val,
    }

    # Cheap invariant: every feature name in FEATURE_NAMES must be set.
    assert set(feat.keys()) == set(FEATURE_NAMES), (
        f"feature dict keys diverged from FEATURE_NAMES "
        f"(extra={set(feat.keys()) - set(FEATURE_NAMES)}, "
        f"missing={set(FEATURE_NAMES) - set(feat.keys())})"
    )
    return feat


def to_vector(features: dict[str, float]) -> list[float]:
    """Materialize the feature dict as a column-ordered vector."""
    return [float(features[name]) for name in FEATURE_NAMES]


def batch_extract(
    entries: Iterable[dict],
) -> tuple[list[list[float]], list[dict], list[dict]]:
    """Extract features for many entries at once. Skips entries with no expression.

    Returns (X, raw_dicts, skipped_entries):
      X            — list of feature vectors (column-ordered)
      raw_dicts    — list of feature dicts (for inspection / SHAP later)
      skipped     — entries that lacked an `expression` field
    """
    X: list[list[float]] = []
    raw: list[dict] = []
    skipped: list[dict] = []
    for e in entries:
        expr = e.get("expression")
        if not expr:
            skipped.append(e)
            continue
        feat = extract_features(expr, e.get("settings"))
        X.append(to_vector(feat))
        raw.append(feat)
    return X, raw, skipped
