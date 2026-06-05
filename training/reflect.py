"""Compare a hypothesis to its simulation result and write the reflection back.

Called by brain_api after each sim completes. Two-tier reflection:

  1. Rule-based (always runs, no LLM): compares predicted sharpe/turnover bands to
     actual; flags refutation against the stated criterion; writes verdict.
  2. LLM reflection (optional, runs if Ollama reachable): 2-3 sentences explaining
     WHAT WAS LEARNED — was the mechanism right but the operationalization wrong?
     Was the regime wrong? Should the hypothesis be retired or extended?

The reflection is appended to the per-alpha JSON in data/hypotheses/<file>.json.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import URLError

ROOT = Path(__file__).resolve().parent.parent
HYPOTHESES_DIR = ROOT / "data" / "hypotheses"

OLLAMA_URL = os.environ.get("ALPHAQUANT_OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = "qwen2.5:7b-instruct"
TIMEOUT_S = 120


# -------------------------------------------- range parsing
_RANGE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)")


def _parse_range(s) -> tuple[float, float] | None:
    if not isinstance(s, str):
        return None
    m = _RANGE_RE.search(s)
    if not m:
        return None
    lo, hi = float(m.group(1)), float(m.group(2))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _band_verdict(actual, band: tuple[float, float] | None) -> str:
    if not isinstance(actual, (int, float)) or band is None:
        return "?"
    lo, hi = band
    if lo <= actual <= hi:
        return "in_band"
    if actual > hi:
        return "above_band"
    return "below_band"


# -------------------------------------------- rule-based core
def rule_based_reflection(hypothesis: dict, result: dict) -> dict:
    """Deterministic reflection. Compares prediction to actual on sharpe/turnover,
    classifies the outcome, and emits a verdict + short lesson string."""
    pred = (hypothesis or {}).get("prediction") or {}
    pred_sharpe = _parse_range(pred.get("sharpe") or pred.get("sharpe_range"))
    pred_turnover = _parse_range(pred.get("turnover_pct") or pred.get("turnover_range_pct"))

    sharpe = result.get("sharpe") if isinstance(result, dict) else None
    fitness = result.get("fitness") if isinstance(result, dict) else None
    turnover = result.get("turnover_pct") if isinstance(result, dict) else None
    status = result.get("status") if isinstance(result, dict) else None

    sharpe_verdict = _band_verdict(sharpe, pred_sharpe)
    turnover_verdict = _band_verdict(turnover, pred_turnover)

    # Top-level verdict
    if status not in ("OK", None):
        verdict = "errored"
    elif isinstance(sharpe, (int, float)) and sharpe >= 1.25 and isinstance(fitness, (int, float)) and fitness >= 1.0:
        verdict = "confirmed"
    elif isinstance(sharpe, (int, float)) and sharpe <= -0.5:
        verdict = "refuted_inverted"  # the sign is backwards — invert the expression
    elif isinstance(sharpe, (int, float)) and -0.5 < sharpe < 0.5:
        verdict = "refuted_no_edge"
    elif isinstance(sharpe, (int, float)) and 0.5 <= sharpe < 1.25:
        verdict = "partial_support"
    else:
        verdict = "inconclusive"

    # Build a one-line lesson the prompt block can show in future generations.
    lesson_parts: list[str] = []
    if verdict == "refuted_inverted":
        lesson_parts.append("mechanism direction was wrong — try inverting the sign (leading -)")
    elif verdict == "refuted_no_edge":
        mech = hypothesis.get("mechanism", "?")
        lesson_parts.append(f"the {mech} mechanism did not produce edge here; do not reuse this composition")
    elif verdict == "partial_support":
        lesson_parts.append("direction is right but edge is weak — tighten with smoothing or stricter neutralization")
    elif verdict == "confirmed":
        lesson_parts.append("hypothesis held — borrow this structure, vary fields/windows for diversity")
    if turnover_verdict == "above_band":
        lesson_parts.append(f"turnover {turnover:.0f}% above predicted band — add ts_decay_linear or lengthen windows")
    elif turnover_verdict == "below_band":
        lesson_parts.append(f"turnover {turnover:.0f}% below predicted band — add a daily-varying leg")

    return {
        "method": "rule_based",
        "verdict": verdict,
        "sharpe_actual": sharpe,
        "sharpe_predicted_range": pred_sharpe,
        "sharpe_band": sharpe_verdict,
        "fitness_actual": fitness,
        "turnover_actual": turnover,
        "turnover_predicted_range": pred_turnover,
        "turnover_band": turnover_verdict,
        "lesson": "; ".join(lesson_parts) or "no clear lesson — result was ambiguous",
        "reflected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# -------------------------------------------- LLM elaboration (optional)
def _build_reflection_prompt(payload: dict) -> str:
    return f"""You are a quant researcher reviewing a single alpha simulation result against the
hypothesis that was generated for it. Your job: in 2-3 sentences, explain what was LEARNED.

The hypothesis predicted a specific mechanism, sign, and regime. The simulation produced
sharpe/fitness/turnover numbers. Was the mechanism correct? Was the sign correct but the
operationalization noisy? Was the regime assumption wrong? Should the hypothesis be retired,
extended, or inverted?

Be concrete. Reference the specific numbers. Do not hedge. If the result confirms or refutes
the hypothesis, say so explicitly. Return STRICT JSON:

{{
  "verdict": "confirmed | partial_support | refuted_no_edge | refuted_inverted | inconclusive | errored",
  "what_was_learned": "2-3 sentences explaining what we now know",
  "next_mutation": "concrete next step — if confirmed: how to extend it; if refuted: what to try instead"
}}

INPUT:
{json.dumps(payload, indent=2, default=str)}
"""


def _ollama_call(prompt: str, model: str) -> dict | None:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3, "num_ctx": 8192, "num_predict": 800},
    }
    req = urlrequest.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "alphaquant/1.0",
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = json.loads(resp.read())
    except URLError:
        return None
    raw = (data.get("response") or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def llm_reflection(hypothesis: dict, result: dict, model: str | None = None) -> dict | None:
    """Optional LLM elaboration on top of the rule-based reflection. Returns None
    if Ollama is unreachable or the response is malformed — caller must handle that."""
    model = model or os.environ.get("ALPHAQUANT_MODEL") or DEFAULT_MODEL
    payload = {
        "hypothesis": hypothesis,
        "simulation_result": {
            "status": result.get("status"),
            "sharpe": result.get("sharpe"),
            "fitness": result.get("fitness"),
            "turnover_pct": result.get("turnover_pct"),
            "returns": result.get("returns"),
            "drawdown": result.get("drawdown"),
            "checks_pass": result.get("checks_pass"),
        },
    }
    resp = _ollama_call(_build_reflection_prompt(payload), model=model)
    if not resp:
        return None
    resp["method"] = "llm"
    resp["model"] = model
    resp["reflected_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return resp


# -------------------------------------------- public entrypoint
def reflect_on_record(alpha: dict, result: dict, *, use_llm: bool = True) -> dict | None:
    """Find the per-alpha hypothesis file (saved at generation time), compute a
    reflection, and write {result, reflection} into it. Returns the merged
    reflection dict (or None if no hypothesis file was found)."""
    if not HYPOTHESES_DIR.exists():
        return None
    name = alpha.get("name") or ""
    expression = alpha.get("expression") or ""
    target: Path | None = None
    candidates = sorted(HYPOTHESES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for fp in candidates[:200]:
        try:
            doc = json.loads(fp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if doc.get("alpha_name") == name and doc.get("expression") == expression:
            target = fp
            break
        if doc.get("expression") == expression and (doc.get("result") is None):
            target = fp
            break
    if target is None:
        return None

    try:
        doc = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if doc.get("reflection"):
        return doc.get("reflection")

    hypothesis = doc.get("hypothesis") or {}
    rule_ref = rule_based_reflection(hypothesis, result)
    merged: dict = dict(rule_ref)

    if use_llm:
        llm_ref = llm_reflection(hypothesis, result)
        if llm_ref:
            merged.update({
                "what_was_learned": llm_ref.get("what_was_learned"),
                "next_mutation": llm_ref.get("next_mutation"),
                "llm_verdict": llm_ref.get("verdict"),
                "method": "rule_based+llm",
            })

    doc["result"] = {
        "status": result.get("status"),
        "sharpe": result.get("sharpe"),
        "fitness": result.get("fitness"),
        "turnover_pct": result.get("turnover_pct"),
        "returns": result.get("returns"),
        "drawdown": result.get("drawdown"),
        "alpha_id": result.get("alpha_id"),
        "submission_pass": result.get("submission_pass"),
    }
    doc["reflection"] = merged
    try:
        target.write_text(json.dumps(doc, indent=2) + "\n")
    except OSError:
        pass
    return merged
