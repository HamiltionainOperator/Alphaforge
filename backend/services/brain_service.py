from __future__ import annotations

import os
import threading
import time
from typing import Any

from brain_api import (
    BrainSession,
    _load_targets,
    _log_result,
    fetch_alpha_pnl,
    fetch_alpha_stats,
    load_credentials,
    meets_submission_targets,
)


_session: BrainSession | None = None
_session_lock = threading.RLock()
_log_lock = threading.Lock()
_status: dict[str, Any] = {
    "state": "disconnected",
    "message": "Brain session has not been authenticated.",
    "authenticated_at": None,
    "email": None,
}


def get_status() -> dict[str, Any]:
    with _session_lock:
        return dict(_status)


def connect() -> dict[str, Any]:
    global _session
    with _session_lock:
        if _session is not None:
            _status.update({
                "state": "connected",
                "message": "Brain session is authenticated.",
            })
            return dict(_status)

        email, password = _read_credentials()
        _status.update({
            "state": "authenticating",
            "message": "Authenticating with WorldQuant Brain.",
            "email": _mask_email(email),
        })
        try:
            _session = BrainSession(email, password)
        except Exception as exc:  # noqa: BLE001
            _status.update({
                "state": "disconnected",
                "message": f"{type(exc).__name__}: {exc}",
                "authenticated_at": None,
            })
            raise

        _status.update({
            "state": "connected",
            "message": "Brain session is authenticated.",
            "authenticated_at": _utc_now(),
            "email": _mask_email(email),
        })
        return dict(_status)


def get_session() -> BrainSession:
    with _session_lock:
        if _session is not None:
            return _session
    connect()
    with _session_lock:
        if _session is None:
            raise RuntimeError("Brain session was not created.")
        return _session


def simulate_alpha(
    expression: str,
    settings: dict[str, Any] | None = None,
    alpha: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session = get_session()
    clean_expression = (expression or "").strip()
    clean_settings = settings or {}
    alpha_meta = {
        "name": (alpha or {}).get("name"),
        "expression": clean_expression,
        "archetype": (alpha or {}).get("archetype"),
        "hypothesis": (alpha or {}).get("hypothesis"),
        "settings": clean_settings,
        "expected_sharpe": (alpha or {}).get("expected_sharpe"),
    }

    started = time.time()
    try:
        result = session.simulate(clean_expression, clean_settings)
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "EXCEPTION",
            "error": f"{type(exc).__name__}: {exc}",
        }
    elapsed = round(time.time() - started, 1)
    result["elapsed_s"] = elapsed

    targets = _load_targets()
    result["checks"] = _extract_checks(result)
    result["submission_pass"] = meets_submission_targets(result, targets)
    result["submittable"], result["failures"] = _summarize_submission(result, targets)
    # Free-creation mode (default): do not persist any history (lessons.jsonl,
    # memory.json, feedback.json, bad_operator_field_pairs.json). Set
    # ALPHAFORGE_PERSIST_MEMORY=1 to re-enable logging for offline analysis.
    if os.getenv("ALPHAFORGE_PERSIST_MEMORY", "0").strip().lower() in ("1", "true", "yes", "on"):
        record = {
            "timestamp": _utc_now(),
            "elapsed_s": elapsed,
            "alpha": alpha_meta,
            "result": result,
        }
        with _log_lock:
            _log_result(record, targets)
    return result


def fetch_details(alpha_id: str) -> dict[str, Any]:
    """Fetch the full picture for a simulated alpha: IS/OS stats, all BRAIN
    checks, correlations, the daily PnL curve, and a submittable/failures verdict."""
    if not (alpha_id or "").strip():
        return {"status": "ERROR", "error": "no alpha_id provided"}
    session = get_session()
    stats = fetch_alpha_stats(session, alpha_id)
    if stats.get("status") != "OK":
        return {"status": stats.get("status", "ERROR"), "alpha_id": alpha_id,
                "error": stats.get("error") or f"could not fetch stats for {alpha_id}"}

    targets = _load_targets()
    checks = _extract_checks(stats)
    submittable, failures = _summarize_submission({**stats, "checks": checks}, targets)
    pnl = _shape_pnl(fetch_alpha_pnl(session, alpha_id))

    return {
        "status": "OK",
        "alpha_id": alpha_id,
        "stats": {
            "sharpe": stats.get("sharpe"),
            "fitness": stats.get("fitness"),
            "turnover_pct": stats.get("turnover_pct"),
            "returns": stats.get("returns"),
            "drawdown": stats.get("drawdown"),
            "margin": stats.get("margin"),
            "long_count": stats.get("long_count"),
            "short_count": stats.get("short_count"),
            "os_sharpe": stats.get("os_sharpe"),
            "os_fitness": stats.get("os_fitness"),
            "os_turnover_pct": stats.get("os_turnover_pct"),
            "os_returns": stats.get("os_returns"),
        },
        "checks": checks,
        "correlations": {
            "self": stats.get("self_correlation"),
            "prod": stats.get("prod_correlation"),
        },
        "concentrated_weight": stats.get("concentrated_weight"),
        "pnl": pnl,
        "submittable": submittable,
        "failures": failures,
    }


def submit_alpha_to_pool(alpha_id: str) -> dict[str, Any]:
    """Submit a simulated alpha to the BRAIN alpha pool (real account action)."""
    if not (alpha_id or "").strip():
        return {"status": "ERROR", "error": "no alpha_id provided"}
    session = get_session()
    try:
        return session.submit_to_pool(alpha_id)
    except Exception as exc:  # noqa: BLE001
        return {"status": "SUBMIT_ERROR", "alpha_id": alpha_id, "error": f"{type(exc).__name__}: {exc}"}


def _extract_checks(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw = (result.get("_raw_is") or {}).get("checks")
    if not isinstance(raw, list):
        return []
    out = []
    for c in raw:
        if isinstance(c, dict) and c.get("name"):
            out.append({
                "name": c.get("name"),
                "result": c.get("result"),
                "value": c.get("value"),
                "limit": c.get("limit"),
            })
    return out


def _summarize_submission(result: dict[str, Any], targets: dict[str, Any] | None = None) -> tuple[bool, list[dict[str, Any]]]:
    """Return (submittable, failures). Failures combine missed numeric gates with
    any FAIL-level BRAIN check — this is what feeds the LLM's failure repair."""
    t = targets or _load_targets()
    submittable = bool(meets_submission_targets(result, t))
    sharpe_min = float(t.get("sharpe_min", 1.25))
    fitness_min = float(t.get("fitness_min", 1.0))
    turnover_min = float(t.get("turnover_min_pct", 1))
    turnover_max = float(t.get("turnover_max_pct", 70))

    failures: list[dict[str, Any]] = []
    sharpe = result.get("sharpe")
    fitness = result.get("fitness")
    turnover = result.get("turnover_pct")

    if not isinstance(sharpe, (int, float)) or sharpe < sharpe_min:
        failures.append({"kind": "gate", "name": "Sharpe", "value": sharpe, "limit": sharpe_min,
                         "message": f"Sharpe {_fmt(sharpe)} is below the {sharpe_min} minimum."})
    if not isinstance(fitness, (int, float)) or fitness < fitness_min:
        failures.append({"kind": "gate", "name": "Fitness", "value": fitness, "limit": fitness_min,
                         "message": f"Fitness {_fmt(fitness)} is below the {fitness_min} minimum."})
    if isinstance(turnover, (int, float)):
        if turnover < turnover_min:
            failures.append({"kind": "gate", "name": "Turnover", "value": turnover, "limit": turnover_min,
                             "message": f"Turnover {turnover:.1f}% is below the {turnover_min}% floor."})
        elif turnover > turnover_max:
            failures.append({"kind": "gate", "name": "Turnover", "value": turnover, "limit": turnover_max,
                             "message": f"Turnover {turnover:.1f}% exceeds the {turnover_max}% cap."})

    for c in result.get("checks", []):
        if c.get("result") == "FAIL":
            failures.append({"kind": "check", "name": c.get("name"), "value": c.get("value"), "limit": c.get("limit"),
                             "message": f"BRAIN check {c.get('name')} failed (value {_fmt(c.get('value'))}, limit {_fmt(c.get('limit'))})."})
    return submittable, failures


def _shape_pnl(series: list[tuple[str, float]] | None) -> dict[str, Any] | None:
    """Turn a daily PnL series into a downsampled cumulative curve for charting."""
    if not series:
        return None
    cumulative: list[float] = []
    running = 0.0
    for _, value in series:
        running += value
        cumulative.append(running)
    dates = [d for d, _ in series]

    max_points = 160
    if len(cumulative) > max_points:
        step = len(cumulative) / max_points
        idx = [int(i * step) for i in range(max_points)]
        idx[-1] = len(cumulative) - 1
        points = [cumulative[i] for i in idx]
    else:
        points = cumulative

    return {
        "points": [round(p, 2) for p in points],
        "n": len(cumulative),
        "start": dates[0],
        "end": dates[-1],
        "final": round(cumulative[-1], 2),
        "max": round(max(cumulative), 2),
        "min": round(min(cumulative), 2),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return "n/a" if value is None else str(value)


def _read_credentials() -> tuple[str, str]:
    email = os.getenv("BRAIN_EMAIL", "").strip()
    password = os.getenv("BRAIN_PASSWORD", "").strip()
    if email and password:
        return email, password

    creds = load_credentials()
    return str(creds["email"]), str(creds["password"])


def _mask_email(email: str) -> str:
    if "@" not in email:
        return "***"
    name, domain = email.split("@", 1)
    if not name:
        return f"***@{domain}"
    return f"{name[0]}***@{domain}"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

