"""
WorldQuant Brain API client.

Workflow:
    BrainSession.simulate(expression, settings)
        → POST /simulations → poll Location URL → fetch /alphas/{id}
        → returns normalized stats dict (turnover already converted bp → %).

    submit_batch(alphas, max_concurrent=10)
        → runs simulations concurrently against one authenticated session,
          auto-logs each result to data/memory.json (passing) or
          data/feedback.json (failing). Re-authenticates on 401.

NOTE: This module performs *simulation* (Brain's backtest). It does NOT
auto-submit alphas to the Alpha Pool — that's a manual step in the Brain
web UI after you review the stats. By design (per Phase 3 plan).

Endpoints (documented Brain API):
    Base: https://api.worldquantbrain.com
    POST /authentication                 — HTTP Basic auth → session cookie
    POST /simulations                    → 201 Location: /simulations/{id}
    GET  /simulations/{id}               → poll status until COMPLETE
    GET  /alphas/{alpha_id}              → fetch is.stats (sharpe, fitness, ...)

Public API:
    BrainSession(email, password)
    BrainSession.simulate(expression, settings) -> dict
    submit_alpha(session, alpha_dict) -> dict
    submit_batch(alphas, max_concurrent=10) -> list[dict]
    load_credentials() -> dict
"""
from __future__ import annotations

import json
import re
import sys
import time
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from requests.auth import HTTPBasicAuth

from math_engine import classify_mechanism_family

ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "brain_kb.json"
CRED_PATH = ROOT / "credentials.json"
MEMORY_PATH = ROOT / "data" / "memory.json"
FEEDBACK_PATH = ROOT / "data" / "feedback.json"
BAD_PAIRS_PATH = ROOT / "data" / "bad_operator_field_pairs.json"
LESSONS_PATH = ROOT / "data" / "lessons.jsonl"
SESSION_COOKIE_PATH = ROOT / "data" / ".wq_session_cookies.json"

BASE_URL = "https://api.worldquantbrain.com"
AUTH_URL = f"{BASE_URL}/authentication"
SIM_URL = f"{BASE_URL}/simulations"

USER_AGENT = "alphaquant/1.0 (Brain client)"
REQUEST_TIMEOUT = 30
POLL_INTERVAL_MIN = 1.0
POLL_INTERVAL_MAX = 8.0
SIM_TIMEOUT_S = 900  # 15 minutes per simulation
SUBMIT_TIMEOUT_S = 600  # 10 minutes for a submission to resolve its checks
RATE_LIMIT_BACKOFF = 10  # seconds on 429


def load_credentials() -> dict:
    if not CRED_PATH.exists():
        raise FileNotFoundError(
            f"\ncredentials.json missing at {CRED_PATH}\n"
            f"  1. cp credentials.json.example credentials.json\n"
            f"  2. edit credentials.json with your Brain email + password\n"
        )
    data = json.loads(CRED_PATH.read_text())
    if not data.get("email") or not data.get("password"):
        raise ValueError("credentials.json must have non-empty 'email' and 'password' fields")
    if data["email"] == "your.email@example.com":
        raise ValueError("credentials.json still has placeholder values — please fill in your real Brain login")
    return data


def _load_targets() -> dict:
    kb = json.loads(KB_PATH.read_text())
    return kb.get("fitness_formula", {}).get("targets", {})


def meets_submission_targets(result: dict, targets: dict | None = None) -> bool:
    """True when a simulated alpha meets all submission gates.

    Gates:
      - simulation status OK
      - BRAIN checks block has no FAIL-level check results
      - sharpe >= sharpe_min
      - fitness >= fitness_min
      - turnover within [turnover_min_pct, turnover_max_pct]
    """
    t = targets or _load_targets()
    sharpe_min = float(t.get("sharpe_min", 1.25))
    fitness_min = float(t.get("fitness_min", 1.0))
    turnover_min = float(t.get("turnover_min_pct", 1))
    turnover_max = float(t.get("turnover_max_pct", 70))

    return (
        result.get("status") == "OK"
        and bool(result.get("checks_pass"))
        and isinstance(result.get("sharpe"), (int, float))
        and result["sharpe"] >= sharpe_min
        and isinstance(result.get("fitness"), (int, float))
        and result["fitness"] >= fitness_min
        and isinstance(result.get("turnover_pct"), (int, float))
        and turnover_min <= result["turnover_pct"] <= turnover_max
    )


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


# ----------------------------------------------------------------- BrainSession
class BrainSession:
    """Authenticated HTTP session for Brain. Thread-safe across simulate() calls.

    The underlying requests.Session is thread-safe for our usage (GET/POST per
    call, no shared mutable state beyond cookies). Auth refresh is guarded by a
    lock so concurrent 401s don't trigger N parallel re-auths.
    """

    def __init__(self, email: str, password: str) -> None:
        self.email = email
        self.password = password
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self._auth_lock = threading.Lock()
        self._authenticate()

    # ── Cookie persistence ─────────────────────────────────────────────────────

    def _save_cookies(self) -> None:
        """Persist session cookies to disk so biometrics aren't repeated."""
        try:
            SESSION_COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
            cookies = {c.name: c.value for c in self.s.cookies}
            import json as _json
            SESSION_COOKIE_PATH.write_text(_json.dumps(cookies))
        except Exception:
            pass

    def _load_cookies(self) -> bool:
        """Load saved cookies. Returns True if cookies were found and loaded."""
        try:
            if not SESSION_COOKIE_PATH.exists():
                return False
            import json as _json
            cookies = _json.loads(SESSION_COOKIE_PATH.read_text())
            if not cookies:
                return False
            self.s.cookies.update(cookies)
            return True
        except Exception:
            return False

    def _verify_session(self) -> bool:
        """Ping a lightweight endpoint to check if the saved session is still valid."""
        try:
            r = self.s.get(f"{BASE_URL}/users/self", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    # ── Auth ───────────────────────────────────────────────────────────────────

    def _authenticate(self) -> None:
        with self._auth_lock:
            # Try saved cookies first — avoids biometric entirely if session is alive
            if self._load_cookies() and self._verify_session():
                print("Session restored from saved cookies — no biometric needed.")
                return

            for attempt in range(5):
                r = self.s.post(
                    AUTH_URL,
                    auth=HTTPBasicAuth(self.email, self.password),
                    timeout=REQUEST_TIMEOUT,
                )
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 60))
                    print(f"Auth throttled — waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
                # BRAIN periodically requires a browser-based biometric (Persona)
                # check. It answers the auth POST with 401 + WWW-Authenticate: persona
                # and a Location pointing at the inquiry. Walk the user through it; the
                # successful persona POST (201) authenticates THIS session and returns
                # the session response, so we adopt it directly. Re-POSTing to AUTH_URL
                # would start a FRESH inquiry and 401 again — that was the old bug.
                if r.status_code == 401 and self._is_biometric_challenge(r):
                    r = self._complete_biometric(r)
                if r.status_code == 401:
                    raise PermissionError(f"Brain auth failed (401). Check credentials.json. Body: {r.text[:200]}")
                r.raise_for_status()
                # Session cookie is now set — save it for next run
                self._save_cookies()
                return
            raise PermissionError("Auth failed after 5 attempts (throttled).")

    @staticmethod
    def _is_biometric_challenge(r: requests.Response) -> bool:
        if (r.headers.get("WWW-Authenticate") or "").lower() == "persona":
            return True
        return '"inquiry"' in (r.text or "")

    def _complete_biometric(self, challenge: requests.Response) -> requests.Response:
        """Walk the user through WorldQuant's Persona biometric verification.

        The inquiry URL must be opened in a browser and then re-POSTed within
        this same session to finalize verification. The finalizing POST returns
        201 with the auth cookie set on this session — that response IS the
        authenticated session, and it's what we return so the caller can use it
        WITHOUT re-hitting AUTH_URL (which would only spawn a fresh inquiry).
        In API/server mode there may be no useful terminal prompt, so this opens
        the inquiry URL once and polls the finalizing POST until Persona is done.
        """
        persona_url = urljoin(challenge.url, challenge.headers.get("Location", ""))
        print("\n" + "=" * 72)
        print("WorldQuant BRAIN requires biometric (Persona) verification.")
        print("Open this URL in your browser and complete the check:")
        print(f"\n  {persona_url}\n")
        print("=" * 72)
        try:
            webbrowser.open(persona_url)
        except Exception:  # noqa: BLE001
            pass

        deadline = time.time() + 300
        print("Waiting for Persona completion; polling every 5 seconds.")
        while time.time() < deadline:
            r = self.s.post(persona_url, timeout=REQUEST_TIMEOUT)
            if r.status_code in (200, 201):
                print("Biometric verified.")
                return r
            print(f"  Persona not verified yet (status {r.status_code}); polling again in 5s.")
            time.sleep(5)
        raise PermissionError("Biometric verification not completed within 5 minutes.")

    def _refresh_if_needed(self, response: requests.Response) -> bool:
        if response.status_code == 401:
            self._authenticate()
            return True
        return False

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", REQUEST_TIMEOUT)
        r = self.s.request(method, url, **kwargs)
        if r.status_code == 401:
            self._refresh_if_needed(r)
            r = self.s.request(method, url, **kwargs)
        if r.status_code == 429:
            time.sleep(RATE_LIMIT_BACKOFF)
            r = self.s.request(method, url, **kwargs)
        return r

    @staticmethod
    def _is_concurrency_limit(r: requests.Response) -> bool:
        """True when BRAIN rejects a submit because too many simulations are already
        running (concurrent-slot limit). Surfaces as 429, or a 4xx whose body mentions
        a concurrency/limit condition."""
        if r.status_code == 429:
            return True
        if r.status_code >= 400:
            body = (r.text or "").lower()
            return any(k in body for k in (
                "concurrent", "simulation_limit", "simulation limit",
                "too many", "limit exceeded", "slot",
            ))
        return False

    # ---------------------------------------------------------- simulate
    def simulate(self, expression: str, settings: dict) -> dict:
        """Run a single simulation. Returns normalized stats dict."""
        payload = {
            "type": "REGULAR",
            "settings": _normalize_settings(settings),
            "regular": expression,
        }
        r = self._request("POST", SIM_URL, json=payload)
        # Concurrent-limit warning: let the server queue breathe, then retry the submit
        # once. (_request already does a 10s backoff on a bare 429; this adds a short,
        # submission-specific pause that also covers non-429 concurrency error bodies.)
        if self._is_concurrency_limit(r):
            print(f"  [sim] concurrent-limit warning (status {r.status_code}); "
                  f"sleeping 5s to let the queue breathe before retry...", flush=True)
            time.sleep(5)
            r = self._request("POST", SIM_URL, json=payload)
        if r.status_code >= 400:
            return {"status": "SUBMIT_ERROR", "http_status": r.status_code, "error": r.text[:500]}
        sim_url = r.headers.get("Location")
        if not sim_url:
            return {"status": "SUBMIT_ERROR", "error": "no Location header on simulation submit", "raw": r.text[:500]}

        return self._wait_for_result(sim_url)

    def _wait_for_result(self, sim_url: str) -> dict:
        deadline = time.time() + SIM_TIMEOUT_S
        wait = POLL_INTERVAL_MIN
        while time.time() < deadline:
            r = self._request("GET", sim_url)
            if r.status_code >= 400:
                return {"status": "POLL_ERROR", "http_status": r.status_code, "error": r.text[:300]}
            data = r.json() if r.text else {}
            status = (data.get("status") or "").upper()

            # Honor server-suggested retry delay
            retry_hint = float(r.headers.get("Retry-After", 0) or 0)
            if retry_hint:
                time.sleep(min(retry_hint, POLL_INTERVAL_MAX))
                continue

            if status in ("COMPLETE", "WARNING"):
                alpha_id = data.get("alpha") or data.get("alphaId")
                if alpha_id:
                    return self._fetch_alpha_stats(alpha_id, sim_status=status, sim_payload=data)
                return {"status": status, "raw": data}
            if status in ("ERROR", "FAIL", "FAILED"):
                return {"status": "FAIL", "error": data.get("message") or data.get("error") or "sim failed", "raw": data}

            time.sleep(wait)
            wait = min(wait * 1.5, POLL_INTERVAL_MAX)
        return {"status": "TIMEOUT", "error": f"simulation did not complete within {SIM_TIMEOUT_S}s"}

    def _fetch_alpha_stats(self, alpha_id: str, sim_status: str, sim_payload: dict) -> dict:
        url = f"{BASE_URL}/alphas/{alpha_id}"
        r = self._request("GET", url)
        if r.status_code >= 400:
            return {"status": "STATS_ERROR", "alpha_id": alpha_id, "http_status": r.status_code, "error": r.text[:300]}
        data = r.json()
        is_block = data.get("is") or {}
        checks = is_block.get("checks") or []

        # Brain's `is.turnover` is a fraction in [0, 1] (0.6232 ≡ 62.32% daily turnover).
        # To get a percentage value we MULTIPLY by 100, not divide. The earlier comment
        # in brain_kb.json claimed basis-points encoding (1397 = 13.97%) — that's wrong.
        raw_turnover = is_block.get("turnover")
        turnover_pct = (raw_turnover * 100.0) if isinstance(raw_turnover, (int, float)) else None

        os_block = data.get("os") or {}
        raw_os_turnover = os_block.get("turnover")
        os_turnover_pct = (raw_os_turnover * 100.0) if isinstance(raw_os_turnover, (int, float)) else None

        return {
            "status": "OK",
            "sim_status": sim_status,
            "alpha_id": alpha_id,
            "sharpe": is_block.get("sharpe"),
            "fitness": is_block.get("fitness"),
            "turnover_pct": turnover_pct,
            "returns": is_block.get("returns"),
            "drawdown": is_block.get("drawdown"),
            "margin": is_block.get("margin"),
            "long_count": is_block.get("longCount"),
            "short_count": is_block.get("shortCount"),
            "self_correlation": _extract_check(checks, "SELF_CORRELATION"),
            "prod_correlation": _extract_check(checks, "PROD_CORRELATION"),
            "concentrated_weight": _extract_check(checks, "CONCENTRATED_WEIGHT"),
            "checks_pass": all(c.get("result") == "PASS" for c in checks if c.get("result") in ("PASS", "FAIL")),
            "_raw_is": is_block,
            "os_sharpe": os_block.get("sharpe"),
            "os_fitness": os_block.get("fitness"),
            "os_turnover_pct": os_turnover_pct,
            "os_returns": os_block.get("returns"),
            "_raw_os": os_block,
        }

    # ---------------------------------------------------------- submit
    def submit_to_pool(self, alpha_id: str) -> dict:
        """Submit an already-simulated alpha to the BRAIN alpha pool.

        Mirrors simulate()'s submit+poll shape on the /alphas/{id}/submit
        resource: BRAIN processes the submission asynchronously and signals
        "still working" with a Retry-After header. We poll until the checks
        resolve, then interpret the result. Every failure path returns a dict
        (never raises) so the caller can surface exactly what BRAIN said.

        NOTE: submission is a real, account-level action on WorldQuant BRAIN.
        Call it only on an explicit user request.
        """
        if not alpha_id:
            return {"status": "ERROR", "error": "no alpha_id provided"}
        url = f"{BASE_URL}/alphas/{alpha_id}/submit"
        try:
            r = self._request("POST", url)
        except Exception as exc:  # noqa: BLE001
            return {"status": "SUBMIT_ERROR", "alpha_id": alpha_id, "error": f"{type(exc).__name__}: {exc}"}
        if self._is_concurrency_limit(r):
            time.sleep(5)
            r = self._request("POST", url)

        deadline = time.time() + SUBMIT_TIMEOUT_S
        while r.status_code < 400:
            retry = float(r.headers.get("Retry-After", 0) or 0)
            if retry <= 0:
                break  # resolved
            if time.time() > deadline:
                return {"status": "SUBMIT_TIMEOUT", "alpha_id": alpha_id,
                        "error": f"submission did not resolve within {SUBMIT_TIMEOUT_S}s"}
            time.sleep(min(retry, POLL_INTERVAL_MAX))
            try:
                r = self._request("GET", url)
            except Exception as exc:  # noqa: BLE001
                return {"status": "SUBMIT_ERROR", "alpha_id": alpha_id, "error": f"{type(exc).__name__}: {exc}"}

        if r.status_code >= 400:
            return {"status": "SUBMIT_ERROR", "alpha_id": alpha_id,
                    "http_status": r.status_code, "error": (r.text or "")[:600]}
        try:
            data = r.json() if r.text else {}
        except Exception:  # noqa: BLE001
            data = {}
        return self._interpret_submission(alpha_id, data, r.status_code)

    @staticmethod
    def _interpret_submission(alpha_id: str, data: Any, http_status: int) -> dict:
        block = data.get("is") if isinstance(data, dict) else None
        raw_checks = (block or {}).get("checks") if isinstance(block, dict) else None
        if not isinstance(raw_checks, list) and isinstance(data, dict):
            raw_checks = data.get("checks")
        checks = [
            {"name": c.get("name"), "result": c.get("result"), "value": c.get("value"), "limit": c.get("limit")}
            for c in (raw_checks or [])
            if isinstance(c, dict)
        ]
        failed = [c for c in checks if c.get("result") == "FAIL"]
        if failed:
            status, message = "REJECTED", "Submission rejected by BRAIN checks: " + ", ".join(
                str(c.get("name")) for c in failed
            )
        else:
            status, message = "SUBMITTED", "Alpha submitted to the BRAIN alpha pool."
        return {
            "status": status,
            "alpha_id": alpha_id,
            "http_status": http_status,
            "message": message,
            "checks": checks,
            "failed_checks": failed,
        }

    # ---------------------------------------------------------- data catalog

    def fetch_data_fields(
        self,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        limit: int = 5000,
    ) -> dict:
        """Fetch the full field catalog from Brain and save to fields_full.json.

        Returns a summary dict: {"saved": bool, "total": int, "path": str}.
        """
        from datetime import datetime, timezone
        fields_path = ROOT / "data" / "brain_docs" / "fields_full.json"

        all_fields: dict[str, Any] = {}
        offset = 0
        page_size = min(limit, 200)  # Brain typically paginates at 200
        while True:
            url = (
                f"{BASE_URL}/data-fields"
                f"?region={region}&delay={delay}&universe={universe}"
                f"&limit={page_size}&offset={offset}&instrumentType=EQUITY"
            )
            try:
                r = self._request("GET", url)
            except Exception as exc:  # noqa: BLE001
                return {"saved": False, "error": str(exc), "total": len(all_fields)}
            if r.status_code >= 400:
                return {"saved": False, "http_status": r.status_code,
                        "error": r.text[:400], "total": len(all_fields)}
            try:
                data = r.json()
            except Exception:  # noqa: BLE001
                break

            # Brain may return a list or {"results": [...], "count": N}
            items: list[dict] = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("results") or data.get("fields") or data.get("data") or []

            if not items:
                break

            for item in items:
                fid = item.get("id") or item.get("fieldId") or item.get("name")
                if not fid:
                    continue
                all_fields[fid] = {
                    "id": fid,
                    "description": item.get("description") or item.get("desc") or "",
                    "dataset_id": item.get("datasetId") or item.get("dataset_id") or "",
                    "dataset_name": item.get("datasetName") or item.get("dataset_name") or "",
                    "category": item.get("category") or "",
                    "subcategory": item.get("subcategory") or "",
                    "type": item.get("type") or "MATRIX",
                    "coverage": float(item.get("coverage") or 0),
                    "user_count": int(item.get("userCount") or item.get("user_count") or 0),
                    "alpha_count": int(item.get("alphaCount") or item.get("alpha_count") or 0),
                    "region": region,
                    "universe": universe,
                    "delay": delay,
                    "themes": item.get("themes") or [],
                }

            offset += len(items)
            if offset >= limit or len(items) < page_size:
                break

        if not all_fields:
            return {"saved": False, "error": "no fields returned by Brain API", "total": 0}

        # Merge with existing file to preserve any locally-enriched data.
        existing: dict[str, Any] = {}
        try:
            existing = json.loads(fields_path.read_text()).get("fields", {})
        except (OSError, json.JSONDecodeError):
            pass

        merged = {**existing, **all_fields}
        out = {
            "_comment": "Full Brain field catalog. Generated by BrainSession.fetch_data_fields().",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "region": region,
            "universe": universe,
            "delay": delay,
            "instrument_type": "EQUITY",
            "fields": merged,
        }
        fields_path.parent.mkdir(parents=True, exist_ok=True)
        fields_path.write_text(json.dumps(out, indent=2))
        return {"saved": True, "total": len(merged), "path": str(fields_path)}

    def list_my_alphas(self, limit: int = 500) -> list[dict]:
        """Fetch all alphas we have simulated from Brain and save to my_alphas.json.

        Returns the list of alpha dicts (each with id, expression, IS/OS stats).
        """
        my_alphas_path = ROOT / "data" / "brain_docs" / "my_alphas.json"
        results: list[dict] = []
        offset = 0
        page_size = min(limit, 100)
        while True:
            url = f"{BASE_URL}/alphas?limit={page_size}&offset={offset}&order=-dateCreated"
            try:
                r = self._request("GET", url)
            except Exception as exc:  # noqa: BLE001
                print(f"[brain] list_my_alphas fetch error: {exc}")
                break
            if r.status_code >= 400:
                print(f"[brain] list_my_alphas HTTP {r.status_code}: {r.text[:200]}")
                break
            try:
                data = r.json()
            except Exception:  # noqa: BLE001
                break

            items: list[dict] = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("results") or data.get("alphas") or []

            if not items:
                break

            results.extend(items)
            offset += len(items)
            if offset >= limit or len(items) < page_size:
                break

        my_alphas_path.parent.mkdir(parents=True, exist_ok=True)
        my_alphas_path.write_text(json.dumps(results, indent=2))
        print(f"[brain] saved {len(results)} personal alphas → {my_alphas_path}")
        return results


def _extract_check(checks: list[dict], name: str) -> dict | None:
    for c in checks:
        if c.get("name") == name:
            return {"result": c.get("result"), "value": c.get("value"), "limit": c.get("limit")}
    return None


def fetch_alpha_stats(session: "BrainSession", alpha_id: str) -> dict:
    """Fetch full alpha stats (IS + OS) for an already-simulated alpha.

    Returns a dict with is_* and os_* fields, or {"status": "ERROR"} on failure.
    """
    if not alpha_id:
        return {"status": "ERROR", "error": "no alpha_id"}
    url = f"{BASE_URL}/alphas/{alpha_id}"
    r = session._request("GET", url)
    if r.status_code >= 400:
        return {"status": "ERROR", "alpha_id": alpha_id, "http_status": r.status_code}
    try:
        data = r.json()
    except Exception as e:
        return {"status": "ERROR", "alpha_id": alpha_id, "error": str(e)}
    return session._fetch_alpha_stats(alpha_id, sim_status="COMPLETE", sim_payload=data)


# --- PnL fetch for local pairwise self-correlation -----------------------
# BRAIN exposes per-alpha daily PnL via the `recordsets/pnl` resource. This
# is the same series BRAIN uses internally for its SELF_CORRELATION check.
# Fetching it locally lets us compute the full pairwise correlation MATRIX
# across our pool (BRAIN's own check only returns the max correlation vs
# pool, not the matrix).

def fetch_alpha_pnl(session: "BrainSession", alpha_id: str) -> list[tuple[str, float]] | None:
    """Fetch the daily PnL series for a simulated alpha.

    Returns a list of (date_iso, pnl_value) tuples sorted by date, or None
    if the alpha has no recordset (unsimulated or expired).
    """
    if not alpha_id:
        return None
    url = f"{BASE_URL}/alphas/{alpha_id}/recordsets/pnl"
    r = session._request("GET", url)
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        return None
    try:
        doc = r.json()
    except Exception:
        return None
    # BRAIN's recordset shape: {"schema": [...], "records": [[date_iso, pnl, ...], ...]}
    records = doc.get("records") if isinstance(doc, dict) else None
    if not isinstance(records, list):
        return None
    out: list[tuple[str, float]] = []
    for row in records:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        date_iso = str(row[0]) if row[0] is not None else None
        try:
            pnl = float(row[1])
        except (TypeError, ValueError):
            continue
        if date_iso:
            out.append((date_iso, pnl))
    out.sort(key=lambda x: x[0])
    return out or None


def _normalize_settings(settings: dict) -> dict:
    """Map our internal settings keys to Brain's API field names."""
    base = {
        "instrumentType": "EQUITY",
        "region": "USA",
        # Universe is pinned to TOP3000 everywhere — never simulate on a narrower
        # universe regardless of what upstream settings (or the LLM) requested.
        "universe": "TOP3000",
        "delay": int(settings.get("delay", 1)),
        "decay": int(settings.get("decay", 4)),
        "neutralization": settings.get("neutralization", "Subindustry").upper() if isinstance(settings.get("neutralization", "Subindustry"), str) else "SUBINDUSTRY",
        "truncation": float(settings.get("truncation", 0.05)),
        "pasteurization": settings.get("pasteurization", "ON").upper(),
        "unitHandling": settings.get("unitHandling", "VERIFY"),
        "nanHandling": settings.get("nanHandling", "OFF"),
        "language": settings.get("language", "FASTEXPR"),
        "visualization": False,
    }
    return base


# ----------------------------------------------------------------- batch + log
def submit_alpha(session: BrainSession, alpha: dict) -> dict:
    """Simulate one alpha. Returns a record ready for logging."""
    t0 = time.time()
    result = session.simulate(alpha["expression"], alpha.get("settings", {}))
    elapsed = time.time() - t0
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(elapsed, 1),
        "alpha": {
            "name": alpha.get("name"),
            "expression": alpha["expression"],
            "archetype": alpha.get("archetype"),
            "logic": alpha.get("logic"),
            "settings": alpha.get("settings", {}),
            "expected_sharpe": alpha.get("expected_sharpe"),
            "paper_source": alpha.get("paper_source"),
        },
        "result": result,
    }


def _result_score(res: dict) -> tuple[int, float, float]:
    """Rank a sim result for picking the best sweep variant: passing first, then
    fitness, then sharpe. Errored/no-result configs sort last."""
    if not isinstance(res, dict) or res.get("status") not in ("OK", None):
        return (0, -1e9, -1e9)
    p = 1 if res.get("submission_pass") else 0
    fit = res.get("fitness")
    sh = res.get("sharpe")
    fit = fit if isinstance(fit, (int, float)) else -1e9
    sh = sh if isinstance(sh, (int, float)) else -1e9
    return (p, fit, sh)


def _norm_expr(e: str) -> str:
    """Whitespace-insensitive expression compare for inversion dedup."""
    return re.sub(r"\s+", "", e or "")


def _diagnose_result(res: dict) -> str:
    """Read an ACTUAL sim result and name the single dominant reason this lead
    missed the submission gate. This is what makes the adaptive loop 'understand'
    the failure instead of blindly sweeping — each mode maps to a different fix."""
    sh = res.get("sharpe")
    fit = res.get("fitness")
    tov = res.get("turnover_pct")
    cw = res.get("concentrated_weight") or {}
    sc = res.get("self_correlation") or {}
    if isinstance(sh, (int, float)) and sh < 0:
        return "sign_wrong"
    if isinstance(tov, (int, float)) and tov > 70:
        return "turnover_high"          # churning too fast — smooth the signal
    if isinstance(tov, (int, float)) and tov < 5:
        return "turnover_low"           # near-static — needs more daily variation
    if isinstance(cw, dict) and cw.get("result") == "FAIL":
        return "concentrated"           # weight piled on too few names — spread it
    # Check correlation BEFORE fitness so a passing-fitness but high-corr alpha
    # is correctly identified — expression-level tweaks can't fix correlation.
    if isinstance(sc, dict) and sc.get("result") == "FAIL":
        return "corr_self"              # too correlated with existing pool — need new fields
    if isinstance(fit, (int, float)) and fit < 1.0:
        return "fitness_low"            # edge is there, returns/turnover too thin
    return "other"


# For each diagnosed failure, the ORDER of fix-types to try first. Labels are
# matched as substrings against the candidate labels built in _plan_repairs
# ("set:*" = settings tweak, "expr:*" = expression rewrite). The fix that most
# directly addresses the diagnosis is tried first so a passing config is usually
# found within the first sim or two (the loop early-stops the moment one passes).
_REPAIR_PRIORITY = {
    # too much churn: smooth in the expression first, then lean on decay settings.
    "turnover_high": ("expr:decay", "set:hi_decay", "expr:win", "expr:double"),
    # too static: re-introduce daily variation via truncation (universe is fixed).
    "turnover_low":  ("set:hi_trunc",),
    # weight too concentrated: loosen per-name truncation.
    "concentrated":  ("set:lo_trunc",),
    # has edge, thin returns: concentrate (truncation), preserve return via Market
    # neutralization, then add a quality gate to the expression.
    "fitness_low":   ("set:hi_trunc", "set:neut", "expr:quality", "expr:subindustry"),
    # self-correlation too high: expression tweaks won't help — need totally different
    # fields. Sentinel triggers diversity regeneration in the forge layer.
    "corr_self":     ("__diversity_regen__",),
    "other":         (),
}


def _plan_repairs(alpha: dict, res: dict, budget: int) -> list[tuple[str, dict]]:
    """Build a RESULT-AWARE, priority-ordered list of repair candidates for a lead
    that has edge but missed a gate. Candidates tweak the settings (truncation /
    decay / neutralization / universe) AND the expression itself (smoothing, quality
    gate, window lengthening — via the deterministic NearMissMutator), and are
    ordered so the fixes targeting the diagnosed failure are tried first.

    Returns up to `budget` (label, candidate_alpha) pairs. Each candidate is a full
    alpha dict ready to simulate."""
    mode = _diagnose_result(res)
    base_settings = alpha.get("settings") or {}
    pool: list[tuple[str, dict]] = []

    # --- settings-level tweaks from the math-engine grid ---
    # NEVER change the universe during a repair: keep whatever the alpha was built
    # on (TOP3000 by default). Drop any grid variant that flips the universe.
    base_universe = base_settings.get("universe", "TOP3000")
    try:
        from math_engine import MathEngine
        for label, s in MathEngine().settings_variants(
            alpha["expression"], base_settings, max_variants=budget + 2
        ):
            if label == "base":
                continue
            if s.get("universe", base_universe) != base_universe:
                continue  # universe is fixed during repair — skip top1000 et al.
            pool.append((f"set:{label}", {**alpha, "settings": s}))
    except Exception as ex:  # noqa: BLE001
        print(f"  [repair] settings variants failed for {alpha.get('name')}: {ex}")

    # --- a low-truncation settings tweak: the grid only offers HIGHER truncation,
    #     but a CONCENTRATED_WEIGHT failure needs the opposite (spread weight out). ---
    if mode == "concentrated":
        cur = float(base_settings.get("truncation", 0.05) or 0.05)
        lo = {**base_settings, "truncation": round(max(0.01, cur / 2), 3)}
        pool.insert(0, ("set:lo_trunc", {**alpha, "settings": lo}))

    # --- expression-level repairs: NearMissMutator already maps failure mode →
    #     targeted rewrite (decay-wrap, quality gate, window lengthening, sign flip)
    #     and self-corrects neutralization. Reuse it so the loop can fix the ALPHA,
    #     not just the knobs. ---
    try:
        from near_miss_mutator import NearMissMutator
        nm = {
            "name": alpha.get("name"),
            "expression": alpha["expression"],
            "archetype": alpha.get("archetype"),
            "settings": base_settings,
            "sharpe": res.get("sharpe"),
            "fitness": res.get("fitness"),
            "turnover_pct": res.get("turnover_pct"),
        }
        for m in NearMissMutator().mutate(nm, max_mutations=budget + 2):
            pool.append((
                f"expr:{m.get('_mutation', 'mut')}",
                {**alpha, "expression": m["expression"], "settings": m["settings"]},
            ))
    except Exception as ex:  # noqa: BLE001
        print(f"  [repair] expression mutations failed for {alpha.get('name')}: {ex}")

    # --- order by the diagnosis: matching fix-types first, in their listed order;
    #     everything else after, original order preserved. ---
    priority = _REPAIR_PRIORITY.get(mode, ())

    def rank(item: tuple[str, dict]) -> int:
        label = item[0]
        for i, key in enumerate(priority):
            if key in label:
                return i
        return len(priority) + 1

    pool.sort(key=rank)

    # de-dup by (expression, normalized settings); keep first (highest-priority).
    seen: set[str] = set()
    plan: list[tuple[str, dict]] = []
    for label, cand in pool:
        key = _norm_expr(cand["expression"]) + "|" + json.dumps(
            _normalize_settings(cand.get("settings", {})), sort_keys=True
        )
        if key in seen:
            continue
        seen.add(key)
        plan.append((label, cand))
    return plan[:budget]


def simulate_alpha_adaptive(
    session: BrainSession,
    alpha: dict,
    sweep: int = 4,
    invert: bool = True,
) -> dict:
    """Budget-aware per-alpha simulation. Runs the baseline config, then:

      * if the baseline has a strong-negative sharpe (<= -0.3), simulate the
        SIGN-INVERTED expression — a wrong sign is a lead, not a failure
        (~31% of historical sims were negative-sharpe and discarded);
      * else if the baseline is a LEAD (sharpe >= 0.7 or fitness >= 0.4) but not
        yet passing, sweep a small fitness-oriented settings grid
        (math_engine.settings_variants) and keep whichever config scores best.

    Returns the single best record (by pass/fitness/sharpe), tagged with the
    winning variant in result['variant']. Sims run sequentially WITHIN an alpha,
    so total BRAIN concurrency is unchanged (still bounded by submit_batch's
    max_concurrent). Hopeless near-zero alphas cost just one sim.
    """
    targets = _load_targets()

    def _sim(a: dict, variant: str) -> dict:
        rec = submit_alpha(session, a)
        r = rec.setdefault("result", {})
        r["variant"] = variant
        r["submission_pass"] = meets_submission_targets(r, targets)
        return rec

    base_rec = _sim(alpha, "base")
    res = base_rec["result"]
    if res.get("status") not in ("OK", None):
        return base_rec  # errored — nothing to sweep

    best = base_rec
    sh = res.get("sharpe")

    # 1) Sign-inversion on strong-negative leads.
    if invert and not res.get("submission_pass") and isinstance(sh, (int, float)) and sh <= -0.3:
        try:
            from training.invert_refuted import _invert_expression
            inv_expr = _invert_expression(alpha["expression"])
        except Exception:
            inv_expr = None
        if inv_expr and _norm_expr(inv_expr) != _norm_expr(alpha["expression"]):
            inv_alpha = dict(alpha)
            inv_alpha["expression"] = inv_expr
            inv_alpha["name"] = f"{alpha.get('name') or 'alpha'}_inv"
            inv_rec = _sim(inv_alpha, "inverted")
            if _result_score(inv_rec["result"]) > _result_score(best["result"]):
                best = inv_rec
        return best  # sign was the issue; don't also sweep settings

    # 2) Result-aware repair on leads that have edge but miss a gate. Instead of a
    #    blind settings grid, we DIAGNOSE why the base config missed (turnover too
    #    high, weight too concentrated, fitness too thin, ...) and try the fixes that
    #    target that failure first — tweaking the settings AND/OR the expression.
    if sweep and not res.get("submission_pass"):
        fit = res.get("fitness")
        fit = fit if isinstance(fit, (int, float)) else -1.0
        is_lead = (isinstance(sh, (int, float)) and sh >= 0.7) or fit >= 0.4
        if is_lead:
            mode = _diagnose_result(res)
            plan = _plan_repairs(alpha, res, budget=sweep)
            print(f"  [repair] {alpha.get('name')}: diagnosed '{mode}' "
                  f"(sh={sh}, fit={res.get('fitness')}, tov={res.get('turnover_pct')}) "
                  f"→ trying {len(plan)} targeted fix(es): {[lbl for lbl, _ in plan]}")
            for label, cand in plan:
                vrec = _sim(cand, label)
                if _result_score(vrec["result"]) > _result_score(best["result"]):
                    best = vrec
                if best["result"].get("submission_pass"):
                    print(f"  [repair] {alpha.get('name')}: '{label}' cleared the gate "
                          f"(fit={best['result'].get('fitness')}) — stopping.")
                    break  # a passing config is good enough — stop spending budget
    return best


def submit_batch(
    alphas: list[dict],
    max_concurrent: int = 10,
    session: "BrainSession | None" = None,
    sweep: int = 0,
    invert: bool = True,
) -> list[dict]:
    """Run simulations concurrently. Auto-logs each result.

    Pass an existing authenticated `session` to reuse it across calls — this is
    what rounds mode does, so the BRAIN Persona biometric is completed ONCE rather
    than re-challenged on every round (a fresh session has no auth cookie). When
    omitted, a new session is created and authenticated for this batch.

    When `sweep` > 0, each alpha is simulated adaptively (settings sweep on leads,
    sign-inversion on strong-negatives) via simulate_alpha_adaptive and only the
    best-scoring config is recorded.
    """
    if not alphas:
        return []
    if session is None:
        creds = load_credentials()
        session = BrainSession(creds["email"], creds["password"])

    targets = _load_targets()
    records: list[dict] = []
    log_lock = threading.Lock()

    def _do_one(alpha):
        try:
            if sweep:
                rec = simulate_alpha_adaptive(session, alpha, sweep=sweep, invert=invert)
            else:
                rec = submit_alpha(session, alpha)
        except Exception as ex:
            rec = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "alpha": {"name": alpha.get("name"), "expression": alpha.get("expression")},
                "result": {"status": "EXCEPTION", "error": f"{type(ex).__name__}: {ex}"},
            }
        rec.setdefault("result", {})
        rec["result"]["submission_pass"] = meets_submission_targets(rec["result"], targets)
        # Hypothesis-loop: compare prediction to result and write reflection back into
        # data/hypotheses/<file>.json. Best-effort — never let reflection failures kill a sim.
        try:
            from training.reflect import reflect_on_record  # local import to avoid hard dep
            reflection = reflect_on_record(rec["alpha"], rec["result"])
            if reflection:
                rec["reflection"] = reflection
        except Exception as ex:  # noqa: BLE001
            print(f"  [reflect] skipped for {alpha.get('name')}: {type(ex).__name__}: {ex}")
        with log_lock:
            _log_result(rec, targets)
        return rec

    # Periodic progress reporter so users can see the script is alive while
    # waiting on the slowest simulation (each can take 2-5 min on BRAIN's side).
    total = len(alphas)
    pending_names = {a.get("name") or a.get("expression", "?")[:30] for a in alphas}
    pending_lock = threading.Lock()
    stop_watcher = threading.Event()
    t_start = time.time()

    def _watcher() -> None:
        while not stop_watcher.wait(30):
            with pending_lock:
                remaining = sorted(pending_names)
            if not remaining:
                continue
            elapsed = int(time.time() - t_start)
            print(f"  [batch] {len(remaining)}/{total} still simulating ({elapsed}s elapsed): "
                  f"{', '.join(remaining)[:120]}", flush=True)

    watcher = threading.Thread(target=_watcher, daemon=True)
    watcher.start()

    print(f"  [batch] submitted {total} alphas to BRAIN (max_concurrent={max_concurrent})", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
            futures = {ex.submit(_do_one, a): a for a in alphas}
            for fut in as_completed(futures):
                rec = fut.result()
                records.append(rec)
                with pending_lock:
                    pending_names.discard(rec["alpha"].get("name") or rec["alpha"].get("expression", "?")[:30])
                _print_brief(rec)
    finally:
        stop_watcher.set()
        watcher.join(timeout=1)

    return records


def _print_brief(rec: dict) -> None:
    a = rec["alpha"]
    r = rec["result"]
    name = (a.get("name") or "?")[:30]
    status = r.get("status", "?")
    if status == "OK":
        p = "PASS" if r.get("submission_pass") else "FAIL"
        print(f"  [OK ] {name:<30s} pass={p:<4s} sharpe={r.get('sharpe', '?')!s:>6s} fitness={r.get('fitness', '?')!s:>6s} turnover={r.get('turnover_pct', '?')!s:>6s}% (elapsed {rec.get('elapsed_s', '?')}s)")
    else:
        err = (r.get("error") or "")[:100]
        print(f"  [{status[:4]:<4s}] {name:<30s} {err}")


def _log_result(rec: dict, targets: dict) -> None:
    """Append rec to memory.json (if passes targets) or feedback.json (otherwise)."""
    r = rec.get("result", {})
    a = rec.get("alpha", {})

    sharpe_min = float(targets.get("sharpe_min", 1.25))
    fitness_min = float(targets.get("fitness_min", 1.0))
    turnover_min = float(targets.get("turnover_min_pct", 1))
    turnover_max = float(targets.get("turnover_max_pct", 70))

    passing = bool(r.get("submission_pass")) if "submission_pass" in r else meets_submission_targets(r, targets)

    # Per-sim lesson: written for every result (pass AND fail) the moment it
    # completes. The alpha_generator surfaces the most-recent lessons at the
    # top of the next round's prompt so the LLM learns per-alpha, not per-batch.
    try:
        _append_lesson(a, r)
    except Exception as ex:  # noqa: BLE001
        print(f"  [lesson] skipped for {a.get('name')}: {type(ex).__name__}: {ex}")

    if passing:
        store = _load_json(MEMORY_PATH, {"entries": []})
        store.setdefault("entries", [])

        # Stat-tuple dedup: catch the "superset expression with a no-op
        # cross-leg" case. When the LLM proposes E and E*rank(static_field),
        # BRAIN's truncation+neutralization can produce IDENTICAL sharpe,
        # fitness, turnover — the cross-leg adds nothing. Without this check
        # both alphas land in memory.json and inflate the pool with a 1.000-
        # correlation duplicate. Tolerance: 0.005 on each axis (≈ sharpe
        # difference 0.5pp, fitness 0.5pp, turnover 0.5pp).
        new_sh = r.get("sharpe")
        new_ft = r.get("fitness")
        new_tv = r.get("turnover_pct")
        duplicate_of: str | None = None
        if all(isinstance(x, (int, float)) for x in (new_sh, new_ft, new_tv)):
            for existing in store["entries"]:
                e_sh = existing.get("sharpe")
                e_ft = existing.get("fitness")
                e_tv = existing.get("turnover_pct")
                if not all(isinstance(x, (int, float)) for x in (e_sh, e_ft, e_tv)):
                    continue
                if (abs(new_sh - e_sh) < 0.005
                        and abs(new_ft - e_ft) < 0.005
                        and abs(new_tv - e_tv) < 0.005):
                    duplicate_of = existing.get("alpha_id") or existing.get("name") or "?"
                    break

        if duplicate_of is not None:
            # Reroute to feedback.json with a specific failure reason. The
            # alpha really did simulate cleanly — it's just structurally
            # redundant with an existing graduate.
            print(f"  [memory-dedup] {a.get('name')} duplicates {duplicate_of} "
                  f"(sh={new_sh}/ft={new_ft}/tov={new_tv}) — routing to feedback")
            reason = f"duplicate_of_graduate {duplicate_of}: identical (sh, ft, tov) within 0.005"
            store_fb = _load_json(FEEDBACK_PATH, {"entries": []})
            store_fb.setdefault("entries", [])
            store_fb["entries"].append({
                "timestamp": rec["timestamp"],
                "name": a.get("name"),
                "expression": a.get("expression"),
                "archetype": a.get("archetype"),
                "settings": a.get("settings", {}),
                "sharpe": r.get("sharpe"),
                "fitness": r.get("fitness"),
                "turnover_pct": r.get("turnover_pct"),
                "returns": r.get("returns"),
                "status": "OK",
                "error": None,
                "reason": reason,
                "alpha_id": r.get("alpha_id"),
                "submission_pass": False,
                "duplicate_of": duplicate_of,
            })
            FEEDBACK_PATH.write_text(json.dumps(store_fb, indent=2) + "\n")
            return

        store["entries"].append({
            "timestamp": rec["timestamp"],
            "name": a.get("name"),
            "expression": a.get("expression"),
            "archetype": a.get("archetype"),
            "settings": a.get("settings", {}),
            "sharpe": r.get("sharpe"),
            "fitness": r.get("fitness"),
            "turnover_pct": r.get("turnover_pct"),
            "returns": r.get("returns"),
            "drawdown": r.get("drawdown"),
            "alpha_id": r.get("alpha_id"),
        })
        MEMORY_PATH.write_text(json.dumps(store, indent=2) + "\n")
    else:
        reason = _classify_failure(r, sharpe_min, fitness_min, turnover_min, turnover_max)
        # If BRAIN returned an operator-input rejection, learn the (op, field) pair
        # so future generations skip it pre-flight.
        _record_bad_pair(a.get("expression"), r.get("error"))
        store = _load_json(FEEDBACK_PATH, {"entries": []})
        store.setdefault("entries", [])
        store["entries"].append({
            "timestamp": rec["timestamp"],
            "name": a.get("name"),
            "expression": a.get("expression"),
            "archetype": a.get("archetype"),
            "settings": a.get("settings", {}),
            "sharpe": r.get("sharpe"),
            "fitness": r.get("fitness"),
            "turnover_pct": r.get("turnover_pct"),
            "returns": r.get("returns"),
            "reason": reason,
            "status": r.get("status"),
            "error": r.get("error"),
            "alpha_id": r.get("alpha_id"),
        })
        FEEDBACK_PATH.write_text(json.dumps(store, indent=2) + "\n")


# Lock to avoid concurrent corruption when threads write the same JSON file.
_BAD_PAIRS_LOCK = threading.Lock()
_LESSONS_LOCK = threading.Lock()


def _synthesize_lesson(alpha: dict, result: dict) -> str:
    """Compress a sim outcome into a one-line directive for the next round.
    Generic enough to read at a glance, specific enough to be actionable.

    Examples produced:
      "PASS sh=2.06 fit=1.17 — outer ts_decay_linear(rank(...), 6) on quality leg works; replicate structure"
      "FAIL fit=0.74 tov=39.6% — sharpe close but fitness short; lower turnover, keep sign"
      "FAIL sh=-0.27 — wrong sign; INVERT the leading - on this composition"
      "REJECT (parse) — rank() does not accept event-typed nws18_qcm; route through ts_corr"
    """
    expr = (alpha.get("expression") or "")
    arch = alpha.get("archetype") or "novel"
    status = result.get("status")
    sh = result.get("sharpe")
    fit = result.get("fitness")
    tov = result.get("turnover_pct")

    # Non-OK status (parse/sim error)
    if status != "OK":
        err = (result.get("error") or "")[:120]
        return f"REJECT [{arch}] — {status}: {err}"

    sh_s = f"sh={sh:.2f}" if isinstance(sh, (int, float)) else "sh=?"
    fit_s = f"fit={fit:.2f}" if isinstance(fit, (int, float)) else "fit=?"
    tov_s = f"tov={tov:.1f}%" if isinstance(tov, (int, float)) else "tov=?"

    if result.get("submission_pass"):
        return f"PASS [{arch}] {sh_s} {fit_s} {tov_s} — replicate structure, vary fields/windows"

    # Actionable failure dispositions, most-informative first.
    if isinstance(tov, (int, float)) and tov > 70:
        return f"FAIL [{arch}] {sh_s} {fit_s} {tov_s} — turnover overshoot; wrap fast leg in ts_decay_linear(...,6-8)"
    if isinstance(tov, (int, float)) and tov < 1:
        return f"FAIL [{arch}] {sh_s} {fit_s} {tov_s} — turnover dead; add daily-varying leg (returns, ts_delta)"
    if isinstance(sh, (int, float)) and sh <= -0.5:
        return f"FAIL [{arch}] {sh_s} {fit_s} — sign wrong; try inverting (- in front of expression)"
    if isinstance(sh, (int, float)) and -0.5 < sh < 0.3:
        return f"FAIL [{arch}] {sh_s} {fit_s} — zero edge; DROP this composition pattern"
    if (isinstance(sh, (int, float)) and sh >= 1.25 and
            isinstance(fit, (int, float)) and fit < 1.0):
        return f"NEAR [{arch}] {sh_s} {fit_s} {tov_s} — sharpe passes, fitness short; lower turnover, keep sign"
    if isinstance(sh, (int, float)) and 0.5 <= sh < 1.25:
        return f"WEAK [{arch}] {sh_s} {fit_s} {tov_s} — right direction; tighten with smoothing or stricter neutralization"
    return f"FAIL [{arch}] {sh_s} {fit_s} {tov_s} — below targets; restructure"


def _append_lesson(alpha: dict, result: dict) -> None:
    """Append a one-line structured lesson to data/lessons.jsonl. JSONL is used
    so threaded writes can simply append without re-reading + re-serializing the
    whole file. The alpha_generator surfaces the last N lessons in the prompt."""
    expr = alpha.get("expression") or ""
    # Compact summary kept under ~150 chars so the prompt block stays scannable.
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "name": (alpha.get("name") or "")[:48],
        "archetype": alpha.get("archetype") or "novel",
        "expression": expr[:200],
        "sharpe": result.get("sharpe"),
        "fitness": result.get("fitness"),
        "turnover_pct": result.get("turnover_pct"),
        "status": result.get("status"),
        "submission_pass": bool(result.get("submission_pass")),
        "mechanism_family": classify_mechanism_family(expr),
        "mechanism_pass": (
            bool(result.get("submission_pass"))
            if "submission_pass" in result
            else meets_submission_targets(result)
        ),
        "lesson": _synthesize_lesson(alpha, result),
    }
    with _LESSONS_LOCK:
        with LESSONS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

# BRAIN error patterns we can mine for (operator, field) pairs.
# Examples observed:
#   "Operator rank does not support event inputs."
#   "Operator ts_decay_linear does not support event inputs."
_OP_REJECT_RE = re.compile(r"Operator\s+([A-Za-z_][A-Za-z_0-9]*)\s+does not support", re.IGNORECASE)


def _direct_wrapped_fields(expression: str, op_name: str) -> list[str]:
    """Return the bare field tokens that `op_name(...)` directly wraps in expression.
    Only counts cases where the first arg is a single field identifier (possibly
    negated): rank(nws18_qcm), ts_decay_linear(scl12_buzz, 4)."""
    out: list[str] = []
    for m in re.finditer(rf"\b{re.escape(op_name)}\s*\(", expression):
        depth = 0
        i = m.end() - 1
        start = i + 1
        while i < len(expression):
            c = expression[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    inner = expression[start:i]
                    # First top-level arg
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
                    arg = inner[:first_end].strip().lstrip("+-").strip()
                    if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", arg):
                        out.append(arg)
                    break
            i += 1
    return out


def _record_bad_pair(expression: str | None, error: str | None) -> None:
    """If BRAIN's error names an operator that 'does not support ... inputs',
    walk the expression for fields directly wrapped by that operator and persist
    each (operator, field) pair to data/bad_operator_field_pairs.json. The
    alpha_generator and math_engine pre-flight loaders read this file, so future
    runs reject the same expressions before they hit BRAIN."""
    if not isinstance(expression, str) or not isinstance(error, str):
        return
    m = _OP_REJECT_RE.search(error)
    if not m:
        return
    op_name = m.group(1)
    fields = _direct_wrapped_fields(expression, op_name)
    if not fields:
        return
    with _BAD_PAIRS_LOCK:
        doc = _load_json(BAD_PAIRS_PATH, {"pairs": []})
        existing = {(p.get("operator"), p.get("field")) for p in doc.get("pairs", []) if isinstance(p, dict)}
        new_entries = 0
        for fld in fields:
            key = (op_name, fld)
            if key in existing:
                continue
            doc["pairs"].append({
                "operator": op_name,
                "field": fld,
                "error": error[:200],
                "first_seen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            existing.add(key)
            new_entries += 1
        if new_entries:
            BAD_PAIRS_PATH.write_text(json.dumps(doc, indent=2) + "\n")


def _classify_failure(r: dict, sharpe_min: float, fitness_min: float, turnover_min: float, turnover_max: float) -> str:
    if r.get("status") != "OK":
        return f"{r.get('status', 'UNKNOWN')}: {r.get('error', '')}"[:200]
    sharpe = r.get("sharpe")
    fitness = r.get("fitness")
    turnover = r.get("turnover_pct")
    reasons = []
    if not isinstance(sharpe, (int, float)) or sharpe < sharpe_min:
        reasons.append(f"sharpe={sharpe} < {sharpe_min}")
    if not isinstance(fitness, (int, float)) or fitness < fitness_min:
        reasons.append(f"fitness={fitness} < {fitness_min}")
    if isinstance(turnover, (int, float)):
        if turnover < turnover_min:
            reasons.append(f"turnover={turnover:.1f}% < {turnover_min}%")
        elif turnover > turnover_max:
            reasons.append(f"turnover={turnover:.1f}% > {turnover_max}%")
    return "; ".join(reasons) or "below targets"


# ----------------------------------------------------------------- CLI
def _cmd_smoke() -> int:
    """Authenticate and simulate one confirmed_working alpha."""
    try:
        creds = load_credentials()
    except (FileNotFoundError, ValueError) as ex:
        print(f"ERROR: {ex}", file=sys.stderr)
        return 1

    kb = json.loads(KB_PATH.read_text())
    working = kb.get("confirmed_working", [])
    if not working:
        print("ERROR: no confirmed_working alphas in brain_kb.json", file=sys.stderr)
        return 1

    pick = working[-1]  # use the alpha101-style one with no scale()
    print(f"Smoke test: authenticating as {creds['email']}...")
    try:
        session = BrainSession(creds["email"], creds["password"])
    except Exception as ex:
        print(f"AUTH FAILED: {ex}", file=sys.stderr)
        return 1
    print("Authenticated.")
    print(f"Simulating: {pick['name']}")
    print(f"  expression: {pick['expression']}")
    print(f"  settings:   {pick['settings']}")

    t0 = time.time()
    result = session.simulate(pick["expression"], pick["settings"])
    elapsed = time.time() - t0

    print(f"\nResult (elapsed {elapsed:.1f}s):")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") == "OK" else 2


def _cmd_run(args: list[str]) -> int:
    """Generate N alphas and submit them.

    Usage: python brain_api.py run [-n N] [--archetype X] [--max-concurrent K]
    """
    import argparse
    ap = argparse.ArgumentParser(prog="brain_api.py run")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--archetype", default=None)
    ap.add_argument("--max-concurrent", type=int, default=10)
    opts = ap.parse_args(args)

    from alpha_generator import generate_batch
    print(f"Generating {opts.n} alphas (archetype={opts.archetype or 'free'})...")
    alphas = generate_batch(n=opts.n, archetype=opts.archetype)
    print(f"Got {len(alphas)} alphas. Submitting to Brain (max_concurrent={opts.max_concurrent})...\n")

    records = submit_batch(alphas, max_concurrent=opts.max_concurrent)

    ok = sum(1 for r in records if r["result"].get("status") == "OK")
    passing = sum(1 for r in records
                  if r["result"].get("status") == "OK"
                  and isinstance(r["result"].get("sharpe"), (int, float))
                  and r["result"]["sharpe"] >= 1.25)
    print(f"\n=== Summary ===")
    print(f"  submitted:        {len(records)}")
    print(f"  simulated OK:     {ok}")
    print(f"  passing targets:  {passing}")
    print(f"  → logged to data/memory.json (passing) and data/feedback.json (failing)")
    return 0


def _cmd_biometric() -> int:
    """Complete BRAIN's interactive biometric (Persona) verification.

    Run this once in your terminal; verification stays valid on the account for
    a few hours, after which `submit`/`run` authenticate without prompting.
    """
    creds = load_credentials()
    print(f"Authenticating as {creds['email']}...")
    try:
        BrainSession(creds["email"], creds["password"])
    except PermissionError as ex:
        print(f"FAILED: {ex}", file=sys.stderr)
        return 1
    print("Authenticated. Biometric verification is now valid for this account.")
    return 0


def _cmd_submit_file(path: str) -> int:
    """Submit alphas from a JSON file (list of alpha dicts or {alphas:[...]})."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        print(f"ERROR: file not found: {p}", file=sys.stderr)
        return 1
    data = json.loads(p.read_text())
    alphas = data if isinstance(data, list) else data.get("alphas", [])
    if not alphas:
        print("ERROR: no alphas found in file", file=sys.stderr)
        return 1
    print(f"Submitting {len(alphas)} alphas from {p.name}...")
    records = submit_batch(alphas)
    print(f"\nDone — {len(records)} records logged.")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("Commands:")
        print("  biometric          Complete BRAIN's interactive Persona verification")
        print("  smoke              Authenticate and simulate one alpha (credentials test)")
        print("  run [-n N]         Generate N alphas and submit them")
        print("  submit <file>      Submit alphas from a JSON file")
        return 0
    cmd, *rest = argv
    if cmd == "biometric":
        return _cmd_biometric()
    if cmd == "smoke":
        return _cmd_smoke()
    if cmd == "run":
        return _cmd_run(rest)
    if cmd == "submit":
        if not rest:
            print("usage: brain_api.py submit <file.json>", file=sys.stderr)
            return 1
        return _cmd_submit_file(rest[0])
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
