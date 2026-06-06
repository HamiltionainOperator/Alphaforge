#!/usr/bin/env python3
"""
refresh_brain_data.py — Refresh Brain field catalog and personal alpha history.

Usage:
    python scripts/refresh_brain_data.py [--fields-only] [--alphas-only]

Requires credentials.json in the project root with Brain email + password.
Writes:
    data/brain_docs/fields_full.json  — updated field catalog with alpha_count stats
    data/brain_docs/my_alphas.json    — list of your simulated alphas from Brain

After running, restart the backend server or call POST /api/intelligence/refresh
to reload the field intelligence cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root or scripts/ directory.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from brain_api import BrainSession, load_credentials


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh Brain field catalog and alpha history")
    parser.add_argument("--fields-only", action="store_true", help="Only refresh field catalog")
    parser.add_argument("--alphas-only", action="store_true", help="Only fetch personal alphas")
    parser.add_argument("--region",   default="USA",    help="Brain region (default: USA)")
    parser.add_argument("--universe", default="TOP3000", help="Universe (default: TOP3000)")
    parser.add_argument("--delay",    default=1, type=int, help="Delay (default: 1)")
    args = parser.parse_args()

    try:
        creds = load_credentials()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}")
        return 1

    print(f"[refresh] authenticating as {creds['email']}...")
    session = BrainSession(creds["email"], creds["password"])

    # Read old field count for comparison.
    fields_path = _ROOT / "data" / "brain_docs" / "fields_full.json"
    old_field_count = 0
    old_fetched_at = None
    try:
        old = json.loads(fields_path.read_text())
        old_field_count = len(old.get("fields", {}))
        old_fetched_at = old.get("fetched_at", "never")
    except (OSError, json.JSONDecodeError):
        pass

    if not args.alphas_only:
        print(f"[refresh] fetching field catalog "
              f"(region={args.region}, universe={args.universe}, delay={args.delay})...")
        result = session.fetch_data_fields(
            region=args.region,
            universe=args.universe,
            delay=args.delay,
        )
        if result.get("saved"):
            print(f"[refresh] fields updated: {old_field_count} → {result['total']} fields "
                  f"(was fetched_at={old_fetched_at})")
        else:
            print(f"[refresh] field fetch failed: {result.get('error') or result}")

    if not args.fields_only:
        print("[refresh] fetching personal alpha history...")
        alphas = session.list_my_alphas(limit=500)
        if alphas:
            # Print a brief summary.
            pass_count = sum(1 for a in alphas if a.get("is", {}).get("sharpe", 0) >= 1.25)
            print(f"[refresh] fetched {len(alphas)} personal alphas "
                  f"({pass_count} with Sharpe ≥ 1.25)")
        else:
            print("[refresh] no personal alphas found (or endpoint unavailable)")

    print("[refresh] done. Restart backend to reload field intelligence cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
