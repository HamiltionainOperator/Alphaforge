"""Targeted test of the 3 leads from the g4 run:
  e1  sh -1.13  -> invert (drop the -1)               ≈ +1.13
  e4  sh -1.31  -> invert                             ≈ +1.31
  e2  sh 1.52 / fit 0.85 / tov 72.6%  -> harden turnover (the strongest lead)

e2's turnover is just over the 70% cap, which is what drags fitness below 1.0.
The fast legs are raw `returns` + close/vwap. We try three turnover cuts:
  - slower decay (6 -> 12)
  - smooth the raw returns leg (returns -> ts_mean(returns, 5))
  - tighter neutralization (sector -> subindustry)
"""
from brain_api import BrainSession, load_credentials, submit_batch

E1 = "-1 * group_neutralize(rank(multiply(multiply(rank(parkinson_volatility_10), rank(subtract(0, ts_mean(fn_comp_non_opt_grants_a, 5)))), rank(subtract(0, ts_mean(returns, 3))))), subindustry)"
E4 = "-1 * group_neutralize(rank(add(rank(ts_rank(divide(actual_eps_value_quarterly, ts_delay(high, 252)), 252)), rank(subtract(0, ts_mean(returns, 3))))), sector)"

# e2 base: -1 * group_neutralize(rank(add(returns, scale(ts_decay_linear(divide(close, vwap), 6)))), sector)
leads = [
    # --- inversions: drop the leading -1 ---
    {"name": "e1_inverted",
     "expression": "group_neutralize(rank(multiply(multiply(rank(parkinson_volatility_10), rank(subtract(0, ts_mean(fn_comp_non_opt_grants_a, 5)))), rank(subtract(0, ts_mean(returns, 3))))), subindustry)",
     "settings": {"neutralization": "None", "decay": 4, "truncation": 0.05}},
    {"name": "e4_inverted",
     "expression": "group_neutralize(rank(add(rank(ts_rank(divide(actual_eps_value_quarterly, ts_delay(high, 252)), 252)), rank(subtract(0, ts_mean(returns, 3))))), sector)",
     "settings": {"neutralization": "None", "decay": 4, "truncation": 0.05}},

    # --- e2 turnover-hardening variants (keep the +sign, it's already 1.52) ---
    {"name": "e2_slowdecay12",
     "expression": "-1 * group_neutralize(rank(add(returns, scale(ts_decay_linear(divide(close, vwap), 12)))), sector)",
     "settings": {"neutralization": "None", "decay": 12, "truncation": 0.05}},
    {"name": "e2_smooth_returns",
     "expression": "-1 * group_neutralize(rank(add(ts_mean(returns, 5), scale(ts_decay_linear(divide(close, vwap), 10)))), sector)",
     "settings": {"neutralization": "None", "decay": 10, "truncation": 0.05}},
    {"name": "e2_subind_decay10",
     "expression": "-1 * group_neutralize(rank(add(ts_mean(returns, 5), scale(ts_decay_linear(divide(close, vwap), 10)))), subindustry)",
     "settings": {"neutralization": "None", "decay": 10, "truncation": 0.05}},
]

if __name__ == "__main__":
    creds = load_credentials()
    session = BrainSession(creds["email"], creds["password"])
    print(f"Testing {len(leads)} leads (2 inversions + 3 e2-hardenings)...")
    records = submit_batch(leads, session=session, max_concurrent=3, sweep=0)
    print("\n=== RESULTS ===")
    for rec in sorted(records, key=lambda r: -(r["result"].get("fitness") or -9)):
        r = rec["result"]
        p = "PASS" if r.get("submission_pass") else "fail"
        print(f"  [{p}] {rec['alpha']['name']:<22} sh={r.get('sharpe')} fit={r.get('fitness')} tov={r.get('turnover_pct')}")
