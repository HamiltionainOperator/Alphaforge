import { useState, useCallback, useRef } from "react";

// ── Brain KB ground truth ──────────────────────────────────────────────────
const ARCHETYPES = [
  { id: "reversal",         label: "Reversal",          color: "#00d4ff", icon: "↩" },
  { id: "fundamental",      label: "Fundamental",        color: "#00ff9d", icon: "📊" },
  { id: "volatility",       label: "Volatility",         color: "#ff6b35", icon: "〜" },
  { id: "options_implied",  label: "Options Implied",    color: "#bf5fff", icon: "Δ" },
  { id: "microstructure",   label: "Microstructure",     color: "#ffd700", icon: "⚡" },
  { id: "earnings_event",   label: "Earnings Event",     color: "#ff3d71", icon: "★" },
  { id: "factor_residual",  label: "Factor Residual",    color: "#17ead9", icon: "⊥" },
  { id: "dispersion",       label: "Dispersion",         color: "#f5a623", icon: "σ" },
  { id: "analyst_revision", label: "Analyst Revision",   color: "#6ee7b7", icon: "↑" },
  { id: "novel",            label: "Novel",              color: "#e879f9", icon: "✦" },
];

const SYSTEM_PROMPT = `You are an elite WorldQuant BRAIN alpha researcher. Your job is to generate ORIGINAL, TESTABLE trading alpha hypotheses for the BRAIN platform.

BRAIN RULES (non-negotiable):
- Valid operators ONLY: rank(), zscore(), ts_mean(x,d), ts_std_dev(x,d), ts_delta(x,d), ts_corr(x,y,d), ts_covariance(x,y,d), ts_rank(x,d), ts_arg_max(x,d), ts_arg_min(x,d), ts_decay_linear(x,d), ts_zscore(x,d), ts_sum(x,d), ts_product(x,d), ts_delay(x,d), group_neutralize(x,group), group_zscore(x,group), group_rank(x,group), group_mean(x,w,group), abs(), log(), sign(), signed_power(x,e), pow(x,e), if_else(cond,a,b), scale()
- FORBIDDEN operators: SMA, EMA, RSI, MACD, ATR, ts_ir, decay_linear, ts_skewness, ts_kurtosis, ts_returns, winsorize, ts_min, ts_max
- Valid price/volume fields: close, open, high, low, vwap, volume, returns, sharesout
- Valid fundamental fields: ebit, ebitda, sales, assets, equity, debt, liabilities, capex
- Valid options fields: implied_volatility_call_10/20/30/60/90/120/180/270/360/720/1080, implied_volatility_put_10/20/30/60/90/120/180/270/360/720/1080, out_of_money_put_call_ratio
- Valid analyst fields: anl4_eps_rev_up_1m, analyst_revision_rank_derivative
- Valid earnings fields: abnormal_return_earnings_release, change_in_eps_surprise, sales_surprise_score
- Valid dispersion fields: fy1_eps_estimate_dispersion_2, fy2_eps_estimate_dispersion, sales_estimate_dispersion
- FORBIDDEN fields: enterprise_value, cap, adv20, adv60, market_cap, fnd6_*, dividends, industry, betweenness
- Use sharesout*close for market cap. Use ts_mean(volume,N) for dollar volume.
- group_mean(x, weight, group) takes EXACTLY 3 args
- group_zscore(x, group) takes EXACTLY 2 args
- if_else branches MUST have same units (don't mix price with unitless)
- Never add scalar epsilon to field denominators under unitHandling=VERIFY
- Cubic moment: ALWAYS negate (-ts_mean(returns*returns*returns,21)) — positive cubic buys lottery stocks and destroys Sharpe
- Raw ts_delta(close,N) without /close is dollar-denominated and biased — always normalize

FITNESS TARGETS:
- Sharpe ≥ 1.25, Fitness ≥ 1.0
- Turnover: 12.5% – 70% (below 12.5% fails fitness floor)
- Self-correlation < 0.7

SETTINGS RULES:
- reversal/microstructure: decay=4, truncation=0.08, neutralization=Subindustry
- fundamental: universe=TOP1000, decay=4, truncation=0.01
- options/volatility: decay=4-6, truncation=0.05
- If group_neutralize() is INSIDE expression → set neutralization=Market (avoid double demean)
- delay=0 only for genuinely intraday signals (close-vs-vwap, IV surface)
- delay=1 for all slow/fundamental signals

CONFIRMED WINNERS for inspiration (do NOT copy, use as structural reference):
1. scale(group_neutralize(zscore(-ts_mean(returns*returns*returns,21))*rank(ts_sum(volume,5)/ts_mean(volume,60)),sector)) → Sharpe 2.99
2. -1*rank(ts_corr(returns,volume,10)) → Sharpe 2.29  
3. rank(-ts_delta(close,1)/ts_mean(close,252))*rank(ts_sum(volume,5)/ts_mean(volume,60)) → Sharpe 2.19
4. -rank(ebit/capex) at delay=0 → Sharpe 2.02

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown, no preamble:
{
  "hypotheses": [
    {
      "id": "hyp_001",
      "title": "Short descriptive title",
      "archetype": "one of: reversal|fundamental|volatility|options_implied|microstructure|earnings_event|factor_residual|dispersion|analyst_revision|novel",
      "claim": "Stocks with [X] will [outperform/underperform] over [N days] because [mechanism]",
      "mechanism": "The structural/behavioral/informational friction that creates the edge",
      "regime_guard": "Specific condition under which this alpha FAILS (be precise, not generic)",
      "expression": "The exact FastExpr expression using only valid Brain operators and fields",
      "settings": {
        "universe": "TOP3000",
        "neutralization": "Subindustry",
        "decay": 4,
        "truncation": 0.05,
        "delay": 1
      },
      "expected_sharpe": "1.5-2.0",
      "expected_turnover_pct": "20-40",
      "novelty_score": 8,
      "novelty_reason": "Why this is different from known winners",
      "risk_flags": ["any warnings about this expression"],
      "fields_used": ["field1", "field2"]
    }
  ]
}`;

function buildUserPrompt(archetype, theme, constraints, count) {
  const archetypeInfo = {
    reversal: "short-horizon mean reversion, price-volume comovement, overextension signals",
    fundamental: "valuation ratios, profitability (ebit/assets, ebit/capex), leverage, accruals — sector-neutralized",
    volatility: "realized vol cross-section, vol regime shifts, IVOL mean-reversion",
    options_implied: "put-call IV skew (implied_volatility_put_30 - implied_volatility_call_30), term-structure spread (IV_30 - IV_180), vol risk premium (IV vs realized)",
    microstructure: "close-vwap divergence, order flow imbalance, intraday range signals, volume burst detection",
    earnings_event: "PEAD using abnormal_return_earnings_release, change_in_eps_surprise, sales_surprise_score",
    factor_residual: "group_zscore to build within-sector residuals, paired with a second cross-sectional signal",
    dispersion: "analyst estimate dispersion (fy1_eps_estimate_dispersion_2), IV time-series dispersion — short high-dispersion stocks",
    analyst_revision: "EPS/sales revision signals via anl4_eps_rev_up_1m or analyst_revision_rank_derivative",
    novel: "compound mechanisms: lottery aversion × volume, residual-of-residual, options × fundamentals composites",
  };

  const focus = archetype !== "any"
    ? `FOCUS ARCHETYPE: ${archetype}. Mechanism hints: ${archetypeInfo[archetype] || "any valid mechanism"}.`
    : `Generate across DIVERSE archetypes — do NOT cluster around reversal or simple momentum. Spread across: options_implied, fundamental, earnings_event, factor_residual, dispersion.`;

  return `Generate ${count} ORIGINAL, NOVEL alpha hypotheses for WorldQuant BRAIN.

${focus}

THEME CONSTRAINT: ${theme || "No specific theme — maximize originality and diversity."}

ADDITIONAL CONSTRAINTS: ${constraints || "None."}

CRITICAL NOVELTY RULES:
1. Do NOT generate simple ts_delta(close,1) reversal unless gated by a completely novel second signal
2. Do NOT generate plain ts_std_dev(returns,21) volatility — must have a novel interaction
3. PREFER unexplored combinations: options × fundamentals, earnings × microstructure, dispersion × reversal
4. Each hypothesis must use at least one field that is NOT close/returns/volume
5. At least one hypothesis must use options fields (implied_volatility_*)
6. Novelty score must be ≥ 7 for all hypotheses

Return ONLY the JSON object. No explanation. No markdown.`;
}

// ── Component ──────────────────────────────────────────────────────────────
export default function HypothesisGenerator() {
  const [archetype, setArchetype]       = useState("any");
  const [theme, setTheme]               = useState("");
  const [constraints, setConstraints]   = useState("");
  const [count, setCount]               = useState(3);
  const [loading, setLoading]           = useState(false);
  const [hypotheses, setHypotheses]     = useState([]);
  const [error, setError]               = useState("");
  const [expanded, setExpanded]         = useState(null);
  const [copied, setCopied]             = useState(null);
  const [statusMsg, setStatusMsg]       = useState("");
  const abortRef                        = useRef(null);

  const generate = useCallback(async () => {
    setLoading(true);
    setError("");
    setHypotheses([]);
    setExpanded(null);
    setStatusMsg("Consulting research corpus…");

    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 4000,
          system: SYSTEM_PROMPT,
          messages: [{ role: "user", content: buildUserPrompt(archetype, theme, constraints, count) }],
        }),
      });

      setStatusMsg("Synthesizing hypotheses…");
      const data = await res.json();

      if (data.error) throw new Error(data.error.message || "API error");

      const raw = data.content?.find(b => b.type === "text")?.text || "";
      // strip any markdown fences
      const clean = raw.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(clean);
      setHypotheses(parsed.hypotheses || []);
      setStatusMsg("");
    } catch (e) {
      setError(e.message);
      setStatusMsg("");
    } finally {
      setLoading(false);
    }
  }, [archetype, theme, constraints, count]);

  const copyExpr = async (id, expr) => {
    try {
      await navigator.clipboard.writeText(expr);
      setCopied(id);
      setTimeout(() => setCopied(null), 1500);
    } catch { }
  };

  const archetypeColor = (id) =>
    ARCHETYPES.find(a => a.id === id)?.color || "#888";

  const noveltyBar = (score) => {
    const pct = (score / 10) * 100;
    const col = score >= 8 ? "#00ff9d" : score >= 6 ? "#ffd700" : "#ff3d71";
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ flex: 1, height: 4, background: "#1a1a2e", borderRadius: 2 }}>
          <div style={{ width: `${pct}%`, height: "100%", background: col, borderRadius: 2, transition: "width 0.5s" }} />
        </div>
        <span style={{ fontSize: 11, color: col, fontFamily: "monospace", minWidth: 20 }}>{score}/10</span>
      </div>
    );
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#060612",
      color: "#e0e0ff",
      fontFamily: "'DM Mono', 'Fira Code', 'Courier New', monospace",
      padding: "32px 24px",
      maxWidth: 900,
      margin: "0 auto",
    }}>
      {/* Header */}
      <div style={{ marginBottom: 40, borderBottom: "1px solid #1a1a3e", paddingBottom: 24 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 4 }}>
          <span style={{ fontSize: 11, letterSpacing: 4, color: "#5555aa", textTransform: "uppercase" }}>AlphaQuant</span>
          <span style={{ fontSize: 11, color: "#333366" }}>v3</span>
        </div>
        <h1 style={{ margin: 0, fontSize: 28, fontWeight: 700, color: "#fff", letterSpacing: -1, fontFamily: "'DM Mono', monospace" }}>
          Hypothesis <span style={{ color: "#00d4ff" }}>Engine</span>
        </h1>
        <p style={{ margin: "8px 0 0", fontSize: 12, color: "#5555aa", letterSpacing: 1 }}>
          WORLDQUANT BRAIN · FASTEXPR · CROSS-SECTIONAL EQUITY
        </p>
      </div>

      {/* Controls */}
      <div style={{
        background: "#0a0a1e",
        border: "1px solid #1a1a3e",
        borderRadius: 8,
        padding: 24,
        marginBottom: 28,
      }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
          {/* Archetype */}
          <div>
            <label style={{ fontSize: 10, letterSpacing: 2, color: "#5555aa", display: "block", marginBottom: 8, textTransform: "uppercase" }}>
              Target Archetype
            </label>
            <select
              value={archetype}
              onChange={e => setArchetype(e.target.value)}
              style={{
                width: "100%", padding: "10px 12px", background: "#060612",
                border: "1px solid #2a2a4e", borderRadius: 4, color: archetypeColor(archetype),
                fontSize: 12, outline: "none", cursor: "pointer",
              }}
            >
              <option value="any" style={{ color: "#888" }}>— Any (auto-diversify) —</option>
              {ARCHETYPES.map(a => (
                <option key={a.id} value={a.id} style={{ color: a.color }}>
                  {a.icon} {a.label}
                </option>
              ))}
            </select>
          </div>

          {/* Count */}
          <div>
            <label style={{ fontSize: 10, letterSpacing: 2, color: "#5555aa", display: "block", marginBottom: 8, textTransform: "uppercase" }}>
              Hypotheses to Generate
            </label>
            <div style={{ display: "flex", gap: 8 }}>
              {[1, 2, 3, 4, 5].map(n => (
                <button
                  key={n}
                  onClick={() => setCount(n)}
                  style={{
                    flex: 1, padding: "10px 0", background: count === n ? "#00d4ff" : "#0a0a1e",
                    border: `1px solid ${count === n ? "#00d4ff" : "#2a2a4e"}`,
                    borderRadius: 4, color: count === n ? "#000" : "#5555aa",
                    fontSize: 13, cursor: "pointer", fontFamily: "monospace", fontWeight: count === n ? 700 : 400,
                  }}
                >{n}</button>
              ))}
            </div>
          </div>
        </div>

        {/* Theme */}
        <div style={{ marginBottom: 16 }}>
          <label style={{ fontSize: 10, letterSpacing: 2, color: "#5555aa", display: "block", marginBottom: 8, textTransform: "uppercase" }}>
            Theme / Research Direction <span style={{ color: "#333366" }}>(optional)</span>
          </label>
          <input
            value={theme}
            onChange={e => setTheme(e.target.value)}
            placeholder="e.g. options market disagreement, capital efficiency, earnings drift × quality"
            style={{
              width: "100%", padding: "10px 12px", background: "#060612",
              border: "1px solid #2a2a4e", borderRadius: 4, color: "#c0c0e0",
              fontSize: 12, outline: "none", boxSizing: "border-box",
            }}
          />
        </div>

        {/* Constraints */}
        <div style={{ marginBottom: 20 }}>
          <label style={{ fontSize: 10, letterSpacing: 2, color: "#5555aa", display: "block", marginBottom: 8, textTransform: "uppercase" }}>
            Additional Constraints <span style={{ color: "#333366" }}>(optional)</span>
          </label>
          <input
            value={constraints}
            onChange={e => setConstraints(e.target.value)}
            placeholder="e.g. delay=0 only, must use options fields, avoid high turnover"
            style={{
              width: "100%", padding: "10px 12px", background: "#060612",
              border: "1px solid #2a2a4e", borderRadius: 4, color: "#c0c0e0",
              fontSize: 12, outline: "none", boxSizing: "border-box",
            }}
          />
        </div>

        <button
          onClick={generate}
          disabled={loading}
          style={{
            width: "100%", padding: "14px 0",
            background: loading ? "#0a0a1e" : "linear-gradient(90deg, #00d4ff, #00ff9d)",
            border: loading ? "1px solid #2a2a4e" : "none",
            borderRadius: 4, color: loading ? "#5555aa" : "#000",
            fontSize: 13, fontWeight: 700, cursor: loading ? "not-allowed" : "pointer",
            letterSpacing: 2, textTransform: "uppercase", fontFamily: "monospace",
            transition: "all 0.2s",
          }}
        >
          {loading ? `⟳  ${statusMsg}` : "⚡  Generate Hypotheses"}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          padding: "12px 16px", background: "#200010", border: "1px solid #ff3d71",
          borderRadius: 4, color: "#ff3d71", fontSize: 12, marginBottom: 20,
        }}>
          ✗ {error}
        </div>
      )}

      {/* Results */}
      {hypotheses.length > 0 && (
        <div>
          <div style={{ fontSize: 10, letterSpacing: 3, color: "#5555aa", textTransform: "uppercase", marginBottom: 16 }}>
            {hypotheses.length} Hypothesis{hypotheses.length > 1 ? "es" : ""} Generated
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {hypotheses.map((h, i) => {
              const aColor = archetypeColor(h.archetype);
              const isOpen = expanded === h.id;
              return (
                <div
                  key={h.id}
                  style={{
                    background: "#0a0a1e",
                    border: `1px solid ${isOpen ? aColor : "#1a1a3e"}`,
                    borderLeft: `3px solid ${aColor}`,
                    borderRadius: 6,
                    overflow: "hidden",
                    transition: "border-color 0.2s",
                  }}
                >
                  {/* Header row */}
                  <div
                    onClick={() => setExpanded(isOpen ? null : h.id)}
                    style={{
                      padding: "16px 20px", cursor: "pointer", display: "flex",
                      alignItems: "flex-start", gap: 14,
                    }}
                  >
                    <span style={{ fontSize: 20, lineHeight: 1, marginTop: 2 }}>
                      {ARCHETYPES.find(a => a.id === h.archetype)?.icon || "◆"}
                    </span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", marginBottom: 6 }}>
                        <span style={{
                          fontSize: 9, letterSpacing: 2, padding: "2px 8px",
                          background: `${aColor}18`, color: aColor, borderRadius: 2,
                          textTransform: "uppercase", border: `1px solid ${aColor}44`,
                        }}>{h.archetype}</span>
                        <span style={{ fontSize: 11, color: "#888", fontFamily: "monospace" }}>
                          S: <span style={{ color: "#ffd700" }}>{h.expected_sharpe}</span>
                          &nbsp;·&nbsp;TO: <span style={{ color: "#00d4ff" }}>{h.expected_turnover_pct}%</span>
                        </span>
                      </div>
                      <div style={{ fontSize: 14, color: "#dde", fontWeight: 600, marginBottom: 6 }}>{h.title}</div>
                      <div style={{ width: "60%", minWidth: 140 }}>{noveltyBar(h.novelty_score)}</div>
                    </div>
                    <span style={{ color: "#3a3a6a", fontSize: 16, flexShrink: 0 }}>{isOpen ? "▲" : "▼"}</span>
                  </div>

                  {/* Expanded detail */}
                  {isOpen && (
                    <div style={{ padding: "0 20px 20px", borderTop: "1px solid #1a1a3e" }}>
                      {/* Claim */}
                      <div style={{ marginTop: 16, marginBottom: 14 }}>
                        <div style={{ fontSize: 9, letterSpacing: 2, color: "#5555aa", textTransform: "uppercase", marginBottom: 6 }}>Hypothesis</div>
                        <div style={{ fontSize: 13, color: "#b0b0d0", lineHeight: 1.6 }}>{h.claim}</div>
                      </div>

                      {/* Mechanism + Regime Guard */}
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
                        <div style={{ background: "#060612", padding: 12, borderRadius: 4, border: "1px solid #1a1a3e" }}>
                          <div style={{ fontSize: 9, letterSpacing: 2, color: "#00d4ff", textTransform: "uppercase", marginBottom: 6 }}>Mechanism</div>
                          <div style={{ fontSize: 11, color: "#8888cc", lineHeight: 1.5 }}>{h.mechanism}</div>
                        </div>
                        <div style={{ background: "#060612", padding: 12, borderRadius: 4, border: "1px solid #2a1a1a" }}>
                          <div style={{ fontSize: 9, letterSpacing: 2, color: "#ff3d71", textTransform: "uppercase", marginBottom: 6 }}>Regime Guard</div>
                          <div style={{ fontSize: 11, color: "#8888cc", lineHeight: 1.5 }}>{h.regime_guard}</div>
                        </div>
                      </div>

                      {/* Expression */}
                      <div style={{ marginBottom: 14 }}>
                        <div style={{ fontSize: 9, letterSpacing: 2, color: "#5555aa", textTransform: "uppercase", marginBottom: 6 }}>FastExpr</div>
                        <div style={{
                          background: "#030310", padding: "12px 14px", borderRadius: 4,
                          border: "1px solid #1a1a3e", display: "flex", alignItems: "flex-start", gap: 10,
                        }}>
                          <code style={{
                            flex: 1, fontSize: 12, color: "#00ff9d", lineHeight: 1.6,
                            wordBreak: "break-all", whiteSpace: "pre-wrap",
                          }}>{h.expression}</code>
                          <button
                            onClick={() => copyExpr(h.id, h.expression)}
                            style={{
                              flexShrink: 0, padding: "4px 10px", background: copied === h.id ? "#00ff9d22" : "#0a0a1e",
                              border: `1px solid ${copied === h.id ? "#00ff9d" : "#2a2a4e"}`,
                              borderRadius: 3, color: copied === h.id ? "#00ff9d" : "#5555aa",
                              fontSize: 10, cursor: "pointer", fontFamily: "monospace",
                            }}
                          >{copied === h.id ? "✓" : "copy"}</button>
                        </div>
                      </div>

                      {/* Settings + Fields */}
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
                        <div style={{ background: "#060612", padding: 12, borderRadius: 4, border: "1px solid #1a1a3e" }}>
                          <div style={{ fontSize: 9, letterSpacing: 2, color: "#5555aa", textTransform: "uppercase", marginBottom: 8 }}>Settings</div>
                          {Object.entries(h.settings || {}).map(([k, v]) => (
                            <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                              <span style={{ fontSize: 10, color: "#555588" }}>{k}</span>
                              <span style={{ fontSize: 10, color: "#aaaadd", fontFamily: "monospace" }}>{String(v)}</span>
                            </div>
                          ))}
                        </div>
                        <div style={{ background: "#060612", padding: 12, borderRadius: 4, border: "1px solid #1a1a3e" }}>
                          <div style={{ fontSize: 9, letterSpacing: 2, color: "#5555aa", textTransform: "uppercase", marginBottom: 8 }}>Fields Used</div>
                          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                            {(h.fields_used || []).map(f => (
                              <span key={f} style={{
                                fontSize: 10, padding: "2px 6px", background: "#0a0a2e",
                                border: "1px solid #2a2a4e", borderRadius: 2, color: "#8888cc",
                              }}>{f}</span>
                            ))}
                          </div>
                          {h.risk_flags?.length > 0 && (
                            <div style={{ marginTop: 8 }}>
                              {h.risk_flags.map((r, ri) => (
                                <div key={ri} style={{ fontSize: 10, color: "#ff6b35", marginBottom: 2 }}>⚠ {r}</div>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Novelty reason */}
                      <div style={{ background: "#060612", padding: 12, borderRadius: 4, border: `1px solid ${aColor}22` }}>
                        <div style={{ fontSize: 9, letterSpacing: 2, color: aColor, textTransform: "uppercase", marginBottom: 6 }}>Novelty Rationale</div>
                        <div style={{ fontSize: 11, color: "#8888cc", lineHeight: 1.5 }}>{h.novelty_reason}</div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Empty state */}
      {!loading && hypotheses.length === 0 && !error && (
        <div style={{
          textAlign: "center", padding: "60px 0", color: "#2a2a5e",
          border: "1px dashed #1a1a3e", borderRadius: 6,
        }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>⚡</div>
          <div style={{ fontSize: 13, letterSpacing: 2, textTransform: "uppercase" }}>Ready to generate</div>
          <div style={{ fontSize: 11, marginTop: 6, color: "#1a1a4e" }}>Configure above and hit Generate</div>
        </div>
      )}

      <div style={{ marginTop: 32, fontSize: 10, color: "#2a2a4e", letterSpacing: 1, textAlign: "center" }}>
        BRAIN FASTEXPR · TOP3000 · USA · FITNESS ≥ 1.0 · SHARPE ≥ 1.25
      </div>
    </div>
  );
}
