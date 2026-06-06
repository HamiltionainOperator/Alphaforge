import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BadgeCheck,
  Brain,
  Check,
  Clock,
  Copy,
  Cpu,
  ExternalLink,
  FlaskConical,
  Gauge,
  GitCompare,
  Globe,
  Hash,
  LineChart,
  Loader2,
  Minus,
  Plus,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Timer,
  Trash2,
  Wifi,
  WifiOff,
  Lightbulb,
  Wrench,
  X,
  Zap,
} from "lucide-react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const WS_BASE = (import.meta.env.VITE_WS_BASE_URL || "").replace(/\/$/, "");

const ARCHETYPES = [
  ["any", "any"],
  ["reversal", "reversal"],
  ["microstructure", "microstructure"],
  ["volatility", "volatility"],
  ["fundamental", "fundamental"],
  ["analyst_revision", "analyst rev."],
  ["earnings_event", "earnings event"],
  ["options_implied", "options/IV"],
  ["factor_residual", "factor residual"],
  ["dispersion", "dispersion"],
];

const VERDICT_STYLE = {
  PASS: { color: "var(--lime)", bg: "var(--lime-dim)", Icon: ShieldCheck, label: "PASS" },
  WARN: { color: "var(--amber)", bg: "var(--amber-dim)", Icon: AlertTriangle, label: "WARN" },
  FAIL: { color: "var(--red)", bg: "var(--red-dim)", Icon: X, label: "FAIL" },
};

const SIM_STYLE = {
  queued: { color: "var(--ink-dim)", bg: "rgba(218,224,214,0.08)", label: "queued" },
  running: { color: "var(--steel)", bg: "var(--steel-dim)", label: "running" },
  completed: { color: "var(--lime)", bg: "var(--lime-dim)", label: "completed" },
  failed: { color: "var(--red)", bg: "var(--red-dim)", label: "failed" },
};

export default function AlphaForge() {
  const [intent, setIntent] = useState("");
  const [arch, setArch] = useState("any");
  const [count, setCount] = useState(3);
  const [webResearch, setWebResearch] = useState(true);
  const [providers, setProviders] = useState([]);
  const [genProvider, setGenProvider] = useState("openrouter");
  const [researchProvider, setResearchProvider] = useState("openrouter");
  const [repairProvider, setRepairProvider] = useState("openrouter");
  const [thinkProvider, setThinkProvider] = useState("claude_code");
  const [connection, setConnection] = useState({
    state: "disconnected",
    message: "Brain session is disconnected.",
  });
  const [phase, setPhase] = useState("idle");
  const [alphas, setAlphas] = useState([]);
  const [brief, setBrief] = useState("");
  const [sources, setSources] = useState([]);
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState("");
  const [copiedId, setCopiedId] = useState(null);
  const [repairingId, setRepairingId] = useState(null);
  const [open, setOpen] = useState({});
  const [editOpen, setEditOpen] = useState({});
  const [settingsDraft, setSettingsDraft] = useState({});
  const [startedAt, setStartedAt] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const [hypotheses, setHypotheses] = useState([]);
  const [hypothesizing, setHypothesizing] = useState(false);
  const [hypTheme, setHypTheme] = useState("");
  const [selectedHypothesis, setSelectedHypothesis] = useState(null);
  const [tokenUsage, setTokenUsage] = useState({ input_tokens: 0, output_tokens: 0, total_tokens: 0, calls: 0 });
  const [thinkMode, setThinkMode] = useState("adaptive");
  const [jobs, setJobs] = useState([]);
  const [jobsTab, setJobsTab] = useState(false);
  const [schedulingJob, setSchedulingJob] = useState(false);
  const socketRef = useRef(null);

  const running = phase === "researching" || phase === "generating" || phase === "simulating";
  const canForge = !running && connection.state === "connected";

  const summary = useMemo(() => {
    return alphas.reduce(
      (acc, alpha) => {
        const verdict = alpha.validation?.verdict || "FAIL";
        acc[verdict] = (acc[verdict] || 0) + 1;
        const simState = alpha.simulation_state || simulationState(alpha.simulation);
        acc[simState] = (acc[simState] || 0) + 1;
        return acc;
      },
      { PASS: 0, WARN: 0, FAIL: 0, queued: 0, running: 0, completed: 0, failed: 0 }
    );
  }, [alphas]);

  useEffect(() => {
    let active = true;
    fetch(apiUrl("/api/connect"))
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (active && data?.state) {
          setConnection(data);
        }
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    fetch(apiUrl("/api/providers"))
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!active || !data?.providers) return;
        setProviders(data.providers);
        const priority = ["openrouter", "claude_code", "anthropic"];
        const pick = priority.find((id) => data.providers.some((p) => p.id === id && p.available))
          || data.providers.find((p) => p.available)?.id;
        if (pick) {
          setGenProvider(pick);
          setResearchProvider(pick);
          setRepairProvider(pick);
        }
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!running || !startedAt) return undefined;
    const timer = window.setInterval(() => {
      setElapsed(Math.round((Date.now() - startedAt) / 100) / 10);
    }, 400);
    return () => window.clearInterval(timer);
  }, [running, startedAt]);

  useEffect(() => {
    return () => {
      if (socketRef.current) socketRef.current.close();
    };
  }, []);

  useEffect(() => {
    async function fetchUsage() {
      try {
        const data = await fetch(apiUrl("/api/usage")).then((r) => r.json());
        setTokenUsage(data);
      } catch {}
    }
    fetchUsage();
    const id = setInterval(fetchUsage, 4000);
    return () => clearInterval(id);
  }, []);

  const fetchJobs = useCallback(async () => {
    try {
      const data = await fetch(apiUrl("/api/jobs")).then((r) => r.json());
      setJobs(data.jobs || []);
    } catch {}
  }, []);

  useEffect(() => {
    fetchJobs();
    const id = setInterval(fetchJobs, 5000);
    return () => clearInterval(id);
  }, [fetchJobs]);

  async function connectBrain() {
    setError("");
    setConnection({ state: "authenticating", message: "Authenticating with WorldQuant Brain." });
    addLog("Brain authentication started.");
    try {
      const data = await apiPost("/api/connect", {});
      setConnection(data);
      addLog("Brain session connected.");
    } catch (err) {
      setConnection({ state: "disconnected", message: err.message });
      setError(err.message);
      addLog(`Brain connection failed: ${err.message}`);
    }
  }

  async function scheduleJob() {
    if (schedulingJob) return;
    setSchedulingJob(true);
    addLog("Scheduling background forge job…");
    try {
      const data = await apiPost("/api/jobs", {
        intent: intent.trim(),
        archetype: arch,
        count,
        web_research: webResearch,
        gen_provider: genProvider,
        research_provider: researchProvider,
        hypothesis_data: selectedHypothesis || null,
        think_mode: thinkMode,
        label: intent.trim() || `${arch} × ${count}`,
      });
      addLog(`Job ${data.job_id} queued — runs in the background even if you close this tab.`);
      await fetchJobs();
      setJobsTab(true);
    } catch (err) {
      addLog(`Schedule failed: ${err.message}`);
    } finally {
      setSchedulingJob(false);
    }
  }

  async function deleteJob(jobId) {
    try {
      await fetch(apiUrl(`/api/jobs/${jobId}`), { method: "DELETE" });
      await fetchJobs();
    } catch {}
  }

  function forge() {
    if (running) return;
    if (socketRef.current) socketRef.current.close();

    setError("");
    setAlphas([]);
    setBrief("");
    setSources([]);
    setLogs([]);
    setOpen({});
    setPhase(webResearch ? "researching" : "generating");
    setStartedAt(Date.now());
    setElapsed(0);
    addLog("Forge request queued.");

    const payload = {
      intent: intent.trim(),
      archetype: arch,
      count,
      web_research: webResearch,
      gen_provider: genProvider,
      research_provider: researchProvider,
      hypothesis_data: selectedHypothesis || null,
      think_mode: thinkMode,
      think_provider: thinkMode === "deep" ? thinkProvider : genProvider,
    };

    let completed = false;
    let fallbackStarted = false;
    const ws = new WebSocket(wsUrl("/ws/forge"));
    socketRef.current = ws;

    ws.onopen = () => {
      addLog("Forge stream connected.");
      ws.send(JSON.stringify(payload));
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.event === "forge_completed" || message.event === "forge_failed") {
        completed = true;
      }
      handleForgeEvent(message);
    };

    ws.onerror = () => {
      if (!fallbackStarted) {
        fallbackStarted = true;
        addLog("Forge stream unavailable. Falling back to POST /api/forge.");
        ws.close();
        forgeViaHttp(payload);
      }
    };

    ws.onclose = () => {
      socketRef.current = null;
      if (!completed && !fallbackStarted) {
        setPhase("error");
        setError("Forge stream closed before completion.");
        addLog("Forge stream closed before completion.");
      }
    };
  }

  async function forgeViaHttp(payload) {
    try {
      setPhase(payload.web_research ? "researching" : "generating");
      const data = await apiPost("/api/forge", payload);
      if (data.research) {
        setBrief(data.research.brief || "");
        setSources(data.research.sources || []);
      }
      const finished = (data.alphas || []).map((alpha, index) => ({
        ...alpha,
        id: alpha.id || `a${index}`,
        simulation_state: alpha.simulation_state || simulationState(alpha.simulation),
      }));
      setAlphas(finished);
      setPhase("completed");
      finished.forEach((alpha) => {
        if (alpha.simulation?.status === "OK" && alpha.simulation?.alpha_id) {
          fetchDetails(alpha.id, alpha.simulation.alpha_id);
        }
      });
      const finalElapsed = startedAt ? Math.round((Date.now() - startedAt) / 100) / 10 : elapsed;
      setElapsed(finalElapsed);
      addLog(`Forge completed with ${finished.length} alpha${finished.length === 1 ? "" : "s"}.`);
    } catch (err) {
      setPhase("error");
      setError(err.message);
      addLog(`Forge failed: ${err.message}`);
    }
  }

  async function runHypothesize() {
    if (hypothesizing) return;
    setHypothesizing(true);
    setHypotheses([]);
    addLog("Hypothesis generation started…");
    try {
      const data = await apiPost("/api/hypothesize", {
        archetype: arch,
        theme: hypTheme.trim(),
        count: 10,
        provider: genProvider,
      });
      setHypotheses(data.hypotheses || []);
      addLog(`Hypothesis generation complete: ${(data.hypotheses || []).length} novel hypotheses.`);
    } catch (err) {
      addLog(`Hypothesis generation failed: ${err.message}`);
    } finally {
      setHypothesizing(false);
    }
  }

  function handleForgeEvent(message) {
    switch (message.event) {
      case "research_started":
        setPhase("researching");
        addLog(`Web research started${message.provider ? ` (${providerLabel(message.provider)})` : ""}.`);
        break;
      case "research_completed":
        setBrief(message.brief || "");
        setSources(message.sources || []);
        addLog(`Research brief ready (${(message.sources || []).length} source${(message.sources || []).length === 1 ? "" : "s"}).`);
        break;
      case "research_skipped":
        addLog(`Web research skipped: ${message.reason || "unavailable"}. Using model knowledge.`);
        break;
      case "generation_started":
        setPhase("generating");
        addLog(`Generation started for ${message.count} alpha${message.count === 1 ? "" : "s"}${message.provider ? ` (${providerLabel(message.provider)})` : ""}.`);
        break;
      case "alpha_generated":
        setAlphas((prev) => upsertAlpha(prev, message.index, {
          ...message.alpha,
          simulation_state: "queued",
        }));
        addLog(`Alpha ${message.index + 1} generated and queued.`);
        break;
      case "simulation_started":
        setPhase("simulating");
        setAlphas((prev) => prev.map((alpha, index) => (
          index === message.index ? { ...alpha, simulation_state: "running" } : alpha
        )));
        addLog(`Simulation ${message.index + 1} started.`);
        break;
      case "simulation_completed":
        setAlphas((prev) => prev.map((alpha, index) => (
          index === message.index
            ? {
                ...alpha,
                ...(message.alpha || {}),
                simulation: message.simulation,
                simulation_state: simulationState(message.simulation),
              }
            : alpha
        )));
        addLog(`Simulation ${message.index + 1} ${message.simulation?.status === "OK" ? "completed" : "failed"}.`);
        if (message.simulation?.status === "OK" && message.simulation?.alpha_id && message.alpha_id) {
          fetchDetails(message.alpha_id, message.simulation.alpha_id);
        }
        break;
      case "forge_completed":
        setPhase("completed");
        setElapsed(message.elapsed_s || elapsed);
        addLog(`Forge completed in ${formatSeconds(message.elapsed_s || elapsed)}.`);
        break;
      case "forge_failed":
        setPhase("error");
        setError(message.error || "Forge failed.");
        addLog(`Forge failed: ${message.error || "unknown error"}`);
        break;
      default:
        addLog(message.event || "Unknown backend event.");
        break;
    }
  }

  async function copyExpr(id, expression) {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(expression);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = expression;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      setCopiedId(id);
      window.setTimeout(() => setCopiedId(null), 1200);
    } catch {
      setCopiedId(null);
    }
  }

  function patchAlpha(id, patch) {
    setAlphas((prev) => prev.map((item) => (
      item.id === id ? { ...item, ...(typeof patch === "function" ? patch(item) : patch) } : item
    )));
  }

  async function repairAlpha(alpha) {
    if (!alpha?.expression || repairingId) return;
    const id = alpha.id;
    const failures = alpha.simulation?.failures || alpha.details?.failures || [];
    const sim = alpha.simulation || {};
    setRepairingId(id);
    addLog(`Repairing ${alpha.name || "alpha"} with OpenRouter${failures.length ? " from BRAIN failures" : ""}.`);
    try {
      const data = await apiPost("/api/repair", {
        expression: alpha.expression,
        issues: alpha.validation?.issues || [],
        failures,
        simulation: { sharpe: sim.sharpe, fitness: sim.fitness, turnover_pct: sim.turnover_pct, status: sim.status, error: sim.error },
        name: alpha.name || "",
        archetype: alpha.archetype || "any",
        hypothesis: alpha.hypothesis || "",
        intent: intent.trim(),
        settings: alpha.settings || {},
        expected_sharpe: alpha.expected_sharpe || "",
        regime_note: alpha.regime_note || "",
        turnover_note: alpha.turnover_note || "",
        simulate: connection.state === "connected",
        provider: repairProvider,
      });
      if (data.alpha) {
        setAlphas((prev) => prev.map((item) => (item.id === id ? { ...data.alpha, id } : item)));
        addLog(`Repaired: ${data.alpha.change_note || "expression updated"}.`);
        const newSimId = data.alpha.simulation?.alpha_id;
        if (newSimId && data.alpha.simulation?.status === "OK") fetchDetails(id, newSimId);
      }
    } catch (err) {
      addLog(`Repair failed: ${err.message}`);
    } finally {
      setRepairingId(null);
    }
  }

  async function fetchDetails(id, alphaId) {
    if (!alphaId) return;
    patchAlpha(id, { detailsLoading: true, detailsError: "" });
    try {
      const data = await apiPost("/api/alpha/details", { alpha_id: alphaId });
      if (data.status === "OK") {
        patchAlpha(id, { details: data, detailsLoading: false });
      } else {
        patchAlpha(id, { detailsLoading: false, detailsError: data.error || data.status || "details unavailable" });
      }
    } catch (err) {
      patchAlpha(id, { detailsLoading: false, detailsError: err.message });
    }
  }

  async function submitAlpha(alpha) {
    const alphaId = alpha.simulation?.alpha_id;
    if (!alphaId) { addLog("No simulated alpha_id to submit."); return; }
    if (alpha.submitting) return;
    const ok = window.confirm(
      `Submit "${alpha.name || alpha.id}" to the WorldQuant BRAIN alpha pool?\n\n` +
      `Expression:\n${alpha.expression}\n\nThis is a real, account-level action.`
    );
    if (!ok) return;
    patchAlpha(alpha.id, { submitting: true, submission: null });
    addLog(`Submitting ${alpha.name || alpha.id} (${alphaId}) to BRAIN.`);
    try {
      const data = await apiPost("/api/alpha/submit", { alpha_id: alphaId });
      patchAlpha(alpha.id, { submitting: false, submission: data });
      addLog(`Submit ${data.status}: ${data.message || data.error || ""}`);
    } catch (err) {
      patchAlpha(alpha.id, { submitting: false, submission: { status: "SUBMIT_ERROR", error: err.message } });
      addLog(`Submit failed: ${err.message}`);
    }
  }

  async function resimulate(alpha) {
    if (alpha.resimulating) return;
    if (connection.state !== "connected") { addLog("Connect Brain before simulating."); return; }
    const draft = settingsDraft[alpha.id] || alpha.settings || {};
    patchAlpha(alpha.id, { resimulating: true });
    addLog(`Re-simulating ${alpha.name || alpha.id} with edited settings.`);
    try {
      const sim = await apiPost("/api/simulate", { expression: alpha.expression, settings: draft });
      const state = sim.status === "OK" ? "completed" : "failed";
      patchAlpha(alpha.id, {
        simulation: sim,
        simulation_state: state,
        settings: { ...alpha.settings, ...draft, universe: "TOP3000" },
        resimulating: false,
        details: null,
        detailsError: "",
        submission: null,
      });
      addLog(`Re-simulation ${sim.status}${sim.status === "OK" ? ` — Sharpe ${sim.sharpe ?? "?"}` : ""}.`);
      if (sim.alpha_id && sim.status === "OK") fetchDetails(alpha.id, sim.alpha_id);
    } catch (err) {
      patchAlpha(alpha.id, { resimulating: false });
      addLog(`Re-simulation failed: ${err.message}`);
    }
  }

  function toggleSettingsEditor(alpha) {
    setSettingsDraft((prev) => (prev[alpha.id] ? prev : { ...prev, [alpha.id]: { ...(alpha.settings || {}) } }));
    setEditOpen((prev) => ({ ...prev, [alpha.id]: !prev[alpha.id] }));
  }

  function updateDraft(id, key, value) {
    setSettingsDraft((prev) => ({ ...prev, [id]: { ...prev[id], [key]: value } }));
  }

  function addLog(text) {
    const stamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setLogs((prev) => [...prev.slice(-99), { id: `${Date.now()}-${Math.random()}`, time: stamp, text }]);
  }

  const connectIcon = connection.state === "connected" ? Wifi : WifiOff;
  const forgeLabel = phase === "researching"
    ? "Researching"
    : phase === "generating"
      ? "Generating"
      : phase === "simulating"
        ? "Simulating"
        : "Forge Alphas";

  return (
    <div className="af-shell">
      <style>{CSS}</style>
      <div className="af-grid-bg" />
      <ServerStatusBox />

      <main className="af-main">
        <header className="af-header">
          <div>
            <div className="af-kicker">
              <Cpu size={14} />
              autonomous fastexpr synthesis - worldquant brain
            </div>
            <h1>AlphaForge</h1>
          </div>
          <div className="af-status-wrap">
            <StatusPill state={connection.state} icon={connectIcon} label={connection.state} />
            <button className="af-secondary" disabled={connection.state === "authenticating" || running} onClick={connectBrain}>
              {connection.state === "authenticating" ? <Loader2 size={15} className="spin" /> : <Wifi size={15} />}
              Connect Brain
            </button>
          </div>
        </header>

        <section className="af-panel">
          <div className="af-connect-note">
            <span>{connection.message || "Brain session is disconnected."}</span>
            {connection.email && <span>{connection.email}</span>}
          </div>

          <div className="af-intent-header">
            <Label icon={FlaskConical}>research intent</Label>
            <div className="af-hyp-controls">
              <input
                className="af-hyp-theme"
                value={hypTheme}
                onChange={(e) => setHypTheme(e.target.value)}
                placeholder="theme (optional)"
                disabled={hypothesizing || running}
              />
              <button
                className="af-secondary af-hyp-btn"
                disabled={hypothesizing || running}
                onClick={runHypothesize}
                title="Generate novel signal hypotheses and click one to load it as intent"
              >
                {hypothesizing ? <Loader2 size={14} className="spin" /> : <Lightbulb size={14} />}
                {hypothesizing ? "thinking…" : "Hypothesize"}
              </button>
            </div>
          </div>
          <textarea
            className="af-input"
            value={intent}
            onChange={(event) => { setIntent(event.target.value); setSelectedHypothesis(null); }}
            placeholder="e.g. low-turnover post-earnings drift, idiosyncratic volatility residual reversal, option skew dispersion versus realized vol"
            rows={3}
            disabled={running}
          />
          {selectedHypothesis && (
            <div className="af-hyp-active">
              <Lightbulb size={12} />
              <span className="af-hyp-active-label">hypothesis guide active:</span>
              <strong className="af-hyp-active-title">{selectedHypothesis.title}</strong>
              <span className="af-hyp-active-fields">
                fields: {(selectedHypothesis.fields_suggested || []).join(", ")}
              </span>
              <button
                className="af-hyp-active-clear"
                onClick={() => setSelectedHypothesis(null)}
                title="Clear hypothesis guide — switch to free-form generation"
              >
                <X size={11} />
              </button>
            </div>
          )}
          {hypotheses.length > 0 && (
            <div className="af-hyp-panel">
              <div className="af-hyp-panel-head">
                <Lightbulb size={13} />
                {hypotheses.length} novel hypotheses — click one to load as intent
              </div>
              {hypotheses.map((h, i) => (
                <button
                  key={i}
                  className={`af-hyp-card${selectedHypothesis === h ? " af-hyp-card-selected" : ""}`}
                  onClick={() => {
                    setSelectedHypothesis(h);
                    setIntent(h.claim);
                    if (h.archetype && h.archetype !== "novel" && h.archetype !== "any") {
                      setArch(h.archetype);
                    }
                  }}
                  title="Click to load this hypothesis — fields, mechanism, and archetype will guide generation"
                >
                  <div className="af-hyp-card-top">
                    <span className="af-tag">{h.archetype}</span>
                    <span className="af-hyp-title">{h.title}</span>
                    <span className="af-hyp-novelty" title="Novelty score">★ {h.novelty_score}/10</span>
                  </div>
                  <div className="af-hyp-claim">{h.claim}</div>
                  <div className="af-hyp-meta">
                    <span>fields: {(h.fields_suggested || []).join(", ")}</span>
                    <span>turnover: {h.expected_turnover_range}</span>
                    <span>sharpe: {h.expected_sharpe_range}</span>
                  </div>
                </button>
              ))}
            </div>
          )}

          <div className="af-controls">
            <div className="af-control-block">
              <Label icon={Sparkles}>archetype</Label>
              <div className="af-chip-row">
                {ARCHETYPES.map(([value, label]) => (
                  <button
                    key={value}
                    className={`af-chip${arch === value ? " active" : ""}`}
                    disabled={running}
                    onClick={() => setArch(value)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div className="af-count-block">
              <Label icon={Hash}>count</Label>
              <div className="af-stepper">
                <button disabled={running || count <= 1} onClick={() => setCount((value) => Math.max(1, value - (value > 50 ? 10 : value > 10 ? 5 : 1)))}>
                  <Minus size={14} />
                </button>
                <input
                  className="af-count-input"
                  type="number"
                  min="1"
                  max="100"
                  value={count}
                  disabled={running}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (!Number.isNaN(v)) setCount(Math.max(1, Math.min(100, v)));
                  }}
                />
                <button disabled={running || count >= 100} onClick={() => setCount((value) => Math.min(100, value + (value >= 50 ? 10 : value >= 10 ? 5 : 1)))}>
                  <Plus size={14} />
                </button>
              </div>
              {count > 20 && (
                <div className="af-count-note">~{Math.round(count * 0.5)}–{Math.round(count * 1.2)} min</div>
              )}
            </div>

            <button
              type="button"
              className={`af-toggle${webResearch ? " on" : ""}`}
              disabled={running}
              onClick={() => setWebResearch((value) => !value)}
              title="Ground generation in a live web-research brief"
            >
              <Globe size={14} />
              {webResearch ? "web research on" : "web research off"}
              <span className="af-toggle-dot" />
            </button>

            <button className="af-forge" disabled={!canForge} onClick={forge}>
              {running ? <Loader2 size={17} className="spin" /> : <Zap size={17} />}
              {forgeLabel}
            </button>
          </div>

          <div className="af-think-row">
            <Label icon={Brain}>thinking mode</Label>
            <div className="af-think-chips">
              {[["standard","standard","single-shot, fastest"],["adaptive","adaptive","plan → build (default)"],["deep","deep think","4 specialists critique → refine → build → verify"]].map(([val, label, tip]) => (
                <button
                  key={val}
                  className={`af-chip${thinkMode === val ? " active" : ""}${val === "deep" ? " af-chip-deep" : ""}`}
                  disabled={running}
                  onClick={() => setThinkMode(val)}
                  title={tip}
                >
                  {val === "deep" && <Brain size={11} />}
                  {label}
                </button>
              ))}
            </div>
            {thinkMode === "deep" && (
              <div className="af-think-engine-row">
                <span className="af-think-engine-label">
                  <Brain size={11} />
                  think engine
                </span>
                <select
                  className="af-think-engine-select"
                  value={thinkProvider}
                  disabled={running}
                  onChange={(e) => setThinkProvider(e.target.value)}
                  title="LLM used for critics, synthesis, refine, and expression verify — can differ from the generate engine"
                >
                  {(providers.length ? providers : [{id:"claude_code",label:"Claude Code",available:true},{id:"openrouter",label:"OpenRouter (owl-alpha)",available:true},{id:"anthropic",label:"Anthropic API",available:true}]).map((p) => (
                    <option key={p.id} value={p.id} disabled={!p.available}>
                      {p.label}{p.available ? "" : " (unavailable)"}
                    </option>
                  ))}
                </select>
                <span className="af-think-note">
                  4 critics · synthesis · refine · verify — set to Claude Code for best reasoning
                </span>
              </div>
            )}
          </div>

          <div className="af-schedule-row">
            <button
              className="af-secondary af-schedule-btn"
              disabled={running || schedulingJob}
              onClick={scheduleJob}
              title="Queue a background forge job — runs even if you close this tab"
            >
              {schedulingJob ? <Loader2 size={14} className="spin" /> : <Clock size={14} />}
              {schedulingJob ? "scheduling…" : "schedule (background)"}
            </button>
            {jobs.some((j) => j.status === "running" || j.status === "queued") && (
              <span className="af-job-live">
                <Loader2 size={12} className="spin" />
                {jobs.filter((j) => j.status === "running").length} job(s) running in background
              </span>
            )}
            <button
              className={`af-jobs-tab-btn${jobsTab ? " active" : ""}`}
              onClick={() => setJobsTab((v) => !v)}
            >
              Jobs {jobs.length > 0 && <span className="af-job-badge">{jobs.length}</span>}
            </button>
          </div>

          <div className="af-engines">
            <EngineSelect label="generate" value={genProvider} onChange={setGenProvider} providers={providers} disabled={running} />
            <EngineSelect label="research" value={researchProvider} onChange={setResearchProvider} providers={providers} disabled={running} />
            <EngineSelect label="repair" value={repairProvider} onChange={setRepairProvider} providers={providers} disabled={false} />
          </div>
        </section>

        {jobsTab && (
          <JobsPanel jobs={jobs} onDelete={deleteJob} onRefresh={fetchJobs} />
        )}

        {brief && (
          <section className="af-research">
            <Label icon={Globe}>research brief</Label>
            <p>{brief}</p>
            {sources.length > 0 && (
              <>
                <div className="af-sources-head">sources</div>
                <div className="af-sources">
                  {sources.slice(0, 10).map((source, sourceIndex) => (
                    <a
                      key={`${source.url}-${sourceIndex}`}
                      href={source.url}
                      target="_blank"
                      rel="noreferrer"
                      className="af-src"
                    >
                      <ExternalLink size={12} />
                      {(source.title || source.url).slice(0, 60)}
                    </a>
                  ))}
                </div>
              </>
            )}
          </section>
        )}

        {error && (
          <section className="af-error">
            <AlertTriangle size={15} />
            <span>{error}</span>
          </section>
        )}

        {(alphas.length > 0 || running || phase === "completed") && (
          <section className="af-summary">
            <Stat n={alphas.length} label="forged" color="var(--ink)" />
            <Stat n={summary.PASS} label="pass" color="var(--lime)" />
            <Stat n={summary.WARN} label="warn" color="var(--amber)" />
            <Stat n={summary.FAIL} label="fail" color="var(--red)" />
            <Stat n={summary.completed} label="completed" color="var(--steel)" />
            <span className="af-elapsed"><Timer size={13} />{formatSeconds(elapsed)}</span>
            {tokenUsage.total_tokens > 0 && (
              <span className="af-token-usage" title={`${tokenUsage.input_tokens.toLocaleString()} in · ${tokenUsage.output_tokens.toLocaleString()} out · ${tokenUsage.calls} calls`}>
                <Cpu size={13} />
                {tokenUsage.total_tokens.toLocaleString()} tokens
                <button className="af-token-reset" onClick={async () => { await fetch(apiUrl("/api/usage/reset"), { method: "POST" }); setTokenUsage({ input_tokens: 0, output_tokens: 0, total_tokens: 0, calls: 0 }); }} title="Reset token counter">×</button>
              </span>
            )}
          </section>
        )}

        <section className="af-results">
          {alphas.map((alpha, index) => (
            <AlphaCard
              key={alpha.id || index}
              alpha={alpha}
              index={index}
              isOpen={Boolean(open[alpha.id || index])}
              onToggle={() => setOpen((prev) => ({ ...prev, [alpha.id || index]: !prev[alpha.id || index] }))}
              onCopy={() => copyExpr(alpha.id || index, alpha.expression)}
              copied={copiedId === (alpha.id || index)}
              onRepair={() => repairAlpha(alpha)}
              repairing={repairingId === (alpha.id || index)}
              busy={Boolean(repairingId)}
              connected={connection.state === "connected"}
              onSubmit={() => submitAlpha(alpha)}
              onResimulate={() => resimulate(alpha)}
              onFetchDetails={() => fetchDetails(alpha.id, alpha.simulation?.alpha_id)}
              draft={settingsDraft[alpha.id]}
              onDraftChange={(key, value) => updateDraft(alpha.id, key, value)}
              editOpen={Boolean(editOpen[alpha.id])}
              onToggleEdit={() => toggleSettingsEditor(alpha)}
            />
          ))}
        </section>

        {phase === "idle" && (
          <section className="af-empty">
            <Activity size={22} />
            <span>Connect Brain, choose a direction, then forge alphas.</span>
          </section>
        )}

        <section className="af-log-panel">
          <div className="af-log-head">
            <Label icon={Activity}>logs</Label>
            {running && <span><Loader2 size={13} className="spin" /> live</span>}
          </div>
          <div className="af-log-list">
            {logs.length === 0 ? (
              <div className="af-log-empty">No backend events yet.</div>
            ) : logs.map((item) => (
              <div key={item.id} className="af-log-line">
                <time>{item.time}</time>
                <span>{item.text}</span>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

function EngineSelect({ label, value, onChange, providers, disabled }) {
  const options = providers.length ? providers : [{ id: "openrouter", label: "OpenRouter (owl-alpha)", available: true }];
  return (
    <label className="af-engine">
      <span><Cpu size={11} />{label} engine</span>
      <select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
        {options.map((p) => (
          <option key={p.id} value={p.id} disabled={!p.available} title={p.available ? "" : p.reason || ""}>
            {p.label}{p.available ? "" : " (unavailable)"}
          </option>
        ))}
      </select>
    </label>
  );
}

function AlphaCard({
  alpha, index, isOpen, onToggle, onCopy, copied, onRepair, repairing, busy,
  connected, onSubmit, onResimulate, onFetchDetails, draft, onDraftChange, editOpen, onToggleEdit,
}) {
  const validation = alpha.validation || { verdict: "FAIL", issues: [], turnover: 0 };
  const verdictStyle = VERDICT_STYLE[validation.verdict] || VERDICT_STYLE.FAIL;
  const simulation = alpha.simulation || {};
  const simState = alpha.simulation_state || simulationState(simulation);
  const issueCount = validation.issues?.length || 0;
  const simRan = simState === "completed" || simState === "failed";
  const simFailures = simulation.failures || [];
  const notSubmittable = simState === "completed" && simulation.submittable === false;
  const canRepair = Boolean(alpha.expression)
    && (validation.verdict === "WARN" || validation.verdict === "FAIL" || notSubmittable || simState === "failed");
  const repairLabel = (notSubmittable || simState === "failed") ? "repair from failures" : "repair with OpenRouter";

  return (
    <article className="af-card">
      <div className="af-card-head">
        <div className="af-title-row">
          <span className="af-index">A{String(index + 1).padStart(2, "0")}</span>
          <h2>{alpha.name || "alpha"}</h2>
          {alpha.archetype && <span className="af-tag">{alpha.archetype}</span>}
          {alpha.repaired && (
            <span className="af-tag af-repaired">
              <Wrench size={10} />
              repaired
            </span>
          )}
        </div>
        <div className="af-badge-row">
          <VerdictBadge styleDef={verdictStyle} />
          <SimulationBadge state={simState} />
        </div>
      </div>

      <div className="af-expr">
        <code>{alpha.expression || "No expression returned."}</code>
        {alpha.expression && (
          <button className="af-copy" onClick={onCopy} title="Copy expression">
            {copied ? <Check size={14} /> : <Copy size={14} />}
          </button>
        )}
      </div>

      <div className="af-metrics">
        <Metric icon={Gauge} label="Sharpe" value={metricValue(simulation.sharpe)} />
        <Metric icon={BadgeCheck} label="Fitness" value={metricValue(simulation.fitness)} />
        <Metric icon={Activity} label="Turnover" value={percentValue(simulation.turnover_pct)} />
        <Metric icon={GitCompare} label="Self-corr" value={corrValue(simulation.self_correlation)} tone={corrTone(simulation.self_correlation)} />
        <Metric icon={Hash} label="Alpha ID" value={simulation.alpha_id || "-"} compact />
        <Metric icon={Timer} label="Elapsed" value={formatSeconds(simulation.elapsed_s)} />
      </div>

      <div className="af-meta-row">
        <TurnoverBar value={validation.turnover ?? validation.predicted_turnover ?? 0} />
        <SettingsPills settings={alpha.settings} />
      </div>

      {alpha.hypothesis && <p className="af-hypothesis">{alpha.hypothesis}</p>}

      {(alpha.regime_note || alpha.turnover_note || alpha.change_note) && (
        <div className="af-notes">
          {alpha.regime_note && <NoteLine label="regime" text={alpha.regime_note} />}
          {alpha.turnover_note && <NoteLine label="turnover" text={alpha.turnover_note} />}
          {alpha.change_note && <NoteLine label="repair" text={alpha.change_note} accent />}
        </div>
      )}

      {simulation.status && simulation.status !== "OK" && simulation.status !== "queued" && (
        <div className="af-sim-error">
          <AlertTriangle size={14} />
          <span>{simulation.error || simulation.status}</span>
        </div>
      )}

      {simRan && (
        <div className="af-postsim">
          <div className="af-actions">
            {simulation.submittable && simulation.alpha_id ? (
              <button className="af-submit" disabled={alpha.submitting || !connected} onClick={onSubmit}
                title={connected ? "Submit to the BRAIN alpha pool" : "Connect Brain first"}>
                {alpha.submitting ? <Loader2 size={13} className="spin" /> : <Send size={13} />}
                {alpha.submitting ? "submitting…" : "submit to BRAIN"}
              </button>
            ) : notSubmittable ? (
              <span className="af-flag"><AlertTriangle size={12} /> not submittable</span>
            ) : null}
            <button className="af-ghost" onClick={onToggleEdit}>
              <SlidersHorizontal size={13} />{editOpen ? "close settings" : "edit settings"}
            </button>
            {simulation.alpha_id && !alpha.details && !alpha.detailsLoading && (
              <button className="af-ghost" onClick={onFetchDetails}><LineChart size={13} />load details</button>
            )}
          </div>

          {editOpen && (
            <SettingsEditor
              draft={draft || alpha.settings}
              onChange={onDraftChange}
              onSimulate={onResimulate}
              busy={alpha.resimulating}
              disabled={!connected}
            />
          )}

          {alpha.submission && <SubmissionResult submission={alpha.submission} />}

          {alpha.detailsLoading && (
            <div className="af-detail-loading"><Loader2 size={13} className="spin" /> fetching PnL, correlations &amp; checks…</div>
          )}
          {alpha.detailsError && (
            <div className="af-sim-error"><AlertTriangle size={14} /><span>details: {alpha.detailsError}</span></div>
          )}
          {alpha.details && <DetailsPanel details={alpha.details} />}
        </div>
      )}

      <div className="af-validator">
        <div className="af-validator-head">
          <button className="af-disclose" onClick={onToggle}>
            <span>{isOpen ? "-" : "+"}</span>
            validator - {issueCount} {issueCount === 1 ? "note" : "notes"}
          </button>
          {canRepair && onRepair && (
            <button className="af-repair" disabled={repairing || busy} onClick={onRepair}>
              {repairing ? <Loader2 size={13} className="spin" /> : <Wrench size={13} />}
              {repairing ? "repairing…" : repairLabel}
            </button>
          )}
        </div>
        {isOpen && (
          <div className="af-issues">
            {issueCount === 0 ? (
              <div className="af-clean">
                <ShieldCheck size={14} />
                deterministic validator passed
              </div>
            ) : validation.issues.map((issue, issueIndex) => (
              <div key={`${issue.rule}-${issueIndex}`} className="af-issue">
                <span className={issue.verdict === "FAIL" ? "fail" : "warn"}>{issue.verdict}</span>
                <div>
                  <p>{issue.message}</p>
                  {issue.fix && <small>{issue.fix}</small>}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </article>
  );
}

function JobsPanel({ jobs, onDelete, onRefresh }) {
  const [expanded, setExpanded] = useState({});

  if (!jobs.length) {
    return (
      <section className="af-jobs-panel">
        <div className="af-jobs-head">
          <Label icon={Clock}>background jobs</Label>
          <button className="af-ghost" onClick={onRefresh}><Activity size={13} />refresh</button>
        </div>
        <div className="af-jobs-empty">No jobs scheduled yet. Use "schedule (background)" to queue a forge job that runs even when this tab is closed.</div>
      </section>
    );
  }

  return (
    <section className="af-jobs-panel">
      <div className="af-jobs-head">
        <Label icon={Clock}>background jobs</Label>
        <button className="af-ghost" onClick={onRefresh}><Activity size={13} />refresh</button>
      </div>
      <div className="af-jobs-list">
        {jobs.map((job) => {
          const isExp = Boolean(expanded[job.id]);
          const statusColor = {
            queued: "var(--ink-dim)",
            running: "var(--steel)",
            completed: "var(--lime)",
            failed: "var(--red)",
          }[job.status] || "var(--ink-dim)";
          const rs = job.result_summary;
          return (
            <div key={job.id} className="af-job-card">
              <div className="af-job-row">
                <div className="af-job-info">
                  <span className="af-job-id">{job.id}</span>
                  <span className="af-job-label">{job.label || "(no label)"}</span>
                  <span className="af-job-status" style={{ color: statusColor }}>
                    {job.status === "running" && <Loader2 size={11} className="spin" />}
                    {job.status}
                  </span>
                  {rs && (
                    <span className="af-job-stats">{rs.passed}/{rs.count} passed</span>
                  )}
                </div>
                <div className="af-job-actions">
                  {job.logs?.length > 0 && (
                    <button className="af-ghost" onClick={() => setExpanded((p) => ({ ...p, [job.id]: !p[job.id] }))}>
                      {isExp ? "hide logs" : "logs"}
                    </button>
                  )}
                  {(job.status === "completed" || job.status === "failed") && (
                    <button className="af-ghost af-job-del" onClick={() => onDelete(job.id)} title="Remove job">
                      <Trash2 size={12} />
                    </button>
                  )}
                </div>
              </div>
              {isExp && job.logs?.length > 0 && (
                <div className="af-job-logs">
                  {job.logs.map((line, i) => (
                    <div key={i} className="af-log-line"><span>{line}</span></div>
                  ))}
                </div>
              )}
              {job.status === "completed" && job.result?.alphas && (
                <div className="af-job-result-note">
                  {job.result.alphas.length} alpha(s) generated — reload to view or run a new forge to load them.
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

const PHASE_LABEL = {
  researching: "Researching",
  generating:  "Generating",
  simulating:  "Simulating",
};

function ServerStatusBox() {
  const [srv, setSrv] = useState(null);

  useEffect(() => {
    let active = true;
    async function poll() {
      try {
        const data = await fetch(apiUrl("/api/status")).then((r) => r.json());
        if (active) setSrv(data);
      } catch {}
    }
    poll();
    const id = setInterval(poll, 2000);
    return () => { active = false; clearInterval(id); };
  }, []);

  if (!srv || srv.phase === "idle") return null;

  const phaseLabel = PHASE_LABEL[srv.phase] || srv.phase;
  const progress = srv.count > 0 ? `${srv.current}/${srv.count}` : "";
  const archLabel = srv.archetype && srv.archetype !== "any" ? ` · ${srv.archetype}` : "";
  const jobBadge = srv.job_id ? <span className="af-ss-job">bg</span> : null;
  const thinkBadge = srv.think_mode === "deep" ? <span className="af-ss-deep"><Brain size={9} />deep</span> : null;

  return (
    <div className="af-server-status">
      <Loader2 size={12} className="spin af-ss-spin" />
      <span className="af-ss-phase">{phaseLabel}</span>
      {progress && <span className="af-ss-progress">{progress}</span>}
      {srv.label && (
        <span className="af-ss-label" title={srv.label}>
          {srv.label.slice(0, 32)}{srv.label.length > 32 ? "…" : ""}
        </span>
      )}
      {archLabel && <span className="af-ss-arch">{archLabel}</span>}
      {thinkBadge}
      {jobBadge}
      <span className="af-ss-time">{formatSeconds(srv.elapsed_s)}</span>
    </div>
  );
}

function Label({ icon: Icon, children }) {
  return (
    <div className="af-label">
      <Icon size={13} />
      {children}
    </div>
  );
}

function StatusPill({ state, icon: Icon, label }) {
  return (
    <span className={`af-status ${state}`}>
      <Icon size={14} />
      {label}
    </span>
  );
}

function VerdictBadge({ styleDef }) {
  const Icon = styleDef.Icon;
  return (
    <span className="af-verdict" style={{ color: styleDef.color, background: styleDef.bg }}>
      <Icon size={13} />
      {styleDef.label}
    </span>
  );
}

function SimulationBadge({ state }) {
  const style = SIM_STYLE[state] || SIM_STYLE.failed;
  return (
    <span className="af-sim-badge" style={{ color: style.color, background: style.bg }}>
      {state === "running" && <Loader2 size={13} className="spin" />}
      {style.label}
    </span>
  );
}

function Metric({ icon: Icon, label, value, compact = false, tone }) {
  return (
    <div className={`af-metric${compact ? " compact" : ""}`}>
      <div>
        <Icon size={13} />
        <span>{label}</span>
      </div>
      <strong style={tone ? { color: tone } : undefined}>{value}</strong>
    </div>
  );
}

function Stat({ n, label, color }) {
  return (
    <span className="af-stat">
      <i style={{ background: color, boxShadow: `0 0 10px ${color}` }} />
      <strong style={{ color }}>{n}</strong>
      {label}
    </span>
  );
}

function TurnoverBar({ value }) {
  const turnover = Number.isFinite(Number(value)) ? Number(value) : 0;
  const color = turnover < 12.5 || turnover > 70 ? "var(--amber)" : "var(--lime)";
  return (
    <div className="af-turnover">
      <div className="af-turnover-head">
        <span>pred. turnover</span>
        <strong style={{ color }}>{turnover.toFixed(1)}%</strong>
      </div>
      <div className="af-bar">
        <div style={{ width: `${Math.min(100, Math.max(2, turnover))}%`, background: color }} />
        <i style={{ left: "12.5%" }} />
        <i style={{ left: "70%" }} />
      </div>
    </div>
  );
}

function SettingsPills({ settings = {} }) {
  const entries = [
    ["universe", settings.universe],
    ["neut", settings.neutralization],
    ["decay", settings.decay],
    ["trunc", settings.truncation],
    ["delay", settings.delay],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");
  return (
    <div className="af-pills">
      {entries.map(([key, value]) => (
        <span key={key}>{key}:{value}</span>
      ))}
    </div>
  );
}

function NoteLine({ label, text, accent = false }) {
  return (
    <div className="af-note-line">
      <span>{label}</span>
      <p className={accent ? "accent" : undefined}>{text}</p>
    </div>
  );
}

const NEUT_OPTIONS = ["None", "Market", "Sector", "Industry", "Subindustry"];

function SettingsEditor({ draft, onChange, onSimulate, busy, disabled }) {
  const d = draft || {};
  return (
    <div className="af-settings-editor">
      <div className="af-set-grid">
        <label>universe<input value="TOP3000" disabled /></label>
        <label>neutralization
          <select value={d.neutralization || "Subindustry"} onChange={(e) => onChange("neutralization", e.target.value)} disabled={busy}>
            {NEUT_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </label>
        <label>decay
          <input type="number" min="0" max="30" value={d.decay ?? 4} onChange={(e) => onChange("decay", e.target.value)} disabled={busy} />
        </label>
        <label>truncation
          <input type="number" step="0.01" min="0" max="0.2" value={d.truncation ?? 0.05} onChange={(e) => onChange("truncation", e.target.value)} disabled={busy} />
        </label>
        <label>delay
          <select value={d.delay ?? 1} onChange={(e) => onChange("delay", Number(e.target.value))} disabled={busy}>
            <option value={0}>0</option>
            <option value={1}>1</option>
          </select>
        </label>
      </div>
      <button className="af-resim" onClick={onSimulate} disabled={busy || disabled} title={disabled ? "Connect Brain first" : "Re-simulate with these settings"}>
        {busy ? <Loader2 size={13} className="spin" /> : <Activity size={13} />}
        {busy ? "simulating…" : "simulate with these settings"}
      </button>
    </div>
  );
}

function SubmissionResult({ submission }) {
  const status = (submission.status || "").toUpperCase();
  const ok = status === "SUBMITTED";
  return (
    <div className={`af-submission ${ok ? "ok" : "err"}`}>
      <div className="af-submission-head">
        {ok ? <BadgeCheck size={15} /> : <AlertTriangle size={15} />}
        <strong>{status || "RESULT"}</strong>
        <span>{submission.message || submission.error || ""}</span>
      </div>
      {submission.failed_checks?.length > 0 && <ChecksGrid checks={submission.failed_checks} />}
    </div>
  );
}

function DetailsPanel({ details }) {
  const { pnl, correlations, checks, stats } = details;
  return (
    <div className="af-details">
      {pnl && pnl.points?.length > 1 && (
        <div className="af-pnl-wrap">
          <div className="af-detail-head">
            <span>cumulative pnl</span>
            <span>{pnl.n} days · final {fmtNum(pnl.final)} · peak {fmtNum(pnl.max)}</span>
          </div>
          <PnlChart pnl={pnl} />
        </div>
      )}
      <div className="af-detail-row">
        <Corr label="self-corr" c={correlations?.self} />
        <Corr label="prod-corr" c={correlations?.prod} />
        {stats?.os_sharpe != null && <Corr label="OS sharpe" c={{ value: stats.os_sharpe }} />}
        {stats?.drawdown != null && <Corr label="drawdown" c={{ value: stats.drawdown }} />}
      </div>
      <ChecksGrid checks={checks} />
    </div>
  );
}

function Corr({ label, c }) {
  if (!c || c.value == null) return null;
  const result = (c.result || "").toUpperCase();
  const color = result === "FAIL" ? "var(--red)" : result === "PASS" ? "var(--lime)" : "var(--ink)";
  return (
    <div className="af-corr">
      <span>{label}</span>
      <strong style={{ color }}>{fmtNum(c.value)}{c.limit != null ? ` / ${fmtNum(c.limit)}` : ""}</strong>
    </div>
  );
}

function ChecksGrid({ checks }) {
  if (!checks?.length) return null;
  return (
    <div className="af-checks">
      {checks.map((c, i) => {
        const result = (c.result || "").toUpperCase();
        const cls = result === "PASS" ? "pass" : result === "FAIL" ? "fail" : "pend";
        return (
          <div key={`${c.name}-${i}`} className={`af-check ${cls}`} title={checkTitle(c)}>
            <span className="af-check-name">{prettyCheck(c.name)}</span>
            <span className="af-check-res">{result || "—"}</span>
          </div>
        );
      })}
    </div>
  );
}

function PnlChart({ pnl }) {
  const pts = pnl?.points || [];
  if (pts.length < 2) return null;
  const W = 600;
  const H = 80;
  const pad = 4;
  const min = Math.min(...pts);
  const max = Math.max(...pts);
  const range = max - min || 1;
  const stepX = (W - pad * 2) / (pts.length - 1);
  const y = (v) => H - pad - ((v - min) / range) * (H - pad * 2);
  const line = pts.map((v, i) => `${i === 0 ? "M" : "L"}${(pad + i * stepX).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const lastX = (pad + (pts.length - 1) * stepX).toFixed(1);
  const area = `${line} L${lastX},${H - pad} L${pad},${H - pad} Z`;
  const up = pts[pts.length - 1] >= 0;
  const color = up ? "var(--lime)" : "var(--red)";
  const zeroY = min <= 0 && 0 <= max ? y(0).toFixed(1) : null;
  return (
    <svg className="af-pnl" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="cumulative pnl">
      <path d={area} fill={color} opacity="0.12" />
      {zeroY != null && (
        <line x1={pad} x2={W - pad} y1={zeroY} y2={zeroY} stroke="var(--ink-faint)" strokeDasharray="4 4" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      )}
      <path d={line} fill="none" stroke={color} strokeWidth="1.6" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function prettyCheck(name) {
  return String(name || "").replace(/_/g, " ").toLowerCase();
}

function checkTitle(c) {
  const parts = [prettyCheck(c.name)];
  if (c.value != null) parts.push(`value ${fmtNum(c.value)}`);
  if (c.limit != null) parts.push(`limit ${fmtNum(c.limit)}`);
  return parts.join(" · ");
}

function fmtNum(value) {
  if (value == null || value === "") return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  return Math.abs(n) >= 1000 ? n.toFixed(0) : n.toFixed(2);
}

function upsertAlpha(items, index, alpha) {
  const next = [...items];
  next[index] = alpha;
  return next.filter(Boolean);
}

function simulationState(simulation) {
  if (!simulation || !simulation.status) return "queued";
  if (simulation.status === "queued") return "queued";
  if (simulation.status === "OK") return "completed";
  return "failed";
}

async function apiPost(path, body) {
  let response;
  try {
    response = await fetch(apiUrl(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
  } catch (err) {
    throw new Error(networkErrorMessage(err));
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || data.error || `${path} failed (${response.status})`);
  }
  return data;
}

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

function wsUrl(path) {
  if (WS_BASE) return `${WS_BASE}${path}`;
  if (API_BASE) {
    return `${API_BASE.replace(/^https:/, "wss:").replace(/^http:/, "ws:")}${path}`;
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}

function networkErrorMessage(err) {
  const raw = err?.message || "Failed to fetch";
  return `${raw}. Backend not reachable through ${apiUrl("/api")}. Start FastAPI with npm run api and restart Vite with npm run dev.`;
}

function metricValue(value) {
  return typeof value === "number" ? value.toFixed(2) : "-";
}

function corrValue(correlation) {
  if (!correlation || correlation.value == null) return "-";
  const n = Number(correlation.value);
  return Number.isFinite(n) ? n.toFixed(2) : "-";
}

function corrTone(correlation) {
  if (!correlation) return undefined;
  const result = (correlation.result || "").toUpperCase();
  if (result === "FAIL") return "var(--red)";
  if (result === "PASS") return "var(--lime)";
  return undefined;
}

function providerLabel(id) {
  return { openrouter: "OpenRouter", claude_code: "Claude Code", anthropic: "Anthropic" }[id] || id;
}

function percentValue(value) {
  return typeof value === "number" ? `${value.toFixed(1)}%` : "-";
}

function formatSeconds(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}m ${seconds}s`;
}

const CSS = `
:root{
  --bg:#0b0d0c;
  --panel:#111412;
  --panel-hi:#171c18;
  --line:rgba(223,231,215,0.10);
  --ink:#eceee8;
  --ink-dim:#a4ab9e;
  --ink-faint:#666f61;
  --lime:#c7f24f;
  --lime-dim:rgba(199,242,79,0.13);
  --amber:#f1b454;
  --amber-dim:rgba(241,180,84,0.13);
  --red:#ef7165;
  --red-dim:rgba(239,113,101,0.13);
  --steel:#7fc0d2;
  --steel-dim:rgba(127,192,210,0.13);
  --violet:#b99bff;
  --mono:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
  --sans:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
*{box-sizing:border-box}
body{margin:0}
button,textarea{font:inherit}
.af-shell{min-height:100vh;background:radial-gradient(circle at 18% 0%,rgba(127,192,210,0.12),transparent 32%),radial-gradient(circle at 82% 10%,rgba(199,242,79,0.10),transparent 28%),var(--bg);color:var(--ink);font-family:var(--sans);position:relative;overflow:hidden}
.af-grid-bg{position:fixed;inset:0;background-image:linear-gradient(var(--line) 1px,transparent 1px),linear-gradient(90deg,var(--line) 1px,transparent 1px);background-size:44px 44px;mask-image:linear-gradient(to bottom,#000 0%,rgba(0,0,0,0.65) 48%,transparent 100%);opacity:.58;pointer-events:none}
.af-main{width:min(1120px,calc(100vw - 32px));margin:0 auto;padding:42px 0 72px;position:relative;z-index:1}
.af-header{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin-bottom:24px}
.af-kicker{display:flex;align-items:center;gap:8px;color:var(--ink-faint);font-family:var(--mono);font-size:11px;letter-spacing:.15em;text-transform:uppercase}
.af-kicker svg{color:var(--lime)}
h1{font-size:clamp(44px,8vw,78px);line-height:.92;margin:12px 0 0;letter-spacing:0;font-weight:760}
.af-status-wrap{display:flex;align-items:center;gap:10px;flex-wrap:wrap;justify-content:flex-end}
.af-status,.af-verdict,.af-sim-badge{display:inline-flex;align-items:center;gap:7px;border-radius:8px;padding:7px 10px;font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase}
.af-status{border:1px solid var(--line);background:rgba(255,255,255,.03);color:var(--ink-dim)}
.af-status.connected{color:var(--lime);background:var(--lime-dim);border-color:rgba(199,242,79,.32)}
.af-status.authenticating{color:var(--steel);background:var(--steel-dim);border-color:rgba(127,192,210,.32)}
.af-status.disconnected{color:var(--red);background:var(--red-dim);border-color:rgba(239,113,101,.30)}
.af-secondary,.af-forge{display:inline-flex;align-items:center;justify-content:center;gap:9px;border:none;border-radius:9px;cursor:pointer;transition:transform .14s,box-shadow .18s,opacity .18s}
.af-secondary{height:38px;padding:0 14px;color:var(--ink);background:rgba(255,255,255,.05);border:1px solid var(--line);font-family:var(--mono);font-size:12px}
.af-secondary:hover:not(:disabled){border-color:rgba(223,231,215,.24);transform:translateY(-1px)}
.af-secondary:disabled,.af-forge:disabled{opacity:.55;cursor:default}
.af-panel,.af-card,.af-log-panel{background:linear-gradient(180deg,var(--panel),#0e100f);border:1px solid var(--line);border-radius:8px;box-shadow:0 18px 60px rgba(0,0,0,.22)}
.af-panel{padding:22px}
.af-connect-note{display:flex;align-items:center;justify-content:space-between;gap:12px;color:var(--ink-dim);font-family:var(--mono);font-size:12px;border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:18px}
.af-label{display:flex;align-items:center;gap:7px;color:var(--ink-faint);font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase}
.af-label svg{color:var(--lime)}
.af-input{width:100%;margin-top:10px;background:#090b0a;border:1px solid var(--line);border-radius:8px;color:var(--ink);font-size:14px;line-height:1.55;padding:13px 14px;resize:vertical;outline:none;transition:border-color .18s,box-shadow .18s}
.af-input:focus{border-color:rgba(199,242,79,.44);box-shadow:0 0 0 3px rgba(199,242,79,.09)}
.af-input::placeholder{color:var(--ink-faint)}
.af-controls{display:grid;grid-template-columns:minmax(0,1fr) auto auto auto;gap:18px;align-items:end;margin-top:20px}
.af-engines{display:flex;flex-wrap:wrap;gap:12px;margin-top:18px;border-top:1px solid var(--line);padding-top:16px}
.af-engine{display:flex;flex-direction:column;gap:6px;flex:1;min-width:148px}
.af-engine>span{display:inline-flex;align-items:center;gap:6px;color:var(--ink-faint);font-family:var(--mono);font-size:10px;letter-spacing:.14em;text-transform:uppercase}
.af-engine>span svg{color:var(--lime)}
.af-engine select{background:#0e110f;border:1px solid var(--line);border-radius:8px;color:var(--ink);font-family:var(--mono);font-size:13px;padding:8px 9px;outline:none;cursor:pointer}
.af-engine select:focus{border-color:rgba(199,242,79,.4)}
.af-engine select:disabled{opacity:.6;cursor:default}
.af-chip-row{display:flex;flex-wrap:wrap;gap:7px;margin-top:9px}
.af-chip{background:transparent;color:var(--ink-dim);border:1px solid var(--line);border-radius:8px;padding:7px 11px;font-family:var(--mono);font-size:12px;cursor:pointer}
.af-chip:hover:not(:disabled){color:var(--ink);border-color:rgba(223,231,215,.24)}
.af-chip.active{background:var(--lime);border-color:var(--lime);color:#12160f;font-weight:700}
.af-count-block{min-width:130px}
.af-stepper{display:flex;align-items:center;gap:4px;border:1px solid var(--line);border-radius:8px;padding:4px;margin-top:9px;background:#090b0a}
.af-stepper button{display:grid;place-items:center;width:30px;height:28px;border:none;border-radius:6px;background:transparent;color:var(--ink-dim);cursor:pointer}
.af-stepper button:hover:not(:disabled){color:var(--lime);background:var(--lime-dim)}
.af-count-input{width:42px;text-align:center;color:var(--lime);font-family:var(--mono);font-size:15px;background:transparent;border:none;outline:none;padding:0;-moz-appearance:textfield}
.af-count-input::-webkit-inner-spin-button,.af-count-input::-webkit-outer-spin-button{-webkit-appearance:none;margin:0}
.af-count-note{margin-top:5px;color:var(--ink-faint);font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-align:center}
.af-forge{height:42px;padding:0 20px;background:var(--lime);color:#11150e;font-weight:800;min-width:154px}
.af-forge:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 16px 36px -18px var(--lime)}
.af-toggle{display:inline-flex;align-items:center;gap:8px;height:42px;padding:0 14px;border:1px solid var(--line);border-radius:9px;background:rgba(255,255,255,.04);color:var(--ink-dim);font-family:var(--mono);font-size:12px;cursor:pointer;white-space:nowrap;transition:border-color .16s,color .16s,transform .14s}
.af-toggle:hover:not(:disabled){transform:translateY(-1px);border-color:rgba(223,231,215,.24)}
.af-toggle:disabled{opacity:.55;cursor:default}
.af-toggle .af-toggle-dot{width:8px;height:8px;border-radius:8px;background:var(--ink-faint);transition:background .16s,box-shadow .16s}
.af-toggle svg{color:var(--ink-faint)}
.af-toggle.on{color:var(--steel);border-color:rgba(127,192,210,.4)}
.af-toggle.on svg{color:var(--steel)}
.af-toggle.on .af-toggle-dot{background:var(--steel);box-shadow:0 0 8px var(--steel)}
.af-intent-header{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:6px}
.af-intent-header>.af-label{margin-bottom:0}
.af-hyp-controls{display:flex;align-items:center;gap:8px;flex-shrink:0}
.af-hyp-theme{height:32px;padding:0 10px;background:rgba(255,255,255,.05);border:1px solid var(--line);border-radius:7px;color:var(--ink);font-family:var(--mono);font-size:12px;width:160px;outline:none;transition:border-color .15s}
.af-hyp-theme:focus{border-color:rgba(223,231,215,.3)}
.af-hyp-theme::placeholder{color:var(--ink-faint)}
.af-hyp-btn{height:32px;padding:0 12px;font-size:12px;background:rgba(255,255,255,.06);border:1px solid var(--line);border-radius:7px;color:var(--ink);white-space:nowrap}
.af-hyp-btn:hover:not(:disabled){background:rgba(255,255,255,.1);border-color:rgba(223,231,215,.25)}
.af-hyp-panel{margin-top:10px;display:flex;flex-direction:column;gap:7px}
.af-hyp-panel-head{display:flex;align-items:center;gap:6px;color:var(--ink-faint);font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;margin-bottom:2px}
.af-hyp-card{text-align:left;background:rgba(255,255,255,.03);border:1px solid var(--line);border-radius:9px;padding:11px 14px;cursor:pointer;transition:background .15s,border-color .15s;width:100%}
.af-hyp-card:hover{background:rgba(255,255,255,.07);border-color:rgba(223,231,215,.22)}
.af-hyp-card-selected{background:rgba(199,242,79,.07);border-color:rgba(199,242,79,.4)}
.af-hyp-active{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-top:8px;padding:8px 11px;background:rgba(199,242,79,.06);border:1px solid rgba(199,242,79,.32);border-radius:8px;font-family:var(--mono);font-size:11px}
.af-hyp-active svg{color:var(--lime);flex-shrink:0}
.af-hyp-active-label{color:var(--lime);letter-spacing:.06em;text-transform:uppercase;white-space:nowrap}
.af-hyp-active-title{color:var(--ink);font-weight:700;white-space:nowrap}
.af-hyp-active-fields{color:var(--ink-faint);flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.af-hyp-active-clear{display:grid;place-items:center;width:20px;height:20px;border:none;border-radius:4px;background:rgba(255,255,255,.06);color:var(--ink-faint);cursor:pointer;flex-shrink:0;margin-left:auto}
.af-hyp-active-clear:hover{color:var(--red);background:var(--red-dim)}
.af-hyp-card-top{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.af-hyp-title{color:var(--ink);font-weight:600;font-size:13px;flex:1}
.af-hyp-novelty{font-family:var(--mono);font-size:11px;color:var(--amber);white-space:nowrap}
.af-hyp-claim{color:var(--ink);font-size:12.5px;line-height:1.5;margin-bottom:6px}
.af-hyp-meta{display:flex;flex-wrap:wrap;gap:10px;font-family:var(--mono);font-size:11px;color:var(--ink-faint)}
.af-research{margin-top:16px;padding:18px 20px;background:linear-gradient(180deg,var(--panel),#0e100f);border:1px solid rgba(127,192,210,.28);border-radius:8px;box-shadow:0 18px 60px rgba(0,0,0,.22)}
.af-research p{margin:12px 0 0;color:var(--ink);font-size:13.5px;line-height:1.7;white-space:pre-wrap}
.af-sources-head{margin:16px 0 9px;color:var(--ink-faint);font-family:var(--mono);font-size:10px;letter-spacing:.2em;text-transform:uppercase}
.af-sources{display:flex;flex-wrap:wrap;gap:7px}
.af-src{display:inline-flex;align-items:center;gap:6px;color:var(--steel);text-decoration:none;border:1px solid rgba(127,192,210,.28);border-radius:7px;padding:5px 9px;font-family:var(--mono);font-size:11px;transition:background .15s}
.af-src:hover{background:rgba(127,192,210,.1)}
.af-src svg{flex-shrink:0}
.af-error{display:flex;align-items:center;gap:9px;margin-top:16px;border:1px solid rgba(239,113,101,.32);background:var(--red-dim);color:var(--red);border-radius:8px;padding:13px 14px;font-size:13px}
.af-summary{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:28px 2px 16px;color:var(--ink-dim);font-family:var(--mono);font-size:12px}
.af-stat{display:inline-flex;align-items:center;gap:7px}
.af-stat i{width:7px;height:7px;border-radius:7px}
.af-elapsed{display:inline-flex;align-items:center;gap:6px;color:var(--steel)}
.af-token-usage{display:inline-flex;align-items:center;gap:6px;color:var(--amber);font-family:var(--mono);font-size:12px;margin-left:4px}
.af-token-reset{background:none;border:none;color:var(--ink-faint);cursor:pointer;font-size:14px;padding:0 2px;line-height:1;margin-left:2px}
.af-token-reset:hover{color:var(--red)}
.af-results{display:flex;flex-direction:column;gap:14px}
.af-card{padding:18px}
.af-card-head{display:flex;align-items:flex-start;justify-content:space-between;gap:14px}
.af-title-row,.af-badge-row{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.af-index{font-family:var(--mono);font-size:12px;color:var(--lime)}
h2{margin:0;font-size:20px;line-height:1.2;letter-spacing:0}
.af-tag{border:1px solid var(--line);border-radius:7px;padding:4px 8px;color:var(--ink-dim);font-family:var(--mono);font-size:11px}
.af-sim-badge{min-width:92px;justify-content:center}
.af-expr{position:relative;margin-top:15px;background:#070908;border:1px solid var(--line);border-radius:8px;padding:14px 46px 14px 14px;overflow:auto}
.af-expr code{white-space:pre-wrap;word-break:break-word;color:#e3ebcf;font-family:var(--mono);font-size:13px;line-height:1.55}
.af-copy{position:absolute;right:9px;top:9px;width:30px;height:30px;border:1px solid var(--line);border-radius:7px;background:#111412;color:var(--ink-dim);display:grid;place-items:center;cursor:pointer}
.af-copy:hover{color:var(--lime);border-color:rgba(199,242,79,.34)}
.af-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(92px,1fr));gap:9px;margin-top:14px}
.af-metric{min-height:66px;border:1px solid var(--line);border-radius:8px;background:rgba(255,255,255,.025);padding:10px}
.af-metric div{display:flex;align-items:center;gap:6px;color:var(--ink-faint);font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase}
.af-metric div svg{color:var(--steel)}
.af-metric strong{display:block;margin-top:9px;color:var(--ink);font-family:var(--mono);font-size:16px;line-height:1.25;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.af-metric.compact strong{font-size:12px}
.af-meta-row{display:flex;align-items:flex-end;gap:18px;flex-wrap:wrap;margin-top:14px}
.af-turnover{width:min(260px,100%)}
.af-turnover-head{display:flex;justify-content:space-between;color:var(--ink-faint);font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;margin-bottom:7px}
.af-bar{height:7px;position:relative;background:#080a09;border:1px solid var(--line);border-radius:7px}
.af-bar div{height:100%;border-radius:7px}
.af-bar i{position:absolute;top:-3px;width:1px;height:12px;background:var(--ink-faint);opacity:.8}
.af-pills{display:flex;flex-wrap:wrap;gap:6px}
.af-pills span{border:1px solid var(--line);border-radius:7px;background:rgba(255,255,255,.025);color:var(--ink-dim);font-family:var(--mono);font-size:11px;padding:5px 8px}
.af-hypothesis{margin:14px 0 0;color:var(--ink);font-size:13.5px;line-height:1.65}
.af-notes{display:flex;flex-direction:column;gap:6px;margin-top:12px}
.af-note-line{display:grid;grid-template-columns:74px minmax(0,1fr);gap:10px;font-size:12px;line-height:1.55}
.af-note-line span{color:var(--ink-faint);font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;padding-top:2px}
.af-note-line p{margin:0;color:var(--ink-dim)}
.af-note-line p.accent{color:var(--lime)}
.af-sim-error{display:flex;gap:8px;align-items:flex-start;margin-top:13px;color:var(--red);background:var(--red-dim);border:1px solid rgba(239,113,101,.28);border-radius:8px;padding:10px 11px;font-size:12.5px;line-height:1.45}
.af-postsim{margin-top:15px;border-top:1px dashed var(--line);padding-top:14px;display:flex;flex-direction:column;gap:12px}
.af-actions{display:flex;flex-wrap:wrap;align-items:center;gap:9px}
.af-submit{display:inline-flex;align-items:center;gap:8px;height:34px;padding:0 16px;border:none;border-radius:8px;background:var(--lime);color:#11150e;font-family:var(--mono);font-size:12.5px;font-weight:700;cursor:pointer;transition:transform .14s,box-shadow .18s,opacity .18s}
.af-submit:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 12px 28px -14px var(--lime)}
.af-submit:disabled{opacity:.5;cursor:default}
.af-flag{display:inline-flex;align-items:center;gap:6px;color:var(--amber);background:var(--amber-dim);border:1px solid rgba(241,180,84,.3);border-radius:8px;padding:6px 11px;font-family:var(--mono);font-size:11.5px}
.af-ghost{display:inline-flex;align-items:center;gap:7px;height:34px;padding:0 12px;border:1px solid var(--line);border-radius:8px;background:rgba(255,255,255,.04);color:var(--ink-dim);font-family:var(--mono);font-size:12px;cursor:pointer;transition:color .15s,border-color .15s}
.af-ghost:hover{color:var(--ink);border-color:rgba(223,231,215,.24)}
.af-settings-editor{border:1px solid var(--line);border-radius:8px;background:#0a0c0b;padding:14px}
.af-set-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px}
.af-set-grid label{display:flex;flex-direction:column;gap:6px;color:var(--ink-faint);font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase}
.af-set-grid input,.af-set-grid select{background:#0e110f;border:1px solid var(--line);border-radius:7px;color:var(--ink);font-family:var(--mono);font-size:13px;padding:8px 9px;outline:none}
.af-set-grid input:focus,.af-set-grid select:focus{border-color:rgba(199,242,79,.4)}
.af-set-grid input:disabled{color:var(--ink-faint);opacity:.7}
.af-resim{display:inline-flex;align-items:center;gap:8px;margin-top:12px;height:34px;padding:0 14px;border:1px solid rgba(127,192,210,.4);border-radius:8px;background:var(--steel-dim);color:var(--steel);font-family:var(--mono);font-size:12px;cursor:pointer;transition:background .15s}
.af-resim:hover:not(:disabled){background:rgba(127,192,210,.18)}
.af-resim:disabled{opacity:.5;cursor:default}
.af-submission{border-radius:8px;padding:11px 12px;border:1px solid var(--line)}
.af-submission.ok{color:var(--lime);background:var(--lime-dim);border-color:rgba(199,242,79,.32)}
.af-submission.err{color:var(--red);background:var(--red-dim);border-color:rgba(239,113,101,.3)}
.af-submission-head{display:flex;align-items:center;gap:9px;font-size:13px}
.af-submission-head strong{font-family:var(--mono);letter-spacing:.08em}
.af-submission-head span{color:var(--ink-dim);font-size:12px}
.af-details{display:flex;flex-direction:column;gap:13px}
.af-pnl-wrap{display:flex;flex-direction:column;gap:7px}
.af-detail-head{display:flex;justify-content:space-between;gap:10px;color:var(--ink-faint);font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase}
.af-pnl{width:100%;height:80px;display:block;background:#070908;border:1px solid var(--line);border-radius:8px}
.af-detail-row{display:flex;flex-wrap:wrap;gap:18px}
.af-corr{display:flex;flex-direction:column;gap:4px}
.af-corr span{color:var(--ink-faint);font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase}
.af-corr strong{font-family:var(--mono);font-size:14px}
.af-checks{display:flex;flex-wrap:wrap;gap:6px}
.af-check{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:7px;padding:5px 9px;font-family:var(--mono);font-size:11px;background:rgba(255,255,255,.02)}
.af-check-name{color:var(--ink-dim)}
.af-check-res{font-weight:600;letter-spacing:.05em}
.af-check.pass .af-check-res{color:var(--lime)}
.af-check.fail{border-color:rgba(239,113,101,.4)}
.af-check.fail .af-check-res{color:var(--red)}
.af-check.pend .af-check-res{color:var(--ink-faint)}
.af-detail-loading{display:inline-flex;align-items:center;gap:8px;color:var(--steel);font-family:var(--mono);font-size:12px}
.af-validator{border-top:1px solid var(--line);margin-top:15px;padding-top:13px}
.af-validator-head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
.af-disclose{display:inline-flex;align-items:center;gap:8px;background:transparent;border:none;color:var(--ink-dim);font-family:var(--mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;padding:0;cursor:pointer}
.af-disclose:hover{color:var(--ink)}
.af-repair{display:inline-flex;align-items:center;gap:8px;padding:7px 13px;border:1px solid rgba(199,242,79,.4);border-radius:8px;background:var(--lime-dim);color:var(--lime);font-family:var(--mono);font-size:12px;cursor:pointer;transition:background .15s,transform .14s}
.af-repair:hover:not(:disabled){background:rgba(199,242,79,.2);transform:translateY(-1px)}
.af-repair:disabled{opacity:.6;cursor:default}
.af-repaired{display:inline-flex;align-items:center;gap:5px;color:var(--lime);border-color:rgba(199,242,79,.4)}
.af-issues{display:flex;flex-direction:column;gap:10px;margin-top:12px}
.af-clean{display:flex;align-items:center;gap:8px;color:var(--lime);font-size:12.5px}
.af-issue{display:flex;gap:10px;align-items:flex-start}
.af-issue>span{font-family:var(--mono);font-size:10px;letter-spacing:.1em;width:40px;padding-top:2px}
.af-issue>span.fail{color:var(--red)}
.af-issue>span.warn{color:var(--amber)}
.af-issue p{margin:0;color:var(--ink);font-size:12.5px;line-height:1.5}
.af-issue small{display:block;margin-top:3px;color:var(--ink-dim);font-size:11.5px;line-height:1.45}
.af-empty{display:flex;align-items:center;justify-content:center;gap:10px;margin-top:28px;min-height:104px;border:1px dashed var(--line);border-radius:8px;color:var(--ink-faint);font-family:var(--mono);font-size:13px}
.af-log-panel{margin-top:18px;padding:16px}
.af-log-head{display:flex;justify-content:space-between;align-items:center;gap:12px}
.af-log-head>span{display:inline-flex;align-items:center;gap:6px;color:var(--steel);font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.1em}
.af-log-list{margin-top:12px;max-height:220px;overflow:auto;border:1px solid var(--line);border-radius:8px;background:#080a09}
.af-log-empty{padding:13px;color:var(--ink-faint);font-family:var(--mono);font-size:12px}
.af-log-line{display:grid;grid-template-columns:86px minmax(0,1fr);gap:10px;padding:9px 11px;border-bottom:1px solid var(--line);font-family:var(--mono);font-size:12px}
.af-log-line:last-child{border-bottom:none}
.af-log-line time{color:var(--ink-faint)}
.af-log-line span{color:var(--ink-dim)}
.af-server-status{position:fixed;bottom:22px;right:22px;z-index:9999;display:inline-flex;align-items:center;gap:7px;padding:9px 14px;background:rgba(17,20,18,0.92);border:1px solid rgba(127,192,210,.5);border-radius:12px;backdrop-filter:blur(12px);box-shadow:0 8px 32px rgba(0,0,0,.45),0 0 0 1px rgba(127,192,210,.12);font-family:var(--mono);font-size:12px;color:var(--ink);animation:af-ss-in .25s ease}
@keyframes af-ss-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.af-ss-spin{color:var(--steel);flex-shrink:0}
.af-ss-phase{color:var(--steel);font-weight:600;letter-spacing:.06em;text-transform:uppercase;font-size:11px}
.af-ss-progress{color:var(--lime);font-weight:700;font-size:13px;min-width:28px}
.af-ss-label{color:var(--ink-dim);max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.af-ss-arch{color:var(--ink-faint)}
.af-ss-deep{display:inline-flex;align-items:center;gap:3px;color:var(--violet);font-size:10px;border:1px solid rgba(185,155,255,.35);border-radius:5px;padding:1px 5px}
.af-ss-job{color:var(--amber);background:var(--amber-dim);border:1px solid rgba(241,180,84,.3);border-radius:5px;padding:1px 5px;font-size:10px}
.af-ss-time{color:var(--ink-faint);margin-left:2px}
.af-think-row{display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin-top:16px;padding-top:14px;border-top:1px solid var(--line)}
.af-think-chips{display:flex;gap:6px}
.af-think-engine-row{display:flex;align-items:center;gap:9px;flex-wrap:wrap;width:100%;margin-top:2px}
.af-think-engine-label{display:inline-flex;align-items:center;gap:5px;color:var(--violet);font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;white-space:nowrap}
.af-think-engine-select{background:#0e110f;border:1px solid rgba(185,155,255,.35);border-radius:7px;color:var(--violet);font-family:var(--mono);font-size:12px;padding:6px 9px;outline:none;cursor:pointer}
.af-think-engine-select:focus{border-color:rgba(185,155,255,.6)}
.af-chip-deep{border-color:rgba(185,155,255,.4);color:var(--violet)}
.af-chip-deep:hover:not(:disabled){color:var(--violet);border-color:rgba(185,155,255,.6)}
.af-chip-deep.active{background:rgba(185,155,255,.18);border-color:var(--violet);color:var(--violet);font-weight:700}
.af-think-note{color:var(--violet);font-family:var(--mono);font-size:11px;opacity:.8}
.af-schedule-row{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-top:10px}
.af-schedule-btn{color:var(--steel);border-color:rgba(127,192,210,.4)}
.af-schedule-btn:hover:not(:disabled){background:var(--steel-dim)}
.af-job-live{display:inline-flex;align-items:center;gap:6px;color:var(--steel);font-family:var(--mono);font-size:11px}
.af-jobs-tab-btn{display:inline-flex;align-items:center;gap:6px;height:32px;padding:0 12px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--ink-dim);font-family:var(--mono);font-size:12px;cursor:pointer;margin-left:auto}
.af-jobs-tab-btn:hover{color:var(--ink);border-color:rgba(223,231,215,.28)}
.af-jobs-tab-btn.active{border-color:rgba(127,192,210,.4);color:var(--steel);background:var(--steel-dim)}
.af-job-badge{display:inline-flex;align-items:center;justify-content:center;width:17px;height:17px;border-radius:17px;background:var(--steel);color:#0b0d0c;font-size:10px;font-weight:700;margin-left:2px}
.af-jobs-panel{margin-top:16px;padding:18px;background:linear-gradient(180deg,var(--panel),#0e100f);border:1px solid rgba(127,192,210,.2);border-radius:8px}
.af-jobs-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:14px}
.af-jobs-empty{color:var(--ink-faint);font-family:var(--mono);font-size:12px;line-height:1.6}
.af-jobs-list{display:flex;flex-direction:column;gap:9px}
.af-job-card{border:1px solid var(--line);border-radius:8px;padding:12px 14px;background:rgba(255,255,255,.02)}
.af-job-row{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
.af-job-info{display:flex;align-items:center;gap:10px;flex-wrap:wrap;min-width:0}
.af-job-id{font-family:var(--mono);font-size:10px;color:var(--ink-faint);letter-spacing:.08em}
.af-job-label{font-size:13px;color:var(--ink);flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.af-job-status{display:inline-flex;align-items:center;gap:5px;font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase}
.af-job-stats{font-family:var(--mono);font-size:11px;color:var(--lime)}
.af-job-actions{display:flex;align-items:center;gap:7px;flex-shrink:0}
.af-job-del{color:var(--ink-faint)}
.af-job-del:hover{color:var(--red);border-color:rgba(239,113,101,.4)}
.af-job-logs{margin-top:10px;max-height:160px;overflow:auto;border:1px solid var(--line);border-radius:7px;background:#070908}
.af-job-logs .af-log-line{grid-template-columns:1fr}
.af-job-result-note{margin-top:9px;color:var(--ink-faint);font-family:var(--mono);font-size:11px;border-top:1px solid var(--line);padding-top:9px}
.spin{animation:afspin .9s linear infinite}
@keyframes afspin{to{transform:rotate(360deg)}}
::-webkit-scrollbar{height:8px;width:8px}
::-webkit-scrollbar-thumb{background:rgba(223,231,215,.16);border-radius:8px}
::-webkit-scrollbar-track{background:transparent}
@media (max-width:880px){
  .af-header{flex-direction:column}
  .af-status-wrap{justify-content:flex-start}
  .af-controls{grid-template-columns:1fr}
  .af-forge{width:100%}
}
@media (max-width:560px){
  .af-main{width:min(100vw - 22px,1120px);padding-top:26px}
  .af-panel,.af-card,.af-log-panel{padding:14px}
  .af-card-head{flex-direction:column}
  .af-connect-note{flex-direction:column;align-items:flex-start}
  .af-note-line{grid-template-columns:1fr}
}
`;
