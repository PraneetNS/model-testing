"use client";
import React, { useState, useEffect, useCallback } from "react";
import {
  ShieldCheck, ShieldAlert, Shield, RefreshCw,
  Award, GitBranch, CheckCircle2, XCircle,
  AlertTriangle, Copy, ExternalLink,
} from "lucide-react";
import { apiGet, apiPost, API_BASE } from "@/lib/api";

// ── Types ────────────────────────────────────────────────────────────────────

interface GovernanceScore {
  model_id: string;
  overall_score: number;
  live_score: number;
  verdict: string;
  component_scores: Record<string, number>;
  component_weights: Record<string, number>;
  drift_penalty: number;
  perf_penalty: number;
  data_freshness_hours: number | null;
  computed_at: string;
  recommendations: string[];
}

interface CertResult {
  model_id: string;
  cert_hash: string;
  verdict: string;
  overall_score: number;
  live_score: number;
  issued_at: string | null;
  download_url: string;
  message: string;
}

interface GateResult {
  model_id: string;
  passed: boolean;
  score: number;
  verdict: string;
  gate_results: Array<{ metric: string; value: number; threshold: number; verdict: string; message: string }>;
  failures: string[];
  warnings: string[];
  checked_at: string;
}

// ── Primitives ───────────────────────────────────────────────────────────────

const VerdictBadge = ({ verdict }: { verdict: string }) => {
  const styles: Record<string, string> = {
    CERTIFIED: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
    CONDITIONAL: "bg-amber-500/10 text-amber-400 border-amber-500/30",
    FAILED: "bg-red-500/10 text-red-400 border-red-500/30",
  };
  const icons: Record<string, typeof ShieldCheck> = {
    CERTIFIED: ShieldCheck,
    CONDITIONAL: ShieldAlert,
    FAILED: Shield,
  };
  const cls = styles[verdict] || styles.CONDITIONAL;
  const Icon = icons[verdict] || Shield;
  return (
    <span className={`inline-flex items-center gap-1.5 text-[10px] font-black uppercase px-3 py-1 rounded-full border ${cls}`}>
      <Icon className="w-3 h-3" />
      {verdict}
    </span>
  );
};

const ScoreGauge = ({ score, label }: { score: number; label: string }) => {
  const color =
    score >= 80 ? "#10b981" : score >= 60 ? "#f59e0b" : "#ef4444";
  const circumference = 2 * Math.PI * 40;
  const offset = circumference - (score / 100) * circumference;
  return (
    <div className="flex flex-col items-center gap-2">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="40" fill="none" stroke="#1f2937" strokeWidth="8" />
        <circle
          cx="50" cy="50" r="40" fill="none"
          stroke={color} strokeWidth="8"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 50 50)"
          style={{ transition: "stroke-dashoffset 0.8s ease" }}
        />
        <text x="50" y="50" textAnchor="middle" dominantBaseline="middle"
          fill={color} fontSize="18" fontWeight="900">
          {score.toFixed(0)}
        </text>
      </svg>
      <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">{label}</p>
    </div>
  );
};

const ComponentBar = ({ label, score, weight }: { label: string; score: number; weight: number }) => {
  const color =
    score >= 80 ? "bg-emerald-500" : score >= 60 ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-3">
      <p className="text-[10px] font-black uppercase text-slate-500 w-24 shrink-0">
        {label.replace(/_/g, " ")}
      </p>
      <div className="flex-1 bg-white/5 rounded-full h-1.5">
        <div className={`h-full rounded-full ${color} transition-all duration-700`}
          style={{ width: `${score}%` }} />
      </div>
      <p className="text-[10px] font-black text-white w-8 text-right">{score.toFixed(0)}</p>
      <p className="text-[9px] text-slate-600 w-8">{(weight * 100).toFixed(0)}%</p>
    </div>
  );
};

// ── Model Selector ───────────────────────────────────────────────────────────

const ModelSelector = ({
  models, selected, onSelect,
}: { models: Array<{ id: string; name: string }>; selected: string; onSelect: (id: string) => void }) => (
  <div className="flex flex-col gap-1.5">
    <p className="text-[9px] font-black uppercase tracking-widest text-slate-600">Select Model</p>
    <div className="flex flex-wrap gap-2">
      {models.map((m) => (
        <button
          key={m.id}
          onClick={() => onSelect(m.id)}
          className={`px-3 py-1.5 rounded-lg text-[10px] font-black border transition-all ${
            selected === m.id
              ? "bg-orange-500/10 border-orange-500/30 text-orange-400"
              : "bg-white/[0.02] border-white/[0.06] text-slate-500 hover:border-white/20 hover:text-slate-300"
          }`}
        >
          {m.name}
        </button>
      ))}
    </div>
  </div>
);

// ── Main Module ──────────────────────────────────────────────────────────────

export default function GovernanceModule({ state, setState, onAction }: any) {
  const modelId: string = state?.selectedModelId || "";
  const [models, setModels] = useState<Array<{ id: string; name: string }>>([]);
  const [score, setScore] = useState<GovernanceScore | null>(null);
  const [cert, setCert] = useState<CertResult | null>(null);
  const [gateResult, setGateResult] = useState<GateResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [certLoading, setCertLoading] = useState(false);
  const [gateLoading, setGateLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");

  // Fetch available models once
  useEffect(() => {
    apiGet<any>("/api/v1/models")
      .then((d) => {
        const items: Array<{ id: string; name: string }> = Array.isArray(d)
          ? d.map((m: any) => ({ id: m.model_id || m.id, name: m.name }))
          : Array.isArray(d?.items)
          ? d.items.map((m: any) => ({ id: m.model_id || m.id, name: m.name }))
          : [];
        setModels(items);
        // Auto-select first if nothing selected
        if (!state?.selectedModelId && items.length > 0) {
          setState((prev: any) => ({ ...prev, selectedModelId: items[0].id }));
        }
      })
      .catch(() => {});
  }, []);

  const fetchScore = useCallback(async () => {
    if (!modelId) return;
    setLoading(true);
    setError("");
    try {
      const data = await apiGet<GovernanceScore>(`/api/v1/governance/${modelId}/score`);
      setScore(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [modelId]);

  useEffect(() => {
    fetchScore();
    setCert(null);
    setGateResult(null);
  }, [fetchScore]);

  const generateCertificate = async () => {
    if (!modelId) return;
    setCertLoading(true);
    setError("");
    try {
      const data = await apiPost<CertResult>(`/api/v1/governance/${modelId}/certify`, {
        force_regenerate: false,
      });
      setCert(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setCertLoading(false);
    }
  };

  const runGate = async () => {
    if (!modelId) return;
    setGateLoading(true);
    setError("");
    try {
      const data = await apiPost<GateResult>(`/api/v1/governance/${modelId}/gate`, {
        policy_config: {
          min_governance_score: 60,
          max_psi: 0.25,
          min_accuracy: 0.70,
        },
      });
      setGateResult(data);
    } catch (e: any) {
      // 422 gate-fail returns detail object — still valid
      try {
        const parsed = JSON.parse(e.message.split("failed 422: ")[1] || "{}");
        if (parsed.model_id) {
          setGateResult({ ...parsed, passed: false });
          return;
        }
      } catch {}
      setError(e.message);
    } finally {
      setGateLoading(false);
    }
  };

  const copyVerifyUrl = () => {
    if (!cert?.cert_hash) return;
    const url = `${window.location.origin}/verify/${cert.cert_hash}`;
    navigator.clipboard.writeText(url);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const selectModel = (id: string) => {
    setState((prev: any) => ({ ...prev, selectedModelId: id }));
    setScore(null);
    setCert(null);
    setGateResult(null);
    setError("");
  };

  return (
    <div className="space-y-6 pb-10">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-white font-black text-lg tracking-tight">Governance Score</h2>
          <p className="text-slate-500 text-xs mt-0.5">Composite compliance posture · live decay · certification</p>
        </div>
        <button
          onClick={fetchScore}
          disabled={!modelId || loading}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-slate-400 text-xs hover:bg-white/10 transition-colors disabled:opacity-40"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {/* ── Model Selector ──────────────────────────────────────────────── */}
      {models.length > 0 && (
        <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-5">
          <ModelSelector models={models} selected={modelId} onSelect={selectModel} />
        </div>
      )}

      {!modelId && models.length === 0 && (
        <div className="flex items-center justify-center h-40 bg-[#0E1014] border border-white/[0.06] rounded-2xl">
          <p className="text-slate-500 text-sm">No models registered — run a seed or register a model first.</p>
        </div>
      )}

      {!modelId && models.length > 0 && (
        <div className="flex items-center justify-center h-40 bg-[#0E1014] border border-white/[0.06] rounded-2xl">
          <p className="text-slate-500 text-sm">Select a model above to view its governance score.</p>
        </div>
      )}

      {/* ── Error Banner ─────────────────────────────────────────────────── */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-red-400 text-xs flex items-start gap-2">
          <XCircle className="w-4 h-4 shrink-0 mt-0.5" />
          {error}
        </div>
      )}

      {/* ── Loading skeleton ─────────────────────────────────────────────── */}
      {loading && (
        <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-8 flex justify-center">
          <div className="w-8 h-8 border-2 border-orange-400 border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {/* ── Score Gauges ─────────────────────────────────────────────────── */}
      {score && !loading && (
        <>
          <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-6">
            <p className="text-[9px] font-black uppercase tracking-widest text-slate-600 mb-6">Composite Score Overview</p>
            <div className="flex items-center justify-around flex-wrap gap-6">
              <ScoreGauge score={score.overall_score} label="Audit Score" />
              <ScoreGauge score={score.live_score} label="Live Score" />
              <div className="flex flex-col items-center gap-3">
                <VerdictBadge verdict={score.verdict} />
                <div className="text-center space-y-1">
                  <p className="text-[9px] font-black uppercase text-slate-600 tracking-widest">Drift Penalty</p>
                  <p className="text-amber-400 font-black text-sm">
                    -{(score.drift_penalty * 100).toFixed(1)}%
                  </p>
                </div>
                {score.data_freshness_hours != null && (
                  <div className="text-center">
                    <p className="text-[9px] font-black uppercase text-slate-600 tracking-widest">Data Age</p>
                    <p className="text-slate-400 font-black text-xs">{score.data_freshness_hours.toFixed(0)}h</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Component Breakdown */}
          <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-6">
            <p className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-5">Component Breakdown</p>
            <div className="space-y-3.5">
              {Object.entries(score.component_scores).map(([key, val]) => (
                <ComponentBar
                  key={key}
                  label={key}
                  score={val}
                  weight={score.component_weights?.[key] || 0}
                />
              ))}
            </div>
          </div>

          {/* Recommendations */}
          {score.recommendations.length > 0 && (
            <div className="bg-amber-500/5 border border-amber-500/20 rounded-2xl p-5">
              <p className="text-[10px] font-black uppercase tracking-widest text-amber-400 mb-4">
                Governance Recommendations
              </p>
              <ul className="space-y-2.5">
                {score.recommendations.map((r, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-slate-400">
                    <AlertTriangle className="w-3 h-3 text-amber-400 mt-0.5 shrink-0" />
                    {r}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}

      {/* ── Certificate Section ──────────────────────────────────────────── */}
      <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-6">
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-2">
            <Award className="w-4 h-4 text-orange-400" />
            <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">Compliance Certificate</p>
          </div>
          <button
            onClick={generateCertificate}
            disabled={!modelId || certLoading}
            className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-orange-500/10 border border-orange-500/20 text-orange-400 text-xs font-black hover:bg-orange-500/20 transition-colors disabled:opacity-40"
          >
            {certLoading ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Award className="w-3 h-3" />}
            Generate Certificate
          </button>
        </div>

        {cert ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between bg-white/[0.02] rounded-xl p-3 border border-white/[0.04]">
              <div className="min-w-0">
                <p className="text-[9px] text-slate-600 uppercase font-black mb-0.5">Certificate Hash</p>
                <p className="text-xs font-mono text-slate-300 truncate">
                  {cert.cert_hash?.slice(0, 32)}…
                </p>
              </div>
              <div className="flex gap-2 ml-3 shrink-0">
                <button
                  onClick={copyVerifyUrl}
                  title="Copy verify URL"
                  className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                >
                  {copied
                    ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    : <Copy className="w-3.5 h-3.5 text-slate-400" />}
                </button>
                <a
                  href={`/verify/${cert.cert_hash}`}
                  target="_blank"
                  rel="noreferrer"
                  title="Open public verify page"
                  className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <ExternalLink className="w-3.5 h-3.5 text-slate-400" />
                </a>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <VerdictBadge verdict={cert.verdict} />
              <span className="text-xs text-slate-500">Score: {cert.overall_score?.toFixed(1)}/100</span>
              {cert.issued_at && (
                <span className="text-xs text-slate-600">
                  Issued {new Date(cert.issued_at).toLocaleDateString()}
                </span>
              )}
            </div>
            <p className="text-[10px] text-slate-600">{cert.message}</p>
          </div>
        ) : (
          <p className="text-slate-600 text-xs">
            No certificate generated yet. Click &quot;Generate Certificate&quot; to create one.
          </p>
        )}
      </div>

      {/* ── CI/CD Gate ──────────────────────────────────────────────────── */}
      <div className="bg-[#0E1014] border border-white/[0.06] rounded-2xl p-6">
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-blue-400" />
            <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">CI/CD Policy Gate</p>
          </div>
          <button
            onClick={runGate}
            disabled={!modelId || gateLoading}
            className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-black hover:bg-blue-500/20 transition-colors disabled:opacity-40"
          >
            {gateLoading ? <RefreshCw className="w-3 h-3 animate-spin" /> : <GitBranch className="w-3 h-3" />}
            Run Gate Check
          </button>
        </div>

        {gateResult ? (
          <div className="space-y-3">
            <div
              className={`flex items-center gap-3 rounded-xl p-3 ${
                gateResult.passed
                  ? "bg-emerald-500/10 border border-emerald-500/20"
                  : "bg-red-500/10 border border-red-500/20"
              }`}
            >
              {gateResult.passed
                ? <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                : <XCircle className="w-5 h-5 text-red-400" />}
              <div>
                <p className={`text-sm font-black ${gateResult.passed ? "text-emerald-400" : "text-red-400"}`}>
                  Gate {gateResult.passed ? "PASSED" : "FAILED"}
                </p>
                <p className="text-[10px] text-slate-500">
                  Score: {gateResult.score?.toFixed(1)}/100 · {gateResult.gate_results?.length ?? 0} checks run
                </p>
              </div>
            </div>

            {/* Gate check rows */}
            {gateResult.gate_results?.length > 0 && (
              <div className="space-y-1.5 mt-2">
                {gateResult.gate_results.map((g, i) => (
                  <div
                    key={i}
                    className={`flex items-center justify-between text-[10px] px-3 py-2 rounded-lg ${
                      g.verdict === "PASS"
                        ? "bg-emerald-500/5 text-emerald-400"
                        : g.verdict === "WARN"
                        ? "bg-amber-500/5 text-amber-400"
                        : "bg-red-500/5 text-red-400"
                    }`}
                  >
                    <span className="font-black">{g.metric}</span>
                    <span className="font-mono">
                      {g.value >= 0 ? g.value.toFixed(3) : "N/A"} / {g.threshold}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {gateResult.failures?.length > 0 && (
              <div className="space-y-1">
                {gateResult.failures.map((f, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs text-red-400 bg-red-500/5 rounded-lg px-3 py-2">
                    <XCircle className="w-3 h-3 shrink-0" />
                    {f}
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <p className="text-slate-600 text-xs">
            Click &quot;Run Gate Check&quot; to validate against the active governance policy.
          </p>
        )}
      </div>
    </div>
  );
}
