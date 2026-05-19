'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, BrainCircuit, Send, AlertTriangle, CheckCircle, Clock } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000') + '/api/v1';
const HDR = { 'Content-Type': 'application/json', 'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'dev-secret-key' };

function RiskBar({ label, value, max = 1 }: { label: string; value: number; max?: number }) {
  const pct = Math.min(100, (value / max) * 100);
  const color = pct >= 70 ? '#C0392B' : pct >= 40 ? '#B35A00' : '#1A5F3A';
  return (
    <div>
      <div className="flex justify-between mb-1">
        <span className="text-[11px] text-muted">{label}</span>
        <span className="text-[11px] font-semibold" style={{ color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-1.5 bg-stone rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function LLMEvalPage() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [modelName, setModelName] = useState('gpt-4');
  const [referenceFacts, setReferenceFacts] = useState('');
  const [evaluating, setEvaluating] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [histLoading, setHistLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = async () => {
    setHistLoading(true);
    try {
      const r = await fetch(`${BASE}/llm/history?limit=10`, { headers: HDR });
      if (r.ok) setHistory(await r.json());
    } catch { }
    finally { setHistLoading(false); }
  };

  useEffect(() => { loadHistory(); }, []);

  const evaluate = async () => {
    if (!prompt.trim() || !response.trim()) { setError('Prompt and response are required.'); return; }
    setEvaluating(true); setError(null); setResult(null);
    try {
      const body: any = { prompt, response, model_name: modelName };
      if (referenceFacts.trim()) {
        body.reference_facts = referenceFacts.split('\n').filter(Boolean);
      }
      const r = await fetch(`${BASE}/llm/evaluate`, {
        method: 'POST', headers: HDR, body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? `HTTP ${r.status}`);
      setResult(d);
      loadHistory();
    } catch (e: any) { setError(e.message); }
    finally { setEvaluating(false); }
  };

  const ev = result?.evaluation;
  const riskLevel: string = ev?.llm_risk_level ?? '';
  const riskScore: number = ev?.llm_risk_score ?? 0;

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">LLM Evaluation</h1>
          <p className="text-[11px] text-muted">Prompt safety · toxicity · hallucination · injection</p>
        </div>
        {ev && (
          <Badge variant={riskLevel === 'LOW' ? 'certified' : riskLevel === 'MEDIUM' ? 'conditional' : 'failed'}>
            {riskLevel} RISK
          </Badge>
        )}
      </div>

      <div className="flex-1 p-8 space-y-6 overflow-auto">
        {error && <div className="p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger">⚠ {error}</div>}

        {/* Input form */}
        <div className="bg-white border border-stone rounded-card p-6">
          <h2 className="text-[14px] font-semibold text-ink mb-4">Evaluate prompt / response pair</h2>
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Prompt *</label>
              <textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={5}
                placeholder="Enter the LLM prompt..."
                className="w-full px-3 py-2.5 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest resize-none" />
            </div>
            <div>
              <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Model Response *</label>
              <textarea value={response} onChange={e => setResponse(e.target.value)} rows={5}
                placeholder="Enter the model's response..."
                className="w-full px-3 py-2.5 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest resize-none" />
            </div>
          </div>
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Model name</label>
              <input value={modelName} onChange={e => setModelName(e.target.value)}
                placeholder="e.g. gpt-4, claude-3"
                className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
            </div>
            <div>
              <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Reference facts (one per line, for hallucination check)</label>
              <textarea value={referenceFacts} onChange={e => setReferenceFacts(e.target.value)} rows={2}
                placeholder="Known facts to verify against..."
                className="w-full px-3 py-2 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest resize-none" />
            </div>
          </div>
          <Button variant="primary" size="sm" onClick={evaluate} disabled={evaluating} className="gap-2">
            {evaluating ? <><RefreshCw size={13} className="animate-spin" />Evaluating…</> : <><Send size={13} />Evaluate</>}
          </Button>
        </div>

        {/* Results */}
        {ev && (
          <>
            {/* Risk summary */}
            <div className="grid md:grid-cols-4 gap-4">
              {[
                { label: 'LLM Risk Score', value: `${(riskScore * 100).toFixed(0)}%`, bad: riskScore > 0.5 },
                { label: 'Toxicity', value: `${((ev.toxicity_score ?? 0) * 100).toFixed(0)}%`, bad: ev.toxicity_score > 0.4 },
                { label: 'Hallucination Risk', value: `${((ev.hallucination_risk ?? 0) * 100).toFixed(0)}%`, bad: ev.hallucination_risk > 0.4 },
                { label: 'Stability', value: `${((ev.stability_score ?? 1) * 100).toFixed(0)}%`, bad: ev.stability_score < 0.6 },
              ].map(({ label, value, bad }) => (
                <div key={label} className="bg-white border border-stone rounded-card p-4">
                  <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">{label}</p>
                  <p className={`text-[22px] font-bold ${bad ? 'text-danger' : 'text-forest'}`}>{value}</p>
                </div>
              ))}
            </div>

            {/* Detailed checks */}
            <div className="bg-white border border-stone rounded-card p-6">
              <h2 className="text-[14px] font-semibold text-ink mb-5">Risk breakdown</h2>
              <div className="space-y-4">
                <RiskBar label="Toxicity Score" value={ev.toxicity_score ?? 0} />
                <RiskBar label="Hallucination Risk" value={ev.hallucination_risk ?? 0} />
                <RiskBar label="LLM Risk Score" value={ev.llm_risk_score ?? 0} />
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-[11px] text-muted">Stability Score (higher = better)</span>
                    <span className={`text-[11px] font-semibold ${(ev.stability_score ?? 1) >= 0.7 ? 'text-forest' : 'text-danger'}`}>
                      {((ev.stability_score ?? 1) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-stone rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all bg-forest"
                      style={{ width: `${Math.min(100, (ev.stability_score ?? 1) * 100)}%` }} />
                  </div>
                </div>
              </div>

              {/* Flags */}
              <div className="mt-5 space-y-2">
                {[
                  { label: 'Prompt injection detected', flag: ev.prompt_injection_flag, bad: true },
                  { label: 'Refusal detected', flag: ev.refusal_detected, bad: false },
                ].map(({ label, flag, bad }) => (
                  flag != null && (
                    <div key={label} className="flex items-center gap-2">
                      {flag
                        ? bad ? <AlertTriangle size={13} className="text-danger" /> : <CheckCircle size={13} className="text-forest" />
                        : <CheckCircle size={13} className="text-forest" />}
                      <span className={`text-[12px] ${flag && bad ? 'text-danger font-semibold' : 'text-muted'}`}>
                        {label}: {flag ? 'Yes' : 'No'}
                      </span>
                    </div>
                  )
                ))}
              </div>
            </div>

            {/* Policy result */}
            {result?.policy && (
              <div className="bg-white border border-stone rounded-card p-6">
                <h2 className="text-[14px] font-semibold text-ink mb-3">Policy verdict</h2>
                <div className="grid md:grid-cols-3 gap-4 text-[12px]">
                  {Object.entries(result.policy).slice(0, 6).map(([k, v]) => (
                    <div key={k}>
                      <p className="text-muted capitalize">{k.replace(/_/g, ' ')}</p>
                      <p className="font-semibold text-ink">{String(v ?? '—')}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* History */}
        <div className="bg-white border border-stone rounded-card overflow-hidden">
          <div className="flex items-center justify-between px-6 py-4 border-b border-stone">
            <h2 className="text-[14px] font-semibold text-ink">Recent evaluations</h2>
            <button onClick={loadHistory} className="text-muted hover:text-ink transition-colors">
              <RefreshCw size={13} strokeWidth={1.5} className={histLoading ? 'animate-spin' : ''} />
            </button>
          </div>
          {histLoading ? (
            <div className="p-6 flex justify-center"><RefreshCw size={16} className="animate-spin text-muted" /></div>
          ) : history.length === 0 ? (
            <div className="p-6 text-center text-[13px] text-muted">No evaluations yet.</div>
          ) : (
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-stone">
                  {['Risk Score', 'Risk Level', 'Toxicity', 'Injection', 'Hallucination', 'Time'].map(h => (
                    <th key={h} className="text-left px-5 py-2.5 text-[10px] font-semibold uppercase tracking-[0.04em] text-muted">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.map((h: any) => (
                  <tr key={h.id} className="border-b border-stone/40 hover:bg-[#F7F6F2] transition-colors">
                    <td className="px-5 py-2.5 text-[13px] font-semibold"
                      style={{ color: (h.llm_risk_score ?? 0) > 0.5 ? '#C0392B' : '#1A5F3A' }}>
                      {((h.llm_risk_score ?? 0) * 100).toFixed(0)}%
                    </td>
                    <td className="px-5 py-2.5">
                      <Badge variant={h.llm_risk_level === 'LOW' ? 'certified' : h.llm_risk_level === 'MEDIUM' ? 'conditional' : 'failed'}>
                        {h.llm_risk_level ?? '—'}
                      </Badge>
                    </td>
                    <td className="px-5 py-2.5 text-[12px] text-muted">{((h.toxicity_score ?? 0) * 100).toFixed(0)}%</td>
                    <td className="px-5 py-2.5 text-[12px]">
                      {h.injection_flag ? <span className="text-danger font-semibold">⚠ Yes</span> : <span className="text-forest">No</span>}
                    </td>
                    <td className="px-5 py-2.5 text-[12px] text-muted">{((h.hallucination_risk ?? 0) * 100).toFixed(0)}%</td>
                    <td className="px-5 py-2.5 text-[11px] text-muted flex items-center gap-1">
                      <Clock size={10} /> {h.created_at ? new Date(h.created_at).toLocaleString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
