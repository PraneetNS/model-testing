'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, ShieldCheck, TrendingDown, AlertTriangle, CheckCircle } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { modelsApi } from '@/lib/api';

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000') + '/api/v1';
const HDR = { 'Content-Type': 'application/json', 'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'dev-secret-key' };

async function fetchScore(modelId: string) {
  const r = await fetch(`${BASE}/governance/${modelId}/score`, { headers: HDR });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function fetchLiveScore(modelId: string) {
  const r = await fetch(`${BASE}/governance/${modelId}/score/live`, { headers: HDR });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function fetchTrend() {
  const r = await fetch(`${BASE}/governance/trend?days=30`, { headers: HDR });
  if (!r.ok) return { trend: [] };
  return r.json();
}

function ScoreRing({ score, size = 120 }: { score: number; size?: number }) {
  const r = size / 2 - 10;
  const circ = 2 * Math.PI * r;
  const offset = circ - (score / 100) * circ;
  const color = score >= 75 ? '#1A5F3A' : score >= 50 ? '#B35A00' : '#C0392B';
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#E8E4DE" strokeWidth="9" />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="9"
        strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ transition: 'stroke-dashoffset 0.9s ease' }} />
      <text x={size / 2} y={size / 2 - 5} textAnchor="middle" fontSize="20" fontWeight="700" fill={color}>{score.toFixed(0)}</text>
      <text x={size / 2} y={size / 2 + 12} textAnchor="middle" fontSize="9" fill="#888">/100</text>
    </svg>
  );
}

function MiniSparkline({ data }: { data: { date: string; avg_score: number }[] }) {
  if (!data.length) return <div className="h-14 bg-stone/30 rounded animate-pulse" />;
  const max = Math.max(...data.map(d => d.avg_score));
  const min = Math.min(...data.map(d => d.avg_score));
  const H = 56, W = 300;
  const pts = data.map((d, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H - ((d.avg_score - min) / (max - min + 1)) * H;
    return `${x},${y}`;
  }).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-14" preserveAspectRatio="none">
      <polyline fill="none" stroke="#1A5F3A" strokeWidth="2" points={pts} />
    </svg>
  );
}

export default function GovernanceScorePage() {
  const [models, setModels] = useState<any[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [score, setScore] = useState<any>(null);
  const [live, setLive] = useState<any>(null);
  const [trend, setTrend] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [certifying, setCertifying] = useState(false);
  const [certResult, setCertResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    modelsApi.list(1, 50).then(r => {
      setModels(r.items ?? []);
      if (r.items?.length) setSelectedId(r.items[0].model_id);
    }).catch(() => {});
    fetchTrend().then(r => setTrend(r.trend ?? []));
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    setLoading(true); setError(null); setScore(null); setLive(null); setCertResult(null);
    Promise.all([fetchScore(selectedId), fetchLiveScore(selectedId)])
      .then(([s, l]) => { setScore(s); setLive(l); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [selectedId]);

  const certify = async () => {
    if (!selectedId) return;
    setCertifying(true); setCertResult(null);
    try {
      const r = await fetch(`${BASE}/governance/${selectedId}/certify`, { method: 'POST', headers: HDR, body: '{}' });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? `HTTP ${r.status}`);
      setCertResult(d);
    } catch (e: any) { setError(e.message); }
    finally { setCertifying(false); }
  };

  const govScore = score?.overall_score ?? live?.base_audit_score ?? 0;
  const liveScore = live?.live_score ?? govScore;
  const verdict = govScore >= 75 ? 'CERTIFIED' : govScore >= 50 ? 'CONDITIONAL' : 'FAILED';

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Governance Score</h1>
          <p className="text-[11px] text-muted">Live score · cert · gate</p>
        </div>
        {govScore > 0 && (
          <Badge variant={verdict === 'CERTIFIED' ? 'certified' : verdict === 'CONDITIONAL' ? 'conditional' : 'failed'}>
            {verdict}
          </Badge>
        )}
      </div>

      <div className="flex-1 p-8 space-y-6 overflow-auto">
        {/* Model selector */}
        <div className="bg-white border border-stone rounded-card p-5">
          <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Select model</label>
          <select
            value={selectedId}
            onChange={e => setSelectedId(e.target.value)}
            className="w-full max-w-xs h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest bg-white"
          >
            {models.length === 0 && <option value="">No models registered</option>}
            {models.map(m => <option key={m.model_id} value={m.model_id}>{m.name} (v{m.latest_version})</option>)}
          </select>
        </div>

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger">⚠ {error}</div>
        )}

        {loading && (
          <div className="bg-white border border-stone rounded-card p-8 flex justify-center">
            <RefreshCw size={20} className="animate-spin text-muted" />
          </div>
        )}

        {!loading && (score || live) && (
          <>
            {/* Score cards */}
            <div className="grid md:grid-cols-3 gap-5">
              {/* Audit score */}
              <div className="bg-white border border-stone rounded-card p-6 flex flex-col items-center">
                <ScoreRing score={Math.round(govScore)} />
                <p className="text-[13px] font-semibold text-ink mt-2">Audit Score</p>
                <p className="text-[11px] text-muted">From last audit run</p>
              </div>
              {/* Live score */}
              <div className="bg-white border border-stone rounded-card p-6 flex flex-col items-center">
                <ScoreRing score={Math.round(liveScore)} />
                <p className="text-[13px] font-semibold text-ink mt-2">Live Score</p>
                <p className="text-[11px] text-muted">With drift & perf decay</p>
              </div>
              {/* Penalties */}
              <div className="bg-white border border-stone rounded-card p-6 space-y-4">
                <p className="text-[13px] font-semibold text-ink">Score Factors</p>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-[12px] text-muted">
                      <TrendingDown size={13} className="text-danger" /> Drift penalty
                    </div>
                    <span className="text-[13px] font-semibold text-danger">
                      -{((live?.drift_penalty ?? 0) * 100).toFixed(1)}pts
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-[12px] text-muted">
                      <TrendingDown size={13} className="text-warning" /> Perf penalty
                    </div>
                    <span className="text-[13px] font-semibold text-warning">
                      -{((live?.perf_penalty ?? 0) * 100).toFixed(1)}pts
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-[12px] text-muted">
                      <AlertTriangle size={13} className={live?.drift_detected ? 'text-danger' : 'text-forest'} />
                      Drift detected
                    </div>
                    <span className={`text-[12px] font-semibold ${live?.drift_detected ? 'text-danger' : 'text-forest'}`}>
                      {live?.drift_detected == null ? '—' : live.drift_detected ? 'Yes' : 'No'}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Component scores */}
            {score?.component_scores && Object.keys(score.component_scores).length > 0 && (
              <div className="bg-white border border-stone rounded-card p-6">
                <h2 className="text-[14px] font-semibold text-ink mb-4">Component Scores</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(score.component_scores).map(([key, val]: [string, any]) => {
                    const v = Math.round(val);
                    const color = v >= 75 ? '#1A5F3A' : v >= 50 ? '#B35A00' : '#C0392B';
                    return (
                      <div key={key} className="bg-[#F7F6F2] rounded-[8px] p-4">
                        <p className="text-[10px] text-muted uppercase tracking-[0.05em] mb-2 capitalize">{key.replace(/_/g, ' ')}</p>
                        <div className="flex items-end gap-1">
                          <span className="text-[22px] font-bold" style={{ color }}>{v}</span>
                          <span className="text-[11px] text-muted mb-1">/100</span>
                        </div>
                        <div className="mt-2 h-1.5 bg-stone rounded-full overflow-hidden">
                          <div className="h-full rounded-full transition-all" style={{ width: `${v}%`, background: color }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {score?.recommendations?.length > 0 && (
              <div className="bg-white border border-stone rounded-card p-6">
                <h2 className="text-[14px] font-semibold text-ink mb-4">Recommendations</h2>
                <div className="space-y-2">
                  {score.recommendations.map((rec: string, i: number) => (
                    <div key={i} className="flex items-start gap-2 text-[13px]">
                      <CheckCircle size={14} className="text-forest flex-shrink-0 mt-0.5" />
                      <span className="text-ink-soft">{rec}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Trend sparkline */}
            {trend.length > 0 && (
              <div className="bg-white border border-stone rounded-card p-6">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-[14px] font-semibold text-ink">30-day score trend (all models)</h2>
                  <span className="text-[11px] text-muted">{trend.length} data points</span>
                </div>
                <MiniSparkline data={trend} />
                <div className="flex justify-between mt-1">
                  <span className="text-[10px] text-muted">{trend[0]?.date}</span>
                  <span className="text-[10px] text-muted">{trend[trend.length - 1]?.date}</span>
                </div>
              </div>
            )}

            {/* Certify */}
            <div className="bg-white border border-stone rounded-card p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-[14px] font-semibold text-ink">Issue Compliance Certificate</h2>
                  <p className="text-[12px] text-muted mt-0.5">Generates a cryptographically-signed report card. Share the verification URL with auditors.</p>
                </div>
                <Button variant="primary" size="sm" onClick={certify} disabled={certifying || !selectedId} className="gap-2">
                  {certifying ? <><RefreshCw size={13} className="animate-spin" />Generating…</> : <><ShieldCheck size={13} />Certify Model</>}
                </Button>
              </div>
              {certResult && (
                <div className="mt-4 p-4 bg-mist border border-forest/30 rounded-[8px] space-y-2">
                  <div className="flex items-center gap-2 text-forest font-medium text-[13px]">
                    <CheckCircle size={14} /> Certificate issued — {certResult.verdict}
                  </div>
                  <p className="text-[12px] text-muted font-mono break-all">{certResult.cert_hash}</p>
                  <p className="text-[12px] text-muted">
                    Share: <span className="text-forest font-medium">{certResult.download_url}</span>
                  </p>
                </div>
              )}
            </div>
          </>
        )}

        {!loading && !score && !live && !error && models.length > 0 && (
          <div className="bg-white border border-stone rounded-card p-10 text-center text-[13px] text-muted">
            Select a model above to view its governance score.
          </div>
        )}

        {!loading && models.length === 0 && (
          <div className="bg-white border border-stone rounded-card p-10 text-center">
            <p className="text-[13px] text-muted">No models registered yet. Run a Model Audit first to generate governance scores.</p>
          </div>
        )}
      </div>
    </div>
  );
}
