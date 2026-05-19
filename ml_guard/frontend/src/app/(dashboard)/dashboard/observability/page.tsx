'use client';

import { useState, useEffect, useRef } from 'react';
import { RefreshCw, Activity, Cpu, Clock, AlertTriangle, CheckCircle } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000') + '/api/v1';
const HDR = { 'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'dev-secret-key' };

async function apiFetch(path: string) {
  const r = await fetch(`${BASE}${path}`, { headers: HDR });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function StatCard({ label, value, sub, icon: Icon, color = '#1A5F3A' }: any) {
  return (
    <div className="bg-white border border-stone rounded-card p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="text-[11px] text-muted uppercase tracking-[0.05em]">{label}</p>
        <Icon size={14} strokeWidth={1.5} style={{ color }} />
      </div>
      <p className="text-[26px] font-bold text-ink" style={{ letterSpacing: '-0.03em' }}>{value}</p>
      {sub && <p className="text-[11px] text-muted mt-0.5">{sub}</p>}
    </div>
  );
}

export default function ObservabilityPage() {
  const [feed, setFeed] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = async () => {
    try {
      const [feedData, statsData] = await Promise.all([
        apiFetch('/observe/feed?limit=30'),
        apiFetch('/observe/stats'),
      ]);
      setFeed(feedData.events ?? []);
      setStats(statsData);
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      timerRef.current = setInterval(load, 10000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [autoRefresh]);

  const severityColor = (s: string) =>
    s === 'CRITICAL' || s === 'HIGH' ? '#C0392B' : s === 'MEDIUM' ? '#B35A00' : '#1A5F3A';

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Observability</h1>
          <p className="text-[11px] text-muted">Real-time prediction events · latency · anomaly flags</p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-[12px] text-muted cursor-pointer">
            <input type="checkbox" checked={autoRefresh} onChange={e => setAutoRefresh(e.target.checked)} className="accent-forest" />
            Auto-refresh (10s)
          </label>
          <button onClick={load} className="text-muted hover:text-ink transition-colors">
            <RefreshCw size={16} strokeWidth={1.5} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      <div className="flex-1 p-8 space-y-6 overflow-auto">
        {error && <div className="p-3 bg-red-50 border border-red-200 rounded-card text-[12px] text-danger">⚠ {error} — backend may be offline</div>}

        {/* Stats */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Registered Models" value={stats.total_models ?? 0} sub="in registry" icon={Cpu} />
            <StatCard label="Active Alerts" value={stats.alerts_today ?? 0} sub="unresolved" icon={AlertTriangle} color="#C0392B" />
            <StatCard label="Avg Gov. Score" value={stats.avg_governance_score ? `${stats.avg_governance_score}/100` : '—'} sub="across all models" icon={CheckCircle} />
            <StatCard label="Active Contracts" value={stats.active_contracts ?? '—'} sub="behavioral contracts" icon={Activity} color="#0369A1" />
          </div>
        )}

        {/* Live feed */}
        <div className="bg-white border border-stone rounded-card overflow-hidden">
          <div className="flex items-center justify-between px-6 py-4 border-b border-stone">
            <h2 className="text-[14px] font-semibold text-ink">Live event feed</h2>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-forest animate-pulse" />
              <span className="text-[11px] text-muted">{feed.length} events</span>
            </div>
          </div>

          {loading && feed.length === 0 ? (
            <div className="p-8 flex justify-center"><RefreshCw size={20} className="animate-spin text-muted" /></div>
          ) : feed.length === 0 ? (
            <div className="p-8 text-center">
              <Activity size={28} className="mx-auto text-muted mb-2" strokeWidth={1.25} />
              <p className="text-[13px] text-muted">No prediction events yet. Deploy a model and start sending predictions.</p>
            </div>
          ) : (
            <div className="divide-y divide-stone/40">
              {feed.map((ev: any) => (
                <div key={ev.id} className="flex items-start gap-4 px-6 py-3 hover:bg-[#F7F6F2] transition-colors">
                  <div className="w-2 h-2 rounded-full mt-1.5 flex-shrink-0"
                    style={{ background: severityColor(ev.severity) }} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-[12px] font-semibold text-ink">{ev.event_type?.replace(/_/g, ' ')}</span>
                      <span className="text-[11px] text-muted font-mono">{ev.model_id?.slice(0, 8)}…</span>
                      <Badge
                        variant={ev.severity === 'HIGH' || ev.severity === 'CRITICAL' ? 'failed' : ev.severity === 'MEDIUM' ? 'conditional' : 'certified'}
                        className="text-[9px]"
                      >
                        {ev.severity}
                      </Badge>
                    </div>
                    <p className="text-[12px] text-muted truncate">{ev.message}</p>
                  </div>
                  <div className="flex items-center gap-1 text-[10px] text-muted flex-shrink-0">
                    <Clock size={10} />
                    {ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : '—'}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Model health breakdown */}
        {stats?.models_by_verdict && Object.keys(stats.models_by_verdict).length > 0 && (
          <div className="bg-white border border-stone rounded-card p-6">
            <h2 className="text-[14px] font-semibold text-ink mb-4">Models by governance verdict</h2>
            <div className="flex gap-4 flex-wrap">
              {Object.entries(stats.models_by_verdict).map(([verdict, count]: [string, any]) => {
                const color = verdict === 'CERTIFIED' ? '#1A5F3A' : verdict === 'CONDITIONAL' ? '#B35A00' : '#C0392B';
                return (
                  <div key={verdict} className="flex items-center gap-2 px-4 py-2 rounded-[8px]"
                    style={{ background: `${color}12` }}>
                    <span className="text-[20px] font-bold" style={{ color }}>{count}</span>
                    <span className="text-[11px] font-medium" style={{ color }}>{verdict}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
