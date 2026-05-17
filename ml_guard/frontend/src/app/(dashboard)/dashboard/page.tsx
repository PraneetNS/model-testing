'use client';

import { useEffect, useState } from 'react';
import { Bell, TrendingUp, TrendingDown, Minus, RefreshCw } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { observeApi, modelsApi, alertsApi, type ModelItem, type AlertEvent } from '@/lib/api';
import { useAuth } from '@/context/AuthContext';
import Link from 'next/link';

function StatCard({ label, value, sub, trend }: { label: string; value: string | number; sub?: string; trend?: 'up' | 'down' | 'neutral' }) {
  return (
    <div className="bg-white border border-stone rounded-card p-5">
      <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">{label}</p>
      <p className="text-[28px] font-bold text-ink leading-none mb-1" style={{ letterSpacing: '-0.03em' }}>{value}</p>
      {sub && (
        <div className="flex items-center gap-1 mt-1">
          {trend === 'up' && <TrendingUp size={11} className="text-forest" />}
          {trend === 'down' && <TrendingDown size={11} className="text-danger" />}
          {trend === 'neutral' && <Minus size={11} className="text-muted" />}
          <span className="text-[11px] text-muted">{sub}</span>
        </div>
      )}
    </div>
  );
}

function ScoreBadge({ score }: { score: number | null }) {
  if (score === null) return <span className="text-[12px] text-muted">—</span>;
  const v: 'certified' | 'conditional' | 'failed' = score >= 80 ? 'certified' : score >= 60 ? 'conditional' : 'failed';
  return <Badge variant={v}>{score.toFixed(0)}</Badge>;
}

function SkeletonRow() {
  return (
    <tr className="border-b border-stone/50">
      {[1,2,3,4].map(i => (
        <td key={i} className="py-3 pr-4">
          <div className="h-3 bg-stone rounded-full animate-pulse" style={{ width: `${60 + i*10}%` }} />
        </td>
      ))}
    </tr>
  );
}

export default function DashboardOverviewPage() {
  const { user } = useAuth();
  const [models, setModels] = useState<ModelItem[]>([]);
  const [alerts, setAlerts] = useState<AlertEvent[]>([]);
  const [stats, setStats] = useState<{ total_models: number; active_contracts: number; alerts_today: number; avg_governance_score: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  async function loadData() {
    try {
      setError(null);
      const [modelsRes, alertsRes] = await Promise.all([
        modelsApi.list(1, 10),
        alertsApi.listEvents(5, false),
      ]);
      setModels(modelsRes.items ?? []);
      setAlerts(alertsRes.items ?? []);

      // Compute stats from real data
      const scores = (modelsRes.items ?? [])
        .map(m => m.latest_governance_score)
        .filter((s): s is number => s !== null);
      setStats({
        total_models: modelsRes.total ?? 0,
        active_contracts: 0, // placeholder — contracts endpoint
        alerts_today: alertsRes.total ?? 0,
        avg_governance_score: scores.length ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length) : 0,
      });
    } catch (e: any) {
      setError(e.message ?? 'Failed to load dashboard data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => { loadData(); }, []);

  const refresh = async () => { setRefreshing(true); await loadData(); };

  const initials = user?.displayName
    ? user.displayName.split(' ').map((n: string) => n[0]).slice(0, 2).join('').toUpperCase()
    : 'OP';

  return (
    <div className="flex flex-col min-h-screen">
      {/* Top bar */}
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Overview</h1>
          <p className="text-[11px] text-muted">
            {user?.displayName ? `Welcome back, ${user.displayName.split(' ')[0]}` : 'Dashboard / Overview'}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={refresh}
            disabled={refreshing}
            className="text-muted hover:text-ink transition-colors duration-150 disabled:opacity-50"
            aria-label="Refresh"
          >
            <RefreshCw size={16} strokeWidth={1.5} className={refreshing ? 'animate-spin' : ''} />
          </button>
          <Link href="/dashboard/alerts" className="relative text-muted hover:text-ink transition-colors duration-150" aria-label="Alerts">
            <Bell size={18} strokeWidth={1.5} />
            {(stats?.alerts_today ?? 0) > 0 && (
              <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-danger rounded-full" />
            )}
          </Link>
          <div className="w-8 h-8 rounded-full bg-forest flex items-center justify-center text-white text-[11px] font-semibold">
            {user?.photoURL
              ? <img src={user.photoURL} className="w-8 h-8 rounded-full object-cover" alt="avatar" />
              : initials}
          </div>
        </div>
      </div>

      <div className="flex-1 p-8 overflow-auto">
        {/* Error banner */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger flex items-center justify-between">
            <span>⚠ {error} — showing cached data or backend may be offline.</span>
            <button onClick={refresh} className="text-forest underline text-[12px]">Retry</button>
          </div>
        )}

        {/* Stat cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {loading ? (
            Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="bg-white border border-stone rounded-card p-5 animate-pulse">
                <div className="h-3 bg-stone rounded-full w-1/2 mb-3" />
                <div className="h-7 bg-stone rounded-full w-3/4" />
              </div>
            ))
          ) : (
            <>
              <StatCard label="Registered Models" value={stats?.total_models ?? 0} sub="in registry" trend="neutral" />
              <StatCard label="Active Alerts" value={stats?.alerts_today ?? 0} sub="unresolved" trend={(stats?.alerts_today ?? 0) > 0 ? 'down' : 'neutral'} />
              <StatCard label="Avg Gov. Score" value={stats?.avg_governance_score ? `${stats.avg_governance_score}/100` : '—'} sub="across all models" trend={((stats?.avg_governance_score ?? 0) >= 80) ? 'up' : 'down'} />
              <StatCard label="Active Contracts" value={stats?.active_contracts ?? '—'} sub="behavioral contracts" trend="neutral" />
            </>
          )}
        </div>

        {/* 2-col */}
        <div className="grid lg:grid-cols-[58%_42%] gap-6 mb-6">
          {/* Models table */}
          <div className="bg-white border border-stone rounded-card p-6">
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-[14px] font-semibold text-ink">Recent model activity</h2>
              <Link href="/dashboard/models" className="text-[12px] text-forest hover:underline">View all →</Link>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr>
                    {['Model', 'Version', 'Score', 'Risk', 'Registered'].map(h => (
                      <th key={h} className="text-left pb-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted border-b border-stone">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {loading
                    ? Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)
                    : models.length === 0
                    ? (
                      <tr><td colSpan={5} className="py-8 text-center text-[13px] text-muted">No models registered yet. <Link href="/dashboard/models" className="text-forest underline">Register one →</Link></td></tr>
                    )
                    : models.slice(0, 8).map(m => (
                      <tr key={m.model_id} className="border-b border-stone/50 hover:bg-[#F0EDE8] transition-colors duration-100">
                        <td className="py-3 pr-4">
                          <Link href={`/dashboard/models/${m.model_id}`} className="font-medium text-ink hover:text-forest transition-colors text-[13px]">
                            {m.name}
                          </Link>
                        </td>
                        <td className="py-3 pr-4 text-[12px] font-mono text-muted">v{m.latest_version}</td>
                        <td className="py-3 pr-4"><ScoreBadge score={m.latest_governance_score} /></td>
                        <td className="py-3 pr-4">
                          {m.latest_risk_class
                            ? <span className="text-[11px] font-medium px-2 py-0.5 rounded-badge bg-stone text-ink-soft">{m.latest_risk_class}</span>
                            : <span className="text-[12px] text-muted">—</span>}
                        </td>
                        <td className="py-3 text-[12px] text-muted">{new Date(m.created_at).toLocaleDateString()}</td>
                      </tr>
                    ))
                  }
                </tbody>
              </table>
            </div>
          </div>

          {/* Active alerts */}
          <div className="bg-white border border-stone rounded-card p-6">
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-[14px] font-semibold text-ink">Active alerts</h2>
              <Link href="/dashboard/alerts" className="text-[12px] text-forest hover:underline">View all →</Link>
            </div>
            {loading ? (
              Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="flex gap-3 pb-3 mb-3 border-b border-stone/50 last:border-0 animate-pulse">
                  <div className="h-5 w-16 bg-stone rounded-badge" />
                  <div className="flex-1">
                    <div className="h-3 bg-stone rounded-full w-3/4 mb-1.5" />
                    <div className="h-2.5 bg-stone rounded-full w-1/2" />
                  </div>
                </div>
              ))
            ) : alerts.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-[13px] text-muted">No active alerts. System healthy ✓</p>
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                {alerts.map(alert => (
                  <div key={alert.id} className="flex items-start gap-3 pb-3 border-b border-stone/50 last:border-0">
                    <Badge
                      variant={alert.severity === 'HIGH' || alert.severity === 'CRITICAL' ? 'failed' : alert.severity === 'MEDIUM' ? 'conditional' : 'certified'}
                      className="mt-0.5 flex-shrink-0 text-[9px]"
                    >
                      {alert.severity}
                    </Badge>
                    <div className="flex-1 min-w-0">
                      <p className="text-[13px] font-medium text-ink truncate">{alert.message}</p>
                      <p className="text-[11px] text-muted">{new Date(alert.created_at).toLocaleString()}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Governance score sparkline */}
        <div className="bg-white border border-stone rounded-card p-6">
          <h2 className="text-[14px] font-semibold text-ink mb-5">Model governance score distribution</h2>
          {loading ? (
            <div className="h-[120px] bg-stone rounded-card animate-pulse" />
          ) : models.length === 0 ? (
            <p className="text-[13px] text-muted text-center py-6">Register and audit models to see score distribution.</p>
          ) : (
            <div className="flex items-end gap-2 h-[120px]">
              {models.map(m => {
                const score = m.latest_governance_score ?? 0;
                const color = score >= 80 ? '#1A5F3A' : score >= 60 ? '#B35A00' : '#C0392B';
                return (
                  <div key={m.model_id} className="flex-1 flex flex-col items-center justify-end gap-1 group relative">
                    <div
                      className="w-full rounded-t-sm transition-all duration-300 cursor-pointer"
                      style={{ height: `${Math.max(8, (score / 100) * 100)}px`, background: color, opacity: 0.85 }}
                    />
                    <span className="text-[9px] text-muted truncate w-full text-center">{m.name.split('-')[0]}</span>
                    {/* Tooltip */}
                    <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 bg-ink text-white text-[10px] px-2 py-1 rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      {m.name}: {score === 0 ? 'Not audited' : `${score}/100`}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
