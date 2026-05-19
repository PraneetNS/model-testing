'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, Rocket, CheckCircle, XCircle, RotateCcw, Layers } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000') + '/api/v1';
const HDR = { 'Content-Type': 'application/json', 'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'dev-secret-key' };

async function apiFetch(path: string, opts: RequestInit = {}) {
  const r = await fetch(`${BASE}${path}`, { ...opts, headers: { ...HDR, ...(opts.headers ?? {}) } });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail ?? `HTTP ${r.status}`);
  return d;
}

const ENV_COLORS: Record<string, string> = {
  DEV: '#0369A1', STAGING: '#B35A00', PRODUCTION: '#1A5F3A',
};
const STATUS_VARIANT: Record<string, 'certified' | 'conditional' | 'failed'> = {
  ACTIVE: 'certified', ROLLED_BACK: 'failed', PENDING: 'conditional',
};

export default function DeploymentsPage() {
  const [deployments, setDeployments] = useState<any[]>([]);
  const [environments, setEnvironments] = useState<any[]>([]);
  const [envFilter, setEnvFilter] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rolling, setRolling] = useState<string | null>(null);

  const load = async () => {
    setLoading(true); setError(null);
    try {
      const qs = envFilter ? `?environment=${envFilter}` : '';
      const [deps, envs] = await Promise.all([
        apiFetch(`/deployments${qs}`),
        apiFetch('/deployments/environments'),
      ]);
      setDeployments(deps.items ?? []);
      setEnvironments(Array.isArray(envs) ? envs : []);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, [envFilter]);

  const rollback = async (deploymentId: string) => {
    if (!confirm('Roll back this deployment?')) return;
    setRolling(deploymentId);
    try {
      await apiFetch(`/deployments/rollback?deployment_id=${deploymentId}`, { method: 'POST', body: '{}' });
      await load();
    } catch (e: any) { setError(e.message); }
    finally { setRolling(null); }
  };

  // Env stats
  const envStats = environments.map(e => ({
    name: e.name,
    count: deployments.filter(d => d.environment === e.name && d.status === 'ACTIVE').length,
  }));

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Deployments</h1>
          <p className="text-[11px] text-muted">Environment tracking · promotion · rollback</p>
        </div>
        <button onClick={load} className="text-muted hover:text-ink transition-colors">
          <RefreshCw size={16} strokeWidth={1.5} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      <div className="flex-1 p-8 space-y-6 overflow-auto">
        {error && <div className="p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger">⚠ {error}</div>}

        {/* Environment cards */}
        {envStats.length > 0 && (
          <div className="grid grid-cols-3 gap-4">
            {envStats.map(e => (
              <button key={e.name}
                onClick={() => setEnvFilter(envFilter === e.name ? '' : e.name)}
                className={`bg-white border rounded-card p-5 text-left transition-all ${envFilter === e.name ? 'border-forest ring-2 ring-forest/20' : 'border-stone hover:border-forest/50'}`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="w-8 h-8 rounded-[8px] flex items-center justify-center"
                    style={{ background: `${ENV_COLORS[e.name] ?? '#888'}18` }}>
                    <Layers size={14} style={{ color: ENV_COLORS[e.name] ?? '#888' }} />
                  </div>
                  <span className="text-[24px] font-bold text-ink" style={{ letterSpacing: '-0.03em' }}>{e.count}</span>
                </div>
                <p className="text-[13px] font-semibold text-ink">{e.name}</p>
                <p className="text-[11px] text-muted">active deployments</p>
              </button>
            ))}
          </div>
        )}

        {/* Filter bar */}
        <div className="flex gap-2">
          <button onClick={() => setEnvFilter('')}
            className={`px-3 py-1 text-[12px] font-medium rounded-badge border transition-colors ${!envFilter ? 'bg-forest text-white border-forest' : 'bg-white text-muted border-stone hover:border-forest'}`}>
            All
          </button>
          {['DEV', 'STAGING', 'PRODUCTION'].map(e => (
            <button key={e} onClick={() => setEnvFilter(envFilter === e ? '' : e)}
              className={`px-3 py-1 text-[12px] font-medium rounded-badge border transition-colors ${envFilter === e ? 'text-white border-transparent' : 'bg-white text-muted border-stone hover:border-forest'}`}
              style={envFilter === e ? { background: ENV_COLORS[e] } : {}}>
              {e}
            </button>
          ))}
        </div>

        {/* Deployments table */}
        {loading ? (
          <div className="bg-white border border-stone rounded-card p-8 flex justify-center">
            <RefreshCw size={20} className="animate-spin text-muted" />
          </div>
        ) : deployments.length === 0 ? (
          <div className="bg-white border border-stone rounded-card p-10 text-center">
            <Rocket size={28} className="mx-auto text-muted mb-2" strokeWidth={1.25} />
            <p className="text-[13px] text-muted">No deployments found. Promote a model version to create one.</p>
          </div>
        ) : (
          <div className="bg-white border border-stone rounded-card overflow-hidden">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-stone">
                  {['Model', 'Version', 'Environment', 'Gov. Score', 'Status', 'Deployed', ''].map(h => (
                    <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {deployments.map((d: any) => (
                  <tr key={d.deployment_id} className="border-b border-stone/50 hover:bg-[#F7F6F2] transition-colors">
                    <td className="px-5 py-3 text-[13px] font-medium text-ink">{d.model_name}</td>
                    <td className="px-5 py-3 text-[12px] font-mono text-muted">v{d.version_number ?? '—'}</td>
                    <td className="px-5 py-3">
                      <span className="text-[11px] font-bold px-2 py-0.5 rounded-badge text-white"
                        style={{ background: ENV_COLORS[d.environment] ?? '#888' }}>
                        {d.environment}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-[13px] font-semibold"
                      style={{ color: d.governance_score >= 75 ? '#1A5F3A' : d.governance_score >= 50 ? '#B35A00' : '#C0392B' }}>
                      {d.governance_score != null ? `${d.governance_score?.toFixed(0)}/100` : '—'}
                    </td>
                    <td className="px-5 py-3">
                      <Badge variant={STATUS_VARIANT[d.status] ?? 'conditional'}>{d.status}</Badge>
                    </td>
                    <td className="px-5 py-3 text-[12px] text-muted">
                      {d.deployed_at ? new Date(d.deployed_at).toLocaleDateString() : '—'}
                    </td>
                    <td className="px-5 py-3">
                      {d.status === 'ACTIVE' && (
                        <button onClick={() => rollback(d.deployment_id)}
                          disabled={rolling === d.deployment_id}
                          className="flex items-center gap-1 text-[11px] text-muted hover:text-danger transition-colors disabled:opacity-50">
                          <RotateCcw size={11} strokeWidth={1.5} />
                          {rolling === d.deployment_id ? 'Rolling…' : 'Rollback'}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
