'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, GitBranch, CheckCircle, XCircle, Plus, Zap } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { modelsApi } from '@/lib/api';

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000') + '/api/v1';
const HDR = { 'Content-Type': 'application/json', 'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'dev-secret-key' };

async function apiFetch(path: string, opts: RequestInit = {}) {
  const r = await fetch(`${BASE}${path}`, { ...opts, headers: { ...HDR, ...(opts.headers ?? {}) } });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail ?? `HTTP ${r.status}`);
  return d;
}

export default function CICDPage() {
  const [integrations, setIntegrations] = useState<any[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [gateModel, setGateModel] = useState('');
  const [gating, setGating] = useState(false);
  const [gateResult, setGateResult] = useState<any>(null);
  const [showAdd, setShowAdd] = useState(false);
  const [newInt, setNewInt] = useState({ provider: 'github', repo_url: '', webhook_secret: '' });
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const [ints, mods] = await Promise.all([
        apiFetch('/ci/integrations'),
        modelsApi.list(1, 50),
      ]);
      setIntegrations(Array.isArray(ints) ? ints : []);
      setModels(mods.items ?? []);
      if (mods.items?.length) setGateModel(mods.items[0].name);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const runGate = async () => {
    if (!gateModel) return;
    setGating(true); setGateResult(null);
    try {
      const d = await apiFetch(`/ci/audit?model_name=${encodeURIComponent(gateModel)}`, { method: 'POST', body: '{}' });
      setGateResult(d);
    } catch (e: any) { setError(e.message); }
    finally { setGating(false); }
  };

  const addIntegration = async () => {
    if (!newInt.repo_url) return;
    setAdding(true);
    try {
      await apiFetch(`/ci/integrations?provider=${newInt.provider}&repo_url=${encodeURIComponent(newInt.repo_url)}&webhook_secret=${newInt.webhook_secret}`, { method: 'POST', body: '{}' });
      setShowAdd(false);
      await load();
    } catch (e: any) { setError(e.message); }
    finally { setAdding(false); }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">CI/CD</h1>
          <p className="text-[11px] text-muted">GitHub · GitLab · Jenkins governance gates</p>
        </div>
        <Button variant="primary" size="sm" onClick={() => setShowAdd(s => !s)} className="gap-2">
          <Plus size={13} /> Add Integration
        </Button>
      </div>

      <div className="flex-1 p-8 space-y-6 overflow-auto">
        {error && <div className="p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger">⚠ {error}</div>}

        {/* Gate check */}
        <div className="bg-white border border-stone rounded-card p-6">
          <h2 className="text-[14px] font-semibold text-ink mb-4">Run governance gate check</h2>
          <p className="text-[12px] text-muted mb-4">Simulate a CI/CD gate — checks if a model passes the governance threshold for deployment approval.</p>
          <div className="flex gap-3 items-end flex-wrap">
            <div>
              <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Model name</label>
              <input value={gateModel} onChange={e => setGateModel(e.target.value)} list="model-names"
                placeholder="e.g. credit-risk-v4"
                className="h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest w-64" />
              <datalist id="model-names">
                {models.map(m => <option key={m.model_id} value={m.name} />)}
              </datalist>
            </div>
            <Button variant="primary" size="sm" onClick={runGate} disabled={gating || !gateModel} className="gap-2">
              {gating ? <><RefreshCw size={13} className="animate-spin" />Checking…</> : <><Zap size={13} />Run Gate Check</>}
            </Button>
          </div>

          {gateResult && (
            <div className={`mt-4 p-4 border rounded-[8px] ${gateResult.deployment_allowed ? 'bg-mist border-forest/30' : 'bg-red-50 border-red-200'}`}>
              <div className="flex items-center gap-3 mb-2">
                {gateResult.deployment_allowed
                  ? <CheckCircle size={16} className="text-forest" />
                  : <XCircle size={16} className="text-danger" />}
                <span className="text-[14px] font-semibold text-ink">{gateResult.message}</span>
              </div>
              <div className="grid grid-cols-3 gap-4 text-[12px]">
                <div><p className="text-muted">Governance Score</p><p className="font-bold text-ink text-[16px]">{gateResult.governance_score?.toFixed(0)}/100</p></div>
                <div><p className="text-muted">Risk Level</p><p className="font-semibold text-ink">{gateResult.risk_level}</p></div>
                <div><p className="text-muted">Deployment</p>
                  <Badge variant={gateResult.deployment_allowed ? 'certified' : 'failed'}>
                    {gateResult.deployment_allowed ? 'ALLOWED' : 'BLOCKED'}
                  </Badge>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Add integration form */}
        {showAdd && (
          <div className="bg-white border border-stone rounded-card p-6">
            <h2 className="text-[14px] font-semibold text-ink mb-4">Register CI/CD integration</h2>
            <div className="grid md:grid-cols-3 gap-4 mb-4">
              <div>
                <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Provider</label>
                <select value={newInt.provider} onChange={e => setNewInt(i => ({ ...i, provider: e.target.value }))}
                  className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest bg-white">
                  <option value="github">GitHub</option>
                  <option value="gitlab">GitLab</option>
                  <option value="jenkins">Jenkins</option>
                </select>
              </div>
              <div>
                <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Repository URL</label>
                <input value={newInt.repo_url} onChange={e => setNewInt(i => ({ ...i, repo_url: e.target.value }))}
                  placeholder="https://github.com/org/repo"
                  className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
              </div>
              <div>
                <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Webhook secret</label>
                <input type="password" value={newInt.webhook_secret} onChange={e => setNewInt(i => ({ ...i, webhook_secret: e.target.value }))}
                  placeholder="Optional HMAC secret"
                  className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
              </div>
            </div>
            <div className="flex gap-2">
              <Button variant="primary" size="sm" onClick={addIntegration} disabled={adding || !newInt.repo_url}>
                {adding ? <><RefreshCw size={13} className="animate-spin" />Adding…</> : 'Add Integration'}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setShowAdd(false)}>Cancel</Button>
            </div>
          </div>
        )}

        {/* Integrations list */}
        <div className="bg-white border border-stone rounded-card overflow-hidden">
          <div className="flex items-center justify-between px-6 py-4 border-b border-stone">
            <h2 className="text-[14px] font-semibold text-ink">Active integrations</h2>
            <button onClick={load} className="text-muted hover:text-ink transition-colors">
              <RefreshCw size={14} strokeWidth={1.5} className={loading ? 'animate-spin' : ''} />
            </button>
          </div>
          {loading ? (
            <div className="p-8 flex justify-center"><RefreshCw size={20} className="animate-spin text-muted" /></div>
          ) : integrations.length === 0 ? (
            <div className="p-8 text-center">
              <GitBranch size={28} className="mx-auto text-muted mb-2" strokeWidth={1.25} />
              <p className="text-[13px] text-muted">No CI/CD integrations yet. Add one above to enable governance gates.</p>
            </div>
          ) : (
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-stone">
                  {['Provider', 'Repository', 'Branch', 'Status', 'Last Run'].map(h => (
                    <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {integrations.map((i: any) => (
                  <tr key={i.id} className="border-b border-stone/50 hover:bg-[#F7F6F2] transition-colors">
                    <td className="px-5 py-3">
                      <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-stone text-ink-soft">{i.provider}</span>
                    </td>
                    <td className="px-5 py-3">
                      <div>
                        <p className="text-[13px] font-medium text-ink">{i.repo_name}</p>
                        <p className="text-[10px] text-muted">{i.repo_url}</p>
                      </div>
                    </td>
                    <td className="px-5 py-3 text-[12px] font-mono text-muted">{i.branch_pattern ?? 'main'}</td>
                    <td className="px-5 py-3">
                      <div className="flex items-center gap-1.5">
                        <span className={`w-1.5 h-1.5 rounded-full ${i.is_active ? 'bg-forest' : 'bg-muted'}`} />
                        <span className="text-[12px] text-muted">{i.is_active ? 'Active' : 'Inactive'}</span>
                      </div>
                    </td>
                    <td className="px-5 py-3 text-[12px] text-muted">
                      {i.last_run_at ? new Date(i.last_run_at).toLocaleDateString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Webhook info */}
        <div className="bg-white border border-stone rounded-card p-6">
          <h2 className="text-[14px] font-semibold text-ink mb-3">Webhook endpoint</h2>
          <p className="text-[12px] text-muted mb-3">Configure your Git provider to send webhook events to this URL. ML Guard will automatically run governance checks on every PR.</p>
          <div className="bg-[#0F0F0E] rounded-[8px] px-4 py-3 font-mono text-[12px] text-[#3ECF8E]">
            POST {typeof window !== 'undefined' ? window.location.origin : ''}/api/v1/webhooks/github
          </div>
        </div>
      </div>
    </div>
  );
}
