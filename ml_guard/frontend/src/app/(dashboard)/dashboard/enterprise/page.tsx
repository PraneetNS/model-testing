'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, Building2, Users, Key, Shield, BarChart2, Activity, CheckCircle, Plus, Copy, Eye, EyeOff } from 'lucide-react';
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

function StatCard({ icon: Icon, label, value, color = '#1A5F3A' }: any) {
  return (
    <div className="bg-white border border-stone rounded-card p-5">
      <div className="flex items-center justify-between mb-3">
        <p className="text-[11px] text-muted uppercase tracking-[0.05em]">{label}</p>
        <Icon size={14} strokeWidth={1.5} style={{ color }} />
      </div>
      <p className="text-[26px] font-bold text-ink" style={{ letterSpacing: '-0.03em' }}>{value ?? '—'}</p>
    </div>
  );
}

type Tab = 'summary' | 'orgs' | 'policies' | 'api-keys';

export default function EnterpriseHubPage() {
  const [tab, setTab] = useState<Tab>('summary');
  const [summary, setSummary] = useState<any>(null);
  const [orgs, setOrgs] = useState<any[]>([]);
  const [policies, setPolicies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Org create form
  const [showOrgForm, setShowOrgForm] = useState(false);
  const [orgForm, setOrgForm] = useState({ name: '', slug: '', plan: 'free' });
  const [creatingOrg, setCreatingOrg] = useState(false);

  // API key
  const [selectedOrgId, setSelectedOrgId] = useState('');
  const [keyLabel, setKeyLabel] = useState('');
  const [newKey, setNewKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [generatingKey, setGeneratingKey] = useState(false);

  const load = async () => {
    setLoading(true); setError(null);
    try {
      const [summ, orgList, polList] = await Promise.all([
        apiFetch('/enterprise/summary'),
        apiFetch('/orgs'),
        apiFetch('/enterprise/policies'),
      ]);
      setSummary(summ);
      setOrgs(Array.isArray(orgList) ? orgList : []);
      setPolicies(Array.isArray(polList) ? polList : []);
      if (orgList?.length) setSelectedOrgId(orgList[0].id);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const createOrg = async () => {
    if (!orgForm.name || !orgForm.slug) return;
    setCreatingOrg(true);
    try {
      await apiFetch('/orgs', { method: 'POST', body: JSON.stringify(orgForm) });
      setShowOrgForm(false); setOrgForm({ name: '', slug: '', plan: 'free' });
      await load();
    } catch (e: any) { setError(e.message); }
    finally { setCreatingOrg(false); }
  };

  const generateApiKey = async () => {
    if (!selectedOrgId || !keyLabel) return;
    setGeneratingKey(true); setNewKey('');
    try {
      const d = await apiFetch(`/orgs/${selectedOrgId}/api-keys`, {
        method: 'POST', body: JSON.stringify({ label: keyLabel, scopes: ['audit', 'behavior', 'monitor'] }),
      });
      setNewKey(d.key ?? '');
    } catch (e: any) { setError(e.message); }
    finally { setGeneratingKey(false); }
  };

  const TABS: { id: Tab; label: string; icon: any }[] = [
    { id: 'summary', label: 'Summary', icon: BarChart2 },
    { id: 'orgs', label: 'Organizations', icon: Building2 },
    { id: 'policies', label: 'Policies', icon: Shield },
    { id: 'api-keys', label: 'API Keys', icon: Key },
  ];

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Enterprise Hub</h1>
          <p className="text-[11px] text-muted">Multi-tenant · RBAC · API keys · policies</p>
        </div>
        <button onClick={load} className="text-muted hover:text-ink transition-colors">
          <RefreshCw size={16} strokeWidth={1.5} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-stone bg-white px-8">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button key={id} onClick={() => setTab(id)}
            className={`flex items-center gap-2 px-4 py-3 text-[13px] font-medium border-b-2 transition-colors ${tab === id ? 'border-forest text-forest' : 'border-transparent text-muted hover:text-ink'}`}>
            <Icon size={13} strokeWidth={1.5} /> {label}
          </button>
        ))}
      </div>

      <div className="flex-1 p-8 space-y-6 overflow-auto">
        {error && <div className="p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger">⚠ {error}</div>}

        {/* SUMMARY TAB */}
        {tab === 'summary' && (
          <>
            {loading ? (
              <div className="flex justify-center p-8"><RefreshCw size={20} className="animate-spin text-muted" /></div>
            ) : summary ? (
              <>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <StatCard icon={Building2} label="Organizations" value={summary.total_organizations ?? orgs.length} />
                  <StatCard icon={Users} label="Users" value={summary.total_users} color="#0369A1" />
                  <StatCard icon={Activity} label="Total Scans" value={summary.total_scans} color="#B35A00" />
                  <StatCard icon={BarChart2} label="Avg Gov. Score" value={summary.avg_governance_score ? `${summary.avg_governance_score.toFixed(0)}/100` : '—'} />
                </div>

                {/* Health cards */}
                {summary.health && (
                  <div className="grid md:grid-cols-3 gap-4">
                    {Object.entries(summary.health).map(([key, val]: [string, any]) => (
                      <div key={key} className="bg-white border border-stone rounded-card p-4 flex items-center gap-3">
                        <CheckCircle size={16} className={val ? 'text-forest' : 'text-danger'} />
                        <div>
                          <p className="text-[12px] font-semibold text-ink capitalize">{key.replace(/_/g, ' ')}</p>
                          <p className="text-[11px] text-muted">{String(val)}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Gate breakdown */}
                {summary.gate_breakdown && (
                  <div className="bg-white border border-stone rounded-card p-6">
                    <h2 className="text-[14px] font-semibold text-ink mb-4">Gate status breakdown</h2>
                    <div className="flex gap-5 flex-wrap">
                      {Object.entries(summary.gate_breakdown).map(([status, count]: [string, any]) => (
                        <div key={status} className="flex items-center gap-2">
                          <Badge variant={status === 'PASSED' ? 'certified' : status === 'WARNING' ? 'conditional' : 'failed'}>{status}</Badge>
                          <span className="text-[15px] font-bold text-ink">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center text-muted p-8">No enterprise data available.</div>
            )}
          </>
        )}

        {/* ORGS TAB */}
        {tab === 'orgs' && (
          <>
            <div className="flex justify-end">
              <Button variant="primary" size="sm" onClick={() => setShowOrgForm(s => !s)} className="gap-2">
                <Plus size={13} /> Create Organization
              </Button>
            </div>

            {showOrgForm && (
              <div className="bg-white border border-stone rounded-card p-6">
                <h2 className="text-[14px] font-semibold text-ink mb-4">New organization</h2>
                <div className="grid md:grid-cols-3 gap-4 mb-4">
                  <div>
                    <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Name *</label>
                    <input value={orgForm.name} onChange={e => setOrgForm(f => ({ ...f, name: e.target.value }))}
                      placeholder="Acme Corp" className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
                  </div>
                  <div>
                    <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Slug *</label>
                    <input value={orgForm.slug} onChange={e => setOrgForm(f => ({ ...f, slug: e.target.value }))}
                      placeholder="acme-corp" className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest font-mono" />
                  </div>
                  <div>
                    <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Plan</label>
                    <select value={orgForm.plan} onChange={e => setOrgForm(f => ({ ...f, plan: e.target.value }))}
                      className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest bg-white">
                      {['free', 'pro', 'enterprise'].map(p => <option key={p}>{p}</option>)}
                    </select>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button variant="primary" size="sm" onClick={createOrg} disabled={creatingOrg || !orgForm.name}>
                    {creatingOrg ? <><RefreshCw size={13} className="animate-spin" />Creating…</> : 'Create'}
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => setShowOrgForm(false)}>Cancel</Button>
                </div>
              </div>
            )}

            <div className="bg-white border border-stone rounded-card overflow-hidden">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b border-stone">
                    {['Name', 'Slug', 'Plan', 'Created'].map(h => (
                      <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {orgs.length === 0 ? (
                    <tr><td colSpan={4} className="px-5 py-8 text-center text-[13px] text-muted">No organizations yet.</td></tr>
                  ) : orgs.map((o: any) => (
                    <tr key={o.id} className="border-b border-stone/50 hover:bg-[#F7F6F2] transition-colors">
                      <td className="px-5 py-3 text-[13px] font-semibold text-ink">{o.name}</td>
                      <td className="px-5 py-3 text-[12px] font-mono text-muted">{o.slug}</td>
                      <td className="px-5 py-3">
                        <span className="text-[11px] font-medium px-2 py-0.5 rounded-badge bg-stone text-ink-soft uppercase">{o.plan}</span>
                      </td>
                      <td className="px-5 py-3 text-[12px] text-muted">{o.created_at ? new Date(o.created_at).toLocaleDateString() : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {/* POLICIES TAB */}
        {tab === 'policies' && (
          <div className="bg-white border border-stone rounded-card overflow-hidden">
            <div className="px-6 py-4 border-b border-stone">
              <h2 className="text-[14px] font-semibold text-ink">Governance policies</h2>
            </div>
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-stone">
                  {['Name', 'Version', 'Status', 'Notes', 'Created'].map(h => (
                    <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {policies.length === 0 ? (
                  <tr><td colSpan={5} className="px-5 py-8 text-center text-[13px] text-muted">No policies configured.</td></tr>
                ) : policies.map((p: any) => (
                  <tr key={p.id} className="border-b border-stone/50 hover:bg-[#F7F6F2] transition-colors">
                    <td className="px-5 py-3 text-[13px] font-medium text-ink">{p.name}</td>
                    <td className="px-5 py-3 text-[12px] font-mono text-muted">{p.version ?? '—'}</td>
                    <td className="px-5 py-3">
                      <Badge variant={p.is_active ? 'certified' : 'conditional'}>{p.is_active ? 'Active' : 'Inactive'}</Badge>
                    </td>
                    <td className="px-5 py-3 text-[12px] text-muted">{p.notes || '—'}</td>
                    <td className="px-5 py-3 text-[12px] text-muted">{p.created_at ? new Date(p.created_at).toLocaleDateString() : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* API KEYS TAB */}
        {tab === 'api-keys' && (
          <>
            <div className="bg-white border border-stone rounded-card p-6">
              <h2 className="text-[14px] font-semibold text-ink mb-4">Generate API key</h2>
              <div className="flex gap-3 items-end flex-wrap">
                <div>
                  <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Organization</label>
                  <select value={selectedOrgId} onChange={e => setSelectedOrgId(e.target.value)}
                    className="h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest bg-white">
                    {orgs.map(o => <option key={o.id} value={o.id}>{o.name}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Label</label>
                  <input value={keyLabel} onChange={e => setKeyLabel(e.target.value)}
                    placeholder="e.g. CI pipeline key" className="h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
                </div>
                <Button variant="primary" size="sm" onClick={generateApiKey} disabled={generatingKey || !selectedOrgId || !keyLabel} className="gap-2">
                  {generatingKey ? <><RefreshCw size={13} className="animate-spin" />Generating…</> : <><Key size={13} />Generate Key</>}
                </Button>
              </div>

              {newKey && (
                <div className="mt-4 p-4 bg-amber-50 border border-amber-200 rounded-[8px]">
                  <p className="text-[12px] font-semibold text-amber-700 mb-2">⚠ Copy this key now — it will not be shown again.</p>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 text-[12px] font-mono text-ink bg-white px-3 py-2 rounded-[6px] border border-stone overflow-x-auto">
                      {showKey ? newKey : '•'.repeat(Math.min(newKey.length, 48))}
                    </code>
                    <button onClick={() => setShowKey(s => !s)} className="text-muted hover:text-ink transition-colors">
                      {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
                    </button>
                    <button onClick={() => navigator.clipboard.writeText(newKey)} className="text-muted hover:text-forest transition-colors">
                      <Copy size={14} />
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div className="bg-white border border-stone rounded-card p-6">
              <p className="text-[13px] text-muted">
                API keys are hashed with SHA-256 before storage. The raw key is only shown once at generation time.
                Use the key in the <code className="text-forest text-[12px]">X-API-Key</code> header for all authenticated requests.
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
