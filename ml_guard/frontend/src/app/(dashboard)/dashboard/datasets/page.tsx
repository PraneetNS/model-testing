'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, Database, Upload, Plus, Trash2, CheckCircle, AlertTriangle } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000') + '/api/v1';
const HDR = { 'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'dev-secret-key' };

async function apiFetch(path: string, opts: RequestInit = {}) {
  const r = await fetch(`${BASE}${path}`, { ...opts, headers: { ...HDR, ...(opts.headers ?? {}) } });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail ?? `HTTP ${r.status}`);
  return d;
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [form, setForm] = useState({ name: '', description: '', source_type: 'csv', location: '' });
  const [showForm, setShowForm] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const d = await apiFetch('/datasets');
      setDatasets(d.items ?? d ?? []);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const create = async () => {
    if (!form.name) return;
    setCreating(true);
    try {
      await apiFetch('/datasets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      setShowForm(false);
      setForm({ name: '', description: '', source_type: 'csv', location: '' });
      await load();
    } catch (e: any) { setError(e.message); }
    finally { setCreating(false); }
  };

  const del = async (id: string) => {
    if (!confirm('Delete this dataset?')) return;
    try { await apiFetch(`/datasets/${id}`, { method: 'DELETE' }); await load(); }
    catch (e: any) { setError(e.message); }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Datasets</h1>
          <p className="text-[11px] text-muted">Data lineage · health · schema tracking</p>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={load} className="gap-2">
            <RefreshCw size={13} strokeWidth={1.5} className={loading ? 'animate-spin' : ''} /> Refresh
          </Button>
          <Button variant="primary" size="sm" onClick={() => setShowForm(s => !s)} className="gap-2">
            <Plus size={13} /> Register Dataset
          </Button>
        </div>
      </div>

      <div className="flex-1 p-8 space-y-5 overflow-auto">
        {error && <div className="p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger">⚠ {error}</div>}

        {/* Create form */}
        {showForm && (
          <div className="bg-white border border-stone rounded-card p-6">
            <h2 className="text-[14px] font-semibold text-ink mb-4">Register new dataset</h2>
            <div className="grid md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Name *</label>
                <input value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
                  placeholder="e.g. credit-train-2025" className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
              </div>
              <div>
                <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Source Type</label>
                <select value={form.source_type} onChange={e => setForm(f => ({ ...f, source_type: e.target.value }))}
                  className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest bg-white">
                  {['csv', 'parquet', 's3', 'gcs', 'postgres', 'bigquery'].map(t => <option key={t}>{t}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Location / URI</label>
                <input value={form.location} onChange={e => setForm(f => ({ ...f, location: e.target.value }))}
                  placeholder="s3://bucket/path or /local/path" className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
              </div>
              <div>
                <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Description</label>
                <input value={form.description} onChange={e => setForm(f => ({ ...f, description: e.target.value }))}
                  placeholder="Optional description" className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
              </div>
            </div>
            <div className="flex gap-2">
              <Button variant="primary" size="sm" onClick={create} disabled={creating || !form.name}>
                {creating ? <><RefreshCw size={13} className="animate-spin" />Registering…</> : 'Register'}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setShowForm(false)}>Cancel</Button>
            </div>
          </div>
        )}

        {/* Dataset list */}
        {loading ? (
          <div className="bg-white border border-stone rounded-card p-8 flex justify-center">
            <RefreshCw size={20} className="animate-spin text-muted" />
          </div>
        ) : datasets.length === 0 ? (
          <div className="bg-white border border-stone rounded-card p-12 text-center">
            <Database size={32} className="mx-auto text-muted mb-3" strokeWidth={1.25} />
            <p className="text-[14px] font-semibold text-ink mb-1">No datasets registered</p>
            <p className="text-[12px] text-muted">Register reference and production datasets to enable drift monitoring and data quality checks.</p>
          </div>
        ) : (
          <div className="bg-white border border-stone rounded-card overflow-hidden">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b border-stone">
                  {['Name', 'Source', 'Rows', 'Features', 'Status', 'Registered', ''].map(h => (
                    <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {datasets.map((d: any) => (
                  <tr key={d.id ?? d.dataset_id} className="border-b border-stone/50 hover:bg-[#F7F6F2] transition-colors">
                    <td className="px-5 py-3">
                      <div>
                        <p className="text-[13px] font-semibold text-ink">{d.name}</p>
                        <p className="text-[11px] text-muted">{d.description ?? '—'}</p>
                      </div>
                    </td>
                    <td className="px-5 py-3">
                      <span className="text-[11px] font-medium px-2 py-0.5 rounded-badge bg-stone text-ink-soft uppercase">{d.source_type ?? d.format ?? '—'}</span>
                    </td>
                    <td className="px-5 py-3 text-[13px] text-muted font-mono">{d.row_count?.toLocaleString() ?? '—'}</td>
                    <td className="px-5 py-3 text-[13px] text-muted">{d.feature_count ?? d.num_features ?? '—'}</td>
                    <td className="px-5 py-3">
                      {d.health_status === 'healthy' || d.status === 'active' ? (
                        <div className="flex items-center gap-1 text-forest text-[12px]"><CheckCircle size={12} /> Healthy</div>
                      ) : (
                        <div className="flex items-center gap-1 text-warning text-[12px]"><AlertTriangle size={12} /> {d.health_status ?? d.status ?? 'Unknown'}</div>
                      )}
                    </td>
                    <td className="px-5 py-3 text-[12px] text-muted">
                      {d.created_at ? new Date(d.created_at).toLocaleDateString() : '—'}
                    </td>
                    <td className="px-5 py-3">
                      <button onClick={() => del(d.id ?? d.dataset_id)} className="text-muted hover:text-danger transition-colors">
                        <Trash2 size={13} strokeWidth={1.5} />
                      </button>
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
