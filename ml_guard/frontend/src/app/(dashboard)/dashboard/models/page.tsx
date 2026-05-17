'use client';

import { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';
import { Plus, Search, RefreshCw, Package } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { modelsApi, type ModelItem } from '@/lib/api';

function ScoreBadge({ score }: { score: number | null }) {
  if (score === null) return <span className="text-[12px] text-muted">—</span>;
  const v: 'certified' | 'conditional' | 'failed' = score >= 80 ? 'certified' : score >= 60 ? 'conditional' : 'failed';
  return <Badge variant={v}>{score.toFixed(0)}</Badge>;
}

function RegisterModal({ onClose, onSuccess }: { onClose: () => void; onSuccess: () => void }) {
  const [name, setName] = useState('');
  const [desc, setDesc] = useState('');
  const [owner, setOwner] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      await modelsApi.register({ model_name: name, description: desc, owner });
      onSuccess();
      onClose();
    } catch (err: any) {
      setError(err.message ?? 'Failed to register model');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-ink/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-card w-full max-w-md p-6 shadow-xl">
        <h2 className="text-[16px] font-semibold text-ink mb-4">Register new model</h2>
        <form onSubmit={submit} className="flex flex-col gap-4">
          <div>
            <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Model name *</label>
            <input
              value={name}
              onChange={e => setName(e.target.value)}
              required
              placeholder="e.g. credit-risk-v4"
              className="w-full h-10 px-3 text-[14px] text-ink bg-white border border-stone rounded-[8px] outline-none focus:border-forest"
            />
            <p className="text-[11px] text-muted mt-1">Only letters, numbers, hyphens, underscores, dots</p>
          </div>
          <div>
            <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Description</label>
            <input
              value={desc}
              onChange={e => setDesc(e.target.value)}
              placeholder="Brief model description"
              className="w-full h-10 px-3 text-[14px] text-ink bg-white border border-stone rounded-[8px] outline-none focus:border-forest"
            />
          </div>
          <div>
            <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Owner / team</label>
            <input
              value={owner}
              onChange={e => setOwner(e.target.value)}
              placeholder="e.g. ml-team@company.com"
              className="w-full h-10 px-3 text-[14px] text-ink bg-white border border-stone rounded-[8px] outline-none focus:border-forest"
            />
          </div>
          {error && <p className="text-[12px] text-danger">{error}</p>}
          <div className="flex gap-3 justify-end pt-2">
            <Button type="button" variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
            <Button type="submit" variant="primary" size="sm" disabled={loading}>
              {loading ? 'Registering…' : 'Register model'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [total, setTotal] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await modelsApi.list(1, 100);
      setModels(res.items ?? []);
      setTotal(res.total ?? 0);
    } catch (e: any) {
      setError(e.message ?? 'Failed to load models');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const filtered = models.filter(m =>
    m.name.toLowerCase().includes(search.toLowerCase()) ||
    (m.provider ?? '').toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Model Registry</h1>
          <p className="text-[11px] text-muted">{total} model{total !== 1 ? 's' : ''} registered</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={load} className="text-muted hover:text-ink transition-colors" aria-label="Refresh">
            <RefreshCw size={15} strokeWidth={1.5} className={loading ? 'animate-spin' : ''} />
          </button>
          <Button variant="primary" size="sm" className="gap-1.5" onClick={() => setShowModal(true)}>
            <Plus size={14} strokeWidth={2} />
            Register model
          </Button>
        </div>
      </div>

      <div className="flex-1 p-8">
        {/* Search */}
        <div className="relative mb-5 max-w-sm">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" strokeWidth={1.5} />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search models…"
            className="w-full h-9 pl-8 pr-3 text-[13px] text-ink bg-white border border-stone rounded-[8px] outline-none focus:border-forest"
          />
        </div>

        {error && (
          <div className="mb-5 p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger flex items-center justify-between">
            <span>⚠ {error}</span>
            <button onClick={load} className="text-forest underline text-[12px]">Retry</button>
          </div>
        )}

        <div className="bg-white border border-stone rounded-card overflow-hidden">
          <table className="w-full border-collapse">
            <thead className="bg-[#F7F6F2]">
              <tr>
                {['Model', 'Provider / Owner', 'Versions', 'Latest Gov. Score', 'Risk Class', 'Registered'].map(h => (
                  <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted border-b border-stone">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: 6 }).map((_, i) => (
                  <tr key={i} className="border-b border-stone/50 animate-pulse">
                    {[1,2,3,4,5,6].map(j => (
                      <td key={j} className="px-5 py-3.5">
                        <div className="h-3 bg-stone rounded-full" style={{ width: `${50 + j*8}%` }} />
                      </td>
                    ))}
                  </tr>
                ))
                : filtered.length === 0
                ? (
                  <tr>
                    <td colSpan={6} className="py-16 text-center">
                      <Package size={32} className="mx-auto text-stone mb-3" strokeWidth={1} />
                      <p className="text-[14px] font-medium text-ink mb-1">No models found</p>
                      <p className="text-[13px] text-muted mb-4">
                        {search ? 'No models match your search.' : 'Register your first model to get started.'}
                      </p>
                      {!search && (
                        <Button variant="primary" size="sm" onClick={() => setShowModal(true)}>
                          Register first model
                        </Button>
                      )}
                    </td>
                  </tr>
                )
                : filtered.map(m => (
                  <tr key={m.model_id} className="border-b border-stone/50 hover:bg-[#F0EDE8] transition-colors duration-100">
                    <td className="px-5 py-3.5">
                      <Link href={`/dashboard/models/${m.model_id}`} className="font-medium text-ink hover:text-forest transition-colors text-[13px]">
                        {m.name}
                      </Link>
                    </td>
                    <td className="px-5 py-3.5 text-[13px] text-ink-soft">{m.provider || '—'}</td>
                    <td className="px-5 py-3.5 text-[13px] text-muted font-mono">{m.version_count}</td>
                    <td className="px-5 py-3.5"><ScoreBadge score={m.latest_governance_score} /></td>
                    <td className="px-5 py-3.5">
                      {m.latest_risk_class
                        ? <span className="text-[11px] font-medium px-2 py-0.5 rounded-badge bg-stone text-ink-soft">{m.latest_risk_class}</span>
                        : <span className="text-[12px] text-muted">—</span>}
                    </td>
                    <td className="px-5 py-3.5 text-[12px] text-muted">{new Date(m.created_at).toLocaleDateString()}</td>
                  </tr>
                ))
              }
            </tbody>
          </table>
        </div>
      </div>

      {showModal && (
        <RegisterModal onClose={() => setShowModal(false)} onSuccess={load} />
      )}
    </div>
  );
}
