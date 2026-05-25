'use client';

import { useEffect, useState, useCallback } from 'react';
import { RefreshCw, Box, AlertTriangle, CheckCircle, Search } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { modelsApi, inventoryApi, type ModelItem } from '@/lib/api';

interface AibomComponent {
  name: string; version: string; type: string; hash: string; cves: number;
}

export default function AibomPage() {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [selected, setSelected] = useState('');
  const [components, setComponents] = useState<AibomComponent[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingAibom, setLoadingAibom] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');

  useEffect(() => {
    modelsApi.list(1, 100)
      .then(r => {
        setModels(r.items ?? []);
        if (r.items?.length) setSelected(r.items[0].model_id);
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoadingAibom(true); setError(null);
    inventoryApi.aibom(selected)
      .then((r: any) => setComponents(r.components ?? []))
      .catch(e => {
        if (e.status === 404) {
          setComponents([]);
        } else {
          setError(e.message);
        }
      })
      .finally(() => setLoadingAibom(false));
  }, [selected]);

  const filtered = components.filter(c =>
    c.name?.toLowerCase().includes(search.toLowerCase()) ||
    c.type?.toLowerCase().includes(search.toLowerCase())
  );

  const totalCves = components.reduce((s, c) => s + (c.cves ?? 0), 0);
  const types = Array.from(new Set(components.map(c => c.type)));

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">AIBOM</h1>
          <p className="text-[11px] text-muted">AI Bill of Materials — components, dependencies &amp; integrity</p>
        </div>
        <div className="flex items-center gap-2">
          {totalCves > 0
            ? <Badge variant="failed">{totalCves} CVE{totalCves > 1 ? 's' : ''}</Badge>
            : components.length > 0
            ? <Badge variant="certified">No CVEs</Badge>
            : null}
        </div>
      </div>

      <div className="flex-1 p-8 space-y-5">
        {/* Model selector + search */}
        <div className="flex items-center gap-3 flex-wrap">
          <div>
            <label className="text-[12px] font-medium text-ink-soft mr-2">Model:</label>
            {loading ? (
              <div className="inline-block h-9 w-48 bg-stone rounded-[8px] animate-pulse" />
            ) : (
              <select value={selected} onChange={e => setSelected(e.target.value)}
                className="h-9 px-3 text-[13px] border border-stone rounded-[8px] bg-white text-ink outline-none focus:border-forest">
                {models.length === 0
                  ? <option value="">No models</option>
                  : models.map(m => <option key={m.model_id} value={m.model_id}>{m.name}</option>)}
              </select>
            )}
          </div>
          <div className="relative">
            <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" strokeWidth={1.5} />
            <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search components…"
              className="h-9 pl-8 pr-3 text-[13px] border border-stone rounded-[8px] bg-white outline-none focus:border-forest w-52" />
          </div>
        </div>

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger">⚠ {error}</div>
        )}

        {/* Stats */}
        {components.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Total components', value: components.length },
              { label: 'CVEs found', value: totalCves },
              { label: 'Types', value: types.length },
              { label: 'Hashed', value: components.filter(c => c.hash).length },
            ].map(s => (
              <div key={s.label} className="bg-white border border-stone rounded-card p-5">
                <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-1">{s.label}</p>
                <p className={`text-[26px] font-bold leading-none ${s.label === 'CVEs found' && s.value > 0 ? 'text-danger' : 'text-ink'}`}>{s.value}</p>
              </div>
            ))}
          </div>
        )}

        {/* Table */}
        <div className="bg-white border border-stone rounded-card overflow-hidden">
          <div className="px-6 py-4 border-b border-stone bg-[#F7F6F2]">
            <h2 className="text-[14px] font-semibold text-ink">Components</h2>
            <p className="text-[11px] text-muted mt-0.5">All libraries, datasets, and dependencies with integrity hashes</p>
          </div>
          <table className="w-full border-collapse">
            <thead>
              <tr>
                {['Component', 'Type', 'Version', 'SHA-256', 'CVEs', 'Status'].map(h => (
                  <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted border-b border-stone">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loadingAibom
                ? Array.from({ length: 6 }).map((_, i) => (
                  <tr key={i} className="border-b border-stone/50 animate-pulse">
                    {[1,2,3,4,5,6].map(j => (
                      <td key={j} className="px-5 py-3.5"><div className="h-3 bg-stone rounded-full" style={{ width: `${40 + j*8}%` }} /></td>
                    ))}
                  </tr>
                ))
                : filtered.length === 0
                ? (
                  <tr>
                    <td colSpan={6} className="py-16 text-center">
                      <Box size={32} className="mx-auto text-stone mb-3" strokeWidth={1} />
                      <p className="text-[14px] font-medium text-ink mb-1">
                        {components.length === 0 ? 'No AIBOM data for this model' : 'No matching components'}
                      </p>
                      <p className="text-[13px] text-muted">
                        {components.length === 0 ? 'Run a model audit to generate AIBOM data.' : 'Try a different search term.'}
                      </p>
                    </td>
                  </tr>
                )
                : filtered.map((c, i) => (
                  <tr key={i} className="border-b border-stone/50 hover:bg-[#F7F6F2] transition-colors">
                    <td className="px-5 py-3.5 text-[13px] font-medium text-ink">{c.name}</td>
                    <td className="px-5 py-3.5">
                      <span className="text-[11px] font-medium px-2 py-0.5 rounded-badge bg-stone text-ink-soft capitalize">{c.type}</span>
                    </td>
                    <td className="px-5 py-3.5 text-[13px] font-mono text-muted">{c.version || '—'}</td>
                    <td className="px-5 py-3.5 text-[11px] font-mono text-muted">{c.hash ? `${c.hash.slice(0, 16)}…` : '—'}</td>
                    <td className="px-5 py-3.5">
                      {(c.cves ?? 0) === 0
                        ? <span className="text-[12px] text-forest flex items-center gap-1"><CheckCircle size={12} />0</span>
                        : <span className="flex items-center gap-1 text-[12px] font-semibold text-danger"><AlertTriangle size={12} />{c.cves}</span>}
                    </td>
                    <td className="px-5 py-3.5">
                      <Badge variant={(c.cves ?? 0) === 0 ? 'certified' : 'failed'}>
                        {(c.cves ?? 0) === 0 ? 'CLEAN' : 'VULNERABLE'}
                      </Badge>
                    </td>
                  </tr>
                ))
              }
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
