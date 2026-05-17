'use client';

import { useEffect, useState, useCallback } from 'react';
import { RefreshCw, FileText, Download, Play } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { modelsApi, api, type ModelItem } from '@/lib/api';

interface ReportItem {
  id: string; model_id: string; report_type: string; created_at: string; file_url?: string;
}

export default function ReportsPage() {
  const [reports, setReports] = useState<ReportItem[]>([]);
  const [models, setModels] = useState<ModelItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState('');

  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [reps, mods] = await Promise.all([
        api.get<{ items: ReportItem[] }>('/reports').catch(() => ({ items: [] })),
        modelsApi.list(1, 100),
      ]);
      setReports(reps.items ?? []);
      setModels(mods.items ?? []);
      if (mods.items?.length) setSelectedModel(mods.items[0].model_id);
    } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  const generate = async () => {
    if (!selectedModel) return;
    setGenerating(selectedModel);
    try {
      await api.post(`/reports/${selectedModel}/pdf`, {});
      await load();
    } catch (e: any) { setError(e.message); } finally { setGenerating(null); }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Reports</h1>
          <p className="text-[11px] text-muted">Generate and download governance audit reports</p>
        </div>
        <button onClick={load} className="text-muted hover:text-ink transition-colors" aria-label="Refresh">
          <RefreshCw size={15} strokeWidth={1.5} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      <div className="flex-1 p-8 space-y-6">
        {/* Generate */}
        <div className="bg-white border border-stone rounded-card p-6">
          <h2 className="text-[14px] font-semibold text-ink mb-4">Generate report</h2>
          <div className="flex items-center gap-3">
            <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}
              className="h-10 px-3 text-[13px] border border-stone rounded-[8px] bg-white outline-none focus:border-forest min-w-[240px]">
              {models.length === 0
                ? <option value="">No models registered</option>
                : models.map(m => <option key={m.model_id} value={m.model_id}>{m.name}</option>)}
            </select>
            <Button variant="primary" size="sm" className="gap-2" onClick={generate}
              disabled={!selectedModel || !!generating}>
              {generating
                ? <><RefreshCw size={13} className="animate-spin" />Generating…</>
                : <><Play size={13} />Generate PDF Report</>}
            </Button>
          </div>
          <p className="text-[11px] text-muted mt-2">Report will include governance score, drift analysis, performance metrics, and advisories.</p>
        </div>

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger flex items-center justify-between">
            <span>⚠ {error}</span>
            <button onClick={load} className="text-forest underline text-[12px]">Retry</button>
          </div>
        )}

        {/* Reports list */}
        <div className="bg-white border border-stone rounded-card overflow-hidden">
          <div className="px-6 py-4 border-b border-stone bg-[#F7F6F2]">
            <h2 className="text-[14px] font-semibold text-ink">Generated reports</h2>
          </div>
          {loading ? (
            Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="px-6 py-4 border-b border-stone/50 flex gap-4 animate-pulse">
                <div className="h-8 w-8 bg-stone rounded" />
                <div className="flex-1">
                  <div className="h-3 bg-stone rounded w-1/2 mb-2" />
                  <div className="h-2.5 bg-stone rounded w-1/3" />
                </div>
              </div>
            ))
          ) : reports.length === 0 ? (
            <div className="py-16 text-center">
              <FileText size={32} className="mx-auto text-stone mb-3" strokeWidth={1} />
              <p className="text-[14px] font-medium text-ink mb-1">No reports generated yet</p>
              <p className="text-[13px] text-muted">Select a model above and click Generate to create your first report.</p>
            </div>
          ) : (
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  {['Report', 'Type', 'Model ID', 'Created', 'Download'].map(h => (
                    <th key={h} className="text-left px-6 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted border-b border-stone">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {reports.map(r => (
                  <tr key={r.id} className="border-b border-stone/50 hover:bg-[#F7F6F2] transition-colors">
                    <td className="px-6 py-3.5">
                      <div className="flex items-center gap-3">
                        <FileText size={16} className="text-forest flex-shrink-0" strokeWidth={1.5} />
                        <span className="text-[13px] font-medium text-ink font-mono">{r.id.slice(0, 12)}…</span>
                      </div>
                    </td>
                    <td className="px-6 py-3.5">
                      <span className="text-[11px] font-medium px-2 py-0.5 rounded-badge bg-mist text-forest capitalize">{r.report_type}</span>
                    </td>
                    <td className="px-6 py-3.5 text-[12px] font-mono text-muted">{r.model_id.slice(0, 12)}…</td>
                    <td className="px-6 py-3.5 text-[12px] text-muted">{new Date(r.created_at).toLocaleString()}</td>
                    <td className="px-6 py-3.5">
                      {r.file_url ? (
                        <a href={r.file_url} target="_blank" rel="noopener noreferrer"
                          className="flex items-center gap-1.5 text-[12px] text-forest hover:underline">
                          <Download size={12} strokeWidth={1.5} />Download
                        </a>
                      ) : (
                        <span className="text-[12px] text-muted">Processing…</span>
                      )}
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
