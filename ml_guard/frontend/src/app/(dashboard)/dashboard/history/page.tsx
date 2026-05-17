'use client';

import { useEffect, useState, useCallback } from 'react';
import { RefreshCw, Clock, ChevronRight, GitCompare } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { api } from '@/lib/api';

interface ScanRecord {
  id: string; model_id: string; scan_type: string;
  governance_score: number | null; gate_status: string | null;
  checks_run: string[]; trigger_source: string | null;
  duration_ms: number | null; created_at: string;
}

interface ScanDetail extends ScanRecord { results_json: any; }

function GateBadge({ status }: { status: string | null }) {
  if (!status) return <span className="text-[12px] text-muted">—</span>;
  const v: any = status === 'PASS' || status === 'PASSED' ? 'certified' : status === 'BLOCK' || status === 'FAILED' ? 'failed' : 'conditional';
  return <Badge variant={v}>{status}</Badge>;
}

function timeAgo(ts: string) {
  const d = Date.now() - new Date(ts).getTime();
  const m = Math.floor(d / 60000);
  return m < 60 ? `${m}m ago` : m < 1440 ? `${Math.floor(m/60)}h ago` : `${Math.floor(m/1440)}d ago`;
}

function JsonSection({ title, data }: { title: string; data: any }) {
  if (!data || (typeof data === 'object' && Object.keys(data).length === 0)) return null;
  return (
    <div className="mb-4">
      <p className="text-[11px] font-semibold uppercase tracking-[0.05em] text-muted mb-2">{title}</p>
      <pre className="bg-[#F7F6F2] rounded-[8px] p-4 text-[11px] font-mono text-ink-soft overflow-x-auto max-h-48 overflow-y-auto">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}

export default function ScanHistoryPage() {
  const [scans, setScans] = useState<ScanRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<ScanDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [scanTypeFilter, setScanTypeFilter] = useState('');
  const [compareA, setCompareA] = useState('');
  const [compareB, setCompareB] = useState('');
  const [compareResult, setCompareResult] = useState<any>(null);
  const [comparing, setComparing] = useState(false);
  const [activeTab, setActiveTab] = useState<'history' | 'compare'>('history');

  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const qs = scanTypeFilter ? `?scan_type=${scanTypeFilter}&limit=100` : '?limit=100';
      const data = await api.get<ScanRecord[]>(`/history${qs}`);
      setScans(Array.isArray(data) ? data : []);
    } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  }, [scanTypeFilter]);

  useEffect(() => { load(); }, [load]);

  const openDetail = async (id: string) => {
    setLoadingDetail(true);
    try {
      const d = await api.get<ScanDetail>(`/history/${id}`);
      setSelected(d);
    } catch (e: any) { setError(e.message); } finally { setLoadingDetail(false); }
  };

  const runCompare = async () => {
    if (!compareA || !compareB) return;
    setComparing(true);
    try {
      const r = await api.get<any>(`/compare?scan_a=${compareA}&scan_b=${compareB}`);
      setCompareResult(r);
    } catch (e: any) { setError(e.message); } finally { setComparing(false); }
  };

  const scanTypes = Array.from(new Set(scans.map(s => s.scan_type).filter(Boolean)));

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Scan History</h1>
          <p className="text-[11px] text-muted">{scans.length} scan records</p>
        </div>
        <button onClick={load} className="text-muted hover:text-ink transition-colors" aria-label="Refresh">
          <RefreshCw size={15} strokeWidth={1.5} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex px-8 bg-white border-b border-stone gap-6">
        {(['history', 'compare'] as const).map(tab => (
          <button key={tab} onClick={() => setActiveTab(tab)}
            className={`py-3 text-[13px] font-medium border-b-2 transition-colors capitalize -mb-px ${
              activeTab === tab ? 'border-forest text-forest' : 'border-transparent text-muted hover:text-ink'
            }`}>
            {tab === 'compare' ? <span className="flex items-center gap-1.5"><GitCompare size={13} />Compare Scans</span> : 'History'}
          </button>
        ))}
      </div>

      <div className="flex-1 flex overflow-hidden">
        {activeTab === 'history' && (
          <>
            {/* List */}
            <div className={`flex flex-col border-r border-stone bg-white overflow-y-auto ${selected ? 'w-[420px] flex-shrink-0' : 'flex-1'}`}>
              {/* Filter */}
              <div className="px-5 py-3 border-b border-stone flex items-center gap-2">
                <span className="text-[11px] text-muted">Type:</span>
                <select value={scanTypeFilter} onChange={e => setScanTypeFilter(e.target.value)}
                  className="h-7 px-2 text-[12px] border border-stone rounded-[6px] bg-white text-ink outline-none">
                  <option value="">All</option>
                  {scanTypes.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>

              {error && <div className="mx-4 my-3 p-3 bg-red-50 border border-red-200 rounded text-[12px] text-danger">⚠ {error}</div>}

              {loading ? (
                Array.from({ length: 8 }).map((_, i) => (
                  <div key={i} className="px-5 py-4 border-b border-stone/50 animate-pulse">
                    <div className="h-3 bg-stone rounded-full w-3/4 mb-2" />
                    <div className="h-2.5 bg-stone rounded-full w-1/2" />
                  </div>
                ))
              ) : scans.length === 0 ? (
                <div className="flex-1 flex items-center justify-center">
                  <div className="text-center py-16">
                    <Clock size={32} className="mx-auto text-stone mb-3" strokeWidth={1} />
                    <p className="text-[14px] font-medium text-ink mb-1">No scan history yet</p>
                    <p className="text-[13px] text-muted">Run a model audit to see results here.</p>
                  </div>
                </div>
              ) : (
                scans.map(scan => (
                  <button key={scan.id} onClick={() => openDetail(scan.id)}
                    className={`w-full text-left px-5 py-4 border-b border-stone/50 hover:bg-[#F7F6F2] transition-colors flex items-start gap-3 ${selected?.id === scan.id ? 'bg-mist border-l-2 border-forest' : ''}`}>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-[11px] font-medium px-1.5 py-0.5 rounded bg-stone text-ink-soft capitalize">{scan.scan_type}</span>
                        <GateBadge status={scan.gate_status} />
                      </div>
                      <p className="text-[12px] font-mono text-muted truncate">{scan.id.slice(0, 16)}…</p>
                      <div className="flex items-center gap-3 mt-1">
                        {scan.governance_score != null && (
                          <span className="text-[11px] text-muted">Score: <strong className="text-ink">{scan.governance_score.toFixed(0)}</strong></span>
                        )}
                        <span className="text-[11px] text-muted">{timeAgo(scan.created_at)}</span>
                      </div>
                    </div>
                    <ChevronRight size={14} className="text-muted flex-shrink-0 mt-1" />
                  </button>
                ))
              )}
            </div>

            {/* Detail panel */}
            {selected && (
              <div className="flex-1 overflow-y-auto p-6">
                <div className="flex items-center justify-between mb-5">
                  <h2 className="text-[15px] font-semibold text-ink">Scan Detail</h2>
                  <button onClick={() => setSelected(null)} className="text-[12px] text-muted hover:text-ink">Close ✕</button>
                </div>

                {loadingDetail ? (
                  <div className="space-y-3 animate-pulse">
                    {Array.from({ length: 6 }).map((_, i) => <div key={i} className="h-8 bg-stone rounded" />)}
                  </div>
                ) : (
                  <>
                    {/* Summary */}
                    <div className="grid grid-cols-2 gap-3 mb-6">
                      {[
                        ['Scan ID', selected.id.slice(0, 16) + '…'],
                        ['Type', selected.scan_type],
                        ['Gate status', selected.gate_status ?? '—'],
                        ['Gov. score', selected.governance_score?.toFixed(0) ?? '—'],
                        ['Trigger', selected.trigger_source ?? '—'],
                        ['Duration', selected.duration_ms ? `${selected.duration_ms}ms` : '—'],
                        ['Checks', (selected.checks_run ?? []).join(', ') || '—'],
                        ['Created', new Date(selected.created_at).toLocaleString()],
                      ].map(([k, v]) => (
                        <div key={String(k)} className="bg-[#F7F6F2] rounded-[8px] px-3 py-2">
                          <p className="text-[10px] text-muted uppercase tracking-[0.04em] mb-0.5">{k}</p>
                          <p className="text-[13px] font-medium text-ink">{String(v)}</p>
                        </div>
                      ))}
                    </div>

                    {/* Results JSON sections */}
                    {selected.results_json && (
                      <>
                        <JsonSection title="Metrics" data={selected.results_json.metrics} />
                        <JsonSection title="Governance" data={selected.results_json.governance} />
                        <JsonSection title="Policy" data={selected.results_json.policy} />
                        <JsonSection title="Calibration" data={selected.results_json.calibration} />
                        <JsonSection title="Leakage" data={selected.results_json.leakage} />
                        <JsonSection title="Top Drifted Features" data={selected.results_json.top_drifted_ranked?.slice(0, 5)} />
                        <JsonSection title="Advisories" data={selected.results_json.advisories} />
                      </>
                    )}
                  </>
                )}
              </div>
            )}
          </>
        )}

        {activeTab === 'compare' && (
          <div className="flex-1 p-8 space-y-6 overflow-y-auto">
            <div className="bg-white border border-stone rounded-card p-6">
              <h2 className="text-[14px] font-semibold text-ink mb-5">Compare two scans side-by-side</h2>
              <div className="grid md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Scan A (baseline)</label>
                  <select value={compareA} onChange={e => setCompareA(e.target.value)}
                    className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] bg-white outline-none focus:border-forest">
                    <option value="">Select scan…</option>
                    {scans.map(s => <option key={s.id} value={s.id}>{s.scan_type} — {s.id.slice(0, 8)} — {timeAgo(s.created_at)}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Scan B (current)</label>
                  <select value={compareB} onChange={e => setCompareB(e.target.value)}
                    className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] bg-white outline-none focus:border-forest">
                    <option value="">Select scan…</option>
                    {scans.map(s => <option key={s.id} value={s.id}>{s.scan_type} — {s.id.slice(0, 8)} — {timeAgo(s.created_at)}</option>)}
                  </select>
                </div>
              </div>
              {error && <p className="mb-4 text-[12px] text-danger">⚠ {error}</p>}
              <Button variant="primary" size="sm" className="gap-2" onClick={runCompare} disabled={!compareA || !compareB || comparing}>
                {comparing ? <><RefreshCw size={13} className="animate-spin" />Comparing…</> : <><GitCompare size={13} />Compare Scans</>}
              </Button>
            </div>

            {compareResult && (
              <>
                {/* Summary row */}
                <div className="grid md:grid-cols-2 gap-4">
                  {['scan_a', 'scan_b'].map((k, i) => {
                    const s = compareResult[k];
                    return (
                      <div key={k} className="bg-white border border-stone rounded-card p-5">
                        <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">Scan {i === 0 ? 'A (Baseline)' : 'B (Current)'}</p>
                        <p className="text-[24px] font-bold text-ink">{s?.score?.toFixed(0) ?? '—'}<span className="text-[13px] text-muted">/100</span></p>
                        <Badge variant={s?.gate === 'PASS' || s?.gate === 'PASSED' ? 'certified' : 'failed'} className="mt-2">{s?.gate ?? '—'}</Badge>
                        <p className="text-[11px] text-muted mt-2">{s?.created_at ? new Date(s.created_at).toLocaleString() : ''}</p>
                      </div>
                    );
                  })}
                </div>

                {/* Delta */}
                {compareResult.governance_delta != null && (
                  <div className="bg-white border border-stone rounded-card p-5">
                    <p className="text-[12px] font-medium text-ink-soft mb-1">Governance score delta (B − A)</p>
                    <p className={`text-[28px] font-bold ${compareResult.governance_delta >= 0 ? 'text-forest' : 'text-danger'}`}>
                      {compareResult.governance_delta >= 0 ? '+' : ''}{compareResult.governance_delta?.toFixed(2)}
                    </p>
                  </div>
                )}

                {/* Metrics comparison */}
                {compareResult.metrics_comparison && Object.keys(compareResult.metrics_comparison).length > 0 && (
                  <div className="bg-white border border-stone rounded-card p-6">
                    <h3 className="text-[14px] font-semibold text-ink mb-4">Metrics comparison</h3>
                    <table className="w-full border-collapse">
                      <thead>
                        <tr>{['Metric', 'Scan A', 'Scan B', 'Delta'].map(h => (
                          <th key={h} className="text-left pb-2 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted border-b border-stone">{h}</th>
                        ))}</tr>
                      </thead>
                      <tbody>
                        {Object.entries(compareResult.metrics_comparison).map(([metric, vals]: any) => (
                          <tr key={metric} className="border-b border-stone/50">
                            <td className="py-2.5 text-[13px] font-medium text-ink capitalize">{metric.replace(/_/g, ' ')}</td>
                            <td className="py-2.5 text-[13px] text-muted">{vals.scan_a?.toFixed(4) ?? '—'}</td>
                            <td className="py-2.5 text-[13px] text-muted">{vals.scan_b?.toFixed(4) ?? '—'}</td>
                            <td className={`py-2.5 text-[13px] font-medium ${vals.delta > 0 ? 'text-forest' : vals.delta < 0 ? 'text-danger' : 'text-muted'}`}>
                              {vals.delta != null ? `${vals.delta > 0 ? '+' : ''}${vals.delta.toFixed(4)}` : '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
