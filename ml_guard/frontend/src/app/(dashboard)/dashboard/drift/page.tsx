'use client';

import { useEffect, useState } from 'react';
import { RefreshCw, Activity, AlertTriangle, CheckCircle, Zap } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { modelsApi, driftApi, type ModelItem, type DriftReport, type DriftHistoryItem } from '@/lib/api';

function SeverityBadge({ severity }: { severity: string }) {
  const variant = severity === 'CRITICAL' ? 'failed' : severity === 'WARNING' ? 'conditional' : 'certified';
  return <Badge variant={variant}>{severity}</Badge>;
}

function FeatureRow({ feat }: { feat: DriftReport['feature_results'][0] }) {
  const pct = Math.min(100, (feat.score / 0.4) * 100);
  const color = feat.severity === 'CRITICAL' ? '#C0392B' : feat.severity === 'WARNING' ? '#B35A00' : '#1A5F3A';
  return (
    <tr className="border-b border-stone/50 hover:bg-[#F0EDE8] transition-colors">
      <td className="py-3 pr-4 text-[13px] font-medium text-ink">{feat.feature}</td>
      <td className="py-3 pr-4 text-[12px] font-mono text-muted">{feat.method.toUpperCase()}</td>
      <td className="py-3 pr-4">
        <div className="flex items-center gap-2">
          <div className="w-20 h-1.5 bg-stone rounded-full overflow-hidden">
            <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
          </div>
          <span className="text-[12px] font-mono text-ink">{feat.score.toFixed(4)}</span>
        </div>
      </td>
      <td className="py-3 pr-4"><SeverityBadge severity={feat.severity} /></td>
      <td className="py-3 text-[12px] text-muted">{feat.ref_count} / {feat.cur_count}</td>
    </tr>
  );
}

export default function DriftMonitorPage() {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [report, setReport] = useState<DriftReport | null>(null);
  const [history, setHistory] = useState<DriftHistoryItem[]>([]);
  const [loadingModels, setLoadingModels] = useState(true);
  const [loadingReport, setLoadingReport] = useState(false);
  const [triggering, setTriggering] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [method, setMethod] = useState<'ks' | 'psi'>('ks');

  useEffect(() => {
    modelsApi.list(1, 100)
      .then(res => {
        setModels(res.items ?? []);
        if (res.items?.length) setSelectedModel(res.items[0].model_id);
      })
      .catch(e => setError(e.message))
      .finally(() => setLoadingModels(false));
  }, []);

  useEffect(() => {
    if (!selectedModel) return;
    setLoadingReport(true);
    setError(null);
    Promise.all([
      driftApi.latestReport(selectedModel).catch(() => null),
      driftApi.history(selectedModel, 20).catch(() => []),
    ])
      .then(([rep, hist]) => {
        setReport(rep);
        setHistory(Array.isArray(hist) ? hist : []);
      })
      .catch(e => setError(e.message))
      .finally(() => setLoadingReport(false));
  }, [selectedModel]);

  const triggerScan = async () => {
    if (!selectedModel) return;
    setTriggering(true);
    setError(null);
    try {
      await driftApi.trigger(selectedModel, method);
      // Reload report
      const [rep, hist] = await Promise.all([
        driftApi.latestReport(selectedModel).catch(() => null),
        driftApi.history(selectedModel, 20).catch(() => []),
      ]);
      setReport(rep);
      setHistory(Array.isArray(hist) ? hist : []);
    } catch (e: any) {
      setError(e.message ?? 'Drift scan failed');
    } finally {
      setTriggering(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Drift Monitor</h1>
          <p className="text-[11px] text-muted">Real-time feature & embedding drift detection</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={method}
            onChange={e => setMethod(e.target.value as 'ks' | 'psi')}
            className="h-9 px-3 text-[13px] border border-stone rounded-[8px] bg-white text-ink outline-none focus:border-forest"
          >
            <option value="ks">KS Test</option>
            <option value="psi">PSI</option>
          </select>
          <Button variant="primary" size="sm" className="gap-1.5" onClick={triggerScan} disabled={!selectedModel || triggering}>
            {triggering
              ? <><RefreshCw size={13} strokeWidth={1.5} className="animate-spin" /> Scanning…</>
              : <><Zap size={13} strokeWidth={1.5} /> Run scan</>}
          </Button>
        </div>
      </div>

      <div className="flex-1 p-8 space-y-6">
        {/* Model selector */}
        <div className="flex items-center gap-3">
          <label className="text-[12px] font-medium text-ink-soft">Model:</label>
          {loadingModels ? (
            <div className="h-9 w-48 bg-stone rounded-[8px] animate-pulse" />
          ) : (
            <select
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
              className="h-9 px-3 text-[13px] border border-stone rounded-[8px] bg-white text-ink outline-none focus:border-forest"
            >
              {models.length === 0
                ? <option value="">No models registered</option>
                : models.map(m => <option key={m.model_id} value={m.model_id}>{m.name}</option>)
              }
            </select>
          )}
        </div>

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger">⚠ {error}</div>
        )}

        {/* Report summary */}
        {loadingReport ? (
          <div className="grid grid-cols-4 gap-4">
            {[1,2,3,4].map(i => <div key={i} className="bg-white border border-stone rounded-card p-5 h-24 animate-pulse" />)}
          </div>
        ) : !report ? (
          <div className="bg-white border border-stone rounded-card p-10 text-center">
            <Activity size={32} className="mx-auto text-stone mb-3" strokeWidth={1} />
            <p className="text-[14px] font-medium text-ink mb-1">No drift reports yet</p>
            <p className="text-[13px] text-muted mb-4">Ingest predictions or run a manual scan to get started.</p>
            <Button variant="primary" size="sm" onClick={triggerScan} disabled={triggering || !selectedModel}>
              Run first scan
            </Button>
          </div>
        ) : (
          <>
            {/* Score cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-white border border-stone rounded-card p-5">
                <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">Overall Score</p>
                <p className="text-[28px] font-bold leading-none" style={{ letterSpacing: '-0.03em', color: report.overall_drift_score >= 0.3 ? '#C0392B' : report.overall_drift_score >= 0.2 ? '#B35A00' : '#1A5F3A' }}>
                  {report.overall_drift_score.toFixed(4)}
                </p>
              </div>
              <div className="bg-white border border-stone rounded-card p-5">
                <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">Drift Detected</p>
                <div className="flex items-center gap-2 mt-2">
                  {report.drift_detected
                    ? <><AlertTriangle size={20} className="text-danger" /><span className="text-[16px] font-semibold text-danger">YES</span></>
                    : <><CheckCircle size={20} className="text-forest" /><span className="text-[16px] font-semibold text-forest">NO</span></>}
                </div>
              </div>
              <div className="bg-white border border-stone rounded-card p-5">
                <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">Features Analyzed</p>
                <p className="text-[28px] font-bold text-ink leading-none">{report.feature_results?.length ?? 0}</p>
              </div>
              <div className="bg-white border border-stone rounded-card p-5">
                <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">Sample Count</p>
                <p className="text-[28px] font-bold text-ink leading-none">{report.sample_count?.toLocaleString() ?? '—'}</p>
              </div>
            </div>

            {/* Feature results table */}
            {report.feature_results && report.feature_results.length > 0 && (
              <div className="bg-white border border-stone rounded-card p-6">
                <h2 className="text-[14px] font-semibold text-ink mb-5">Per-feature drift results</h2>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr>
                        {['Feature', 'Method', 'Score', 'Severity', 'Ref / Cur count'].map(h => (
                          <th key={h} className="text-left pb-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted border-b border-stone">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {report.feature_results.map(f => <FeatureRow key={f.feature} feat={f} />)}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* History sparkline */}
            {history.length > 1 && (
              <div className="bg-white border border-stone rounded-card p-6">
                <h2 className="text-[14px] font-semibold text-ink mb-5">Drift score history</h2>
                <div className="flex items-end gap-1.5 h-[80px]">
                  {history.slice().reverse().map((h, i) => {
                    const pct = Math.min(100, (h.overall_drift_score / 0.5) * 100);
                    const color = h.drift_detected ? '#C0392B' : '#1A5F3A';
                    return (
                      <div key={h.id} className="flex-1 flex flex-col items-center justify-end gap-1 group relative">
                        <div className="w-full rounded-t-sm" style={{ height: `${Math.max(4, pct)}%`, background: color, opacity: 0.8 }} />
                        <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 bg-ink text-white text-[10px] px-2 py-1 rounded whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none z-10">
                          {h.overall_drift_score.toFixed(4)} — {new Date(h.created_at).toLocaleDateString()}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <p className="text-[11px] text-muted mt-2">Last {history.length} scan{history.length !== 1 ? 's' : ''} · Red = drift detected</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
