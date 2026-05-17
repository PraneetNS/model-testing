'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { Download, RefreshCw, ChevronLeft, Zap, AlertTriangle } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { modelsApi, driftApi, contractsApi, governanceApi, reportsApi, predictionsApi, inventoryApi, type ModelDetail, type ModelVersion, type DriftReport, type Contract, type GovernanceScore } from '@/lib/api';

const TABS = ['Overview', 'Versions', 'Drift', 'Contracts', 'Governance', 'Predictions', 'AIBOM'];

function ScoreBadge({ score }: { score: number | null | undefined }) {
  if (score == null) return <span className="text-[12px] text-muted">Not audited</span>;
  const v: 'certified' | 'conditional' | 'failed' = score >= 80 ? 'certified' : score >= 60 ? 'conditional' : 'failed';
  return <Badge variant={v}>{score.toFixed(0)} / 100</Badge>;
}

function DimBar({ label, score }: { label: string; score: number }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[13px] text-ink-soft">{label}</span>
        <span className="text-[13px] font-semibold text-ink">{score}</span>
      </div>
      <div className="h-1.5 bg-stone rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${score}%`, background: score >= 80 ? '#1A5F3A' : score >= 60 ? '#B35A00' : '#C0392B' }} />
      </div>
    </div>
  );
}

export default function ModelDetailPage() {
  const params = useParams();
  const router = useRouter();
  const modelId = params?.id as string;

  const [model, setModel] = useState<ModelDetail | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [drift, setDrift] = useState<DriftReport | null>(null);
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [governance, setGovernance] = useState<GovernanceScore | null>(null);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [aibom, setAibom] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('Overview');
  const [generatingReport, setGeneratingReport] = useState(false);
  const [triggeringAudit, setTriggeringAudit] = useState(false);

  useEffect(() => {
    if (!modelId) return;
    setLoading(true);
    setError(null);

    Promise.all([
      modelsApi.get(modelId),
      modelsApi.versions(modelId).catch(() => ({ versions: [] })),
      driftApi.latestReport(modelId).catch(() => null),
      contractsApi.list(modelId).catch(() => ({ items: [], total: 0 })),
      governanceApi.score(modelId).catch(() => null),
      predictionsApi.list(modelId, 20).catch(() => ({ items: [], total: 0 })),
      inventoryApi.aibom(modelId).catch(() => ({ components: [] })),
    ])
      .then(([m, v, d, c, g, preds, aibomRes]) => {
        setModel(m);
        setVersions(v.versions ?? []);
        setDrift(d);
        setContracts(c.items ?? []);
        setGovernance(g);
        setPredictions(preds.items ?? []);
        setAibom((aibomRes as any).components ?? []);
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [modelId]);

  const runAudit = async () => {
    setTriggeringAudit(true);
    try {
      await governanceApi.runAudit(modelId);
      // Refresh governance score
      const g = await governanceApi.score(modelId).catch(() => null);
      setGovernance(g);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setTriggeringAudit(false);
    }
  };

  const downloadReport = async () => {
    setGeneratingReport(true);
    try {
      await reportsApi.generatePdf(modelId);
      alert('Report generation started. Check /dashboard/reports for download link.');
    } catch (e: any) {
      setError(e.message);
    } finally {
      setGeneratingReport(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col min-h-screen">
        <div className="flex items-center px-8 h-16 border-b border-stone bg-white gap-4">
          <div className="h-5 w-48 bg-stone rounded-full animate-pulse" />
        </div>
        <div className="p-8 space-y-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-24 bg-white border border-stone rounded-card animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !model) {
    return (
      <div className="flex flex-col min-h-screen items-center justify-center p-8">
        <AlertTriangle size={32} className="text-danger mb-3" />
        <p className="text-[15px] font-semibold text-ink mb-2">{error ?? 'Model not found'}</p>
        <Button variant="ghost" size="sm" onClick={() => router.push('/dashboard/models')}>
          <ChevronLeft size={14} /> Back to registry
        </Button>
      </div>
    );
  }

  const verdict = model.governance_score != null
    ? (model.governance_score >= 80 ? 'CERTIFIED' : model.governance_score >= 60 ? 'CONDITIONAL' : 'FAILED')
    : null;

  return (
    <div className="flex flex-col min-h-screen">
      {/* Top bar */}
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div className="flex items-center gap-3">
          <Link href="/dashboard/models" className="text-muted hover:text-ink transition-colors">
            <ChevronLeft size={18} strokeWidth={1.5} />
          </Link>
          <div>
            <h1 className="text-[17px] font-semibold text-ink flex items-center gap-2">
              {model.name}
              <span className="font-mono text-[13px] text-muted">v{model.latest_version}</span>
            </h1>
            <p className="text-[11px] text-muted">Dashboard / Models / {model.name}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {verdict && <Badge variant={verdict.toLowerCase() as any}>{verdict}</Badge>}
          <Button variant="ghost" size="sm" className="gap-1.5" onClick={runAudit} disabled={triggeringAudit}>
            {triggeringAudit ? <RefreshCw size={13} strokeWidth={1.5} className="animate-spin" /> : <Zap size={13} strokeWidth={1.5} />}
            {triggeringAudit ? 'Auditing…' : 'Run audit'}
          </Button>
          <Button variant="ghost" size="sm" className="gap-1.5" onClick={downloadReport} disabled={generatingReport}>
            <Download size={13} strokeWidth={1.5} />
            Certificate
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex px-8 bg-white border-b border-stone gap-6">
        {TABS.map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`py-3 text-[13px] font-medium border-b-2 transition-colors -mb-px ${
              activeTab === tab ? 'border-forest text-forest' : 'border-transparent text-muted hover:text-ink'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {error && (
        <div className="mx-8 mt-4 p-3 bg-red-50 border border-red-200 rounded-card text-[12px] text-danger">⚠ {error}</div>
      )}

      <div className="flex-1 p-8 overflow-auto">
        {/* OVERVIEW TAB */}
        {activeTab === 'Overview' && (
          <div className="grid lg:grid-cols-2 gap-6">
            <div className="bg-white border border-stone rounded-card p-6">
              <h3 className="text-[14px] font-semibold text-ink mb-5">Model details</h3>
              <dl className="grid grid-cols-2 gap-4">
                {[
                  ['Name', model.name],
                  ['Provider / Owner', model.provider || '—'],
                  ['Risk tier', model.risk_tier || '—'],
                  ['Environment', model.deployment_environment || '—'],
                  ['Business owner', model.business_owner || '—'],
                  ['Technical owner', model.technical_owner || '—'],
                  ['Versions', model.version_count],
                  ['Registered', new Date(model.created_at).toLocaleDateString()],
                ].map(([k, v]) => (
                  <div key={String(k)}>
                    <dt className="text-[11px] text-muted uppercase tracking-[0.04em] mb-0.5">{k}</dt>
                    <dd className="text-[14px] font-medium text-ink">{String(v)}</dd>
                  </div>
                ))}
              </dl>
            </div>
            <div className="bg-white border border-stone rounded-card p-6">
              <h3 className="text-[14px] font-semibold text-ink mb-5">Governance score</h3>
              {governance ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between mb-4">
                    <ScoreBadge score={governance.overall_score} />
                    <Badge variant={governance.verdict?.toLowerCase() === 'certified' ? 'certified' : governance.verdict?.toLowerCase() === 'conditional' ? 'conditional' : 'failed'}>
                      {governance.verdict}
                    </Badge>
                  </div>
                  {Object.entries(governance.dimension_scores ?? {}).map(([dim, score]) => (
                    <DimBar key={dim} label={dim} score={typeof score === 'number' ? score : 0} />
                  ))}
                </div>
              ) : (
                <div className="text-center py-6">
                  <p className="text-[13px] text-muted mb-3">No audit results yet.</p>
                  <Button variant="primary" size="sm" onClick={runAudit} disabled={triggeringAudit}>
                    <Zap size={12} strokeWidth={1.5} /> Run audit now
                  </Button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* VERSIONS TAB */}
        {activeTab === 'Versions' && (
          <div className="bg-white border border-stone rounded-card overflow-hidden">
            <table className="w-full border-collapse">
              <thead className="bg-[#F7F6F2]">
                <tr>
                  {['Version', 'Framework', 'Governance score', 'Risk class', 'Deployments', 'Created'].map(h => (
                    <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted border-b border-stone">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {versions.length === 0
                  ? <tr><td colSpan={6} className="py-10 text-center text-[13px] text-muted">No versions yet.</td></tr>
                  : versions.map(v => (
                    <tr key={v.version_id} className="border-b border-stone/50 hover:bg-[#F0EDE8] transition-colors">
                      <td className="px-5 py-3.5 text-[13px] font-mono text-ink">v{v.version_number}</td>
                      <td className="px-5 py-3.5 text-[13px] text-muted">{v.framework || '—'}</td>
                      <td className="px-5 py-3.5"><ScoreBadge score={v.governance_score} /></td>
                      <td className="px-5 py-3.5">
                        {v.risk_class ? <span className="text-[11px] font-medium px-2 py-0.5 rounded-badge bg-stone text-ink-soft">{v.risk_class}</span> : '—'}
                      </td>
                      <td className="px-5 py-3.5">
                        {v.deployments.length > 0
                          ? v.deployments.map(d => (
                            <span key={d.environment} className="text-[11px] mr-1.5 px-1.5 py-0.5 rounded bg-mist text-forest font-medium">{d.environment}</span>
                          ))
                          : <span className="text-[12px] text-muted">Not deployed</span>}
                      </td>
                      <td className="px-5 py-3.5 text-[12px] text-muted">{new Date(v.created_at).toLocaleDateString()}</td>
                    </tr>
                  ))
                }
              </tbody>
            </table>
          </div>
        )}

        {/* DRIFT TAB */}
        {activeTab === 'Drift' && (
          drift ? (
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-white border border-stone rounded-card p-5">
                  <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">Overall Drift Score</p>
                  <p className="text-[24px] font-bold" style={{ color: drift.drift_detected ? '#C0392B' : '#1A5F3A' }}>{drift.overall_drift_score.toFixed(4)}</p>
                </div>
                <div className="bg-white border border-stone rounded-card p-5">
                  <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">Status</p>
                  <Badge variant={drift.drift_detected ? 'failed' : 'certified'}>{drift.drift_detected ? 'DRIFT DETECTED' : 'STABLE'}</Badge>
                </div>
                <div className="bg-white border border-stone rounded-card p-5">
                  <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">Sample Count</p>
                  <p className="text-[24px] font-bold text-ink">{drift.sample_count?.toLocaleString() ?? '—'}</p>
                </div>
              </div>
              {drift.feature_results?.length > 0 && (
                <div className="bg-white border border-stone rounded-card p-6">
                  <h3 className="text-[14px] font-semibold text-ink mb-4">Per-feature drift</h3>
                  {drift.feature_results.map(f => (
                    <div key={f.feature} className="flex items-center gap-4 mb-3 pb-3 border-b border-stone/50 last:border-0">
                      <span className="text-[13px] text-ink w-40 truncate">{f.feature}</span>
                      <div className="flex-1 h-1.5 bg-stone rounded-full overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${Math.min(100, (f.score / 0.4) * 100)}%`, background: f.severity === 'CRITICAL' ? '#C0392B' : f.severity === 'WARNING' ? '#B35A00' : '#1A5F3A' }} />
                      </div>
                      <span className="text-[12px] font-mono text-muted w-16 text-right">{f.score.toFixed(4)}</span>
                      <Badge variant={f.severity === 'CRITICAL' ? 'failed' : f.severity === 'WARNING' ? 'conditional' : 'certified'}>{f.severity}</Badge>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="bg-white border border-stone rounded-card p-10 text-center">
              <p className="text-[13px] text-muted">No drift report available for this model yet.</p>
              <Link href="/dashboard/drift" className="text-[13px] text-forest underline mt-2 block">Go to Drift Monitor →</Link>
            </div>
          )
        )}

        {/* CONTRACTS TAB */}
        {activeTab === 'Contracts' && (
          <div className="bg-white border border-stone rounded-card overflow-hidden">
            <table className="w-full border-collapse">
              <thead className="bg-[#F7F6F2]">
                <tr>
                  {['Contract', 'Type', 'Status', 'Breach rate'].map(h => (
                    <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted border-b border-stone">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {contracts.length === 0
                  ? <tr><td colSpan={4} className="py-10 text-center text-[13px] text-muted">No contracts for this model.</td></tr>
                  : contracts.map(c => (
                    <tr key={c.id} className="border-b border-stone/50 hover:bg-[#F0EDE8] transition-colors">
                      <td className="px-5 py-3.5 text-[13px] font-medium text-ink">{c.name}</td>
                      <td className="px-5 py-3.5"><span className="text-[11px] px-2 py-0.5 rounded-badge bg-stone text-ink-soft">{c.contract_type}</span></td>
                      <td className="px-5 py-3.5">
                        <Badge variant={c.status === 'CERTIFIED' ? 'certified' : c.status === 'CONDITIONAL' ? 'conditional' : 'failed'}>{c.status}</Badge>
                      </td>
                      <td className="px-5 py-3.5 text-[13px] text-muted">{c.breach_rate !== undefined ? `${(c.breach_rate * 100).toFixed(2)}%` : '—'}</td>
                    </tr>
                  ))
                }
              </tbody>
            </table>
          </div>
        )}

        {/* GOVERNANCE TAB */}
        {activeTab === 'Governance' && (
          governance ? (
            <div className="grid lg:grid-cols-2 gap-6">
              <div className="bg-white border border-stone rounded-card p-6 space-y-4">
                <h3 className="text-[14px] font-semibold text-ink">Dimension scores</h3>
                {Object.entries(governance.dimension_scores ?? {}).map(([dim, score]) => (
                  <DimBar key={dim} label={dim} score={typeof score === 'number' ? score : 0} />
                ))}
              </div>
              <div className="bg-white border border-stone rounded-card p-6">
                <h3 className="text-[14px] font-semibold text-ink mb-5">Audit summary</h3>
                <dl className="grid grid-cols-2 gap-4">
                  {[
                    ['Overall score', `${governance.overall_score?.toFixed(0) ?? '—'} / 100`],
                    ['Verdict', governance.verdict ?? '—'],
                    ['Computed at', governance.computed_at ? new Date(governance.computed_at).toLocaleString() : '—'],
                  ].map(([k, v]) => (
                    <div key={String(k)}>
                      <dt className="text-[11px] text-muted uppercase tracking-[0.04em] mb-0.5">{k}</dt>
                      <dd className="text-[14px] font-medium text-ink">{String(v)}</dd>
                    </div>
                  ))}
                </dl>
                <div className="mt-6">
                  <Button variant="primary" size="sm" onClick={runAudit} disabled={triggeringAudit}>
                    {triggeringAudit ? <><RefreshCw size={12} className="animate-spin" /> Re-auditing…</> : <><Zap size={12} /> Re-run audit</>}
                  </Button>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white border border-stone rounded-card p-10 text-center">
              <p className="text-[13px] text-muted mb-3">No governance audit has been run yet.</p>
              <Button variant="primary" size="sm" onClick={runAudit} disabled={triggeringAudit}>Run audit</Button>
            </div>
          )
        )}

        {/* PREDICTIONS TAB */}
        {activeTab === 'Predictions' && (
          <div className="bg-white border border-stone rounded-card overflow-hidden">
            <table className="w-full border-collapse">
              <thead className="bg-[#F7F6F2]">
                <tr>
                  {['Prediction', 'Confidence', 'Latency', 'Timestamp'].map(h => (
                    <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted border-b border-stone">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {predictions.length === 0
                  ? <tr><td colSpan={4} className="py-10 text-center text-[13px] text-muted">No prediction logs for this model.</td></tr>
                  : predictions.map((p, i) => (
                    <tr key={p.id ?? i} className="border-b border-stone/50 hover:bg-[#F0EDE8] transition-colors">
                      <td className="px-5 py-3.5 text-[13px] font-medium text-ink">{String(p.prediction)}</td>
                      <td className="px-5 py-3.5 text-[13px] text-muted">{p.confidence != null ? `${(p.confidence * 100).toFixed(1)}%` : '—'}</td>
                      <td className="px-5 py-3.5 text-[13px] text-muted">{p.latency_ms != null ? `${p.latency_ms}ms` : '—'}</td>
                      <td className="px-5 py-3.5 text-[12px] text-muted">{new Date(p.timestamp).toLocaleString()}</td>
                    </tr>
                  ))
                }
              </tbody>
            </table>
          </div>
        )}

        {/* AIBOM TAB */}
        {activeTab === 'AIBOM' && (
          <div className="bg-white border border-stone rounded-card overflow-hidden">
            <div className="px-6 py-4 border-b border-stone bg-[#F7F6F2]">
              <h3 className="text-[14px] font-semibold text-ink">AI Bill of Materials</h3>
              <p className="text-[12px] text-muted mt-0.5">All components, dependencies, and datasets with integrity hashes</p>
            </div>
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  {['Component', 'Type', 'Version', 'SHA-256', 'CVEs'].map(h => (
                    <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted border-b border-stone">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {aibom.length === 0
                  ? <tr><td colSpan={5} className="py-10 text-center text-[13px] text-muted">AIBOM not yet generated for this model.</td></tr>
                  : aibom.map((item: any, i: number) => (
                    <tr key={i} className="border-b border-stone/50 hover:bg-[#F0EDE8] transition-colors">
                      <td className="px-5 py-3.5 text-[13px] font-medium text-ink">{item.name}</td>
                      <td className="px-5 py-3.5"><span className="text-[11px] px-2 py-0.5 rounded-badge bg-stone text-ink-soft">{item.type}</span></td>
                      <td className="px-5 py-3.5 text-[13px] font-mono text-muted">{item.version || '—'}</td>
                      <td className="px-5 py-3.5 text-[11px] font-mono text-muted">{item.hash ? item.hash.slice(0, 16) + '…' : '—'}</td>
                      <td className="px-5 py-3.5">
                        {item.cves === 0 || item.cves === null
                          ? <span className="text-[12px] text-forest">0 CVEs</span>
                          : <Badge variant="failed">{item.cves} CVE{item.cves > 1 ? 's' : ''}</Badge>}
                      </td>
                    </tr>
                  ))
                }
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
