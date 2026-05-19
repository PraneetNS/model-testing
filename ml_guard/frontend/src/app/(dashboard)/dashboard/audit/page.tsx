'use client';

import { useState, useRef } from 'react';
import { Upload, Play, RefreshCw, CheckCircle, XCircle, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';

const BASE_URL = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000') + '/api/v1';
const token = () => (typeof window !== 'undefined' ? localStorage.getItem('niyantrana_token') : null);

const ALL_CHECKS = ['drift', 'performance', 'fairness', 'security', 'explainability', 'calibration', 'leakage'];

function FileDropZone({ label, accept, file, onChange }: {
  label: string; accept: string; file: File | null;
  onChange: (f: File | null) => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <div
      onClick={() => ref.current?.click()}
      className={`relative border-2 border-dashed rounded-card p-5 cursor-pointer transition-colors text-center ${
        file ? 'border-forest bg-mist' : 'border-stone hover:border-forest/60'
      }`}
    >
      <input ref={ref} type="file" accept={accept} className="hidden" onChange={e => onChange(e.target.files?.[0] ?? null)} />
      <Upload size={20} className={`mx-auto mb-2 ${file ? 'text-forest' : 'text-muted'}`} strokeWidth={1.5} />
      <p className="text-[13px] font-medium text-ink">{file ? file.name : label}</p>
      <p className="text-[11px] text-muted mt-0.5">{file ? `${(file.size / 1024).toFixed(1)} KB` : accept}</p>
      {file && (
        <button onClick={e => { e.stopPropagation(); onChange(null); }}
          className="absolute top-2 right-2 text-muted hover:text-danger text-[10px]">✕</button>
      )}
    </div>
  );
}

function ScoreGauge({ score, label }: { score: number; label: string }) {
  const color = score >= 75 ? '#1A5F3A' : score >= 50 ? '#B35A00' : '#C0392B';
  const r = 52, circ = 2 * Math.PI * r;
  const offset = circ - (score / 100) * circ;
  return (
    <div className="flex flex-col items-center">
      <svg width="130" height="130" viewBox="0 0 130 130">
        <circle cx="65" cy="65" r={r} fill="none" stroke="#E8E4DE" strokeWidth="10" />
        <circle cx="65" cy="65" r={r} fill="none" stroke={color} strokeWidth="10"
          strokeDasharray={circ} strokeDashoffset={offset}
          strokeLinecap="round" transform="rotate(-90 65 65)"
          style={{ transition: 'stroke-dashoffset 0.8s ease' }} />
        <text x="65" y="60" textAnchor="middle" fontSize="22" fontWeight="700" fill={color}>{score.toFixed(0)}</text>
        <text x="65" y="78" textAnchor="middle" fontSize="10" fill="#888">/100</text>
      </svg>
      <p className="text-[12px] font-semibold text-ink mt-1">{label}</p>
    </div>
  );
}

function MetricPill({ label, value, good }: { label: string; value: string; good?: boolean }) {
  return (
    <div className="bg-white border border-stone rounded-card px-4 py-3 flex flex-col items-center">
      <span className="text-[11px] text-muted uppercase tracking-[0.04em] mb-1">{label}</span>
      <span className={`text-[18px] font-bold ${good === false ? 'text-danger' : good ? 'text-forest' : 'text-ink'}`}>{value}</span>
    </div>
  );
}

function CollapsibleSection({ title, badge, children }: { title: string; badge?: React.ReactNode; children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="bg-white border border-stone rounded-card overflow-hidden">
      <button onClick={() => setOpen(o => !o)}
        className="flex items-center justify-between w-full px-6 py-4 hover:bg-[#F7F6F2] transition-colors">
        <div className="flex items-center gap-3">
          <h3 className="text-[14px] font-semibold text-ink">{title}</h3>
          {badge}
        </div>
        {open ? <ChevronUp size={16} className="text-muted" /> : <ChevronDown size={16} className="text-muted" />}
      </button>
      {open && <div className="px-6 pb-6 border-t border-stone/50 pt-4">{children}</div>}
    </div>
  );
}

export default function ModelAuditPage() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [trainFile, setTrainFile] = useState<File | null>(null);
  const [valFile, setValFile] = useState<File | null>(null);
  const [modelName, setModelName] = useState('');
  const [labelCol, setLabelCol] = useState('target');
  const [checks, setChecks] = useState<string[]>(ALL_CHECKS);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState('');

  const toggleCheck = (c: string) =>
    setChecks(prev => prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c]);

  const runAudit = async () => {
    if (!modelFile) { setError('Model file (.pkl or .onnx) is required.'); return; }
    if (!trainFile && !valFile) { setError('At least one dataset (training or validation) is required.'); return; }

    setRunning(true); setError(null); setResult(null);
    setProgress('Uploading files…');

    const fd = new FormData();
    fd.append('model_file', modelFile);
    if (trainFile) fd.append('train_file', trainFile);
    if (valFile) fd.append('val_file', valFile);
    fd.append('model_name', modelName || modelFile.name.replace(/\.[^.]+$/, ''));
    fd.append('label_col', labelCol);
    checks.forEach(c => fd.append('selected', c));

    try {
      setProgress('Running audit (this may take 30–60s)…');
      const res = await fetch(`${BASE_URL}/audit/run`, {
        method: 'POST',
        headers: token() ? { Authorization: `Bearer ${token()}` } : {},
        body: fd,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
      setProgress('');
    } catch (e: any) {
      setError(e.message ?? 'Audit failed');
      setProgress('');
    } finally {
      setRunning(false);
    }
  };

  const gateStatus: 'PASS' | 'BLOCK' | 'WARN' | null = result?.policy?.gate_status ?? null;
  const govScore: number = result?.governance?.governance_score ?? 0;
  const verdict = govScore >= 75 ? 'CERTIFIED' : govScore >= 50 ? 'CONDITIONAL' : 'FAILED';

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Model Audit</h1>
          <p className="text-[11px] text-muted">Upload model + datasets → full governance audit</p>
        </div>
        {result && (
          <Badge variant={gateStatus === 'PASS' ? 'certified' : gateStatus === 'BLOCK' ? 'failed' : 'conditional'}>
            {gateStatus ?? verdict}
          </Badge>
        )}
      </div>

      <div className="flex-1 p-8 space-y-6">
        {/* Upload Form */}
        <div className="bg-white border border-stone rounded-card p-6">
          <h2 className="text-[14px] font-semibold text-ink mb-5">Upload artifacts</h2>
          <div className="grid md:grid-cols-3 gap-4 mb-5">
            <FileDropZone label="Model File (.pkl / .onnx)" accept=".pkl,.joblib,.onnx" file={modelFile} onChange={setModelFile} />
            <FileDropZone label="Training Dataset (.csv)" accept=".csv" file={trainFile} onChange={setTrainFile} />
            <FileDropZone label="Validation Dataset (.csv)" accept=".csv" file={valFile} onChange={setValFile} />
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-5">
            <div>
              <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Model name (optional)</label>
              <input value={modelName} onChange={e => setModelName(e.target.value)}
                placeholder="e.g. credit-risk-v4"
                className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
            </div>
            <div>
              <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Label column</label>
              <input value={labelCol} onChange={e => setLabelCol(e.target.value)}
                placeholder="target"
                className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
            </div>
          </div>

          <div className="mb-5">
            <label className="block text-[12px] font-medium text-ink-soft mb-2">Checks to run</label>
            <div className="flex flex-wrap gap-2">
              {ALL_CHECKS.map(c => (
                <button key={c} onClick={() => toggleCheck(c)}
                  className={`px-3 py-1 text-[12px] font-medium rounded-badge border transition-colors capitalize ${
                    checks.includes(c) ? 'bg-forest text-white border-forest' : 'bg-white text-ink-soft border-stone hover:border-forest'
                  }`}>{c}
                </button>
              ))}
            </div>
          </div>

          {error && <p className="mb-4 text-[12px] text-danger bg-red-50 border border-red-200 rounded-[8px] px-4 py-2">⚠ {error}</p>}

          <Button variant="primary" size="sm" className="gap-2" onClick={runAudit} disabled={running}>
            {running
              ? <><RefreshCw size={13} strokeWidth={1.5} className="animate-spin" />{progress}</>
              : <><Play size={13} strokeWidth={1.5} />Run Full Audit</>}
          </Button>
        </div>

        {/* Results */}
        {result && (
          <>
            {/* Score gauges */}
            <div className="bg-white border border-stone rounded-card p-6">
              <h2 className="text-[14px] font-semibold text-ink mb-6">Audit results</h2>
              <div className="flex flex-wrap gap-8 justify-center mb-6">
                <ScoreGauge score={govScore} label="Governance Score" />
                {result.risk_score != null && <ScoreGauge score={100 - result.risk_score} label="Safety Score" />}
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {result.metrics?.accuracy != null && (
                  <MetricPill label="Val Accuracy" value={`${(result.metrics.accuracy * 100).toFixed(1)}%`} good={result.metrics.accuracy > 0.8} />
                )}
                {result.metrics?.f1 != null && (
                  <MetricPill label="F1 Score" value={result.metrics.f1.toFixed(3)} good={result.metrics.f1 > 0.7} />
                )}
                {result.overfitting_gap?.accuracy_gap != null && (
                  <MetricPill label="Overfitting Gap" value={`${(result.overfitting_gap.accuracy_gap * 100).toFixed(1)}%`} good={result.overfitting_gap.accuracy_gap < 0.05} />
                )}
                {result.risk_level && (
                  <MetricPill label="Risk Level" value={result.risk_level} good={result.risk_level === 'LOW'} />
                )}
              </div>
            </div>

            {/* Gate Status */}
            {result.policy && (
              <CollapsibleSection title="Policy Gate" badge={
                <Badge variant={result.policy.gate_status === 'PASS' ? 'certified' : result.policy.gate_status === 'BLOCK' ? 'failed' : 'conditional'}>
                  {result.policy.gate_status}
                </Badge>
              }>
                <div className="grid md:grid-cols-3 gap-4 text-[13px]">
                  {Object.entries(result.policy).map(([k, v]) => (
                    k !== 'gate_status' && (
                      <div key={k} className="flex justify-between border-b border-stone/50 pb-2">
                        <span className="text-muted capitalize">{k.replace(/_/g, ' ')}</span>
                        <span className="font-medium text-ink">{String(v)}</span>
                      </div>
                    )
                  ))}
                </div>
              </CollapsibleSection>
            )}

            {/* Drift Results */}
            {result.top_drifted_ranked?.length > 0 && (
              <CollapsibleSection title="Feature Drift" badge={
                <span className="text-[11px] text-muted">{result.top_drifted_ranked.filter((f: any) => f.severity !== 'OK').length} drifted features</span>
              }>
                <div className="space-y-3">
                  {result.top_drifted_ranked.slice(0, 10).map((f: any) => (
                    <div key={f.feature} className="flex items-center gap-4">
                      <span className="text-[13px] text-ink w-40 truncate">{f.feature}</span>
                      <div className="flex-1 h-2 bg-stone rounded-full overflow-hidden">
                        <div className="h-full rounded-full transition-all" style={{
                          width: `${Math.min(100, (f.psi / 0.5) * 100)}%`,
                          background: f.severity === 'CRITICAL' ? '#C0392B' : f.severity === 'WARNING' ? '#B35A00' : '#1A5F3A'
                        }} />
                      </div>
                      <span className="text-[12px] font-mono text-muted w-16 text-right">PSI {f.psi.toFixed(4)}</span>
                      <Badge variant={f.severity === 'CRITICAL' ? 'failed' : f.severity === 'WARNING' ? 'conditional' : 'certified'}>{f.severity}</Badge>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}

            {/* Calibration */}
            {result.calibration && Object.keys(result.calibration).length > 0 && (
              <CollapsibleSection title="Calibration">
                <div className="grid md:grid-cols-3 gap-4">
                  {Object.entries(result.calibration).map(([k, v]) => (
                    <div key={k} className="flex justify-between border-b border-stone/50 pb-2 text-[13px]">
                      <span className="text-muted capitalize">{k.replace(/_/g, ' ')}</span>
                      <span className="font-medium text-ink">{typeof v === 'boolean' ? (v ? 'Yes' : 'No') : String(v)}</span>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}

            {/* Leakage */}
            {result.leakage && Object.keys(result.leakage).length > 0 && (
              <CollapsibleSection title="Data Leakage Detection">
                <div className="grid md:grid-cols-2 gap-4">
                  {Object.entries(result.leakage).map(([k, v]) => (
                    <div key={k} className="flex justify-between border-b border-stone/50 pb-2 text-[13px]">
                      <span className="text-muted capitalize">{k.replace(/_/g, ' ')}</span>
                      <span className="font-medium text-ink">{typeof v === 'boolean' ? (v ? '⚠ Detected' : '✓ Clean') : String(v)}</span>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}

            {/* Advisories */}
            {result.advisories?.length > 0 && (
              <CollapsibleSection title="Advisories" badge={
                <span className="text-[11px] px-2 py-0.5 rounded-badge bg-amber-50 text-amber-700 font-medium">{result.advisories.length} items</span>
              }>
                <div className="space-y-3">
                  {result.advisories.map((a: any, i: number) => (
                    <div key={i} className="flex items-start gap-3 pb-3 border-b border-stone/50 last:border-0">
                      {a.severity === 'HIGH' || a.severity === 'CRITICAL'
                        ? <XCircle size={15} className="text-danger flex-shrink-0 mt-0.5" />
                        : a.severity === 'MEDIUM'
                        ? <AlertTriangle size={15} className="text-warning flex-shrink-0 mt-0.5" />
                        : <CheckCircle size={15} className="text-forest flex-shrink-0 mt-0.5" />}
                      <div>
                        <p className="text-[13px] font-medium text-ink">{a.message ?? a.title ?? String(a)}</p>
                        {a.recommendation && <p className="text-[12px] text-muted mt-0.5">{a.recommendation}</p>}
                      </div>
                    </div>
                  ))}
                </div>
              </CollapsibleSection>
            )}

            {/* Model fingerprint */}
            {result.fingerprint && (
              <div className="bg-white border border-stone rounded-card px-6 py-4 flex items-center justify-between">
                <span className="text-[12px] text-muted">Model fingerprint (SHA-256)</span>
                <span className="text-[12px] font-mono text-ink-soft">{result.fingerprint}</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
