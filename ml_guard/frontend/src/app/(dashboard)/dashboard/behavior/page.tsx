'use client';

import { useState, useRef } from 'react';
import { Upload, Play, RefreshCw, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000') + '/api/v1';

const SCENARIOS = [
  { id: 'sensitivity_analysis', label: 'Sensitivity Analysis', desc: 'Finite-difference partial derivatives per feature' },
  { id: 'monte_carlo_stability', label: 'Monte Carlo Stability', desc: 'Gaussian noise injection × 100, flip rate & stability' },
  { id: 'ood_boundary_test', label: 'OOD Boundary Test', desc: 'Synthetic extremes (min−3σ, max+3σ)' },
  { id: 'adversarial_permutation', label: 'Adversarial Permutation', desc: 'Permutation importance: shuffle each feature' },
  { id: 'noise_perturbation', label: 'Noise Perturbation', desc: 'Single Gaussian noise run (σ=0.1×std)' },
  { id: 'extreme_values', label: 'Extreme Values', desc: 'Feed min/max row values uniformly' },
  { id: 'missing_data_injection', label: 'Missing Data Injection', desc: '30% NaN injection, imputed with column mean' },
  { id: 'boundary_inputs', label: 'Boundary Inputs', desc: 'Predict at 5th and 95th percentiles' },
  { id: 'adversarial_shifts', label: 'Adversarial Shifts', desc: 'Shift all features by +2 standard deviations' },
];

function FileZone({ label, file, onChange, accept }: { label: string; file: File | null; onChange: (f: File | null) => void; accept: string }) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <div onClick={() => ref.current?.click()}
      className={`border-2 border-dashed rounded-card p-4 cursor-pointer text-center transition-colors ${file ? 'border-forest bg-mist' : 'border-stone hover:border-forest/60'}`}>
      <input ref={ref} type="file" accept={accept} className="hidden" onChange={e => onChange(e.target.files?.[0] ?? null)} />
      <Upload size={16} className={`mx-auto mb-1 ${file ? 'text-forest' : 'text-muted'}`} />
      <p className="text-[12px] font-medium text-ink">{file ? file.name : label}</p>
      <p className="text-[10px] text-muted">{accept}</p>
    </div>
  );
}

export default function BehaviorTestPage() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [refFile, setRefFile] = useState<File | null>(null);
  const [labelCol, setLabelCol] = useState('target');
  const [selected, setSelected] = useState<string[]>(['monte_carlo_stability', 'noise_perturbation', 'extreme_values']);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const toggle = (id: string) =>
    setSelected(p => p.includes(id) ? p.filter(x => x !== id) : [...p, id]);

  const run = async () => {
    if (!modelFile || !refFile) { setError('Model and reference dataset are required.'); return; }
    if (!selected.length) { setError('Select at least one scenario.'); return; }
    setRunning(true); setError(null); setResult(null);
    const fd = new FormData();
    fd.append('model_file', modelFile);
    fd.append('ref_file', refFile);
    fd.append('scenarios', selected.join(','));
    fd.append('label_col', labelCol);
    try {
      const r = await fetch(`${BASE}/behavior/test`, {
        method: 'POST',
        headers: { 'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'dev-secret-key' },
        body: fd,
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? `HTTP ${r.status}`);
      setResult(d);
    } catch (e: any) { setError(e.message); }
    finally { setRunning(false); }
  };

  const score: number = result?.robustness_score ?? 0;
  const status: string = result?.status ?? '';

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Behavior Test</h1>
          <p className="text-[11px] text-muted">Scenario robustness · stress testing · boundary analysis</p>
        </div>
        {result && (
          <Badge variant={status === 'PASSED' ? 'certified' : 'failed'}>{status} — {score.toFixed(0)}/100</Badge>
        )}
      </div>

      <div className="flex-1 p-8 space-y-6 overflow-auto">
        {error && <div className="p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger">⚠ {error}</div>}

        {/* Upload */}
        <div className="bg-white border border-stone rounded-card p-6">
          <h2 className="text-[14px] font-semibold text-ink mb-4">Artifacts</h2>
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <FileZone label="Model (.pkl / .onnx)" file={modelFile} onChange={setModelFile} accept=".pkl,.joblib,.onnx" />
            <FileZone label="Reference Dataset (.csv)" file={refFile} onChange={setRefFile} accept=".csv" />
          </div>
          <div className="max-w-xs">
            <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Label column</label>
            <input value={labelCol} onChange={e => setLabelCol(e.target.value)}
              className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
          </div>
        </div>

        {/* Scenarios */}
        <div className="bg-white border border-stone rounded-card p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-[14px] font-semibold text-ink">Select scenarios ({selected.length} selected)</h2>
            <div className="flex gap-2">
              <button onClick={() => setSelected(SCENARIOS.map(s => s.id))} className="text-[11px] text-forest hover:underline">All</button>
              <span className="text-muted">·</span>
              <button onClick={() => setSelected([])} className="text-[11px] text-muted hover:text-ink">None</button>
            </div>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
            {SCENARIOS.map(s => (
              <button key={s.id} onClick={() => toggle(s.id)}
                className={`text-left p-3 rounded-[8px] border transition-colors ${selected.includes(s.id) ? 'border-forest bg-mist' : 'border-stone hover:border-forest/50'}`}>
                <div className="flex items-center gap-2 mb-1">
                  <div className={`w-3 h-3 rounded-full flex-shrink-0 border ${selected.includes(s.id) ? 'bg-forest border-forest' : 'border-stone'}`} />
                  <span className="text-[12px] font-semibold text-ink">{s.label}</span>
                </div>
                <p className="text-[11px] text-muted pl-5">{s.desc}</p>
              </button>
            ))}
          </div>
          <div className="mt-4">
            <Button variant="primary" size="sm" onClick={run} disabled={running} className="gap-2">
              {running ? <><RefreshCw size={13} className="animate-spin" />Running tests…</> : <><Play size={13} />Run Behavior Tests</>}
            </Button>
          </div>
        </div>

        {/* Results */}
        {result && (
          <>
            {/* Score banner */}
            <div className={`border rounded-card p-5 flex items-center gap-4 ${status === 'PASSED' ? 'bg-mist border-forest/30' : 'bg-red-50 border-red-200'}`}>
              {status === 'PASSED' ? <CheckCircle size={20} className="text-forest" /> : <XCircle size={20} className="text-danger" />}
              <div>
                <p className="text-[15px] font-semibold text-ink">Robustness Score: {score.toFixed(1)}/100</p>
                <p className="text-[12px] text-muted">Baseline variance: {result.baseline_variance?.toFixed(6) ?? '—'}</p>
              </div>
            </div>

            {/* Per-scenario results */}
            <div className="space-y-3">
              {Object.entries(result.stress_results ?? {}).map(([key, val]: [string, any]) => {
                const stable = val?.stability_flag === 'STABLE' || (val?.stability_score != null && val.stability_score > 0.7);
                const hasError = !!val?.error;
                return (
                  <div key={key} className="bg-white border border-stone rounded-card p-5">
                    <div className="flex items-center gap-3 mb-3">
                      {hasError ? <AlertTriangle size={14} className="text-warning" />
                        : stable ? <CheckCircle size={14} className="text-forest" />
                        : <XCircle size={14} className="text-danger" />}
                      <h3 className="text-[13px] font-semibold text-ink capitalize">{key.replace(/_/g, ' ')}</h3>
                      {!hasError && (
                        <Badge variant={stable ? 'certified' : 'failed'} className="ml-auto">
                          {val?.stability_flag ?? (stable ? 'STABLE' : 'UNSTABLE')}
                        </Badge>
                      )}
                    </div>
                    {hasError ? (
                      <p className="text-[12px] text-danger">Error: {val.error}</p>
                    ) : (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[12px]">
                        {val?.stability_score != null && (
                          <div><p className="text-muted">Stability</p><p className="font-semibold text-ink">{(val.stability_score * 100).toFixed(1)}%</p></div>
                        )}
                        {val?.flip_rate != null && (
                          <div><p className="text-muted">Flip Rate</p><p className="font-semibold text-ink">{(val.flip_rate * 100).toFixed(2)}%</p></div>
                        )}
                        {val?.output_variance != null && (
                          <div><p className="text-muted">Output Variance</p><p className="font-semibold text-ink">{val.output_variance.toFixed(4)}</p></div>
                        )}
                        {val?.n_predictions != null && (
                          <div><p className="text-muted">Predictions</p><p className="font-semibold text-ink">{val.n_predictions}</p></div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
