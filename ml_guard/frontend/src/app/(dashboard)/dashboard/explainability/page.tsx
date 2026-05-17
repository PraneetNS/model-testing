'use client';

import { useState, useRef } from 'react';
import { Upload, Play, RefreshCw, BarChart2 } from 'lucide-react';
import { Button } from '@/components/ui/Button';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
const token = () => typeof window !== 'undefined' ? localStorage.getItem('niyantrana_token') : null;

function FileDropZone({ label, accept, file, onChange }: {
  label: string; accept: string; file: File | null; onChange: (f: File | null) => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <div onClick={() => ref.current?.click()}
      className={`border-2 border-dashed rounded-card p-5 cursor-pointer transition-colors text-center relative ${
        file ? 'border-forest bg-mist' : 'border-stone hover:border-forest/60'
      }`}>
      <input ref={ref} type="file" accept={accept} className="hidden" onChange={e => onChange(e.target.files?.[0] ?? null)} />
      <Upload size={18} className={`mx-auto mb-2 ${file ? 'text-forest' : 'text-muted'}`} strokeWidth={1.5} />
      <p className="text-[13px] font-medium text-ink">{file ? file.name : label}</p>
      <p className="text-[11px] text-muted mt-0.5">{file ? `${(file.size/1024).toFixed(1)} KB` : accept}</p>
      {file && <button onClick={e => { e.stopPropagation(); onChange(null); }} className="absolute top-2 right-2 text-muted hover:text-danger text-[11px]">✕</button>}
    </div>
  );
}

export default function ExplainabilityPage() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [maxSamples, setMaxSamples] = useState(50);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!modelFile || !datasetFile) { setError('Both model and dataset files are required.'); return; }
    setRunning(true); setError(null); setResult(null);
    const fd = new FormData();
    fd.append('model_file', modelFile);
    fd.append('dataset_file', datasetFile);
    fd.append('max_samples', String(maxSamples));
    try {
      const res = await fetch(`${BASE_URL}/explainability/compute`, {
        method: 'POST',
        headers: token() ? { Authorization: `Bearer ${token()}` } : {},
        body: fd,
      });
      if (!res.ok) { const b = await res.json().catch(() => ({})); throw new Error(b.detail ?? `HTTP ${res.status}`); }
      setResult(await res.json());
    } catch (e: any) { setError(e.message); } finally { setRunning(false); }
  };

  const maxImp = result?.feature_importance?.[0]?.importance ?? 1;

  return (
    <div className="flex flex-col min-h-screen">
      <div className="px-8 h-16 border-b border-stone bg-white flex items-center">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Explainability</h1>
          <p className="text-[11px] text-muted">SHAP feature importance — upload model + dataset</p>
        </div>
      </div>

      <div className="flex-1 p-8 space-y-6">
        {/* Upload */}
        <div className="bg-white border border-stone rounded-card p-6">
          <h2 className="text-[14px] font-semibold text-ink mb-5">Upload artifacts</h2>
          <div className="grid md:grid-cols-2 gap-4 mb-5">
            <FileDropZone label="Model File (.pkl)" accept=".pkl,.joblib" file={modelFile} onChange={setModelFile} />
            <FileDropZone label="Dataset (.csv)" accept=".csv" file={datasetFile} onChange={setDatasetFile} />
          </div>
          <div className="flex items-center gap-4 mb-5">
            <label className="text-[12px] font-medium text-ink-soft">Max samples:</label>
            <input type="number" value={maxSamples} min={5} max={200}
              onChange={e => setMaxSamples(Number(e.target.value))}
              className="w-24 h-9 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
            <span className="text-[11px] text-muted">Higher = more accurate but slower</span>
          </div>
          {error && <p className="mb-4 text-[12px] text-danger bg-red-50 border border-red-200 rounded-[8px] px-4 py-2">⚠ {error}</p>}
          <Button variant="primary" size="sm" className="gap-2" onClick={run} disabled={running}>
            {running ? <><RefreshCw size={13} className="animate-spin" />Computing SHAP…</> : <><Play size={13} />Compute Explainability</>}
          </Button>
        </div>

        {/* Results */}
        {result && (
          <>
            {/* Score + method */}
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white border border-stone rounded-card p-5 col-span-1">
                <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">Interpretability Score</p>
                <p className="text-[36px] font-bold leading-none" style={{ color: result.interpretability_score >= 60 ? '#1A5F3A' : '#B35A00' }}>
                  {result.interpretability_score?.toFixed(0)}
                  <span className="text-[16px] text-muted font-normal">/100</span>
                </p>
              </div>
              <div className="bg-white border border-stone rounded-card p-5">
                <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">Method Used</p>
                <p className="text-[16px] font-semibold text-ink capitalize">{result.method?.replace(/_/g, ' ')}</p>
              </div>
              <div className="bg-white border border-stone rounded-card p-5">
                <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">Top Feature</p>
                <p className="text-[15px] font-semibold text-ink truncate">{result.top_features?.[0] ?? '—'}</p>
              </div>
            </div>

            {/* Feature importance chart */}
            <div className="bg-white border border-stone rounded-card p-6">
              <h2 className="text-[14px] font-semibold text-ink mb-5 flex items-center gap-2">
                <BarChart2 size={16} strokeWidth={1.5} className="text-forest" />
                Feature Importance (Top {result.feature_importance?.length ?? 0})
              </h2>
              <div className="space-y-3">
                {result.feature_importance?.map((f: any, i: number) => {
                  const pct = maxImp > 0 ? (f.importance / maxImp) * 100 : 0;
                  const colors = ['#1A5F3A', '#2E7D52', '#3D8B6A', '#4CAF80', '#6EC99A', '#8FD9B0', '#AADCC5', '#C5EDD9'];
                  return (
                    <div key={f.feature} className="flex items-center gap-4">
                      <span className="text-[13px] text-ink-soft w-8 text-right text-muted">#{i + 1}</span>
                      <span className="text-[13px] font-medium text-ink w-44 truncate">{f.feature}</span>
                      <div className="flex-1 h-6 bg-stone rounded-[4px] overflow-hidden relative">
                        <div className="h-full rounded-[4px] transition-all duration-500 flex items-center pl-2"
                          style={{ width: `${pct}%`, background: colors[i % colors.length] }}>
                          {pct > 15 && <span className="text-[10px] text-white font-medium">{f.importance.toFixed(4)}</span>}
                        </div>
                        {pct <= 15 && <span className="absolute left-full ml-2 top-1/2 -translate-y-1/2 text-[10px] text-muted">{f.importance.toFixed(4)}</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Top 5 features */}
            <div className="bg-white border border-stone rounded-card p-6">
              <h2 className="text-[14px] font-semibold text-ink mb-4">Top 5 influential features</h2>
              <div className="flex flex-wrap gap-2">
                {result.top_features?.map((f: string, i: number) => (
                  <span key={f} className="px-3 py-1.5 rounded-badge text-[12px] font-medium" style={{
                    background: i === 0 ? '#1A5F3A' : i === 1 ? '#2E7D52' : '#4CAF80',
                    color: 'white'
                  }}>
                    #{i + 1} {f}
                  </span>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
