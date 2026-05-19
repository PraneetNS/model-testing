'use client';

import { useState, useRef } from 'react';
import { Upload, Play, RefreshCw, Scale, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { API_BASE, apiUploadHeaders } from '@/lib/api';

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

function MetricCard({ label, value, threshold, inverse = false }: { 
  label: string; value: number; threshold: number; inverse?: boolean 
}) {
  const absVal = Math.abs(value);
  const isOk = inverse ? absVal > threshold : absVal < threshold;
  
  return (
    <div className="bg-white border border-stone rounded-card p-5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] text-muted uppercase tracking-[0.05em]">{label}</span>
        {isOk ? <CheckCircle size={14} className="text-forest" /> : <AlertTriangle size={14} className="text-danger" />}
      </div>
      <p className="text-[24px] font-bold text-ink mb-1">{value.toFixed(4)}</p>
      <div className="flex items-center gap-1.5">
        <div className="flex-1 h-1.5 bg-stone rounded-full overflow-hidden">
          <div className={`h-full rounded-full transition-all duration-500 ${isOk ? 'bg-forest' : 'bg-danger'}`} 
               style={{ width: `${Math.min(100, (absVal / (threshold * 2)) * 100)}%` }} />
        </div>
        <span className="text-[10px] text-muted">Limit: {threshold}</span>
      </div>
    </div>
  );
}

export default function FairnessPage() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [dataFile, setDataFile] = useState<File | null>(null);
  const [sensitiveColumn, setSensitiveColumn] = useState('gender');
  const [labelCol, setLabelCol] = useState('target');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = async () => {
    if (!modelFile || !dataFile) { setError('Both model and dataset files are required.'); return; }
    if (!sensitiveColumn) { setError('Sensitive column name is required.'); return; }

    setRunning(true); setError(null); setResult(null);

    const fd = new FormData();
    fd.append('model_file', modelFile);
    fd.append('data_file', dataFile);
    fd.append('sensitive_column', sensitiveColumn);
    fd.append('label_col', labelCol);

    try {
      const res = await fetch(`${API_BASE}/fairness/analyze`, {
        method: 'POST',
        headers: apiUploadHeaders(),
        body: fd,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message ?? 'Analysis failed');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Fairness & Bias Detection</h1>
          <p className="text-[11px] text-muted">Analyze model bias across sensitive attributes</p>
        </div>
        {result && (
          <Badge variant={result.fairness?.fairness_flag ? 'failed' : 'certified'}>
            {result.fairness?.fairness_flag ? 'BIAS DETECTED' : 'FAIRNESS CERTIFIED'}
          </Badge>
        )}
      </div>

      <div className="flex-1 p-8 space-y-6">
        {/* Input Form */}
        <div className="bg-white border border-stone rounded-card p-6">
          <h2 className="text-[14px] font-semibold text-ink mb-5">Audit configuration</h2>
          <div className="grid md:grid-cols-2 gap-4 mb-5">
            <FileDropZone label="Model File (.pkl / .onnx)" accept=".pkl,.joblib,.onnx" file={modelFile} onChange={setModelFile} />
            <FileDropZone label="Evaluation Dataset (.csv)" accept=".csv" file={dataFile} onChange={setDataFile} />
          </div>

          <div className="grid md:grid-cols-2 gap-4 mb-5">
            <div>
              <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Sensitive column (e.g. race, gender)</label>
              <input value={sensitiveColumn} onChange={e => setSensitiveColumn(e.target.value)}
                placeholder="gender"
                className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
            </div>
            <div>
              <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Target column</label>
              <input value={labelCol} onChange={e => setLabelCol(e.target.value)}
                placeholder="target"
                className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
            </div>
          </div>

          {error && <p className="mb-4 text-[12px] text-danger bg-red-50 border border-red-200 rounded-[8px] px-4 py-2">⚠ {error}</p>}

          <Button variant="primary" size="sm" className="gap-2" onClick={runAnalysis} disabled={running}>
            {running
              ? <><RefreshCw size={13} strokeWidth={1.5} className="animate-spin" />Analyzing fairness…</>
              : <><Scale size={13} strokeWidth={1.5} />Run Fairness Audit</>}
          </Button>
        </div>

        {/* Results */}
        {result && (
          <>
            {/* Top metrics */}
            <div className="grid md:grid-cols-3 gap-4">
              <MetricCard 
                label="Statistical Parity Diff" 
                value={result.fairness?.statistical_parity_diff ?? 0} 
                threshold={0.1} 
              />
              <MetricCard 
                label="Disparate Impact Ratio" 
                value={result.fairness?.disparate_impact_ratio ?? 1} 
                threshold={0.8} 
                inverse
              />
              <MetricCard 
                label="Equal Opportunity Diff" 
                value={result.fairness?.equal_opportunity_diff ?? 0} 
                threshold={0.1} 
              />
            </div>

            {/* Group Metrics Table */}
            <div className="bg-white border border-stone rounded-card overflow-hidden">
              <div className="px-6 py-4 border-b border-stone bg-[#F7F6F2] flex items-center justify-between">
                <div>
                  <h3 className="text-[14px] font-semibold text-ink">Per-group metrics</h3>
                  <p className="text-[11px] text-muted mt-0.5">Sensitive attribute: <span className="font-medium text-forest">{result.sensitive_column}</span></p>
                </div>
                <div className="text-right">
                  <span className="text-[11px] text-muted block">Total samples</span>
                  <span className="text-[14px] font-bold text-ink">{result.n_samples}</span>
                </div>
              </div>
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-white border-b border-stone">
                    {['Group', 'Sample Count', 'Selection Rate', 'Accuracy', 'Precision', 'Recall'].map(h => (
                      <th key={h} className="text-left px-6 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.fairness?.group_metrics ?? {}).map(([group, metrics]: [string, any]) => (
                    <tr key={group} className="border-b border-stone/50 hover:bg-[#F7F6F2] transition-colors">
                      <td className="px-6 py-3.5 text-[13px] font-medium text-ink">{group}</td>
                      <td className="px-6 py-3.5 text-[13px] text-muted">{metrics.count}</td>
                      <td className="px-6 py-3.5 text-[13px] text-ink">{metrics.selection_rate?.toFixed(3)}</td>
                      <td className="px-6 py-3.5 text-[13px] text-ink">{metrics.accuracy?.toFixed(3)}</td>
                      <td className="px-6 py-3.5 text-[13px] text-ink">{metrics.precision?.toFixed(3)}</td>
                      <td className="px-6 py-3.5 text-[13px] text-ink">{metrics.recall?.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Recommendations */}
            <div className="bg-mist border border-forest/20 rounded-card p-6">
              <h3 className="text-[14px] font-semibold text-forest mb-3 flex items-center gap-2">
                <Info size={16} /> Fairness Advisory
              </h3>
              <p className="text-[13px] text-ink-soft leading-relaxed">
                {result.fairness?.fairness_flag 
                  ? `Significant bias detected for group "${result.sensitive_column}". Consider re-weighting samples or using debiasing algorithms like Reject Option Classification (ROC). Ensure your training data has representative coverage for all demographic groups.`
                  : "No significant bias detected across the analyzed sensitive attributes. The model meets the fairness thresholds for statistical parity and disparate impact."}
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
