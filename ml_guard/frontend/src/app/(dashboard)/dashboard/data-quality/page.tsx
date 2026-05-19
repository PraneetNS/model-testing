'use client';

import { useState, useRef } from 'react';
import { Upload, Play, RefreshCw, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000') + '/api/v1';

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

export default function DataQualityPage() {
  const [dataFile, setDataFile] = useState<File | null>(null);
  const [refFile, setRefFile] = useState<File | null>(null);
  const [targetCol, setTargetCol] = useState('target');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!dataFile) { setError('Dataset file required.'); return; }
    setRunning(true); setError(null); setResult(null);
    const fd = new FormData();
    fd.append('dataset_file', dataFile);
    if (refFile) fd.append('reference_file', refFile);
    fd.append('target_column', targetCol);
    try {
      const r = await fetch(`${BASE}/data-quality/validate`, {
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

  const score: number = result?.quality_score ?? 0;
  const statusColor = (s: string) =>
    s === 'EXCELLENT' || s === 'PASS' ? 'certified' : s === 'GOOD' ? 'certified' : s === 'WARNING' || s === 'FAIL' && score >= 50 ? 'conditional' : 'failed';

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Data Quality</h1>
          <p className="text-[11px] text-muted">Schema validation · freshness · distribution checks</p>
        </div>
        {result && (
          <Badge variant={statusColor(result.status)}>{result.status} — {score.toFixed(0)}/100</Badge>
        )}
      </div>

      <div className="flex-1 p-8 space-y-6 overflow-auto">
        {error && <div className="p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger">⚠ {error}</div>}

        {/* Upload */}
        <div className="bg-white border border-stone rounded-card p-6">
          <h2 className="text-[14px] font-semibold text-ink mb-4">Upload dataset</h2>
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <FileZone label="Dataset to validate (.csv)" file={dataFile} onChange={setDataFile} accept=".csv" />
            <FileZone label="Reference dataset (.csv, optional)" file={refFile} onChange={setRefFile} accept=".csv" />
          </div>
          <div className="flex gap-4 items-end">
            <div>
              <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Target column</label>
              <input value={targetCol} onChange={e => setTargetCol(e.target.value)}
                className="h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
            </div>
            <Button variant="primary" size="sm" onClick={run} disabled={running || !dataFile} className="gap-2">
              {running ? <><RefreshCw size={13} className="animate-spin" />Validating…</> : <><Play size={13} />Validate Dataset</>}
            </Button>
          </div>
        </div>

        {/* Results */}
        {result && (
          <>
            {/* Score overview */}
            <div className="grid md:grid-cols-4 gap-4">
              {[
                { label: 'Quality Score', value: `${score.toFixed(0)}/100`, good: score >= 75 },
                { label: 'Checks Passed', value: `${result.checks_passed}/${result.total_checks}`, good: result.checks_passed === result.total_checks },
                { label: 'Critical Issues', value: result.critical_issues ?? 0, good: result.critical_issues === 0 },
                { label: 'Row Count', value: result.details?.row_count?.toLocaleString() ?? '—', good: true },
              ].map(({ label, value, good }) => (
                <div key={label} className="bg-white border border-stone rounded-card p-4">
                  <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">{label}</p>
                  <p className={`text-[22px] font-bold ${good ? 'text-forest' : 'text-danger'}`}>{value}</p>
                </div>
              ))}
            </div>

            {/* Checks detail */}
            <div className="bg-white border border-stone rounded-card overflow-hidden">
              <div className="px-6 py-4 border-b border-stone">
                <h2 className="text-[14px] font-semibold text-ink">Check results</h2>
              </div>
              <div className="divide-y divide-stone/40">
                {Object.entries(result.report ?? {}).map(([check, data]: [string, any]) => (
                  <div key={check} className="flex items-center gap-4 px-6 py-3">
                    {data.status === 'PASS'
                      ? <CheckCircle size={14} className="text-forest flex-shrink-0" />
                      : <XCircle size={14} className="text-danger flex-shrink-0" />}
                    <div className="flex-1">
                      <p className="text-[13px] font-medium text-ink capitalize">{check.replace(/_/g, ' ')}</p>
                      <p className="text-[11px] text-muted">{data.message}</p>
                    </div>
                    <Badge variant={data.status === 'PASS' ? 'certified' : 'failed'}>{data.status}</Badge>
                  </div>
                ))}
              </div>
            </div>

            {/* Details */}
            {result.details && (
              <div className="bg-white border border-stone rounded-card p-6">
                <h2 className="text-[14px] font-semibold text-ink mb-3">Dataset metadata</h2>
                <div className="grid md:grid-cols-3 gap-4 text-[13px]">
                  <div><p className="text-muted text-[11px]">Rows</p><p className="font-semibold text-ink">{result.details.row_count?.toLocaleString() ?? '—'}</p></div>
                  <div><p className="text-muted text-[11px]">Features</p><p className="font-semibold text-ink">{result.details.feature_count ?? '—'}</p></div>
                  <div><p className="text-muted text-[11px]">Schema Hash</p><p className="font-mono text-[11px] text-muted">{result.details.schema_hash?.slice(0, 16) ?? '—'}…</p></div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
