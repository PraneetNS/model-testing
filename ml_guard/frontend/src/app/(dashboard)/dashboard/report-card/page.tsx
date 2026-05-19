'use client';

import { useState, useEffect } from 'react';
import { FileText, RefreshCw, Download, CheckCircle, XCircle, Shield } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { modelsApi } from '@/lib/api';

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000') + '/api/v1';
const HDR = { 'Content-Type': 'application/json', 'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'dev-secret-key' };

export default function ReportCardPage() {
  const [models, setModels] = useState<any[]>([]);
  const [selectedId, setSelectedId] = useState('');
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [verifyHash, setVerifyHash] = useState('');
  const [verifying, setVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    modelsApi.list(1, 50).then(r => {
      setModels(r.items ?? []);
      if (r.items?.length) setSelectedId(r.items[0].model_id);
    }).catch(() => {});
  }, []);

  const generate = async () => {
    if (!selectedId) return;
    setGenerating(true); setError(null); setResult(null);
    try {
      const r = await fetch(`${BASE}/governance/${selectedId}/certify`, {
        method: 'POST', headers: HDR, body: JSON.stringify({ force_regenerate: false }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? `HTTP ${r.status}`);
      setResult(d);
    } catch (e: any) { setError(e.message); }
    finally { setGenerating(false); }
  };

  const verify = async () => {
    if (!verifyHash.trim()) return;
    setVerifying(true); setVerifyResult(null);
    try {
      const r = await fetch(`${BASE}/governance/verify/${verifyHash.trim()}`, { headers: HDR });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? `HTTP ${r.status}`);
      setVerifyResult(d);
    } catch (e: any) { setError(e.message); }
    finally { setVerifying(false); }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Report Card</h1>
          <p className="text-[11px] text-muted">Compliance certificates — cryptographically signed</p>
        </div>
      </div>

      <div className="flex-1 p-8 space-y-6 overflow-auto">
        {error && <div className="p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger">⚠ {error}</div>}

        {/* Generate card */}
        <div className="bg-white border border-stone rounded-card p-6">
          <h2 className="text-[14px] font-semibold text-ink mb-4">Generate compliance certificate</h2>
          <div className="flex gap-3 items-end">
            <div className="flex-1 max-w-xs">
              <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Model</label>
              <select value={selectedId} onChange={e => setSelectedId(e.target.value)}
                className="w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest bg-white">
                {models.length === 0 && <option value="">No models registered</option>}
                {models.map(m => <option key={m.model_id} value={m.model_id}>{m.name}</option>)}
              </select>
            </div>
            <Button variant="primary" size="sm" onClick={generate} disabled={generating || !selectedId} className="gap-2">
              {generating ? <><RefreshCw size={13} className="animate-spin" />Generating…</> : <><Shield size={13} />Generate Certificate</>}
            </Button>
          </div>

          {result && (
            <div className="mt-5 border border-forest/30 rounded-[10px] overflow-hidden">
              {/* Certificate header */}
              <div className="flex items-center justify-between px-6 py-4" style={{ background: '#0F0F0E' }}>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-[#1A5F3A]/20 flex items-center justify-center">
                    <Shield size={18} className="text-[#3ECF8E]" />
                  </div>
                  <div>
                    <p className="text-[13px] font-semibold text-white">ML Guard Compliance Certificate</p>
                    <p className="text-[11px]" style={{ color: '#555552' }}>{result.issued_at ? new Date(result.issued_at).toLocaleString() : 'Just now'}</p>
                  </div>
                </div>
                <Badge variant={result.verdict === 'CERTIFIED' ? 'certified' : result.verdict === 'CONDITIONAL' ? 'conditional' : 'failed'}>
                  {result.verdict}
                </Badge>
              </div>

              {/* Certificate body */}
              <div className="p-6 space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-[13px]">
                  <div>
                    <p className="text-[10px] text-muted uppercase tracking-[0.05em] mb-1">Model ID</p>
                    <p className="font-medium text-ink font-mono text-[12px]">{result.model_id}</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-muted uppercase tracking-[0.05em] mb-1">Overall Score</p>
                    <p className="font-bold text-ink text-[18px]">{result.overall_score?.toFixed(1)}<span className="text-[12px] text-muted font-normal">/100</span></p>
                  </div>
                  <div>
                    <p className="text-[10px] text-muted uppercase tracking-[0.05em] mb-1">Live Score</p>
                    <p className="font-bold text-forest text-[18px]">{result.live_score?.toFixed(1) ?? '—'}<span className="text-[12px] text-muted font-normal">/100</span></p>
                  </div>
                </div>

                <div className="border-t border-stone/50 pt-4">
                  <p className="text-[10px] text-muted uppercase tracking-[0.05em] mb-1.5">Certificate Hash (SHA-256)</p>
                  <p className="font-mono text-[11px] text-ink-soft break-all bg-[#F7F6F2] px-3 py-2 rounded-[6px]">{result.cert_hash}</p>
                </div>

                <div className="flex gap-2 flex-wrap">
                  <button
                    onClick={() => { setVerifyHash(result.cert_hash); }}
                    className="flex items-center gap-1.5 text-[12px] text-forest hover:underline"
                  >
                    <CheckCircle size={12} /> Verify this certificate
                  </button>
                  <span className="text-muted">·</span>
                  <span className="text-[12px] text-muted">Share: <span className="text-forest font-medium">{result.download_url}</span></span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Verify card */}
        <div className="bg-white border border-stone rounded-card p-6">
          <h2 className="text-[14px] font-semibold text-ink mb-4">Verify a certificate</h2>
          <p className="text-[12px] text-muted mb-4">Enter a certificate hash to verify its authenticity and current compliance status.</p>
          <div className="flex gap-3 items-end">
            <div className="flex-1">
              <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Certificate hash</label>
              <input value={verifyHash} onChange={e => setVerifyHash(e.target.value)}
                placeholder="e.g. a3f8d2..."
                className="w-full h-10 px-3 text-[13px] font-mono border border-stone rounded-[8px] outline-none focus:border-forest" />
            </div>
            <Button variant="ghost" size="sm" onClick={verify} disabled={verifying || !verifyHash.trim()} className="gap-2">
              {verifying ? <><RefreshCw size={13} className="animate-spin" />Verifying…</> : 'Verify'}
            </Button>
          </div>

          {verifyResult && (
            <div className={`mt-4 p-4 border rounded-[8px] ${verifyResult.valid ? 'bg-mist border-forest/30' : 'bg-red-50 border-red-200'}`}>
              <div className="flex items-center gap-2 mb-3">
                {verifyResult.valid
                  ? <CheckCircle size={15} className="text-forest" />
                  : <XCircle size={15} className="text-danger" />}
                <span className="text-[13px] font-semibold text-ink">
                  {verifyResult.valid ? 'Certificate Valid' : 'Certificate Invalid / Revoked'}
                </span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-[12px]">
                {[
                  ['Verdict', verifyResult.verdict],
                  ['Score', verifyResult.overall_score?.toFixed(1)],
                  ['Still Compliant', verifyResult.still_compliant ? 'Yes' : 'No'],
                  ['Issued At', verifyResult.issued_at ? new Date(verifyResult.issued_at).toLocaleDateString() : '—'],
                  ['Revoked', verifyResult.is_revoked ? `Yes — ${verifyResult.revocation_reason}` : 'No'],
                  ['Drift Events', verifyResult.drift_events_since_issue ?? '0'],
                ].map(([k, v]) => (
                  <div key={k}>
                    <p className="text-muted">{k}</p>
                    <p className="font-medium text-ink">{String(v ?? '—')}</p>
                  </div>
                ))}
              </div>
              {verifyResult.message && <p className="mt-3 text-[12px] text-muted italic">{verifyResult.message}</p>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
