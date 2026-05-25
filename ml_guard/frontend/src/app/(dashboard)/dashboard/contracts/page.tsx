'use client';

import { useEffect, useState, useCallback } from 'react';
import { RefreshCw, ShieldCheck, Plus, Play } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { contractsApi, type Contract } from '@/lib/api';

function StatusBadge({ status }: { status: string }) {
  const v: 'certified' | 'conditional' | 'failed' =
    status === 'CERTIFIED' || status === 'PASSED' ? 'certified' :
    status === 'CONDITIONAL' ? 'conditional' : 'failed';
  return <Badge variant={v}>{status}</Badge>;
}

export default function ContractsPage() {
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [evaluating, setEvaluating] = useState<string | null>(null);
  const [total, setTotal] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await contractsApi.list();
      setContracts(res.items ?? []);
      setTotal(res.total ?? 0);
    } catch (e: any) {
      setError(e.message ?? 'Failed to load contracts');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const evaluate = async (contractId: string) => {
    setEvaluating(contractId);
    try {
      const result = await contractsApi.evaluate(contractId);
      // Update local state optimistically
      setContracts(prev => prev.map(c =>
        c.id === contractId
          ? { ...c, status: result.verdict, breach_rate: result.breach_rate }
          : c
      ));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setEvaluating(null);
    }
  };

  const certified = contracts.filter(c => c.status === 'CERTIFIED' || c.status === 'PASSED').length;
  const failed = contracts.filter(c => c.status === 'FAILED' || c.status === 'BREACHED').length;

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Behavioral Contracts</h1>
          <p className="text-[11px] text-muted">{total} contracts · {certified} certified · {failed} failed</p>
        </div>
        <button onClick={load} className="text-muted hover:text-ink transition-colors" aria-label="Refresh">
          <RefreshCw size={15} strokeWidth={1.5} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      <div className="flex-1 p-8">
        {error && (
          <div className="mb-5 p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger flex items-center justify-between">
            <span>⚠ {error}</span>
            <button onClick={load} className="text-forest underline text-[12px]">Retry</button>
          </div>
        )}

        <div className="bg-white border border-stone rounded-card overflow-hidden">
          <table className="w-full border-collapse">
            <thead className="bg-[#F7F6F2]">
              <tr>
                {['Contract name', 'Model', 'Type', 'Status', 'Breach rate', 'Actions'].map(h => (
                  <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted border-b border-stone">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} className="border-b border-stone/50 animate-pulse">
                    {[1,2,3,4,5,6].map(j => (
                      <td key={j} className="px-5 py-3.5">
                        <div className="h-3 bg-stone rounded-full" style={{ width: `${40 + j*10}%` }} />
                      </td>
                    ))}
                  </tr>
                ))
                : contracts.length === 0
                ? (
                  <tr>
                    <td colSpan={6} className="py-16 text-center">
                      <ShieldCheck size={32} className="mx-auto text-stone mb-3" strokeWidth={1} />
                      <p className="text-[14px] font-medium text-ink mb-1">No behavioral contracts</p>
                      <p className="text-[13px] text-muted">Create contracts via the backend API or SDK to enforce model behavior.</p>
                    </td>
                  </tr>
                )
                : contracts.map(c => (
                  <tr key={c.id} className="border-b border-stone/50 hover:bg-[#F0EDE8] transition-colors">
                    <td className="px-5 py-3.5 text-[13px] font-medium text-ink">{c.name}</td>
                    <td className="px-5 py-3.5 text-[13px] text-muted font-mono text-[12px]">
                      {c.model_name || `${c.model_id.slice(0, 8)}…`}
                    </td>
                    <td className="px-5 py-3.5">
                      <span className="text-[11px] font-medium px-2 py-0.5 rounded-badge bg-stone text-ink-soft">{c.contract_type}</span>
                    </td>
                    <td className="px-5 py-3.5"><StatusBadge status={c.status} /></td>
                    <td className="px-5 py-3.5 text-[13px] text-muted">
                      {c.breach_rate !== undefined ? `${(c.breach_rate * 100).toFixed(2)}%` : '—'}
                    </td>
                    <td className="px-5 py-3.5">
                      <button
                        onClick={() => evaluate(c.id)}
                        disabled={evaluating === c.id}
                        className="flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium text-forest border border-forest rounded-[6px] hover:bg-mist transition-colors disabled:opacity-50"
                      >
                        {evaluating === c.id ? <RefreshCw size={10} className="animate-spin" /> : <Play size={10} strokeWidth={2} />}
                        Evaluate
                      </button>
                    </td>
                  </tr>
                ))
              }
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
