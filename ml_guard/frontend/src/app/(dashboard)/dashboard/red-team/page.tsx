'use client';

import { useEffect, useState, useCallback } from 'react';
import { RefreshCw, Zap, AlertTriangle, CheckCircle, Plus } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { redTeamApi, modelsApi, type RedTeamSession, type ModelItem } from '@/lib/api';

const ATTACK_TYPES = ['adversarial_inputs', 'prompt_injection', 'fairness_probing', 'model_inversion', 'membership_inference'];

function StatusBadge({ status }: { status: string }) {
  const v: 'certified' | 'conditional' | 'failed' =
    status === 'COMPLETED' || status === 'PASSED' ? 'certified' :
    status === 'RUNNING' ? 'conditional' : 'failed';
  return <Badge variant={v}>{status}</Badge>;
}

function CreateSessionModal({ models, onClose, onSuccess }: {
  models: ModelItem[];
  onClose: () => void;
  onSuccess: () => void;
}) {
  const [modelId, setModelId] = useState(models[0]?.model_id ?? '');
  const [name, setName] = useState('');
  const [attacks, setAttacks] = useState<string[]>(['adversarial_inputs']);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const toggle = (a: string) =>
    setAttacks(prev => prev.includes(a) ? prev.filter(x => x !== a) : [...prev, a]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!attacks.length) { setError('Select at least one attack type'); return; }
    setLoading(true);
    setError('');
    try {
      await redTeamApi.create({ model_id: modelId, session_name: name, attack_types: attacks });
      onSuccess();
      onClose();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-ink/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-card w-full max-w-md p-6 shadow-xl">
        <h2 className="text-[16px] font-semibold text-ink mb-4">New red team session</h2>
        <form onSubmit={submit} className="flex flex-col gap-4">
          <div>
            <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Session name *</label>
            <input value={name} onChange={e => setName(e.target.value)} required placeholder="e.g. Q2 fairness audit"
              className="w-full h-10 px-3 text-[14px] border border-stone rounded-[8px] outline-none focus:border-forest" />
          </div>
          <div>
            <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Model *</label>
            <select value={modelId} onChange={e => setModelId(e.target.value)}
              className="w-full h-10 px-3 text-[14px] border border-stone rounded-[8px] outline-none focus:border-forest bg-white">
              {models.map(m => <option key={m.model_id} value={m.model_id}>{m.name}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-[12px] font-medium text-ink-soft mb-2">Attack types *</label>
            <div className="flex flex-wrap gap-2">
              {ATTACK_TYPES.map(a => (
                <button key={a} type="button" onClick={() => toggle(a)}
                  className={`px-2.5 py-1 text-[11px] font-medium rounded-badge border transition-colors ${
                    attacks.includes(a) ? 'bg-forest text-white border-forest' : 'bg-white text-ink-soft border-stone hover:border-forest'
                  }`}>
                  {a.replace(/_/g, ' ')}
                </button>
              ))}
            </div>
          </div>
          {error && <p className="text-[12px] text-danger">{error}</p>}
          <div className="flex gap-3 justify-end pt-2">
            <Button type="button" variant="ghost" size="sm" onClick={onClose}>Cancel</Button>
            <Button type="submit" variant="primary" size="sm" disabled={loading}>
              {loading ? 'Creating…' : 'Create session'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default function RedTeamPage() {
  const [sessions, setSessions] = useState<RedTeamSession[]>([]);
  const [models, setModels] = useState<ModelItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [running, setRunning] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [sessRes, modRes] = await Promise.all([
        redTeamApi.list(),
        modelsApi.list(1, 100),
      ]);
      setSessions(sessRes.items ?? []);
      setModels(modRes.items ?? []);
    } catch (e: any) {
      setError(e.message ?? 'Failed to load red team sessions');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const runSession = async (sessionId: string) => {
    setRunning(sessionId);
    try {
      await redTeamApi.run(sessionId);
      // Refresh
      const res = await redTeamApi.list();
      setSessions(res.items ?? []);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRunning(null);
    }
  };

  const completed = sessions.filter(s => s.status === 'COMPLETED').length;
  const totalVulns = sessions.reduce((sum, s) => sum + (s.vulnerability_count ?? 0), 0);

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Red Team</h1>
          <p className="text-[11px] text-muted">{sessions.length} sessions · {totalVulns} vulnerabilities found</p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={load} className="text-muted hover:text-ink transition-colors" aria-label="Refresh">
            <RefreshCw size={15} strokeWidth={1.5} className={loading ? 'animate-spin' : ''} />
          </button>
          <Button variant="primary" size="sm" className="gap-1.5" onClick={() => setShowModal(true)}>
            <Plus size={14} strokeWidth={2} /> New session
          </Button>
        </div>
      </div>

      <div className="flex-1 p-8">
        {/* Summary */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          {[
            { label: 'Total sessions', value: sessions.length },
            { label: 'Completed', value: completed },
            { label: 'Vulnerabilities found', value: totalVulns },
          ].map(s => (
            <div key={s.label} className="bg-white border border-stone rounded-card p-5">
              <p className="text-[11px] text-muted uppercase tracking-[0.05em] mb-2">{s.label}</p>
              <p className="text-[28px] font-bold text-ink leading-none">{s.value}</p>
            </div>
          ))}
        </div>

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
                {['Session', 'Model', 'Attack types', 'Vulnerabilities', 'Status', 'Created', 'Actions'].map(h => (
                  <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted border-b border-stone">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: 4 }).map((_, i) => (
                  <tr key={i} className="border-b border-stone/50 animate-pulse">
                    {[1,2,3,4,5,6,7].map(j => (
                      <td key={j} className="px-5 py-3.5"><div className="h-3 bg-stone rounded-full" style={{ width: `${40+j*8}%` }} /></td>
                    ))}
                  </tr>
                ))
                : sessions.length === 0
                ? (
                  <tr>
                    <td colSpan={7} className="py-16 text-center">
                      <Zap size={32} className="mx-auto text-stone mb-3" strokeWidth={1} />
                      <p className="text-[14px] font-medium text-ink mb-1">No red team sessions</p>
                      <p className="text-[13px] text-muted mb-4">Create a session to start adversarial testing of your models.</p>
                      <Button variant="primary" size="sm" onClick={() => setShowModal(true)}>Create first session</Button>
                    </td>
                  </tr>
                )
                : sessions.map(s => (
                  <tr key={s.id} className="border-b border-stone/50 hover:bg-[#F0EDE8] transition-colors">
                    <td className="px-5 py-3.5 text-[13px] font-medium text-ink">{s.session_name}</td>
                    <td className="px-5 py-3.5 text-[12px] font-mono text-muted">{s.model_id.slice(0, 8)}…</td>
                    <td className="px-5 py-3.5">
                      <div className="flex flex-wrap gap-1">
                        {(s.attack_types ?? []).slice(0, 2).map(a => (
                          <span key={a} className="text-[10px] px-1.5 py-0.5 rounded bg-stone text-ink-soft">{a.replace(/_/g, ' ')}</span>
                        ))}
                        {(s.attack_types ?? []).length > 2 && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-stone text-ink-soft">+{(s.attack_types ?? []).length - 2}</span>
                        )}
                      </div>
                    </td>
                    <td className="px-5 py-3.5">
                      {(s.vulnerability_count ?? 0) > 0
                        ? <span className="text-[13px] font-semibold text-danger">{s.vulnerability_count}</span>
                        : <span className="text-[13px] text-forest flex items-center gap-1"><CheckCircle size={12} />0</span>}
                    </td>
                    <td className="px-5 py-3.5"><StatusBadge status={s.status} /></td>
                    <td className="px-5 py-3.5 text-[12px] text-muted">{new Date(s.created_at).toLocaleDateString()}</td>
                    <td className="px-5 py-3.5">
                      {s.status !== 'COMPLETED' && (
                        <button
                          onClick={() => runSession(s.id)}
                          disabled={running === s.id}
                          className="flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium text-forest border border-forest rounded-[6px] hover:bg-mist transition-colors disabled:opacity-50"
                        >
                          {running === s.id ? <RefreshCw size={10} className="animate-spin" /> : <Zap size={10} />}
                          {running === s.id ? 'Running…' : 'Run'}
                        </button>
                      )}
                    </td>
                  </tr>
                ))
              }
            </tbody>
          </table>
        </div>
      </div>

      {showModal && models.length > 0 && (
        <CreateSessionModal models={models} onClose={() => setShowModal(false)} onSuccess={load} />
      )}
    </div>
  );
}
