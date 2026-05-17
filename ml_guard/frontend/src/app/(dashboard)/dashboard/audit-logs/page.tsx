'use client';

import { useEffect, useState, useCallback } from 'react';
import { RefreshCw, Search, Shield, User, Clock, Terminal } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { api } from '@/lib/api';

interface AuditLog {
  id: string; action: string; resource_type: string;
  resource_id: string | null; details: any; created_at: string;
}

function timeAgo(ts: string) {
  const d = Date.now() - new Date(ts).getTime();
  const m = Math.floor(d / 60000);
  if (m < 1) return 'just now';
  return m < 60 ? `${m}m ago` : m < 1440 ? `${Math.floor(m/60)}h ago` : `${Math.floor(m/1440)}d ago`;
}

export default function AuditLogsPage() {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');

  const load = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const data = await api.get<AuditLog[]>('/audit-logs?limit=100');
      setLogs(Array.isArray(data) ? data : []);
    } catch (e: any) { setError(e.message); } finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  const filtered = logs.filter(l => 
    l.action.toLowerCase().includes(search.toLowerCase()) ||
    l.resource_type.toLowerCase().includes(search.toLowerCase()) ||
    (l.resource_id && l.resource_id.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Platform Audit Logs</h1>
          <p className="text-[11px] text-muted">Immutable trail of all system activities</p>
        </div>
        <button onClick={load} className="text-muted hover:text-ink transition-colors" aria-label="Refresh">
          <RefreshCw size={15} strokeWidth={1.5} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      <div className="flex-1 p-8 space-y-5">
        {/* Search */}
        <div className="relative max-w-[400px]">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
          <input value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search logs (action, resource, ID)…"
            className="w-full h-10 pl-9 pr-4 text-[13px] border border-stone rounded-[8px] bg-white outline-none focus:border-forest" />
        </div>

        {error && <div className="p-4 bg-red-50 border border-red-200 rounded text-[13px] text-danger">⚠ {error}</div>}

        <div className="bg-white border border-stone rounded-card overflow-hidden">
          <table className="w-full border-collapse">
            <thead>
              <tr className="bg-[#F7F6F2] border-b border-stone">
                {['Action', 'Resource', 'Resource ID', 'Details', 'Timestamp'].map(h => (
                  <th key={h} className="text-left px-6 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                Array.from({ length: 10 }).map((_, i) => (
                  <tr key={i} className="border-b border-stone/50 animate-pulse">
                    {[1,2,3,4,5].map(j => (
                      <td key={j} className="px-6 py-4"><div className="h-3 bg-stone rounded-full w-2/3" /></td>
                    ))}
                  </tr>
                ))
              ) : filtered.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-20 text-center">
                    <Terminal size={32} className="mx-auto text-stone mb-3" strokeWidth={1} />
                    <p className="text-[14px] font-medium text-ink">No logs found</p>
                  </td>
                </tr>
              ) : (
                filtered.map(log => (
                  <tr key={log.id} className="border-b border-stone/50 hover:bg-[#F7F6F2] transition-colors">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2.5">
                        <div className="w-7 h-7 rounded-full bg-mist flex items-center justify-center text-forest">
                          <Shield size={12} strokeWidth={2} />
                        </div>
                        <span className="text-[13px] font-semibold text-ink">{log.action}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-[11px] font-medium px-2 py-0.5 rounded-badge bg-stone text-ink-soft uppercase tracking-wider">{log.resource_type}</span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-[12px] font-mono text-muted">{log.resource_id ? `${log.resource_id.slice(0, 12)}…` : '—'}</span>
                    </td>
                    <td className="px-6 py-4 max-w-[300px]">
                      <p className="text-[12px] text-ink-soft truncate" title={JSON.stringify(log.details)}>
                        {typeof log.details === 'object' ? JSON.stringify(log.details) : String(log.details)}
                      </p>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex flex-col">
                        <span className="text-[12px] text-ink">{timeAgo(log.created_at)}</span>
                        <span className="text-[10px] text-muted">{new Date(log.created_at).toLocaleTimeString()}</span>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
