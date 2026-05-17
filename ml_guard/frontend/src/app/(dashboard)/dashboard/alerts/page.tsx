'use client';

import { useEffect, useState, useCallback } from 'react';
import { RefreshCw, Bell, BellOff, Check, Filter } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { alertsApi, type AlertEvent, type AlertRule } from '@/lib/api';

function SeverityBadge({ sev }: { sev: string }) {
  const v: 'failed' | 'conditional' | 'certified' =
    sev === 'HIGH' || sev === 'CRITICAL' ? 'failed' : sev === 'MEDIUM' ? 'conditional' : 'certified';
  return <Badge variant={v}>{sev}</Badge>;
}

function timeAgo(ts: string) {
  const diff = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export default function AlertsPage() {
  const [events, setEvents] = useState<AlertEvent[]>([]);
  const [rules, setRules] = useState<AlertRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'active' | 'resolved' | 'all'>('active');
  const [activeTab, setActiveTab] = useState<'events' | 'rules'>('events');
  const [resolving, setResolving] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resolved = filter === 'resolved' ? true : filter === 'active' ? false : undefined;
      const [evRes, rulesRes] = await Promise.all([
        alertsApi.listEvents(100, resolved),
        alertsApi.listRules(),
      ]);
      setEvents(evRes.items ?? []);
      setRules(rulesRes.items ?? []);
    } catch (e: any) {
      setError(e.message ?? 'Failed to load alerts');
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => { load(); }, [load]);

  const resolveAlert = async (id: string) => {
    setResolving(id);
    try {
      await alertsApi.resolve(id);
      setEvents(prev => prev.map(e => e.id === id ? { ...e, resolved: true } : e));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setResolving(null);
    }
  };

  const active = events.filter(e => !e.resolved).length;
  const resolved = events.filter(e => e.resolved).length;

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Alert Center</h1>
          <p className="text-[11px] text-muted">{active} active · {resolved} resolved</p>
        </div>
        <button onClick={load} className="text-muted hover:text-ink transition-colors" aria-label="Refresh">
          <RefreshCw size={15} strokeWidth={1.5} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex px-8 bg-white border-b border-stone gap-6">
        {(['events', 'rules'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`py-3 text-[13px] font-medium border-b-2 transition-colors capitalize -mb-px ${
              activeTab === tab ? 'border-forest text-forest' : 'border-transparent text-muted hover:text-ink'
            }`}
          >
            {tab === 'events' ? `Events ${active > 0 ? `(${active})` : ''}` : `Alert Rules (${rules.length})`}
          </button>
        ))}
      </div>

      <div className="flex-1 p-8">
        {error && (
          <div className="mb-5 p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger flex items-center justify-between">
            <span>⚠ {error}</span>
            <button onClick={load} className="text-forest underline text-[12px]">Retry</button>
          </div>
        )}

        {activeTab === 'events' && (
          <>
            {/* Filter */}
            <div className="flex items-center gap-2 mb-5">
              <Filter size={13} className="text-muted" strokeWidth={1.5} />
              {(['active', 'resolved', 'all'] as const).map(f => (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`px-3 py-1 text-[12px] font-medium rounded-badge transition-colors capitalize ${
                    filter === f ? 'bg-forest text-white' : 'bg-stone text-ink-soft hover:bg-[#ddd9d3]'
                  }`}
                >
                  {f}
                </button>
              ))}
            </div>

            {loading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="bg-white border border-stone rounded-card p-5 mb-3 animate-pulse flex gap-4">
                  <div className="h-5 w-16 bg-stone rounded-badge" />
                  <div className="flex-1">
                    <div className="h-3 bg-stone rounded-full w-3/4 mb-2" />
                    <div className="h-2.5 bg-stone rounded-full w-1/2" />
                  </div>
                </div>
              ))
            ) : events.length === 0 ? (
              <div className="text-center py-16">
                <BellOff size={32} className="mx-auto text-stone mb-3" strokeWidth={1} />
                <p className="text-[14px] font-medium text-ink mb-1">No alerts</p>
                <p className="text-[13px] text-muted">System is healthy — no {filter} alerts.</p>
              </div>
            ) : (
              <div className="flex flex-col gap-3 max-w-[900px]">
                {events.map(alert => (
                  <div
                    key={alert.id}
                    className={`bg-white border rounded-card p-5 flex items-start gap-4 transition-all ${
                      alert.resolved ? 'opacity-50 border-stone' : 'border-stone hover:shadow-sm'
                    }`}
                  >
                    <SeverityBadge sev={alert.severity} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-3">
                        <p className="text-[14px] font-medium text-ink">{alert.message}</p>
                        <span className="text-[11px] text-muted flex-shrink-0">{timeAgo(alert.created_at)}</span>
                      </div>
                      <p className="text-[12px] text-muted mt-0.5">Alert ID: {alert.id.slice(0, 8)}…</p>
                    </div>
                    {!alert.resolved && (
                      <button
                        onClick={() => resolveAlert(alert.id)}
                        disabled={resolving === alert.id}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium text-forest border border-forest rounded-[6px] hover:bg-mist transition-colors disabled:opacity-50"
                      >
                        {resolving === alert.id ? <RefreshCw size={11} className="animate-spin" /> : <Check size={11} strokeWidth={2} />}
                        Resolve
                      </button>
                    )}
                    {alert.resolved && (
                      <span className="flex items-center gap-1 text-[11px] text-forest">
                        <Check size={11} strokeWidth={2} /> Resolved
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {activeTab === 'rules' && (
          <div className="max-w-[900px]">
            {loading ? (
              Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="bg-white border border-stone rounded-card p-5 mb-3 animate-pulse h-20" />
              ))
            ) : rules.length === 0 ? (
              <div className="text-center py-16">
                <Bell size={32} className="mx-auto text-stone mb-3" strokeWidth={1} />
                <p className="text-[14px] font-medium text-ink mb-1">No alert rules configured</p>
                <p className="text-[13px] text-muted">Create alert rules in the backend to start receiving notifications.</p>
              </div>
            ) : (
              <div className="bg-white border border-stone rounded-card overflow-hidden">
                <table className="w-full border-collapse">
                  <thead className="bg-[#F7F6F2]">
                    <tr>
                      {['Rule name', 'Metric', 'Severity', 'Status', 'Created'].map(h => (
                        <th key={h} className="text-left px-5 py-3 text-[11px] font-semibold uppercase tracking-[0.04em] text-muted border-b border-stone">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rules.map(rule => (
                      <tr key={rule.id} className="border-b border-stone/50 hover:bg-[#F0EDE8] transition-colors">
                        <td className="px-5 py-3.5 text-[13px] font-medium text-ink">{rule.name}</td>
                        <td className="px-5 py-3.5 text-[13px] text-muted font-mono">{rule.metric}</td>
                        <td className="px-5 py-3.5"><SeverityBadge sev={rule.severity} /></td>
                        <td className="px-5 py-3.5">
                          <span className={`text-[11px] font-medium px-2 py-0.5 rounded-badge ${rule.is_active ? 'bg-mist text-forest' : 'bg-stone text-muted'}`}>
                            {rule.is_active ? 'Active' : 'Inactive'}
                          </span>
                        </td>
                        <td className="px-5 py-3.5 text-[12px] text-muted">{new Date(rule.created_at).toLocaleDateString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
