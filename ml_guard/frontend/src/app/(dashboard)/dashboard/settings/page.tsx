'use client';

import { useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { Button } from '@/components/ui/Button';
import { Key, User, Bell, Shield, Database, RefreshCw, Check } from 'lucide-react';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

function Section({ title, icon: Icon, children }: { title: string; icon: any; children: React.ReactNode }) {
  return (
    <div className="bg-white border border-stone rounded-card overflow-hidden">
      <div className="flex items-center gap-3 px-6 py-4 border-b border-stone bg-[#F7F6F2]">
        <Icon size={15} strokeWidth={1.5} className="text-forest" />
        <h2 className="text-[14px] font-semibold text-ink">{title}</h2>
      </div>
      <div className="p-6">{children}</div>
    </div>
  );
}

function Field({ label, value, type = 'text', readOnly = false, onChange }: {
  label: string; value: string; type?: string; readOnly?: boolean; onChange?: (v: string) => void;
}) {
  return (
    <div className="mb-4">
      <label className="block text-[12px] font-medium text-ink-soft mb-1.5">{label}</label>
      <input type={type} value={value} readOnly={readOnly} onChange={e => onChange?.(e.target.value)}
        className={`w-full h-10 px-3 text-[13px] border border-stone rounded-[8px] outline-none transition-colors ${
          readOnly ? 'bg-[#F7F6F2] text-muted cursor-default' : 'bg-white text-ink focus:border-forest'
        }`} />
    </div>
  );
}

export default function SettingsPage() {
  const { user, logout } = useAuth();
  const [apiUrl, setApiUrl] = useState(BASE_URL);
  const [apiKey, setApiKey] = useState('');
  const [saved, setSaved] = useState(false);
  const [testing, setTesting] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'unknown' | 'online' | 'offline'>('unknown');

  const saveSettings = () => {
    if (apiKey) localStorage.setItem('niyantrana_token', apiKey);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const testConnection = async () => {
    setTesting(true);
    try {
      const res = await fetch(`${apiUrl}/drift/health`, { signal: AbortSignal.timeout(5000) });
      setBackendStatus(res.ok ? 'online' : 'offline');
    } catch {
      setBackendStatus('offline');
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <div className="px-8 h-16 border-b border-stone bg-white flex items-center justify-between">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Settings</h1>
          <p className="text-[11px] text-muted">Platform configuration and account settings</p>
        </div>
        <Button variant="primary" size="sm" className="gap-2" onClick={saveSettings}>
          {saved ? <><Check size={13} />Saved!</> : 'Save changes'}
        </Button>
      </div>

      <div className="flex-1 p-8 space-y-5 max-w-[760px]">
        {/* Account */}
        <Section title="Account" icon={User}>
          <Field label="Display name" value={user?.displayName ?? ''} readOnly />
          <Field label="Email" value={user?.email ?? ''} readOnly />
          <Field label="User ID" value={user?.uid ?? ''} readOnly />
          <div className="mt-4 pt-4 border-t border-stone">
            <Button variant="ghost" size="sm" onClick={() => logout()} className="text-danger hover:text-danger hover:bg-red-50">
              Sign out
            </Button>
          </div>
        </Section>

        {/* Backend connection */}
        <Section title="Backend Connection" icon={Database}>
          <Field label="API Base URL" value={apiUrl} onChange={setApiUrl} />
          <div className="flex items-center gap-3 mt-2">
            <Button variant="ghost" size="sm" className="gap-2" onClick={testConnection} disabled={testing}>
              {testing ? <><RefreshCw size={12} className="animate-spin" />Testing…</> : 'Test connection'}
            </Button>
            {backendStatus !== 'unknown' && (
              <span className={`flex items-center gap-1.5 text-[12px] font-medium ${backendStatus === 'online' ? 'text-forest' : 'text-danger'}`}>
                <span className={`w-2 h-2 rounded-full ${backendStatus === 'online' ? 'bg-forest' : 'bg-danger'}`} />
                {backendStatus === 'online' ? 'Backend online' : 'Backend offline'}
              </span>
            )}
          </div>
        </Section>

        {/* API Key */}
        <Section title="Authentication" icon={Key}>
          <p className="text-[12px] text-muted mb-4">
            Set a backend API token to authenticate requests. This is stored in localStorage.
          </p>
          <Field label="Backend API token" value={apiKey} type="password" onChange={setApiKey}
          />
          <p className="text-[11px] text-muted mt-1">Leave blank to use unauthenticated mode (dev only).</p>
        </Section>

        {/* Notifications */}
        <Section title="Notifications" icon={Bell}>
          <div className="space-y-3">
            {[
              { label: 'Drift alerts', desc: 'Notify when drift score exceeds threshold', key: 'drift_alerts' },
              { label: 'Contract breaches', desc: 'Notify on behavioral contract violations', key: 'contract_alerts' },
              { label: 'Audit completions', desc: 'Notify when a governance audit finishes', key: 'audit_alerts' },
              { label: 'Security events', desc: 'Notify on red team findings or security flags', key: 'security_alerts' },
            ].map(n => (
              <label key={n.key} className="flex items-start gap-3 cursor-pointer group">
                <input type="checkbox" defaultChecked className="mt-0.5 accent-forest" />
                <div>
                  <p className="text-[13px] font-medium text-ink group-hover:text-forest transition-colors">{n.label}</p>
                  <p className="text-[11px] text-muted">{n.desc}</p>
                </div>
              </label>
            ))}
          </div>
        </Section>

        {/* Governance thresholds */}
        <Section title="Governance Thresholds" icon={Shield}>
          <p className="text-[12px] text-muted mb-4">Configure default pass/fail thresholds for audits.</p>
          <div className="grid md:grid-cols-2 gap-4">
            {[
              { label: 'Min. governance score (PASS)', key: 'gov_pass', defaultVal: '75' },
              { label: 'Max. drift PSI (WARNING)', key: 'drift_warn', defaultVal: '0.15' },
              { label: 'Max. drift PSI (CRITICAL)', key: 'drift_crit', defaultVal: '0.25' },
              { label: 'Max. overfitting gap', key: 'overfit', defaultVal: '0.10' },
            ].map(t => (
              <div key={t.key}>
                <label className="block text-[12px] font-medium text-ink-soft mb-1.5">{t.label}</label>
                <input type="number" step="0.01" defaultValue={t.defaultVal}
                  className="w-full h-9 px-3 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest" />
              </div>
            ))}
          </div>
          <p className="text-[11px] text-muted mt-3">Note: Threshold changes apply to future audits only.</p>
        </Section>

        {/* About */}
        <div className="bg-[#F7F6F2] border border-stone rounded-card px-6 py-4">
          <p className="text-[12px] font-medium text-ink mb-1">Niyantrana Platform</p>
          <p className="text-[11px] text-muted">AI Governance · Model Auditing · Drift Detection</p>
          <p className="text-[11px] text-muted mt-1">API: <span className="font-mono">{BASE_URL}</span></p>
        </div>
      </div>
    </div>
  );
}
