'use client';

import { useState } from 'react';
import { Tabs } from '@/components/ui/Tabs';
import { Button } from '@/components/ui/Button';
import { DataTable } from '@/components/ui/DataTable';
import { Badge } from '@/components/ui/Badge';
import { CodeBlock } from '@/components/ui/CodeBlock';
import { Plus, Eye, EyeOff, Copy, Check } from 'lucide-react';

const SETTINGS_TABS = [
  { id: 'api-keys', label: 'API Keys' },
  { id: 'notifications', label: 'Notifications' },
  { id: 'integrations', label: 'Integrations' },
  { id: 'team', label: 'Team' },
];

const API_KEYS = [
  { label: 'Production key', scopes: 'read, write, audit', lastUsed: '1h ago', expires: 'Never' },
  { label: 'CI/CD key', scopes: 'audit, gate', lastUsed: '3h ago', expires: '2026-12-31' },
  { label: 'Read-only key', scopes: 'read', lastUsed: '2d ago', expires: 'Never' },
];

function ApiKeysTab() {
  const [showNew, setShowNew] = useState(false);
  const [copied, setCopied] = useState(false);
  const newKey = 'niy_live_a3f8c2b1d4e7f9c2b1d4e7f9c2b1d4e7';

  const handleCopy = async () => {
    await navigator.clipboard.writeText(newKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-5">
        <div>
          <h3 className="text-[15px] font-semibold text-ink">API Keys</h3>
          <p className="text-[13px] text-muted mt-0.5">Manage keys for SDK and CI/CD access.</p>
        </div>
        <Button variant="primary" size="sm" className="gap-1.5" onClick={() => setShowNew(true)}>
          <Plus size={14} strokeWidth={2} />
          Create new key
        </Button>
      </div>

      {/* New key modal */}
      {showNew && (
        <div className="bg-mist border border-forest/20 rounded-card p-5 mb-6">
          <p className="text-[13px] font-semibold text-ink mb-2">⚠️ Copy this now — it will never be shown again.</p>
          <div className="flex items-center gap-2">
            <code className="flex-1 text-[13px] font-mono bg-white border border-stone rounded-[6px] px-3 py-2 text-ink overflow-x-auto">
              {newKey}
            </code>
            <button onClick={handleCopy} className="flex items-center gap-1.5 text-[12px] text-forest hover:text-ink-soft transition-colors duration-150">
              {copied ? <Check size={14} strokeWidth={2} /> : <Copy size={14} strokeWidth={1.5} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
          <button onClick={() => setShowNew(false)} className="mt-3 text-[12px] text-muted underline underline-offset-4">
            I've saved it, dismiss
          </button>
        </div>
      )}

      <div className="bg-white border border-stone rounded-card p-0 overflow-hidden">
        <DataTable
          data={API_KEYS as unknown as Record<string, unknown>[]}
          columns={[
            { key: 'label', header: 'Label', render: (v) => <span className="font-medium text-ink">{String(v)}</span> },
            { key: 'scopes', header: 'Scopes', render: (v) => <span className="font-mono text-[12px] text-muted">{String(v)}</span> },
            { key: 'lastUsed', header: 'Last used' },
            { key: 'expires', header: 'Expires' },
            {
              key: 'label',
              header: '',
              render: () => (
                <button className="text-[12px] text-danger hover:underline underline-offset-4">Revoke</button>
              ),
            },
          ]}
        />
      </div>
    </div>
  );
}

function PlaceholderSettings({ label }: { label: string }) {
  return (
    <div className="py-10 text-center">
      <p className="text-[14px] text-muted">{label} settings — coming soon.</p>
    </div>
  );
}

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState('api-keys');

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Settings</h1>
          <p className="text-[11px] text-muted">Dashboard / Settings</p>
        </div>
      </div>

      <div className="px-8 bg-white border-b border-stone">
        <Tabs tabs={SETTINGS_TABS} activeTab={activeTab} onTabChange={setActiveTab} />
      </div>

      <div className="flex-1 p-8">
        <div className="max-w-[800px]">
          {activeTab === 'api-keys' && <ApiKeysTab />}
          {activeTab === 'notifications' && <PlaceholderSettings label="Notification" />}
          {activeTab === 'integrations' && <PlaceholderSettings label="Integration" />}
          {activeTab === 'team' && <PlaceholderSettings label="Team" />}
        </div>
      </div>
    </div>
  );
}
