'use client';

import { useState } from 'react';
import { Tabs } from '@/components/ui/Tabs';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { DataTable } from '@/components/ui/DataTable';
import { Download, Check } from 'lucide-react';

const MODEL_TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'drift', label: 'Drift' },
  { id: 'contracts', label: 'Contracts' },
  { id: 'red-team', label: 'Red Team' },
  { id: 'aibom', label: 'AIBOM' },
  { id: 'compliance', label: 'Compliance' },
  { id: 'history', label: 'History' },
];

const CONTRACTS = [
  { name: 'Fairness Parity', type: 'Fairness', status: 'CERTIFIED', breachRate: '0.0%', lastChecked: '2m ago' },
  { name: 'Confidence Threshold ≥ 0.85', type: 'Threshold', status: 'CERTIFIED', breachRate: '0.8%', lastChecked: '2m ago' },
  { name: 'PII Non-Disclosure', type: 'Security', status: 'CERTIFIED', breachRate: '0.0%', lastChecked: '2m ago' },
  { name: 'Max Prediction Latency', type: 'Performance', status: 'CONDITIONAL', breachRate: '3.2%', lastChecked: '2m ago' },
];

const AIBOM_ITEMS = [
  { component: 'Model weights', hash: 'sha256:a3f8c2...', cves: '0 CVEs', type: 'Model' },
  { component: 'scikit-learn 1.3.0', hash: 'sha256:b1d4e7...', cves: '0 CVEs', type: 'Library' },
  { component: 'numpy 1.24.3', hash: 'sha256:c9a2f1...', cves: '1 CVE (low)', type: 'Library' },
  { component: 'Training dataset v3', hash: 'sha256:d7b3e9...', cves: 'N/A', type: 'Dataset' },
];

const RADAR_DATA = [
  { dim: 'Performance', score: 88 },
  { dim: 'Security', score: 96 },
  { dim: 'Fairness', score: 91 },
  { dim: 'Behavioral', score: 89 },
  { dim: 'Compliance', score: 93 },
];

function OverviewTab() {
  return (
    <div className="grid lg:grid-cols-2 gap-6 mt-6">
      {/* Radar chart (simple bar representation) */}
      <div className="bg-white border border-stone rounded-card p-6">
        <h3 className="text-[14px] font-semibold text-ink mb-5">Dimension scores</h3>
        <div className="flex flex-col gap-4">
          {RADAR_DATA.map((d) => (
            <div key={d.dim}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[13px] text-ink-soft">{d.dim}</span>
                <span className="text-[13px] font-semibold text-ink">{d.score}</span>
              </div>
              <div className="h-1.5 bg-stone rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{ width: `${d.score}%`, background: '#1A5F3A' }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Score summary */}
      <div className="bg-white border border-stone rounded-card p-6">
        <h3 className="text-[14px] font-semibold text-ink mb-5">Audit summary</h3>
        <div className="grid grid-cols-2 gap-4">
          {[
            { label: 'Governance score', value: '91 / 100' },
            { label: 'Verdict', value: 'CERTIFIED' },
            { label: 'Total predictions', value: '847,291' },
            { label: 'Breach rate', value: '0.8%' },
            { label: 'Latency (p50)', value: '38ms' },
            { label: 'Certificate', value: 'SHA-256 sealed' },
          ].map((s) => (
            <div key={s.label}>
              <p className="text-[11px] text-muted uppercase tracking-[0.04em] mb-0.5">{s.label}</p>
              <p className="text-[14px] font-semibold text-ink">{s.value}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ContractsTab() {
  return (
    <div className="mt-6 bg-white border border-stone rounded-card p-6">
      <h3 className="text-[14px] font-semibold text-ink mb-5">Active behavioral contracts</h3>
      <DataTable
        data={CONTRACTS as unknown as Record<string, unknown>[]}
        columns={[
          { key: 'name', header: 'Contract', render: (v) => <span className="font-medium text-ink">{String(v)}</span> },
          { key: 'type', header: 'Type' },
          { key: 'status', header: 'Status', render: (v) => <Badge variant={String(v).toLowerCase() as 'certified' | 'conditional'}>{String(v)}</Badge> },
          { key: 'breachRate', header: 'Breach rate' },
          { key: 'lastChecked', header: 'Last checked' },
        ]}
      />
    </div>
  );
}

function AIBOMTab() {
  return (
    <div className="mt-6 bg-white border border-stone rounded-card p-6">
      <h3 className="text-[14px] font-semibold text-ink mb-5">AI Bill of Materials</h3>
      <DataTable
        data={AIBOM_ITEMS as unknown as Record<string, unknown>[]}
        columns={[
          { key: 'component', header: 'Component', render: (v) => <span className="font-medium text-ink">{String(v)}</span> },
          { key: 'type', header: 'Type' },
          { key: 'hash', header: 'SHA-256', render: (v) => <span className="font-mono text-[12px] text-muted">{String(v)}</span> },
          {
            key: 'cves',
            header: 'CVEs',
            render: (v) => {
              const s = String(v);
              return <span className={s === '0 CVEs' || s === 'N/A' ? 'text-forest text-[13px]' : 'text-warning text-[13px]'}>{s}</span>;
            },
          },
        ]}
      />
    </div>
  );
}

function PlaceholderTab({ label }: { label: string }) {
  return (
    <div className="mt-6 bg-white border border-stone rounded-card p-10 text-center">
      <p className="text-[14px] text-muted">{label} data available after next audit run.</p>
    </div>
  );
}

export default function ModelDetailClient({ modelId }: { modelId: string }) {
  const [activeTab, setActiveTab] = useState('overview');

  const tabContent: Record<string, React.ReactNode> = {
    overview: <OverviewTab />,
    contracts: <ContractsTab />,
    aibom: <AIBOMTab />,
    drift: <PlaceholderTab label="Drift monitoring" />,
    'red-team': <PlaceholderTab label="Red team sessions" />,
    compliance: <PlaceholderTab label="Compliance mapping" />,
    history: <PlaceholderTab label="Audit history" />,
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Top bar */}
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">credit-risk-v4 <span className="font-mono text-[13px] text-muted ml-2">v4.2.1</span></h1>
          <p className="text-[11px] text-muted">Dashboard / Models / {modelId}</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="certified">CERTIFIED</Badge>
          <Button variant="ghost" size="sm" className="gap-1.5">
            <Download size={14} strokeWidth={1.5} />
            Download certificate
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <div className="px-8 bg-white border-b border-stone">
        <Tabs tabs={MODEL_TABS} activeTab={activeTab} onTabChange={setActiveTab} />
      </div>

      {/* Tab content */}
      <div className="flex-1 p-8">
        {tabContent[activeTab]}
      </div>
    </div>
  );
}
