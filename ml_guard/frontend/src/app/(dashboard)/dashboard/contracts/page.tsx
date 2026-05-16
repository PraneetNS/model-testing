import type { Metadata } from 'next';
import { DataTable } from '@/components/ui/DataTable';
import { Badge } from '@/components/ui/Badge';

export const metadata: Metadata = { title: 'Contracts — Niyantrana Dashboard' };

const CONTRACTS = [
  { name: 'Fairness Parity', model: 'credit-risk-v4', type: 'Fairness', status: 'CERTIFIED', breachRate: '0.0%', checked: '2m ago' },
  { name: 'Confidence Threshold', model: 'fraud-detect-prod', type: 'Threshold', status: 'CONDITIONAL', breachRate: '3.2%', checked: '2m ago' },
  { name: 'PII Non-Disclosure', model: 'llm-support-v2', type: 'Security', status: 'FAILED', breachRate: '12.1%', checked: '5m ago' },
  { name: 'Max Latency < 100ms', model: 'churn-predictor', type: 'Performance', status: 'CERTIFIED', breachRate: '0.1%', checked: '5m ago' },
];

export default function ContractsPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Behavioral Contracts</h1>
          <p className="text-[11px] text-muted">Dashboard / Contracts</p>
        </div>
      </div>
      <div className="flex-1 p-8">
        <div className="bg-white border border-stone rounded-card p-6">
          <DataTable
            data={CONTRACTS as unknown as Record<string, unknown>[]}
            columns={[
              { key: 'name', header: 'Contract', render: (v) => <span className="font-medium text-ink">{String(v)}</span> },
              { key: 'model', header: 'Model' },
              { key: 'type', header: 'Type' },
              { key: 'status', header: 'Status', render: (v) => <Badge variant={String(v).toLowerCase() as 'certified' | 'conditional' | 'failed'}>{String(v)}</Badge> },
              { key: 'breachRate', header: 'Breach rate' },
              { key: 'checked', header: 'Last checked' },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
