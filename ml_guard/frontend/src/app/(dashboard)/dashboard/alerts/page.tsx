import type { Metadata } from 'next';
import { Badge } from '@/components/ui/Badge';

export const metadata: Metadata = { title: 'Alerts — Niyantrana Dashboard' };

const ALERTS = [
  { title: 'Feature drift detected', model: 'fraud-detect-prod', feature: 'income_ratio', severity: 'danger', time: '1h ago', detail: 'PSI = 0.31 (threshold: 0.25)' },
  { title: 'Contract breach', model: 'llm-support-v2', feature: 'PII Non-Disclosure', severity: 'danger', time: '2h ago', detail: '3 PII tokens leaked in 100 predictions' },
  { title: 'Governance score drop', model: 'rag-qa-prod', feature: 'Overall', severity: 'warning', time: '4h ago', detail: 'Score fell from 82 → 73' },
  { title: 'Audit completed', model: 'credit-risk-v4', feature: 'Full audit', severity: 'certified', time: '5h ago', detail: 'Score: 91/100 — CERTIFIED' },
];

export default function AlertsPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Alert Center</h1>
          <p className="text-[11px] text-muted">Dashboard / Alerts</p>
        </div>
      </div>
      <div className="flex-1 p-8">
        <div className="flex flex-col gap-3 max-w-[800px]">
          {ALERTS.map((alert, i) => (
            <div key={i} className="bg-white border border-stone rounded-card p-5 flex items-start gap-4">
              <Badge variant={alert.severity as 'danger' | 'warning' | 'certified'} className="mt-0.5 flex-shrink-0">
                {alert.severity}
              </Badge>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <p className="text-[14px] font-semibold text-ink">{alert.title}</p>
                  <span className="text-[12px] text-muted">{alert.time}</span>
                </div>
                <p className="text-[13px] text-muted mb-1">{alert.model} · {alert.feature}</p>
                <p className="text-[13px] text-ink-soft">{alert.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
