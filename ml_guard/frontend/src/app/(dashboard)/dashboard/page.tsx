import type { Metadata } from 'next';
import { StatCard } from '@/components/ui/StatCard';
import { DataTable } from '@/components/ui/DataTable';
import { Badge } from '@/components/ui/Badge';
import { Bell } from 'lucide-react';

export const metadata: Metadata = { title: 'Overview — Niyantrana Dashboard' };

const RECENT_MODELS = [
  { name: 'credit-risk-v4', score: 91, verdict: 'CERTIFIED', audited: '2 hours ago' },
  { name: 'fraud-detect-prod', score: 78, verdict: 'CONDITIONAL', audited: '5 hours ago' },
  { name: 'churn-predictor', score: 94, verdict: 'CERTIFIED', audited: '1 day ago' },
  { name: 'llm-support-v2', score: 52, verdict: 'FAILED', audited: '2 days ago' },
  { name: 'image-class-v1', score: 88, verdict: 'CERTIFIED', audited: '3 days ago' },
];

const ALERTS = [
  { title: 'Drift detected', model: 'fraud-detect-prod', severity: 'warning', time: '1h ago' },
  { title: 'Contract breach', model: 'llm-support-v2', severity: 'danger', time: '2h ago' },
  { title: 'Audit completed', model: 'credit-risk-v4', severity: 'certified', time: '2h ago' },
];

function ScorePill({ score }: { score: number }) {
  const variant = score >= 80 ? 'certified' : score >= 60 ? 'conditional' : 'failed';
  return <Badge variant={variant}>{score}</Badge>;
}

function VerdictBadge({ verdict }: { verdict: string }) {
  const v = verdict.toLowerCase() as 'certified' | 'conditional' | 'failed';
  return <Badge variant={v}>{verdict}</Badge>;
}

export default function DashboardOverviewPage() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Top bar */}
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Overview</h1>
          <p className="text-[11px] text-muted">Dashboard / Overview</p>
        </div>
        <div className="flex items-center gap-3">
          <button className="relative text-muted hover:text-ink transition-colors duration-150" aria-label="Notifications">
            <Bell size={18} strokeWidth={1.5} />
            <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-danger rounded-full" />
          </button>
          <div className="w-8 h-8 rounded-full bg-forest flex items-center justify-center text-white text-[11px] font-semibold">PR</div>
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 p-8">
        {/* Stat cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard label="Total Models" value="12" trend="up" trendLabel="+2 this week" />
          <StatCard label="Active Contracts" value="47" trend="up" trendLabel="+5 this month" />
          <StatCard label="Alerts Today" value="3" trend="down" trendLabel="-1 vs yesterday" />
          <StatCard label="Avg Gov. Score" value="91.4" trend="up" trendLabel="+1.2 pts" />
        </div>

        {/* 2-col grid */}
        <div className="grid lg:grid-cols-[60%_40%] gap-6 mb-8">
          {/* Recent model activity */}
          <div className="bg-white border border-stone rounded-card p-6">
            <h2 className="text-[14px] font-semibold text-ink mb-5">Recent model activity</h2>
            <DataTable
              data={RECENT_MODELS as Record<string, unknown>[]}
              columns={[
                { key: 'name', header: 'Model', render: (v) => <span className="font-medium text-ink">{String(v)}</span> },
                { key: 'score', header: 'Score', render: (v) => <ScorePill score={Number(v)} /> },
                { key: 'verdict', header: 'Verdict', render: (v) => <VerdictBadge verdict={String(v)} /> },
                { key: 'audited', header: 'Last audited' },
              ]}
            />
          </div>

          {/* Active alerts */}
          <div className="bg-white border border-stone rounded-card p-6">
            <h2 className="text-[14px] font-semibold text-ink mb-5">Active alerts</h2>
            <div className="flex flex-col gap-3">
              {ALERTS.map((alert, i) => (
                <div key={i} className="flex items-start gap-3 pb-3 border-b border-stone/50 last:border-0">
                  <Badge variant={alert.severity as 'warning' | 'danger' | 'certified'} className="mt-0.5 flex-shrink-0">
                    {alert.severity}
                  </Badge>
                  <div className="flex-1 min-w-0">
                    <p className="text-[13px] font-medium text-ink">{alert.title}</p>
                    <p className="text-[12px] text-muted">{alert.model}</p>
                  </div>
                  <span className="text-[11px] text-muted flex-shrink-0">{alert.time}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Governance score trend (placeholder chart) */}
        <div className="bg-white border border-stone rounded-card p-6">
          <h2 className="text-[14px] font-semibold text-ink mb-5">Governance score trend</h2>
          <div className="h-[160px] flex items-end gap-2">
            {[78, 82, 79, 85, 88, 87, 90, 91, 89, 91, 93, 91].map((v, i) => (
              <div key={i} className="flex-1 flex flex-col items-center justify-end gap-1">
                <div
                  className="w-full rounded-t-sm transition-all duration-300"
                  style={{
                    height: `${(v / 100) * 140}px`,
                    background: v >= 80 ? '#1A5F3A' : v >= 60 ? '#B35A00' : '#C0392B',
                    opacity: 0.8,
                  }}
                />
                <span className="text-[9px] text-muted">{['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'][i]}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
