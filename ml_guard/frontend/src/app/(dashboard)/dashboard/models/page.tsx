import type { Metadata } from 'next';
import Link from 'next/link';
import { DataTable } from '@/components/ui/DataTable';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Plus } from 'lucide-react';

export const metadata: Metadata = { title: 'Models — Niyantrana Dashboard' };

const MODELS = [
  { id: 'mdl-001', name: 'credit-risk-v4', version: 'v4.2.1', score: 91, verdict: 'CERTIFIED', framework: 'scikit-learn', updated: '2h ago' },
  { id: 'mdl-002', name: 'fraud-detect-prod', version: 'v2.0.0', score: 78, verdict: 'CONDITIONAL', framework: 'XGBoost', updated: '5h ago' },
  { id: 'mdl-003', name: 'churn-predictor', version: 'v3.1.0', score: 94, verdict: 'CERTIFIED', framework: 'PyTorch', updated: '1d ago' },
  { id: 'mdl-004', name: 'llm-support-v2', version: 'v2.0.0', score: 52, verdict: 'FAILED', framework: 'Transformers', updated: '2d ago' },
  { id: 'mdl-005', name: 'image-class-v1', version: 'v1.5.3', score: 88, verdict: 'CERTIFIED', framework: 'TensorFlow', updated: '3d ago' },
  { id: 'mdl-006', name: 'rag-qa-prod', version: 'v1.0.0', score: 73, verdict: 'CONDITIONAL', framework: 'LangChain', updated: '4d ago' },
];

export default function ModelsPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Model Registry</h1>
          <p className="text-[11px] text-muted">Dashboard / Models</p>
        </div>
        <Button variant="primary" size="sm" className="gap-1.5">
          <Plus size={14} strokeWidth={2} />
          Add model
        </Button>
      </div>
      <div className="flex-1 p-8">
        <div className="bg-white border border-stone rounded-card p-6">
          <DataTable
            data={MODELS as unknown as Record<string, unknown>[]}
            columns={[
              {
                key: 'name',
                header: 'Model',
                render: (v, row) => (
                  <Link href={`/dashboard/models/${(row as typeof MODELS[0]).id}`} className="font-medium text-ink hover:text-forest underline underline-offset-4 decoration-transparent hover:decoration-forest transition-all duration-150">
                    {String(v)}
                  </Link>
                ),
              },
              { key: 'version', header: 'Version', render: (v) => <span className="font-mono text-[12px] text-muted">{String(v)}</span> },
              { key: 'framework', header: 'Framework' },
              {
                key: 'score',
                header: 'Score',
                render: (v) => {
                  const n = Number(v);
                  const variant = n >= 80 ? 'certified' : n >= 60 ? 'conditional' : 'failed';
                  return <Badge variant={variant}>{n}</Badge>;
                },
              },
              {
                key: 'verdict',
                header: 'Verdict',
                render: (v) => <Badge variant={String(v).toLowerCase() as 'certified' | 'conditional' | 'failed'}>{String(v)}</Badge>,
              },
              { key: 'updated', header: 'Last audited' },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
