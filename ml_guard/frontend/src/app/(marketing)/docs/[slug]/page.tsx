import type { Metadata } from 'next';
import Link from 'next/link';
import { CodeBlock } from '@/components/ui/CodeBlock';
import { ChevronRight } from 'lucide-react';

const DOC_CONTENT: Record<string, { title: string; body: React.ReactNode }> = {
  'quick-start': {
    title: 'Quick Start',
    body: (
      <div className="prose-doc">
        <p>Audit your first model in under 5 minutes.</p>
        <h2>1. Install the SDK</h2>
        <CodeBlock code="pip install niyantrana" language="bash" />
        <h2>2. Authenticate</h2>
        <CodeBlock code={`export NIYANTRANA_API_KEY="niy_your_key_here"`} language="bash" />
        <h2>3. Run your first audit</h2>
        <CodeBlock
          code={`from niyantrana import NiyantranaClient\n\nclient = NiyantranaClient()\nresult = client.audit_model(\n    model_path="./my_model.pkl",\n    test_data="./test.csv",\n    contracts=["fairness_parity", "confidence_threshold"]\n)\n\nprint(result.governance_score)  # 91.4\nprint(result.verdict)           # CERTIFIED\nresult.download_certificate("./cert.pdf")`}
          language="python"
        />
        <h2>4. Check the dashboard</h2>
        <p>Log in to <Link href="/dashboard" className="text-forest underline underline-offset-4">your dashboard</Link> to view the full report, drift trends, and compliance mappings.</p>
      </div>
    ),
  },
  'behavioral-contracts': {
    title: 'Behavioral Contracts',
    body: (
      <div className="prose-doc">
        <p>A Behavioral Contract is a machine-readable promise your model must keep. Every prediction is validated against it in real time.</p>
        <h2>Defining a contract</h2>
        <CodeBlock
          code={`contracts:\n  - name: confidence_threshold\n    type: threshold\n    metric: prediction_confidence\n    operator: ">="  \n    value: 0.85\n    on_breach: log_and_alert\n\n  - name: fairness_parity\n    type: fairness\n    metric: demographic_parity\n    groups: [age_group, gender]\n    max_disparity: 0.05\n    on_breach: block`}
          language="python"
        />
        <h2>Breach handling</h2>
        <p>When a contract is breached, Niyantrana can: <code className="text-code">log</code>, <code className="text-code">alert</code>, or <code className="text-code">block</code> the prediction. Each breach is recorded with a cryptographic timestamp.</p>
      </div>
    ),
  },
  'aibom': {
    title: 'AIBOM — AI Bill of Materials',
    body: (
      <div className="prose-doc">
        <p>An AIBOM is a structured inventory of everything your model depends on, required under EU AI Act Article 13.</p>
        <h2>Generate an AIBOM</h2>
        <CodeBlock
          code={`niyantrana aibom generate --model ./my_model.pkl --output aibom.json`}
          language="bash"
        />
        <h2>AIBOM contents</h2>
        <ul>
          <li>Model weights SHA-256 hash</li>
          <li>Training dataset provenance and hashes</li>
          <li>All Python dependencies with CVE status</li>
          <li>Framework versions (scikit-learn, PyTorch, etc.)</li>
        </ul>
      </div>
    ),
  },
};

const DEFAULT_DOC = {
  title: 'Documentation',
  body: <p>This documentation page is coming soon. <Link href="/docs" className="text-forest underline underline-offset-4">Return to docs index</Link>.</p>,
};

const NAV = [
  { slug: 'quick-start', label: 'Quick start' },
  { slug: 'installation', label: 'Installation' },
  { slug: 'authentication', label: 'Authentication' },
  { slug: 'behavioral-contracts', label: 'Behavioral Contracts' },
  { slug: 'governance-scoring', label: 'Governance Scoring' },
  { slug: 'drift-detection', label: 'Drift Detection' },
  { slug: 'aibom', label: 'AIBOM' },
  { slug: 'api-reference', label: 'API Reference' },
  { slug: 'cicd-setup', label: 'CI/CD Setup' },
  { slug: 'huggingface', label: 'HuggingFace' },
  { slug: 'mlflow', label: 'MLflow / W&B' },
  { slug: 'slack-teams', label: 'Slack / Teams' },
];

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const doc = DOC_CONTENT[slug] ?? DEFAULT_DOC;
  return {
    title: `${doc.title} — Niyantrana Docs`,
    description: `Niyantrana documentation: ${doc.title}`,
  };
}

export default async function DocSlugPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const doc = DOC_CONTENT[slug] ?? DEFAULT_DOC;

  return (
    <div className="flex min-h-screen">
      {/* Left sidebar */}
      <aside className="hidden lg:block w-[240px] flex-shrink-0 border-r border-stone bg-white">
        <div className="sticky top-16 p-5 overflow-y-auto max-h-[calc(100vh-64px)]">
          <p className="text-[10px] font-bold uppercase tracking-[0.08em] text-muted mb-3">Documentation</p>
          <nav className="flex flex-col gap-0.5">
            {NAV.map((item) => (
              <Link
                key={item.slug}
                href={`/docs/${item.slug}`}
                className={`px-3 py-2 text-[13px] rounded-[6px] transition-colors duration-150 ${
                  item.slug === slug
                    ? 'bg-mist text-forest font-medium'
                    : 'text-ink-soft hover:text-ink hover:bg-ivory'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 px-8 py-12 max-w-[760px]">
        {/* Breadcrumb */}
        <div className="flex items-center gap-1 text-[12px] text-muted mb-8">
          <Link href="/docs" className="hover:text-ink transition-colors duration-150">Docs</Link>
          <ChevronRight size={12} strokeWidth={1.5} />
          <span className="text-ink-soft">{doc.title}</span>
        </div>

        <h1 className="text-3xl font-bold text-ink mb-6" style={{ letterSpacing: '-0.02em' }}>
          {doc.title}
        </h1>

        <div className="text-[15px] text-ink-soft leading-relaxed [&_h2]:text-xl [&_h2]:font-semibold [&_h2]:text-ink [&_h2]:mt-8 [&_h2]:mb-3 [&_h2]:tracking-[-0.01em] [&_p]:mb-4 [&_ul]:mb-4 [&_ul]:pl-4 [&_li]:mb-1.5 [&_li]:list-disc">
          {doc.body}
        </div>
      </main>
    </div>
  );
}
