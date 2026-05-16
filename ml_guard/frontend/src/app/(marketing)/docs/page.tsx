import type { Metadata } from 'next';
import Link from 'next/link';
import { CodeBlock } from '@/components/ui/CodeBlock';
import { ChevronRight } from 'lucide-react';

export const metadata: Metadata = {
  title: 'Documentation — Niyantrana AI Governance',
  description: 'Guides, API reference, and SDK documentation for the Niyantrana platform.',
};

const DOC_SECTIONS = [
  {
    title: 'Getting Started',
    items: [
      { slug: 'quick-start', label: 'Quick start', description: 'Audit your first model in 5 minutes.' },
      { slug: 'installation', label: 'Installation', description: 'SDK setup and authentication.' },
      { slug: 'authentication', label: 'Authentication', description: 'API keys and OAuth flow.' },
    ],
  },
  {
    title: 'Core Concepts',
    items: [
      { slug: 'behavioral-contracts', label: 'Behavioral Contracts', description: 'Define and enforce model promises.' },
      { slug: 'governance-scoring', label: 'Governance Scoring', description: 'How the 0–100 score is calculated.' },
      { slug: 'drift-detection', label: 'Drift Detection', description: 'PSI, KS-Test, Jensen-Shannon.' },
      { slug: 'aibom', label: 'AIBOM', description: 'AI Bill of Materials explained.' },
    ],
  },
  {
    title: 'Integrations',
    items: [
      { slug: 'huggingface', label: 'HuggingFace', description: 'Pull and audit Hub models.' },
      { slug: 'mlflow', label: 'MLflow / W&B', description: 'Sync runs from experiment tracking.' },
      { slug: 'slack-teams', label: 'Slack / Teams', description: 'Governance alerts in your channels.' },
    ],
  },
  {
    title: 'Deployment',
    items: [
      { slug: 'cicd-setup', label: 'CI/CD Setup', description: 'GitHub Actions governance gate.' },
      { slug: 'api-reference', label: 'API Reference', description: 'Full REST API documentation.' },
    ],
  },
];

export default function DocsPage() {
  return (
    <div className="container-site py-16">
      <div className="max-w-[760px] mb-14">
        <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-forest mb-4">Documentation</p>
        <h1 className="text-4xl font-bold text-ink mb-4" style={{ letterSpacing: '-0.03em', lineHeight: 1.1 }}>
          Build governance into your ML workflow.
        </h1>
        <p className="text-[16px] text-ink-soft leading-relaxed">
          Everything you need to audit models, enforce behavioral contracts, and generate cryptographic certificates.
        </p>
      </div>

      {/* Quick start code snippet */}
      <div className="mb-14 max-w-[640px]">
        <p className="text-[13px] font-medium text-ink mb-3">Install the SDK</p>
        <CodeBlock code="pip install niyantrana" language="bash" />
        <div className="mt-3">
          <CodeBlock
            code={`from niyantrana import NiyantranaClient\n\nclient = NiyantranaClient(api_key="niy_...")\nresult = client.audit_model("./my_model.pkl")\nprint(result.governance_score)  # 91.4\nprint(result.verdict)           # CERTIFIED`}
            language="python"
          />
        </div>
      </div>

      {/* Section cards */}
      <div className="grid md:grid-cols-2 gap-8">
        {DOC_SECTIONS.map((section) => (
          <div key={section.title}>
            <h2 className="text-[13px] font-semibold text-ink uppercase tracking-[0.04em] mb-4 pb-2 border-b border-stone">
              {section.title}
            </h2>
            <ul className="flex flex-col gap-1">
              {section.items.map((item) => (
                <li key={item.slug}>
                  <Link
                    href={`/docs/${item.slug}`}
                    className="flex items-center justify-between p-3 rounded-[8px] hover:bg-mist group transition-colors duration-150"
                  >
                    <div>
                      <p className="text-[14px] font-medium text-ink group-hover:text-forest transition-colors duration-150">
                        {item.label}
                      </p>
                      <p className="text-[12px] text-muted">{item.description}</p>
                    </div>
                    <ChevronRight size={14} strokeWidth={1.5} className="text-muted group-hover:text-forest transition-colors duration-150 flex-shrink-0" />
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
