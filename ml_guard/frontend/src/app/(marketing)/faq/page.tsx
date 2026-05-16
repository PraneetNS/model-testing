import type { Metadata } from 'next';
import { Accordion } from '@/components/ui/Accordion';

export const metadata: Metadata = {
  title: 'FAQ — Niyantrana AI Governance',
  description: "Answers to common questions about Niyantrana's governance platform.",
};

const FAQ_ITEMS = [
  { question: 'What is a Behavioral Contract?', answer: 'A Behavioral Contract is a machine-readable promise your model must keep. You define conditions — e.g. "confidence must be ≥ 0.85 on protected groups" — and Niyantrana validates every prediction against those conditions in real time, logging any breach with a cryptographic timestamp.' },
  { question: 'How does the governance score work?', answer: 'The governance score (0–100) is a weighted aggregate across five dimensions: Performance, Fairness, Security, Behavioral Compliance, and Regulatory Alignment. Each dimension is scored independently, and the weights can be customised per your organisation\'s risk policy.' },
  { question: 'What is an AIBOM?', answer: 'An AI Bill of Materials (AIBOM) is a structured inventory of everything your model depends on: model weights (SHA-256 hash), training datasets, third-party libraries, and their known CVEs. It gives you full supply chain visibility and is required under EU AI Act Article 13.' },
  { question: 'How does Niyantrana differ from Evidently AI?', answer: 'Evidently AI focuses on data and model monitoring dashboards. Niyantrana adds cryptographic audit certificates, behavioral contract enforcement, CI/CD governance gates, LLM red teaming, and actuarial insurance scoring — making it a full governance platform, not just a monitoring tool.' },
  { question: 'Can I pull models directly from HuggingFace?', answer: 'Yes. Niyantrana integrates with the HuggingFace Hub API. You provide a model ID and revision, and the platform pulls, hashes, and audits the model automatically.' },
  { question: 'What compliance frameworks does Niyantrana map to?', answer: 'Niyantrana currently maps to EU AI Act, NIST AI RMF (Govern, Map, Measure, Manage), ISO/IEC 42001, SOC 2 Type II, and GDPR Article 22. Each governance report card includes a framework-specific compliance section.' },
  { question: 'How does the CI/CD gate work?', answer: 'Install the Niyantrana GitHub Action in your workflow. When a model is submitted, the gate calls the Niyantrana API, runs the full audit, and returns a pass/fail. If the governance score is below your threshold or a behavioral contract is violated, the deployment is blocked.' },
  { question: 'Is my model data sent to your servers?', answer: 'Only metadata and prediction samples are sent. Model weights are hashed locally by the SDK — the hash (not the weights) is sent. For on-premise Enterprise deployments, nothing leaves your VPC.' },
  { question: 'What is the Insurance Score?', answer: 'The Insurance Score is an actuarial risk rating (0–1000) modelled after property insurance underwriting. It factors in model failure modes, deployment context, prediction stakes, and governance posture. It\'s designed for AI liability insurance underwriters and risk officers.' },
  { question: 'How do I get a governance certificate?', answer: 'After running an audit, go to the Model Detail page and click "Download certificate." The PDF is SHA-256 sealed with a QR code linking to the public audit record. Certificates are versioned and immutable.' },
];

export default function FAQPage() {
  return (
    <div className="container-site py-20">
      <div className="max-w-[720px] mx-auto">
        <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-forest mb-4">FAQ</p>
        <h1 className="text-h1 text-5xl text-ink mb-4" style={{ letterSpacing: '-0.03em', lineHeight: 1.1 }}>
          Frequently asked questions.
        </h1>
        <p className="text-[16px] text-ink-soft leading-relaxed mb-14">
          Everything you need to know about Niyantrana. Can't find your answer?{' '}
          <a href="mailto:hello@niyantrana.ai" className="text-forest underline underline-offset-4">
            Email us
          </a>.
        </p>
        <div className="bg-white border border-stone rounded-card px-8 py-2">
          <Accordion items={FAQ_ITEMS} />
        </div>
      </div>
    </div>
  );
}
