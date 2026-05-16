import type { Metadata } from 'next';
import Link from 'next/link';
import { HeroIllustration } from '@/components/marketing/HeroIllustration';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { Accordion } from '@/components/ui/Accordion';
import {
  ShieldCheck, Activity, FileText, GitBranch,
  Zap, Eye, Package, BarChart3, ArrowRight,
  Check, X as XIcon,
} from 'lucide-react';

export const metadata: Metadata = {
  title: 'Niyantrana — AI Governance Platform',
  description: 'Behavioral contracts, drift detection, cryptographic audit certificates. Prove your AI is safe.',
};

const FEATURES = [
  {
    icon: ShieldCheck,
    category: 'Enforcement',
    title: 'Behavioral Contracts',
    body: 'Define promises your model must keep. Every prediction, validated in real time.',
  },
  {
    icon: Activity,
    category: 'Monitoring',
    title: 'Drift Sentinel',
    body: 'PSI, KS-Test, Jensen-Shannon. Feature-level granularity. Fire-and-forget ingestion.',
  },
  {
    icon: FileText,
    category: 'Audit',
    title: 'Governance Report Cards',
    body: 'PDF audit certificates with SHA-256 tamper-proof seals. CERTIFIED, CONDITIONAL, or FAILED.',
  },
  {
    icon: GitBranch,
    category: 'CI/CD',
    title: 'CI/CD Governance Gate',
    body: 'Models that fail governance never reach production. Integrated into GitHub Actions.',
  },
  {
    icon: Zap,
    category: 'Security',
    title: 'LLM Red Teaming',
    body: 'Jailbreak detection, PII leakage, toxicity scoring, prompt injection mitigation.',
  },
  {
    icon: Eye,
    category: 'Observability',
    title: 'RAG Observability',
    body: 'Grounding fidelity, retrieval hit rate, hallucination risk. Dashboarded over time.',
  },
  {
    icon: Package,
    category: 'Supply Chain',
    title: 'AI Bill of Materials',
    body: 'SHA-256 hashes, CVE scanning, training data provenance. Full supply chain visibility.',
  },
  {
    icon: BarChart3,
    category: 'Risk',
    title: 'Insurance Score',
    body: 'Actuarial risk rating from 0–1000. Designed for AI liability insurance underwriting.',
  },
];

const HOW_IT_WORKS = [
  {
    step: 1,
    title: 'Upload or pull',
    body: 'Submit your model via our SDK, CI script, or directly from HuggingFace Hub.',
  },
  {
    step: 2,
    title: 'Audit runs',
    body: 'Scoring engine evaluates performance, fairness, security, and behavioral compliance.',
  },
  {
    step: 3,
    title: 'Contracts enforced',
    body: 'Behavioral promises are validated against every live prediction.',
  },
  {
    step: 4,
    title: 'Certificate issued',
    body: 'PDF report card generated, SHA-256 sealed, lineage recorded.',
  },
];

const PRICING = [
  {
    name: 'Free',
    price: '$0',
    description: 'For individuals exploring AI governance.',
    features: ['2 models', '1,000 predictions/mo', '1 governance report', 'Community support'],
    cta: 'Start free',
    highlighted: false,
  },
  {
    name: 'Pro',
    price: '$149',
    period: '/mo',
    description: 'For teams shipping AI to production.',
    features: ['Unlimited models', '100k predictions/mo', 'All features', 'Slack integration', 'Priority support'],
    cta: 'Start Pro trial',
    highlighted: true,
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    description: 'For regulated industries.',
    features: ['Custom limits', 'SSO', 'On-prem option', 'SLA guarantee', 'Dedicated support'],
    cta: 'Contact sales',
    highlighted: false,
  },
];

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

const INTEGRATIONS = [
  'HuggingFace', 'MLflow', 'Weights & Biases', 'Slack', 'Microsoft Teams',
  'GitHub Actions', 'Kaggle', 'OpenML', 'Snowflake', 'BigQuery', 'AWS S3', 'GCS',
];

export default function LandingPage() {
  return (
    <>
      {/* ─── HERO ─────────────────────────────── */}
      <section className="min-h-[calc(100vh-64px)] flex flex-col lg:flex-row items-center gap-12 lg:gap-20 container-site py-20 lg:py-24">
        {/* Left: Copy */}
        <div className="flex-1 max-w-[560px]">
          {/* Eyebrow */}
          <a
            href="/docs/aibom"
            className="inline-flex items-center gap-2 mb-6 px-3 py-1.5 rounded-badge border border-forest/40 bg-mist text-forest text-xs font-semibold hover:border-forest transition-colors duration-200"
          >
            Introducing AIBOM — AI Bill of Materials
            <ArrowRight size={12} strokeWidth={2} />
          </a>

          {/* H1 */}
          <h1 className="text-display text-5xl lg:text-6xl mb-6 text-ink" style={{ letterSpacing: '-0.04em', lineHeight: 1.0 }}>
            AI governance,<br />
            you can actually{' '}
            <span className="wavy-underline">prove</span>.
          </h1>

          {/* Subtext */}
          <p className="text-[17px] text-ink-soft leading-relaxed mb-8 max-w-[480px]">
            Niyantrana turns your models into auditable assets — behavioral contracts, drift detection, cryptographic certificates, and supply chain integrity. All in one platform.
          </p>

          {/* CTAs */}
          <div className="flex flex-wrap gap-3 mb-10">
            <Link href="/signup">
              <Button variant="primary" size="lg" data-cursor="cta">
                Start governing free
              </Button>
            </Link>
            <Link href="/dashboard">
              <Button variant="ghost" size="lg">
                See a live audit →
              </Button>
            </Link>
          </div>

          {/* Social proof */}
          <div className="flex items-center gap-3 text-[13px] text-muted">
            <span>Trusted by teams at</span>
            {['Acme AI', 'FinSafe Corp', 'DataOrbit'].map((co) => (
              <span key={co} className="font-semibold text-ink-soft opacity-40">{co}</span>
            ))}
          </div>
        </div>

        {/* Right: Dashboard illustration */}
        <div className="flex-1 w-full max-w-[640px]">
          <HeroIllustration />
        </div>
      </section>

      {/* ─── TRUST BAR ────────────────────────── */}
      <section style={{ background: '#0F0F0E', padding: '20px 0' }}>
        <div className="flex items-center justify-center gap-0 overflow-x-auto">
          <span className="text-[13px] text-[#888884] whitespace-nowrap px-6">Compliant with</span>
          {['EU AI Act', 'NIST AI RMF', 'ISO 42001', 'SOC 2', 'GDPR'].map((item, i) => (
            <div key={item} className="flex items-center">
              {i > 0 && <span className="w-px h-4 bg-[#333] mx-0" />}
              <span className="px-6 text-[13px] text-[#888884] whitespace-nowrap font-medium">
                {item}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* ─── PROBLEM / SOLUTION ───────────────── */}
      <section className="container-site py-24">
        <div className="grid md:grid-cols-2 gap-6">
          {/* Problem */}
          <Card padding="lg">
            <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-muted mb-3">The problem</p>
            <h2 className="text-h2 text-3xl mb-4 text-ink" style={{ letterSpacing: '-0.02em' }}>
              AI ethics is<br />measured in vibes.
            </h2>
            <p className="text-sm text-ink-soft leading-relaxed mb-6">
              Governance today means PDF reports, Excel checklists, and a prayer. When a model fails, no one knows why, no one has a record, and compliance is a lie.
            </p>
            <ul className="flex flex-col gap-3">
              {[
                'No audit trail for model decisions',
                'Drift goes undetected until damage is done',
                'Compliance reports are stale the moment they\'re generated',
              ].map((item) => (
                <li key={item} className="flex items-start gap-3 text-sm text-ink-soft">
                  <XIcon size={14} strokeWidth={2} className="text-danger flex-shrink-0 mt-0.5" />
                  {item}
                </li>
              ))}
            </ul>
          </Card>

          {/* Solution */}
          <div className="rounded-card p-8" style={{ background: '#E1F5EE' }}>
            <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-forest mb-3">The solution</p>
            <h2 className="text-h2 text-3xl mb-4 text-ink" style={{ letterSpacing: '-0.02em' }}>
              Governance-as-Code.
            </h2>
            <p className="text-sm text-ink-soft leading-relaxed mb-6">
              Niyantrana treats your behavioral promises as enforceable contracts. Every prediction is validated. Every breach is recorded. Every certificate is cryptographically sealed.
            </p>
            <ul className="flex flex-col gap-3">
              {[
                'Real-time behavioral contract enforcement',
                'Cryptographic audit certificates (SHA-256)',
                'CI/CD governance gates — bad models never ship',
              ].map((item) => (
                <li key={item} className="flex items-start gap-3 text-sm text-ink-soft">
                  <Check size={14} strokeWidth={2} className="text-forest flex-shrink-0 mt-0.5" />
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* ─── FEATURE GRID ─────────────────────── */}
      <section id="features" className="container-site py-24">
        <div className="text-center mb-14">
          <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-forest mb-3">What Niyantrana does</p>
          <h2 className="text-h2 text-4xl text-ink" style={{ letterSpacing: '-0.02em' }}>
            Every dimension of AI risk. One platform.
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {FEATURES.map((f, i) => {
            const Icon = f.icon;
            return (
              <div
                key={f.title}
                className="bg-white border border-stone rounded-card p-6 transition-colors duration-200 hover:border-mint"
                style={{ animationDelay: `${i * 80}ms` }}
              >
                <div className="w-10 h-10 rounded-icon bg-mist flex items-center justify-center mb-4">
                  <Icon size={20} strokeWidth={1.5} className="text-forest" />
                </div>
                <p className="text-[10px] font-bold uppercase tracking-[0.08em] text-forest mb-2">{f.category}</p>
                <h3 className="text-[15px] font-semibold text-ink mb-2 leading-snug">{f.title}</h3>
                <p className="text-[13px] text-ink-soft leading-relaxed">{f.body}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* ─── HOW IT WORKS ─────────────────────── */}
      <section className="container-site py-24">
        <div className="text-center mb-14">
          <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-forest mb-3">How it works</p>
          <h2 className="text-h2 text-4xl text-ink" style={{ letterSpacing: '-0.02em' }}>
            From code push to certified model.
          </h2>
        </div>
        <div className="relative">
          {/* Dashed connecting line (desktop) */}
          <div className="hidden lg:block absolute top-6 left-[calc(12.5%+20px)] right-[calc(12.5%+20px)] h-px border-t-2 border-dashed border-forest/30" />
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
            {HOW_IT_WORKS.map((s) => (
              <div key={s.step} className="flex flex-col items-center text-center lg:items-start lg:text-left">
                <div className="w-12 h-12 rounded-full bg-forest flex items-center justify-center mb-4 flex-shrink-0 relative z-10">
                  <span className="text-white text-[15px] font-bold">{s.step}</span>
                </div>
                <h3 className="text-[15px] font-semibold text-ink mb-2">{s.title}</h3>
                <p className="text-[13px] text-ink-soft leading-relaxed">{s.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── TESTIMONIAL ──────────────────────── */}
      <section style={{ background: '#0F0F0E', padding: '80px 0' }}>
        <div className="container-site flex justify-center">
          <div className="relative max-w-[680px] text-center">
            {/* Large quotation mark */}
            <span
              aria-hidden="true"
              style={{
                position: 'absolute',
                top: '-40px',
                left: '50%',
                transform: 'translateX(-50%)',
                fontSize: '120px',
                lineHeight: 1,
                color: '#1A5F3A',
                opacity: 0.3,
                fontFamily: 'Georgia, serif',
                pointerEvents: 'none',
              }}
            >
              "
            </span>
            <blockquote className="relative z-10">
              <p
                style={{
                  fontFamily: 'Inter, system-ui',
                  fontSize: '20px',
                  fontStyle: 'italic',
                  fontWeight: 400,
                  color: '#E8E5DF',
                  lineHeight: 1.65,
                  marginBottom: '24px',
                }}
              >
                "We went from 'I think the model is fair' to 'here is a cryptographic certificate that proves it.' Niyantrana changed how we talk to our compliance team."
              </p>
              <footer style={{ fontSize: '13px', color: '#888884' }}>
                — Head of ML, Fortune 500 Financial Services Firm
              </footer>
            </blockquote>
          </div>
        </div>
      </section>

      {/* ─── INTEGRATIONS ─────────────────────── */}
      <section className="container-site py-24">
        <div className="text-center mb-12">
          <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-forest mb-3">Integrations</p>
          <h2 className="text-h2 text-4xl text-ink" style={{ letterSpacing: '-0.02em' }}>
            Works with everything you already use.
          </h2>
        </div>
        <div className="flex flex-wrap justify-center gap-6">
          {INTEGRATIONS.map((name) => (
            <div
              key={name}
              className="px-5 py-3 bg-white border border-stone rounded-[10px] text-[13px] font-medium text-muted hover:text-ink-soft hover:border-stone/80 transition-colors duration-300 cursor-default"
            >
              {name}
            </div>
          ))}
        </div>
      </section>

      {/* ─── PRICING ──────────────────────────── */}
      <section id="pricing" className="container-site py-24">
        <div className="text-center mb-14">
          <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-forest mb-3">Pricing</p>
          <h2 className="text-h2 text-4xl text-ink" style={{ letterSpacing: '-0.02em' }}>
            Simple, honest pricing.
          </h2>
        </div>
        <div className="grid md:grid-cols-3 gap-6 max-w-[960px] mx-auto">
          {PRICING.map((plan) => (
            <div
              key={plan.name}
              className={`rounded-card p-8 ${
                plan.highlighted
                  ? 'bg-white border-[1.5px] border-forest'
                  : 'bg-white border border-stone'
              }`}
            >
              {plan.highlighted && (
                <Badge variant="certified" className="mb-4">Most popular</Badge>
              )}
              <h3 className="text-[18px] font-semibold text-ink mb-1">{plan.name}</h3>
              <p className="text-[13px] text-ink-soft mb-4">{plan.description}</p>
              <div className="flex items-baseline gap-1 mb-6">
                <span className="text-3xl font-bold text-ink">{plan.price}</span>
                {plan.period && <span className="text-sm text-muted">{plan.period}</span>}
              </div>
              <ul className="flex flex-col gap-2.5 mb-8">
                {plan.features.map((f) => (
                  <li key={f} className="flex items-center gap-2 text-[13px] text-ink-soft">
                    <Check size={13} strokeWidth={2} className="text-forest flex-shrink-0" />
                    {f}
                  </li>
                ))}
              </ul>
              <Link href="/signup">
                <Button
                  variant={plan.highlighted ? 'primary' : 'ghost'}
                  size="md"
                  className="w-full justify-center"
                >
                  {plan.cta}
                </Button>
              </Link>
            </div>
          ))}
        </div>
      </section>

      {/* ─── FAQ ──────────────────────────────── */}
      <section className="container-site py-24">
        <div className="max-w-[720px] mx-auto">
          <div className="text-center mb-12">
            <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-forest mb-3">FAQ</p>
            <h2 className="text-h2 text-4xl text-ink" style={{ letterSpacing: '-0.02em' }}>
              Common questions.
            </h2>
          </div>
          <div className="bg-white border border-stone rounded-card px-8 py-2">
            <Accordion items={FAQ_ITEMS} />
          </div>
        </div>
      </section>

      {/* ─── FINAL CTA ────────────────────────── */}
      <section className="container-site py-16 mb-8">
        <div className="bg-ink rounded-[20px] p-12 text-center">
          <h2 className="text-3xl font-bold text-white mb-4" style={{ letterSpacing: '-0.02em' }}>
            Prove it, don't promise it.
          </h2>
          <p className="text-[#888884] text-[15px] mb-8 max-w-[480px] mx-auto leading-relaxed">
            Every model. Every deployment. Audited.
            Start your first governance audit in under 5 minutes.
          </p>
          <div className="flex flex-wrap gap-3 justify-center">
            <Link href="/signup">
              <Button variant="accent" size="lg" data-cursor="cta">
                Start governing free
              </Button>
            </Link>
            <Link href="/docs">
              <Button
                size="lg"
                className="bg-transparent text-white border border-white/20 hover:border-white/40"
              >
                Read the docs
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
