'use client';

import Link from 'next/link';
import {
  ScanLine, BarChart2, FileText, Scale, Lightbulb, TestTube,
  Package, Database, Activity, Eye, FlaskConical, Bell,
  GitBranch, Rocket, BrainCircuit, Bot, Zap, Building2,
  ArrowRight, CheckCircle,
} from 'lucide-react';

const FEATURES = [
  {
    icon: ScanLine, href: '/dashboard/audit', color: '#1A5F3A',
    title: 'Model Audit', sub: 'Core Compliance & Risk',
    desc: 'Upload .pkl / .onnx models and run a full governance audit: drift, fairness, calibration, leakage, security, explainability — in one click.',
  },
  {
    icon: BarChart2, href: '/dashboard/governance-score', color: '#1A5F3A',
    title: 'Governance Score', sub: 'Live score · cert · gate',
    desc: 'Composite live score (0–100) with drift & performance decay. Issue compliance certificates, trigger CI/CD gates.',
  },
  {
    icon: FileText, href: '/dashboard/report-card', color: '#B35A00',
    title: 'Report Card', sub: 'Compliance certificates',
    desc: 'Generate cryptographically-signed report cards. Share a verification URL with auditors & regulators.',
  },
  {
    icon: Scale, href: '/dashboard/fairness', color: '#1A5F3A',
    title: 'Fairness Audit', sub: 'Bias & equity metrics',
    desc: 'Demographic parity, equalized odds, disparate impact. Detect and quantify bias across protected attributes.',
  },
  {
    icon: Lightbulb, href: '/dashboard/explainability', color: '#B35A00',
    title: 'Explainability', sub: 'SHAP & importance',
    desc: 'SHAP-based feature importance. Understand which features drive your model's decisions globally and per-prediction.',
  },
  {
    icon: TestTube, href: '/dashboard/behavior', color: '#7C3AED',
    title: 'Behavior Test', sub: 'Scenario robustness',
    desc: 'Run behavioral contracts: edge cases, perturbation tests, invariance checks. Validate model behavior beyond accuracy.',
  },
  {
    icon: Package, href: '/dashboard/models', color: '#1A5F3A',
    title: 'Model Registry', sub: 'Version control',
    desc: 'Register, version, and track all production models. Compare governance scores across versions and deployments.',
  },
  {
    icon: Database, href: '/dashboard/datasets', color: '#0369A1',
    title: 'Datasets', sub: 'Data lineage & health',
    desc: 'Manage reference and production datasets. Track schema, freshness, distribution shifts, and data lineage in real time.',
  },
  {
    icon: Activity, href: '/dashboard/drift', color: '#C0392B',
    title: 'Drift Monitor', sub: 'Real-time feature drift',
    desc: 'KS-test, PSI, and Wasserstein distance. Continuous monitoring with configurable alert thresholds.',
  },
  {
    icon: Eye, href: '/dashboard/observability', color: '#0369A1',
    title: 'Observability', sub: 'Prediction & latency feed',
    desc: 'Live feed of prediction events, latency histograms, and anomaly flags. Full pipeline observability in real time.',
  },
  {
    icon: FlaskConical, href: '/dashboard/data-quality', color: '#B35A00',
    title: 'Data Quality', sub: 'Schema & freshness checks',
    desc: 'Validate inbound data: missing values, type mismatches, out-of-range distributions, and freshness SLAs.',
  },
  {
    icon: GitBranch, href: '/dashboard/ci', color: '#7C3AED',
    title: 'CI/CD Gates', sub: 'Automated policy enforcement',
    desc: 'Sync with GitHub Actions, GitLab CI, Jenkins. Block deployments that fail governance thresholds automatically.',
  },
  {
    icon: Rocket, href: '/dashboard/deployments', color: '#0369A1',
    title: 'Deployments', sub: 'Environment tracking',
    desc: 'Track models across staging, production, and canary environments. One-click rollback and blue-green comparisons.',
  },
  {
    icon: BrainCircuit, href: '/dashboard/llm-eval', color: '#7C3AED',
    title: 'LLM Evaluation', sub: 'Prompt safety & quality',
    desc: 'Evaluate LLM prompt/response pairs for injection, toxicity, hallucination risk, and stability across responses.',
  },
  {
    icon: Bot, href: '/dashboard/advisor', color: '#1A5F3A',
    title: 'AI Advisor', sub: 'Actionable recommendations',
    desc: 'Context-aware governance advisories. The advisor synthesizes audit results into prioritized, actionable fix plans.',
  },
  {
    icon: Zap, href: '/dashboard/red-team', color: '#C0392B',
    title: 'Red Team', sub: 'Adversarial attack simulation',
    desc: 'Systematically probe model robustness with adversarial perturbations, boundary attacks, and model inversion.',
  },
  {
    icon: Building2, href: '/dashboard/enterprise', color: '#0369A1',
    title: 'Enterprise Hub', sub: 'Multi-tenant org management',
    desc: 'Org-level RBAC, SSO, tenant isolation, API key management, billing enforcement, and audit trail.',
  },
];

const STATS = [
  { value: '17', label: 'Governance modules' },
  { value: '7', label: 'Audit check types' },
  { value: '100%', label: 'API driven' },
  { value: 'Real-time', label: 'Monitoring' },
];

export default function ShowcasePage() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Top bar */}
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white flex-shrink-0">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Platform Showcase</h1>
          <p className="text-[11px] text-muted">ML Guard v7.2 — Enterprise AI Governance Platform</p>
        </div>
      </div>

      <div className="flex-1 p-8 overflow-auto">
        {/* Hero */}
        <div
          className="rounded-card p-8 mb-8 relative overflow-hidden"
          style={{ background: '#0F0F0E' }}
        >
          <div className="relative z-10">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-medium mb-4"
              style={{ background: 'rgba(62,207,142,0.15)', color: '#3ECF8E' }}>
              <span className="w-1.5 h-1.5 rounded-full bg-[#3ECF8E] animate-pulse" />
              System Ready · All modules operational
            </div>
            <h2 className="text-3xl font-bold text-white mb-3" style={{ letterSpacing: '-0.03em' }}>
              The contract your model must keep.
            </h2>
            <p className="text-[14px] max-w-xl mb-6" style={{ color: '#888884' }}>
              ML Guard is an enterprise-grade AI governance platform. Audit, monitor, certify and control every ML model across your organization — from training to production.
            </p>
            <div className="flex gap-8">
              {STATS.map(s => (
                <div key={s.label}>
                  <p className="text-[22px] font-bold text-white" style={{ letterSpacing: '-0.02em' }}>{s.value}</p>
                  <p className="text-[11px]" style={{ color: '#555552' }}>{s.label}</p>
                </div>
              ))}
            </div>
          </div>
          {/* Decorative grid lines */}
          <div className="absolute inset-0 opacity-[0.04]"
            style={{ backgroundImage: 'linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
        </div>

        {/* Checklist banner */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
          {['EU AI Act ready', 'SOC 2 aligned', 'GDPR compatible', 'ISO 42001 aware'].map(label => (
            <div key={label} className="flex items-center gap-2 bg-white border border-stone rounded-card px-4 py-3">
              <CheckCircle size={14} className="text-forest flex-shrink-0" strokeWidth={2} />
              <span className="text-[12px] font-medium text-ink">{label}</span>
            </div>
          ))}
        </div>

        {/* Feature grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map(({ icon: Icon, href, color, title, sub, desc }) => (
            <Link key={href} href={href}
              className="bg-white border border-stone rounded-card p-5 hover:border-forest/50 hover:shadow-sm transition-all duration-150 group flex flex-col"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 rounded-[8px] flex items-center justify-center flex-shrink-0"
                  style={{ background: `${color}18` }}>
                  <Icon size={15} strokeWidth={1.75} style={{ color }} />
                </div>
                <div>
                  <p className="text-[13px] font-semibold text-ink leading-tight">{title}</p>
                  <p className="text-[10px] text-muted">{sub}</p>
                </div>
              </div>
              <p className="text-[12px] text-ink-soft leading-relaxed flex-1">{desc}</p>
              <div className="flex items-center gap-1 mt-3 text-[11px] font-medium text-forest opacity-0 group-hover:opacity-100 transition-opacity">
                Open <ArrowRight size={11} strokeWidth={2} />
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
