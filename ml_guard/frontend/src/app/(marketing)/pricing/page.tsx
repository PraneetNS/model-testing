import type { Metadata } from 'next';
import Link from 'next/link';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Check } from 'lucide-react';

export const metadata: Metadata = {
  title: 'Pricing — Niyantrana AI Governance',
  description: 'Simple, honest pricing for AI governance. Free, Pro, and Enterprise tiers.',
};

const PLANS = [
  {
    name: 'Free',
    price: '$0',
    period: '',
    description: 'For individuals exploring AI governance.',
    features: [
      '2 models',
      '1,000 predictions/mo',
      '1 governance report',
      'Community support',
      'Public API access',
    ],
    cta: 'Start free',
    href: '/signup',
    highlighted: false,
  },
  {
    name: 'Pro',
    price: '$149',
    period: '/mo',
    description: 'For teams shipping AI to production.',
    features: [
      'Unlimited models',
      '100,000 predictions/mo',
      'All features unlocked',
      'Slack + Teams integration',
      'Priority support',
      'Custom scoring weights',
      'GitHub Actions gate',
    ],
    cta: 'Start Pro trial',
    href: '/signup?plan=pro',
    highlighted: true,
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    period: '',
    description: 'For regulated industries.',
    features: [
      'Custom prediction limits',
      'SSO (SAML / OIDC)',
      'On-premise deployment',
      'SLA guarantee (99.9%)',
      'Dedicated support',
      'Custom compliance mapping',
      'Private audit ledger',
    ],
    cta: 'Contact sales',
    href: 'mailto:sales@niyantrana.ai',
    highlighted: false,
  },
];

export default function PricingPage() {
  return (
    <div className="container-site py-20">
      <div className="text-center mb-16">
        <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-forest mb-4">Pricing</p>
        <h1 className="text-5xl font-bold text-ink mb-4" style={{ letterSpacing: '-0.03em', lineHeight: 1.1 }}>
          Simple, honest pricing.
        </h1>
        <p className="text-[16px] text-ink-soft max-w-[480px] mx-auto leading-relaxed">
          No per-seat nonsense. No hidden overage fees. One number, everything included.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6 max-w-[1000px] mx-auto mb-16">
        {PLANS.map((plan) => (
          <div
            key={plan.name}
            className={`rounded-card p-8 flex flex-col ${
              plan.highlighted
                ? 'bg-white border-[1.5px] border-forest'
                : 'bg-white border border-stone'
            }`}
          >
            {plan.highlighted && (
              <Badge variant="certified" className="mb-4 self-start">Most popular</Badge>
            )}
            <h2 className="text-[20px] font-semibold text-ink mb-1">{plan.name}</h2>
            <p className="text-[13px] text-muted mb-5">{plan.description}</p>
            <div className="flex items-baseline gap-1 mb-6">
              <span className="text-4xl font-bold text-ink">{plan.price}</span>
              {plan.period && <span className="text-sm text-muted">{plan.period}</span>}
            </div>
            <ul className="flex flex-col gap-3 mb-8 flex-1">
              {plan.features.map((f) => (
                <li key={f} className="flex items-center gap-2.5 text-[13px] text-ink-soft">
                  <Check size={13} strokeWidth={2} className="text-forest flex-shrink-0" />
                  {f}
                </li>
              ))}
            </ul>
            <Link href={plan.href}>
              <Button
                variant={plan.highlighted ? 'primary' : 'ghost'}
                size="md"
                className="w-full justify-center"
                data-cursor={plan.highlighted ? 'cta' : 'pointer'}
              >
                {plan.cta}
              </Button>
            </Link>
          </div>
        ))}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-6 max-w-[600px] mx-auto text-center">
        {[
          { value: '3,200+', label: 'Models audited' },
          { value: '91.4', label: 'Avg governance score' },
          { value: '< 45ms', label: 'Median contract latency' },
        ].map((s) => (
          <div key={s.label}>
            <p className="text-2xl font-bold text-ink mb-1">{s.value}</p>
            <p className="text-[12px] text-muted">{s.label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
