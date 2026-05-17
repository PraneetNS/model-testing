import Link from 'next/link';
import { Logo } from '@/components/ui/Logo';

const PRODUCT_LINKS = [
  { label: 'Behavioral Contracts', href: '/docs/behavioral-contracts' },
  { label: 'Drift Sentinel', href: '/docs/drift-detection' },
  { label: 'Governance Reports', href: '/docs/governance-scoring' },
  { label: 'CI/CD Gate', href: '/docs/cicd-setup' },
  { label: 'LLM Red Teaming', href: '/docs/red-teaming' },
  { label: 'AIBOM', href: '/docs/aibom' },
];

const DEV_LINKS = [
  { label: 'Documentation', href: '/docs' },
  { label: 'API Reference', href: '/docs/api-reference' },
  { label: 'Python SDK', href: '/docs/sdk' },
  { label: 'GitHub', href: 'https://github.com', external: true },
];

const COMPANY_LINKS = [
  { label: 'About', href: '/about' },
  { label: 'Blog', href: '/blog' },
  { label: 'Careers', href: '/careers' },
  { label: 'Security', href: '/security' },
];

interface FooterLinkProps {
  href: string;
  label: string;
  external?: boolean;
}

function FooterLink({ href, label, external }: FooterLinkProps) {
  return (
    <li>
      <Link
        href={href}
        target={external ? '_blank' : undefined}
        rel={external ? 'noopener noreferrer' : undefined}
        className="text-[#888884] hover:text-white transition-colors duration-150 text-[13px]"
      >
        {label}
      </Link>
    </li>
  );
}

export function Footer() {
  return (
    <footer style={{ background: '#0F0F0E' }}>
      <div className="container-site py-16">
        {/* 4-column grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 md:gap-8">
          {/* Brand column */}
          <div className="md:col-span-1">
            <Logo size="sm" showWordmark wordmarkColor="#FFFFFF" />
            <p className="mt-4 text-[13px] text-[#888884] leading-relaxed max-w-[200px]">
              AI governance, made measurable.
            </p>

          </div>

          {/* Product */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[#555552] mb-4">
              Product
            </h4>
            <ul className="flex flex-col gap-3">
              {PRODUCT_LINKS.map((l) => <FooterLink key={l.href} {...l} />)}
            </ul>
          </div>

          {/* Developers */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[#555552] mb-4">
              Developers
            </h4>
            <ul className="flex flex-col gap-3">
              {DEV_LINKS.map((l) => <FooterLink key={l.href} {...l} />)}
            </ul>
          </div>

          {/* Company */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[#555552] mb-4">
              Company
            </h4>
            <ul className="flex flex-col gap-3">
              {COMPANY_LINKS.map((l) => <FooterLink key={l.href} {...l} />)}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-12 pt-6 border-t border-[#1a1a18] flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-[12px] text-[#555552]">
            © 2026 FireFlink ML Research. Niyantrana is Governance-as-Code.
          </p>
          <div className="flex items-center gap-4">
            <Link href="/privacy" className="text-[12px] text-[#555552] hover:text-white transition-colors duration-150">
              Privacy
            </Link>
            <Link href="/terms" className="text-[12px] text-[#555552] hover:text-white transition-colors duration-150">
              Terms
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
