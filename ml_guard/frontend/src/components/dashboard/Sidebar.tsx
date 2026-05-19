'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Logo } from '@/components/ui/Logo';
import { useAuth } from '@/context/AuthContext';
import {
  LayoutDashboard, Package, ShieldCheck, Activity,
  Zap, Scale, Bell, Settings, BookOpen, LogOut, Box,
  ScanLine, Lightbulb, Clock, FileText, Users, Terminal,
  Star, Database, BarChart2, FlaskConical, GitBranch,
  Rocket, BrainCircuit, Bot, Building2, Eye, FileBarChart,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const NAV_SECTIONS = [
  {
    label: 'Platform Info',
    items: [
      { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
      { href: '/dashboard/showcase', label: 'Showcase', icon: Star, sub: 'Platform overview' },
      { href: '/docs', label: 'Documentation', icon: BookOpen, sub: 'Installation & Guides' },
    ],
  },
  {
    label: 'Governance',
    items: [
      { href: '/dashboard/audit', label: 'Model Audit', icon: ScanLine, sub: 'Core Compliance & Risk' },
      { href: '/dashboard/governance-score', label: 'Governance Score', icon: BarChart2, sub: 'Live score · cert · gate' },
      { href: '/dashboard/report-card', label: 'Report Card', icon: FileText, sub: 'Compliance certificates' },
      { href: '/dashboard/fairness', label: 'Fairness', icon: Scale, sub: 'Bias & equity metrics' },
      { href: '/dashboard/explainability', label: 'Explainability', icon: Lightbulb, sub: 'SHAP & importance' },
      { href: '/dashboard/behavior', label: 'Behavior Test', icon: FlaskConical, sub: 'Scenario robustness' },
    ],
  },
  {
    label: 'Asset Tracking',
    items: [
      { href: '/dashboard/models', label: 'Model Registry', icon: Package, sub: 'Version control' },
      { href: '/dashboard/datasets', label: 'Datasets', icon: Database, sub: 'Data lineage' },
    ],
  },
  {
    label: 'Monitoring',
    items: [
      { href: '/dashboard/drift', label: 'Drift Monitor', icon: Activity, sub: 'Feature & concept drift' },
      { href: '/dashboard/observability', label: 'Observability', icon: Eye, sub: 'Real-time metrics' },
      { href: '/dashboard/data-quality', label: 'Data Quality', icon: FlaskConical, sub: 'Schema & freshness' },
      { href: '/dashboard/alerts', label: 'Alerts', icon: Bell, sub: 'Rules & events' },
    ],
  },
  {
    label: 'Pipeline',
    items: [
      { href: '/dashboard/ci', label: 'CI/CD', icon: GitBranch, sub: 'Gate checks' },
      { href: '/dashboard/deployments', label: 'Deployments', icon: Rocket, sub: 'Environments' },
      { href: '/dashboard/history', label: 'Scan History', icon: Clock, sub: 'Past audit runs' },
    ],
  },
  {
    label: 'Intelligence',
    items: [
      { href: '/dashboard/llm-eval', label: 'LLM Evaluation', icon: BrainCircuit, sub: 'Prompt safety & quality' },
      { href: '/dashboard/advisor', label: 'AI Advisor', icon: Bot, sub: 'Recommendations' },
      { href: '/dashboard/red-team', label: 'Red Team', icon: Zap, sub: 'Adversarial testing' },
    ],
  },
  {
    label: 'Enterprise',
    items: [
      { href: '/dashboard/enterprise', label: 'Enterprise Hub', icon: Building2, sub: 'Org & tenants' },
      { href: '/dashboard/compliance', label: 'Compliance', icon: Users, sub: 'Frameworks' },
      { href: '/dashboard/contracts', label: 'Contracts', icon: ShieldCheck, sub: 'Behavioral contracts' },
      { href: '/dashboard/aibom', label: 'AIBOM', icon: Box, sub: 'AI Bill of Materials' },
      { href: '/dashboard/audit-logs', label: 'Audit Logs', icon: Terminal, sub: 'Immutable trail' },
      { href: '/dashboard/reports', label: 'Reports', icon: FileBarChart, sub: 'PDF exports' },
      { href: '/dashboard/settings', label: 'Settings', icon: Settings, sub: 'Configuration' },
    ],
  },
];

export function DashboardSidebar() {
  const pathname = usePathname();
  const { user, logout, isDev } = useAuth();

  const isActive = (href: string) => {
    if (href === '/dashboard') return pathname === '/dashboard';
    return pathname.startsWith(href);
  };

  const initials = user?.displayName
    ? user.displayName.split(' ').map((n: string) => n[0]).slice(0, 2).join('').toUpperCase()
    : user?.email?.slice(0, 2).toUpperCase() ?? 'OP';

  return (
    <aside className="fixed top-0 left-0 h-screen w-[220px] flex flex-col z-40" style={{ background: '#0F0F0E' }}>
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-5 h-14 border-b border-[#1a1a18] flex-shrink-0">
        <Logo size="sm" showWordmark wordmarkColor="#FFFFFF" />
        {isDev && (
          <span className="ml-auto text-[9px] font-medium px-1.5 py-0.5 rounded bg-amber-900/60 text-amber-400 uppercase tracking-[0.05em]">DEV</span>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-2 px-2" style={{ scrollbarWidth: 'none' }}>
        {NAV_SECTIONS.map(section => (
          <div key={section.label} className="mb-2">
            <p className="text-[9px] font-semibold uppercase tracking-[0.08em] text-[#333330] px-3 mb-0.5">{section.label}</p>
            {section.items.map(({ href, label, icon: Icon }) => {
              const active = isActive(href);
              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    'flex items-center gap-2.5 px-3 py-1.5 text-[12px] font-medium transition-colors duration-100 mb-0.5',
                    active
                      ? 'text-white border-l-[2px] border-[#3ECF8E] pl-[calc(12px-2px)]'
                      : 'text-[#666662] hover:text-[#AAAA9E] hover:bg-white/[0.04] border-l-[2px] border-transparent pl-[calc(12px-2px)]',
                  )}
                  style={active ? { background: 'rgba(26,95,58,0.55)', borderRadius: '0 6px 6px 0' } : { borderRadius: '0 6px 6px 0' }}
                >
                  <Icon size={13} strokeWidth={1.5} className="flex-shrink-0" />
                  {label}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      {/* User + logout */}
      <div className="p-3 border-t border-[#1a1a18] flex-shrink-0">
        <div className="flex items-center gap-2 mb-2">
          {user?.photoURL ? (
            <img src={user.photoURL} alt="avatar" className="w-6 h-6 rounded-full object-cover flex-shrink-0" />
          ) : (
            <div className="w-6 h-6 rounded-full bg-[#1A5F3A] flex items-center justify-center text-white text-[9px] font-semibold flex-shrink-0">
              {initials}
            </div>
          )}
          <div className="flex-1 min-w-0">
            <p className="text-[11px] font-medium text-white truncate">{user?.displayName ?? 'Operator'}</p>
            <p className="text-[9px] text-[#444442] truncate">{user?.email ?? (isDev ? 'dev@local' : '')}</p>
          </div>
        </div>
        <button
          onClick={() => logout()}
          className="flex items-center gap-2 text-[10px] text-[#444442] hover:text-white transition-colors duration-150 w-full"
        >
          <LogOut size={11} strokeWidth={1.5} />
          Log out
        </button>
      </div>
    </aside>
  );
}
