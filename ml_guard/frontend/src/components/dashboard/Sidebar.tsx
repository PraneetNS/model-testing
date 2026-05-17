'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Logo } from '@/components/ui/Logo';
import { useAuth } from '@/context/AuthContext';
import {
  LayoutDashboard, Package, ShieldCheck, Activity,
  Zap, Scale, Bell, Settings, BookOpen, LogOut, Box,
  ScanLine, Lightbulb, Clock, FileText, Users, Terminal,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const NAV_SECTIONS = [
  {
    label: 'Overview',
    items: [
      { href: '/dashboard', label: 'Overview', icon: LayoutDashboard },
    ],
  },
  {
    label: 'Governance',
    items: [
      { href: '/dashboard/audit', label: 'Model Audit', icon: ScanLine },
      { href: '/dashboard/fairness', label: 'Fairness Audit', icon: Scale },
      { href: '/dashboard/explainability', label: 'Explainability', icon: Lightbulb },
      { href: '/dashboard/models', label: 'Model Registry', icon: Package },
      { href: '/dashboard/contracts', label: 'Contracts', icon: ShieldCheck },
      { href: '/dashboard/compliance', label: 'Compliance', icon: Users },
    ],
  },
  {
    label: 'Monitoring',
    items: [
      { href: '/dashboard/drift', label: 'Drift Monitor', icon: Activity },
      { href: '/dashboard/red-team', label: 'Red Team', icon: Zap },
      { href: '/dashboard/alerts', label: 'Alerts', icon: Bell },
    ],
  },
  {
    label: 'Analysis',
    items: [
      { href: '/dashboard/history', label: 'Scan History', icon: Clock },
      { href: '/dashboard/reports', label: 'Reports', icon: FileText },
      { href: '/dashboard/audit-logs', label: 'Audit Logs', icon: Terminal },
      { href: '/dashboard/aibom', label: 'AIBOM', icon: Box },
    ],
  },
  {
    label: 'System',
    items: [
      { href: '/dashboard/settings', label: 'Settings', icon: Settings },
      { href: '/docs', label: 'Docs', icon: BookOpen },
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
      <div className="flex items-center gap-2.5 px-5 h-16 border-b border-[#1a1a18] flex-shrink-0">
        <Logo size="sm" showWordmark wordmarkColor="#FFFFFF" />
        {isDev && (
          <span className="ml-auto text-[9px] font-medium px-1.5 py-0.5 rounded bg-amber-900/60 text-amber-400 uppercase tracking-[0.05em]">DEV</span>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-3 px-2">
        {NAV_SECTIONS.map(section => (
          <div key={section.label} className="mb-3">
            <p className="text-[9px] font-semibold uppercase tracking-[0.08em] text-[#444442] px-3 mb-1">{section.label}</p>
            {section.items.map(({ href, label, icon: Icon }) => {
              const active = isActive(href);
              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    'flex items-center gap-3 px-3 py-2 text-[12px] font-medium transition-colors duration-100 mb-0.5',
                    active
                      ? 'text-white border-l-[2px] border-mint pl-[calc(12px-2px)]'
                      : 'text-[#888884] hover:text-white hover:bg-white/[0.06] border-l-[2px] border-transparent pl-[calc(12px-2px)]',
                  )}
                  style={active ? { background: 'rgba(26,95,58,0.7)', borderRadius: '0 6px 6px 0' } : { borderRadius: '0 6px 6px 0' }}
                >
                  <Icon size={14} strokeWidth={1.5} className="flex-shrink-0" />
                  {label}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      {/* User + logout */}
      <div className="p-4 border-t border-[#1a1a18] flex-shrink-0">
        <div className="flex items-center gap-2.5 mb-3">
          {user?.photoURL ? (
            <img src={user.photoURL} alt="avatar" className="w-7 h-7 rounded-full object-cover flex-shrink-0" />
          ) : (
            <div className="w-7 h-7 rounded-full bg-forest flex items-center justify-center text-white text-[10px] font-semibold flex-shrink-0">
              {initials}
            </div>
          )}
          <div className="flex-1 min-w-0">
            <p className="text-[11px] font-medium text-white truncate">{user?.displayName ?? 'Operator'}</p>
            <p className="text-[10px] text-[#555552] truncate">{user?.email ?? (isDev ? 'dev@local' : '')}</p>
          </div>
        </div>
        <button
          onClick={() => logout()}
          className="flex items-center gap-2 text-[11px] text-[#555552] hover:text-white transition-colors duration-150 w-full"
        >
          <LogOut size={12} strokeWidth={1.5} />
          Log out
        </button>
      </div>
    </aside>
  );
}
