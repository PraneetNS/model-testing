'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Logo } from '@/components/ui/Logo';
import {
  LayoutDashboard, Package, ShieldCheck, Activity,
  Zap, FileText, Scale, Bell, Settings, BookOpen, LogOut,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const NAV_ITEMS = [
  { href: '/dashboard', label: 'Overview', icon: LayoutDashboard },
  { href: '/dashboard/models', label: 'Models', icon: Package },
  { href: '/dashboard/contracts', label: 'Contracts', icon: ShieldCheck },
  { href: '/dashboard/drift', label: 'Drift Monitor', icon: Activity },
  { href: '/dashboard/red-team', label: 'Red Team', icon: Zap },
  { href: '/dashboard/aibom', label: 'AIBOM', icon: Package },
  { href: '/dashboard/compliance', label: 'Compliance', icon: Scale },
  { href: '/dashboard/alerts', label: 'Alerts', icon: Bell },
  { href: '/dashboard/settings', label: 'Settings', icon: Settings },
  { href: '/docs', label: 'Docs', icon: BookOpen },
];

export function DashboardSidebar() {
  const pathname = usePathname();

  const isActive = (href: string) => {
    if (href === '/dashboard') return pathname === '/dashboard';
    return pathname.startsWith(href);
  };

  return (
    <aside
      className="fixed top-0 left-0 h-screen w-[220px] flex flex-col z-40"
      style={{ background: '#0F0F0E' }}
    >
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-5 h-16 border-b border-[#1a1a18]">
        <Logo size="sm" showWordmark wordmarkColor="#FFFFFF" />
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto py-3 px-2">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = isActive(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-r-none text-[13px] font-medium transition-colors duration-100 mb-0.5',
                active
                  ? 'text-white border-l-[3px] border-mint pl-[calc(12px-3px)]'
                  : 'text-[#888884] hover:text-white hover:bg-white/[0.06] border-l-[3px] border-transparent pl-[calc(12px-3px)]',
              )}
              style={active ? { background: '#1A5F3A', borderRadius: '0 6px 6px 0' } : { borderRadius: '0 6px 6px 0' }}
            >
              <Icon size={16} strokeWidth={1.5} className="flex-shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Bottom: user + logout */}
      <div className="p-4 border-t border-[#1a1a18]">
        <div className="flex items-center gap-2.5 mb-3">
          <div className="w-7 h-7 rounded-full bg-forest flex items-center justify-center text-white text-[11px] font-semibold flex-shrink-0">
            PR
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-[12px] font-medium text-white truncate">Priya Rajan</p>
            <p className="text-[11px] text-[#555552] truncate">priya@acme.ai</p>
          </div>
        </div>
        <Link
          href="/login"
          className="flex items-center gap-2 text-[12px] text-[#555552] hover:text-white transition-colors duration-150"
        >
          <LogOut size={13} strokeWidth={1.5} />
          Log out
        </Link>
      </div>
    </aside>
  );
}
