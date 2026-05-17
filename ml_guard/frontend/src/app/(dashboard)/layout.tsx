'use client';

import { DashboardSidebar } from '@/components/dashboard/Sidebar';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { user, loading, isDev } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user && !isDev) {
      router.push('/login');
    }
  }, [user, loading, isDev, router]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center" style={{ background: '#F7F6F2' }}>
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 rounded-full border-2 border-forest border-t-transparent animate-spin" />
          <p className="text-[13px] text-muted">Loading…</p>
        </div>
      </div>
    );
  }

  if (!user && !isDev) return null;

  return (
    <div className="flex min-h-screen" style={{ background: '#F7F6F2' }}>
      <DashboardSidebar />
      <div className="flex-1 ml-[220px] flex flex-col min-h-screen overflow-hidden">
        {children}
      </div>
    </div>
  );
}
