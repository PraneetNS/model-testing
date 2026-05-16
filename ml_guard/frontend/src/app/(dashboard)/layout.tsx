import { DashboardSidebar } from '@/components/dashboard/Sidebar';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Dashboard — Niyantrana',
  description: 'Niyantrana AI governance dashboard.',
};

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen" style={{ background: '#F7F6F2' }}>
      <DashboardSidebar />
      {/* Main content offset by sidebar */}
      <div className="flex-1 ml-[220px] flex flex-col min-h-screen">
        {children}
      </div>
    </div>
  );
}
