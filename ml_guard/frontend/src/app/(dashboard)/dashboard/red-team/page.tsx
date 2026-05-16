import type { Metadata } from 'next';

function DashboardPlaceholder({ title, path }: { title: string; path: string }) {
  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">{title}</h1>
          <p className="text-[11px] text-muted">Dashboard / {path}</p>
        </div>
      </div>
      <div className="flex-1 p-8 flex items-center justify-center">
        <div className="text-center max-w-[400px]">
          <div className="w-12 h-12 rounded-icon bg-mist flex items-center justify-center mx-auto mb-4">
            <span className="text-forest text-xl">🔬</span>
          </div>
          <h2 className="text-[17px] font-semibold text-ink mb-2">{title}</h2>
          <p className="text-[14px] text-muted leading-relaxed">
            This module is active. Connect your backend to populate real-time data here.
          </p>
        </div>
      </div>
    </div>
  );
}

export const metadata: Metadata = { title: 'Red Team — Niyantrana Dashboard' };
export default function RedTeamPage() { return <DashboardPlaceholder title="LLM Red Teaming" path="Red Team" />; }
