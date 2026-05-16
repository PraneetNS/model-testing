import type { Metadata } from 'next';

export const metadata: Metadata = { title: 'Compliance — Niyantrana Dashboard' };
export default function CompliancePage() {
  const FRAMEWORKS = [
    { name: 'EU AI Act', score: 87, status: 'CERTIFIED', articles: ['Art. 9', 'Art. 13', 'Art. 17'] },
    { name: 'NIST AI RMF', score: 91, status: 'CERTIFIED', articles: ['Govern', 'Map', 'Measure', 'Manage'] },
    { name: 'ISO 42001', score: 79, status: 'CONDITIONAL', articles: ['6.1', '8.4', '9.1'] },
    { name: 'SOC 2 Type II', score: 94, status: 'CERTIFIED', articles: ['CC6', 'CC7', 'A1'] },
    { name: 'GDPR', score: 96, status: 'CERTIFIED', articles: ['Art. 22', 'Art. 35'] },
  ];

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">Regulatory Compliance</h1>
          <p className="text-[11px] text-muted">Dashboard / Compliance</p>
        </div>
      </div>
      <div className="flex-1 p-8">
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {FRAMEWORKS.map((fw) => (
            <div key={fw.name} className="bg-white border border-stone rounded-card p-6">
              <div className="flex items-start justify-between mb-4">
                <h3 className="text-[15px] font-semibold text-ink">{fw.name}</h3>
                <span className={`text-[10px] font-bold uppercase tracking-[0.04em] px-2 py-0.5 rounded-badge ${
                  fw.status === 'CERTIFIED' ? 'bg-mist text-forest' : 'bg-amber-50 text-amber-700'
                }`}>{fw.status}</span>
              </div>
              <div className="mb-4">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[12px] text-muted">Compliance score</span>
                  <span className="text-[13px] font-semibold text-ink">{fw.score}/100</span>
                </div>
                <div className="h-1.5 bg-stone rounded-full overflow-hidden">
                  <div className="h-full rounded-full" style={{ width: `${fw.score}%`, background: '#1A5F3A' }} />
                </div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {fw.articles.map((a) => (
                  <span key={a} className="text-[11px] font-mono bg-sage text-forest px-2 py-0.5 rounded-[4px]">{a}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
