import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'About — Niyantrana AI Governance',
  description: 'We believe AI should be accountable — not just capable.',
};

const TEAM = [
  { name: 'Priya Rajan', role: 'CEO & Co-founder', initials: 'PR' },
  { name: 'Arjun Mehta', role: 'CTO & Co-founder', initials: 'AM' },
  { name: 'Leila Osei', role: 'Head of Research', initials: 'LO' },
  { name: 'Sam Torres', role: 'Engineering Lead', initials: 'ST' },
  { name: 'Nina Patel', role: 'Head of Compliance', initials: 'NP' },
  { name: 'Marco Diaz', role: 'Product Design', initials: 'MD' },
];

export default function AboutPage() {
  return (
    <div className="container-site py-20">
      {/* Hero */}
      <div className="max-w-[680px] mb-20">
        <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-forest mb-4">About</p>
        <h1 className="text-5xl font-bold text-ink mb-6" style={{ letterSpacing: '-0.03em', lineHeight: 1.1 }}>
          We believe AI should be accountable — not just capable.
        </h1>
        <p className="text-[16px] text-ink-soft leading-relaxed mb-6">
          Niyantrana (Sanskrit: "control / governance") was born from a simple frustration: every AI team we knew was shipping models with nothing more than a spreadsheet and good intentions.
        </p>
        <p className="text-[16px] text-ink-soft leading-relaxed">
          Regulation is arriving. Insurance requirements are hardening. Boards are asking questions that ML teams can't yet answer with evidence. We built Niyantrana to close that gap — to make governance something you prove, not something you promise.
        </p>
      </div>

      {/* Mission block */}
      <div className="rounded-[20px] p-10 mb-20" style={{ background: '#E1F5EE' }}>
        <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-forest mb-3">Our mission</p>
        <p className="text-[20px] font-semibold text-ink leading-relaxed max-w-[600px]" style={{ letterSpacing: '-0.01em' }}>
          "Make AI governance as rigorous and verifiable as software testing — so that every model in production has a cryptographic proof of its behavior."
        </p>
      </div>

      {/* Team */}
      <div className="mb-20">
        <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-forest mb-4">The team</p>
        <h2 className="text-3xl font-semibold text-ink mb-10" style={{ letterSpacing: '-0.02em' }}>
          Built by ML engineers and compliance leads.
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-5">
          {TEAM.map((member) => (
            <div key={member.name} className="flex flex-col items-center text-center gap-3">
              <div
                className="w-16 h-16 rounded-full flex items-center justify-center text-white font-semibold text-[15px]"
                style={{ background: '#1A5F3A' }}
              >
                {member.initials}
              </div>
              <div>
                <p className="text-[13px] font-semibold text-ink">{member.name}</p>
                <p className="text-[12px] text-muted">{member.role}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Built at */}
      <div className="rounded-card p-8 border border-stone bg-white flex flex-col md:flex-row items-start md:items-center gap-6">
        <div className="flex-1">
          <p className="text-[11px] font-bold uppercase tracking-[0.08em] text-forest mb-2">Built at</p>
          <h3 className="text-xl font-semibold text-ink mb-2">FireFlink ML Research</h3>
          <p className="text-[14px] text-ink-soft leading-relaxed max-w-[480px]">
            Niyantrana is the flagship product of FireFlink ML Research, an applied AI safety lab focused on making production AI systems measurably safer and more accountable.
          </p>
        </div>

      </div>
    </div>
  );
}
