export function HeroIllustration() {
  return (
    <div
      className="w-full rounded-[20px] border border-stone overflow-hidden"
      style={{ background: '#fff', minHeight: '420px' }}
    >
      <svg
        viewBox="0 0 680 420"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-auto"
        role="img"
        aria-label="Niyantrana governance dashboard preview"
      >
        {/* Sidebar */}
        <rect width="160" height="420" fill="#0F0F0E" />
        {/* Sidebar logo area */}
        <rect x="16" y="20" width="128" height="28" rx="6" fill="#1a1a18" />
        <text x="28" y="39" fill="#4CAF80" fontSize="10" fontFamily="Inter, system-ui" fontWeight="600">● Niyantrana</text>

        {/* Sidebar nav items */}
        {['Overview', 'Models', 'Contracts', 'Drift', 'Red Team', 'AIBOM', 'Compliance', 'Alerts'].map((item, i) => (
          <g key={item}>
            <rect
              x="8"
              y={72 + i * 34}
              width="144"
              height="28"
              rx="4"
              fill={i === 0 ? '#1A5F3A' : 'transparent'}
            />
            {i === 0 && <rect x="8" y={72 + i * 34} width="3" height="28" fill="#4CAF80" />}
            <text
              x="24"
              y={72 + i * 34 + 18}
              fill={i === 0 ? '#fff' : '#888884'}
              fontSize="10"
              fontFamily="Inter, system-ui"
              fontWeight={i === 0 ? '500' : '400'}
            >
              {item}
            </text>
          </g>
        ))}

        {/* Main area */}
        <rect x="160" y="0" width="520" height="420" fill="#F7F6F2" />

        {/* Top bar */}
        <rect x="160" y="0" width="520" height="44" fill="#fff" />
        <text x="176" y="28" fill="#0F0F0E" fontSize="13" fontFamily="Inter, system-ui" fontWeight="600">Dashboard Overview</text>

        {/* Governance Score Radial Gauge */}
        <g transform="translate(460, 130)">
          {/* Gauge arc background */}
          <circle cx="0" cy="0" r="56" stroke="#E8E5DF" strokeWidth="8" fill="none" />
          {/* Gauge arc foreground — 91% of circle */}
          <circle
            cx="0"
            cy="0"
            r="56"
            stroke="#1A5F3A"
            strokeWidth="8"
            fill="none"
            strokeDasharray={`${(91 / 100) * 2 * Math.PI * 56} ${2 * Math.PI * 56}`}
            strokeDashoffset={2 * Math.PI * 56 * 0.25}
            strokeLinecap="round"
          />
          <text x="0" y="-6" textAnchor="middle" fill="#0F0F0E" fontSize="22" fontFamily="Inter, system-ui" fontWeight="700">91</text>
          <text x="0" y="10" textAnchor="middle" fill="#888884" fontSize="8" fontFamily="Inter, system-ui">/100</text>

          {/* CERTIFIED badge */}
          <rect x="-30" y="22" width="60" height="16" rx="8" fill="#E1F5EE" />
          <text x="0" y="34" textAnchor="middle" fill="#1A5F3A" fontSize="7" fontFamily="Inter, system-ui" fontWeight="700">CERTIFIED</text>
        </g>

        {/* Score label */}
        <text x="460" y="210" textAnchor="middle" fill="#888884" fontSize="9" fontFamily="Inter, system-ui">Governance Score</text>

        {/* Mini drift chart */}
        <rect x="176" y="64" width="240" height="80" rx="8" fill="#fff" stroke="#E8E5DF" strokeWidth="1" />
        <text x="190" y="82" fill="#888884" fontSize="8" fontFamily="Inter, system-ui" fontWeight="600" letterSpacing="0.04em">DRIFT MONITOR</text>
        {/* Chart line */}
        <polyline
          points="190,120 210,118 230,117 250,116 270,119 290,118 310,115 330,116 350,114 370,117 390,116"
          fill="none"
          stroke="#1A5F3A"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* Chart baseline */}
        <line x1="190" y1="130" x2="400" y2="130" stroke="#E8E5DF" strokeWidth="1" />

        {/* Behavioral contracts list */}
        <rect x="176" y="160" width="260" height="100" rx="8" fill="#fff" stroke="#E8E5DF" strokeWidth="1" />
        <text x="190" y="178" fill="#888884" fontSize="8" fontFamily="Inter, system-ui" fontWeight="600" letterSpacing="0.04em">BEHAVIORAL CONTRACTS</text>

        {[
          { name: 'Fairness Parity', status: 'PASS' },
          { name: 'Confidence Threshold ≥ 0.85', status: 'PASS' },
          { name: 'PII Non-Disclosure', status: 'PASS' },
        ].map((contract, i) => (
          <g key={contract.name}>
            <text x="190" y={200 + i * 22} fill="#3D3D3A" fontSize="9" fontFamily="Inter, system-ui">{contract.name}</text>
            <rect x="380" y={190 + i * 22} width="36" height="14" rx="7" fill="#E1F5EE" />
            <text x="398" y={201 + i * 22} textAnchor="middle" fill="#1A5F3A" fontSize="7" fontFamily="Inter, system-ui" fontWeight="700">{contract.status}</text>
          </g>
        ))}

        {/* Stats row */}
        {[
          { label: 'Models', value: '12' },
          { label: 'Contracts', value: '47' },
          { label: 'Alerts', value: '3' },
          { label: 'Avg Score', value: '91.4' },
        ].map((stat, i) => (
          <g key={stat.label}>
            <rect x={460 + i * 60} y={240} width="52" height="40" rx="6" fill="#fff" stroke="#E8E5DF" strokeWidth="1" />
            <text x={486 + i * 60} y={258} textAnchor="middle" fill="#0F0F0E" fontSize="11" fontFamily="Inter, system-ui" fontWeight="700">{stat.value}</text>
            <text x={486 + i * 60} y={272} textAnchor="middle" fill="#888884" fontSize="7" fontFamily="Inter, system-ui">{stat.label}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export default HeroIllustration;
