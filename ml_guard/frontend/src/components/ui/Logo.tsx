'use client';

interface LogoProps {
  size?: 'sm' | 'md';
  showWordmark?: boolean;
  wordmarkColor?: string;
}

const SIZES = {
  sm: 24,
  md: 42,
};

export function Logo({ size = 'sm', showWordmark = true, wordmarkColor = '#0F0F0E' }: LogoProps) {
  const px = SIZES[size];
  const fontSize = size === 'sm' ? 16 : 24;

  return (
    <div className="flex items-center gap-2.5 no-select">
      <svg
        width={px}
        height={px}
        viewBox="0 0 42 42"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-label="Niyantrana logomark"
        role="img"
      >
        {/* Background */}
        <rect width="42" height="42" rx="10" fill="#0F0F0E" />

        {/* Outer ring */}
        <circle cx="21" cy="21" r="16" stroke="#2A7A50" strokeWidth="1.5" fill="none" />

        {/* N tick mark */}
        <line x1="21" y1="4" x2="21" y2="7" stroke="#2A7A50" strokeWidth="1.5" strokeLinecap="round" />
        {/* S tick mark */}
        <line x1="21" y1="35" x2="21" y2="38" stroke="#2A7A50" strokeWidth="1.5" strokeLinecap="round" />
        {/* E tick mark */}
        <line x1="35" y1="21" x2="38" y2="21" stroke="#2A7A50" strokeWidth="1.5" strokeLinecap="round" />
        {/* W tick mark */}
        <line x1="4" y1="21" x2="7" y2="21" stroke="#2A7A50" strokeWidth="1.5" strokeLinecap="round" />

        {/* Inner ring */}
        <circle cx="21" cy="21" r="10" stroke="#4CAF80" strokeWidth="1.5" fill="none" />

        {/* Center dot */}
        <circle cx="21" cy="21" r="2.5" fill="#4CAF80" />

        {/* North arrow (directional indicator from center ring, pointing up) */}
        <line x1="21" y1="11" x2="21" y2="18" stroke="#4CAF80" strokeWidth="1.5" strokeLinecap="round" />
        <polyline points="18.5,13.5 21,11 23.5,13.5" stroke="#4CAF80" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      </svg>

      {showWordmark && (
        <span
          style={{
            fontFamily: 'Inter, system-ui, sans-serif',
            fontWeight: 700,
            fontSize: `${fontSize}px`,
            letterSpacing: '-0.03em',
            color: wordmarkColor,
            lineHeight: 1,
          }}
        >
          Niyantrana
        </span>
      )}
    </div>
  );
}

export default Logo;
