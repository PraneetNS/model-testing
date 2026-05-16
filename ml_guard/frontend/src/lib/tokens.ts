// Niyantrana Brand Tokens
// DO NOT import colours directly in components — use Tailwind classes
// This file is the single source of truth for the design system.

export const colors = {
  ivory: '#F7F6F2',
  stone: '#E8E5DF',
  ink: '#0F0F0E',
  inkSoft: '#3D3D3A',
  muted: '#888884',
  forest: '#1A5F3A',
  mint: '#4CAF80',
  mist: '#E1F5EE',
  sage: '#E8F0E8',
  white: '#FFFFFF',
  danger: '#C0392B',
  warning: '#B35A00',
  // Logo ring colours
  outerRing: '#2A7A50',
  innerRing: '#4CAF80',
  logoBg: '#0F0F0E',
} as const;

export const typography = {
  display: { weight: 800, tracking: '-0.04em', lineHeight: '1.0' },
  h1: { weight: 700, tracking: '-0.03em', lineHeight: '1.1' },
  h2: { weight: 600, tracking: '-0.02em' },
  h3: { weight: 600, tracking: '-0.01em' },
  body: { weight: 400, tracking: '0em', lineHeight: '1.65' },
  caption: { weight: 500, tracking: '0.02em' },
  code: { weight: 400 },
} as const;

export const spacing = {
  containerMax: '1120px',
  containerPadMobile: '24px',
  containerPadDesktop: '48px',
} as const;

export const borderRadius = {
  card: '14px',
  button: '8px',
  badge: '999px',
  icon: '10px',
} as const;

export const transitions = {
  fast: '150ms ease',
  default: '200ms ease',
  slow: '500ms ease-out',
  accordion: '250ms ease-in-out',
} as const;
