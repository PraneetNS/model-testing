import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        ivory: '#F7F6F2',
        stone: '#E8E5DF',
        ink: '#0F0F0E',
        'ink-soft': '#3D3D3A',
        muted: '#888884',
        forest: '#1A5F3A',
        mint: '#4CAF80',
        mist: '#E1F5EE',
        sage: '#E8F0E8',
        danger: '#C0392B',
        warning: '#B35A00',
        brand: {
          outer: '#2A7A50',
          inner: '#4CAF80',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      fontSize: {
        '2xs': ['10px', { lineHeight: '1.4', letterSpacing: '0.04em' }],
      },
      letterSpacing: {
        display: '-0.04em',
        h1: '-0.03em',
        h2: '-0.02em',
        h3: '-0.01em',
        caption: '0.02em',
        wide: '0.04em',
      },
      lineHeight: {
        display: '1.0',
        heading: '1.1',
        relaxed: '1.65',
      },
      borderRadius: {
        card: '14px',
        badge: '999px',
        icon: '10px',
      },
      maxWidth: {
        container: '1120px',
      },
      animation: {
        'spin-slow': 'spin 2s linear infinite',
        'spin-once': 'spin 0.6s ease-in-out',
        'pulse-ring': 'pulse-ring 3s ease-in-out infinite',
        'fade-in': 'fade-in 0.2s ease-out',
        'slide-up': 'slide-up 0.5s ease-out',
        'progress-fill': 'progress-fill 0.8s ease-out forwards',
        'scroll-logos': 'scroll-logos 30s linear infinite',
      },
      keyframes: {
        'pulse-ring': {
          '0%, 100%': { transform: 'scale(1.0)' },
          '50%': { transform: 'scale(1.03)' },
        },
        'fade-in': {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        'slide-up': {
          from: { opacity: '0', transform: 'translateY(24px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'progress-fill': {
          from: { width: '0%' },
          to: { width: '100%' },
        },
        'scroll-logos': {
          '0%': { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-50%)' },
        },
      },
      transitionDuration: {
        '150': '150ms',
        '200': '200ms',
        '250': '250ms',
        '500': '500ms',
      },
      backgroundImage: {
        'forest-mint': 'linear-gradient(90deg, #1A5F3A, #4CAF80)',
      },
    },
  },
  plugins: [],
};

export default config;
