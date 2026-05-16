'use client';

import { useEffect, useState, useRef } from 'react';
import { usePathname } from 'next/navigation';
import { Logo } from './Logo';

export function RouteLoader({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [loading, setLoading] = useState(false);
  const [visible, setVisible] = useState(false);
  const prevPath = useRef(pathname);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    if (prevPath.current !== pathname) {
      prevPath.current = pathname;
      setLoading(true);
      setVisible(true);

      // Never show loader more than 800ms
      timerRef.current = setTimeout(() => {
        setLoading(false);
        setTimeout(() => setVisible(false), 200);
      }, 800);
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [pathname]);

  return (
    <>
      {/* Page content fades during load */}
      <div
        style={{
          opacity: loading ? 0.3 : 1,
          transition: 'opacity 200ms ease',
        }}
      >
        {children}
      </div>

      {/* Route loader overlay */}
      {visible && (
        <div
          aria-live="polite"
          aria-label="Loading page"
          style={{
            position: 'fixed',
            inset: 0,
            background: '#0F0F0E',
            opacity: loading ? 0.85 : 0,
            zIndex: 9998,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '16px',
            transition: 'opacity 200ms ease',
            pointerEvents: loading ? 'all' : 'none',
          }}
        >
          {/* Spinning logomark */}
          <div
            style={{
              animation: 'spin-slow 0.6s ease-in-out infinite',
            }}
          >
            <Logo size="md" showWordmark={false} />
          </div>

          {/* Wordmark */}
          <span
            style={{
              fontFamily: 'Inter, system-ui, sans-serif',
              fontWeight: 600,
              fontSize: '13px',
              color: '#ffffff',
              letterSpacing: '0.04em',
            }}
          >
            Niyantrana
          </span>

          {/* Progress bar */}
          <div
            style={{
              width: '160px',
              height: '2px',
              background: 'rgba(255,255,255,0.1)',
              borderRadius: '999px',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                height: '100%',
                background: 'linear-gradient(90deg, #1A5F3A, #4CAF80)',
                animation: 'progress-fill 0.8s ease-out forwards',
                borderRadius: '999px',
              }}
            />
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin-slow {
          to { transform: rotate(360deg); }
        }
        @keyframes progress-fill {
          from { width: 0%; }
          to { width: 100%; }
        }
      `}</style>
    </>
  );
}

export default RouteLoader;
