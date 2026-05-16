'use client';

import { useEffect, useRef } from 'react';

export function useScrollAnimation() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
          }
        });
      },
      { threshold: 0.2 }
    );

    const el = ref.current;
    if (!el) return;

    const targets = el.querySelectorAll('.animate-up');
    targets.forEach((t) => observer.observe(t));

    return () => observer.disconnect();
  }, []);

  return ref;
}

export function AnimatedSection({ children, className }: { children: React.ReactNode; className?: string }) {
  const ref = useScrollAnimation();
  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  );
}
