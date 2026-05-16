'use client';

import { useEffect, useRef } from 'react';

export function CustomCursor() {
  const cursorRef = useRef<HTMLDivElement>(null);
  const mouseX = useRef(0);
  const mouseY = useRef(0);
  const curX = useRef(0);
  const curY = useRef(0);
  const stateRef = useRef<'default' | 'pointer' | 'cta' | 'text'>('default');

  useEffect(() => {
    // Hide default cursor
    document.documentElement.style.cursor = 'none';

    const onMouseMove = (e: MouseEvent) => {
      mouseX.current = e.clientX;
      mouseY.current = e.clientY;

      const target = e.target as Element;
      if (target.closest('[data-cursor="cta"]')) {
        stateRef.current = 'cta';
      } else if (
        target.closest('button') ||
        target.closest('a') ||
        target.closest('[data-cursor="pointer"]')
      ) {
        stateRef.current = 'pointer';
      } else if (target.closest('p') || target.closest('[data-cursor="text"]')) {
        stateRef.current = 'text';
      } else {
        stateRef.current = 'default';
      }
    };

    const loop = () => {
      curX.current += (mouseX.current - curX.current) * 0.12;
      curY.current += (mouseY.current - curY.current) * 0.12;

      const el = cursorRef.current;
      if (el) {
        const state = stateRef.current;

        if (state === 'cta') {
          el.style.width = '36px';
          el.style.height = '36px';
          el.style.background = '#4CAF80';
          el.style.border = 'none';
          el.style.mixBlendMode = 'normal';
        } else if (state === 'pointer') {
          el.style.width = '36px';
          el.style.height = '36px';
          el.style.background = 'transparent';
          el.style.border = '2px solid #1A5F3A';
          el.style.mixBlendMode = 'normal';
        } else if (state === 'text') {
          el.style.width = '8px';
          el.style.height = '8px';
          el.style.background = '#0F0F0E';
          el.style.border = '1px solid #1A5F3A';
          el.style.mixBlendMode = 'difference';
        } else {
          el.style.width = '13px';
          el.style.height = '13px';
          el.style.background = '#0F0F0E';
          el.style.border = 'none';
          el.style.mixBlendMode = 'difference';
        }

        el.style.transform = `translate(${curX.current - parseFloat(el.style.width) / 2}px, ${curY.current - parseFloat(el.style.height) / 2}px)`;
      }

      requestAnimationFrame(loop);
    };

    window.addEventListener('mousemove', onMouseMove);
    const frame = requestAnimationFrame(loop);

    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      cancelAnimationFrame(frame);
      document.documentElement.style.cursor = '';
    };
  }, []);

  return (
    <div
      ref={cursorRef}
      aria-hidden="true"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '13px',
        height: '13px',
        borderRadius: '50%',
        background: '#0F0F0E',
        pointerEvents: 'none',
        zIndex: 9999,
        transition: 'width 0.15s ease, height 0.15s ease, background 0.15s ease, border 0.15s ease',
        willChange: 'transform',
      }}
    />
  );
}

export default CustomCursor;
