'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, X } from 'lucide-react';
import { Logo } from '@/components/ui/Logo';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/utils';
import { useAuth } from '@/context/AuthContext';

const NAV_LINKS = [
  { label: 'Platform', href: '/#features' },
  { label: 'Docs', href: '/docs' },
  { label: 'Pricing', href: '/pricing' },
  { label: 'FAQ', href: '/faq' },
  { label: 'About', href: '/about' },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const pathname = usePathname();
  const { user } = useAuth();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const isActive = (href: string) => {
    if (href.startsWith('/#')) return pathname === '/';
    return pathname.startsWith(href);
  };

  return (
    <header
      className={cn(
        'sticky top-0 z-50 w-full border-b border-stone',
        'transition-all duration-200',
        scrolled ? 'backdrop-blur-nav bg-ivory/90' : 'bg-ivory'
      )}
    >
      <div className="container-site">
        <div className="flex items-center justify-between h-16">
          {/* Left: Logo */}
          <Link href="/" aria-label="Niyantrana home">
            <Logo size="sm" showWordmark />
          </Link>

          {/* Center: Nav links (desktop) */}
          <nav className="hidden md:flex items-center gap-6" aria-label="Main navigation">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  'text-[13px] font-medium no-underline transition-colors duration-150',
                  'hover:text-ink hover:underline',
                  'underline-offset-4 decoration-forest',
                  isActive(link.href) ? 'text-ink underline decoration-forest' : 'text-ink-soft'
                )}
              >
                {link.label}
              </Link>
            ))}
          </nav>

          {/* Right: CTAs + YC badge */}
          <div className="hidden md:flex items-center gap-3">

            {user ? (
              <Link href="/dashboard">
                <Button variant="primary" size="sm" data-cursor="cta">Dashboard</Button>
              </Link>
            ) : (
              <>
                <Link href="/login">
                  <Button variant="ghost" size="sm">Log in</Button>
                </Link>
                <Link href="/signup">
                  <Button variant="primary" size="sm" data-cursor="cta">Get started</Button>
                </Link>
              </>
            )}
          </div>

          {/* Mobile: hamburger */}
          <button
            className="md:hidden text-ink-soft hover:text-ink transition-colors duration-150"
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label={menuOpen ? 'Close menu' : 'Open menu'}
            data-cursor="pointer"
          >
            {menuOpen ? <X size={20} strokeWidth={1.5} /> : <Menu size={20} strokeWidth={1.5} />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden border-t border-stone bg-ivory">
          <div className="container-site py-4 flex flex-col gap-1">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMenuOpen(false)}
                className={cn(
                  'py-3 text-[15px] font-medium border-b border-stone/50 last:border-0',
                  isActive(link.href) ? 'text-forest' : 'text-ink-soft'
                )}
              >
                {link.label}
              </Link>
            ))}
            <div className="flex flex-col gap-2 mt-4">
              {user ? (
                <Link href="/dashboard" onClick={() => setMenuOpen(false)}>
                  <Button variant="primary" size="md" className="w-full justify-center">Dashboard</Button>
                </Link>
              ) : (
                <>
                  <Link href="/login" onClick={() => setMenuOpen(false)}>
                    <Button variant="ghost" size="md" className="w-full justify-center">Log in</Button>
                  </Link>
                  <Link href="/signup" onClick={() => setMenuOpen(false)}>
                    <Button variant="primary" size="md" className="w-full justify-center">Get started</Button>
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </header>
  );
}

export default Navbar;
