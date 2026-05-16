'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/Button';
import { Logo } from '@/components/ui/Logo';
import { useAuth } from '@/context/AuthContext';

export default function LoginPage() {
  const { signInWithGoogle } = useAuth();

  return (
    <div className="min-h-screen flex" style={{ background: '#F7F6F2' }}>
      {/* Left: Brand */}
      <div
        className="hidden lg:flex flex-col justify-between p-14 w-[480px] flex-shrink-0"
        style={{ background: '#0F0F0E' }}
      >
        <Logo size="sm" showWordmark wordmarkColor="#FFFFFF" />
        <div>
          <blockquote
            className="text-[22px] font-light leading-relaxed mb-6"
            style={{ color: '#E8E5DF', letterSpacing: '-0.01em' }}
          >
            "The contract your model must keep."
          </blockquote>
          <p className="text-[13px]" style={{ color: '#888884' }}>
            3,200+ models audited. 91.4 average governance score.
            Every deployment, certified.
          </p>
        </div>
        <div className="flex items-center gap-2 text-[12px]" style={{ color: '#555552' }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: '#FF6600' }} />
          Backed by YC W26
        </div>
      </div>

      {/* Right: Form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-[400px]">
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-ink mb-2" style={{ letterSpacing: '-0.02em' }}>
              Welcome back
            </h1>
            <p className="text-[14px] text-muted">
              Don't have an account?{' '}
              <Link href="/signup" className="text-forest underline underline-offset-4">
                Sign up free
              </Link>
            </p>
          </div>

          <form className="flex flex-col gap-4" onSubmit={(e) => e.preventDefault()}>
            <div>
              <label htmlFor="email" className="block text-[12px] font-medium text-ink-soft mb-1.5">
                Email address
              </label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                placeholder="you@company.com"
                className="w-full h-10 px-3 text-[14px] text-ink bg-white border border-stone rounded-[8px] outline-none focus:border-forest transition-colors duration-150 placeholder:text-muted"
              />
            </div>
            <div>
              <label htmlFor="password" className="block text-[12px] font-medium text-ink-soft mb-1.5">
                Password
              </label>
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                placeholder="••••••••"
                className="w-full h-10 px-3 text-[14px] text-ink bg-white border border-stone rounded-[8px] outline-none focus:border-forest transition-colors duration-150 placeholder:text-muted"
              />
            </div>
            <div className="flex items-center justify-end">
              <a href="#" className="text-[12px] text-forest underline underline-offset-4">
                Forgot password?
              </a>
            </div>

            <Button variant="primary" size="md" className="w-full justify-center mt-2" data-cursor="cta">
              Sign in
            </Button>
          </form>

          <div className="flex items-center gap-3 my-6">
            <div className="flex-1 h-px bg-stone" />
            <span className="text-[12px] text-muted">or</span>
            <div className="flex-1 h-px bg-stone" />
          </div>

          <Button 
            variant="ghost" 
            size="md" 
            className="w-full justify-center gap-2"
            onClick={() => signInWithGoogle()}
          >
            <svg width="18" height="18" viewBox="0 0 24 24">
              <path
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                fill="#4285F4"
              />
              <path
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-1 .67-2.28 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                fill="#34A853"
              />
              <path
                d="M5.84 14.09c-.22-.67-.35-1.39-.35-2.09s.13-1.42.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z"
                fill="#FBBC05"
              />
              <path
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                fill="#EA4335"
              />
            </svg>
            Continue with Google
          </Button>
        </div>
      </div>
    </div>
  );
}
