'use client';

import { cn } from '@/lib/utils';
import React from 'react';

type ButtonVariant = 'primary' | 'ghost' | 'accent' | 'danger';
type ButtonSize = 'sm' | 'md' | 'lg';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  asChild?: boolean;
  href?: string;
}

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    'bg-ink text-white border border-ink hover:bg-[#1a1a18] active:scale-[0.98]',
  ghost:
    'bg-transparent text-ink border border-ink hover:bg-stone active:scale-[0.98]',
  accent:
    'bg-forest text-white border border-forest hover:bg-[#154d2e] active:scale-[0.98]',
  danger:
    'bg-danger text-white border border-danger hover:bg-[#a93226] active:scale-[0.98]',
};

const sizeClasses: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-xs rounded-[6px]',
  md: 'h-10 px-4 text-sm rounded-[8px]',
  lg: 'h-12 px-6 text-base rounded-[8px]',
};

export function Button({
  variant = 'primary',
  size = 'md',
  className,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      data-cursor={variant === 'primary' ? 'cta' : 'pointer'}
      className={cn(
        'inline-flex items-center justify-center gap-2 font-medium',
        'transition-all duration-150 cursor-pointer',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-forest',
        'disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none',
        variantClasses[variant],
        sizeClasses[size],
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}

export default Button;
