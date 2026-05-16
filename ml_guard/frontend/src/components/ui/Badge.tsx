'use client';

import { cn } from '@/lib/utils';

type BadgeVariant = 'certified' | 'conditional' | 'failed' | 'monitoring' | 'new' | 'danger';

interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  className?: string;
}

const variantClasses: Record<BadgeVariant, string> = {
  certified: 'bg-mist text-forest border border-forest/20',
  conditional: 'bg-amber-50 text-amber-700 border border-amber-200',
  failed: 'bg-red-50 text-danger border border-danger/20',
  monitoring: 'bg-sage text-forest border border-forest/20',
  new: 'bg-ink text-white border border-ink',
  danger: 'bg-danger text-white border border-danger',
};

export function Badge({ variant = 'certified', children, className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-0.5',
        'text-[10px] font-bold uppercase tracking-[0.04em]',
        'rounded-badge leading-none',
        variantClasses[variant],
        className
      )}
    >
      {children}
    </span>
  );
}

export default Badge;
