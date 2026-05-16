'use client';

import { cn } from '@/lib/utils';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface StatCardProps {
  label: string;
  value: string | number;
  trend?: 'up' | 'down' | 'flat';
  trendLabel?: string;
  className?: string;
}

export function StatCard({ label, value, trend, trendLabel, className }: StatCardProps) {
  return (
    <div className={cn('bg-white border border-stone rounded-[12px] p-5', className)}>
      <p className="text-[12px] font-medium text-muted uppercase tracking-[0.04em] mb-3">
        {label}
      </p>
      <p className="text-[28px] font-bold text-ink leading-none mb-2">
        {value}
      </p>
      {trend && trendLabel && (
        <div className={cn(
          'flex items-center gap-1 text-[11px] font-medium',
          trend === 'up' ? 'text-forest' : trend === 'down' ? 'text-danger' : 'text-muted'
        )}>
          {trend === 'up' && <TrendingUp size={11} strokeWidth={1.5} />}
          {trend === 'down' && <TrendingDown size={11} strokeWidth={1.5} />}
          {trend === 'flat' && <Minus size={11} strokeWidth={1.5} />}
          <span>{trendLabel}</span>
        </div>
      )}
    </div>
  );
}

export default StatCard;
