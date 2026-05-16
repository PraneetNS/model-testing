'use client';

import { cn } from '@/lib/utils';

interface Tab {
  id: string;
  label: string;
}

interface TabsProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (id: string) => void;
  className?: string;
}

export function Tabs({ tabs, activeTab, onTabChange, className }: TabsProps) {
  return (
    <div className={cn('flex border-b border-stone', className)}>
      {tabs.map((tab) => {
        const isActive = tab.id === activeTab;
        return (
          <button
            key={tab.id}
            data-cursor="pointer"
            onClick={() => onTabChange(tab.id)}
            className={cn(
              'px-4 py-3 text-sm font-medium relative',
              'transition-colors duration-150',
              'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-forest',
              isActive
                ? 'text-ink'
                : 'text-muted hover:text-ink-soft'
            )}
          >
            {tab.label}
            {/* Underline indicator */}
            {isActive && (
              <span
                className="absolute bottom-0 left-0 right-0 h-[2px] bg-forest"
                style={{ transition: 'opacity 150ms ease' }}
              />
            )}
          </button>
        );
      })}
    </div>
  );
}

export default Tabs;
