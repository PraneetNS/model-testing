'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Search, X } from 'lucide-react';

interface SearchResult {
  title: string;
  description?: string;
  href: string;
}

const MOCK_RESULTS: SearchResult[] = [
  { title: 'Quick Start Guide', description: 'Get up and running in minutes', href: '/docs/quick-start' },
  { title: 'Behavioral Contracts', description: 'Define and enforce model promises', href: '/docs/behavioral-contracts' },
  { title: 'Governance Scoring', description: 'How scores are calculated', href: '/docs/governance-scoring' },
  { title: 'Drift Detection', description: 'PSI, KS-Test, Jensen-Shannon', href: '/docs/drift-detection' },
  { title: 'AIBOM', description: 'AI Bill of Materials explained', href: '/docs/aibom' },
  { title: 'CI/CD Gate', description: 'Block bad models from shipping', href: '/docs/cicd-setup' },
  { title: 'API Reference', description: 'Full REST API documentation', href: '/docs/api-reference' },
  { title: 'Pricing', description: 'Free, Pro, and Enterprise tiers', href: '/pricing' },
];

interface SearchDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

function SearchDialog({ isOpen, onClose }: SearchDialogProps) {
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const results = MOCK_RESULTS.filter(
    (r) =>
      query === '' ||
      r.title.toLowerCase().includes(query.toLowerCase()) ||
      r.description?.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
      setQuery('');
      setActiveIndex(0);
    }
  }, [isOpen]);

  const handleKey = useCallback(
    (e: KeyboardEvent) => {
      if (!isOpen) return;
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveIndex((i) => Math.min(i + 1, results.length - 1));
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveIndex((i) => Math.max(i - 1, 0));
      }
      if (e.key === 'Enter' && results[activeIndex]) {
        window.location.href = results[activeIndex].href;
        onClose();
      }
    },
    [isOpen, onClose, results, activeIndex]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [handleKey]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[9990] flex items-start justify-center pt-24"
      style={{ background: 'rgba(15,15,14,0.7)', backdropFilter: 'blur(4px)' }}
      onClick={onClose}
    >
      <div
        className="w-full max-w-[560px] mx-4 rounded-[14px] overflow-hidden"
        style={{
          background: '#0F0F0E',
          border: '1px solid #1a1a18',
          animation: 'slide-up 0.2s ease-out',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input */}
        <div className="flex items-center gap-3 px-4 py-4 border-b border-[#1a1a18]">
          <Search size={16} strokeWidth={1.5} className="text-[#888884] flex-shrink-0" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Search documentation, features..."
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActiveIndex(0);
            }}
            className="flex-1 bg-transparent text-white text-sm outline-none placeholder:text-[#888884]"
            style={{ fontFamily: 'Inter, system-ui, sans-serif' }}
          />
          <button
            onClick={onClose}
            className="text-[#888884] hover:text-white transition-colors duration-150"
          >
            <X size={14} strokeWidth={1.5} />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-[360px] overflow-y-auto p-2">
          {results.length === 0 ? (
            <p className="py-8 text-center text-sm text-[#888884]">No results found</p>
          ) : (
            results.map((result, i) => (
              <a
                key={result.href}
                href={result.href}
                onClick={onClose}
                className={`
                  flex flex-col gap-0.5 px-3 py-2.5 rounded-[8px] transition-colors duration-100
                  ${i === activeIndex ? 'bg-forest' : 'hover:bg-[#1a1a18]'}
                `}
                onMouseEnter={() => setActiveIndex(i)}
              >
                <span className={`text-sm font-medium ${i === activeIndex ? 'text-white' : 'text-[#E8E5DF]'}`}>
                  {result.title}
                </span>
                {result.description && (
                  <span className={`text-xs ${i === activeIndex ? 'text-white/70' : 'text-[#888884]'}`}>
                    {result.description}
                  </span>
                )}
              </a>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-4 px-4 py-2.5 border-t border-[#1a1a18]">
          <span className="text-[11px] text-[#888884]">↑↓ navigate</span>
          <span className="text-[11px] text-[#888884]">↵ select</span>
          <span className="text-[11px] text-[#888884]">Esc close</span>
        </div>
      </div>
    </div>
  );
}

// Hook to control the search dialog
export function useSearchDialog() {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  return { isOpen, open: () => setIsOpen(true), close: () => setIsOpen(false) };
}

export { SearchDialog };
export default SearchDialog;
