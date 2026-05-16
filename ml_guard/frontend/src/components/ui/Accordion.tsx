'use client';

import { useState, useRef, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AccordionItem {
  question: string;
  answer: string;
}

interface AccordionProps {
  items: AccordionItem[];
  className?: string;
}

function AccordionRow({ item, isOpen, onToggle }: {
  item: AccordionItem;
  isOpen: boolean;
  onToggle: () => void;
}) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(0);

  useEffect(() => {
    if (contentRef.current) {
      setHeight(contentRef.current.scrollHeight);
    }
  }, [item.answer]);

  return (
    <div className="border-b border-stone last:border-b-0">
      <button
        onClick={onToggle}
        data-cursor="pointer"
        className="w-full flex items-center justify-between py-5 text-left group"
        aria-expanded={isOpen}
      >
        <span className={cn(
          'text-sm font-semibold transition-colors duration-150',
          isOpen ? 'text-forest' : 'text-ink group-hover:text-forest'
        )}>
          {item.question}
        </span>
        <ChevronDown
          size={16}
          strokeWidth={1.5}
          className={cn(
            'text-muted flex-shrink-0 ml-4 transition-transform duration-250',
            isOpen ? 'rotate-180 text-forest' : ''
          )}
        />
      </button>

      {/* Max-height transition accordion — NOT display:none */}
      <div
        style={{
          maxHeight: isOpen ? `${height}px` : '0px',
          overflow: 'hidden',
          transition: 'max-height 250ms ease-in-out',
        }}
      >
        <div ref={contentRef} className="pb-5">
          <p className="text-sm text-ink-soft leading-relaxed">{item.answer}</p>
        </div>
      </div>
    </div>
  );
}

export function Accordion({ items, className }: AccordionProps) {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const toggle = (i: number) => {
    setOpenIndex(openIndex === i ? null : i);
  };

  return (
    <div className={cn('divide-y-0', className)}>
      {items.map((item, i) => (
        <AccordionRow
          key={i}
          item={item}
          isOpen={openIndex === i}
          onToggle={() => toggle(i)}
        />
      ))}
    </div>
  );
}

export default Accordion;
