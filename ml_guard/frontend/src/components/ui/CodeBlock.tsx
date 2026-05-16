'use client';

import { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { cn } from '@/lib/utils';

interface CodeBlockProps {
  code: string;
  language?: string;
  className?: string;
}

export function CodeBlock({ code, language = 'bash', className }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div
      className={cn(
        'relative rounded-[10px] overflow-hidden border border-[#1a1a18]',
        className
      )}
      style={{ background: '#0F0F0E' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[#1a1a18]">
        <span className="text-[11px] font-medium text-[#888884] uppercase tracking-[0.04em]">
          {language}
        </span>
        <button
          onClick={handleCopy}
          data-cursor="pointer"
          className="flex items-center gap-1.5 text-[11px] text-[#888884] hover:text-white transition-colors duration-150"
          aria-label="Copy code"
        >
          {copied ? (
            <>
              <Check size={12} strokeWidth={1.5} />
              <span>Copied</span>
            </>
          ) : (
            <>
              <Copy size={12} strokeWidth={1.5} />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code */}
      <pre className="p-4 overflow-x-auto">
        <code
          className="text-[13px] leading-relaxed"
          style={{
            fontFamily: 'JetBrains Mono, monospace',
            color: '#4CAF80',
          }}
        >
          {code}
        </code>
      </pre>
    </div>
  );
}

export default CodeBlock;
