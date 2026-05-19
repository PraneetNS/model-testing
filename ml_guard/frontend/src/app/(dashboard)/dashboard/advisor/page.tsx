'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, Bot, Send, Lightbulb, AlertTriangle, CheckCircle, Sparkles } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { modelsApi } from '@/lib/api';

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000') + '/api/v1';
const HDR = { 'Content-Type': 'application/json', 'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'dev-secret-key' };

const SUGGESTED_QUESTIONS = [
  'Why is my governance score low?',
  'What is causing the drift penalty?',
  'How do I fix the overfitting gap?',
  'Why was deployment blocked?',
  'How can I improve my fairness score?',
  'What does a CRITICAL advisory mean?',
];

function MarkdownText({ text }: { text: string }) {
  // Simple markdown rendering: bold, headers, bullets
  const lines = text.split('\n');
  return (
    <div className="space-y-1.5">
      {lines.map((line, i) => {
        if (line.startsWith('### ')) return <h3 key={i} className="text-[13px] font-bold text-ink mt-3 mb-1">{line.slice(4)}</h3>;
        if (line.startsWith('## ')) return <h2 key={i} className="text-[14px] font-bold text-ink mt-4 mb-1">{line.slice(3)}</h2>;
        if (line.startsWith('- ') || line.startsWith('• ')) return <div key={i} className="flex items-start gap-2 text-[13px] text-ink-soft pl-2"><span className="text-muted mt-1">•</span><span>{line.slice(2)}</span></div>;
        if (line.startsWith('**') && line.endsWith('**')) return <p key={i} className="text-[13px] font-semibold text-ink">{line.slice(2, -2)}</p>;
        if (!line.trim()) return <div key={i} className="h-1" />;
        return <p key={i} className="text-[13px] text-ink-soft leading-relaxed">{line}</p>;
      })}
    </div>
  );
}

export default function AIAdvisorPage() {
  const [scanId, setScanId] = useState('');
  const [question, setQuestion] = useState('Why is my governance score low?');
  const [useLLM, setUseLLM] = useState(false);
  const [asking, setAsking] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<{ q: string; r: any }[]>([]);

  const ask = async () => {
    if (!question.trim()) return;
    setAsking(true); setError(null); setResult(null);
    try {
      const endpoint = useLLM ? '/advisory/explain-with-llm' : '/advisory/explain';
      const body: any = { question };
      if (scanId.trim()) body.scan_id = scanId.trim();
      else body.results_json = {}; // will trigger local fallback
      const r = await fetch(`${BASE}${endpoint}`, { method: 'POST', headers: HDR, body: JSON.stringify(body) });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? `HTTP ${r.status}`);
      setResult(d);
      setHistory(h => [{ q: question, r: d }, ...h.slice(0, 9)]);
    } catch (e: any) { setError(e.message); }
    finally { setAsking(false); }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex items-center justify-between px-8 h-16 border-b border-stone bg-white">
        <div>
          <h1 className="text-[17px] font-semibold text-ink">AI Advisor</h1>
          <p className="text-[11px] text-muted">Context-aware governance recommendations</p>
        </div>
        {result?.advisory_type && (
          <Badge variant="certified">{result.advisory_type === 'llm' ? 'Groq LLM' : 'Local Analysis'}</Badge>
        )}
      </div>

      <div className="flex-1 p-8 overflow-auto">
        <div className="grid lg:grid-cols-[1fr_340px] gap-6">
          {/* Left: main */}
          <div className="space-y-5">
            {/* Input card */}
            <div className="bg-white border border-stone rounded-card p-6">
              <div className="flex items-center gap-2 mb-4">
                <Bot size={16} className="text-forest" />
                <h2 className="text-[14px] font-semibold text-ink">Ask the AI Advisor</h2>
              </div>
              <p className="text-[12px] text-muted mb-4">
                The advisor analyzes governance results from a past audit scan and provides prioritized, actionable recommendations.
              </p>

              <div className="space-y-3 mb-4">
                <div>
                  <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Scan ID (optional)</label>
                  <input value={scanId} onChange={e => setScanId(e.target.value)}
                    placeholder="Leave blank for generic guidance"
                    className="w-full h-10 px-3 text-[13px] font-mono border border-stone rounded-[8px] outline-none focus:border-forest" />
                </div>
                <div>
                  <label className="block text-[12px] font-medium text-ink-soft mb-1.5">Your question</label>
                  <textarea value={question} onChange={e => setQuestion(e.target.value)} rows={3}
                    className="w-full px-3 py-2.5 text-[13px] border border-stone rounded-[8px] outline-none focus:border-forest resize-none" />
                </div>
              </div>

              {/* Suggested questions */}
              <div className="mb-4">
                <p className="text-[11px] text-muted mb-2">Suggested questions:</p>
                <div className="flex flex-wrap gap-2">
                  {SUGGESTED_QUESTIONS.map(q => (
                    <button key={q} onClick={() => setQuestion(q)}
                      className="px-3 py-1 text-[11px] font-medium border border-stone rounded-badge hover:border-forest text-muted hover:text-ink transition-colors">
                      {q}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex items-center gap-4">
                <Button variant="primary" size="sm" onClick={ask} disabled={asking} className="gap-2">
                  {asking ? <><RefreshCw size={13} className="animate-spin" />Analyzing…</> : <><Send size={13} />Ask Advisor</>}
                </Button>
                <label className="flex items-center gap-2 text-[12px] text-muted cursor-pointer">
                  <input type="checkbox" checked={useLLM} onChange={e => setUseLLM(e.target.checked)} className="accent-forest" />
                  <Sparkles size={12} className="text-amber-500" /> Use Groq LLM (if configured)
                </label>
              </div>
            </div>

            {error && <div className="p-4 bg-red-50 border border-red-200 rounded-card text-[13px] text-danger">⚠ {error}</div>}

            {/* Response */}
            {result && (
              <div className="bg-white border border-stone rounded-card overflow-hidden">
                <div className="flex items-center justify-between px-6 py-4 border-b border-stone"
                  style={{ background: '#0F0F0E' }}>
                  <div className="flex items-center gap-2">
                    <Bot size={15} className="text-[#3ECF8E]" />
                    <span className="text-[13px] font-semibold text-white">ML Guard AI Advisor</span>
                  </div>
                  <span className="text-[10px] px-2 py-0.5 rounded-full"
                    style={{ background: 'rgba(62,207,142,0.15)', color: '#3ECF8E' }}>
                    {result.advisory_type === 'llm' ? `Groq · ${result.provider}` : 'Local Analysis'}
                  </span>
                </div>
                <div className="p-6">
                  {result.governance_score != null && (
                    <div className="flex gap-4 mb-5 p-3 bg-[#F7F6F2] rounded-[8px]">
                      <div><p className="text-[10px] text-muted">Governance Score</p><p className="text-[16px] font-bold text-ink">{result.governance_score?.toFixed(0)}/100</p></div>
                      {result.gate_status && (
                        <div><p className="text-[10px] text-muted">Gate Status</p>
                          <Badge variant={result.gate_status === 'PASS' || result.gate_status === 'PASSED' ? 'certified' : 'failed'}>{result.gate_status}</Badge>
                        </div>
                      )}
                    </div>
                  )}
                  <MarkdownText text={result.explanation ?? ''} />
                  {result.disclaimer && (
                    <p className="text-[10px] text-muted mt-5 italic border-t border-stone/50 pt-3">{result.disclaimer}</p>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right: history */}
          <div className="space-y-3">
            <p className="text-[12px] font-semibold text-ink-soft uppercase tracking-[0.05em]">Session history</p>
            {history.length === 0 ? (
              <div className="bg-white border border-stone rounded-card p-5 text-center">
                <Lightbulb size={24} className="mx-auto text-muted mb-2" strokeWidth={1.25} />
                <p className="text-[12px] text-muted">Ask a question to start your advisory session.</p>
              </div>
            ) : (
              history.map(({ q, r }, i) => (
                <button key={i} onClick={() => setResult(r)}
                  className="w-full bg-white border border-stone rounded-card p-4 text-left hover:border-forest/50 transition-colors">
                  <p className="text-[12px] font-medium text-ink truncate mb-1">{q}</p>
                  <p className="text-[11px] text-muted">{r.advisory_type === 'llm' ? '✦ Groq LLM' : '⚙ Local analysis'}</p>
                </button>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
