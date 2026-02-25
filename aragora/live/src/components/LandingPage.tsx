'use client';

import { useState, useCallback, FormEvent } from 'react';
import Link from 'next/link';
import { DebateResultPreview, RETURN_URL_KEY, PENDING_DEBATE_KEY, type DebateResponse } from './DebateResultPreview';
import { getCurrentReturnUrl, normalizeReturnUrl } from '@/utils/returnUrl';

interface LandingPageProps {
  apiBase?: string;
  wsUrl?: string;
  onDebateStarted?: (debateId: string) => void;
  onEnterDashboard?: () => void;
}

export function LandingPage({ apiBase, onEnterDashboard }: LandingPageProps) {
  const [question, setQuestion] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<DebateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const resolvedApiBase = apiBase || 'https://api.aragora.ai';

  const saveDebateBeforeLogin = useCallback(() => {
    if (result) {
      sessionStorage.setItem(PENDING_DEBATE_KEY, JSON.stringify(result));
      const debateDestination = result.id ? `/debates/${encodeURIComponent(result.id)}` : getCurrentReturnUrl();
      sessionStorage.setItem(RETURN_URL_KEY, normalizeReturnUrl(debateDestination));
    }
  }, [result]);

  async function runDebate(topic: string) {
    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${resolvedApiBase}/api/v1/playground/debate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, question: topic, rounds: 2, agents: 3, source: 'landing' }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        setError(data?.error || `Request failed (${res.status})`);
        return;
      }

      setResult(await res.json());
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Network error';
      setError(message);
    } finally {
      setIsRunning(false);
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (question.trim()) {
      runDebate(question.trim());
    }
  }

  return (
    <main className="min-h-screen bg-bg text-text">
      {/* Nav */}
      <nav className="border-b border-border bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <span className="font-mono text-acid-green font-bold text-sm tracking-wider">
            ARAGORA
          </span>
          <div className="flex items-center gap-4">
            <a href="#how-it-works" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors hidden sm:block">
              How it works
            </a>
            <Link href="/oracle" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors hidden sm:block">
              Oracle
            </Link>
            {onEnterDashboard ? (
              <button
                onClick={() => { saveDebateBeforeLogin(); onEnterDashboard(); }}
                className="text-xs font-mono px-3 py-1.5 border border-acid-green/40 text-text-muted hover:text-acid-green hover:border-acid-green transition-colors"
              >
                Log in
              </button>
            ) : (
              <Link
                href="/login"
                onClick={saveDebateBeforeLogin}
                className="text-xs font-mono px-3 py-1.5 border border-acid-green/40 text-text-muted hover:text-acid-green hover:border-acid-green transition-colors"
              >
                Log in
              </Link>
            )}
            <Link
              href="/signup"
              onClick={saveDebateBeforeLogin}
              className="text-xs font-mono px-3 py-1.5 bg-acid-green text-bg hover:bg-acid-green/80 transition-colors font-bold"
            >
              Sign up free
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="py-20 sm:py-32 px-4">
        <div className="max-w-2xl mx-auto text-center">
          <h1 className="font-mono text-3xl sm:text-5xl text-text mb-6 leading-tight">
            Don&apos;t trust one AI.
            <br />
            <span className="text-acid-green">Make them argue.</span>
          </h1>
          <p className="font-mono text-sm text-text-muted max-w-lg mx-auto mb-12 leading-relaxed">
            Multiple AI models debate your question, stress-test each answer,
            and deliver an audit-ready verdict you can actually defend.
          </p>

          <form onSubmit={handleSubmit} className="text-left max-w-xl mx-auto">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What decision are you facing?"
              disabled={isRunning}
              rows={2}
              className="w-full bg-surface border border-border text-text px-4 py-3 font-mono text-sm placeholder:text-text-muted/50 focus:outline-none focus:border-acid-green transition-colors resize-none disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={isRunning || !question.trim()}
              className="w-full mt-3 font-mono text-sm px-8 py-3 bg-acid-green text-bg font-bold hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {isRunning ? 'Agents debating...' : 'Run a free debate'}
            </button>
          </form>

          {isRunning && (
            <div className="flex items-center justify-center py-8">
              <div className="flex items-center gap-3 text-acid-green">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span className="text-sm font-mono">Agents are debating...</span>
              </div>
            </div>
          )}

          {error && (
            <div className="border border-crimson bg-crimson/10 p-4 mt-6 text-left max-w-xl mx-auto">
              <p className="text-sm text-crimson font-mono">{error}</p>
            </div>
          )}

          {result && <DebateResultPreview result={result} />}
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="py-20 px-4 border-t border-border">
        <div className="max-w-3xl mx-auto">
          <h2 className="font-mono text-sm text-text-muted text-center mb-12 tracking-widest uppercase">
            How it works
          </h2>
          <div className="space-y-12">
            {[
              { step: '01', title: 'You ask a question', desc: 'Any decision, strategy, or architecture question you need vetted.' },
              { step: '02', title: 'AI agents debate it', desc: 'Claude, GPT, Gemini, Mistral, and others argue every angle. Different models catch different blind spots.' },
              { step: '03', title: 'You get a decision receipt', desc: 'An audit-ready verdict with evidence chains, confidence scores, and dissenting views preserved.' },
            ].map((item) => (
              <div key={item.step} className="flex gap-6 items-start">
                <span className="font-mono text-acid-green text-sm mt-0.5 flex-shrink-0">{item.step}</span>
                <div>
                  <h3 className="font-mono text-base text-text mb-1">{item.title}</h3>
                  <p className="font-mono text-sm text-text-muted leading-relaxed">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Why debate */}
      <section className="py-20 px-4 border-t border-border">
        <div className="max-w-3xl mx-auto">
          <h2 className="font-mono text-sm text-text-muted text-center mb-4 tracking-widest uppercase">
            Why this matters
          </h2>
          <p className="font-mono text-lg text-center text-text mb-12 max-w-xl mx-auto leading-relaxed">
            A single AI hallucinates, agrees with you, and contradicts itself.
            Adversarial debate fixes all three.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { problem: 'Hallucination', fix: 'Cross-model verification catches fabrications before they reach you.' },
              { problem: 'Sycophancy', fix: 'Agents are structurally incentivized to disagree and find flaws.' },
              { problem: 'Inconsistency', fix: 'Debate convergence produces stable, defensible positions.' },
            ].map((item) => (
              <div key={item.problem}>
                <h3 className="font-mono text-sm text-acid-green mb-2">{item.problem}</h3>
                <p className="font-mono text-xs text-text-muted leading-relaxed">{item.fix}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Bottom CTA */}
      <section className="py-20 px-4 border-t border-border">
        <div className="max-w-2xl mx-auto text-center">
          <p className="font-mono text-sm text-text-muted mb-6">
            No signup required. First result in under 30 seconds.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <button
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="font-mono text-sm px-8 py-3 bg-acid-green text-bg font-bold hover:opacity-90 transition-opacity"
            >
              Try it now
            </button>
            <Link
              href="/signup"
              className="font-mono text-sm px-8 py-3 border border-border text-text-muted hover:border-acid-green hover:text-acid-green transition-colors"
            >
              Create an account
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-6 px-4 border-t border-border">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <span className="font-mono text-xs text-text-muted/50">
            Aragora
          </span>
          <div className="flex items-center gap-6">
            <a href="/about" className="font-mono text-xs text-text-muted/50 hover:text-text-muted transition-colors">About</a>
            <a href="/pricing" className="font-mono text-xs text-text-muted/50 hover:text-text-muted transition-colors">Pricing</a>
            <a href="mailto:support@aragora.ai" className="font-mono text-xs text-text-muted/50 hover:text-text-muted transition-colors">Support</a>
          </div>
        </div>
      </footer>
    </main>
  );
}
