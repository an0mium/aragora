'use client';

import { useState, FormEvent } from 'react';
import Link from 'next/link';
import { DebateResultPreview, type DebateResponse } from './DebateResultPreview';

interface LandingPageProps {
  apiBase?: string;
  wsUrl?: string;
  onDebateStarted?: (debateId: string) => void;
  onEnterDashboard?: () => void;
}

const FAILURE_MODES = [
  {
    title: 'HALLUCINATION',
    problem: 'LLMs fabricate facts with confidence',
    fix: 'Cross-model verification catches fabrications before they reach you',
    accent: 'acid-green',
  },
  {
    title: 'SYCOPHANCY',
    problem: 'LLMs agree with whatever you say',
    fix: 'Adversarial agents are structurally incentivized to disagree',
    accent: 'acid-cyan',
  },
  {
    title: 'INCONSISTENCY',
    problem: 'Ask twice, get contradictory answers',
    fix: 'Debate convergence produces stable, defensible positions',
    accent: 'acid-green',
  },
];

const STEPS = [
  {
    num: '01',
    title: 'QUESTION',
    desc: 'Submit any decision, architecture, or strategy for vetting',
    icon: '?',
  },
  {
    num: '02',
    title: 'AGENTS DEBATE',
    desc: '30+ heterogeneous AI agents argue, critique, and red-team each proposal',
    icon: '#',
  },
  {
    num: '03',
    title: 'DECISION RECEIPT',
    desc: 'Get an audit-ready verdict with evidence chains and dissenting views preserved',
    icon: '!',
  },
];

const FEATURES = [
  {
    title: 'ADVERSARIAL DEBATE',
    desc: '30+ heterogeneous AI agents (Claude, GPT, Gemini, Mistral, Grok, DeepSeek, Qwen) argue, critique, and converge. Real diversity. Real disagreement.',
    accent: 'acid-green',
    icon: '>',
  },
  {
    title: 'DECISION RECEIPTS',
    desc: 'Cryptographic audit trails with SHA-256 hashing. Every claim linked to evidence. Dissenting views preserved with full reasoning chains.',
    accent: 'acid-cyan',
    icon: '#',
  },
  {
    title: 'EU AI ACT READY',
    desc: 'Article 12/13/14 compliance artifact generation. Event logs, tech docs, oversight models, and bias safeguards -- bundled with integrity hashes.',
    accent: 'acid-green',
    icon: '%',
  },
  {
    title: 'GAUNTLET MODE',
    desc: 'Adversarial stress-testing for critical decisions. Red-team your architecture, strategy, or security posture with attack/defend cycles.',
    accent: 'acid-cyan',
    icon: '!',
  },
];

const STATS = [
  { value: '30+', label: 'AI AGENTS' },
  { value: '6', label: 'PROVIDERS' },
  { value: '<30s', label: 'FIRST RESULT' },
  { value: '100%', label: 'AUDIT TRAIL' },
];

const FOOTER_LINKS = [
  { href: 'https://github.com/an0mium/aragora', label: 'GitHub' },
  { href: '/about', label: 'Docs' },
  { href: 'https://api.aragora.ai/docs', label: 'API Reference' },
  { href: 'https://status.aragora.ai', label: 'Status' },
];

export function LandingPage({ apiBase, onEnterDashboard }: LandingPageProps) {
  const [question, setQuestion] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<DebateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const resolvedApiBase = apiBase || 'https://api.aragora.ai';

  async function runDebate(topic: string) {
    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${resolvedApiBase}/api/v1/playground/debate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, rounds: 2, agents: 3 }),
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
      {/* ── NAV ── */}
      <nav className="border-b border-border bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <span className="font-mono text-acid-green font-bold text-sm tracking-wider">
            ARAGORA
          </span>
          <div className="flex items-center gap-4">
            <a href="#how-it-works" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors hidden sm:block">
              [HOW IT WORKS]
            </a>
            <button
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors hidden sm:block"
            >
              [TRY NOW]
            </button>
            <a href="https://github.com/an0mium/aragora" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors hidden sm:block">
              [GITHUB]
            </a>
            {onEnterDashboard ? (
              <button
                onClick={onEnterDashboard}
                className="text-xs font-mono px-3 py-1.5 border border-acid-green text-acid-green hover:bg-acid-green hover:text-bg transition-colors"
              >
                LOG IN
              </button>
            ) : (
              <Link
                href="/auth/login"
                className="text-xs font-mono px-3 py-1.5 border border-acid-green text-acid-green hover:bg-acid-green hover:text-bg transition-colors"
              >
                LOG IN
              </Link>
            )}
          </div>
        </div>
      </nav>

      {/* ── HERO with live debate input ── */}
      <section className="py-16 sm:py-24 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <h1 className="font-mono text-2xl sm:text-4xl lg:text-5xl text-text mb-6 leading-tight">
            Don&apos;t trust a single AI.{' '}
            <span className="text-acid-green">Trust a debate.</span>
          </h1>
          <p className="font-mono text-sm sm:text-base text-text-muted max-w-2xl mx-auto mb-8 leading-relaxed">
            Submit any question. Watch AI agents argue. Get a decision receipt.
          </p>

          {/* Debate input form */}
          <form onSubmit={handleSubmit} className="text-left">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What decision are you facing?"
              disabled={isRunning}
              rows={2}
              className="w-full bg-[var(--surface)] border border-[var(--border)] text-[var(--text)] px-4 py-3 font-mono text-sm placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--acid-green)] transition-colors resize-none disabled:opacity-50"
            />

            <div className="flex justify-center mt-6">
              <button
                type="submit"
                disabled={isRunning || !question.trim()}
                className="font-mono text-sm px-8 py-3 bg-acid-green text-bg font-bold hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isRunning ? 'AGENTS DEBATING...' : 'RUN FREE DEBATE'}
              </button>
            </div>
          </form>

          {/* Loading spinner */}
          {isRunning && (
            <div className="flex items-center justify-center py-8">
              <div className="flex items-center gap-3 text-[var(--acid-green)]">
                <svg
                  className="animate-spin h-5 w-5"
                  viewBox="0 0 24 24"
                  fill="none"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                <span className="text-sm font-mono">Agents are debating...</span>
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="border border-[var(--crimson)] bg-[var(--crimson)]/10 p-4 mt-6 text-left">
              <p className="text-sm text-[var(--crimson)] font-mono">{error}</p>
            </div>
          )}

          {/* Inline results with gated receipt */}
          {result && <DebateResultPreview result={result} />}
        </div>
      </section>

      {/* ── PROBLEM STATEMENT ── */}
      <section className="py-16 px-4 border-t border-border">
        <div className="max-w-5xl mx-auto">
          <h2 className="font-mono text-lg sm:text-xl text-center text-text mb-3">
            LLMs hallucinate. They agree with you. They contradict themselves.
          </h2>
          <p className="font-mono text-xs text-center text-text-muted mb-10 max-w-xl mx-auto">
            Individual models are unreliable. Adversarial debate between diverse models fixes that.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {FAILURE_MODES.map((mode) => (
              <div
                key={mode.title}
                className={`border border-${mode.accent}/30 bg-surface/50 p-5`}
              >
                <h3 className={`font-mono text-sm text-${mode.accent} mb-2`}>
                  {mode.title}
                </h3>
                <p className="font-mono text-xs text-text-muted/70 mb-3 line-through decoration-crimson/40">
                  {mode.problem}
                </p>
                <p className="font-mono text-xs text-text-muted">
                  {mode.fix}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── HOW IT WORKS ── */}
      <section id="how-it-works" className="py-16 px-4 border-t border-border">
        <div className="max-w-5xl mx-auto">
          <h2 className="font-mono text-lg sm:text-xl text-center text-acid-green mb-10">
            {'>'} HOW IT WORKS
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {STEPS.map((step, i) => (
              <div key={step.num} className="flex flex-col items-center text-center">
                <div className="w-14 h-14 border-2 border-acid-green flex items-center justify-center mb-4">
                  <span className="font-mono text-2xl text-acid-green">
                    {step.icon}
                  </span>
                </div>
                <span className="font-mono text-xs text-text-muted mb-1">
                  [{step.num}]
                </span>
                <h3 className="font-mono text-sm text-text mb-2">
                  {step.title}
                </h3>
                <p className="font-mono text-xs text-text-muted leading-relaxed">
                  {step.desc}
                </p>
                {i < STEPS.length - 1 && (
                  <span className="hidden md:block absolute right-0 top-1/2 text-acid-green/40 text-xl translate-x-1/2" aria-hidden="true">
                    {/* Arrow handled by spacing */}
                  </span>
                )}
              </div>
            ))}
          </div>
          {/* Step flow arrows for desktop */}
          <div className="hidden md:flex justify-center items-center gap-0 mt-[-4.5rem] mb-4 pointer-events-none" aria-hidden="true">
            <div className="w-1/3 flex justify-end pr-4">
              <span className="font-mono text-acid-green/40 text-lg">---{'>'}</span>
            </div>
            <div className="w-1/3 flex justify-end pr-4">
              <span className="font-mono text-acid-green/40 text-lg">---{'>'}</span>
            </div>
            <div className="w-1/3" />
          </div>
        </div>
      </section>

      {/* ── KEY FEATURES ── */}
      <section className="py-16 px-4 border-t border-border">
        <div className="max-w-5xl mx-auto">
          <h2 className="font-mono text-lg sm:text-xl text-center text-acid-green mb-10">
            {'>'} KEY CAPABILITIES
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {FEATURES.map((feature) => (
              <div
                key={feature.title}
                className={`border border-${feature.accent}/30 bg-surface/30 p-5 hover:border-${feature.accent}/60 transition-colors`}
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className={`text-${feature.accent} font-mono text-lg`}>
                    {feature.icon}
                  </span>
                  <h3 className={`font-mono text-sm text-${feature.accent}`}>
                    {feature.title}
                  </h3>
                </div>
                <p className="font-mono text-xs text-text-muted leading-relaxed">
                  {feature.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── QUICKSTART ── */}
      <section id="quickstart" className="py-16 px-4 border-t border-border">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="font-mono text-lg sm:text-xl text-acid-green mb-6">
            {'>'} QUICKSTART
          </h2>
          <p className="font-mono text-xs text-text-muted mb-6">
            Get started in under a minute. Run a demo debate locally:
          </p>
          <div className="bg-bg border border-acid-green/40 p-6 text-left font-mono shadow-terminal">
            <div className="text-xs text-text-muted mb-2">$ # Install and run a demo debate</div>
            <div className="text-sm text-acid-green mb-1">
              <span className="text-text-muted select-none">$ </span>
              pip install aragora
            </div>
            <div className="text-sm text-acid-green">
              <span className="text-text-muted select-none">$ </span>
              aragora review --demo
            </div>
          </div>
          <p className="font-mono text-xs text-text-muted/60 mt-4">
            Or{' '}
            <button
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="text-acid-cyan hover:text-acid-green transition-colors"
            >
              scroll up
            </button>{' '}
            to try a free debate right in your browser.
          </p>
        </div>
      </section>

      {/* ── STATS BAR ── */}
      <section className="py-10 px-4 border-t border-acid-green/30 bg-surface/30">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-6 text-center">
            {STATS.map((stat) => (
              <div key={stat.label}>
                <div className="font-mono text-xl sm:text-2xl text-acid-green font-bold">
                  {stat.value}
                </div>
                <div className="font-mono text-xs text-text-muted mt-1">
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="py-8 px-4 border-t border-border">
        <div className="max-w-5xl mx-auto">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <span className="font-mono text-xs text-text-muted">
              ARAGORA // Decision Integrity Platform
            </span>
            <div className="flex items-center gap-4">
              {FOOTER_LINKS.map((link) => (
                <a
                  key={link.label}
                  href={link.href}
                  className="font-mono text-xs text-text-muted hover:text-acid-green transition-colors"
                >
                  {link.label}
                </a>
              ))}
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}
