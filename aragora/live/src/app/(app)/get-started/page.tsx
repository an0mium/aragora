'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react'; // useState used by usePlatformStats
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { API_BASE_URL } from '@/config';

// ---------------------------------------------------------------------------
// Live data hook — replaces hardcoded constants with backend reality
// ---------------------------------------------------------------------------

function usePlatformStats() {
  const [stats, setStats] = useState([
    { label: 'Agent Types', value: '—', color: 'text-[var(--acid-green)]' },
    { label: 'Knowledge Adapters', value: '—', color: 'text-[var(--acid-cyan)]' },
    { label: 'Debates Run', value: '—', color: 'text-amber-400' },
  ]);

  useEffect(() => {
    const url = API_BASE_URL;
    // Best-effort fetch — falls back to static values on any error
    Promise.allSettled([
      fetch(`${url}/api/v1/health`, { signal: AbortSignal.timeout(5000) }).then((r) => r.json()),
      fetch(`${url}/api/v1/debates?limit=1&offset=0`, { signal: AbortSignal.timeout(5000) }).then(
        (r) => r.json()
      ),
    ]).then(([healthResult, debatesResult]) => {
      const health =
        healthResult.status === 'fulfilled' ? (healthResult.value as Record<string, unknown>) : {};
      const debates =
        debatesResult.status === 'fulfilled'
          ? (debatesResult.value as Record<string, unknown>)
          : {};

      const agentCount = (health.agent_types as number) ?? 43;
      const adapterCount = (health.km_adapters as number) ?? 45;
      const debateCount = (debates.total as number) ?? 0;

      setStats([
        {
          label: 'Agent Types',
          value: String(agentCount),
          color: 'text-[var(--acid-green)]',
        },
        {
          label: 'Knowledge Adapters',
          value: String(adapterCount),
          color: 'text-[var(--acid-cyan)]',
        },
        {
          label: 'Debates Run',
          value: debateCount > 0 ? String(debateCount) : '0',
          color: 'text-amber-400',
        },
      ]);
    });
  }, []);

  return stats;
}

// ---------------------------------------------------------------------------
// Static data (templates, links — these are UI navigation, not metrics)
// ---------------------------------------------------------------------------

const DEMO_AGENTS = [
  { name: 'Claude (Anthropic)', role: 'Proposer', color: 'text-[var(--acid-green)]' },
  { name: 'GPT-4 (OpenAI)', role: 'Critic', color: 'text-[var(--acid-cyan)]' },
  { name: 'Gemini (Google)', role: 'Synthesizer', color: 'text-amber-400' },
];

// Static navigation data removed — replaced by focused onboarding funnel.
// Pipeline, Templates, Quick Links sections consolidated into "Explore More".

// ---------------------------------------------------------------------------
// Section wrapper
// ---------------------------------------------------------------------------

function Section({
  id,
  step,
  title,
  children,
}: {
  id: string;
  step: number;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section
      id={id}
      className="bg-[var(--surface)] border border-[var(--border)] p-5 transition-colors hover:border-[var(--acid-green)]/40"
    >
      <div className="flex items-center gap-3 mb-4">
        <span className="flex items-center justify-center w-7 h-7 bg-[var(--acid-green)]/20 text-[var(--acid-green)] font-mono text-sm font-bold border border-[var(--acid-green)]/40">
          {step}
        </span>
        <h2 className="text-sm font-mono text-[var(--acid-green)] uppercase tracking-wider">
          {title}
        </h2>
      </div>
      {children}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function GetStartedPage() {
  const platformStats = usePlatformStats();

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-[var(--bg)] text-[var(--text)] relative z-10">
        <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
          {/* Breadcrumb */}
          <div className="flex items-center gap-3 mb-2">
            <Link
              href="/dashboard"
              className="text-xs font-mono text-[var(--text-muted)] hover:text-[var(--acid-green)] transition-colors"
            >
              DASHBOARD
            </Link>
            <span className="text-xs font-mono text-[var(--text-muted)]">/</span>
            <span className="text-xs font-mono text-[var(--acid-green)]">GET STARTED</span>
          </div>

          {/* Page header */}
          <div className="mb-2">
            <h1 className="text-2xl font-mono text-[var(--acid-green)] mb-1">
              {'>'} GET STARTED WITH ARAGORA
            </h1>
            <p className="text-sm font-mono text-[var(--text-muted)] max-w-2xl">
              See the full platform in 5 minutes. Follow each step below to run
              your first debate, explore self-improvement, and discover enterprise
              features.
            </p>
          </div>

          {/* ---------------------------------------------------------------- */}
          {/* 1. Welcome */}
          {/* ---------------------------------------------------------------- */}
          <Section id="welcome" step={1} title="Welcome to the Decision Integrity Platform">
            <p className="text-xs font-mono text-[var(--text-muted)] mb-4 max-w-2xl">
              Aragora orchestrates AI agents to adversarially vet decisions against
              your organization&apos;s knowledge, then delivers audit-ready decision
              receipts to any channel.
            </p>

            <div className="grid grid-cols-3 gap-3">
              {platformStats.map((stat) => (
                <div
                  key={stat.label}
                  className="bg-[var(--bg)] border border-[var(--border)] p-3 text-center"
                >
                  <div className={`text-2xl font-mono font-bold ${stat.color}`}>
                    {stat.value}
                  </div>
                  <div className="text-[10px] font-mono text-[var(--text-muted)] mt-1 uppercase">
                    {stat.label}
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* ---------------------------------------------------------------- */}
          {/* 2. First Debate */}
          {/* ---------------------------------------------------------------- */}
          <Section id="first-debate" step={2} title="Run Your First Debate">
            <p className="text-xs font-mono text-[var(--text-muted)] mb-4 max-w-2xl">
              Launch a multi-agent debate where different AI models propose,
              critique, and synthesize a decision. Try the example below or write
              your own topic.
            </p>

            <div className="bg-[var(--bg)] border border-[var(--acid-green)]/30 p-4">
              {/* Topic */}
              <div className="mb-3">
                <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase">
                  Topic
                </span>
                <div className="text-sm font-mono text-[var(--text)] mt-0.5">
                  &quot;Should we adopt microservices for our next product?&quot;
                </div>
              </div>

              {/* Agent lineup */}
              <div className="mb-4">
                <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase">
                  Agents
                </span>
                <div className="flex flex-wrap gap-2 mt-1">
                  {DEMO_AGENTS.map((agent) => (
                    <div
                      key={agent.name}
                      className="flex items-center gap-2 bg-[var(--surface)] border border-[var(--border)] px-3 py-1.5"
                    >
                      <span className={`text-xs font-mono font-bold ${agent.color}`}>
                        {agent.role.toUpperCase()}
                      </span>
                      <span className="text-xs font-mono text-[var(--text-muted)]">
                        {agent.name}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Protocol details */}
              <div className="flex items-center gap-4 mb-4 text-[10px] font-mono text-[var(--text-muted)]">
                <span>Rounds: 3</span>
                <span>Consensus: majority</span>
                <span>Convergence: enabled</span>
              </div>

              {/* Action */}
              <Link
                href="/arena?topic=Should+we+adopt+microservices+for+our+next+product%3F&rounds=3&consensus=majority"
                className="inline-block px-4 py-2 text-xs font-mono font-bold bg-[var(--acid-green)] text-[var(--bg)] hover:bg-[var(--acid-green)]/80 transition-colors"
              >
                START DEBATE
              </Link>
            </div>
          </Section>

          {/* ---------------------------------------------------------------- */}
          {/* 3. See Your Receipts */}
          {/* ---------------------------------------------------------------- */}
          <Section id="receipts" step={3} title="Review Your Decision Receipts">
            <p className="text-xs font-mono text-[var(--text-muted)] mb-4 max-w-2xl">
              Every debate produces a cryptographic receipt with the final verdict,
              confidence score, dissenting views, and SHA-256 content hash.
              Receipts are your audit trail for every decision.
            </p>

            <div className="flex items-center gap-3">
              <Link
                href="/receipts"
                className="inline-block px-4 py-2 text-xs font-mono font-bold bg-[var(--acid-green)] text-[var(--bg)] hover:bg-[var(--acid-green)]/80 transition-colors"
              >
                VIEW RECEIPTS
              </Link>
              <Link
                href="/knowledge"
                className="inline-block px-4 py-2 text-xs font-mono font-bold bg-[var(--acid-green)]/20 text-[var(--acid-green)] border border-[var(--acid-green)]/40 hover:bg-[var(--acid-green)]/30 transition-colors"
              >
                KNOWLEDGE BASE
              </Link>
            </div>
          </Section>

          {/* ---------------------------------------------------------------- */}
          {/* Explore More (compact links) */}
          {/* ---------------------------------------------------------------- */}
          <section className="bg-[var(--surface)] border border-[var(--border)] p-5 transition-colors hover:border-[var(--acid-green)]/40">
            <h2 className="text-sm font-mono text-[var(--text-muted)] uppercase tracking-wider mb-3">
              Explore More
            </h2>
            <div className="flex flex-wrap gap-2">
              {[
                { label: 'Pipeline', href: '/pipeline' },
                { label: 'Self-Improve', href: '/self-improve' },
                { label: 'Templates', href: '/marketplace' },
                { label: 'Code Review', href: '/reviews' },
                { label: 'Gauntlet', href: '/gauntlet' },
                { label: 'Settings', href: '/settings' },
              ].map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="px-3 py-1.5 text-xs font-mono bg-[var(--bg)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/40 hover:text-[var(--acid-green)] transition-colors"
                >
                  {link.label}
                </Link>
              ))}
            </div>
          </section>

          {/* ---------------------------------------------------------------- */}
          {/* Footer navigation */}
          {/* ---------------------------------------------------------------- */}
          <div className="flex items-center gap-2 pt-4 border-t border-[var(--border)]">
            <span className="text-xs font-mono text-[var(--text-muted)]">Navigate:</span>
            <Link
              href="/dashboard"
              className="px-3 py-1 text-xs font-mono bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
            >
              DASHBOARD
            </Link>
            <Link
              href="/arena"
              className="px-3 py-1 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30 hover:bg-[var(--acid-green)]/20 transition-colors"
            >
              NEW DEBATE
            </Link>
            <Link
              href="/self-improve"
              className="px-3 py-1 text-xs font-mono bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
            >
              SELF-IMPROVE
            </Link>
            <Link
              href="/marketplace"
              className="px-3 py-1 text-xs font-mono bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
            >
              MARKETPLACE
            </Link>
          </div>
        </div>
      </main>
    </>
  );
}
