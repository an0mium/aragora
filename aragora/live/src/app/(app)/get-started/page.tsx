'use client';

import Link from 'next/link';
import { useState } from 'react';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

const PLATFORM_STATS = [
  { label: 'Agent Types', value: '42', color: 'text-[var(--acid-green)]' },
  { label: 'Knowledge Adapters', value: '34', color: 'text-[var(--acid-cyan)]' },
  { label: 'API Operations', value: '2,000+', color: 'text-amber-400' },
];

const DEMO_AGENTS = [
  { name: 'Claude (Anthropic)', role: 'Proposer', color: 'text-[var(--acid-green)]' },
  { name: 'GPT-4 (OpenAI)', role: 'Critic', color: 'text-[var(--acid-cyan)]' },
  { name: 'Gemini (Google)', role: 'Synthesizer', color: 'text-amber-400' },
];

const PIPELINE_STAGES = [
  { id: 'ideas', label: 'Ideas', icon: '?', description: 'Capture raw proposals' },
  { id: 'goals', label: 'Goals', icon: '^', description: 'Refine into objectives' },
  { id: 'actions', label: 'Actions', icon: '>', description: 'Plan execution steps' },
  { id: 'orchestration', label: 'Orchestration', icon: '@', description: 'Coordinate agents' },
];

const FEATURED_TEMPLATES = [
  {
    name: 'Security Code Review',
    category: 'Security',
    description: 'Multi-agent adversarial code audit with vulnerability detection and fix suggestions.',
    tags: ['security', 'code-review', 'audit'],
    icon: '!',
  },
  {
    name: 'Vendor Comparison',
    category: 'Decision',
    description: 'Structured debate to evaluate vendor proposals against weighted criteria.',
    tags: ['decision', 'procurement', 'comparison'],
    icon: '^',
  },
  {
    name: 'Research Synthesis',
    category: 'Research',
    description: 'Literature review with evidence grading, contradiction detection, and knowledge extraction.',
    tags: ['research', 'synthesis', 'evidence'],
    icon: '?',
  },
];

const QUICK_LINKS = [
  {
    category: 'Security',
    icon: '!',
    links: [
      { label: 'Code Review', href: '/reviews' },
      { label: 'API Scan', href: '/gauntlet/api' },
      { label: 'Red Team', href: '/gauntlet/redteam' },
    ],
  },
  {
    category: 'Compliance',
    icon: '%',
    links: [
      { label: 'GDPR Check', href: '/gauntlet/gdpr' },
      { label: 'HIPAA Audit', href: '/gauntlet/hipaa' },
      { label: 'SOX Review', href: '/gauntlet/sox' },
    ],
  },
  {
    category: 'Architecture',
    icon: '@',
    links: [
      { label: 'Stress Test', href: '/gauntlet' },
      { label: 'Incident Analysis', href: '/gauntlet/incident' },
      { label: 'Graph Debate', href: '/debates/graph' },
    ],
  },
  {
    category: 'Research',
    icon: '?',
    links: [
      { label: 'New Research', href: '/arena' },
      { label: 'Knowledge Base', href: '/knowledge' },
      { label: 'Evidence', href: '/evidence' },
    ],
  },
  {
    category: 'Decisions',
    icon: '^',
    links: [
      { label: 'New Decision', href: '/arena' },
      { label: 'Receipts', href: '/receipts' },
      { label: 'Matrix Debate', href: '/debates/matrix' },
    ],
  },
  {
    category: 'Industry',
    icon: '/',
    links: [
      { label: 'Healthcare', href: '/verticals/healthcare' },
      { label: 'Finance', href: '/verticals/finance' },
      { label: 'Legal', href: '/verticals/legal' },
    ],
  },
];

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
  const [activeStage, setActiveStage] = useState<string>('ideas');

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
              {PLATFORM_STATS.map((stat) => (
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
              Launch a multi-agent debate where heterogeneous AI models propose,
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
          {/* 3. Self-Improvement */}
          {/* ---------------------------------------------------------------- */}
          <Section id="self-improve" step={3} title="Explore Self-Improvement">
            <p className="text-xs font-mono text-[var(--text-muted)] mb-4 max-w-2xl">
              The Nomic Loop is Aragora&apos;s autonomous self-improvement cycle.
              Agents debate improvements, design solutions, implement code, and
              verify changes -- continuously evolving the platform.
            </p>

            {/* Nomic Loop phases */}
            <div className="flex items-center gap-1 mb-4">
              {['Context', 'Debate', 'Design', 'Implement', 'Verify'].map(
                (phase, idx, arr) => (
                  <div key={phase} className="flex items-center gap-1">
                    <div className="bg-[var(--bg)] border border-[var(--acid-cyan)]/30 px-3 py-1.5 text-center">
                      <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase">
                        Phase {idx}
                      </div>
                      <div className="text-xs font-mono text-[var(--acid-cyan)] font-bold">
                        {phase}
                      </div>
                    </div>
                    {idx < arr.length - 1 && (
                      <span className="text-[var(--acid-cyan)]/50 font-mono text-xs">
                        {'>'}
                      </span>
                    )}
                  </div>
                ),
              )}
            </div>

            {/* Sample goal */}
            <div className="bg-[var(--bg)] border border-[var(--border)] p-3 mb-4">
              <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase">
                Sample Goal
              </span>
              <div className="text-sm font-mono text-[var(--text)] mt-0.5">
                &quot;Improve test coverage for the debate orchestration module&quot;
              </div>
            </div>

            <Link
              href="/self-improve"
              className="inline-block px-4 py-2 text-xs font-mono font-bold bg-[var(--acid-cyan)]/20 text-[var(--acid-cyan)] border border-[var(--acid-cyan)]/40 hover:bg-[var(--acid-cyan)]/30 transition-colors"
            >
              TRY SELF-IMPROVEMENT
            </Link>
          </Section>

          {/* ---------------------------------------------------------------- */}
          {/* 4. Pipeline */}
          {/* ---------------------------------------------------------------- */}
          <Section id="pipeline" step={4} title="Browse the Pipeline">
            <p className="text-xs font-mono text-[var(--text-muted)] mb-4 max-w-2xl">
              The Idea-to-Execution pipeline transforms raw proposals into
              orchestrated agent workflows across four stages. Click a stage below
              to preview.
            </p>

            {/* Interactive pipeline */}
            <div className="flex items-stretch gap-2 mb-4">
              {PIPELINE_STAGES.map((stage, idx) => (
                <button
                  key={stage.id}
                  onClick={() => setActiveStage(stage.id)}
                  className={`flex-1 text-left p-3 border transition-colors ${
                    activeStage === stage.id
                      ? 'bg-[var(--acid-green)]/10 border-[var(--acid-green)]/50'
                      : 'bg-[var(--bg)] border-[var(--border)] hover:border-[var(--acid-green)]/30'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className={`font-mono text-sm ${
                        activeStage === stage.id
                          ? 'text-[var(--acid-green)]'
                          : 'text-[var(--text-muted)]'
                      }`}
                    >
                      {stage.icon}
                    </span>
                    <span
                      className={`text-xs font-mono font-bold uppercase ${
                        activeStage === stage.id
                          ? 'text-[var(--acid-green)]'
                          : 'text-[var(--text)]'
                      }`}
                    >
                      {stage.label}
                    </span>
                    {idx < PIPELINE_STAGES.length - 1 && (
                      <span className="ml-auto text-[var(--text-muted)] font-mono text-xs">
                        {'>'}
                      </span>
                    )}
                  </div>
                  <div className="text-[10px] font-mono text-[var(--text-muted)]">
                    {stage.description}
                  </div>
                </button>
              ))}
            </div>

            {/* Stage detail */}
            <div className="bg-[var(--bg)] border border-[var(--border)] p-3 mb-4 text-xs font-mono text-[var(--text-muted)]">
              {activeStage === 'ideas' && (
                <span>
                  Capture brainstorms, feature requests, and improvement proposals
                  from any source -- chat, email, Slack, or direct entry.
                </span>
              )}
              {activeStage === 'goals' && (
                <span>
                  Refine raw ideas into measurable objectives with success
                  criteria, priority scoring, and dependency mapping.
                </span>
              )}
              {activeStage === 'actions' && (
                <span>
                  Decompose goals into concrete tasks with estimated effort,
                  required agents, and execution order.
                </span>
              )}
              {activeStage === 'orchestration' && (
                <span>
                  Coordinate multi-agent execution with circuit breakers, budget
                  limits, and real-time progress tracking.
                </span>
              )}
            </div>

            <Link
              href="/pipeline"
              className="inline-block px-4 py-2 text-xs font-mono font-bold bg-[var(--acid-green)]/20 text-[var(--acid-green)] border border-[var(--acid-green)]/40 hover:bg-[var(--acid-green)]/30 transition-colors"
            >
              OPEN PIPELINE CANVAS
            </Link>
          </Section>

          {/* ---------------------------------------------------------------- */}
          {/* 5. Templates */}
          {/* ---------------------------------------------------------------- */}
          <Section id="templates" step={5} title="Discover Templates">
            <p className="text-xs font-mono text-[var(--text-muted)] mb-4 max-w-2xl">
              Browse pre-built debate templates, agent configurations, and workflow
              patterns from the marketplace. Install and customize for your use case.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
              {FEATURED_TEMPLATES.map((tmpl) => (
                <div
                  key={tmpl.name}
                  className="bg-[var(--bg)] border border-[var(--border)] p-4 hover:border-[var(--acid-green)]/40 transition-colors"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-[var(--acid-green)] font-mono">{tmpl.icon}</span>
                    <span className="text-xs font-mono text-[var(--acid-cyan)] uppercase">
                      {tmpl.category}
                    </span>
                  </div>
                  <h3 className="text-sm font-mono text-[var(--text)] font-bold mb-1">
                    {tmpl.name}
                  </h3>
                  <p className="text-[10px] font-mono text-[var(--text-muted)] mb-3 line-clamp-2">
                    {tmpl.description}
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {tmpl.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-0.5 text-[10px] font-mono bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)]"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <Link
              href="/marketplace"
              className="inline-block px-4 py-2 text-xs font-mono font-bold bg-[var(--acid-green)]/20 text-[var(--acid-green)] border border-[var(--acid-green)]/40 hover:bg-[var(--acid-green)]/30 transition-colors"
            >
              BROWSE MARKETPLACE
            </Link>
          </Section>

          {/* ---------------------------------------------------------------- */}
          {/* 6. Quick Links */}
          {/* ---------------------------------------------------------------- */}
          <section className="bg-[var(--surface)] border border-[var(--border)] p-5 transition-colors hover:border-[var(--acid-green)]/40">
            <div className="flex items-center gap-3 mb-4">
              <span className="flex items-center justify-center w-7 h-7 bg-[var(--acid-green)]/20 text-[var(--acid-green)] font-mono text-sm font-bold border border-[var(--acid-green)]/40">
                +
              </span>
              <h2 className="text-sm font-mono text-[var(--acid-green)] uppercase tracking-wider">
                Quick Links
              </h2>
            </div>

            <p className="text-xs font-mono text-[var(--text-muted)] mb-4">
              Jump to any feature across the platform.
            </p>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              {QUICK_LINKS.map((group) => (
                <div
                  key={group.category}
                  className="bg-[var(--bg)] border border-[var(--border)] p-3"
                >
                  <div className="flex items-center gap-1.5 mb-2">
                    <span className="text-[var(--acid-green)] font-mono text-sm">
                      {group.icon}
                    </span>
                    <span className="text-[10px] font-mono text-[var(--acid-cyan)] font-bold uppercase">
                      {group.category}
                    </span>
                  </div>
                  <div className="space-y-1">
                    {group.links.map((link) => (
                      <Link
                        key={link.href + link.label}
                        href={link.href}
                        className="block text-xs font-mono text-[var(--text-muted)] hover:text-[var(--acid-green)] transition-colors"
                      >
                        {link.label}
                      </Link>
                    ))}
                  </div>
                </div>
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
