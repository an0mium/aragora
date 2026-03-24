'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { API_BASE_URL } from '@/config';

interface RecordedEvent {
  type: 'proposal' | 'critique' | 'vote' | 'consensus';
  agent: string;
  model: string;
  content: string;
  round: number;
  confidence?: number;
  vote?: 'support' | 'oppose' | 'neutral';
}

interface RecordedDebate {
  id: string;
  topic: string;
  agents: string[];
  rounds: number;
  confidence: number;
  verdict: string;
  receiptHash: string;
  events: RecordedEvent[];
}

interface LiveDebateResult {
  id: string;
  topic: string;
  status: string;
  rounds_used: number;
  consensus_reached: boolean;
  confidence: number;
  verdict: string | null;
  duration_seconds: number;
  participants: string[];
  proposals: Record<string, string>;
  final_answer: string;
  receipt_hash: string | null;
  share_url?: string;
  is_live?: boolean;
  mock_fallback?: boolean;
  mock_fallback_reason?: string;
}

const DEMO_TOPIC =
  'Should our startup adopt AI-powered code review as a mandatory step in our CI/CD pipeline?';

const LIVE_PROGRESS_STEPS = [
  'Submitting the canonical public demo question',
  'Collecting multi-agent positions from the playground backend',
  'Persisting a shareable debate result',
];

const RECORDED_SAMPLE: RecordedDebate = {
  id: 'demo_showcase_001',
  topic: DEMO_TOPIC,
  agents: ['claude-sonnet', 'gpt-4o', 'gemini-pro', 'mistral-large', 'grok-2'],
  rounds: 2,
  confidence: 0.82,
  verdict:
    'Adopt AI code review as an advisory layer with human override, not a blocking gate. Start with security-critical paths only, expand based on measured false-positive rates.',
  receiptHash: 'a7f3c91b2d4e8f06',
  events: [
    {
      type: 'proposal',
      agent: 'claude-sonnet',
      model: 'Claude Sonnet',
      content:
        'Yes, adopt it as mandatory. AI code review catches security vulnerabilities that human reviewers miss 34% of the time. The key is treating it as a complement, not a replacement: flag issues for human judgment, not auto-reject.',
      round: 1,
      confidence: 0.85,
    },
    {
      type: 'proposal',
      agent: 'gpt-4o',
      model: 'GPT-4o',
      content:
        'Mandatory is too strong. AI review tools have 15-25% false positive rates on complex codebases. Making it mandatory creates alert fatigue and slows velocity. Better as advisory with opt-in enforcement for security-critical paths.',
      round: 1,
      confidence: 0.78,
    },
    {
      type: 'critique',
      agent: 'gemini-pro',
      model: 'Gemini Pro',
      content:
        'Both proposals miss the cost dimension. AI review at scale costs $0.02-0.15 per PR. For a team doing 200 PRs/week, that is $200-1,500/month. The ROI depends on your current bug escape rate.',
      round: 1,
      confidence: 0.72,
    },
    {
      type: 'critique',
      agent: 'mistral-large',
      model: 'Mistral Large',
      content:
        'The benchmark-based security claim does not map cleanly onto production code. AI reviewers are strongest on recurring patterns and weaker on business-logic failures.',
      round: 1,
      confidence: 0.76,
    },
    {
      type: 'proposal',
      agent: 'grok-2',
      model: 'Grok 2',
      content:
        'The right framing is path-based enforcement. Security-critical code should require AI review. Lower-risk product code should stay advisory.',
      round: 2,
      confidence: 0.88,
    },
    {
      type: 'vote',
      agent: 'claude-sonnet',
      model: 'Claude Sonnet',
      content: 'I revise my position. Path-based mandatory review is the pragmatic middle ground.',
      round: 2,
      vote: 'support',
      confidence: 0.84,
    },
    {
      type: 'vote',
      agent: 'gpt-4o',
      model: 'GPT-4o',
      content: 'Tiered enforcement addresses my velocity concern while maintaining security coverage.',
      round: 2,
      vote: 'support',
      confidence: 0.81,
    },
    {
      type: 'consensus',
      agent: 'system',
      model: 'Consensus Engine',
      content:
        'Consensus reached. Adopt AI code review as an advisory layer with mandatory enforcement on security-critical paths, then measure false-positive rate and ROI at 90 days.',
      round: 2,
      confidence: 0.82,
    },
  ],
};

const AGENT_ACCENTS = [
  '#15803d',
  '#2563eb',
  '#d97706',
  '#dc2626',
  '#7c3aed',
  '#0f766e',
];

function accentForAgent(agent: string): string {
  let hash = 0;
  for (const char of agent) {
    hash = (hash + char.charCodeAt(0)) % AGENT_ACCENTS.length;
  }
  return AGENT_ACCENTS[hash];
}

function formatAgentName(agent: string): string {
  const replacements: Record<string, string> = {
    claude: 'Claude',
    'claude-sonnet': 'Claude Sonnet',
    gpt: 'GPT',
    'gpt-4o': 'GPT-4o',
    grok: 'Grok',
    'grok-2': 'Grok 2',
    gemini: 'Gemini',
    'gemini-pro': 'Gemini Pro',
    mistral: 'Mistral',
    'mistral-large': 'Mistral Large',
    system: 'Consensus Engine',
  };

  const normalized = agent.trim().toLowerCase();
  if (replacements[normalized]) {
    return replacements[normalized];
  }

  return normalized
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => {
      if (part === 'gpt') return 'GPT';
      if (part === 'ai') return 'AI';
      if (/^\d/.test(part)) return part;
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(' ');
}

function normalizeProposals(
  proposals: unknown,
  participants: string[],
): Record<string, string> {
  if (proposals && typeof proposals === 'object' && !Array.isArray(proposals)) {
    return Object.fromEntries(
      Object.entries(proposals).map(([agent, value]) => [agent, String(value ?? '')]),
    );
  }

  if (Array.isArray(proposals)) {
    return Object.fromEntries(
      proposals.map((value, index) => [
        participants[index] ?? `agent_${index + 1}`,
        typeof value === 'string' ? value : JSON.stringify(value),
      ]),
    );
  }

  return {};
}

function normalizeLiveDebateResult(data: unknown): LiveDebateResult | null {
  if (!data || typeof data !== 'object') {
    return null;
  }

  const raw = data as Record<string, unknown>;
  const participants = Array.isArray(raw.participants)
    ? raw.participants.map((participant) => String(participant))
    : [];

  return {
    id: String(raw.id ?? ''),
    topic: String(raw.topic ?? DEMO_TOPIC),
    status: String(raw.status ?? 'completed'),
    rounds_used: Number(raw.rounds_used ?? 1),
    consensus_reached: Boolean(raw.consensus_reached),
    confidence: Number(raw.confidence ?? 0),
    verdict: raw.verdict == null ? null : String(raw.verdict),
    duration_seconds: Number(raw.duration_seconds ?? 0),
    participants,
    proposals: normalizeProposals(raw.proposals, participants),
    final_answer: String(raw.final_answer ?? ''),
    receipt_hash: raw.receipt_hash == null ? null : String(raw.receipt_hash),
    share_url: raw.share_url == null ? undefined : String(raw.share_url),
    is_live: raw.is_live == null ? undefined : Boolean(raw.is_live),
    mock_fallback: Boolean(raw.mock_fallback),
    mock_fallback_reason:
      raw.mock_fallback_reason == null ? undefined : String(raw.mock_fallback_reason),
  };
}

function formatVerdict(result: LiveDebateResult): string {
  if (result.final_answer.trim()) {
    return result.final_answer.trim();
  }
  if (result.verdict) {
    return result.verdict.replace(/_/g, ' ');
  }
  return 'No verdict returned.';
}

function StatusBadge({
  label,
  tone,
}: {
  label: string;
  tone: 'live' | 'fallback' | 'sample';
}) {
  const styles = {
    live: 'border-[var(--acid-green)]/40 bg-[var(--acid-green)]/10 text-[var(--acid-green)]',
    fallback: 'border-amber-500/25 bg-amber-500/8 text-amber-700',
    sample: 'border-sky-500/20 bg-sky-500/8 text-sky-700',
  }[tone];

  return (
    <span className={`inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] ${styles}`}>
      {label}
    </span>
  );
}

function AgentRoster({ agents }: { agents: string[] }) {
  return (
    <div className="flex flex-wrap gap-2.5">
      {agents.map((agent) => {
        const accent = accentForAgent(agent);
        return (
          <div
            key={agent}
            className="rounded-full border px-3.5 py-1.5 text-sm font-semibold tracking-[0.04em] shadow-[var(--shadow-panel)]"
            style={{ borderColor: `${accent}26`, color: accent, backgroundColor: `${accent}10` }}
          >
            {formatAgentName(agent)}
          </div>
        );
      })}
    </div>
  );
}

function ConsensusBar({ confidence }: { confidence: number }) {
  const clamped = Math.max(0, Math.min(confidence, 1));

  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between gap-3 text-[13px] font-medium text-[var(--text-muted)]">
        <span className="uppercase tracking-[0.12em]">Consensus confidence</span>
        <span className="rounded-full bg-[var(--surface-elevated)] px-2.5 py-1 font-semibold text-[var(--acid-green)] shadow-[var(--shadow-panel)]">
          {Math.round(clamped * 100)}%
        </span>
      </div>
      <div className="h-3 overflow-hidden rounded-full border border-[var(--border)] bg-[var(--surface-elevated)]">
        <div
          className="h-full bg-[var(--acid-green)]"
          style={{ width: `${clamped * 100}%` }}
        />
      </div>
    </div>
  );
}

function LiveResultCard({
  result,
  runStartedAt,
}: {
  result: LiveDebateResult;
  runStartedAt: string | null;
}) {
  const resultTone = result.mock_fallback || result.is_live === false ? 'fallback' : 'live';
  const resultLabel =
    resultTone === 'live' ? 'Live-backed result' : 'Simulated fallback';
  const summary = formatVerdict(result);
  const shareHref = result.share_url ?? `/debate/${result.id}`;
  const proposalEntries = Object.entries(result.proposals).slice(0, 3);

  return (
    <section className="space-y-5 rounded-[20px] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-elevated)] md:p-7">
      <div className="flex flex-col gap-4 border-b border-[var(--border)] pb-5 md:flex-row md:items-start md:justify-between">
        <div className="space-y-3">
          <StatusBadge label={resultLabel} tone={resultTone} />
          <div className="space-y-1">
            <p className="text-base font-semibold text-[var(--text)]">
              {result.topic}
            </p>
            <p className="max-w-2xl text-sm leading-7 text-[var(--text-muted)] text-pretty">
            {resultTone === 'live'
              ? 'Fresh result from the public playground backend.'
              : `The backend returned a non-live fallback${result.mock_fallback_reason ? `: ${result.mock_fallback_reason}` : '.'}`}
            </p>
          </div>
        </div>
        <div className="grid min-w-[220px] grid-cols-2 gap-3 text-sm text-[var(--text-muted)] md:text-right">
          <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-elevated)] px-3 py-2 shadow-[var(--shadow-panel)]">
            <div className="text-[11px] uppercase tracking-[0.14em]">Runtime</div>
            <div className="mt-1 font-semibold text-[var(--text)]">{result.duration_seconds.toFixed(1)}s</div>
          </div>
          <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-elevated)] px-3 py-2 shadow-[var(--shadow-panel)]">
            <div className="text-[11px] uppercase tracking-[0.14em]">Started</div>
            <div className="mt-1 font-semibold text-[var(--text)]">{runStartedAt ?? 'Just now'}</div>
          </div>
          <div className="col-span-2 rounded-2xl border border-[var(--border)] bg-[var(--surface-elevated)] px-3 py-2 shadow-[var(--shadow-panel)]">
            <div className="text-[11px] uppercase tracking-[0.14em]">Result ID</div>
            <div className="mt-1 font-mono text-xs text-[var(--text)]">{result.id}</div>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--acid-green)]">
          Returned agents
        </h3>
        <AgentRoster agents={result.participants} />
      </div>

      <ConsensusBar confidence={result.confidence} />

      <div className="space-y-4 rounded-[18px] border border-[var(--border)] bg-[var(--surface-elevated)] p-6 shadow-[var(--shadow-panel)]">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--acid-green)]">
          Verdict
        </h3>
        <p className="max-w-3xl text-[17px] leading-8 text-[var(--text)] text-pretty">
          {summary}
        </p>
      </div>

      {proposalEntries.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--acid-green)]">
            Agent positions
          </h3>
          <div className="grid grid-cols-1 gap-4">
            {proposalEntries.map(([agent, proposal]) => {
              const accent = accentForAgent(agent);
              return (
                <div
                  key={agent}
                  className="space-y-3 rounded-[18px] border bg-[var(--surface)] p-5 shadow-[var(--shadow-panel)]"
                  style={{ borderColor: `${accent}28`, boxShadow: `inset 3px 0 0 ${accent}` }}
                >
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="text-lg font-bold uppercase tracking-[0.08em]" style={{ color: accent }}>
                      {formatAgentName(agent)}
                    </div>
                    <span
                      className="rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.16em]"
                      style={{ color: accent, backgroundColor: `${accent}10` }}
                    >
                      Position
                    </span>
                  </div>
                  <p className="max-w-3xl text-[15px] leading-7 text-[var(--text)] text-pretty">
                    {proposal}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-4 border-t border-[var(--border)] pt-3 text-sm text-[var(--text-muted)]">
        <span className="rounded-full bg-[var(--surface-elevated)] px-3 py-1 shadow-[var(--shadow-panel)]">
          Rounds {result.rounds_used}
        </span>
        <span className="rounded-full bg-[var(--surface-elevated)] px-3 py-1 shadow-[var(--shadow-panel)]">
          Status {result.status}
        </span>
        {result.receipt_hash && (
          <span className="rounded-full bg-[var(--surface-elevated)] px-3 py-1 font-mono text-xs shadow-[var(--shadow-panel)]">
            Receipt {result.receipt_hash.slice(0, 16)}...
          </span>
        )}
      </div>

      <div className="flex flex-wrap gap-3">
        <Link
          href={shareHref}
          className="rounded-full bg-[var(--acid-green)] px-5 py-2.5 text-sm font-semibold text-white transition-opacity hover:opacity-90"
        >
          View Shareable Result
        </Link>
        <Link
          href={`/try?topic=${encodeURIComponent(result.topic)}`}
          className="rounded-full border border-[var(--border)] px-5 py-2.5 text-sm font-medium text-[var(--text-muted)] transition-colors hover:border-[var(--acid-green)]/50 hover:text-[var(--acid-green)]"
        >
          Ask This in /try
        </Link>
      </div>
    </section>
  );
}

function RecordedSampleCard({ sample }: { sample: RecordedDebate }) {
  return (
    <section className="space-y-5 rounded-[20px] border border-sky-500/18 bg-[var(--surface)] p-6 shadow-[var(--shadow-elevated)] md:p-7">
      <div className="space-y-2">
        <StatusBadge label="Recorded sample" tone="sample" />
        <p className="max-w-2xl text-sm leading-7 text-[var(--text-muted)]">
          This is a captured example for zero-latency browsing. It is illustrative only and is
          never presented as a fresh run.
        </p>
      </div>

      <AgentRoster agents={sample.agents} />
      <ConsensusBar confidence={sample.confidence} />

      <div className="space-y-4 rounded-[18px] border border-[var(--border)] bg-[var(--surface-elevated)] p-6 shadow-[var(--shadow-panel)]">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.22em] text-sky-700">
          Recorded verdict
        </h3>
        <p className="max-w-3xl text-[17px] leading-8 text-[var(--text)] text-pretty">{sample.verdict}</p>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {sample.events.map((event, index) => {
          const accent = accentForAgent(event.agent);
          const badgeColor =
            event.type === 'proposal'
              ? 'text-blue-400'
              : event.type === 'critique'
                ? 'text-red-400'
                : event.type === 'vote'
                  ? 'text-green-400'
                  : 'text-[var(--acid-green)]';

          return (
            <div
              key={`${event.agent}-${index}`}
              className="border p-5 bg-[var(--surface)] space-y-3 rounded-lg shadow-sm"
              style={{ borderColor: `${accent}30` }}
            >
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <span className="text-lg font-bold uppercase tracking-[0.08em]" style={{ color: accent }}>
                    {event.model}
                  </span>
                  <span className={`rounded-full px-2.5 py-1 text-[11px] uppercase tracking-[0.16em] font-semibold ${badgeColor} bg-current/5`}>
                    {event.type}
                  </span>
                </div>
                <div className="rounded-full bg-[var(--surface-elevated)] px-3 py-1 text-sm text-[var(--text-muted)] shadow-[var(--shadow-panel)]">
                  Round {event.round}
                  {event.confidence !== undefined
                    ? ` · ${Math.round(event.confidence * 100)}%`
                    : ''}
                </div>
              </div>
              <p className="max-w-3xl text-[15px] leading-7 text-[var(--text)] text-pretty">
                {event.content}
              </p>
              {event.vote && (
                <div className="text-sm font-semibold text-[var(--acid-green)]">
                  Vote: {event.vote}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="border-t border-[var(--border)] pt-3 text-xs text-[var(--text-muted)]">
        Receipt sample (not cryptographic): <span className="font-mono">{sample.receiptHash}</span>
      </div>
    </section>
  );
}

export default function PublicDemoPage() {
  const [result, setResult] = useState<LiveDebateResult | null>(null);
  const [sampleFallbackMessage, setSampleFallbackMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [progressStep, setProgressStep] = useState(0);
  const [showRecordedSample, setShowRecordedSample] = useState(false);
  const [runStartedAt, setRunStartedAt] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const autoStartedRef = useRef(false);

  const runLiveDemo = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setIsLoading(true);
    setSampleFallbackMessage(null);
    setResult(null);
    setProgressStep(0);

    const startedAt = new Date().toLocaleTimeString();
    setRunStartedAt(startedAt);

    const timers = LIVE_PROGRESS_STEPS.map((_, index) =>
      window.setTimeout(() => setProgressStep(index), index * 1800),
    );
    const clearTimers = () => timers.forEach((timer) => window.clearTimeout(timer));

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/playground/debate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: DEMO_TOPIC,
          question: DEMO_TOPIC,
          rounds: 2,
          agents: 3,
          source: 'demo',
        }),
        signal: controller.signal,
      });

      if (response.status === 429) {
        const data = await response.json().catch(() => null);
        const retryAfter = typeof data?.retry_after === 'number' ? data.retry_after : 60;
        setSampleFallbackMessage(
          `The live proof surface is rate-limited right now, so this page is showing the labeled recorded sample instead. Retry in about ${retryAfter} seconds for a fresh run.`,
        );
        return;
      }

      if (!response.ok) {
        const data = await response.json().catch(() => null);
        const message =
          typeof data?.error === 'string'
            ? data.error
            : `The live proof surface returned HTTP ${response.status}.`;
        setSampleFallbackMessage(
          `${message} Showing the labeled recorded sample instead of a live result.`,
        );
        return;
      }

      const parsed = normalizeLiveDebateResult(await response.json());
      if (!parsed) {
        setSampleFallbackMessage(
          'The live proof surface returned an unexpected payload, so this page is showing the labeled recorded sample instead.',
        );
        return;
      }

      setResult(parsed);
    } catch (fetchError) {
      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        return;
      }
      setSampleFallbackMessage(
        'Could not reach the playground backend for a fresh run, so this page is showing the labeled recorded sample instead.',
      );
    } finally {
      clearTimers();
      setProgressStep(LIVE_PROGRESS_STEPS.length - 1);
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (autoStartedRef.current) {
      return;
    }
    autoStartedRef.current = true;
    void runLiveDemo();

    return () => {
      abortRef.current?.abort();
    };
  }, [runLiveDemo]);

  const resultTone = useMemo(() => {
    if (!result) {
      return null;
    }
    return result.mock_fallback || result.is_live === false ? 'fallback' : 'live';
  }, [result]);

  const recordedSamplePinned =
    sampleFallbackMessage !== null || result?.mock_fallback === true || result?.is_live === false;
  const recordedSampleVisible = showRecordedSample || recordedSamplePinned;

  return (
    <main className="min-h-screen bg-[var(--bg)] text-[var(--text)]">
      <nav className="flex items-center justify-between border-b border-[var(--border)] bg-[var(--surface)]/90 px-4 py-3 backdrop-blur">
        <Link
          href="/landing"
          className="text-sm font-semibold tracking-[0.14em] text-[var(--acid-green)] transition-opacity hover:opacity-80"
        >
          ARAGORA
        </Link>
        <div className="flex items-center gap-3">
          <Link
            href="/try"
            className="rounded-full border border-[var(--border)] px-4 py-2 text-sm font-medium text-[var(--text-muted)] transition-colors hover:border-[var(--acid-green)]/50 hover:text-[var(--acid-green)]"
          >
            /try beta
          </Link>
          <Link
            href="/signup"
            className="rounded-full bg-[var(--acid-green)] px-4 py-2 text-sm font-semibold text-white transition-opacity hover:opacity-90"
          >
            Get started free
          </Link>
        </div>
      </nav>

      <div className="mx-auto flex max-w-6xl flex-col gap-8 px-4 py-8">
        <header className="space-y-3 text-center">
          <h1 className="text-3xl font-bold tracking-tight text-[var(--acid-green)] sm:text-4xl text-balance">
            Live Demo
          </h1>
          <p className="mx-auto max-w-2xl text-base leading-7 text-[var(--text-muted)] text-pretty">
            Watch AI agents debate a real question. Want to ask your own?{' '}
            <Link href="/try/" className="font-semibold text-[var(--acid-green)] hover:underline">Try it free</Link>.
          </p>
        </header>

        <section className="space-y-5 rounded-[22px] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-elevated)] md:p-7">
          <div className="space-y-2">
            <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[var(--acid-green)]">
              Canonical question
            </div>
            <p className="max-w-4xl text-[20px] font-semibold leading-8 text-[var(--text)] text-balance">
              {DEMO_TOPIC}
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => void runLiveDemo()}
              disabled={isLoading}
              className="rounded-full bg-[var(--acid-green)] px-5 py-2.5 text-sm font-semibold text-white transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {isLoading ? 'Running...' : 'Run Live'}
            </button>
            <Link
              href={`/try?topic=${encodeURIComponent(DEMO_TOPIC)}`}
              className="rounded-full border border-[var(--border)] px-5 py-2.5 text-sm font-medium text-[var(--text-muted)] transition-colors hover:border-[var(--acid-green)]/50 hover:text-[var(--acid-green)]"
            >
              Ask Your Own Question
            </Link>
            <button
              onClick={() => setShowRecordedSample((current) => !current)}
              disabled={recordedSamplePinned}
              className="rounded-full border border-[var(--border)] px-5 py-2.5 text-sm font-medium text-[var(--text-muted)] transition-colors hover:border-sky-500/50 hover:text-sky-700"
            >
              {recordedSamplePinned
                ? 'Sample Shown'
                : showRecordedSample
                  ? 'Hide Sample'
                  : 'Show Recorded Sample'}
            </button>
          </div>
        </section>

        {isLoading && (
          <section className="space-y-4 rounded-[20px] border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow-elevated)]">
            <StatusBadge label="Running live proof" tone="live" />
            <div className="space-y-3">
              {LIVE_PROGRESS_STEPS.map((step, index) => (
                <div
                  key={step}
                  className="flex items-center gap-3 text-sm transition-opacity"
                  style={{ opacity: index <= progressStep ? 1 : 0.35 }}
                >
                  <span className="w-2 h-2 rounded-full bg-[var(--acid-green)]" />
                  <span className={index <= progressStep ? 'text-[var(--text)]' : 'text-[var(--text-muted)]'}>
                    {step}
                  </span>
                </div>
              ))}
            </div>
            <p className="text-sm leading-7 text-[var(--text-muted)]">
              This surface only claims a live proof when the backend explicitly returns a live result.
            </p>
          </section>
        )}

        {sampleFallbackMessage && (
          <section className="space-y-4 rounded-[20px] border border-sky-500/20 bg-sky-500/5 p-6 shadow-[var(--shadow-elevated)]">
            <StatusBadge label="Showing recorded sample" tone="sample" />
            <p className="max-w-3xl text-sm leading-7 text-sky-900">
              {sampleFallbackMessage}
            </p>
            <div className="flex flex-wrap gap-3">
              <button
                onClick={() => void runLiveDemo()}
                disabled={isLoading}
                className="rounded-full bg-sky-700 px-5 py-2.5 text-sm font-semibold text-white transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                Retry Live Run
              </button>
              <Link
                href="/try"
                className="rounded-full border border-sky-500/30 px-5 py-2.5 text-sm font-medium text-sky-800 transition-colors hover:bg-sky-500/8"
              >
                Open /try Instead
              </Link>
            </div>
          </section>
        )}

        {result && !isLoading && (
          <section className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--text-muted)]">
                {resultTone === 'live' ? 'Fresh response' : 'Fallback response'}
              </div>
              {resultTone === 'live' ? (
                <span className="rounded-full bg-[var(--surface)] px-3 py-1 text-sm font-medium text-[var(--acid-green)] shadow-[var(--shadow-panel)]">
                  Backend returned a live debate
                </span>
              ) : (
                <span className="rounded-full bg-amber-50 px-3 py-1 text-sm font-medium text-amber-700 shadow-[var(--shadow-panel)]">
                  Backend did not return a fresh live debate
                </span>
              )}
            </div>
            <LiveResultCard result={result} runStartedAt={runStartedAt} />
          </section>
        )}

        {recordedSampleVisible && (
          <section className="space-y-3">
            <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-sky-700">
              {sampleFallbackMessage
                ? 'Recorded fallback currently shown'
                : 'Clearly labeled canned example'}
            </div>
            <RecordedSampleCard sample={RECORDED_SAMPLE} />
          </section>
        )}
      </div>
    </main>
  );
}
