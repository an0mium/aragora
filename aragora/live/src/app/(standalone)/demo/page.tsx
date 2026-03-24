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
  '#00ff41',
  '#63b3ed',
  '#f6ad55',
  '#fc8181',
  '#b794f6',
  '#68d391',
];

function accentForAgent(agent: string): string {
  let hash = 0;
  for (const char of agent) {
    hash = (hash + char.charCodeAt(0)) % AGENT_ACCENTS.length;
  }
  return AGENT_ACCENTS[hash];
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
    fallback: 'border-amber-400/40 bg-amber-400/10 text-amber-300',
    sample: 'border-blue-400/40 bg-blue-400/10 text-blue-300',
  }[tone];

  return (
    <span className={`inline-flex items-center px-2 py-1 text-[10px] font-mono uppercase tracking-[0.2em] border ${styles}`}>
      {label}
    </span>
  );
}

function AgentRoster({ agents }: { agents: string[] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {agents.map((agent) => {
        const accent = accentForAgent(agent);
        return (
          <div
            key={agent}
            className="px-2 py-1 border text-xs font-mono"
            style={{ borderColor: `${accent}55`, color: accent }}
          >
            {agent}
          </div>
        );
      })}
    </div>
  );
}

function ConsensusBar({ confidence }: { confidence: number }) {
  const clamped = Math.max(0, Math.min(confidence, 1));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs font-mono text-[var(--text-muted)]">
        <span>Consensus confidence</span>
        <span className="text-[var(--acid-green)]">{Math.round(clamped * 100)}%</span>
      </div>
      <div className="h-2 bg-[var(--surface)] border border-[var(--border)] overflow-hidden">
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
    <section className="border border-[var(--acid-green)]/30 bg-[var(--surface)]/40 p-5 space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="space-y-2">
          <StatusBadge label={resultLabel} tone={resultTone} />
          <div className="text-xs font-mono text-[var(--text-muted)]">
            {resultTone === 'live'
              ? 'Fresh result from the public playground backend.'
              : `The backend returned a non-live fallback${result.mock_fallback_reason ? `: ${result.mock_fallback_reason}` : '.'}`}
          </div>
        </div>
        <div className="text-right text-xs font-mono text-[var(--text-muted)] space-y-1">
          <div>ID: {result.id}</div>
          <div>{result.duration_seconds.toFixed(1)}s runtime</div>
          {runStartedAt && <div>Started {runStartedAt}</div>}
        </div>
      </div>

      <div className="space-y-3">
        <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-[var(--acid-green)]">
          Returned agents
        </div>
        <AgentRoster agents={result.participants} />
      </div>

      <ConsensusBar confidence={result.confidence} />

      <div className="space-y-2">
        <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-[var(--acid-green)]">
          Verdict
        </div>
        <p className="text-sm font-mono text-[var(--text)] leading-relaxed">{summary}</p>
      </div>

      {proposalEntries.length > 0 && (
        <div className="space-y-3">
          <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-[var(--acid-green)]">
            Agent positions
          </div>
          <div className="grid grid-cols-1 gap-3">
            {proposalEntries.map(([agent, proposal]) => {
              const accent = accentForAgent(agent);
              return (
                <div
                  key={agent}
                  className="border p-3 bg-[var(--bg)]/40"
                  style={{ borderColor: `${accent}55` }}
                >
                  <div className="mb-2 text-xs font-mono" style={{ color: accent }}>
                    {agent}
                  </div>
                  <p className="text-xs font-mono text-[var(--text-muted)] leading-relaxed">
                    {proposal}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3 text-xs font-mono text-[var(--text-muted)]">
        <span>Rounds: {result.rounds_used}</span>
        <span>Status: {result.status}</span>
        {result.receipt_hash && <span>Receipt: {result.receipt_hash.slice(0, 16)}...</span>}
      </div>

      <div className="flex flex-wrap gap-3">
        <Link
          href={shareHref}
          className="px-4 py-2 text-xs font-mono bg-[var(--acid-green)] text-[var(--bg)] hover:opacity-90 transition-opacity"
        >
          VIEW SHAREABLE RESULT
        </Link>
        <Link
          href={`/try?topic=${encodeURIComponent(result.topic)}`}
          className="px-4 py-2 text-xs font-mono border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--acid-green)] hover:border-[var(--acid-green)]/50 transition-colors"
        >
          TAKE YOUR OWN QUESTION TO /TRY
        </Link>
      </div>
    </section>
  );
}

function RecordedSampleCard({ sample }: { sample: RecordedDebate }) {
  return (
    <section className="border border-blue-400/30 bg-blue-400/5 p-5 space-y-5">
      <div className="space-y-2">
        <StatusBadge label="Recorded sample" tone="sample" />
        <p className="text-sm font-mono text-[var(--text-muted)] leading-relaxed">
          This is a captured example for zero-latency browsing. It is illustrative only and is
          never presented as a fresh run.
        </p>
      </div>

      <AgentRoster agents={sample.agents} />
      <ConsensusBar confidence={sample.confidence} />

      <div className="space-y-2">
        <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-blue-300">
          Recorded verdict
        </div>
        <p className="text-sm font-mono text-[var(--text)] leading-relaxed">{sample.verdict}</p>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {sample.events.map((event, index) => {
          const accent = accentForAgent(event.agent);
          const badgeColor =
            event.type === 'proposal'
              ? 'text-blue-300'
              : event.type === 'critique'
                ? 'text-red-300'
                : event.type === 'vote'
                  ? 'text-green-300'
                  : 'text-[var(--acid-green)]';

          return (
            <div
              key={`${event.agent}-${index}`}
              className="border p-3 bg-[var(--bg)]/30 space-y-2"
              style={{ borderColor: `${accent}55` }}
            >
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono" style={{ color: accent }}>
                    {event.model}
                  </span>
                  <span className={`text-[10px] font-mono uppercase ${badgeColor}`}>
                    {event.type}
                  </span>
                </div>
                <div className="text-[10px] font-mono text-[var(--text-muted)]">
                  Round {event.round}
                  {event.confidence !== undefined
                    ? ` | ${Math.round(event.confidence * 100)}% confidence`
                    : ''}
                </div>
              </div>
              <p className="text-xs font-mono text-[var(--text-muted)] leading-relaxed">
                {event.content}
              </p>
              {event.vote && (
                <div className="text-[10px] font-mono text-[var(--text-muted)]">
                  Vote: {event.vote}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="text-xs font-mono text-[var(--text-muted)]">
        Receipt sample (not cryptographic): {sample.receiptHash}
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
      <nav className="border-b border-[var(--border)] px-4 py-3 flex items-center justify-between">
        <Link
          href="/landing"
          className="font-mono text-sm text-[var(--acid-green)] hover:opacity-80 transition-opacity"
        >
          ARAGORA
        </Link>
        <div className="flex items-center gap-3">
          <Link
            href="/try"
            className="px-4 py-1.5 text-xs font-mono border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--acid-green)] hover:border-[var(--acid-green)]/50 transition-colors"
          >
            /TRY BETA
          </Link>
          <Link
            href="/signup"
            className="px-4 py-1.5 text-xs font-mono bg-[var(--acid-green)] text-[var(--bg)] hover:opacity-90 transition-opacity"
          >
            GET STARTED FREE
          </Link>
        </div>
      </nav>

      <div className="container mx-auto px-4 py-8 max-w-5xl space-y-8">
        <header className="text-center space-y-4">
          <div className="text-[10px] font-mono text-[var(--acid-green)] uppercase tracking-[0.3em]">
            Truthful Public Demo
          </div>
          <h1 className="text-3xl sm:text-4xl font-mono text-[var(--acid-green)]">
            LIVE PROOF SURFACE
          </h1>
          <p className="text-sm font-mono text-[var(--text-muted)] max-w-3xl mx-auto leading-relaxed">
            This page runs one canonical question against the same public playground backend used
            elsewhere. If the backend returns a simulated fallback instead of a fresh live run, it
            is labeled as such. If the backend is unavailable, the recorded sample appears instead
            of an error. If you want to bring your own question, use <span className="text-[var(--acid-green)]">/try</span>.
          </p>
        </header>

        <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-[var(--surface)] border border-[var(--acid-green)]/30 space-y-2">
            <StatusBadge label="/demo" tone="live" />
            <p className="text-sm font-mono text-[var(--text)]">Canonical live-backed proof</p>
            <p className="text-xs font-mono text-[var(--text-muted)]">
              Fresh run of one public question. Honest fallback labeling if the backend is not live.
            </p>
          </div>
          <div className="p-4 bg-[var(--surface)] border border-[var(--border)] space-y-2">
            <StatusBadge label="/try" tone="live" />
            <p className="text-sm font-mono text-[var(--text)]">Primary beta funnel</p>
            <p className="text-xs font-mono text-[var(--text-muted)]">
              Ask your own question. Keep the existing rate limits, persistence, and share flow.
            </p>
          </div>
          <div className="p-4 bg-[var(--surface)] border border-blue-400/30 space-y-2">
            <StatusBadge label="Recorded sample" tone="sample" />
            <p className="text-sm font-mono text-[var(--text)]">Canned example, clearly labeled</p>
            <p className="text-xs font-mono text-[var(--text-muted)]">
              Available for comparison and fast browsing, but never presented as a live proof.
            </p>
          </div>
        </section>

        <section className="p-4 bg-[var(--surface)] border border-[var(--acid-green)]/30 space-y-3">
          <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-[var(--acid-green)]">
            Canonical question for this surface
          </div>
          <p className="text-sm font-mono text-[var(--text)] leading-relaxed">{DEMO_TOPIC}</p>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => void runLiveDemo()}
              disabled={isLoading}
              className="px-4 py-2 text-xs font-mono bg-[var(--acid-green)] text-[var(--bg)] hover:opacity-90 disabled:opacity-50 transition-opacity"
            >
              {isLoading ? 'RUNNING LIVE PROOF...' : 'RUN LIVE AGAIN'}
            </button>
            <Link
              href={`/try?topic=${encodeURIComponent(DEMO_TOPIC)}`}
              className="px-4 py-2 text-xs font-mono border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--acid-green)] hover:border-[var(--acid-green)]/50 transition-colors"
            >
              TAKE THIS TO /TRY
            </Link>
            <button
              onClick={() => setShowRecordedSample((current) => !current)}
              disabled={recordedSamplePinned}
              className="px-4 py-2 text-xs font-mono border border-blue-400/40 text-blue-300 hover:bg-blue-400/10 transition-colors"
            >
              {recordedSamplePinned
                ? 'RECORDED SAMPLE SHOWN'
                : showRecordedSample
                  ? 'HIDE RECORDED SAMPLE'
                  : 'SHOW RECORDED SAMPLE'}
            </button>
          </div>
        </section>

        {isLoading && (
          <section className="border border-[var(--acid-green)]/30 bg-[var(--surface)]/40 p-5 space-y-4">
            <StatusBadge label="Running live proof" tone="live" />
            <div className="space-y-3">
              {LIVE_PROGRESS_STEPS.map((step, index) => (
                <div
                  key={step}
                  className="flex items-center gap-3 text-sm font-mono transition-opacity"
                  style={{ opacity: index <= progressStep ? 1 : 0.35 }}
                >
                  <span className="w-2 h-2 rounded-full bg-[var(--acid-green)]" />
                  <span className={index <= progressStep ? 'text-[var(--text)]' : 'text-[var(--text-muted)]'}>
                    {step}
                  </span>
                </div>
              ))}
            </div>
            <p className="text-xs font-mono text-[var(--text-muted)]">
              This surface only claims a live proof when the backend explicitly returns a live result.
            </p>
          </section>
        )}

        {sampleFallbackMessage && (
          <section className="border border-blue-400/40 bg-blue-400/10 p-5 space-y-3">
            <StatusBadge label="Showing recorded sample" tone="sample" />
            <p className="text-sm font-mono text-blue-100 leading-relaxed">
              {sampleFallbackMessage}
            </p>
            <div className="flex flex-wrap gap-3">
              <button
                onClick={() => void runLiveDemo()}
                disabled={isLoading}
                className="px-4 py-2 text-xs font-mono bg-blue-300 text-[var(--bg)] hover:opacity-90 disabled:opacity-50 transition-opacity"
              >
                RETRY LIVE RUN
              </button>
              <Link
                href="/try"
                className="px-4 py-2 text-xs font-mono border border-blue-300/50 text-blue-100 hover:bg-blue-300/10 transition-colors"
              >
                OPEN /TRY INSTEAD
              </Link>
            </div>
          </section>
        )}

        {result && !isLoading && (
          <section className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-[var(--text-muted)]">
                {resultTone === 'live' ? 'Fresh response' : 'Fallback response'}
              </div>
              {resultTone === 'live' ? (
                <span className="text-xs font-mono text-[var(--acid-green)]">
                  Backend returned a live debate
                </span>
              ) : (
                <span className="text-xs font-mono text-amber-300">
                  Backend did not return a fresh live debate
                </span>
              )}
            </div>
            <LiveResultCard result={result} runStartedAt={runStartedAt} />
          </section>
        )}

        {recordedSampleVisible && (
          <section className="space-y-3">
            <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-blue-300">
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
