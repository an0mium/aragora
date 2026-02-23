'use client';

import { useState, useEffect, useRef, useCallback } from 'react';

// ---------------------------------------------------------------------------
// Mock debate data
// ---------------------------------------------------------------------------

const TOPIC = 'Should we use microservices or a monolith for our new product?';

interface AgentEntry {
  name: string;
  role: string;
  color: string;
  proposal: string;
}

const AGENTS: AgentEntry[] = [
  {
    name: 'claude-analyst',
    role: 'Analyst',
    color: 'var(--acid-cyan)',
    proposal:
      'Start with a modular monolith. You get the deployment simplicity of a single artifact, ' +
      'but enforce module boundaries at the code level. When you have clear scaling bottlenecks ' +
      'after 6-12 months, extract only those modules into services. This avoids premature ' +
      'distributed-systems complexity (network partitions, eventual consistency, service discovery) ' +
      'while preserving the option to decompose later.',
  },
  {
    name: 'gpt-contrarian',
    role: 'Contrarian',
    color: 'var(--acid-magenta)',
    proposal:
      'Go microservices from day one. The "extract later" argument sounds reasonable but almost ' +
      'never happens in practice -- monolith boundaries rot within months, and the migration cost ' +
      'grows exponentially. If you design service contracts up front, your team can deploy and ' +
      'scale independently. The operational overhead of Kubernetes is a one-time cost; the ' +
      'organizational agility is permanent.',
  },
  {
    name: 'gemini-synthesizer',
    role: 'Synthesizer',
    color: 'var(--acid-green)',
    proposal:
      'Both positions contain valid tradeoffs. The key variable is team size: below 15 engineers, ' +
      'a modular monolith reduces coordination overhead. Above 25, independent deployability ' +
      'becomes critical. My recommendation: start with the modular monolith but invest in CI/CD ' +
      'pipelines and API contracts that make future extraction low-friction. Set a 6-month ' +
      'checkpoint to evaluate scaling pressure.',
  },
];

interface CritiqueEntry {
  from: string;
  to: string;
  fromColor: string;
  text: string;
  severity: number;
}

const CRITIQUES: CritiqueEntry[] = [
  {
    from: 'gpt-contrarian',
    to: 'claude-analyst',
    fromColor: 'var(--acid-magenta)',
    text: 'The "extract later" pattern has a <20% success rate in orgs with >50 engineers. Your proposal lacks a concrete trigger for decomposition.',
    severity: 7.2,
  },
  {
    from: 'claude-analyst',
    to: 'gpt-contrarian',
    fromColor: 'var(--acid-cyan)',
    text: 'You assume Kubernetes operational cost is "one-time" but ignore ongoing incident management, service mesh maintenance, and distributed tracing overhead for a small team.',
    severity: 6.8,
  },
  {
    from: 'gemini-synthesizer',
    to: 'gpt-contrarian',
    fromColor: 'var(--acid-green)',
    text: 'Team size is the missing variable. Your proposal optimizes for scale that may never materialize, violating YAGNI.',
    severity: 5.5,
  },
];

interface VoteEntry {
  agent: string;
  color: string;
  choice: string;
  confidence: number;
}

const VOTES: VoteEntry[] = [
  { agent: 'claude-analyst', color: 'var(--acid-cyan)', choice: 'gemini-synthesizer', confidence: 0.82 },
  { agent: 'gpt-contrarian', color: 'var(--acid-magenta)', choice: 'gemini-synthesizer', confidence: 0.61 },
  { agent: 'gemini-synthesizer', color: 'var(--acid-green)', choice: 'gemini-synthesizer', confidence: 0.91 },
];

const RECEIPT = {
  receipt_id: 'rcpt_pg_demo_4f8a1c',
  verdict: 'CONSENSUS REACHED',
  confidence: 0.87,
  method: 'weighted_majority',
  rounds: 2,
  timestamp: new Date().toISOString().split('T')[0],
  hash: 'sha256:e3b0c44298fc1c14...a495991b7852b855',
};

// ---------------------------------------------------------------------------
// Typing animation hook
// ---------------------------------------------------------------------------

function useTypewriter(text: string, active: boolean, speed = 12): string {
  const [displayed, setDisplayed] = useState('');
  const indexRef = useRef(0);

  useEffect(() => {
    if (!active) return;
    setDisplayed('');
    indexRef.current = 0;

    const interval = setInterval(() => {
      indexRef.current += 1;
      if (indexRef.current >= text.length) {
        setDisplayed(text);
        clearInterval(interval);
      } else {
        setDisplayed(text.slice(0, indexRef.current));
      }
    }, speed);

    return () => clearInterval(interval);
  }, [text, active, speed]);

  return active ? displayed : '';
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Cursor({ visible }: { visible: boolean }) {
  if (!visible) return null;
  return <span className="animate-pulse text-[var(--acid-green)]">_</span>;
}

function PhaseLabel({ label, active }: { label: string; active: boolean }) {
  return (
    <div className={`flex items-center gap-2 mb-4 ${active ? 'opacity-100' : 'opacity-40'} transition-opacity duration-500`}>
      <span className="text-xs font-mono text-[var(--acid-green)]">{'>>'}</span>
      <span className="text-sm font-mono font-bold text-[var(--acid-green)] uppercase tracking-wider">
        {label}
      </span>
      {active && (
        <span className="ml-2 w-2 h-2 rounded-full bg-[var(--acid-green)] animate-pulse" />
      )}
    </div>
  );
}

function AgentProposal({
  agent,
  text,
  typing,
}: {
  agent: AgentEntry;
  text: string;
  typing: boolean;
}) {
  return (
    <div className="border border-[var(--border)] bg-[var(--surface)]/30 p-4 mb-3">
      <div className="flex items-center gap-2 mb-2">
        <span
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: agent.color }}
        />
        <span className="text-xs font-mono font-bold" style={{ color: agent.color }}>
          {agent.name}
        </span>
        <span className="text-xs font-mono text-[var(--text-muted)]">
          [{agent.role}]
        </span>
      </div>
      <p className="text-xs font-mono text-[var(--text)] leading-relaxed whitespace-pre-wrap">
        {text}
        <Cursor visible={typing && text.length > 0 && text.length < agent.proposal.length} />
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

type Phase = 'idle' | 'topic' | 'proposals' | 'critiques' | 'votes' | 'receipt';

const PHASE_ORDER: Phase[] = ['topic', 'proposals', 'critiques', 'votes', 'receipt'];

export function PlaygroundDebate() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [proposalIndex, setProposalIndex] = useState(-1);
  const [critiqueIndex, setCritiqueIndex] = useState(-1);
  const [voteIndex, setVoteIndex] = useState(-1);
  const [showReceipt, setShowReceipt] = useState(false);
  const [started, setStarted] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom as content appears
  const scrollToBottom = useCallback(() => {
    if (containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, []);

  // Typing hooks for each proposal
  const p0 = useTypewriter(AGENTS[0].proposal, proposalIndex >= 0, 8);
  const p1 = useTypewriter(AGENTS[1].proposal, proposalIndex >= 1, 8);
  const p2 = useTypewriter(AGENTS[2].proposal, proposalIndex >= 2, 8);
  const proposals = [p0, p1, p2];

  // Orchestrate the demo sequence
  useEffect(() => {
    if (!started) return;

    const timers: ReturnType<typeof setTimeout>[] = [];
    let t = 0;

    // Topic reveal
    timers.push(setTimeout(() => setPhase('topic'), (t += 300)));

    // Proposals phase
    timers.push(setTimeout(() => { setPhase('proposals'); setProposalIndex(0); }, (t += 1200)));
    timers.push(setTimeout(() => setProposalIndex(1), (t += 3500)));
    timers.push(setTimeout(() => setProposalIndex(2), (t += 3500)));

    // Critiques phase
    timers.push(setTimeout(() => { setPhase('critiques'); setCritiqueIndex(0); }, (t += 4000)));
    timers.push(setTimeout(() => setCritiqueIndex(1), (t += 1500)));
    timers.push(setTimeout(() => setCritiqueIndex(2), (t += 1500)));

    // Votes phase
    timers.push(setTimeout(() => { setPhase('votes'); setVoteIndex(0); }, (t += 2000)));
    timers.push(setTimeout(() => setVoteIndex(1), (t += 600)));
    timers.push(setTimeout(() => setVoteIndex(2), (t += 600)));

    // Receipt phase
    timers.push(setTimeout(() => { setPhase('receipt'); setShowReceipt(true); }, (t += 1500)));

    return () => timers.forEach(clearTimeout);
  }, [started]);

  // Scroll on phase/content changes
  useEffect(() => { scrollToBottom(); }, [phase, proposalIndex, critiqueIndex, voteIndex, showReceipt, p0, p1, p2, scrollToBottom]);

  const phaseIdx = PHASE_ORDER.indexOf(phase);

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Terminal header */}
      <div className="border border-[var(--acid-green)]/40 bg-[var(--surface)]/50">
        <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--acid-green)]/20 bg-[var(--bg)]">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full border border-[var(--acid-green)]/40" />
            <span className="w-3 h-3 rounded-full border border-[var(--acid-green)]/40" />
            <span className="w-3 h-3 rounded-full border border-[var(--acid-green)]/40" />
          </div>
          <span className="text-xs font-mono text-[var(--text-muted)]">
            aragora://playground/demo
          </span>
          <span className="text-xs font-mono text-[var(--acid-green)]">LIVE</span>
        </div>

        <div ref={containerRef} className="p-6 max-h-[600px] overflow-y-auto space-y-4">
          {/* Start state */}
          {!started && (
            <div className="text-center py-12 space-y-6">
              <div className="space-y-2">
                <p className="text-sm font-mono text-[var(--text-muted)]">
                  {'>'} DEMO: Watch 3 AI agents debate a real architecture decision
                </p>
                <p className="text-xs font-mono text-[var(--text-muted)]/60">
                  Proposals, critiques, votes, and a decision receipt -- all in real time
                </p>
              </div>
              <button
                onClick={() => setStarted(true)}
                className="px-8 py-3 font-mono text-sm font-bold bg-[var(--acid-green)] text-[var(--bg)] hover:bg-[var(--acid-green)]/80 transition-colors"
              >
                START DEBATE
              </button>
            </div>
          )}

          {/* Topic */}
          {started && phaseIdx >= 0 && (
            <div className="border-l-2 border-[var(--acid-green)] pl-4">
              <span className="text-xs font-mono text-[var(--text-muted)]">TOPIC</span>
              <p className="text-sm font-mono text-[var(--text)] mt-1 font-bold">
                {TOPIC}
              </p>
              <div className="flex items-center gap-3 mt-2 text-xs font-mono text-[var(--text-muted)]">
                <span>3 agents</span>
                <span>|</span>
                <span>2 rounds</span>
                <span>|</span>
                <span>weighted majority</span>
              </div>
            </div>
          )}

          {/* Proposals */}
          {started && phaseIdx >= 1 && (
            <div>
              <PhaseLabel label="Round 1: Proposals" active={phase === 'proposals'} />
              {AGENTS.map((agent, i) => {
                if (proposalIndex < i) return null;
                return (
                  <AgentProposal
                    key={agent.name}
                    agent={agent}
                    text={proposals[i]}
                    typing={proposalIndex === i}
                  />
                );
              })}
            </div>
          )}

          {/* Critiques */}
          {started && phaseIdx >= 2 && (
            <div>
              <PhaseLabel label="Round 2: Critiques" active={phase === 'critiques'} />
              {CRITIQUES.map((critique, i) => {
                if (critiqueIndex < i) return null;
                return (
                  <div
                    key={i}
                    className="border-l-2 border-[var(--border)] pl-3 mb-3 animate-in fade-in duration-300"
                  >
                    <div className="text-xs font-mono mb-1">
                      <span style={{ color: critique.fromColor }}>{critique.from}</span>
                      <span className="text-[var(--text-muted)]">{' -> '}</span>
                      <span className="text-[var(--text-muted)]">{critique.to}</span>
                      <span className="text-[var(--crimson,#ef4444)] ml-2">
                        severity {critique.severity}/10
                      </span>
                    </div>
                    <p className="text-xs font-mono text-[var(--text-muted)] leading-relaxed">
                      {critique.text}
                    </p>
                  </div>
                );
              })}
            </div>
          )}

          {/* Votes */}
          {started && phaseIdx >= 3 && (
            <div>
              <PhaseLabel label="Voting" active={phase === 'votes'} />
              <div className="space-y-2">
                {VOTES.map((vote, i) => {
                  if (voteIndex < i) return null;
                  return (
                    <div key={i} className="flex items-center gap-2 text-xs font-mono">
                      <span style={{ color: vote.color }}>{vote.agent}</span>
                      <span className="text-[var(--text-muted)]">voted for</span>
                      <span className="text-[var(--acid-green)] font-bold">{vote.choice}</span>
                      <span className="text-[var(--text-muted)]">
                        ({Math.round(vote.confidence * 100)}%)
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Receipt */}
          {started && showReceipt && (
            <div>
              <PhaseLabel label="Decision Receipt" active={phase === 'receipt'} />
              <div className="border border-[var(--acid-green)]/30 bg-[var(--bg)] p-4">
                <div className="flex items-center justify-between mb-4">
                  <span className="px-3 py-1 text-sm font-mono font-bold bg-[var(--acid-green)]/20 text-[var(--acid-green)] border border-[var(--acid-green)]/30">
                    {RECEIPT.verdict}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-[var(--text-muted)]">CONFIDENCE</span>
                    <div className="w-24 h-2 bg-[var(--bg)] border border-[var(--acid-green)]/20 overflow-hidden">
                      <div
                        className="h-full bg-[var(--acid-green)] transition-all duration-1000"
                        style={{ width: `${Math.round(RECEIPT.confidence * 100)}%` }}
                      />
                    </div>
                    <span className="text-xs font-mono text-[var(--acid-green)]">
                      {Math.round(RECEIPT.confidence * 100)}%
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  <div>
                    <span className="text-[var(--text-muted)]">Receipt ID: </span>
                    <span className="text-[var(--acid-cyan)]">{RECEIPT.receipt_id}</span>
                  </div>
                  <div>
                    <span className="text-[var(--text-muted)]">Method: </span>
                    <span className="text-[var(--text)]">{RECEIPT.method}</span>
                  </div>
                  <div>
                    <span className="text-[var(--text-muted)]">Rounds: </span>
                    <span className="text-[var(--text)]">{RECEIPT.rounds}</span>
                  </div>
                  <div>
                    <span className="text-[var(--text-muted)]">Date: </span>
                    <span className="text-[var(--text)]">{RECEIPT.timestamp}</span>
                  </div>
                  <div className="col-span-2">
                    <span className="text-[var(--text-muted)]">Hash: </span>
                    <span className="text-[var(--text-muted)]/70">{RECEIPT.hash}</span>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-[var(--acid-green)]/20">
                  <p className="text-xs font-mono text-[var(--text)] leading-relaxed">
                    <span className="text-[var(--acid-green)] font-bold">Verdict: </span>
                    Start with a modular monolith and invest in API contracts for future extraction.
                    Set a 6-month checkpoint keyed to team size ({'>'} 25 engineers) and
                    scaling pressure as the decomposition trigger.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
