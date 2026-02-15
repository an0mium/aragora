'use client';

import Link from 'next/link';

// ---------------------------------------------------------------------------
// Types (shared with Playground.tsx)
// ---------------------------------------------------------------------------

interface CritiqueResult {
  agent: string;
  target_agent: string;
  issues: string[];
  suggestions: string[];
  severity: number;
}

interface VoteResult {
  agent: string;
  choice: string;
  confidence: number;
  reasoning: string;
}

interface ReceiptResult {
  receipt_id: string;
  question: string;
  verdict: string;
  confidence: number;
  consensus: {
    reached: boolean;
    method: string;
    confidence: number;
    supporting_agents: string[];
    dissenting_agents: string[];
  };
  agents: string[];
  rounds_used: number;
  timestamp: string;
  signature: string | null;
  signature_algorithm: string | null;
}

export interface DebateResponse {
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
  critiques: CritiqueResult[];
  votes: VoteResult[];
  dissenting_views: string[];
  final_answer: string;
  receipt: ReceiptResult | null;
  receipt_hash: string | null;
}

// ---------------------------------------------------------------------------
// Agent color mapping
// ---------------------------------------------------------------------------

const AGENT_COLORS: Record<string, string> = {
  analyst: 'text-[var(--acid-cyan)]',
  critic: 'text-[var(--crimson)]',
  moderator: 'text-[var(--acid-green)]',
  contrarian: 'text-[var(--acid-yellow)]',
  synthesizer: 'text-[var(--acid-magenta)]',
};

function agentColor(name: string): string {
  return AGENT_COLORS[name] || 'text-[var(--acid-cyan)]';
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface DebateResultPreviewProps {
  result: DebateResponse;
}

export function DebateResultPreview({ result }: DebateResultPreviewProps) {
  const handleSignup = () => {
    sessionStorage.setItem('aragora_pending_debate', JSON.stringify(result));
  };

  return (
    <div className="text-left space-y-4 mt-8">
      {/* Summary bar */}
      <div className="border border-[var(--border)] p-4 flex flex-wrap gap-4 items-center text-sm font-mono">
        <span
          className={
            result.consensus_reached
              ? 'text-[var(--acid-green)]'
              : 'text-[var(--warning)]'
          }
        >
          {result.consensus_reached ? 'Consensus Reached' : 'No Consensus'}
        </span>
        <span className="text-[var(--text-muted)]">|</span>
        <span className="text-[var(--text-muted)]">
          Confidence: {(result.confidence * 100).toFixed(0)}%
        </span>
        <span className="text-[var(--text-muted)]">|</span>
        <span className="text-[var(--text-muted)]">
          {result.rounds_used} round{result.rounds_used !== 1 ? 's' : ''}
        </span>
        <span className="text-[var(--text-muted)]">|</span>
        <span className="text-[var(--text-muted)]">
          {result.duration_seconds}s
        </span>
      </div>

      {/* Proposals */}
      <div className="border border-[var(--border)] p-4">
        <h3 className="text-sm text-[var(--acid-green)] mb-4 font-bold font-mono">
          Proposals
        </h3>
        <div className="space-y-4">
          {Object.entries(result.proposals).map(([agent, content]) => (
            <div key={agent}>
              <h4 className={`text-sm font-bold mb-1 font-mono ${agentColor(agent)}`}>
                {agent}
              </h4>
              <p className="text-xs text-[var(--text-muted)] whitespace-pre-wrap leading-relaxed">
                {content}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Critiques (first 3) */}
      {result.critiques.length > 0 && (
        <div className="border border-[var(--border)] p-4">
          <h3 className="text-sm text-[var(--acid-green)] mb-4 font-bold font-mono">
            Critiques
          </h3>
          <div className="space-y-3">
            {result.critiques.slice(0, 3).map((c, i) => (
              <div key={i} className="border-l-2 border-[var(--border)] pl-3">
                <div className="text-xs mb-1 font-mono">
                  <span className={agentColor(c.agent)}>{c.agent}</span>
                  <span className="text-[var(--text-muted)]"> on </span>
                  <span className={agentColor(c.target_agent)}>
                    {c.target_agent}
                  </span>
                  <span className="text-[var(--text-muted)] ml-2">
                    severity {c.severity.toFixed(1)}/10
                  </span>
                </div>
                <ul className="text-xs text-[var(--text-muted)] space-y-0.5">
                  {c.issues.map((issue, j) => (
                    <li key={j} className="flex items-start gap-1">
                      <span className="text-[var(--crimson)]">-</span>
                      {issue}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Votes */}
      {result.votes.length > 0 && (
        <div className="border border-[var(--border)] p-4">
          <h3 className="text-sm text-[var(--acid-green)] mb-4 font-bold font-mono">
            Votes
          </h3>
          <div className="space-y-2">
            {result.votes.map((v, i) => (
              <div key={i} className="text-xs flex items-center gap-2 font-mono">
                <span className={agentColor(v.agent)}>{v.agent}</span>
                <span className="text-[var(--text-muted)]">voted for</span>
                <span className={`font-bold ${agentColor(v.choice)}`}>
                  {v.choice}
                </span>
                <span className="text-[var(--text-muted)]">
                  ({(v.confidence * 100).toFixed(0)}%)
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gated Receipt */}
      <div className="relative">
        <div className="blur-sm pointer-events-none opacity-60">
          <div className="border border-[var(--acid-green)]/30 p-4">
            <h3 className="text-sm text-[var(--acid-green)] mb-3 font-bold font-mono">
              Decision Receipt
            </h3>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div>
                <span className="text-[var(--text-muted)]">Receipt ID: </span>
                <span className="text-[var(--acid-cyan)]">
                  {result.receipt?.receipt_id || 'rcpt_...'}
                </span>
              </div>
              <div>
                <span className="text-[var(--text-muted)]">Verdict: </span>
                <span className="text-[var(--acid-green)]">
                  {result.receipt?.verdict?.replace(/_/g, ' ') || 'consensus_reached'}
                </span>
              </div>
              <div>
                <span className="text-[var(--text-muted)]">Hash: </span>
                <span>sha256:a3b9c...</span>
              </div>
              <div>
                <span className="text-[var(--text-muted)]">Timestamp: </span>
                <span>{result.receipt?.timestamp || new Date().toISOString()}</span>
              </div>
            </div>
          </div>
        </div>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="bg-[var(--surface)] border border-[var(--border)] p-6 text-center shadow-lg">
            <p className="font-mono text-sm text-[var(--text)] mb-4">
              Sign up to download the full decision receipt
            </p>
            <div className="flex gap-3 justify-center">
              <Link
                href="/auth/register"
                onClick={handleSignup}
                className="font-mono text-sm px-4 py-2 bg-[var(--acid-green)] text-[var(--bg)] font-bold hover:opacity-90 transition-opacity"
              >
                Sign Up Free
              </Link>
              <Link
                href="/auth/login"
                className="font-mono text-sm px-4 py-2 border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--acid-green)] hover:text-[var(--acid-green)] transition-colors"
              >
                Log In
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
