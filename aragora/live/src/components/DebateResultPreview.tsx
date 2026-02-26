'use client';

import { useState } from 'react';
import Link from 'next/link';
import { RETURN_URL_STORAGE_KEY } from '@/utils/returnUrl';
import { PerspectiveCard } from './PerspectiveCard';

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

const VERDICT_COLORS: Record<string, string> = {
  approved: 'bg-[var(--acid-green)] text-[var(--bg)]',
  approved_with_conditions: 'bg-[var(--gold,#ffd700)] text-[var(--bg)]',
  needs_review: 'bg-[var(--warning,#ff8c00)] text-[var(--bg)]',
  rejected: 'bg-[var(--crimson)] text-white',
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface DebateResultPreviewProps {
  result: DebateResponse;
}

export const RETURN_URL_KEY = RETURN_URL_STORAGE_KEY;
export const PENDING_DEBATE_KEY = 'aragora_pending_debate';

export function DebateResultPreview({ result }: DebateResultPreviewProps) {
  const [showDetails, setShowDetails] = useState(false);

  const saveDebateAndReturnUrl = () => {
    // Save debate results so the landing page can restore them after login
    sessionStorage.setItem(PENDING_DEBATE_KEY, JSON.stringify(result));
    // Save return URL so OAuth callback redirects back here
    sessionStorage.setItem(RETURN_URL_KEY, window.location.pathname);
  };

  const handleSignup = saveDebateAndReturnUrl;
  const handleLogin = saveDebateAndReturnUrl;

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
        {result.verdict && (
          <>
            <span className="text-[var(--text-muted)]">|</span>
            <span
              className={`px-2 py-0.5 text-xs font-bold ${
                VERDICT_COLORS[result.verdict.toLowerCase().replace(/\s+/g, '_')] ||
                'bg-[var(--surface)] text-[var(--text)]'
              }`}
            >
              {result.verdict.replace(/_/g, ' ').toUpperCase()}
            </span>
          </>
        )}
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

      {/* Perspectives Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.entries(result.proposals)
          .filter(([agent]) => !agent.toLowerCase().includes('synthesizer'))
          .map(([agent, content]) => (
            <PerspectiveCard key={agent} agentName={agent} content={content} />
          ))}
        {Object.entries(result.proposals)
          .filter(([agent]) => agent.toLowerCase().includes('synthesizer'))
          .map(([agent, content]) => (
            <PerspectiveCard key={agent} agentName={agent} content={content} isFullWidth />
          ))}
      </div>

      {/* Collapsible critique details */}
      {(result.critiques.length > 0 || result.votes.length > 0) && (
        <>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="w-full text-left font-mono text-xs px-4 py-2 border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--acid-green)] hover:text-[var(--acid-green)] transition-colors"
          >
            {showDetails ? '[-] Hide critique details' : '[+] Show critique details'}
          </button>
          {showDetails && (
            <>
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
            </>
          )}
        </>
      )}

      {/* Verdict — visible to everyone */}
      {result.final_answer && (
        <div className="border border-[var(--acid-green)]/30 p-4">
          <h3 className="text-sm text-[var(--acid-green)] mb-3 font-bold font-mono">
            Verdict
          </h3>
          <p className="text-sm text-[var(--text)] whitespace-pre-wrap leading-relaxed font-mono">
            {result.final_answer}
          </p>
          {result.dissenting_views.length > 0 && (
            <div className="mt-3 pt-3 border-t border-[var(--border)]">
              <h4 className="text-xs text-[var(--text-muted)] font-bold font-mono mb-2">
                Dissenting Views
              </h4>
              {result.dissenting_views.map((view, i) => (
                <p key={i} className="text-xs text-[var(--text-muted)] whitespace-pre-wrap leading-relaxed mb-1">
                  {view}
                </p>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Receipt — summary visible, full download gated behind signup */}
      <div className="border border-[var(--acid-green)]/30 p-4">
        <h3 className="text-sm text-[var(--acid-green)] mb-3 font-bold font-mono">
          Decision Receipt
        </h3>
        <div className="grid grid-cols-2 gap-2 text-xs font-mono mb-4">
          <div>
            <span className="text-[var(--text-muted)]">Receipt ID: </span>
            <span className="text-[var(--acid-cyan)]">
              {result.receipt?.receipt_id || result.id}
            </span>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Verdict: </span>
            <span className="text-[var(--acid-green)]">
              {result.receipt?.verdict?.replace(/_/g, ' ') || (result.consensus_reached ? 'consensus reached' : 'no consensus')}
            </span>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Hash: </span>
            <span className="text-[var(--text)]">
              {result.receipt_hash ? result.receipt_hash.slice(0, 16) + '...' : 'pending'}
            </span>
          </div>
          <div>
            <span className="text-[var(--text-muted)]">Timestamp: </span>
            <span>{result.receipt?.timestamp || new Date().toISOString()}</span>
          </div>
        </div>
        <div className="border-t border-[var(--border)] pt-3 text-center">
          <p className="font-mono text-xs text-[var(--text-muted)] mb-3">
            Sign up to download the full receipt with evidence chains and cryptographic audit trail
          </p>
          <div className="flex gap-3 justify-center">
            <Link
              href="/signup"
              onClick={handleSignup}
              className="font-mono text-xs px-4 py-2 bg-[var(--acid-green)] text-[var(--bg)] font-bold hover:opacity-90 transition-opacity"
            >
              Sign Up Free
            </Link>
            <Link
              href="/auth/login"
              onClick={handleLogin}
              className="font-mono text-xs px-4 py-2 border border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--acid-green)] hover:text-[var(--acid-green)] transition-colors"
            >
              Log In
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
