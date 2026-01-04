'use client';

import { useState, useEffect, use } from 'react';
import Link from 'next/link';
import { fetchDebateById, type DebateArtifact } from '@/utils/supabase';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { ThemeToggle } from '@/components/ThemeToggle';

// Agent color schemes
const AGENT_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  gemini: { bg: 'bg-purple/10', text: 'text-purple', border: 'border-purple/40' },
  codex: { bg: 'bg-gold/10', text: 'text-gold', border: 'border-gold/40' },
  claude: { bg: 'bg-acid-cyan/10', text: 'text-acid-cyan', border: 'border-acid-cyan/40' },
  grok: { bg: 'bg-crimson/10', text: 'text-crimson', border: 'border-crimson/40' },
  default: { bg: 'bg-acid-green/10', text: 'text-acid-green', border: 'border-acid-green/40' },
};

function getAgentColors(agentName: string) {
  const name = agentName.toLowerCase();
  if (name.startsWith('gemini')) return AGENT_COLORS.gemini;
  if (name.startsWith('codex')) return AGENT_COLORS.codex;
  if (name.startsWith('claude')) return AGENT_COLORS.claude;
  if (name.startsWith('grok')) return AGENT_COLORS.grok;
  return AGENT_COLORS.default;
}

interface TranscriptMessage {
  agent: string;
  role?: string;
  content: string;
  round?: number;
  timestamp?: number;
}

export default function DebateViewerPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [debate, setDebate] = useState<DebateArtifact | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    async function loadDebate() {
      try {
        const data = await fetchDebateById(id);
        if (data) {
          setDebate(data);
        } else {
          setError('Debate not found');
        }
      } catch (e) {
        setError('Failed to load debate');
      } finally {
        setLoading(false);
      }
    }

    loadDebate();
  }, [id]);

  const handleShare = async () => {
    const url = window.location.href;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = url;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>

            <div className="flex items-center gap-3">
              <Link
                href="/"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [BACK TO LIVE]
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-6">
          {loading && (
            <div className="flex items-center justify-center py-20">
              <div className="text-acid-green font-mono animate-pulse">
                {'>'} LOADING DEBATE...
              </div>
            </div>
          )}

          {error && (
            <div className="bg-warning/10 border border-warning/30 rounded-lg p-6 text-center">
              <div className="text-warning text-2xl mb-2">{'>'} ERROR</div>
              <div className="text-text-muted">{error}</div>
              <Link
                href="/"
                className="inline-block mt-4 text-acid-green hover:underline font-mono"
              >
                [RETURN HOME]
              </Link>
            </div>
          )}

          {debate && (
            <div className="space-y-6">
              {/* Debate Header */}
              <div className="bg-surface border border-acid-green/30 p-6">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="text-xs text-text-muted font-mono mb-2">
                      DEBATE // CYCLE {debate.cycle_number} // {debate.phase.toUpperCase()}
                    </div>
                    <h1 className="text-lg font-mono text-acid-green mb-4">
                      {debate.task}
                    </h1>
                    <div className="flex flex-wrap gap-2">
                      {debate.agents.map((agent) => {
                        const colors = getAgentColors(agent);
                        return (
                          <span
                            key={agent}
                            className={`px-2 py-1 text-xs font-mono ${colors.bg} ${colors.text} ${colors.border} border`}
                          >
                            {agent}
                          </span>
                        );
                      })}
                    </div>
                  </div>

                  <div className="flex flex-col items-end gap-2">
                    <button
                      onClick={handleShare}
                      className="px-3 py-1 text-xs font-mono bg-acid-green text-bg hover:bg-acid-green/80 transition-colors"
                    >
                      {copied ? '[COPIED!]' : '[SHARE LINK]'}
                    </button>
                    <div className="text-xs text-text-muted font-mono">
                      {new Date(debate.created_at).toLocaleString()}
                    </div>
                  </div>
                </div>

                {/* Consensus Status */}
                <div className="mt-4 pt-4 border-t border-acid-green/20 flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <span
                      className={`w-2 h-2 rounded-full ${
                        debate.consensus_reached ? 'bg-green-400' : 'bg-yellow-400'
                      }`}
                    />
                    <span className="text-xs font-mono text-text-muted">
                      {debate.consensus_reached ? 'CONSENSUS REACHED' : 'NO CONSENSUS'}
                    </span>
                  </div>
                  <div className="text-xs font-mono text-text-muted">
                    CONFIDENCE: {Math.round(debate.confidence * 100)}%
                  </div>
                  {debate.vote_tally && Object.keys(debate.vote_tally).length > 0 && (
                    <div className="text-xs font-mono text-text-muted">
                      VOTES: {Object.entries(debate.vote_tally).map(([k, v]) => `${k}:${v}`).join(' ')}
                    </div>
                  )}
                </div>
              </div>

              {/* Winning Proposal */}
              {debate.winning_proposal && (
                <div className="bg-gradient-to-br from-accent/10 to-purple-500/10 border-2 border-accent/50 p-6">
                  <div className="text-xs text-accent font-mono mb-2 uppercase tracking-wider">
                    Winning Proposal
                  </div>
                  <div className="text-text whitespace-pre-wrap font-mono text-sm">
                    {debate.winning_proposal}
                  </div>
                </div>
              )}

              {/* Transcript */}
              <div className="bg-surface border border-acid-green/30">
                <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50">
                  <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
                    {'>'} DEBATE TRANSCRIPT
                  </span>
                </div>
                <div className="p-4 space-y-4 max-h-[600px] overflow-y-auto">
                  {(debate.transcript as TranscriptMessage[]).map((msg, idx) => {
                    const colors = getAgentColors(msg.agent || 'system');
                    return (
                      <div
                        key={idx}
                        className={`${colors.bg} border ${colors.border} p-4`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className={`font-mono font-bold text-sm ${colors.text}`}>
                              {(msg.agent || 'SYSTEM').toUpperCase()}
                            </span>
                            {msg.role && (
                              <span className="text-xs text-text-muted border border-text-muted/30 px-1">
                                {msg.role}
                              </span>
                            )}
                            {msg.round !== undefined && msg.round > 0 && (
                              <span className="text-xs text-text-muted">
                                R{msg.round}
                              </span>
                            )}
                          </div>
                          {msg.timestamp && (
                            <span className="text-[10px] text-text-muted font-mono">
                              {new Date(msg.timestamp * 1000).toLocaleTimeString()}
                            </span>
                          )}
                        </div>
                        <div className="text-sm text-text whitespace-pre-wrap">
                          {msg.content}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Metadata */}
              <div className="text-center text-xs font-mono text-text-muted py-4 border-t border-acid-green/20">
                <div>DEBATE ID: {debate.id}</div>
                <div>LOOP: {debate.loop_id}</div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'═'.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA DEBATE VIEWER // PERMALINK
          </p>
          <p className="text-acid-cyan mt-2">
            <a
              href="https://aragora.ai"
              className="hover:text-acid-green transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              [ ARAGORA.AI ]
            </a>
          </p>
          <div className="text-acid-green/50 mt-4">
            {'═'.repeat(40)}
          </div>
        </footer>
      </main>
    </>
  );
}
