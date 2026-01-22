'use client';

import { useState, useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { DebateInput } from '@/components/DebateInput';
import { useRightSidebar } from '@/context/RightSidebarContext';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { API_BASE_URL } from '@/config';
import Link from 'next/link';

export default function ArenaPage() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [recentDebates, setRecentDebates] = useState<{ id: string; question: string; created_at: string }[]>([]);
  const { setContext, clearContext } = useRightSidebar();

  // Handle debate started - navigate to debate viewer
  const handleDebateStarted = useCallback((debateId: string, question: string) => {
    router.push(`/debate/${debateId}`);
  }, [router]);

  // Handle error
  const handleError = useCallback((err: string) => {
    setError(err);
  }, []);

  // Fetch recent debates for the sidebar
  useEffect(() => {
    async function fetchRecent() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/debates?limit=5`);
        if (response.ok) {
          const data = await response.json();
          if (data.debates) {
            setRecentDebates(data.debates.slice(0, 5));
          }
        }
      } catch {
        // Silently fail - recent debates are not critical
      }
    }
    fetchRecent();
  }, []);

  // Set up right sidebar
  useEffect(() => {
    setContext({
      title: 'Debate Arena',
      subtitle: 'Start a new debate',
      statsContent: (
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Mode</span>
            <span className="text-sm font-mono text-[var(--acid-green)]">Standard</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Agents</span>
            <span className="text-sm font-mono text-[var(--acid-cyan)]">4-5</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Rounds</span>
            <span className="text-sm font-mono text-[var(--text)]">4-9</span>
          </div>
        </div>
      ),
      actionsContent: recentDebates.length > 0 ? (
        <div className="space-y-2">
          <span className="text-xs text-[var(--text-muted)] block mb-2">Recent Debates</span>
          {recentDebates.map((debate) => (
            <Link
              key={debate.id}
              href={`/debate/${debate.id}`}
              className="block w-full px-2 py-1.5 text-xs font-mono text-left bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 hover:text-[var(--text)] transition-colors truncate"
            >
              {debate.question?.slice(0, 40) || debate.id.slice(0, 8)}...
            </Link>
          ))}
        </div>
      ) : null,
    });

    return () => clearContext();
  }, [recentDebates, setContext, clearContext]);

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-[calc(100vh-120px)] bg-[var(--bg)] text-[var(--text)] relative z-10">
        <div className="max-w-4xl mx-auto px-4 py-8">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-2xl sm:text-3xl font-mono text-[var(--acid-green)] mb-3">
              {'>'} DEBATE ARENA
            </h1>
            <p className="text-sm text-[var(--text-muted)] font-mono max-w-xl mx-auto">
              Harness the collective intelligence of multiple AI models for better decisions.
              Choose your question and let Claude, GPT, Gemini, Grok & DeepSeek collaborate through debate.
            </p>
          </div>

          {/* Error Banner */}
          {error && (
            <div className="mb-6 bg-[var(--warning)]/10 border border-[var(--warning)]/30 p-4 flex items-center justify-between">
              <span className="text-[var(--warning)] font-mono text-sm">{error}</span>
              <button
                onClick={() => setError(null)}
                className="text-[var(--warning)] hover:text-[var(--warning)]/80"
                aria-label="Dismiss error"
              >

              </button>
            </div>
          )}

          {/* Debate Input */}
          <div className="bg-[var(--surface)] border border-[var(--border)] p-6">
            <DebateInput
              apiBase={API_BASE_URL}
              onDebateStarted={handleDebateStarted}
              onError={handleError}
            />
          </div>

          {/* Quick Info */}
          <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-[var(--surface)]/50 border border-[var(--border)] p-4">
              <div className="text-[var(--acid-green)] font-mono text-lg mb-2">01</div>
              <h3 className="text-sm font-mono text-[var(--text)] mb-1">Enter Question</h3>
              <p className="text-xs text-[var(--text-muted)]">
                Pose any question - philosophical, technical, or creative
              </p>
            </div>
            <div className="bg-[var(--surface)]/50 border border-[var(--border)] p-4">
              <div className="text-[var(--acid-green)] font-mono text-lg mb-2">02</div>
              <h3 className="text-sm font-mono text-[var(--text)] mb-1">AI Models Collaborate</h3>
              <p className="text-xs text-[var(--text-muted)]">
                Watch as diverse AI models discuss, critique, and refine answers together
              </p>
            </div>
            <div className="bg-[var(--surface)]/50 border border-[var(--border)] p-4">
              <div className="text-[var(--acid-green)] font-mono text-lg mb-2">03</div>
              <h3 className="text-sm font-mono text-[var(--text)] mb-1">Consensus Emerges</h3>
              <p className="text-xs text-[var(--text-muted)]">
                Get a battle-tested answer with citations and confidence scores
              </p>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
