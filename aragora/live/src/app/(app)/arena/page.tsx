'use client';

import { useState, useCallback, useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { DebateInput } from '@/components/DebateInput';
import { useRightSidebar } from '@/context/RightSidebarContext';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { API_BASE_URL } from '@/config';
import Link from 'next/link';

// Template configurations for SME/Quickstart templates
const TEMPLATE_CONFIGS: Record<string, { name: string; description: string; format: 'light' | 'full'; mode: string }> = {
  'quickstart/yes-no': {
    name: 'Quick Yes/No Decision',
    description: 'Fast binary decision with AI consensus',
    format: 'light',
    mode: 'decision',
  },
  'quickstart/pros-cons': {
    name: 'Pros & Cons Analysis',
    description: 'Structured advantages and disadvantages analysis',
    format: 'light',
    mode: 'analysis',
  },
  'quickstart/risk-assessment': {
    name: 'Risk Assessment',
    description: 'Quick risk identification and scoring',
    format: 'light',
    mode: 'risk',
  },
  'quickstart/brainstorm': {
    name: 'Brainstorm Session',
    description: 'Multi-perspective idea generation',
    format: 'light',
    mode: 'creative',
  },
  'sme/vendor-evaluation': {
    name: 'Vendor Evaluation',
    description: 'Compare vendors across key criteria',
    format: 'full',
    mode: 'evaluation',
  },
  'sme/hiring-decision': {
    name: 'Hiring Decision',
    description: 'Candidate evaluation with multi-agent analysis',
    format: 'full',
    mode: 'hiring',
  },
  'sme/budget-allocation': {
    name: 'Budget Allocation',
    description: 'Optimize budget distribution',
    format: 'full',
    mode: 'finance',
  },
  'sme/business-decision': {
    name: 'Business Decision',
    description: 'Strategic decision analysis',
    format: 'full',
    mode: 'strategy',
  },
};

// Loading fallback for Suspense boundary
function ArenaLoading() {
  return (
    <main className="min-h-[calc(100vh-120px)] bg-[var(--bg)] text-[var(--text)] relative z-10">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-2xl sm:text-3xl font-mono text-[var(--acid-green)] mb-3">
            {'>'} DEBATE ARENA
          </h1>
          <p className="text-sm text-[var(--text-muted)] font-mono animate-pulse">
            Loading...
          </p>
        </div>
      </div>
    </main>
  );
}

// Wrap the page content in a Suspense boundary for useSearchParams
export default function ArenaPage() {
  return (
    <Suspense fallback={<ArenaLoading />}>
      <ArenaContent />
    </Suspense>
  );
}

function ArenaContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const templateId = searchParams.get('template');
  const templateConfig = templateId ? TEMPLATE_CONFIGS[templateId] : null;
  const [error, setError] = useState<string | null>(null);
  const [recentDebates, setRecentDebates] = useState<{ id: string; question: string; created_at: string }[]>([]);
  const { setContext, clearContext } = useRightSidebar();

  // Handle debate started - navigate to debate viewer
  const handleDebateStarted = useCallback((debateId: string, _question: string) => {
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
              {'>'} {templateConfig ? templateConfig.name.toUpperCase() : 'DEBATE ARENA'}
            </h1>
            <p className="text-sm text-[var(--text-muted)] font-mono max-w-xl mx-auto">
              {templateConfig
                ? templateConfig.description
                : 'Harness the collective intelligence of multiple AI models for better decisions. Choose your question and let Claude, GPT, Gemini, Grok & DeepSeek collaborate through debate.'}
            </p>
            {templateConfig && (
              <div className="mt-3 flex items-center justify-center gap-4">
                <span className="px-2 py-1 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30">
                  {templateConfig.format === 'light' ? 'Quick (~5 min)' : 'Thorough (~15 min)'}
                </span>
                <Link
                  href="/arena"
                  className="text-xs font-mono text-[var(--text-muted)] hover:text-[var(--acid-green)] underline"
                >
                  Switch to standard arena
                </Link>
              </div>
            )}
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
              defaultFormat={templateConfig?.format}
              templateId={templateId || undefined}
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
