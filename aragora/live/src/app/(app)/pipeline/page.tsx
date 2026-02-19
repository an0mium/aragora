'use client';

import { Suspense, useState, useCallback, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import dynamic from 'next/dynamic';
import { usePipeline } from '@/hooks/usePipeline';
import type { PipelineStageType } from '@/components/pipeline-canvas/types';

const PipelineCanvas = dynamic(
  () => import('@/components/pipeline-canvas/PipelineCanvas').then((m) => m.PipelineCanvas),
  { ssr: false, loading: () => <CanvasLoadingState /> },
);

function CanvasLoadingState() {
  return (
    <div className="flex-1 flex items-center justify-center bg-bg">
      <div className="text-center">
        <div className="animate-pulse text-acid-green text-xl font-mono mb-2">
          Loading Pipeline Canvas...
        </div>
        <p className="text-text-muted text-sm">Initializing Xyflow</p>
      </div>
    </div>
  );
}

/** Map transition target stages to the next stage for advancement */
const _NEXT_STAGE: Record<string, PipelineStageType> = {
  ideas: 'goals',
  goals: 'actions',
  actions: 'orchestration',
};

export default function PipelinePage() {
  return (
    <Suspense fallback={<CanvasLoadingState />}>
      <PipelinePageContent />
    </Suspense>
  );
}

function PipelinePageContent() {
  const {
    pipelineData,
    createFromIdeas,
    createFromDebate,
    advanceStage,
    executePipeline,
    loadDemo,
    reset,
    loading,
    executing,
    error,
  } = usePipeline();

  const searchParams = useSearchParams();

  const [showIdeaInput, setShowIdeaInput] = useState(false);
  const [showDebateInput, setShowDebateInput] = useState(false);
  const [ideaText, setIdeaText] = useState('');
  const [debateJson, setDebateJson] = useState('');
  const [debateError, setDebateError] = useState('');
  const [key, setKey] = useState(0);
  const [debateImportStatus, setDebateImportStatus] = useState<string | null>(null);

  // Auto-import from debate when ?from=debate&id=xxx is present
  useEffect(() => {
    const from = searchParams?.get('from');
    const debateId = searchParams?.get('id');
    if (from === 'debate' && debateId && !pipelineData) {
      setDebateImportStatus('Fetching debate results...');
      fetch(`/api/v1/debates/${encodeURIComponent(debateId)}`)
        .then((res) => {
          if (!res.ok) throw new Error(`Failed to fetch debate: ${res.status}`);
          return res.json();
        })
        .then((debate) => {
          // Extract proposals/conclusions as cartographer-style data
          const nodes = (debate.proposals || debate.conclusions || debate.messages || []).map(
            (item: Record<string, unknown>, i: number) => ({
              id: (item.id as string) || `debate-node-${i}`,
              type: (item.type as string) || 'proposal',
              summary: (item.summary as string) || (item.content as string) || '',
              content: (item.content as string) || (item.summary as string) || '',
            }),
          );
          if (nodes.length === 0) {
            setDebateImportStatus('No proposals found in debate');
            return;
          }
          setDebateImportStatus(`Importing ${nodes.length} debate proposals...`);
          return createFromDebate({ nodes, edges: [] });
        })
        .then(() => {
          setDebateImportStatus(null);
          setKey((k) => k + 1);
        })
        .catch((err) => {
          setDebateImportStatus(`Import failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
        });
    }
    // Only run once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFromIdeas = useCallback(async () => {
    const ideas = ideaText
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean);
    if (ideas.length > 0) {
      await createFromIdeas(ideas);
      setShowIdeaInput(false);
      setIdeaText('');
      setKey((k) => k + 1);
    }
  }, [ideaText, createFromIdeas]);

  const handleFromDebate = useCallback(async () => {
    setDebateError('');
    try {
      const data = JSON.parse(debateJson);
      if (!data.nodes || !Array.isArray(data.nodes)) {
        setDebateError('JSON must have a "nodes" array');
        return;
      }
      await createFromDebate(data);
      setShowDebateInput(false);
      setDebateJson('');
      setKey((k) => k + 1);
    } catch {
      setDebateError('Invalid JSON — paste ArgumentCartographer export');
    }
  }, [debateJson, createFromDebate]);

  const handleDemo = useCallback(() => {
    loadDemo();
    setKey((k) => k + 1);
  }, [loadDemo]);

  const handleNew = useCallback(() => {
    reset();
    setKey((k) => k + 1);
  }, [reset]);

  const handleStageAdvance = useCallback(
    (pipelineId: string, stage: PipelineStageType) => {
      advanceStage(pipelineId, stage);
      setKey((k) => k + 1);
    },
    [advanceStage],
  );

  const handleTransitionApprove = useCallback(
    (pipelineId: string, transitionId: string) => {
      // Find the transition to determine target stage
      const transition = pipelineData?.transitions?.find(
        (t) => (t.id as string) === transitionId,
      );
      if (transition) {
        const toStage = transition.to_stage as PipelineStageType;
        advanceStage(pipelineId, toStage);
        setKey((k) => k + 1);
      }
    },
    [pipelineData, advanceStage],
  );

  const handleTransitionReject = useCallback(
    (_pipelineId: string, _transitionId: string) => {
      // Rejection is a no-op for now — the stage stays pending
      // Future: could send rejection reason to backend
    },
    [],
  );

  const handleExecute = useCallback(async () => {
    if (pipelineData?.pipeline_id) {
      await executePipeline(pipelineData.pipeline_id);
    }
  }, [pipelineData, executePipeline]);

  const allStagesComplete = pipelineData?.stage_status
    ? Object.values(pipelineData.stage_status).every((s) => s === 'complete')
    : false;

  return (
    <div className="flex flex-col h-screen bg-bg">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-border bg-surface">
        <div>
          <h1 className="text-xl font-mono font-bold text-text">
            Idea-to-Execution Pipeline
          </h1>
          <p className="text-sm text-text-muted font-mono">
            Ideas &rarr; Goals &rarr; Actions &rarr; Orchestration
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleNew}
            className="px-4 py-2 bg-surface border border-border text-text font-mono text-sm hover:border-text transition-colors rounded"
          >
            New Pipeline
          </button>

          <button
            onClick={() => { setShowIdeaInput(!showIdeaInput); setShowDebateInput(false); }}
            disabled={loading}
            className="px-4 py-2 bg-indigo-600 text-white font-mono text-sm hover:bg-indigo-500 transition-colors rounded"
          >
            From Ideas
          </button>

          <button
            onClick={() => { setShowDebateInput(!showDebateInput); setShowIdeaInput(false); }}
            disabled={loading}
            className="px-4 py-2 bg-violet-600 text-white font-mono text-sm hover:bg-violet-500 transition-colors rounded"
          >
            From Debate
          </button>

          <button
            onClick={handleDemo}
            disabled={loading}
            className="px-4 py-2 bg-emerald-600 text-white font-mono text-sm hover:bg-emerald-500 transition-colors rounded"
          >
            {loading ? 'Loading...' : 'Try Demo'}
          </button>

          {pipelineData && allStagesComplete && (
            <button
              onClick={handleExecute}
              disabled={executing}
              className="px-4 py-2 bg-amber-600 text-white font-mono text-sm hover:bg-amber-500 disabled:opacity-50 transition-colors rounded"
            >
              {executing ? 'Executing...' : 'Execute Pipeline'}
            </button>
          )}
        </div>
      </header>

      {/* Idea input dropdown */}
      {showIdeaInput && (
        <div className="px-6 py-4 border-b border-border bg-surface/50">
          <label className="block text-sm font-mono text-text-muted mb-2">
            Enter ideas (one per line):
          </label>
          <textarea
            value={ideaText}
            onChange={(e) => setIdeaText(e.target.value)}
            rows={4}
            className="w-full max-w-lg bg-bg border border-border rounded p-3 text-sm text-text font-mono resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500"
            placeholder={"Build a rate limiter\nAdd caching layer\nImprove API docs\nSet up monitoring"}
          />
          <div className="flex gap-2 mt-2">
            <button
              onClick={handleFromIdeas}
              disabled={!ideaText.trim() || loading}
              className="px-4 py-2 bg-indigo-600 text-white text-sm font-mono rounded hover:bg-indigo-500 disabled:opacity-50 transition-colors"
            >
              {loading ? 'Generating...' : 'Generate Pipeline'}
            </button>
            <button
              onClick={() => setShowIdeaInput(false)}
              className="px-4 py-2 text-sm font-mono text-text-muted hover:text-text"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Debate JSON input dropdown */}
      {showDebateInput && (
        <div className="px-6 py-4 border-b border-border bg-surface/50">
          <label className="block text-sm font-mono text-text-muted mb-2">
            Paste ArgumentCartographer JSON export:
          </label>
          <textarea
            value={debateJson}
            onChange={(e) => { setDebateJson(e.target.value); setDebateError(''); }}
            rows={6}
            className="w-full max-w-lg bg-bg border border-border rounded p-3 text-sm text-text font-mono resize-none focus:outline-none focus:ring-2 focus:ring-violet-500"
            placeholder={'{\n  "nodes": [\n    {"id": "n1", "type": "proposal", "summary": "...", "content": "..."}\n  ],\n  "edges": [\n    {"source_id": "n2", "target_id": "n1", "relation": "supports"}\n  ]\n}'}
          />
          {debateError && (
            <p className="text-xs text-red-400 font-mono mt-1">{debateError}</p>
          )}
          <div className="flex gap-2 mt-2">
            <button
              onClick={handleFromDebate}
              disabled={!debateJson.trim() || loading}
              className="px-4 py-2 bg-violet-600 text-white text-sm font-mono rounded hover:bg-violet-500 disabled:opacity-50 transition-colors"
            >
              {loading ? 'Generating...' : 'Import Debate'}
            </button>
            <button
              onClick={() => { setShowDebateInput(false); setDebateError(''); }}
              className="px-4 py-2 text-sm font-mono text-text-muted hover:text-text"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Debate import status */}
      {debateImportStatus && (
        <div className="px-6 py-2 bg-violet-500/10 border-b border-violet-500/30">
          <p className="text-sm text-violet-400 font-mono">{debateImportStatus}</p>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="px-6 py-2 bg-red-500/10 border-b border-red-500/30">
          <p className="text-sm text-red-400 font-mono">{error}</p>
        </div>
      )}

      {/* Canvas or empty state */}
      <div className="flex-1 overflow-hidden">
        {pipelineData ? (
          <PipelineCanvas
            key={key}
            pipelineId={pipelineData.pipeline_id}
            initialData={pipelineData}
            onStageAdvance={handleStageAdvance}
            onTransitionApprove={handleTransitionApprove}
            onTransitionReject={handleTransitionReject}
          />
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <h2 className="text-2xl font-mono font-bold text-text mb-4">
                No Pipeline Active
              </h2>
              <p className="text-text-muted mb-6">
                Start by entering ideas, importing a debate, or trying the demo to see the
                four-stage pipeline in action.
              </p>
              <div className="flex gap-3 justify-center">
                <button
                  onClick={() => setShowIdeaInput(true)}
                  className="px-6 py-3 bg-indigo-600 text-white font-mono text-sm rounded hover:bg-indigo-500 transition-colors"
                >
                  Enter Ideas
                </button>
                <button
                  onClick={() => setShowDebateInput(true)}
                  className="px-6 py-3 bg-violet-600 text-white font-mono text-sm rounded hover:bg-violet-500 transition-colors"
                >
                  From Debate
                </button>
                <button
                  onClick={handleDemo}
                  className="px-6 py-3 bg-emerald-600 text-white font-mono text-sm rounded hover:bg-emerald-500 transition-colors"
                >
                  Try Demo
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
