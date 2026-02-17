'use client';

import { useState, useCallback } from 'react';
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

export default function PipelinePage() {
  const {
    pipelineData,
    createFromIdeas,
    advanceStage,
    loadDemo,
    reset,
    loading,
    error,
  } = usePipeline();

  const [showIdeaInput, setShowIdeaInput] = useState(false);
  const [ideaText, setIdeaText] = useState('');
  const [key, setKey] = useState(0);

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
            onClick={() => setShowIdeaInput(!showIdeaInput)}
            disabled={loading}
            className="px-4 py-2 bg-indigo-600 text-white font-mono text-sm hover:bg-indigo-500 transition-colors rounded"
          >
            From Ideas
          </button>

          <button
            onClick={handleDemo}
            disabled={loading}
            className="px-4 py-2 bg-emerald-600 text-white font-mono text-sm hover:bg-emerald-500 transition-colors rounded"
          >
            {loading ? 'Loading...' : 'Try Demo'}
          </button>
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
