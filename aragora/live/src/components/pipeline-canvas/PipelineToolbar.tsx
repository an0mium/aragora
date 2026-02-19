'use client';

import { memo } from 'react';
import { PIPELINE_STAGE_CONFIG, type PipelineStageType } from './types';

const STAGES: PipelineStageType[] = ['ideas', 'goals', 'actions', 'orchestration'];

const AI_GENERATE_STAGES: PipelineStageType[] = ['goals', 'actions', 'orchestration'];

const AI_GENERATE_LABELS: Record<string, string> = {
  goals: 'AI Generate Goals',
  actions: 'AI Generate Actions',
  orchestration: 'AI Generate Orchestration',
};

function getAdvanceLabel(currentStage: PipelineStageType): string {
  const currentIndex = STAGES.indexOf(currentStage);
  if (currentIndex < 0 || currentIndex >= STAGES.length - 1) {
    return 'Advance';
  }
  const nextStage = STAGES[currentIndex + 1];
  const nextConfig = PIPELINE_STAGE_CONFIG[nextStage];
  return `Advance to ${nextConfig.label}`;
}

interface PipelineToolbarProps {
  stage: PipelineStageType;
  nodeCount: number;
  edgeCount: number;
  readOnly?: boolean;
  loading?: boolean;
  onSave?: () => void;
  onClear?: () => void;
  onAIGenerate?: () => void;
  onExecute?: () => void;
  canAdvance?: boolean;
  onAdvance?: () => void;
  onExportReceipt?: () => void;
  pipelineId?: string;
}

export const PipelineToolbar = memo(function PipelineToolbar({
  stage,
  nodeCount,
  edgeCount,
  readOnly = false,
  loading = false,
  onSave,
  onClear,
  onAIGenerate,
  onExecute,
  canAdvance = false,
  onAdvance,
  onExportReceipt,
  pipelineId,
}: PipelineToolbarProps) {
  if (readOnly) {
    return null;
  }

  const showAIGenerate = AI_GENERATE_STAGES.includes(stage);
  const showExecute = stage === 'orchestration';

  return (
    <div className="flex items-center gap-2">
      {/* Save */}
      <button
        onClick={onSave}
        className="px-4 py-2 bg-acid-green text-bg font-mono text-sm font-bold hover:bg-acid-green/80 transition-colors rounded"
      >
        SAVE
      </button>

      {/* AI Generate */}
      {showAIGenerate && onAIGenerate && (
        <button
          onClick={onAIGenerate}
          disabled={loading}
          className="px-4 py-2 bg-indigo-600 text-white font-mono text-sm font-bold hover:bg-indigo-500 transition-colors rounded flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <svg
            className="w-4 h-4"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z" />
            <path d="M18 14l1 3.5L22.5 18l-3.5 1L18 22.5l-1-3.5L13.5 18l3.5-1L18 14z" />
          </svg>
          {loading ? 'Generating...' : AI_GENERATE_LABELS[stage]}
        </button>
      )}

      {/* Clear Stage */}
      <button
        onClick={onClear}
        className="px-4 py-2 bg-surface border border-border text-text font-mono text-sm hover:border-text transition-colors rounded"
      >
        CLEAR STAGE
      </button>

      {/* Execute */}
      {showExecute && onExecute && (
        <button
          onClick={onExecute}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white font-mono text-sm font-bold hover:bg-blue-500 transition-colors rounded flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <svg
            className="w-4 h-4"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
          EXECUTE
        </button>
      )}

      {/* Advance */}
      {canAdvance && onAdvance && (
        <button
          onClick={onAdvance}
          className="px-4 py-2 bg-acid-green text-bg font-mono text-sm font-bold hover:bg-acid-green/80 transition-colors rounded"
        >
          {getAdvanceLabel(stage)}
        </button>
      )}

      {/* Export Receipt */}
      {onExportReceipt && pipelineId && (
        <button
          onClick={onExportReceipt}
          className="px-4 py-2 bg-surface border border-border text-text font-mono text-sm hover:border-text transition-colors rounded flex items-center gap-2"
          data-testid="export-receipt-btn"
        >
          <svg
            className="w-4 h-4"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          EXPORT RECEIPT
        </button>
      )}

      {/* Node/Edge count */}
      <span className="ml-2 text-xs font-mono text-text-muted">
        {nodeCount} nodes | {edgeCount} edges
      </span>
    </div>
  );
});

export default PipelineToolbar;
