'use client';

import { DebateThisButton } from '../DebateThisButton';

export interface TransitionData {
  id: string;
  from_stage: string;
  to_stage: string;
  ai_rationale?: string;
  confidence?: number;
  status?: string;
}

export interface StageTransitionGateProps {
  transition: TransitionData;
  pipelineId: string;
  onApprove?: (pipelineId: string, transitionId: string) => void;
  onReject?: (pipelineId: string, transitionId: string) => void;
}

export function StageTransitionGate({
  transition,
  pipelineId,
  onApprove,
  onReject,
}: StageTransitionGateProps) {
  const confidence = transition.confidence ?? 0;
  const confidencePct = Math.round(confidence * 100);

  return (
    <div className="bg-surface border border-border rounded-lg p-3 max-w-xs">
      <div className="flex items-center gap-2 mb-2">
        <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
        <span className="text-xs font-mono font-bold text-text uppercase">
          {transition.from_stage} &rarr; {transition.to_stage}
        </span>
      </div>

      {transition.ai_rationale && (
        <p className="text-xs text-text-muted mb-2">{transition.ai_rationale}</p>
      )}

      <div className="flex items-center gap-1 mb-2">
        <span className="text-xs text-text-muted font-mono">Confidence:</span>
        <div className="w-16 h-1 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-400 rounded-full"
            style={{ width: `${confidencePct}%` }}
          />
        </div>
        <span className="text-xs text-text font-mono">{confidencePct}%</span>
      </div>

      <div className="flex gap-2 mt-2">
        {onApprove && (
          <button
            onClick={() => onApprove(pipelineId, transition.id)}
            className="flex-1 px-2 py-1 bg-emerald-600 text-white text-xs font-mono rounded hover:bg-emerald-500 transition-colors"
          >
            Approve
          </button>
        )}
        {onReject && (
          <button
            onClick={() => onReject(pipelineId, transition.id)}
            className="flex-1 px-2 py-1 bg-red-600 text-white text-xs font-mono rounded hover:bg-red-500 transition-colors"
          >
            Reject
          </button>
        )}
      </div>

      {/* Debate this transition before approving */}
      <div className="mt-2 pt-2 border-t border-border">
        <DebateThisButton
          question={`Should we transition from ${transition.from_stage} to ${transition.to_stage}? ${transition.ai_rationale || ''}`}
          context={`Pipeline transition gate. Confidence: ${confidencePct}%. Rationale: ${transition.ai_rationale || 'none provided'}`}
          source="pipeline"
          variant="inline"
        />
      </div>
    </div>
  );
}

export default StageTransitionGate;
