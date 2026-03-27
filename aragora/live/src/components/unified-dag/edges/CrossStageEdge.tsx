'use client';

import { EdgeLabelRenderer, type EdgeProps, getBezierPath } from '@xyflow/react';
import { STAGE_COLORS, type DAGEdgeData } from '@/hooks/useUnifiedDAG';

export function CrossStageEdge({
  id,
  data,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
}: EdgeProps) {
  const edgeData = (data ?? {}) as Partial<DAGEdgeData>;
  const sourceColor = edgeData.sourceStage ? STAGE_COLORS[edgeData.sourceStage] : '#6366f1';
  const targetColor = edgeData.targetStage ? STAGE_COLORS[edgeData.targetStage] : '#10b981';
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      <defs>
        <linearGradient id={`grad-${id}`} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor={sourceColor} />
          <stop offset="100%" stopColor={targetColor} />
        </linearGradient>
        <marker
          id={`arrow-${id}`}
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill={targetColor} />
        </marker>
      </defs>
      <path
        d={edgePath}
        fill="none"
        stroke={`url(#grad-${id})`}
        strokeWidth={edgeData.blocking ? 2.5 : 1.75}
        strokeDasharray={edgeData.implicit ? '4 4' : edgeData.crossStage ? '8 4' : undefined}
        markerEnd={`url(#arrow-${id})`}
        style={style}
      >
        {edgeData.crossStage ? (
          <animate
            attributeName="stroke-dashoffset"
            from="18"
            to="0"
            dur="1.4s"
            repeatCount="indefinite"
          />
        ) : null}
      </path>
      {edgeData.label ? (
        <EdgeLabelRenderer>
          <div
            className="pointer-events-none absolute rounded-full border border-border/70 bg-surface/90 px-2 py-0.5 text-[10px] font-mono text-text-muted"
            style={{
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            }}
          >
            {edgeData.label.replace(/_/g, ' ')}
          </div>
        </EdgeLabelRenderer>
      ) : null}
    </>
  );
}
