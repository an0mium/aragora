'use client';

import { memo } from 'react';
import { BaseEdge, getBezierPath, type EdgeProps } from '@xyflow/react';

/**
 * Custom animated edge for showing knowledge flow from KM precedents.
 * Uses a dashed animated stroke with gradient color to indicate
 * knowledge transfer between nodes.
 */
export const KnowledgeFlowEdge = memo(function KnowledgeFlowEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
}: EdgeProps) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  return (
    <>
      <defs>
        <linearGradient id={`km-gradient-${id}`} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#8B5CF6" stopOpacity={0.6} />
          <stop offset="100%" stopColor="#3B82F6" stopOpacity={0.6} />
        </linearGradient>
      </defs>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          ...style,
          stroke: `url(#km-gradient-${id})`,
          strokeWidth: 2,
          strokeDasharray: '6 4',
          animation: 'km-flow 1.5s linear infinite',
        }}
      />
      {/* Animated flow indicator */}
      <circle r={3} fill="#8B5CF6" opacity={0.8}>
        <animateMotion dur="2s" repeatCount="indefinite" path={edgePath} />
      </circle>
      <style>{`
        @keyframes km-flow {
          to { stroke-dashoffset: -20; }
        }
      `}</style>
    </>
  );
});

export default KnowledgeFlowEdge;
