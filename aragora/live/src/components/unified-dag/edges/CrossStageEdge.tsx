'use client';

import { type EdgeProps, getBezierPath } from '@xyflow/react';

export function CrossStageEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
}: EdgeProps) {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <g>
      <defs>
        <linearGradient id={`grad-${id}`} x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="50%" stopColor="#10b981" />
          <stop offset="100%" stopColor="#f59e0b" />
        </linearGradient>
      </defs>
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={`url(#grad-${id})`}
        strokeWidth={2}
        strokeDasharray="6 3"
        style={style}
      >
        <animate
          attributeName="stroke-dashoffset"
          from="18"
          to="0"
          dur="1.5s"
          repeatCount="indefinite"
        />
      </path>
    </g>
  );
}
