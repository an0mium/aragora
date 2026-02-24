'use client';

import { Handle, Position, type NodeProps } from '@xyflow/react';
import type { DAGNodeData } from '@/hooks/useUnifiedDAG';

export function AgentAssignmentNode({ data }: NodeProps) {
  const nodeData = data as unknown as DAGNodeData;
  const agents = (nodeData.metadata?.agents as string[]) || [];
  const track = (nodeData.metadata?.track as string) || 'general';

  return (
    <div className="px-3 py-2 rounded-lg border border-pink-500/40 bg-surface shadow-md min-w-[160px]">
      <Handle type="target" position={Position.Left} className="!bg-pink-500" />
      <div className="text-xs font-mono font-bold text-pink-400 uppercase tracking-wide mb-1">
        Agent Assignment
      </div>
      <div className="text-sm font-medium text-text mb-1 truncate">{nodeData.label}</div>
      <div className="flex flex-wrap gap-1">
        <span className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-pink-500/20 text-pink-300">
          {track}
        </span>
        {agents.slice(0, 3).map((agent) => (
          <span
            key={agent}
            className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-surface border border-border text-text-muted"
          >
            {agent}
          </span>
        ))}
        {agents.length > 3 && (
          <span className="px-1.5 py-0.5 text-[10px] font-mono text-text-muted">
            +{agents.length - 3}
          </span>
        )}
      </div>
      <Handle type="source" position={Position.Right} className="!bg-pink-500" />
    </div>
  );
}
