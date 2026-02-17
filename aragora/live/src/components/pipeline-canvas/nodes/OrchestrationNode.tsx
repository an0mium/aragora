'use client';

import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';

const orchTypeLabels: Record<string, string> = {
  agent_task: 'Agent Task',
  debate: 'Debate',
  human_gate: 'Human Gate',
  parallel_fan: 'Parallel',
  merge: 'Merge',
  verification: 'Verify',
  agent: 'Agent',
};

interface OrchestrationNodeProps {
  data: Record<string, unknown>;
  selected?: boolean;
}

export const OrchestrationNode = memo(function OrchestrationNode({
  data,
  selected,
}: OrchestrationNodeProps) {
  const orchType = (data.orchType || data.orch_type || 'agent_task') as string;
  const label = data.label as string;
  const assignedAgent = (data.assignedAgent || data.assigned_agent) as string | undefined;
  const agentType = (data.agentType || data.agent_type) as string | undefined;
  const capabilities = data.capabilities as string[] | undefined;
  const isAgent = orchType === 'agent';
  const isHumanGate = orchType === 'human_gate';

  return (
    <div
      className={`
        px-4 py-3 border-2 min-w-[180px] max-w-[250px]
        bg-pink-500/20 border-pink-500
        ${isAgent ? 'rounded-full' : 'rounded-lg'}
        ${isHumanGate ? 'border-dashed' : ''}
        ${selected ? 'ring-2 ring-acid-green ring-offset-2 ring-offset-bg' : ''}
        transition-all duration-200
      `}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="w-3 h-3 bg-pink-500 border-2 border-bg"
      />

      <div className="flex items-center gap-2 mb-2">
        <span className="px-1.5 py-0.5 text-xs bg-pink-500/30 text-pink-200 rounded font-mono uppercase">
          {orchTypeLabels[orchType] || orchType}
        </span>
        {assignedAgent && (
          <span className="text-xs text-text-muted font-mono">{assignedAgent}</span>
        )}
      </div>

      <div className="text-sm font-medium text-text mb-1 truncate">
        {label}
      </div>

      {agentType && (
        <div className="text-xs text-pink-300 font-mono mb-1">{agentType}</div>
      )}

      {capabilities && capabilities.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {capabilities.slice(0, 3).map((cap) => (
            <span
              key={cap}
              className="px-1 py-0.5 text-xs bg-pink-500/20 text-pink-200 rounded font-mono"
            >
              {cap}
            </span>
          ))}
        </div>
      )}

      <Handle
        type="source"
        position={Position.Right}
        className="w-3 h-3 bg-pink-500 border-2 border-bg"
      />
    </div>
  );
});

export default OrchestrationNode;
