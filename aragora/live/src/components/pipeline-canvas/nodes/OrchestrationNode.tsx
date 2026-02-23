'use client';

import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import {
  PIPELINE_NODE_TYPE_CONFIGS,
  ORCH_STATUS_COLORS,
  type OrchType,
  type OrchStatus,
} from '../types';

interface OrchestrationNodeProps {
  data: Record<string, unknown>;
  selected?: boolean;
}

export const OrchestrationNode = memo(function OrchestrationNode({
  data,
  selected,
}: OrchestrationNodeProps) {
  const orchType = (data.orchType || data.orch_type || 'agent_task') as OrchType;
  const label = data.label as string;
  const description = data.description as string | undefined;
  const assignedAgent = (data.assignedAgent || data.assigned_agent) as string | undefined;
  const agentType = (data.agentType || data.agent_type) as string | undefined;
  const capabilities = data.capabilities as string[] | undefined;
  const status = (data.status || 'pending') as OrchStatus;
  const lockedBy = data.lockedBy as string | undefined;
  const executionStatus = data.executionStatus as string | undefined;
  const executionDuration = data.executionDuration as string | undefined;
  const executionAgent = data.executionAgent as string | undefined;

  const isAgent = orchType === 'agent_task' || orchType === 'debate';
  const isHumanGate = orchType === 'human_gate';

  const config = PIPELINE_NODE_TYPE_CONFIGS.orchestration[orchType] || PIPELINE_NODE_TYPE_CONFIGS.orchestration.agent_task;
  const statusClass = ORCH_STATUS_COLORS[status] || ORCH_STATUS_COLORS.pending;

  return (
    <div
      className={`
        px-4 py-3 border-2 min-w-[200px] max-w-[270px]
        ${config.color} ${config.borderColor}
        ${isAgent ? 'rounded-full' : 'rounded-lg'}
        ${isHumanGate ? 'border-dashed' : ''}
        ${selected ? 'ring-2 ring-acid-green ring-offset-2 ring-offset-bg' : ''}
        ${lockedBy ? 'opacity-70' : ''}
        transition-all duration-200
      `}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="w-3 h-3 bg-pink-500 border-2 border-bg"
      />

      {/* Header: icon + type badge + status badge */}
      <div className="flex items-center gap-2 mb-2">
        <span className="w-5 h-5 flex items-center justify-center text-xs font-bold rounded bg-pink-500/30 text-pink-200">
          {config.icon}
        </span>
        <span className="px-1.5 py-0.5 text-xs bg-pink-500/30 text-pink-200 rounded font-mono uppercase">
          {config.label}
        </span>
        <span className={`px-1.5 py-0.5 text-xs rounded font-mono ${statusClass}`}>
          {status.replace('_', ' ')}
        </span>
      </div>

      {/* Label */}
      <div className="text-sm font-medium text-text mb-1 line-clamp-2">
        {label}
      </div>

      {/* Description */}
      {description && (
        <div className="text-xs text-text-muted mb-1 line-clamp-2">
          {description}
        </div>
      )}

      {/* Assigned agent */}
      {assignedAgent && (
        <div className="text-xs text-pink-300/80 mb-1">
          agent: {assignedAgent}
        </div>
      )}

      {/* Agent type */}
      {agentType && (
        <div className="text-xs text-pink-300 font-mono mb-1">{agentType}</div>
      )}

      {/* Capabilities badges */}
      {capabilities && capabilities.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1">
          {capabilities.slice(0, 3).map((cap) => (
            <span
              key={cap}
              className="px-1 py-0.5 text-xs bg-pink-500/20 text-pink-200 rounded font-mono"
            >
              {cap}
            </span>
          ))}
          {capabilities.length > 3 && (
            <span className="text-xs text-pink-300 font-mono">
              +{capabilities.length - 3}
            </span>
          )}
        </div>
      )}

      {/* Execution status */}
      {executionStatus && (
        <div className={`mt-2 flex items-center gap-1.5 text-xs font-mono ${
          executionStatus === 'running' ? 'text-amber-400' :
          executionStatus === 'completed' ? 'text-emerald-400' :
          executionStatus === 'failed' ? 'text-red-400' : 'text-text-muted'
        }`}>
          {executionStatus === 'running' && (
            <span className="inline-block w-2 h-2 border border-current border-t-transparent rounded-full animate-spin" />
          )}
          {executionStatus === 'completed' && <span>✓</span>}
          {executionStatus === 'failed' && <span>✗</span>}
          <span>{executionStatus}</span>
          {executionDuration && <span className="text-text-muted">({executionDuration})</span>}
        </div>
      )}
      {executionAgent && executionStatus && (
        <div className="text-xs text-text-muted mt-0.5">
          via {executionAgent}
        </div>
      )}

      {/* Lock indicator */}
      {lockedBy && (
        <div className="mt-1 text-xs text-amber-400">
          Locked by {lockedBy}
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
