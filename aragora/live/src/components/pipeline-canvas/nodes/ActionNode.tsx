'use client';

import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';

const actionTypeLabels: Record<string, string> = {
  task: 'Task',
  epic: 'Epic',
  checkpoint: 'Checkpoint',
  deliverable: 'Deliverable',
  dependency: 'Dependency',
  human_checkpoint: 'Review Gate',
  verification: 'Verification',
};

interface ActionNodeProps {
  data: Record<string, unknown>;
  selected?: boolean;
}

export const ActionNode = memo(function ActionNode({ data, selected }: ActionNodeProps) {
  const stepType = (data.stepType || data.step_type || 'task') as string;
  const label = data.label as string;
  const description = data.description as string | undefined;
  const optional = data.optional as boolean | undefined;
  const timeout = (data.timeoutSeconds || data.timeout) as number | undefined;

  return (
    <div
      className={`
        px-4 py-3 rounded-md border-2 min-w-[190px] max-w-[260px]
        bg-amber-500/20 border-amber-500
        ${selected ? 'ring-2 ring-acid-green ring-offset-2 ring-offset-bg' : ''}
        transition-all duration-200
      `}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="w-3 h-3 bg-amber-500 border-2 border-bg"
      />

      <div className="flex items-center gap-2 mb-2">
        <span className="px-1.5 py-0.5 text-xs bg-amber-500/30 text-amber-200 rounded font-mono uppercase">
          {actionTypeLabels[stepType] || stepType}
        </span>
        {optional && (
          <span className="px-1.5 py-0.5 text-xs bg-gray-500/30 text-gray-300 rounded font-mono">
            optional
          </span>
        )}
      </div>

      <div className="text-sm font-medium text-text mb-1 truncate">
        {label}
      </div>

      {description && (
        <div className="text-xs text-text-muted mb-2 line-clamp-2">
          {description}
        </div>
      )}

      {timeout && timeout > 0 && (
        <div className="text-xs font-mono text-amber-300">
          timeout: {timeout}s
        </div>
      )}

      <Handle
        type="source"
        position={Position.Right}
        className="w-3 h-3 bg-amber-500 border-2 border-bg"
      />
    </div>
  );
});

export default ActionNode;
