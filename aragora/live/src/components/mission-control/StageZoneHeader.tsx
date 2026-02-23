'use client';

import { memo } from 'react';
import {
  PIPELINE_STAGE_CONFIG,
  STAGE_COLOR_CLASSES,
  type PipelineStageType,
} from '../pipeline-canvas/types';

export interface StageZoneHeaderProps {
  stage: PipelineStageType;
  nodeCount: number;
  status: 'pending' | 'active' | 'complete' | 'error';
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

const statusIndicators: Record<string, { dot: string; label: string }> = {
  pending: { dot: 'bg-gray-400', label: 'Pending' },
  active: { dot: 'bg-blue-400 animate-pulse', label: 'Active' },
  complete: { dot: 'bg-emerald-400', label: 'Complete' },
  error: { dot: 'bg-red-400', label: 'Error' },
};

export const StageZoneHeader = memo(function StageZoneHeader({
  stage,
  nodeCount,
  status,
  collapsed,
  onToggleCollapse,
}: StageZoneHeaderProps) {
  const config = PIPELINE_STAGE_CONFIG[stage];
  const colors = STAGE_COLOR_CLASSES[stage];
  const statusInfo = statusIndicators[status] || statusIndicators.pending;

  return (
    <div
      className={`
        flex items-center gap-2 px-3 py-2 rounded-t-lg border-b-2
        ${colors.bg} ${colors.border}
        select-none
      `}
      data-testid={`stage-zone-header-${stage}`}
    >
      <span className={`text-base ${colors.text}`}>{config.icon}</span>

      <span className={`text-xs font-mono font-bold uppercase tracking-wide ${colors.text}`}>
        {config.label}
      </span>

      <span
        className={`px-1.5 py-0.5 text-xs font-mono rounded-full ${colors.bg} ${colors.text}`}
        data-testid={`stage-zone-count-${stage}`}
      >
        {nodeCount}
      </span>

      <span className={`w-2 h-2 rounded-full ${statusInfo.dot}`} title={statusInfo.label} />

      {onToggleCollapse && (
        <button
          onClick={onToggleCollapse}
          className={`ml-auto text-xs ${colors.text} hover:opacity-80 transition-opacity`}
          aria-label={collapsed ? `Expand ${config.label}` : `Collapse ${config.label}`}
          data-testid={`stage-zone-collapse-${stage}`}
        >
          {collapsed ? '\u25B6' : '\u25BC'}
        </button>
      )}
    </div>
  );
});

export default StageZoneHeader;
