'use client';

import { STAGE_COLORS, type DAGStage } from '@/hooks/useUnifiedDAG';

const LANES: { stage: DAGStage; label: string }[] = [
  { stage: 'ideas', label: 'Ideas' },
  { stage: 'goals', label: 'Goals' },
  { stage: 'actions', label: 'Actions' },
  { stage: 'orchestration', label: 'Orchestration' },
];

const LANE_WIDTH = 280;

export function DAGStageLanes() {
  return (
    <div className="absolute inset-0 pointer-events-none flex" style={{ zIndex: 0 }}>
      {LANES.map(({ stage, label }, i) => (
        <div
          key={stage}
          className="h-full flex flex-col items-center"
          style={{
            width: LANE_WIDTH,
            marginLeft: i === 0 ? 0 : 20,
            background: `${STAGE_COLORS[stage]}08`,
            borderRight: `1px dashed ${STAGE_COLORS[stage]}30`,
          }}
        >
          <span
            className="mt-2 px-3 py-1 rounded-full text-xs font-mono font-bold uppercase tracking-wider"
            style={{ color: STAGE_COLORS[stage], background: `${STAGE_COLORS[stage]}15` }}
          >
            {label}
          </span>
        </div>
      ))}
    </div>
  );
}
