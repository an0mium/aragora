'use client';

import {
  DAG_STAGE_LABELS,
  DAG_STAGE_ORDER,
  STAGE_COLORS,
  STAGE_LANE_GAP,
  STAGE_LANE_WIDTH,
  type DAGStage,
  type DAGStageSummary,
} from '@/hooks/useUnifiedDAG';

interface DAGStageLanesProps {
  stages?: DAGStageSummary[];
  activeStage?: DAGStage | null;
}

const STATUS_TONE: Record<string, string> = {
  pending: 'text-text-muted border-border/60 bg-surface/40',
  in_progress: 'text-amber-300 border-amber-500/30 bg-amber-500/10',
  complete: 'text-emerald-300 border-emerald-500/30 bg-emerald-500/10',
  partial: 'text-sky-300 border-sky-500/30 bg-sky-500/10',
  failed: 'text-red-300 border-red-500/30 bg-red-500/10',
  awaiting_human: 'text-fuchsia-300 border-fuchsia-500/30 bg-fuchsia-500/10',
};

export function DAGStageLanes({ stages = [], activeStage = null }: DAGStageLanesProps) {
  const stageById = Object.fromEntries(stages.map((stage) => [stage.stage, stage])) as Partial<
    Record<DAGStage, DAGStageSummary>
  >;

  return (
    <div className="absolute inset-0 pointer-events-none flex" style={{ zIndex: 0 }}>
      {DAG_STAGE_ORDER.map((stage, index) => {
        const stageSummary = stageById[stage];
        const label = stageSummary?.label ?? DAG_STAGE_LABELS[stage];
        const stageStatus = stageSummary?.status ?? 'pending';
        const dependencyLabel = (stageSummary?.dependency_stage_ids ?? []).join(' -> ');
        const isDimmed = activeStage !== null && activeStage !== stage;

        return (
          <div
            key={stage}
            className="h-full flex flex-col transition-opacity"
            style={{
              width: STAGE_LANE_WIDTH,
              marginLeft: index === 0 ? 0 : STAGE_LANE_GAP,
              background: `${STAGE_COLORS[stage]}0a`,
              borderLeft: `1px solid ${STAGE_COLORS[stage]}18`,
              borderRight: `1px dashed ${STAGE_COLORS[stage]}24`,
              opacity: isDimmed ? 0.32 : 1,
            }}
            data-testid={`stage-lane-${stage}`}
          >
            <div className="m-3 rounded-xl border border-border/60 bg-surface/70 px-3 py-2 backdrop-blur-sm">
              <div className="flex items-center justify-between gap-2">
                <span
                  className="text-xs font-mono font-bold uppercase tracking-[0.2em]"
                  style={{ color: STAGE_COLORS[stage] }}
                >
                  {label}
                </span>
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-mono uppercase ${STATUS_TONE[stageStatus] ?? STATUS_TONE.pending}`}
                >
                  {stageStatus.replace('_', ' ')}
                </span>
              </div>
              <div className="mt-2 flex items-center justify-between text-[10px] font-mono text-text-muted">
                <span>{stageSummary?.node_count ?? 0} nodes</span>
                <span>{stageSummary?.status_counts?.succeeded ?? 0} done</span>
              </div>
              {dependencyLabel ? (
                <div className="mt-2 text-[10px] font-mono text-text-muted">
                  Depends on {dependencyLabel}
                </div>
              ) : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}
