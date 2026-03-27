'use client';

/**
 * ExecutionSidebar - Right-hand panel for DAG execution controls.
 *
 * Shows graph validation, executable readiness, per-stage status, and
 * recent execution history for the pipeline DAG.
 */

import { useMemo } from 'react';
import type { Node } from '@xyflow/react';
import {
  DAG_STAGE_ORDER,
  STAGE_COLORS,
  type DAGNodeData,
  type DAGStage,
  type DAGStageSummary,
} from '@/hooks/useUnifiedDAG';

export interface ExecutionHistoryEntry {
  id: string;
  nodeId: string;
  nodeLabel: string;
  status: 'succeeded' | 'failed';
  durationMs: number;
  timestamp: number;
}

interface ExecutionSidebarProps {
  nodes: Node<DAGNodeData>[];
  stages?: DAGStageSummary[];
  executing: boolean;
  onExecuteAll: () => void;
  onAutoAdvance: () => void;
  onValidate: () => void;
  validationErrors: string[];
  executionHistory: ExecutionHistoryEntry[];
  onClose: () => void;
}

interface StageSummary {
  stage: DAGStage;
  label: string;
  status: string;
  total: number;
  ready: number;
  running: number;
  succeeded: number;
  failed: number;
  blocked: number;
  pending: number;
  awaitingHuman: number;
}

function computeStageSummaries(
  nodes: Node<DAGNodeData>[],
  stages: DAGStageSummary[] = [],
): StageSummary[] {
  const stageById = Object.fromEntries(stages.map((stage) => [stage.stage, stage])) as Partial<
    Record<DAGStage, DAGStageSummary>
  >;

  return DAG_STAGE_ORDER.map((stage) => {
    const stageNodes = nodes.filter((node) => node.data.stage === stage);
    const counts = {
      ready: 0,
      running: 0,
      succeeded: 0,
      failed: 0,
      blocked: 0,
      pending: 0,
      awaitingHuman: 0,
    };

    for (const node of stageNodes) {
      switch (node.data.status) {
        case 'ready':
          counts.ready += 1;
          break;
        case 'running':
          counts.running += 1;
          break;
        case 'succeeded':
          counts.succeeded += 1;
          break;
        case 'failed':
          counts.failed += 1;
          break;
        case 'blocked':
          counts.blocked += 1;
          break;
        case 'awaiting_human':
          counts.awaitingHuman += 1;
          break;
        default:
          counts.pending += 1;
          break;
      }
    }

    return {
      stage,
      label: stageById[stage]?.label ?? stage,
      status: stageById[stage]?.status ?? 'pending',
      total: stageNodes.length,
      ...counts,
    };
  });
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60_000).toFixed(1)}m`;
}

export function ExecutionSidebar({
  nodes,
  stages = [],
  executing,
  onExecuteAll,
  onAutoAdvance,
  onValidate,
  validationErrors,
  executionHistory,
  onClose,
}: ExecutionSidebarProps) {
  const summaries = useMemo(() => computeStageSummaries(nodes, stages), [nodes, stages]);
  const totalNodes = nodes.length;
  const succeededNodes = nodes.filter((node) => node.data.status === 'succeeded').length;
  const readyNodes = nodes.filter((node) => node.data.status === 'ready').length;
  const executableReadyNodes = nodes.filter(
    (node) => node.data.status === 'ready' && Boolean(node.data.canExecute),
  ).length;
  const completionPct = totalNodes > 0 ? Math.round((succeededNodes / totalNodes) * 100) : 0;

  return (
    <aside
      className="w-80 h-full border-l border-border bg-surface flex-shrink-0 overflow-y-auto"
      data-testid="execution-sidebar"
    >
      <div className="p-4 space-y-5">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-mono font-bold text-text uppercase tracking-wide">
            Execution
          </h3>
          <button
            onClick={onClose}
            className="text-text-muted hover:text-text text-xs font-mono"
            title="Close"
          >
            {'\u00D7'}
          </button>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-text-muted">
            <span>Overall Progress</span>
            <span className="text-text">{completionPct}%</span>
          </div>
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-emerald-500 rounded-full transition-all duration-500"
              style={{ width: `${completionPct}%` }}
              data-testid="progress-bar"
            />
          </div>
          <div className="flex items-center justify-between text-[10px] font-mono text-text-muted">
            <span>{succeededNodes}/{totalNodes} nodes complete</span>
            <span>{readyNodes} ready</span>
          </div>
          <div className="rounded-lg border border-border/60 bg-bg/40 px-3 py-2 text-[10px] font-mono text-text-muted">
            {executableReadyNodes > 0
              ? `${executableReadyNodes} orchestration node(s) ready to execute`
              : 'No orchestration nodes are ready to execute yet'}
          </div>
        </div>

        <div className="space-y-2">
          <h4 className="text-xs font-mono text-text-muted uppercase tracking-wider">
            Stage Progress
          </h4>
          {summaries.map((summary) => {
            const color = STAGE_COLORS[summary.stage];
            const pct = summary.total > 0 ? Math.round((summary.succeeded / summary.total) * 100) : 0;
            return (
              <div key={summary.stage} className="space-y-1.5 rounded-lg border border-border/40 bg-bg/35 p-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span style={{ color }} className="capitalize font-bold">{summary.stage}</span>
                  <span className="text-text-muted">{summary.succeeded}/{summary.total}</span>
                </div>
                <div className="flex items-center justify-between gap-2 text-[10px] font-mono text-text-muted">
                  <span>{summary.status.replace('_', ' ')}</span>
                  {summary.awaitingHuman > 0 ? (
                    <span className="text-fuchsia-300">{summary.awaitingHuman} waiting</span>
                  ) : null}
                </div>
                <div className="h-1.5 bg-gray-700/50 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${pct}%`, background: color }}
                  />
                </div>
                <div className="flex flex-wrap gap-2 text-[9px] font-mono text-text-muted">
                  {summary.running > 0 && <span className="text-amber-400">{summary.running} running</span>}
                  {summary.ready > 0 && <span className="text-blue-400">{summary.ready} ready</span>}
                  {summary.failed > 0 && <span className="text-red-400">{summary.failed} failed</span>}
                  {summary.blocked > 0 && <span>{summary.blocked} blocked</span>}
                </div>
              </div>
            );
          })}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-mono text-text-muted uppercase tracking-wider">
              Validation
            </h4>
            <button
              onClick={onValidate}
              className="px-2 py-0.5 text-[10px] font-mono rounded bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-600/40 transition-colors"
            >
              Check
            </button>
          </div>
          {validationErrors.length === 0 ? (
            <div className="flex items-center gap-1.5 text-xs font-mono text-emerald-400">
              <span>{'\u2713'}</span>
              <span>Graph is valid and executable</span>
            </div>
          ) : (
            <div className="space-y-1">
              {validationErrors.map((error, index) => (
                <div
                  key={`${error}-${index}`}
                  className="flex items-start gap-1.5 text-[11px] font-mono text-red-400"
                >
                  <span className="mt-0.5">{'\u2717'}</span>
                  <span>{error}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="space-y-2 pt-2 border-t border-border">
          <h4 className="text-xs font-mono text-text-muted uppercase tracking-wider">
            Batch Operations
          </h4>
          <button
            onClick={onExecuteAll}
            disabled={executing || executableReadyNodes === 0}
            className="w-full px-3 py-2 text-sm font-mono rounded bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            data-testid="execute-all-btn"
          >
            {executing ? 'Executing...' : `Execute All Ready (${executableReadyNodes})`}
          </button>
          <button
            onClick={onAutoAdvance}
            disabled={executing || totalNodes === 0}
            className="w-full px-3 py-2 text-sm font-mono rounded bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            data-testid="auto-advance-btn"
          >
            {executing ? 'Advancing...' : 'Auto-Advance All Stages'}
          </button>
          <p className="text-[10px] font-mono text-text-muted">
            Validate dependencies before batch execution. Ready orchestration nodes are executed together.
          </p>
        </div>

        {executionHistory.length > 0 ? (
          <div className="space-y-2 pt-2 border-t border-border">
            <h4 className="text-xs font-mono text-text-muted uppercase tracking-wider">
              Recent Executions
            </h4>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {executionHistory.slice(0, 20).map((entry) => (
                <div
                  key={entry.id}
                  className="flex items-center justify-between px-2 py-1 rounded bg-bg/50 text-[11px] font-mono"
                >
                  <div className="flex items-center gap-1.5 truncate">
                    <span
                      className={`inline-block w-1.5 h-1.5 rounded-full ${
                        entry.status === 'succeeded' ? 'bg-emerald-400' : 'bg-red-400'
                      }`}
                    />
                    <span className="text-text truncate">{entry.nodeLabel}</span>
                  </div>
                  <span className="text-text-muted ml-2 flex-shrink-0">
                    {formatDuration(entry.durationMs)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </aside>
  );
}
