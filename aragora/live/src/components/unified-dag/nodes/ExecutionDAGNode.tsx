'use client';

import { Handle, Position, type NodeProps } from '@xyflow/react';
import { STAGE_COLORS, type DAGNodeData, type DAGStage } from '@/hooks/useUnifiedDAG';

interface ExecutionDAGNodeProps extends NodeProps {
  onExecuteNode?: (nodeId: string) => void;
}

const STATUS_CONFIG: Record<string, { ring: string; bg: string; label: string; pulse: boolean }> = {
  pending: { ring: '#6b7280', bg: 'bg-gray-500/10', label: 'Pending', pulse: false },
  ready: { ring: '#3b82f6', bg: 'bg-blue-500/12', label: 'Ready', pulse: false },
  running: { ring: '#f59e0b', bg: 'bg-amber-500/14', label: 'Running', pulse: true },
  succeeded: { ring: '#10b981', bg: 'bg-emerald-500/14', label: 'Done', pulse: false },
  failed: { ring: '#ef4444', bg: 'bg-red-500/14', label: 'Failed', pulse: false },
  blocked: { ring: '#64748b', bg: 'bg-slate-500/10', label: 'Blocked', pulse: false },
  awaiting_human: { ring: '#d946ef', bg: 'bg-fuchsia-500/12', label: 'Waiting', pulse: false },
};

const STAGE_ICONS: Record<DAGStage, string> = {
  ideas: '\u2726',
  principles: '\u2696',
  goals: '\u25ce',
  actions: '\u2611',
  orchestration: '\u2699',
};

export function ExecutionDAGNode({ id, data, selected, onExecuteNode }: ExecutionDAGNodeProps) {
  const nodeData = data as unknown as DAGNodeData;
  const stage = nodeData.stage || 'ideas';
  const status = nodeData.status || 'pending';
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.pending;
  const stageColor = STAGE_COLORS[stage] || '#6366f1';
  const icon = STAGE_ICONS[stage];
  const metadata = nodeData.metadata || {};
  const blockedBy = Array.isArray(nodeData.blockedBy)
    ? nodeData.blockedBy.filter((nodeId): nodeId is string => typeof nodeId === 'string')
    : [];
  const agentPool = Array.isArray(metadata.agent_pool)
    ? metadata.agent_pool.filter((agent): agent is string => typeof agent === 'string')
    : [];
  const assignedAgents = nodeData.assignedAgent
    ? [nodeData.assignedAgent, ...agentPool.filter((agent) => agent !== nodeData.assignedAgent)]
    : agentPool;
  const confidence =
    typeof nodeData.confidence === 'number' && Number.isFinite(nodeData.confidence)
      ? Math.round(nodeData.confidence * 100)
      : null;
  const progress =
    typeof nodeData.progress === 'number' && Number.isFinite(nodeData.progress)
      ? Math.min(100, Math.max(0, nodeData.progress))
      : 0;
  const canExecute =
    typeof nodeData.canExecute === 'boolean' ? nodeData.canExecute : typeof onExecuteNode === 'function';
  const showRunButton = canExecute && (status === 'ready' || status === 'failed') && typeof onExecuteNode === 'function';
  const outputPreview =
    typeof nodeData.outputPreview === 'string' && nodeData.outputPreview.trim()
      ? nodeData.outputPreview
      : null;

  return (
    <div
      className={`relative rounded-xl border shadow-lg min-w-[220px] max-w-[260px] transition-all ${
        selected ? 'ring-2 ring-indigo-400/60' : ''
      } ${cfg.bg} ${cfg.pulse ? 'animate-pulse' : ''}`}
      style={{ borderColor: `${stageColor}66`, boxShadow: `0 12px 36px ${stageColor}16` }}
      data-testid={`dag-node-${id}`}
    >
      <Handle type="target" position={Position.Left} className="!bg-indigo-400" />

      <div
        className="flex items-center gap-2 px-3 py-2 rounded-t-xl text-[10px] font-mono font-bold uppercase tracking-[0.18em]"
        style={{ background: `${stageColor}18`, color: stageColor }}
      >
        <span>{icon}</span>
        <span>{nodeData.stageLabel ?? stage}</span>
        {nodeData.subtype ? (
          <span className="ml-auto rounded-full border border-current/20 px-1.5 py-0.5 text-[9px] opacity-80">
            {nodeData.subtype}
          </span>
        ) : null}
      </div>

      <div className="px-3 py-3 space-y-2.5">
        <div className="text-sm font-semibold text-text leading-snug" title={nodeData.label}>
          {nodeData.label}
        </div>

        {nodeData.description ? (
          <div className="text-[11px] text-text-muted line-clamp-3">
            {nodeData.description}
          </div>
        ) : null}

        <div className="flex flex-wrap gap-1.5 text-[10px] font-mono text-text-muted">
          {nodeData.dependencyCount > 0 ? (
            <span className="rounded-full border border-border/70 px-2 py-0.5">
              {nodeData.dependencyCount} dep{nodeData.dependencyCount === 1 ? '' : 's'}
            </span>
          ) : null}
          {nodeData.downstreamCount > 0 ? (
            <span className="rounded-full border border-border/70 px-2 py-0.5">
              {nodeData.downstreamCount} downstream
            </span>
          ) : null}
          {confidence !== null ? (
            <span className="rounded-full border border-emerald-500/30 px-2 py-0.5 text-emerald-300">
              {confidence}% conf
            </span>
          ) : null}
        </div>

        {assignedAgents.length > 0 ? (
          <div className="flex flex-wrap gap-1">
            {assignedAgents.slice(0, 3).map((agent) => (
              <span
                key={agent}
                className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-pink-500/15 text-pink-300 border border-pink-500/20"
              >
                {agent}
              </span>
            ))}
            {assignedAgents.length > 3 ? (
              <span className="text-[10px] font-mono text-text-muted">+{assignedAgents.length - 3}</span>
            ) : null}
          </div>
        ) : null}

        {outputPreview ? (
          <div className="rounded-lg border border-border/60 bg-bg/45 px-2.5 py-2 text-[11px] text-text-muted line-clamp-2">
            {outputPreview}
          </div>
        ) : null}

        {status === 'blocked' && blockedBy.length > 0 ? (
          <div className="text-[10px] font-mono text-slate-300">
            Waiting on {blockedBy.length} upstream node{blockedBy.length === 1 ? '' : 's'}
          </div>
        ) : null}
      </div>

      <div className="border-t border-border/70 px-3 py-2">
        <div className="flex items-center gap-2">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: cfg.ring }}
            data-testid={`status-${status}`}
          />
          <span className="text-[10px] font-mono text-text-muted">{cfg.label}</span>
          {showRunButton ? (
            <button
              onClick={(event) => {
                event.stopPropagation();
                onExecuteNode?.(id);
              }}
              className="ml-auto px-2 py-0.5 text-[10px] font-mono rounded bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-600/40 transition-colors"
              title={status === 'failed' ? 'Retry execution' : 'Execute this node'}
              data-testid={`run-btn-${id}`}
            >
              {status === 'failed' ? 'Retry' : 'Run'}
            </button>
          ) : null}
        </div>
        {status === 'running' && progress > 0 ? (
          <div className="mt-2 h-1.5 rounded-full bg-gray-700/60 overflow-hidden">
            <div
              className="h-full rounded-full bg-amber-400 transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        ) : null}
      </div>

      <Handle type="source" position={Position.Right} className="!bg-indigo-400" />
    </div>
  );
}
