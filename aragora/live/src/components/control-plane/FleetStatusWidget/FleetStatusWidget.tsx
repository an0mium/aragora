'use client';

import { useMemo, useCallback } from 'react';
import { FleetHealthGauge } from './FleetHealthGauge';

export type AgentStatus = 'idle' | 'busy' | 'offline' | 'error' | 'rate_limited';

export interface FleetAgent {
  id: string;
  name: string;
  model: string;
  status: AgentStatus;
  current_task_id?: string;
  last_heartbeat?: string;
}

export interface FleetStatusWidgetProps {
  /** List of agents in the fleet */
  agents: FleetAgent[];
  /** Optional: Count of running tasks */
  runningTasks?: number;
  /** Optional: Count of queued tasks */
  queuedTasks?: number;
  /** Callback when clicking the widget (navigate to agent catalog) */
  onViewAgents?: () => void;
  /** Callback when clicking on a specific agent */
  onSelectAgent?: (agent: FleetAgent) => void;
  /** Whether the widget is in compact mode */
  compact?: boolean;
  /** Additional className */
  className?: string;
}

const statusConfig: Record<AgentStatus, { color: string; dotClass: string; label: string }> = {
  idle: { color: 'text-green-400', dotClass: 'bg-green-400', label: 'Idle' },
  busy: { color: 'text-acid-cyan', dotClass: 'bg-acid-cyan animate-pulse', label: 'Busy' },
  offline: { color: 'text-gray-400', dotClass: 'bg-gray-500', label: 'Offline' },
  error: { color: 'text-crimson', dotClass: 'bg-crimson', label: 'Error' },
  rate_limited: { color: 'text-yellow-400', dotClass: 'bg-yellow-400', label: 'Rate Limited' },
};

/**
 * Fleet Status Widget - Quick-view component showing agent fleet overview.
 * Displays agent counts by status, fleet health gauge, and click-through to agent catalog.
 */
export function FleetStatusWidget({
  agents,
  runningTasks = 0,
  queuedTasks = 0,
  onViewAgents,
  onSelectAgent,
  compact = false,
  className = '',
}: FleetStatusWidgetProps) {
  // Calculate fleet statistics
  const stats = useMemo(() => {
    const byStatus: Record<AgentStatus, number> = {
      idle: 0,
      busy: 0,
      offline: 0,
      error: 0,
      rate_limited: 0,
    };

    agents.forEach((agent) => {
      byStatus[agent.status] = (byStatus[agent.status] || 0) + 1;
    });

    const total = agents.length;
    const available = byStatus.idle + byStatus.busy;
    const healthy = total > 0 ? Math.round((available / total) * 100) : 0;

    return {
      total,
      byStatus,
      available,
      healthy,
    };
  }, [agents]);

  // Get busy agents for display
  const busyAgents = useMemo(() => {
    return agents.filter((a) => a.status === 'busy').slice(0, 3);
  }, [agents]);

  const handleWidgetClick = useCallback(() => {
    onViewAgents?.();
  }, [onViewAgents]);

  const handleAgentClick = useCallback(
    (e: React.MouseEvent, agent: FleetAgent) => {
      e.stopPropagation();
      onSelectAgent?.(agent);
    },
    [onSelectAgent]
  );

  if (compact) {
    return (
      <div
        onClick={handleWidgetClick}
        className={`card p-3 cursor-pointer hover:border-acid-green/50 transition-colors ${className}`}
      >
        <div className="flex items-center gap-3">
          <FleetHealthGauge health={stats.healthy} size={48} strokeWidth={4} showLabel={false} />
          <div className="flex-1 min-w-0">
            <div className="text-sm font-mono text-text">
              {stats.byStatus.idle}/{stats.total} agents available
            </div>
            <div className="text-xs text-text-muted">
              {runningTasks} running | {queuedTasks} queued
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`card ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-acid-green animate-pulse" />
          <h3 className="font-mono text-sm text-acid-green">Fleet Status</h3>
        </div>
        {onViewAgents && (
          <button
            onClick={handleWidgetClick}
            className="text-xs text-text-muted hover:text-acid-green transition-colors font-mono"
          >
            View All &rarr;
          </button>
        )}
      </div>

      {/* Content */}
      <div className="p-4">
        <div className="flex items-start gap-6">
          {/* Health Gauge */}
          <FleetHealthGauge health={stats.healthy} size={90} strokeWidth={6} />

          {/* Agent counts by status */}
          <div className="flex-1 grid grid-cols-2 gap-2">
            <div className="bg-surface p-2 rounded">
              <div className="flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full ${statusConfig.idle.dotClass}`} />
                <span className="text-xs text-text-muted font-mono">Idle</span>
              </div>
              <div className="text-lg font-mono text-green-400">{stats.byStatus.idle}</div>
            </div>
            <div className="bg-surface p-2 rounded">
              <div className="flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full ${statusConfig.busy.dotClass}`} />
                <span className="text-xs text-text-muted font-mono">Busy</span>
              </div>
              <div className="text-lg font-mono text-acid-cyan">{stats.byStatus.busy}</div>
            </div>
            <div className="bg-surface p-2 rounded">
              <div className="flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full ${statusConfig.offline.dotClass}`} />
                <span className="text-xs text-text-muted font-mono">Offline</span>
              </div>
              <div className="text-lg font-mono text-gray-400">{stats.byStatus.offline}</div>
            </div>
            <div className="bg-surface p-2 rounded">
              <div className="flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full ${statusConfig.error.dotClass}`} />
                <span className="text-xs text-text-muted font-mono">Error</span>
              </div>
              <div className="text-lg font-mono text-crimson">{stats.byStatus.error}</div>
            </div>
          </div>
        </div>

        {/* Task summary */}
        <div className="mt-4 pt-4 border-t border-border/50 grid grid-cols-2 gap-3">
          <div className="text-center">
            <div className="text-2xl font-mono text-acid-cyan">{runningTasks}</div>
            <div className="text-xs text-text-muted font-mono">Running Tasks</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-mono text-acid-yellow">{queuedTasks}</div>
            <div className="text-xs text-text-muted font-mono">Queued Tasks</div>
          </div>
        </div>

        {/* Active agents (if any busy) */}
        {busyAgents.length > 0 && (
          <div className="mt-4 pt-4 border-t border-border/50">
            <div className="text-xs text-text-muted font-mono mb-2">ACTIVE AGENTS</div>
            <div className="space-y-2">
              {busyAgents.map((agent) => (
                <div
                  key={agent.id}
                  onClick={(e) => handleAgentClick(e, agent)}
                  className="flex items-center gap-2 p-2 bg-surface rounded cursor-pointer hover:bg-surface/80 transition-colors"
                >
                  <span className="w-2 h-2 rounded-full bg-acid-cyan animate-pulse" />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-mono text-text truncate">{agent.name}</div>
                    <div className="text-xs text-text-muted truncate">
                      {agent.current_task_id || agent.model}
                    </div>
                  </div>
                </div>
              ))}
              {stats.byStatus.busy > 3 && (
                <div className="text-xs text-text-muted text-center font-mono">
                  +{stats.byStatus.busy - 3} more active
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default FleetStatusWidget;
