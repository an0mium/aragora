'use client';

import React from 'react';

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  uptime_seconds: number;
  version: string;
  components: {
    name: string;
    status: 'healthy' | 'degraded' | 'unhealthy';
    latency_ms?: number;
    error?: string;
  }[];
}

export interface HealthOverviewProps {
  health: SystemHealth | null;
  loading?: boolean;
}

const STATUS_CONFIG = {
  healthy: { color: 'bg-success', text: 'text-success', label: 'HEALTHY' },
  degraded: { color: 'bg-acid-yellow', text: 'text-acid-yellow', label: 'DEGRADED' },
  unhealthy: { color: 'bg-crimson', text: 'text-crimson', label: 'UNHEALTHY' },
} as const;

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const mins = Math.floor((seconds % 3600) / 60);

  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}

export function HealthOverview({ health, loading = false }: HealthOverviewProps) {
  if (loading) {
    return (
      <div className="bg-surface border border-acid-green/30 p-4 animate-pulse">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-3 h-3 rounded-full bg-acid-green/20" />
          <div className="w-24 h-4 bg-acid-green/10 rounded" />
        </div>
        <div className="grid grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-16 bg-bg rounded" />
          ))}
        </div>
      </div>
    );
  }

  const displayHealth = health ?? {
    status: 'healthy' as const,
    uptime_seconds: 0,
    version: 'unknown',
    components: [],
  };

  const statusConfig = STATUS_CONFIG[displayHealth.status];
  const healthyCount = displayHealth.components.filter(
    (c) => c.status === 'healthy'
  ).length;

  return (
    <div className="bg-surface border border-acid-green/30 p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className={`w-3 h-3 rounded-full ${statusConfig.color} animate-pulse`} />
          <span className={`text-sm font-mono font-bold ${statusConfig.text}`}>
            {statusConfig.label}
          </span>
        </div>
        <span className="text-xs font-mono text-text-muted">
          v{displayHealth.version}
        </span>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="bg-bg p-3 rounded text-center">
          <div className="text-xl font-mono text-acid-green">
            {formatUptime(displayHealth.uptime_seconds)}
          </div>
          <div className="text-xs font-mono text-text-muted">UPTIME</div>
        </div>
        <div className="bg-bg p-3 rounded text-center">
          <div className="text-xl font-mono text-text">
            {healthyCount}/{displayHealth.components.length}
          </div>
          <div className="text-xs font-mono text-text-muted">COMPONENTS</div>
        </div>
        <div className="bg-bg p-3 rounded text-center">
          <div className={`text-xl font-mono ${statusConfig.text}`}>
            {displayHealth.status === 'healthy' ? 'OK' : displayHealth.status.toUpperCase()}
          </div>
          <div className="text-xs font-mono text-text-muted">OVERALL</div>
        </div>
      </div>

      {/* Components list */}
      {displayHealth.components.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-mono text-text-muted uppercase mb-2">
            COMPONENT STATUS
          </div>
          {displayHealth.components.map((component) => {
            const compStatus = STATUS_CONFIG[component.status];
            return (
              <div
                key={component.name}
                className="flex items-center justify-between bg-bg p-2 rounded"
              >
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${compStatus.color}`} />
                  <span className="text-sm font-mono text-text">
                    {component.name}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  {component.latency_ms !== undefined && (
                    <span className="text-xs font-mono text-text-muted">
                      {component.latency_ms}ms
                    </span>
                  )}
                  <span className={`text-xs font-mono ${compStatus.text}`}>
                    {component.status}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
