'use client';

import { useSystemHealth } from '@/hooks/useSystemHealth';

const STATUS_CONFIG: Record<string, { color: string; bg: string; label: string }> = {
  healthy: { color: 'text-emerald-400', bg: 'bg-emerald-400', label: 'ALL SYSTEMS OPERATIONAL' },
  degraded: { color: 'text-amber-400', bg: 'bg-amber-400', label: 'DEGRADED PERFORMANCE' },
  critical: { color: 'text-red-400', bg: 'bg-red-400', label: 'CRITICAL ISSUES' },
};

export function SystemHealthSummary() {
  const { overallStatus, subsystems, lastCheck, loading } = useSystemHealth();

  if (loading) return <div className="animate-pulse p-4 text-[var(--text-muted)] font-mono">Loading system health...</div>;

  const config = STATUS_CONFIG[overallStatus] || STATUS_CONFIG.healthy;
  const entries = Object.entries(subsystems);
  const healthy = entries.filter(([, v]) => v.status === 'healthy').length;
  const degraded = entries.filter(([, v]) => v.status === 'degraded').length;
  const critical = entries.filter(([, v]) => v.status === 'critical').length;

  return (
    <div className="space-y-4">
      <div className="card p-6 flex items-center gap-4">
        <div className={`w-4 h-4 rounded-full ${config.bg} animate-pulse`} />
        <div>
          <h2 className={`font-mono text-lg font-bold ${config.color}`}>{config.label}</h2>
          <p className="font-mono text-xs text-[var(--text-muted)]">Last check: {lastCheck || 'N/A'}</p>
        </div>
      </div>

      <div className="flex gap-4 font-mono text-xs">
        <span className="text-emerald-400">{healthy} healthy</span>
        <span className="text-amber-400">{degraded} degraded</span>
        <span className="text-red-400">{critical} critical</span>
      </div>

      {entries.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {entries.map(([name, info]) => (
            <div key={name} className={`card p-3 border-l-2 ${info.status === 'healthy' ? 'border-emerald-400' : info.status === 'degraded' ? 'border-amber-400' : 'border-red-400'}`}>
              <span className="font-mono text-xs text-[var(--text)]">{name}</span>
              <span className={`block text-[10px] font-mono ${info.status === 'healthy' ? 'text-emerald-400' : info.status === 'degraded' ? 'text-amber-400' : 'text-red-400'}`}>{info.detail}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
