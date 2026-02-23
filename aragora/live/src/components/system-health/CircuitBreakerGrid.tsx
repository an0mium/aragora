'use client';

import { useCircuitBreakers } from '@/hooks/useSystemHealth';

const STATE_COLORS: Record<string, string> = {
  closed: 'text-emerald-400 border-emerald-400/40',
  open: 'text-red-400 border-red-400/40',
  half_open: 'text-amber-400 border-amber-400/40',
};

export function CircuitBreakerGrid() {
  const { breakers, loading } = useCircuitBreakers();

  if (loading) return <div className="animate-pulse p-4 text-[var(--text-muted)] font-mono">Loading circuit breakers...</div>;

  return (
    <div className="space-y-3">
      <h3 className="font-mono text-xs text-[var(--text-muted)] uppercase tracking-wider">Circuit Breakers</h3>
      {breakers.length === 0 ? (
        <p className="text-[var(--text-muted)] font-mono text-sm">No circuit breakers registered.</p>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {breakers.map((b) => (
            <div key={b.name} className="card p-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-[var(--text)] truncate">{b.name}</span>
                <span className={`text-[10px] font-mono px-2 py-0.5 border rounded ${STATE_COLORS[b.state] || ''}`}>
                  {b.state.toUpperCase()}
                </span>
              </div>
              <div className="h-1.5 bg-[var(--bg)] rounded overflow-hidden">
                <div className={`h-full transition-all ${b.success_rate > 0.95 ? 'bg-emerald-400' : b.success_rate > 0.8 ? 'bg-amber-400' : 'bg-red-400'}`} style={{ width: `${b.success_rate * 100}%` }} />
              </div>
              <div className="flex justify-between text-[10px] font-mono text-[var(--text-muted)]">
                <span>Failures: {b.failure_count}</span>
                <span>{(b.success_rate * 100).toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
