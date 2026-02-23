'use client';

import { useSLOStatus } from '@/hooks/useSystemHealth';

export function SLOStatusCards() {
  const { slos, loading } = useSLOStatus();

  if (loading) return <div className="animate-pulse p-4 text-[var(--text-muted)] font-mono">Loading SLOs...</div>;

  return (
    <div className="space-y-3">
      <h3 className="font-mono text-xs text-[var(--text-muted)] uppercase tracking-wider">SLO Compliance</h3>
      {slos.length === 0 ? (
        <p className="text-[var(--text-muted)] font-mono text-sm">No SLOs configured.</p>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {slos.map((s) => (
            <div key={s.name} className={`card p-3 space-y-2 border-l-2 ${s.compliant ? 'border-emerald-400' : 'border-red-400'}`}>
              <span className="font-mono text-xs text-[var(--text)]">{s.name}</span>
              <div className="flex justify-between text-[10px] font-mono">
                <span className="text-[var(--text-muted)]">Target: {(s.target * 100).toFixed(2)}%</span>
                <span className={s.compliant ? 'text-emerald-400' : 'text-red-400'}>{(s.current * 100).toFixed(2)}%</span>
              </div>
              <div className="h-1.5 bg-[var(--bg)] rounded overflow-hidden">
                <div className={`h-full ${s.compliant ? 'bg-emerald-400' : 'bg-red-400'}`} style={{ width: `${Math.min(100, (s.current / s.target) * 100)}%` }} />
              </div>
              <div className="text-[10px] font-mono text-[var(--text-muted)]">
                Burn rate: {s.burn_rate.toFixed(2)}x
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
