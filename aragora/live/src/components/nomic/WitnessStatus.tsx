'use client';

interface WitnessStatusData {
  patrolling: boolean;
  alert_count: number;
  agents_healthy: number;
  agents_total: number;
  recommendations?: string[];
}

interface WitnessStatusProps {
  status: WitnessStatusData;
}

export function WitnessStatus({ status }: WitnessStatusProps) {
  const healthPct =
    status.agents_total > 0
      ? Math.round((status.agents_healthy / status.agents_total) * 100)
      : 0;

  return (
    <div className="space-y-4">
      {/* Patrol indicator */}
      <div className="flex items-center gap-3">
        <div
          className={`w-3 h-3 rounded-full ${
            status.patrolling
              ? 'bg-acid-green animate-pulse'
              : 'bg-text-muted'
          }`}
        />
        <span className="font-mono text-sm text-text">
          Patrol {status.patrolling ? 'Active' : 'Inactive'}
        </span>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-3">
        {/* Alert count */}
        <div className="bg-surface rounded border border-border p-3 text-center">
          <div
            className={`font-mono text-2xl font-bold ${
              status.alert_count > 0 ? 'text-crimson' : 'text-acid-green'
            }`}
          >
            {status.alert_count}
          </div>
          <div className="font-mono text-[10px] text-text-muted uppercase mt-1">
            Alerts
          </div>
        </div>

        {/* Agent health */}
        <div className="bg-surface rounded border border-border p-3 text-center">
          <div className="font-mono text-2xl font-bold text-acid-cyan">
            {status.agents_healthy}/{status.agents_total}
          </div>
          <div className="font-mono text-[10px] text-text-muted uppercase mt-1">
            Agents ({healthPct}%)
          </div>
        </div>
      </div>

      {/* Recommendations */}
      {status.recommendations && status.recommendations.length > 0 && (
        <div className="border-t border-border pt-3">
          <div className="font-mono text-[10px] text-text-muted uppercase mb-2">
            Recommendations
          </div>
          <ul className="space-y-1">
            {status.recommendations.map((rec, idx) => (
              <li
                key={idx}
                className="font-mono text-xs text-acid-yellow flex items-start gap-2"
              >
                <span className="text-text-muted shrink-0">{'>'}</span>
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
