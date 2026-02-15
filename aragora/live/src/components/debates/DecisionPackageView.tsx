'use client';

interface DecisionPackage {
  explanation: string;
  agents: string[];
  rounds: number;
  consensus_reached: boolean;
  confidence: number;
  total_cost: number;
  cost_breakdown: Array<{
    agent: string;
    tokens: number;
    cost: number;
  }>;
  next_steps: string[];
  duration_seconds: number;
}

interface DecisionPackageViewProps {
  pkg: DecisionPackage;
}

export function DecisionPackageView({ pkg }: DecisionPackageViewProps) {
  return (
    <div className="space-y-4">
      {/* Explanation Panel */}
      {pkg.explanation && (
        <div className="bg-[var(--surface)] border border-[var(--border)] p-5">
          <div className="text-xs font-mono text-[var(--acid-green)] mb-3">
            {'>'} EXPLANATION
          </div>
          <p className="text-sm font-mono text-[var(--text)] whitespace-pre-wrap leading-relaxed">
            {pkg.explanation}
          </p>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-[var(--surface)] border border-[var(--border)] p-4">
          <div className="text-xs font-mono text-[var(--text-muted)] mb-1">AGENTS</div>
          <div className="text-lg font-mono text-[var(--acid-green)]">{pkg.agents.length}</div>
          <div className="text-xs font-mono text-[var(--text-muted)] mt-1 truncate">
            {pkg.agents.slice(0, 3).join(', ')}
            {pkg.agents.length > 3 ? ` +${pkg.agents.length - 3}` : ''}
          </div>
        </div>
        <div className="bg-[var(--surface)] border border-[var(--border)] p-4">
          <div className="text-xs font-mono text-[var(--text-muted)] mb-1">ROUNDS</div>
          <div className="text-lg font-mono text-[var(--acid-cyan)]">{pkg.rounds}</div>
          <div className="text-xs font-mono text-[var(--text-muted)] mt-1">
            {pkg.consensus_reached ? 'Converged' : 'Divergent'}
          </div>
        </div>
        <div className="bg-[var(--surface)] border border-[var(--border)] p-4">
          <div className="text-xs font-mono text-[var(--text-muted)] mb-1">COST</div>
          <div className="text-lg font-mono text-[var(--text)]">
            ${typeof pkg.total_cost === 'number' ? pkg.total_cost.toFixed(4) : '--'}
          </div>
          <div className="text-xs font-mono text-[var(--text-muted)] mt-1">
            {pkg.cost_breakdown?.length ?? 0} agents billed
          </div>
        </div>
        <div className="bg-[var(--surface)] border border-[var(--border)] p-4">
          <div className="text-xs font-mono text-[var(--text-muted)] mb-1">DURATION</div>
          <div className="text-lg font-mono text-[var(--text)]">
            {pkg.duration_seconds ? `${Math.round(pkg.duration_seconds)}s` : '--'}
          </div>
          <div className="text-xs font-mono text-[var(--text-muted)] mt-1">
            wall clock
          </div>
        </div>
      </div>

      {/* Participating Agents */}
      {pkg.agents.length > 0 && (
        <div className="bg-[var(--surface)] border border-[var(--border)] p-5">
          <div className="text-xs font-mono text-[var(--acid-green)] mb-3">
            {'>'} PARTICIPATING AGENTS
          </div>
          <div className="flex flex-wrap gap-2">
            {pkg.agents.map((agent, i) => (
              <span
                key={i}
                className="px-2 py-1 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30"
              >
                {agent}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Next Steps */}
      {pkg.next_steps && pkg.next_steps.length > 0 && (
        <div className="bg-[var(--surface)] border border-[var(--acid-cyan)]/30 p-5">
          <div className="text-xs font-mono text-[var(--acid-cyan)] mb-3">
            {'>'} RECOMMENDED NEXT STEPS
          </div>
          <div className="space-y-2">
            {pkg.next_steps.map((step, i) => (
              <div key={i} className="flex items-start gap-2">
                <span className="text-xs font-mono text-[var(--acid-cyan)] mt-0.5">
                  {String(i + 1).padStart(2, '0')}.
                </span>
                <p className="text-sm font-mono text-[var(--text)]">{step}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
