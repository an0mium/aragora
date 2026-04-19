import clsx from 'clsx';

interface StatsHeaderProps {
  pendingCount: number;
  medianDecisionSeconds: number;
  streak: number;
  approvalsToday: number;
  sortBy: 'age' | 'risk' | 'subsystem';
  onSortChange: (value: 'age' | 'risk' | 'subsystem') => void;
}

const SORT_OPTIONS: Array<{ value: 'age' | 'risk' | 'subsystem'; label: string }> = [
  { value: 'age', label: 'Age' },
  { value: 'risk', label: 'Risk' },
  { value: 'subsystem', label: 'Subsystem' },
];

export function StatsHeader({
  pendingCount,
  medianDecisionSeconds,
  streak,
  approvalsToday,
  sortBy,
  onSortChange,
}: StatsHeaderProps) {
  return (
    <section className="rounded-2xl border border-[var(--accent)]/25 bg-[radial-gradient(circle_at_top_left,rgba(0,255,163,0.12),transparent_42%),linear-gradient(180deg,rgba(14,22,28,0.94),rgba(10,14,19,0.94))] p-5 shadow-[0_24px_60px_rgba(0,0,0,0.28)]">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div className="space-y-2">
          <p className="text-[11px] font-theme-data uppercase tracking-[0.28em] text-[var(--accent)]">
            Morning Brief
          </p>
          <h1 className="text-3xl font-theme-data text-text">
            Review Queue
          </h1>
          <p className="max-w-2xl text-sm font-theme-data text-text-muted">
            Brief first, evidence second, diff third. Clear the morning tranche without living in GitHub tabs.
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Pending" value={String(pendingCount)} tone="text-[var(--accent)]" />
          <MetricCard
            label="Median Decision"
            value={`${medianDecisionSeconds.toFixed(0)}s`}
            tone="text-[var(--acid-cyan)]"
          />
          <MetricCard label="Streak" value={String(streak)} tone="text-[var(--acid-yellow)]" />
          <MetricCard label="Approved Today" value={String(approvalsToday)} tone="text-emerald-300" />
        </div>
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-[var(--accent)]/15 pt-4">
        <span className="text-[11px] font-theme-data uppercase tracking-[0.24em] text-text-muted">
          Sort
        </span>
        {SORT_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onSortChange(option.value)}
            className={clsx(
              'rounded-full border px-3 py-1 text-xs font-theme-data transition-colors',
              sortBy === option.value
                ? 'border-[var(--accent)] bg-[var(--accent)]/15 text-[var(--accent)]'
                : 'border-border bg-bg/50 text-text-muted hover:border-[var(--accent)]/30 hover:text-text'
            )}
          >
            {option.label}
          </button>
        ))}
      </div>
    </section>
  );
}

function MetricCard({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: string;
}) {
  return (
    <div className="min-w-[8rem] rounded-xl border border-[var(--accent)]/15 bg-bg/45 px-4 py-3">
      <div className={clsx('text-2xl font-theme-data', tone)}>{value}</div>
      <div className="mt-1 text-[11px] font-theme-data uppercase tracking-[0.18em] text-text-muted">
        {label}
      </div>
    </div>
  );
}
