'use client';

import type { ReviewQueueStats } from '@/hooks/useReviewQueue';
import { formatDecisionSeconds } from './format';

export interface StatsHeaderProps {
  visible: number;
  total: number;
  deferredCount: number;
  stats: ReviewQueueStats | null;
  degraded?: boolean;
  reason?: string;
}

export function StatsHeader({
  visible,
  total,
  deferredCount,
  stats,
  degraded,
  reason,
}: StatsHeaderProps) {
  const median = stats?.median_decision_seconds ?? null;
  const streak = stats?.streak ?? 0;
  const approved = stats?.approved ?? 0;

  const statBoxStyle = {
    backgroundColor: 'var(--surface)',
    borderColor: 'var(--border)',
  };

  return (
    <header
      data-testid="review-queue-stats-header"
      className="mb-10 flex flex-col gap-3"
    >
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <div
          className="flex min-h-[5.5rem] flex-col justify-center rounded-xl border px-5 py-4"
          style={statBoxStyle}
        >
          <div
            className="text-[10px] uppercase tracking-wider"
            style={{ color: 'var(--text-muted)' }}
          >
            In queue
          </div>
          <div
            className="mt-2 font-theme-data text-2xl leading-none"
            style={{ color: 'var(--accent)' }}
            data-testid="review-queue-visible"
          >
            {visible}
          </div>
          {deferredCount > 0 ? (
            <div
              className="mt-2 text-[11px]"
              style={{ color: 'var(--text-muted)' }}
              data-testid="review-queue-deferred-count"
            >
              {deferredCount} deferred
            </div>
          ) : (
            <div
              className="mt-2 text-[11px]"
              style={{ color: 'var(--text-muted)' }}
            >
              {total} total
            </div>
          )}
        </div>

        <div
          className="flex min-h-[5.5rem] flex-col justify-center rounded-xl border px-5 py-4"
          style={statBoxStyle}
        >
          <div
            className="text-[10px] uppercase tracking-wider"
            style={{ color: 'var(--text-muted)' }}
          >
            Median decision
          </div>
          <div
            className="mt-2 font-theme-data text-2xl leading-none"
            style={{ color: 'var(--text)' }}
            data-testid="review-queue-median"
          >
            {formatDecisionSeconds(median)}
          </div>
        </div>

        <div
          className="flex min-h-[5.5rem] flex-col justify-center rounded-xl border px-5 py-4"
          style={statBoxStyle}
        >
          <div
            className="text-[10px] uppercase tracking-wider"
            style={{ color: 'var(--text-muted)' }}
          >
            Streak
          </div>
          <div
            className="mt-2 font-theme-data text-2xl leading-none"
            style={{ color: streak > 0 ? 'var(--accent)' : 'var(--text-muted)' }}
            data-testid="review-queue-streak"
          >
            {streak}
          </div>
        </div>

        <div
          className="flex min-h-[5.5rem] flex-col justify-center rounded-xl border px-5 py-4"
          style={statBoxStyle}
        >
          <div
            className="text-[10px] uppercase tracking-wider"
            style={{ color: 'var(--text-muted)' }}
          >
            Approved today
          </div>
          <div
            className="mt-2 font-theme-data text-2xl leading-none"
            style={{ color: 'var(--text)' }}
            data-testid="review-queue-approved-today"
          >
            {approved}
          </div>
        </div>

        <div
          className="col-span-2 flex min-h-[5.5rem] items-center justify-center rounded-xl border px-5 py-4 sm:col-span-1"
          style={statBoxStyle}
        >
          <div
            className="flex items-center gap-2 text-xs"
            style={{ color: 'var(--text-muted)' }}
          >
            <kbd
              className="rounded-md border font-theme-data"
              style={{
                padding: '0.25rem 0.625rem',
                fontSize: '11px',
                borderColor: 'var(--border)',
                backgroundColor: 'var(--surface-elevated)',
                color: 'var(--text)',
              }}
            >
              ?
            </kbd>
            <span>shortcuts</span>
          </div>
        </div>
      </div>

      {degraded && (
        <div
          role="alert"
          data-testid="review-queue-degraded"
          className="rounded-xl border px-4 py-3 text-xs"
          style={{
            borderColor: 'var(--warning)',
            backgroundColor: 'rgba(255, 255, 0, 0.06)',
            color: 'var(--warning)',
          }}
        >
          Queue running in degraded mode: {reason || 'gh CLI unavailable'}.
        </div>
      )}
    </header>
  );
}
