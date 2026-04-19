import type { MouseEvent } from 'react';

import clsx from 'clsx';

import { BriefPanel } from './BriefPanel';
import type { ReviewQueueDetail, ReviewQueueItem } from './types';
import { formatRelativeAge, laneTone, verdictLabel, verdictTone } from './utils';

interface ReviewQueueCardProps {
  item: ReviewQueueItem;
  selected: boolean;
  expanded: boolean;
  detail: ReviewQueueDetail | null;
  detailLoading: boolean;
  requestChangesOpen: boolean;
  requestChangesDraft: string;
  actionLoading?: 'approve' | 'request_changes' | 'defer' | null;
  onSelect: () => void;
  onToggleExpand: () => void;
  onApprove: () => void;
  onDefer: () => void;
  onOpenDiff: () => void;
  onOpenRequestChanges: () => void;
  onRequestChangesDraftChange: (value: string) => void;
  onRequestChangesSubmit: () => void;
  onRequestChangesCancel: () => void;
}

export function ReviewQueueCard({
  item,
  selected,
  expanded,
  detail,
  detailLoading,
  requestChangesOpen,
  requestChangesDraft,
  actionLoading,
  onSelect,
  onToggleExpand,
  onApprove,
  onDefer,
  onOpenDiff,
  onOpenRequestChanges,
  onRequestChangesDraftChange,
  onRequestChangesSubmit,
  onRequestChangesCancel,
}: ReviewQueueCardProps) {
  const verdict = item.brief?.verdict ?? item.machine_recommendation ?? null;

  return (
    <article
      className={clsx(
        'rounded-2xl border p-4 transition-all',
        selected
          ? 'border-[var(--accent)] bg-[linear-gradient(180deg,rgba(10,18,24,0.96),rgba(10,16,20,0.92))] shadow-[0_18px_50px_rgba(0,0,0,0.24)]'
          : 'border-border bg-surface/70 hover:border-[var(--accent)]/25 hover:bg-surface'
      )}
    >
      <div
        className="w-full text-left"
        onClick={onSelect}
        role="presentation"
      >
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className={clsx('text-[11px] font-theme-data uppercase tracking-[0.18em]', laneTone(item.lane))}>
                {item.lane.replace('_', ' ')}
              </span>
              <span className="text-[11px] font-theme-data uppercase tracking-[0.18em] text-text-muted">
                #{item.number}
              </span>
              <span className="text-[11px] font-theme-data uppercase tracking-[0.18em] text-text-muted">
                {formatRelativeAge(item.created_at)}
              </span>
            </div>
            <h2 className="mt-2 truncate text-xl font-theme-data text-text">{item.title}</h2>
            <p className="mt-2 text-sm font-theme-data text-text-muted">
              {item.lane_reason}
            </p>

            <div className="mt-4 flex flex-wrap gap-2">
              <span className={clsx('rounded-full border px-2.5 py-1 text-[11px] font-theme-data uppercase tracking-[0.14em]', verdictTone(verdict))}>
                {verdictLabel(verdict)}
              </span>
              <span className="rounded-full border border-border px-2.5 py-1 text-[11px] font-theme-data uppercase tracking-[0.14em] text-text-muted">
                {item.checks_summary}
              </span>
              <span className="rounded-full border border-border px-2.5 py-1 text-[11px] font-theme-data uppercase tracking-[0.14em] text-text-muted">
                +{item.additions} / -{item.deletions}
              </span>
              {item.touched_subsystems.slice(0, 2).map((subsystem) => (
                <span
                  key={subsystem}
                  className="rounded-full border border-[var(--accent)]/20 bg-[var(--accent)]/8 px-2.5 py-1 text-[11px] font-theme-data uppercase tracking-[0.14em] text-[var(--accent)]"
                >
                  {subsystem}
                </span>
              ))}
            </div>
          </div>

          <div className="flex flex-col gap-3 lg:min-w-[15rem]">
            <div className="grid grid-cols-3 gap-2 text-center">
              <MiniMetric label="Pass" value={item.status_counts.success} tone="text-emerald-300" />
              <MiniMetric label="Warn" value={item.status_counts.pending} tone="text-[var(--acid-yellow)]" />
              <MiniMetric label="Fail" value={item.status_counts.failure} tone="text-acid-red" />
            </div>

            <div className="flex flex-wrap gap-2">
              <ActionButton
                label="Approve"
                tone="approve"
                busy={actionLoading === 'approve'}
                onClick={(event) => {
                  event.stopPropagation();
                  onApprove();
                }}
              />
              <ActionButton
                label="Request"
                tone="warn"
                busy={actionLoading === 'request_changes'}
                onClick={(event) => {
                  event.stopPropagation();
                  onOpenRequestChanges();
                }}
              />
              <ActionButton
                label="Defer"
                tone="muted"
                busy={actionLoading === 'defer'}
                onClick={(event) => {
                  event.stopPropagation();
                  onDefer();
                }}
              />
              <ActionButton
                label="Diff"
                tone="link"
                busy={false}
                onClick={(event) => {
                  event.stopPropagation();
                  onOpenDiff();
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {requestChangesOpen ? (
        <div className="mt-4 rounded-xl border border-[var(--acid-yellow)]/20 bg-[var(--acid-yellow)]/8 p-4">
          <label className="block text-[11px] font-theme-data uppercase tracking-[0.18em] text-[var(--acid-yellow)]">
            Request changes reason
          </label>
          <textarea
            value={requestChangesDraft}
            onChange={(event) => onRequestChangesDraftChange(event.target.value)}
            className="mt-3 min-h-24 w-full rounded-xl border border-[var(--accent)]/20 bg-bg/70 px-3 py-2 text-sm font-theme-data text-text focus:border-[var(--accent)] focus:outline-none"
            placeholder="Keep the repair loop bounded with one concrete reason."
          />
          <div className="mt-3 flex gap-2">
            <button
              type="button"
              onClick={onRequestChangesSubmit}
              className="rounded-full border border-[var(--acid-yellow)]/35 px-3 py-1.5 text-xs font-theme-data text-[var(--acid-yellow)] hover:bg-[var(--acid-yellow)]/12"
            >
              Submit request
            </button>
            <button
              type="button"
              onClick={onRequestChangesCancel}
              className="rounded-full border border-border px-3 py-1.5 text-xs font-theme-data text-text-muted hover:text-text"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : null}

      {expanded ? (
        <div className="mt-4">
          <BriefPanel detail={detail} loading={detailLoading} />
        </div>
      ) : null}

      <div className="mt-4 flex items-center justify-between border-t border-[var(--accent)]/10 pt-3 text-[11px] font-theme-data uppercase tracking-[0.16em] text-text-muted">
        <span>{item.author || 'unknown author'}</span>
        <button
          type="button"
          onClick={onToggleExpand}
          className="rounded-full border border-border px-3 py-1 text-[11px] font-theme-data text-text-muted hover:border-[var(--accent)]/25 hover:text-text"
        >
          {expanded ? 'Collapse' : 'Expand'}
        </button>
      </div>
    </article>
  );
}

function MiniMetric({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className="rounded-xl border border-[var(--accent)]/10 bg-bg/45 px-3 py-2">
      <div className={clsx('text-lg font-theme-data', tone)}>{value}</div>
      <div className="text-[10px] font-theme-data uppercase tracking-[0.16em] text-text-muted">
        {label}
      </div>
    </div>
  );
}

function ActionButton({
  label,
  tone,
  busy,
  onClick,
}: {
  label: string;
  tone: 'approve' | 'warn' | 'muted' | 'link';
  busy: boolean;
  onClick: (event: MouseEvent<HTMLButtonElement>) => void;
}) {
  const toneClass = {
    approve: 'border-emerald-400/30 text-emerald-300 hover:bg-emerald-400/10',
    warn: 'border-[var(--acid-yellow)]/30 text-[var(--acid-yellow)] hover:bg-[var(--acid-yellow)]/10',
    muted: 'border-border text-text-muted hover:text-text',
    link: 'border-[var(--accent)]/30 text-[var(--accent)] hover:bg-[var(--accent)]/10',
  }[tone];

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      className={clsx(
        'rounded-full border px-3 py-1.5 text-xs font-theme-data transition-colors disabled:cursor-not-allowed disabled:opacity-60',
        toneClass
      )}
    >
      {busy ? 'Working…' : label}
    </button>
  );
}
