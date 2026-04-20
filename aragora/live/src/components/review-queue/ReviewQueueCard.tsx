'use client';

import { useCallback, useEffect, useState } from 'react';
import type { ReviewQueuePR, SettlementAction } from '@/hooks/useReviewQueue';
import { BriefPanel } from './BriefPanel';
import {
  ciGlyph,
  formatAge,
  toneColor,
  verdictGlyph,
} from './format';
import { fetchBrief, type ReviewQueueBrief } from '@/hooks/useReviewQueue';

export interface ReviewQueueCardProps {
  pr: ReviewQueuePR;
  selected: boolean;
  expanded: boolean;
  onSelect: () => void;
  onToggleExpand: () => void;
  onSettle: (action: SettlementAction, options?: { note?: string; reason?: string }) => Promise<void>;
}

export function ReviewQueueCard({
  pr,
  selected,
  expanded,
  onSelect,
  onToggleExpand,
  onSettle,
}: ReviewQueueCardProps) {
  const ci = ciGlyph(pr.ci);
  const verdict = verdictGlyph(pr.brief_present, pr.verdict);
  const [pendingAction, setPendingAction] = useState<SettlementAction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reasonDraft, setReasonDraft] = useState('');
  const [reasonForAction, setReasonForAction] = useState<SettlementAction | null>(null);
  const [brief, setBrief] = useState<ReviewQueueBrief | null>(null);
  const [briefLoading, setBriefLoading] = useState(false);
  const [briefFetched, setBriefFetched] = useState(false);
  const [briefError, setBriefError] = useState<string | null>(null);
  const [hovered, setHovered] = useState(false);

  const loadBrief = useCallback(async () => {
    if (briefFetched || briefLoading) return;
    setBriefLoading(true);
    setBriefError(null);
    try {
      const data = await fetchBrief(pr.number);
      setBrief(data);
      setBriefFetched(true);
    } catch (err) {
      setBriefError((err as Error).message || 'Failed to load brief');
    } finally {
      setBriefLoading(false);
    }
  }, [briefFetched, briefLoading, pr.number]);

  const handleExpand = useCallback(() => {
    onToggleExpand();
  }, [onToggleExpand]);

  useEffect(() => {
    if (expanded) {
      void loadBrief();
    }
  }, [expanded, loadBrief]);

  const runSettle = useCallback(
    async (action: SettlementAction, options?: { note?: string; reason?: string }) => {
      setError(null);
      setPendingAction(action);
      try {
        await onSettle(action, options);
      } catch (err) {
        setError((err as Error).message || 'action failed');
      } finally {
        setPendingAction(null);
      }
    },
    [onSettle],
  );

  const handleApprove = useCallback(() => {
    if (pr.brief_present && pr.verdict && pr.verdict !== 'approve_candidate') {
      const ok = typeof window !== 'undefined'
        ? window.confirm(
            `Brief verdict is ${pr.verdict}. Approve anyway?`,
          )
        : true;
      if (!ok) return;
    } else if (!pr.brief_present) {
      const ok = typeof window !== 'undefined'
        ? window.confirm('No brief on file. Approve without PDB brief?')
        : true;
      if (!ok) return;
    }
    void runSettle('approve');
  }, [pr.brief_present, pr.verdict, runSettle]);

  const handleRequestChanges = useCallback(() => {
    setReasonForAction('request-changes');
    setReasonDraft('');
  }, []);

  const handleDefer = useCallback(() => {
    void runSettle('defer', { reason: 'deferred from web UI' });
  }, [runSettle]);

  const handleOpenDiff = useCallback(() => {
    if (typeof window === 'undefined' || !pr.url) return;
    window.open(pr.url, '_blank', 'noopener,noreferrer');
  }, [pr.url]);

  const submitReason = useCallback(() => {
    const action = reasonForAction;
    if (!action) return;
    const reason = reasonDraft.trim();
    if (!reason) {
      setError('reason is required');
      return;
    }
    setReasonForAction(null);
    void runSettle(action, { reason });
  }, [reasonDraft, reasonForAction, runSettle]);

  return (
    <article
      data-testid={`review-queue-card-${pr.number}`}
      data-selected={selected ? 'true' : 'false'}
      aria-selected={selected}
      tabIndex={selected ? 0 : -1}
      role="option"
      onClick={onSelect}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="cursor-pointer rounded-xl border text-sm transition-all hover:-translate-y-px"
      style={{
        backgroundColor: 'var(--surface)',
        // Selected wins over hover (thicker accent border + glow).
        // Hover on non-selected cards picks up a subtle accent glow border.
        borderColor: selected
          ? 'var(--accent)'
          : hovered
          ? 'var(--accent)'
          : 'var(--border)',
        borderWidth: selected ? '2px' : '1px',
        padding: selected ? 'calc(1.5rem - 1px)' : '1.5rem',
        boxShadow: selected
          ? '0 0 0 1px var(--accent-glow)'
          : hovered
          ? '0 0 0 1px var(--accent-glow), 0 4px 12px rgba(0, 0, 0, 0.08)'
          : 'var(--shadow-panel)',
      }}
    >
      {/* Headline row: number badge + title + diff stats */}
      <div className="flex items-start gap-5">
        {/* Number badge, tone-colored by verdict if brief exists, else by CI */}
        <div
          className="flex shrink-0 flex-col items-center justify-center rounded-lg px-3 py-2 font-theme-data"
          style={{
            minWidth: '4rem',
            backgroundColor: (pr.brief_present ? verdict.tone : ci.tone) === 'ok' ? 'rgba(57, 255, 20, 0.14)'
              : (pr.brief_present ? verdict.tone : ci.tone) === 'warn' ? 'rgba(218, 165, 32, 0.14)'
              : (pr.brief_present ? verdict.tone : ci.tone) === 'fail' ? 'rgba(255, 0, 64, 0.12)'
              : 'var(--surface-elevated)',
            color: (pr.brief_present ? verdict.tone : ci.tone) === 'ok' ? 'var(--accent)'
              : (pr.brief_present ? verdict.tone : ci.tone) === 'warn' ? 'var(--warning)'
              : (pr.brief_present ? verdict.tone : ci.tone) === 'fail' ? 'var(--crimson)'
              : 'var(--text-muted)',
            border: `1px solid ${(pr.brief_present ? verdict.tone : ci.tone) === 'ok' ? 'rgba(57, 255, 20, 0.25)'
              : (pr.brief_present ? verdict.tone : ci.tone) === 'warn' ? 'rgba(255, 255, 0, 0.25)'
              : (pr.brief_present ? verdict.tone : ci.tone) === 'fail' ? 'rgba(255, 0, 64, 0.25)'
              : 'var(--border)'}`,
          }}
          title={pr.brief_present ? verdict.label : ci.label}
        >
          <div className="text-[10px] uppercase tracking-wider opacity-60">PR</div>
          <div className="text-lg leading-tight">{pr.number}</div>
        </div>

        <div className="flex min-w-0 flex-1 flex-col">
          <div
            className="text-base font-medium leading-snug"
            style={{ color: 'var(--text)' }}
          >
            {pr.title}
          </div>
          <div
            className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs"
            style={{ color: 'var(--text-muted)' }}
          >
            <span>by {pr.author || 'unknown'}</span>
            <span aria-hidden="true">·</span>
            <span>{formatAge(pr.age_seconds)}</span>
            <span aria-hidden="true">·</span>
            <span className={toneColor(ci.tone)} title={ci.label}>
              {ci.glyph} {ci.label}
            </span>
            {pr.brief_present && (
              <>
                <span aria-hidden="true">·</span>
                <span className={toneColor(verdict.tone)} title={verdict.label}>
                  brief: {verdict.glyph} {verdict.label}
                </span>
              </>
            )}
          </div>
        </div>

        <div
          className="shrink-0 text-right font-theme-data text-xs leading-tight"
          style={{ color: 'var(--text-muted)' }}
        >
          <div>
            <span style={{ color: 'var(--accent)' }}>+{pr.additions}</span>
          </div>
          <div>
            <span style={{ color: 'var(--crimson)' }}>−{pr.deletions}</span>
          </div>
        </div>
      </div>

      {/* Subsystem tags — flattened: just a muted inline list, no borders */}
      {pr.touched_subsystems.length > 0 && (
        <div
          className="mt-3 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs"
          style={{ color: 'var(--text-muted)' }}
        >
          {pr.touched_subsystems.slice(0, 6).map((sub, i) => (
            <span key={sub}>
              {i > 0 && <span className="mr-2" aria-hidden="true">·</span>}
              <span className="font-theme-data">{sub}</span>
            </span>
          ))}
          {pr.touched_subsystems.length > 6 && (
            <span>
              <span className="mr-2" aria-hidden="true">·</span>
              +{pr.touched_subsystems.length - 6} more
            </span>
          )}
        </div>
      )}

      {/* Action row — Approve is the only emphasized action. Rest are muted text links. */}
      <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2">
        <button
          type="button"
          data-testid={`review-queue-approve-${pr.number}`}
          onClick={(ev) => {
            ev.stopPropagation();
            handleApprove();
          }}
          disabled={pendingAction !== null}
          className="rounded-lg border font-theme-data uppercase tracking-wider transition-colors hover:opacity-80 disabled:opacity-40"
          style={{
            padding: '0.5rem 1rem',
            fontSize: '11px',
            borderColor: 'var(--accent)',
            backgroundColor: 'rgba(57, 255, 20, 0.14)',
            color: 'var(--accent)',
          }}
        >
          {pendingAction === 'approve' ? 'approving…' : 'Approve'}
        </button>
        <button
          type="button"
          data-testid={`review-queue-request-changes-${pr.number}`}
          onClick={(ev) => {
            ev.stopPropagation();
            handleRequestChanges();
          }}
          disabled={pendingAction !== null}
          className="text-xs underline-offset-4 transition-opacity hover:underline hover:opacity-100 disabled:opacity-40"
          style={{ color: 'var(--text-muted)' }}
        >
          {pendingAction === 'request-changes' ? 'requesting…' : 'Request changes'}
        </button>
        <button
          type="button"
          data-testid={`review-queue-defer-${pr.number}`}
          onClick={(ev) => {
            ev.stopPropagation();
            handleDefer();
          }}
          disabled={pendingAction !== null}
          className="text-xs underline-offset-4 transition-opacity hover:underline hover:opacity-100 disabled:opacity-40"
          style={{ color: 'var(--text-muted)' }}
        >
          {pendingAction === 'defer' ? 'deferring…' : 'Defer'}
        </button>
        <button
          type="button"
          data-testid={`review-queue-open-diff-${pr.number}`}
          onClick={(ev) => {
            ev.stopPropagation();
            handleOpenDiff();
          }}
          className="text-xs underline-offset-4 transition-opacity hover:underline hover:opacity-100"
          style={{ color: 'var(--text-muted)' }}
        >
          Open diff ↗
        </button>
        <button
          type="button"
          data-testid={`review-queue-expand-${pr.number}`}
          onClick={(ev) => {
            ev.stopPropagation();
            handleExpand();
          }}
          className="ml-auto text-xs underline-offset-4 transition-opacity hover:underline hover:opacity-100"
          style={{ color: 'var(--text-muted)' }}
          aria-expanded={expanded}
        >
          {expanded ? 'Collapse ▲' : 'Expand ▼'}
        </button>
      </div>

      {error && (
        <div
          role="alert"
          data-testid={`review-queue-error-${pr.number}`}
          className="mt-3 rounded-lg border px-3 py-2 text-xs"
          style={{
            borderColor: 'var(--crimson)',
            backgroundColor: 'rgba(255, 0, 64, 0.12)',
            color: 'var(--crimson)',
          }}
        >
          {error}
        </div>
      )}

      {reasonForAction === 'request-changes' && (
        <div
          data-testid={`review-queue-reason-${pr.number}`}
          className="mt-3 flex flex-col gap-2 rounded-lg border p-3"
          style={{
            borderColor: 'var(--border)',
            backgroundColor: 'var(--surface-elevated)',
          }}
        >
          <label
            className="text-xs"
            style={{ color: 'var(--text-muted)' }}
            htmlFor={`reason-${pr.number}`}
          >
            Reason (required). Kept bounded so the repair loop converges.
          </label>
          <textarea
            id={`reason-${pr.number}`}
            data-testid={`review-queue-reason-input-${pr.number}`}
            value={reasonDraft}
            onChange={(ev) => setReasonDraft(ev.target.value)}
            rows={2}
            className="rounded-lg border px-2 py-1.5 text-sm focus:outline-none"
            style={{
              borderColor: 'var(--border)',
              backgroundColor: 'var(--bg)',
              color: 'var(--text)',
            }}
          />
          <div className="flex gap-2">
            <button
              type="button"
              onClick={submitReason}
              data-testid={`review-queue-reason-submit-${pr.number}`}
              className="rounded-lg border px-3 py-1.5 text-xs font-theme-data uppercase tracking-wider hover:opacity-80"
              style={{
                borderColor: 'var(--crimson)',
                backgroundColor: 'rgba(255, 0, 64, 0.12)',
                color: 'var(--crimson)',
              }}
            >
              Send request-changes
            </button>
            <button
              type="button"
              onClick={() => setReasonForAction(null)}
              className="rounded-lg border px-3 py-1.5 text-xs font-theme-data uppercase tracking-wider hover:opacity-80"
              style={{
                borderColor: 'var(--border)',
                color: 'var(--text-muted)',
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {expanded && (
        <div className="mt-4 space-y-3">
          <BriefPanel brief={brief} loading={briefLoading} error={briefError} />
          <div
            className="rounded-lg border px-4 py-3 text-xs"
            style={{
              borderColor: 'var(--border)',
              backgroundColor: 'var(--surface-elevated)',
              color: 'var(--text)',
            }}
          >
            <div
              className="mb-1 font-theme-data uppercase tracking-wider"
              style={{ color: 'var(--text-muted)' }}
            >
              CI detail
            </div>
            <div>
              {pr.ci.total === 0
                ? 'no checks on this PR'
                : `${pr.ci.success} green · ${pr.ci.pending} pending · ${pr.ci.failure} failing (of ${pr.ci.total})`}
            </div>
          </div>
        </div>
      )}
    </article>
  );
}
