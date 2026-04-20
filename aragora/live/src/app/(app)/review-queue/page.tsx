'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import clsx from 'clsx';

import { KeyboardHelp } from '@/components/review-queue/KeyboardHelp';
import { ReviewQueueList } from '@/components/review-queue/ReviewQueueList';
import { StatsHeader } from '@/components/review-queue/StatsHeader';
import type { ReviewQueueItem } from '@/components/review-queue/types';
import { formatRelativeAge, riskRank, subsystemKey } from '@/components/review-queue/utils';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { useReviewQueue } from '@/hooks/useReviewQueue';

type SortMode = 'age' | 'risk' | 'subsystem';

export default function ReviewQueuePage() {
  const {
    items,
    stats,
    listLoading,
    listError,
    actionError,
    details,
    detailLoading,
    actionLoading,
    loadDetail,
    refreshAll,
    approve,
    requestChanges,
    defer,
  } = useReviewQueue();

  const [sortBy, setSortBy] = useState<SortMode>('age');
  const [selectedNumber, setSelectedNumber] = useState<number | null>(null);
  const [expandedNumber, setExpandedNumber] = useState<number | null>(null);
  const [requestChangesOpenFor, setRequestChangesOpenFor] = useState<number | null>(null);
  const [requestChangesDrafts, setRequestChangesDrafts] = useState<Record<number, string>>({});
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false);
  const [flashMessage, setFlashMessage] = useState<string | null>(null);
  const [showInboxZero, setShowInboxZero] = useState(false);
  const previousPendingCount = useRef<number | null>(null);

  const sortedItems = [...items].sort((left, right) => {
    if (sortBy === 'risk') {
      const riskDelta = riskRank(left) - riskRank(right);
      if (riskDelta !== 0) return riskDelta;
      return left.number - right.number;
    }
    if (sortBy === 'subsystem') {
      const subsystemDelta = subsystemKey(left).localeCompare(subsystemKey(right));
      if (subsystemDelta !== 0) return subsystemDelta;
      return left.number - right.number;
    }
    const leftCreated = left.created_at || '';
    const rightCreated = right.created_at || '';
    if (leftCreated !== rightCreated) {
      return leftCreated.localeCompare(rightCreated);
    }
    return left.number - right.number;
  });

  useEffect(() => {
    if (!sortedItems.length) {
      setSelectedNumber(null);
      return;
    }
    if (!selectedNumber || !sortedItems.some((item) => item.number === selectedNumber)) {
      setSelectedNumber(sortedItems[0].number);
    }
  }, [selectedNumber, sortedItems]);

  useEffect(() => {
    const previous = previousPendingCount.current;
    if (previous !== null && previous > 0 && items.length === 0) {
      setShowInboxZero(true);
    }
    previousPendingCount.current = items.length;
  }, [items.length]);

  const toggleExpand = useCallback(
    async (number: number) => {
      setSelectedNumber(number);
      if (expandedNumber === number) {
        setExpandedNumber(null);
        return;
      }
      setExpandedNumber(number);
      await loadDetail(number);
    },
    [expandedNumber, loadDetail]
  );

  const openRequestChanges = useCallback(
    (number: number) => {
      setSelectedNumber(number);
      setRequestChangesOpenFor(number);
      if (expandedNumber !== number) {
        setExpandedNumber(number);
      }
      void loadDetail(number);
    },
    [expandedNumber, loadDetail]
  );

  const handleApprove = useCallback(
    async (item: ReviewQueueItem) => {
      const needsConfirm = !item.brief || item.brief.verdict !== 'approve_candidate';
      if (
        needsConfirm &&
        !window.confirm(
          item.brief
            ? `Brief verdict is ${item.brief.raw_verdict || item.brief.verdict}. Approve anyway?`
            : `No brief is available for #${item.number}. Approve anyway?`
        )
      ) {
        return;
      }
      const ok = await approve(item.number);
      if (ok) {
        setFlashMessage(`Approved #${item.number}`);
        if (selectedNumber === item.number) {
          setExpandedNumber(null);
          setRequestChangesOpenFor(null);
        }
      }
    },
    [approve, selectedNumber]
  );

  const handleRequestChangesSubmit = useCallback(
    async (number: number) => {
      const reason = (requestChangesDrafts[number] || '').trim();
      if (!reason) return;
      const ok = await requestChanges(number, reason);
      if (ok) {
        setFlashMessage(`Requested changes on #${number}`);
        setRequestChangesOpenFor(null);
        setExpandedNumber(null);
      }
    },
    [requestChanges, requestChangesDrafts]
  );

  const handleDefer = useCallback(
    async (number: number) => {
      const reason = window.prompt('Optional defer note (local only)', '') ?? '';
      const ok = await defer(number, reason);
      if (ok) {
        setFlashMessage(`Deferred #${number} for 4 hours`);
        if (selectedNumber === number) {
          setExpandedNumber(null);
          setRequestChangesOpenFor(null);
        }
      }
    },
    [defer, selectedNumber]
  );

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return;
      }

      const currentIndex = sortedItems.findIndex((item) => item.number === selectedNumber);
      const currentItem = sortedItems[currentIndex] || null;

      if (event.key === '?') {
        event.preventDefault();
        setShowKeyboardHelp((current) => !current);
        return;
      }

      if (!currentItem) return;

      if (event.key === 'j') {
        event.preventDefault();
        const next = sortedItems[Math.min(currentIndex + 1, sortedItems.length - 1)];
        setSelectedNumber(next.number);
        return;
      }

      if (event.key === 'k') {
        event.preventDefault();
        const next = sortedItems[Math.max(currentIndex - 1, 0)];
        setSelectedNumber(next.number);
        return;
      }

      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        void toggleExpand(currentItem.number);
        return;
      }

      if (event.key === 'a') {
        event.preventDefault();
        void handleApprove(currentItem);
        return;
      }

      if (event.key === 'r') {
        event.preventDefault();
        openRequestChanges(currentItem.number);
        return;
      }

      if (event.key === 'd') {
        event.preventDefault();
        void handleDefer(currentItem.number);
        return;
      }

      if (event.key === 'o') {
        event.preventDefault();
        window.open(currentItem.diff_url, '_blank', 'noopener,noreferrer');
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleApprove, handleDefer, openRequestChanges, selectedNumber, sortedItems, toggleExpand]);

  return (
    <>
      <main className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(0,255,163,0.08),transparent_24%),linear-gradient(180deg,var(--bg),#05080b)] px-4 py-6 text-text md:px-6">
        <div className="mx-auto max-w-7xl space-y-6">
          <StatsHeader
            pendingCount={sortedItems.length}
            medianDecisionSeconds={stats.median_decision_seconds}
            streak={stats.streak}
            approvalsToday={stats.approvals_today}
            sortBy={sortBy}
            onSortChange={setSortBy}
          />

          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2 text-[11px] font-theme-data uppercase tracking-[0.18em] text-text-muted">
              <span>{sortedItems.length} PRs in scope</span>
              {sortedItems[0]?.created_at ? (
                <span>oldest {formatRelativeAge(sortedItems[0].created_at)}</span>
              ) : null}
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => void refreshAll()}
                className="rounded-full border border-[var(--accent)]/25 px-3 py-1.5 text-xs font-theme-data text-[var(--accent)] hover:bg-[var(--accent)]/10"
              >
                Refresh
              </button>
              <button
                type="button"
                onClick={() => setShowKeyboardHelp(true)}
                className="rounded-full border border-border px-3 py-1.5 text-xs font-theme-data text-text-muted hover:text-text"
              >
                Shortcuts
              </button>
            </div>
          </div>

          {(listError || actionError) ? (
            <div className="rounded-2xl border border-acid-red/30 bg-acid-red/10 px-4 py-3 text-sm font-theme-data text-acid-red">
              {actionError || listError}
            </div>
          ) : null}

          {flashMessage ? (
            <div className="rounded-2xl border border-emerald-400/25 bg-emerald-400/10 px-4 py-3 text-sm font-theme-data text-emerald-300">
              {flashMessage}
            </div>
          ) : null}

          <PanelErrorBoundary panelName="Review Queue">
            {listLoading ? (
              <div className="rounded-2xl border border-[var(--accent)]/15 bg-surface/70 px-5 py-12 text-center text-sm font-theme-data text-text-muted">
                Loading queue…
              </div>
            ) : sortedItems.length === 0 ? (
              <div className="rounded-2xl border border-[var(--accent)]/15 bg-surface/70 px-5 py-12 text-center">
                <p className="text-[11px] font-theme-data uppercase tracking-[0.28em] text-[var(--accent)]">
                  Inbox Zero
                </p>
                <h2 className="mt-3 text-3xl font-theme-data text-text">
                  Morning tranche cleared.
                </h2>
                <p className="mx-auto mt-3 max-w-xl text-sm font-theme-data text-text-muted">
                  No pending PRs remain in scope. This surface is intentionally optimized for the daily settlement loop, not for camping in diffs.
                </p>
              </div>
            ) : (
              <ReviewQueueList
                items={sortedItems}
                selectedNumber={selectedNumber}
                expandedNumber={expandedNumber}
                details={details}
                detailLoading={detailLoading}
                requestChangesOpenFor={requestChangesOpenFor}
                requestChangesDrafts={requestChangesDrafts}
                actionLoading={actionLoading}
                onSelect={setSelectedNumber}
                onToggleExpand={(number) => void toggleExpand(number)}
                onApprove={(item) => void handleApprove(item)}
                onDefer={(number) => void handleDefer(number)}
                onOpenDiff={(item) => window.open(item.diff_url, '_blank', 'noopener,noreferrer')}
                onOpenRequestChanges={openRequestChanges}
                onRequestChangesDraftChange={(number, value) =>
                  setRequestChangesDrafts((current) => ({ ...current, [number]: value }))
                }
                onRequestChangesSubmit={(number) => void handleRequestChangesSubmit(number)}
                onRequestChangesCancel={() => setRequestChangesOpenFor(null)}
              />
            )}
          </PanelErrorBoundary>
        </div>
      </main>

      <KeyboardHelp open={showKeyboardHelp} onClose={() => setShowKeyboardHelp(false)} />

      {showInboxZero ? (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 px-4">
          <div className="max-w-lg rounded-2xl border border-[var(--accent)]/25 bg-[linear-gradient(180deg,rgba(8,12,16,0.96),rgba(12,18,24,0.96))] p-6 text-center shadow-[0_30px_80px_rgba(0,0,0,0.38)]">
            <p className="text-[11px] font-theme-data uppercase tracking-[0.28em] text-[var(--accent)]">
              Queue Cleared
            </p>
            <h2 className="mt-3 text-3xl font-theme-data text-text">Inbox zero.</h2>
            <p className="mt-3 text-sm font-theme-data text-text-muted">
              The browser tranche is empty. This is the intended UX reward loop for the morning brief workflow.
            </p>
            <button
              type="button"
              onClick={() => setShowInboxZero(false)}
              className={clsx(
                'mt-5 rounded-full border border-[var(--accent)]/25 px-4 py-2 text-xs font-theme-data text-[var(--accent)] hover:bg-[var(--accent)]/10'
              )}
            >
              Keep Moving
            </button>
          </div>
        </div>
      ) : null}
    </>
  );
}
