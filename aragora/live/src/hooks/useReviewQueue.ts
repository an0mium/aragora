'use client';

import { useEffect, useState } from 'react';

import { useAuthFetch, useAuthenticatedFetch } from '@/hooks/useAuthenticatedFetch';
import type {
  ReviewQueueDetail,
  ReviewQueueItem,
  ReviewQueueListResponse,
  ReviewQueueStats,
} from '@/components/review-queue/types';

const DEFAULT_LIST: ReviewQueueListResponse = {
  prs: [],
  count: 0,
  generated_at: '',
  source: 'local-review-queue',
};

const DEFAULT_STATS: ReviewQueueStats = {
  decisions_today: 0,
  approvals_today: 0,
  median_decision_seconds: 0,
  streak: 0,
  source: 'local-review-queue',
};

type ActionName = 'approve' | 'request_changes' | 'defer';

export function useReviewQueue() {
  const listState = useAuthenticatedFetch<ReviewQueueListResponse>('/api/v1/review-queue/prs', {
    defaultData: DEFAULT_LIST,
  });
  const statsState = useAuthenticatedFetch<ReviewQueueStats>('/api/v1/review-queue/stats', {
    defaultData: DEFAULT_STATS,
  });
  const { authFetch } = useAuthFetch();

  const [items, setItems] = useState<ReviewQueueItem[]>([]);
  const [details, setDetails] = useState<Record<number, ReviewQueueDetail>>({});
  const [detailLoading, setDetailLoading] = useState<Record<number, boolean>>({});
  const [actionLoading, setActionLoading] = useState<Record<number, ActionName | null>>({});
  const [actionError, setActionError] = useState<string | null>(null);

  useEffect(() => {
    setItems(listState.data?.prs || []);
  }, [listState.data]);

  async function refreshAll(): Promise<void> {
    await Promise.all([listState.refetch(), statsState.refetch()]);
  }

  async function loadDetail(number: number): Promise<void> {
    if (details[number] || detailLoading[number]) return;
    setDetailLoading((current) => ({ ...current, [number]: true }));
    setActionError(null);
    try {
      const result = await authFetch<ReviewQueueDetail>(`/api/v1/review-queue/prs/${number}`);
      if (result) {
        setDetails((current) => ({ ...current, [number]: result }));
      }
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Failed to load PR detail');
    } finally {
      setDetailLoading((current) => ({ ...current, [number]: false }));
    }
  }

  async function runAction(
    number: number,
    action: ActionName,
    body: Record<string, unknown> = {}
  ): Promise<boolean> {
    setActionLoading((current) => ({ ...current, [number]: action }));
    setActionError(null);
    try {
      const result = await authFetch(
        `/api/v1/review-queue/prs/${number}/${action === 'request_changes' ? 'request-changes' : action}`,
        {
          method: 'POST',
          body: JSON.stringify(body),
        }
      );
      if (!result) return false;
      setItems((current) => current.filter((item) => item.number !== number));
      setDetails((current) => {
        const next = { ...current };
        delete next[number];
        return next;
      });
      await statsState.refetch();
      return true;
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Action failed');
      return false;
    } finally {
      setActionLoading((current) => ({ ...current, [number]: null }));
    }
  }

  return {
    items,
    setItems,
    stats: statsState.data || DEFAULT_STATS,
    listLoading: listState.loading,
    listError: listState.error,
    actionError,
    details,
    detailLoading,
    actionLoading,
    loadDetail,
    refreshAll,
    approve: (number: number) => runAction(number, 'approve'),
    requestChanges: (number: number, reason: string) =>
      runAction(number, 'request_changes', { reason }),
    defer: (number: number, reason: string) => runAction(number, 'defer', { reason }),
  };
}
