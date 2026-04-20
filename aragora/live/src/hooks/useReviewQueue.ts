'use client';

import { useCallback } from 'react';
import { useSWRFetch } from './useSWRFetch';
import { API_BASE_URL } from '@/config';

// ---------------------------------------------------------------------------
// Types (mirror the backend shapes from aragora/server/handlers/review_queue.py)
// ---------------------------------------------------------------------------

export interface CiSummary {
  success: number;
  failure: number;
  pending: number;
  total: number;
}

export interface ReviewQueuePR {
  number: number;
  title: string;
  url: string;
  head_sha: string;
  is_draft: boolean;
  author: string;
  labels: string[];
  additions: number;
  deletions: number;
  changed_files: number;
  created_at: string;
  updated_at: string;
  age_seconds: number | null;
  touched_subsystems: string[];
  ci: CiSummary;
  brief_present: boolean;
  verdict: string | null;
  confidence: number | null;
  deferred: boolean;
}

export interface ReviewQueueListResponse {
  prs: ReviewQueuePR[];
  total: number;
  visible: number;
  deferred_count: number;
  degraded: boolean;
  reason?: string;
}

export interface ReviewQueueBrief {
  pr_number: number;
  head_sha: string;
  verdict: string;
  confidence: number | null;
  logic?: string | null;
  security?: string | null;
  maintainability?: string | null;
  skeptic?: string | null;
  [key: string]: unknown;
}

export interface ReviewQueueStats {
  date: string | null;
  approved: number;
  request_changes: number;
  deferred: number;
  streak: number;
  decision_count: number;
  median_decision_seconds: number | null;
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

function getAccessToken(): string | null {
  if (typeof window === 'undefined') return null;
  const stored = localStorage.getItem('aragora_tokens');
  if (!stored) return null;
  try {
    const parsed = JSON.parse(stored) as { access_token?: string };
    return parsed.access_token || null;
  } catch {
    return null;
  }
}

export function useReviewQueue() {
  const {
    data,
    error,
    isLoading,
    isValidating,
    mutate,
  } = useSWRFetch<ReviewQueueListResponse>('/api/v1/review-queue/prs', {
    refreshInterval: 60000,
  });

  return {
    prs: data?.prs ?? [],
    total: data?.total ?? 0,
    visible: data?.visible ?? 0,
    deferredCount: data?.deferred_count ?? 0,
    degraded: data?.degraded ?? false,
    reason: data?.reason,
    isLoading,
    isValidating,
    error,
    mutate,
  };
}

export function useReviewQueueStats() {
  const { data, isLoading, mutate } = useSWRFetch<{ stats: ReviewQueueStats }>(
    '/api/v1/review-queue/stats',
    { refreshInterval: 30000 },
  );
  return {
    stats: data?.stats ?? null,
    isLoading,
    mutate,
  };
}

export async function fetchBrief(prNumber: number): Promise<ReviewQueueBrief | null> {
  const token = getAccessToken();
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(
    `${API_BASE_URL}/api/v1/review-queue/prs/${prNumber}/brief`,
    { headers },
  );
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`brief request failed: ${res.status}`);
  const data = (await res.json()) as { brief: ReviewQueueBrief };
  return data.brief;
}

export type SettlementAction = 'approve' | 'request-changes' | 'defer';

export interface SettlementOptions {
  note?: string;
  reason?: string;
  hours?: number;
  decisionSeconds?: number;
}

export async function settlePR(
  prNumber: number,
  action: SettlementAction,
  options: SettlementOptions = {},
): Promise<{ status: string; [key: string]: unknown }> {
  const token = getAccessToken();
  const headers: HeadersInit = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const body: Record<string, unknown> = {};
  if (options.note !== undefined) body.note = options.note;
  if (options.reason !== undefined) body.reason = options.reason;
  if (options.hours !== undefined) body.hours = options.hours;
  if (options.decisionSeconds !== undefined) body.decision_seconds = options.decisionSeconds;

  const res = await fetch(
    `${API_BASE_URL}/api/v1/review-queue/prs/${prNumber}/${action}`,
    {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    },
  );
  if (!res.ok) {
    let detail: string;
    try {
      const errData = (await res.json()) as { error?: string };
      detail = errData.error || `${res.status}`;
    } catch {
      detail = `${res.status}`;
    }
    const err = new Error(detail) as Error & { status: number };
    err.status = res.status;
    throw err;
  }
  return res.json() as Promise<{ status: string }>;
}

/**
 * Convenience callback factory — combines settlement + cache invalidation.
 */
export function useSettlePR(onSettled?: () => void) {
  return useCallback(
    async (
      prNumber: number,
      action: SettlementAction,
      options: SettlementOptions = {},
    ) => {
      const result = await settlePR(prNumber, action, options);
      onSettled?.();
      return result;
    },
    [onSettled],
  );
}
