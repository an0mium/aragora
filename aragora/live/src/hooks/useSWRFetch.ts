'use client';

import useSWR, { SWRConfiguration, mutate, useSWRConfig } from 'swr';
import { API_BASE_URL } from '@/config';

const TOKENS_KEY = 'aragora_tokens';

function getAccessToken(): string | null {
  if (typeof window === 'undefined') return null;
  const stored = localStorage.getItem(TOKENS_KEY);
  if (!stored) return null;
  try {
    const parsed = JSON.parse(stored) as { access_token?: string };
    return parsed.access_token || null;
  } catch {
    return null;
  }
}

function isInternalApiRequest(url: string): boolean {
  if (typeof window === 'undefined') return false;
  try {
    const parsed = new URL(url, window.location.origin);
    const apiOrigin = new URL(API_BASE_URL, window.location.origin).origin;
    const isApiPath = parsed.pathname.startsWith('/api/');
    const isTrustedOrigin = parsed.origin === window.location.origin || parsed.origin === apiOrigin;
    return isApiPath && isTrustedOrigin;
  } catch {
    return false;
  }
}

function getAuthHeaders(url: string): HeadersInit {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };
  const token = getAccessToken();
  if (token && isInternalApiRequest(url)) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

/**
 * Global fetcher function for SWR
 * Uses fetchWithRetry for resilience
 */
export async function swrFetcher<T>(url: string): Promise<T> {
  const response = await fetch(url, {
    headers: getAuthHeaders(url),
  });

  if (!response.ok) {
    const error = new Error('API request failed') as Error & { status: number };
    error.status = response.status;
    throw error;
  }

  return response.json();
}

/**
 * Options for useSWRFetch hook
 */
export interface UseSWRFetchOptions<T> extends SWRConfiguration<T> {
  /** Base URL for API requests (defaults to API_BASE_URL) */
  baseUrl?: string;
  /** Whether to fetch immediately (default: true) */
  enabled?: boolean;
}

/**
 * SWR-based data fetching hook with caching and automatic revalidation
 *
 * Features:
 * - Automatic caching with stale-while-revalidate
 * - Deduplication of requests
 * - Automatic revalidation on focus/reconnect
 * - Optimistic updates support
 * - Error retry with exponential backoff
 *
 * @example
 * // Basic usage
 * const { data, error, isLoading } = useSWRFetch('/api/leaderboard');
 *
 * // With options
 * const { data, error, isLoading, mutate } = useSWRFetch('/api/debates', {
 *   refreshInterval: 30000, // Refresh every 30 seconds
 *   revalidateOnFocus: true,
 * });
 *
 * // Conditional fetching
 * const { data } = useSWRFetch(userId ? `/api/users/${userId}` : null);
 */
export function useSWRFetch<T = unknown>(
  endpoint: string | null,
  options: UseSWRFetchOptions<T> = {}
) {
  const {
    baseUrl = API_BASE_URL,
    enabled = true,
    ...swrOptions
  } = options;

  const url = endpoint && enabled ? `${baseUrl}${endpoint}` : null;

  const {
    data,
    error,
    isLoading,
    isValidating,
    mutate: boundMutate,
  } = useSWR<T>(url, swrFetcher, {
    revalidateOnFocus: true,
    revalidateOnReconnect: true,
    dedupingInterval: 2000,
    errorRetryCount: 3,
    errorRetryInterval: 1000,
    ...swrOptions,
  });

  return {
    data: data ?? null,
    error: error as Error | null,
    isLoading,
    isValidating,
    mutate: boundMutate,
  };
}

/**
 * Hook for multiple endpoints with caching
 *
 * @example
 * const { data, errors, isLoading } = useSWRFetchMultiple([
 *   '/api/debates',
 *   '/api/leaderboard',
 *   '/api/agents',
 * ]);
 */
export function useSWRFetchMultiple<T = unknown>(
  endpoints: (string | null)[],
  options: UseSWRFetchOptions<T> = {}
) {
  const { baseUrl = API_BASE_URL, enabled = true } = options;

  const results = endpoints.map((endpoint) => {
    const url = endpoint && enabled ? `${baseUrl}${endpoint}` : null;
    // eslint-disable-next-line react-hooks/rules-of-hooks
    return useSWR<T>(url, swrFetcher);
  });

  return {
    data: results.map((r) => r.data ?? null),
    errors: results.map((r) => r.error as Error | null),
    isLoading: results.some((r) => r.isLoading),
    isValidating: results.some((r) => r.isValidating),
  };
}

/**
 * Prefetch data for a given endpoint
 * Useful for preloading data before navigation
 *
 * @example
 * // Prefetch on hover
 * <Link
 *   href="/debates/123"
 *   onMouseEnter={() => prefetchData('/api/debates/123')}
 * >
 */
export function prefetchData(endpoint: string, baseUrl: string = API_BASE_URL) {
  const url = `${baseUrl}${endpoint}`;
  return mutate(url, swrFetcher(url));
}

/**
 * Invalidate cached data for a given endpoint
 * Forces a revalidation on next access
 *
 * @example
 * // After creating a new debate
 * await createDebate(data);
 * invalidateCache('/api/debates');
 */
export function invalidateCache(endpoint: string, baseUrl: string = API_BASE_URL) {
  const url = `${baseUrl}${endpoint}`;
  return mutate(url);
}

/**
 * Invalidate all cached data matching a pattern
 *
 * @example
 * // Invalidate all debate-related caches
 * invalidateCachePattern(/\/api\/debates/);
 */
export function invalidateCachePattern(pattern: RegExp) {
  return mutate(
    (key) => typeof key === 'string' && pattern.test(key),
    undefined,
    { revalidate: true }
  );
}

/**
 * Update cached data optimistically
 *
 * @example
 * // Optimistic update
 * const newDebate = { id: '123', title: 'New Debate' };
 * updateCache('/api/debates', (current) => [...(current || []), newDebate]);
 */
export function updateCache<T>(
  endpoint: string,
  updater: (current: T | undefined) => T,
  baseUrl: string = API_BASE_URL
) {
  const url = `${baseUrl}${endpoint}`;
  return mutate(url, updater, { revalidate: false });
}

/**
 * Hook to get the global SWR cache for debugging
 */
export function useSWRCache() {
  const { cache } = useSWRConfig();
  return cache;
}

/**
 * Pre-configured hooks for common endpoints
 */

export function useDebates(options?: UseSWRFetchOptions<unknown>) {
  return useSWRFetch('/api/debates', {
    refreshInterval: 30000, // Refresh every 30 seconds
    ...options,
  });
}

export function useDebate(debateId: string | null, options?: UseSWRFetchOptions<unknown>) {
  return useSWRFetch(debateId ? `/api/debates/${debateId}` : null, options);
}

export function useLeaderboard(options?: UseSWRFetchOptions<unknown>) {
  return useSWRFetch('/api/leaderboard', {
    refreshInterval: 60000, // Refresh every minute
    ...options,
  });
}

export function useAgents(options?: UseSWRFetchOptions<unknown>) {
  return useSWRFetch('/api/agents', {
    refreshInterval: 60000,
    ...options,
  });
}

export function useConnectors(options?: UseSWRFetchOptions<unknown>) {
  return useSWRFetch('/api/scheduler/jobs', {
    refreshInterval: 30000,
    ...options,
  });
}

export function useSchedulerStats(options?: UseSWRFetchOptions<unknown>) {
  return useSWRFetch('/api/scheduler/stats', {
    refreshInterval: 10000, // Refresh every 10 seconds
    ...options,
  });
}

export function usePulseTopics(options?: UseSWRFetchOptions<unknown>) {
  return useSWRFetch('/api/pulse/topics', {
    refreshInterval: 60000,
    ...options,
  });
}

export function useMemoryStats(options?: UseSWRFetchOptions<unknown>) {
  return useSWRFetch('/api/memory/stats', {
    refreshInterval: 30000,
    ...options,
  });
}
