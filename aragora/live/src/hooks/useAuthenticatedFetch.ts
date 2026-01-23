'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { useAuth } from '@/context/AuthContext';
import { API_BASE_URL } from '@/config';

interface FetchState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  /** True if user is not authenticated (skipped API call) */
  skipped: boolean;
}

interface UseAuthenticatedFetchOptions<T> {
  /** Skip the fetch if user is not authenticated (default: true) */
  requireAuth?: boolean;
  /** Default data to return when not authenticated */
  defaultData?: T;
  /** Dependencies to trigger refetch */
  deps?: unknown[];
  /** Called on successful fetch */
  onSuccess?: (data: T) => void;
  /** Called on error */
  onError?: (error: Error) => void;
  /** Disable automatic fetch on mount */
  manual?: boolean;
}

/**
 * Hook for making authenticated API calls that gracefully handles
 * unauthenticated users without flooding the console with 401 errors.
 *
 * @example
 * // Basic usage - auto-fetches on mount, skips if not authenticated
 * const { data, loading, error, skipped } = useAuthenticatedFetch<Debate[]>(
 *   '/api/debates',
 * );
 *
 * // With options
 * const { data, refetch } = useAuthenticatedFetch<LeaderboardData>(
 *   '/api/leaderboard',
 *   { defaultData: [], manual: true }
 * );
 */
export function useAuthenticatedFetch<T>(
  endpoint: string,
  options: UseAuthenticatedFetchOptions<T> = {}
): FetchState<T> & { refetch: () => Promise<void> } {
  const {
    requireAuth = true,
    defaultData = null as T,
    deps = [],
    onSuccess,
    onError,
    manual = false,
  } = options;

  const { tokens, isAuthenticated, isLoading: authLoading } = useAuth();
  const [state, setState] = useState<FetchState<T>>({
    data: defaultData,
    loading: !manual,
    error: null,
    skipped: false,
  });

  const mountedRef = useRef(true);

  const fetchData = useCallback(async () => {
    // Wait for auth to finish loading
    if (authLoading) {
      return;
    }

    // Skip if auth required but not authenticated
    if (requireAuth && (!isAuthenticated || !tokens?.access_token)) {
      setState({
        data: defaultData,
        loading: false,
        error: null,
        skipped: true,
      });
      return;
    }

    setState(prev => ({ ...prev, loading: true, error: null, skipped: false }));

    try {
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      };

      if (tokens?.access_token) {
        headers['Authorization'] = `Bearer ${tokens.access_token}`;
      }

      const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;
      const response = await fetch(url, { headers });

      if (!response.ok) {
        // Handle auth errors silently when requireAuth is true
        if (response.status === 401 && requireAuth) {
          if (mountedRef.current) {
            setState({
              data: defaultData,
              loading: false,
              error: null,
              skipped: true,
            });
          }
          return;
        }

        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Request failed: ${response.status}`);
      }

      const data = await response.json();

      if (mountedRef.current) {
        setState({
          data,
          loading: false,
          error: null,
          skipped: false,
        });
        onSuccess?.(data);
      }
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      if (mountedRef.current) {
        setState(prev => ({
          ...prev,
          loading: false,
          error: error.message,
          skipped: false,
        }));
        onError?.(error);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [endpoint, tokens?.access_token, isAuthenticated, authLoading, requireAuth, ...deps]);

  // Auto-fetch on mount (unless manual)
  useEffect(() => {
    mountedRef.current = true;

    if (!manual) {
      fetchData();
    }

    return () => {
      mountedRef.current = false;
    };
  }, [fetchData, manual]);

  return {
    ...state,
    refetch: fetchData,
  };
}

/**
 * Hook that provides auth-aware fetch function for manual API calls.
 * Use this when you need to make POST/PUT/DELETE requests or
 * have complex fetch logic.
 *
 * @example
 * const { authFetch, isAuthenticated } = useAuthFetch();
 *
 * const handleSubmit = async () => {
 *   if (!isAuthenticated) return;
 *   const data = await authFetch('/api/debates', {
 *     method: 'POST',
 *     body: JSON.stringify({ question: '...' }),
 *   });
 * };
 */
export function useAuthFetch() {
  const { tokens, isAuthenticated, isLoading } = useAuth();

  const authFetch = useCallback(
    async <T>(
      endpoint: string,
      init: RequestInit = {}
    ): Promise<T | null> => {
      if (!isAuthenticated || !tokens?.access_token) {
        console.warn(`[useAuthFetch] Skipped ${endpoint} - not authenticated`);
        return null;
      }

      const headers: HeadersInit = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${tokens.access_token}`,
        ...init.headers,
      };

      const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;
      const response = await fetch(url, { ...init, headers });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Request failed: ${response.status}`);
      }

      return response.json();
    },
    [tokens?.access_token, isAuthenticated]
  );

  const getAuthHeaders = useCallback((): HeadersInit => {
    return {
      'Content-Type': 'application/json',
      ...(tokens?.access_token && {
        'Authorization': `Bearer ${tokens.access_token}`,
      }),
    };
  }, [tokens?.access_token]);

  return {
    authFetch,
    getAuthHeaders,
    isAuthenticated,
    isLoading,
  };
}

export default useAuthenticatedFetch;
