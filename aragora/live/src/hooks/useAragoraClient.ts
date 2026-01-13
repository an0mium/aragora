/**
 * React Hook for Aragora SDK Client
 *
 * Provides an AragoraClient instance configured with the current
 * auth token and backend URL from context.
 *
 * Usage:
 * ```typescript
 * function MyComponent() {
 *   const client = useAragoraClient();
 *
 *   useEffect(() => {
 *     client.debates.list().then(setDebates);
 *   }, [client]);
 * }
 * ```
 */

'use client';

import { useMemo, useEffect } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useBackend } from '@/components/BackendSelector';
import { AragoraClient, getClient, clearClient } from '@/lib/aragora-client';

/**
 * Hook that returns an AragoraClient configured with current auth and backend.
 *
 * The client is memoized and only recreated when the token or backend changes.
 */
export function useAragoraClient(): AragoraClient {
  const { tokens } = useAuth();
  const { config: backendConfig } = useBackend();

  const token = tokens?.access_token;
  const baseUrl = backendConfig.api;

  // Memoize the client instance
  const client = useMemo(() => {
    return getClient(token, baseUrl);
  }, [token, baseUrl]);

  return client;
}

/**
 * Hook that clears the client cache on logout.
 * Use this in your auth provider or logout handler.
 */
export function useClientCleanup(): void {
  const { isAuthenticated } = useAuth();

  useEffect(() => {
    if (!isAuthenticated) {
      clearClient();
    }
  }, [isAuthenticated]);
}

/**
 * Hook that returns whether the client is authenticated.
 */
export function useClientAuth(): { isAuthenticated: boolean; isAdmin: boolean } {
  const { isAuthenticated, user } = useAuth();

  return {
    isAuthenticated,
    isAdmin: isAuthenticated && (user?.role === 'admin' || user?.role === 'owner'),
  };
}

export default useAragoraClient;
