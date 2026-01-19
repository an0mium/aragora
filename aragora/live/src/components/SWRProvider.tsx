'use client';

import { SWRConfig } from 'swr';
import { ReactNode } from 'react';
import { swrFetcher } from '@/hooks/useSWRFetch';

interface SWRProviderProps {
  children: ReactNode;
}

/**
 * Global SWR configuration provider
 *
 * Wraps the app with default SWR settings for:
 * - Global fetcher
 * - Error retry behavior
 * - Revalidation settings
 * - Deduplication
 */
export function SWRProvider({ children }: SWRProviderProps) {
  return (
    <SWRConfig
      value={{
        fetcher: swrFetcher,
        revalidateOnFocus: true,
        revalidateOnReconnect: true,
        dedupingInterval: 2000,
        errorRetryCount: 3,
        errorRetryInterval: 1000,
        shouldRetryOnError: (error) => {
          // Don't retry on 4xx errors (client errors)
          if (error && 'status' in error) {
            const status = (error as { status: number }).status;
            return status >= 500;
          }
          return true;
        },
        onError: (error, key) => {
          // Log errors in development
          if (process.env.NODE_ENV === 'development') {
            console.error(`SWR Error [${key}]:`, error);
          }
        },
      }}
    >
      {children}
    </SWRConfig>
  );
}
