/**
 * Retry utility with exponential backoff
 */

interface RetryOptions {
  maxRetries?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  onRetry?: (attempt: number, error: Error) => void;
}

const DEFAULT_OPTIONS: Required<Omit<RetryOptions, 'onRetry'>> = {
  maxRetries: 3,
  baseDelayMs: 1000,
  maxDelayMs: 10000,
};

/**
 * Retry a function with exponential backoff
 */
export async function retry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const { maxRetries, baseDelayMs, maxDelayMs } = { ...DEFAULT_OPTIONS, ...options };
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (attempt < maxRetries) {
        const delay = Math.min(baseDelayMs * Math.pow(2, attempt), maxDelayMs);
        options.onRetry?.(attempt + 1, lastError);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError;
}

/**
 * Fetch with automatic retry on network errors and rate limiting
 */
export async function fetchWithRetry(
  url: string,
  init?: RequestInit,
  options: RetryOptions = {}
): Promise<Response> {
  return retry(async () => {
    const response = await fetch(url, init);

    // Handle rate limiting (429) with exponential backoff
    if (response.status === 429) {
      const retryAfter = response.headers.get('Retry-After');
      const waitTime = retryAfter ? parseInt(retryAfter, 10) * 1000 : 5000;
      console.warn(`[fetchWithRetry] Rate limited on ${url}, waiting ${waitTime}ms`);
      await new Promise((resolve) => setTimeout(resolve, waitTime));
      throw new Error(`Rate limited: ${response.status}`);
    }

    // Retry on server errors (5xx), but not other client errors (4xx)
    if (response.status >= 500) {
      throw new Error(`Server error: ${response.status}`);
    }

    return response;
  }, options);
}

/**
 * Custom hook state for retry functionality
 */
export interface RetryState {
  retryCount: number;
  isRetrying: boolean;
  lastError: string | null;
}

/**
 * Check if an error is retryable (network/server error)
 */
export function isRetryableError(error: unknown): boolean {
  if (error instanceof TypeError) {
    // Network errors like "Failed to fetch"
    return true;
  }
  if (error instanceof Error) {
    const message = error.message.toLowerCase();
    return (
      message.includes('network') ||
      message.includes('timeout') ||
      message.includes('server error') ||
      message.includes('failed to fetch')
    );
  }
  return false;
}
