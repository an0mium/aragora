/**
 * API utility functions for making requests to the Aragora backend.
 */

// Default API base URL - can be overridden by environment variable or component props
const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8080';

interface ApiFetchOptions extends RequestInit {
  baseUrl?: string;
}

interface ApiFetchResult<T = unknown> {
  data?: T;
  error?: string;
}

/**
 * Fetch wrapper with standard configuration and error handling.
 *
 * @param path - API endpoint path (can be relative like '/api/debates' or full URL)
 * @param options - Fetch options including optional baseUrl override
 * @returns The JSON response data
 * @throws Error if the request fails
 */
export async function apiFetch<T = unknown>(
  path: string,
  options: ApiFetchOptions = {}
): Promise<T> {
  const { baseUrl = DEFAULT_API_BASE, ...fetchOptions } = options;

  // Determine the full URL
  let url: string;
  if (path.startsWith('http://') || path.startsWith('https://')) {
    url = path;
  } else if (path.startsWith('/api/')) {
    // Relative API path - prepend base URL
    url = `${baseUrl}${path}`;
  } else if (path.startsWith('/')) {
    // Other relative path
    url = `${baseUrl}${path}`;
  } else {
    // Bare path - prepend base URL with /
    url = `${baseUrl}/${path}`;
  }

  // Set default headers
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...fetchOptions.headers,
  };

  const response = await fetch(url, {
    ...fetchOptions,
    headers,
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => response.statusText);
    throw new Error(`API Error (${response.status}): ${errorText}`);
  }

  // Handle empty responses
  const contentType = response.headers.get('content-type');
  if (!contentType || !contentType.includes('application/json')) {
    return {} as T;
  }

  return response.json();
}

/**
 * Safe fetch wrapper that returns an object with either data or error.
 * Useful for components that want to handle errors without try/catch.
 *
 * @param path - API endpoint path
 * @param options - Fetch options
 * @returns Object with data or error field
 */
export async function apiFetchSafe<T = unknown>(
  path: string,
  options: ApiFetchOptions = {}
): Promise<ApiFetchResult<T>> {
  try {
    const data = await apiFetch<T>(path, options);
    return { data };
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) };
  }
}

/**
 * GET request wrapper.
 */
export async function apiGet<T = unknown>(
  path: string,
  options: Omit<ApiFetchOptions, 'method'> = {}
): Promise<T> {
  return apiFetch<T>(path, { ...options, method: 'GET' });
}

/**
 * POST request wrapper.
 */
export async function apiPost<T = unknown>(
  path: string,
  body?: unknown,
  options: Omit<ApiFetchOptions, 'method' | 'body'> = {}
): Promise<T> {
  return apiFetch<T>(path, {
    ...options,
    method: 'POST',
    body: body ? JSON.stringify(body) : undefined,
  });
}

/**
 * PUT request wrapper.
 */
export async function apiPut<T = unknown>(
  path: string,
  body?: unknown,
  options: Omit<ApiFetchOptions, 'method' | 'body'> = {}
): Promise<T> {
  return apiFetch<T>(path, {
    ...options,
    method: 'PUT',
    body: body ? JSON.stringify(body) : undefined,
  });
}

/**
 * DELETE request wrapper.
 */
export async function apiDelete<T = unknown>(
  path: string,
  options: Omit<ApiFetchOptions, 'method'> = {}
): Promise<T> {
  return apiFetch<T>(path, { ...options, method: 'DELETE' });
}
