/**
 * Aragora Frontend Configuration.
 *
 * Centralized configuration with environment variable overrides.
 * Import these values instead of hardcoding throughout components.
 *
 * PRODUCTION NOTE: Set NEXT_PUBLIC_API_URL and NEXT_PUBLIC_WS_URL in production.
 * Without these, the app defaults to localhost which will fail in production.
 */

// === API Configuration ===
const _API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
const _WS_URL = process.env.NEXT_PUBLIC_WS_URL;
const _CONTROL_PLANE_WS_URL = process.env.NEXT_PUBLIC_CONTROL_PLANE_WS_URL;
const _NOMIC_LOOP_WS_URL = process.env.NEXT_PUBLIC_NOMIC_LOOP_WS_URL;

// Detect production environment - check for production hostnames
function isProductionEnvironment(): boolean {
  if (typeof window === 'undefined') return false;
  const hostname = window.location.hostname;
  // Production if not localhost/127.0.0.1 and not a local IP
  return (
    hostname !== 'localhost' &&
    hostname !== '127.0.0.1' &&
    !hostname.startsWith('192.168.') &&
    !hostname.startsWith('10.') &&
    !hostname.endsWith('.local')
  );
}

// Configuration validation
// In production: API calls use relative URLs (via Next.js rewrites), WS needs explicit URL
// In development: Both default to localhost
if (typeof window !== 'undefined') {
  const isProd = isProductionEnvironment();

  if (isProd && !_WS_URL) {
    // WebSocket URL is required in production (can't use rewrites for WS)
    console.error(
      '[Aragora] CRITICAL: NEXT_PUBLIC_WS_URL not set in production. ' +
      'WebSocket features will not work. Please configure this in your deployment.'
    );
    if (document.body) {
      const errorBanner = document.createElement('div');
      errorBanner.id = 'aragora-config-error';
      errorBanner.style.cssText = 'position:fixed;top:0;left:0;right:0;background:#dc2626;color:white;padding:12px;text-align:center;z-index:99999;font-family:monospace;';
      errorBanner.textContent = 'Configuration Error: Missing NEXT_PUBLIC_WS_URL. Contact your administrator.';
      document.body.prepend(errorBanner);
    }
  } else if (!isProd) {
    // In development, just warn
    if (!_API_BASE_URL) {
      console.warn(
        '[Aragora] NEXT_PUBLIC_API_URL not set, using localhost:8080 fallback (dev mode).'
      );
    }
    if (!_WS_URL) {
      console.warn(
        '[Aragora] NEXT_PUBLIC_WS_URL not set, using ws://localhost:8765/ws fallback (dev mode).'
      );
    }
  }
}

// In production, use relative URLs (Next.js rewrites proxy to backend)
// In development, default to localhost:8080
export const API_BASE_URL = _API_BASE_URL ?? (typeof window !== 'undefined' && isProductionEnvironment() ? '' : 'http://localhost:8080');
export const WS_URL = _WS_URL || 'ws://localhost:8765/ws';
export const CONTROL_PLANE_WS_URL = _CONTROL_PLANE_WS_URL || 'ws://localhost:8766/api/control-plane/stream';
export const NOMIC_LOOP_WS_URL = _NOMIC_LOOP_WS_URL || 'ws://localhost:8767/api/nomic/stream';

// Helper to detect dev/localhost mode (useful for conditional behavior)
export const IS_DEV_MODE = !_API_BASE_URL || API_BASE_URL.includes('localhost');
export const IS_PRODUCTION = typeof window !== 'undefined' && isProductionEnvironment();

// === Debate Defaults ===
// 9-round format: Round 0 (context), Rounds 1-7 (debate), Round 8 (adjudication)
export const DEFAULT_AGENTS = process.env.NEXT_PUBLIC_DEFAULT_AGENTS || 'grok,anthropic-api,openai-api,deepseek,mistral,gemini,qwen,kimi';
export const DEFAULT_ROUNDS = parseInt(process.env.NEXT_PUBLIC_DEFAULT_ROUNDS || '8', 10);
export const MAX_ROUNDS = parseInt(process.env.NEXT_PUBLIC_MAX_ROUNDS || '10', 10);
export const DEFAULT_CONSENSUS = process.env.NEXT_PUBLIC_DEFAULT_CONSENSUS || 'judge';

// === Agent Display Names ===
export const AGENT_DISPLAY_NAMES: Record<string, string> = {
  'grok': 'Grok 4',
  'anthropic-api': 'Opus 4.5',
  'openai-api': 'GPT 5.2',
  'deepseek': 'DeepSeek V3',
  'mistral': 'Mistral Large 3',
  'gemini': 'Gemini 3 Pro',
  'qwen': 'Qwen3 235B',
  'qwen-max': 'Qwen3 Max',
  'kimi': 'Kimi K2',
  'kimi-thinking': 'Kimi K2 Thinking',
  'llama': 'Llama 3.3',
  'llama4-maverick': 'Llama 4 Maverick',
  'llama4-scout': 'Llama 4 Scout',
  'sonar': 'Perplexity Sonar',
  'command-r': 'Cohere Command R+',
  'jamba': 'AI21 Jamba',
  'yi': 'Yi Large',
  'openrouter': 'OpenRouter',
  'deepseek-r1': 'DeepSeek R1',
};

// === Streaming Configuration ===
export const STREAMING_CAPABLE_AGENTS = (process.env.NEXT_PUBLIC_STREAMING_AGENTS || 'grok,anthropic-api,openai-api,mistral').split(',');

// === UI Timeouts ===
export const API_TIMEOUT_MS = parseInt(process.env.NEXT_PUBLIC_API_TIMEOUT || '30000', 10);
export const WS_RECONNECT_DELAY_MS = parseInt(process.env.NEXT_PUBLIC_WS_RECONNECT_DELAY || '3000', 10);
export const COPY_FEEDBACK_DURATION_MS = 2000;

// === Pagination ===
export const DEFAULT_PAGE_SIZE = parseInt(process.env.NEXT_PUBLIC_DEFAULT_PAGE_SIZE || '20', 10);
export const MAX_PAGE_SIZE = 100;

// === Cache TTLs (milliseconds) ===
export const CACHE_TTL_LEADERBOARD = 5 * 60 * 1000;  // 5 minutes
export const CACHE_TTL_DEBATES = 2 * 60 * 1000;      // 2 minutes
export const CACHE_TTL_AGENT = 10 * 60 * 1000;       // 10 minutes

// === Feature Flags ===
export const ENABLE_STREAMING = process.env.NEXT_PUBLIC_ENABLE_STREAMING !== 'false';
export const ENABLE_AUDIENCE = process.env.NEXT_PUBLIC_ENABLE_AUDIENCE !== 'false';

// === Validation ===
export const MAX_QUESTION_LENGTH = 10000;
export const MIN_QUESTION_LENGTH = 10;

// === Environment Status ===
export interface EnvWarning {
  key: string;
  message: string;
  severity: 'warning' | 'error';
}

export function getEnvWarnings(): EnvWarning[] {
  const warnings: EnvWarning[] = [];

  if (!_API_BASE_URL) {
    warnings.push({
      key: 'NEXT_PUBLIC_API_URL',
      message: 'API URL not set, using localhost:8080',
      severity: 'warning',
    });
  }
  if (!_WS_URL) {
    warnings.push({
      key: 'NEXT_PUBLIC_WS_URL',
      message: 'WebSocket URL not set, using ws://localhost:8765/ws',
      severity: 'warning',
    });
  }
  if (!process.env.NEXT_PUBLIC_SUPABASE_URL) {
    warnings.push({
      key: 'NEXT_PUBLIC_SUPABASE_URL',
      message: 'Supabase not configured, history features disabled',
      severity: 'warning',
    });
  }

  return warnings;
}

// === API Fetch Helper ===
export interface ApiFetchResult<T> {
  data: T | null;
  error: string | null;
  status?: number;
  errorCode?: 'AUTH_REQUIRED' | 'FORBIDDEN' | 'RATE_LIMITED' | 'SERVER_ERROR' | 'NETWORK_ERROR' | 'TIMEOUT' | 'UNKNOWN';
}

// Retry configuration for API calls
const API_RETRY_CONFIG = {
  maxAttempts: 3,
  initialDelayMs: 1000,
  maxDelayMs: 10000,
  // Don't retry auth errors (401) - user needs to re-login
  shouldRetry: (error: Error) => {
    const message = error.message.toLowerCase();
    // Don't retry client errors (except rate limiting)
    if (message.includes('401') || message.includes('403') || message.includes('404')) {
      return false;
    }
    // Retry on network errors, timeouts, rate limiting, and server errors
    return (
      message.includes('failed to fetch') ||
      message.includes('network') ||
      message.includes('timeout') ||
      message.includes('aborted') ||
      message.includes('429') ||
      message.includes('500') ||
      message.includes('502') ||
      message.includes('503') ||
      message.includes('504')
    );
  },
};

function classifyHttpError(status: number): ApiFetchResult<never>['errorCode'] {
  if (status === 401) return 'AUTH_REQUIRED';
  if (status === 403) return 'FORBIDDEN';
  if (status === 429) return 'RATE_LIMITED';
  if (status >= 500) return 'SERVER_ERROR';
  return 'UNKNOWN';
}

export async function apiFetch<T>(
  endpoint: string,
  options?: RequestInit,
  retryOptions?: { maxAttempts?: number; skipRetry?: boolean }
): Promise<ApiFetchResult<T>> {
  const maxAttempts = retryOptions?.skipRetry ? 1 : (retryOptions?.maxAttempts ?? API_RETRY_CONFIG.maxAttempts);
  let lastError: Error | null = null;
  let lastStatus: number | undefined;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT_MS);

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      clearTimeout(timeoutId);
      lastStatus = response.status;

      if (!response.ok) {
        const errorText = await response.text().catch(() => '');
        const errorCode = classifyHttpError(response.status);

        // Don't retry auth errors
        if (errorCode === 'AUTH_REQUIRED' || errorCode === 'FORBIDDEN') {
          return {
            data: null,
            error: errorText || `HTTP ${response.status}`,
            status: response.status,
            errorCode,
          };
        }

        // For retryable errors, throw to trigger retry
        lastError = new Error(`HTTP ${response.status}: ${errorText}`);
        if (attempt < maxAttempts && API_RETRY_CONFIG.shouldRetry(lastError)) {
          const delayMs = Math.min(
            API_RETRY_CONFIG.initialDelayMs * Math.pow(2, attempt - 1),
            API_RETRY_CONFIG.maxDelayMs
          );
          await new Promise(resolve => setTimeout(resolve, delayMs));
          continue;
        }

        return {
          data: null,
          error: errorText || `HTTP ${response.status}`,
          status: response.status,
          errorCode,
        };
      }

      const data = await response.json();
      return { data, error: null, status: response.status };
    } catch (e) {
      clearTimeout(timeoutId);
      lastError = e instanceof Error ? e : new Error(String(e));

      if (lastError.name === 'AbortError') {
        return { data: null, error: 'Request timeout', errorCode: 'TIMEOUT' };
      }

      // Check if we should retry
      if (attempt < maxAttempts && API_RETRY_CONFIG.shouldRetry(lastError)) {
        const delayMs = Math.min(
          API_RETRY_CONFIG.initialDelayMs * Math.pow(2, attempt - 1),
          API_RETRY_CONFIG.maxDelayMs
        );
        await new Promise(resolve => setTimeout(resolve, delayMs));
        continue;
      }

      return {
        data: null,
        error: lastError.message || 'Network error',
        errorCode: 'NETWORK_ERROR',
      };
    }
  }

  // Exhausted all retries
  return {
    data: null,
    error: lastError?.message || 'Request failed after retries',
    status: lastStatus,
    errorCode: lastStatus ? classifyHttpError(lastStatus) : 'NETWORK_ERROR',
  };
}
