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

// Warn in browser when env vars are missing (only warn once)
if (typeof window !== 'undefined') {
  if (!_API_BASE_URL) {
    console.warn(
      '[Aragora] NEXT_PUBLIC_API_URL not set, using localhost:8080 fallback. ' +
      'This will fail in production - set the environment variable.'
    );
  }
  if (!_WS_URL) {
    console.warn(
      '[Aragora] NEXT_PUBLIC_WS_URL not set, using ws://localhost:8080 fallback. ' +
      'This will fail in production - set the environment variable.'
    );
  }
}

export const API_BASE_URL = _API_BASE_URL || 'http://localhost:8080';
export const WS_URL = _WS_URL || 'ws://localhost:8080';

// Helper to detect dev/localhost mode (useful for conditional behavior)
export const IS_DEV_MODE = !_API_BASE_URL || API_BASE_URL.includes('localhost');

// === Debate Defaults ===
export const DEFAULT_AGENTS = process.env.NEXT_PUBLIC_DEFAULT_AGENTS || 'grok,anthropic-api,openai-api,deepseek,mistral-api,gemini,qwen-max,kimi';
export const DEFAULT_ROUNDS = parseInt(process.env.NEXT_PUBLIC_DEFAULT_ROUNDS || '3', 10);
export const MAX_ROUNDS = parseInt(process.env.NEXT_PUBLIC_MAX_ROUNDS || '10', 10);

// === Agent Display Names ===
export const AGENT_DISPLAY_NAMES: Record<string, string> = {
  'grok': 'Grok 4',
  'anthropic-api': 'Opus 4.5',
  'openai-api': 'GPT 5.2',
  'deepseek': 'DeepSeek V3',
  'mistral-api': 'Mistral Large 3',
  'codestral': 'Codestral',
  'gemini': 'Gemini 3 Pro',
  'qwen': 'Qwen 2.5',
  'qwen-max': 'Qwen Max',
  'kimi': 'Kimi',
  'yi': 'Yi Large',
  'llama': 'Llama 3.3',
};

// === Streaming Configuration ===
export const STREAMING_CAPABLE_AGENTS = (process.env.NEXT_PUBLIC_STREAMING_AGENTS || 'grok,anthropic-api,openai-api,mistral-api').split(',');

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
