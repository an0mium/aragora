/**
 * Aragora SDK Client for SvelteKit
 */

import { createClient, type AragoraClient } from '@aragora/sdk';
import { env } from '$env/dynamic/private';
import { PUBLIC_ARAGORA_API_URL } from '$env/static/public';

// Server-side client (for load functions)
export function getServerClient(): AragoraClient {
  return createClient({
    baseUrl: env.ARAGORA_API_URL || 'http://localhost:8080',
    apiKey: env.ARAGORA_API_KEY,
  });
}

// Browser client (singleton)
let browserClient: AragoraClient | null = null;

export function getBrowserClient(): AragoraClient {
  if (typeof window === 'undefined') {
    throw new Error('getBrowserClient must be called in browser context');
  }

  if (!browserClient) {
    browserClient = createClient({
      baseUrl: PUBLIC_ARAGORA_API_URL || 'http://localhost:8080',
    });
  }

  return browserClient;
}

export type { Debate, Agent } from '@aragora/sdk';
