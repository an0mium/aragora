/**
 * Aragora SDK Client for Remix
 *
 * Server-side only - use in loaders and actions
 */

import { createClient, type AragoraClient } from '@aragora/sdk';

let client: AragoraClient | null = null;

export function getClient(): AragoraClient {
  if (!client) {
    client = createClient({
      baseUrl: process.env.ARAGORA_API_URL || 'http://localhost:8080',
      apiKey: process.env.ARAGORA_API_KEY,
    });
  }
  return client;
}

export type { Debate, Agent } from '@aragora/sdk';
