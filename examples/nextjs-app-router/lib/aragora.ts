/**
 * Aragora SDK Client Configuration
 *
 * This file sets up the Aragora client for use in Next.js App Router.
 * The client is configured for both server and client-side usage.
 */

import { createClient, AragoraClient } from '@aragora/sdk';

// Server-side client (for Server Components and Route Handlers)
export function getServerClient(): AragoraClient {
  return createClient({
    baseUrl: process.env.ARAGORA_API_URL || 'http://localhost:8080',
    apiKey: process.env.ARAGORA_API_KEY,
  });
}

// Client-side client factory (for Client Components)
let clientInstance: AragoraClient | null = null;

export function getClientSideClient(): AragoraClient {
  if (typeof window === 'undefined') {
    throw new Error('getClientSideClient must be called on the client side');
  }

  if (!clientInstance) {
    clientInstance = createClient({
      baseUrl: process.env.NEXT_PUBLIC_ARAGORA_API_URL || 'http://localhost:8080',
      apiKey: process.env.NEXT_PUBLIC_ARAGORA_API_KEY,
    });
  }

  return clientInstance;
}

// Types re-exported for convenience
export type { Debate, Agent, ConsensusResult } from '@aragora/sdk';
