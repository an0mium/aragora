/**
 * Blockchain Namespace API
 *
 * Provides access to Aragora ERC-8004 blockchain endpoints.
 */

import type { AragoraClient } from '../client';

export interface BlockchainSyncRequest {
  sync_identities?: boolean;
  sync_reputation?: boolean;
  sync_validations?: boolean;
  agent_ids?: number[];
}

export interface RegisterAgentRequest {
  agent_uri: string;
  metadata?: Record<string, string>;
}

export class BlockchainNamespace {
  constructor(private client: AragoraClient) {}

  /**
   * List registered on-chain agents with pagination.
   * @route GET /api/v1/blockchain/agents
   */
  async listAgents(skip = 0, limit = 100): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/blockchain/agents', {
      params: { skip, limit },
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Register a new agent on the Identity Registry.
   * @route POST /api/v1/blockchain/agents
   */
  async registerAgent(request: RegisterAgentRequest): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/blockchain/agents', {
      body: request,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get blockchain connector configuration.
   * @route GET /api/v1/blockchain/config
   */
  async getConfig(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/blockchain/config') as Promise<Record<string, unknown>>;
  }

  /**
   * Get blockchain connector health status.
   * @route GET /api/v1/blockchain/health
   */
  async getHealth(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/blockchain/health') as Promise<Record<string, unknown>>;
  }

  /**
   * Trigger blockchain -> Knowledge Mound sync.
   * @route POST /api/v1/blockchain/sync
   */
  async sync(request: BlockchainSyncRequest = {}): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/blockchain/sync', {
      body: request,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get on-chain agent identity by token ID.
   * @route GET /api/v1/blockchain/agents/{tokenId}
   */
  async getAgent(tokenId: number): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/blockchain/agents/${encodeURIComponent(String(tokenId))}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get reputation summary for an agent token.
   * @route GET /api/v1/blockchain/agents/{tokenId}/reputation
   */
  async getReputation(
    tokenId: number,
    options?: { tag1?: string; tag2?: string }
  ): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/blockchain/agents/${encodeURIComponent(String(tokenId))}/reputation`,
      { params: options }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get validation summary for an agent token.
   * @route GET /api/v1/blockchain/agents/{tokenId}/validations
   */
  async getValidations(
    tokenId: number,
    options?: { tag?: string }
  ): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/blockchain/agents/${encodeURIComponent(String(tokenId))}/validations`,
      { params: options }
    ) as Promise<Record<string, unknown>>;
  }
}
