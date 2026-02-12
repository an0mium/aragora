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

  /** List registered on-chain agents with pagination. */
  async listAgents(skip = 0, limit = 100): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('GET', '/api/v1/blockchain/agents', {
      params: { skip, limit },
    });
  }

  /** Register a new agent on the Identity Registry. */
  async registerAgent(request: RegisterAgentRequest): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('POST', '/api/v1/blockchain/agents', {
      body: request,
    });
  }

  /** Get blockchain connector configuration. */
  async getConfig(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('GET', '/api/v1/blockchain/config');
  }

  /** Get blockchain connector health status. */
  async getHealth(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('GET', '/api/v1/blockchain/health');
  }

  /** Trigger blockchain -> Knowledge Mound sync. */
  async sync(request: BlockchainSyncRequest = {}): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('POST', '/api/v1/blockchain/sync', {
      body: request,
    });
  }

  /** Get on-chain agent identity by token ID. */
  async getAgent(tokenId: number): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/blockchain/agents/${encodeURIComponent(String(tokenId))}`
    );
  }

  /** Get reputation summary for an agent token. */
  async getReputation(
    tokenId: number,
    options?: { tag1?: string; tag2?: string }
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/blockchain/agents/${encodeURIComponent(String(tokenId))}/reputation`,
      { params: options }
    );
  }

  /** Get validation summary for an agent token. */
  async getValidations(
    tokenId: number,
    options?: { tag?: string }
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/blockchain/agents/${encodeURIComponent(String(tokenId))}/validations`,
      { params: options }
    );
  }
}
