/**
 * Agent Bridge Namespace API
 *
 * Provides access to recorded agent-bridge runs.
 */

import type { AragoraClient } from '../client';

export interface AgentBridgeRun {
  run_id?: string;
  status?: string;
  created_at?: string;
  updated_at?: string;
  [key: string]: unknown;
}

export interface ListAgentBridgeRunsOptions {
  status?: string;
  limit?: number;
  offset?: number;
}

export class AgentBridgeAPI {
  constructor(private client: AragoraClient) {}

  /** List recorded agent-bridge runs. */
  async listRuns(
    options?: ListAgentBridgeRunsOptions
  ): Promise<{ runs: AgentBridgeRun[]; total?: number }> {
    const params: Record<string, unknown> = {};
    if (options?.status !== undefined) {
      params.status = options.status;
    }
    if (options?.limit !== undefined) {
      params.limit = options.limit;
    }
    if (options?.offset !== undefined) {
      params.offset = options.offset;
    }
    return this.client.request<{ runs: AgentBridgeRun[]; total?: number }>(
      'GET',
      '/api/v1/agent-bridge/runs',
      { params }
    );
  }
}
