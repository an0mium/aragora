/**
 * Agent Bridge Namespace API
 *
 * Provides access to recorded agent-bridge runs.
 */

import type { AragoraClient } from '../client';

export interface AgentBridgeRun {
  schema_version: 1;
  run_id: string;
  task: string;
  status: string;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  last_turn_index: number;
  next_actor: string | null;
  repair_budget_per_turn: number;
  footer_mode: string;
  worktree_cleanup_mode: string;
  participants: Array<{
    role: string;
    harness: string;
    model: string;
  }>;
  last_event_id: string | null;
}

export interface ListAgentBridgeRunsOptions {
  limit?: number;
  cursor?: string;
}

export interface AgentBridgeRunListResponse {
  schema_version: 1;
  runs: AgentBridgeRun[];
  next_cursor?: string | null;
}

export class AgentBridgeAPI {
  constructor(private client: AragoraClient) {}

  /** List recorded agent-bridge runs. */
  async listRuns(options?: ListAgentBridgeRunsOptions): Promise<AgentBridgeRunListResponse> {
    const params: Record<string, unknown> = {};
    if (options?.limit !== undefined) {
      params.limit = options.limit;
    }
    if (options?.cursor !== undefined) {
      params.cursor = options.cursor;
    }
    return this.client.request<AgentBridgeRunListResponse>('GET', '/api/v1/agent-bridge/runs', {
      params,
    });
  }
}
