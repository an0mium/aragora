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

export interface AgentBridgeActor {
  role: string;
  harness: string;
  model?: string;
  session_id?: string;
  worktree_path?: string;
  worktree_agent_slug?: string;
  branch?: string;
  harness_options?: Record<string, unknown>;
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

export interface StartAgentBridgeRunOptions {
  task: string;
  actors: AgentBridgeActor[];
  run_id?: string;
  next_actor?: string;
  worktree_path?: string;
  worktree_agent_slug?: string;
  repair_budget_per_turn?: number;
}

export interface AgentBridgeTurnRecord {
  schema_version: 1;
  event_id: string;
  run_id: string;
  ts: string;
  event_type: string;
  turn_index: number;
  role: string;
  harness: string;
  session_id: string | null;
  parse_status: string | null;
  payload: Record<string, unknown>;
}

export interface AutoStepAgentBridgeRunOptions {
  prompt?: string;
  context_turns?: number;
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

  /** Start an agent-bridge run without dispatching a turn. */
  async startRun(options: StartAgentBridgeRunOptions): Promise<AgentBridgeRun & { roles: Record<string, unknown> }> {
    return this.client.request<AgentBridgeRun & { roles: Record<string, unknown> }>(
      'POST',
      '/api/v1/agent-bridge/runs',
      { body: options }
    );
  }

  /** Dispatch one bridge turn to a run role. */
  async dispatchTurn(
    runId: string,
    options: { role: string; prompt: string }
  ): Promise<AgentBridgeTurnRecord> {
    return this.client.request<AgentBridgeTurnRecord>(
      'POST',
      `/api/v1/agent-bridge/runs/${runId}/dispatch`,
      { body: options }
    );
  }

  /** Dispatch one turn to the run's next_actor. */
  async autoStep(
    runId: string,
    options?: AutoStepAgentBridgeRunOptions
  ): Promise<AgentBridgeTurnRecord & { auto_step: { role: string; context_turns: number } }> {
    return this.client.request<
      AgentBridgeTurnRecord & { auto_step: { role: string; context_turns: number } }
    >('POST', `/api/v1/agent-bridge/runs/${runId}/auto-step`, {
      body: options ?? {},
    });
  }
}
