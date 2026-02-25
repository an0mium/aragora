/**
 * Selection Namespace API
 *
 * Provides endpoints for agent selection including
 * scoring, team composition, and role assignment.
 */

import type { AragoraClient } from '../client';

/** Agent score from the selection process */
export interface SelectionScore {
  agent_id: string;
  agent_name: string;
  score: number;
  factors: Record<string, number>;
}

/** Selected team member */
export interface SelectedTeamMember {
  agent_id: string;
  agent_name: string;
  role: string;
  score: number;
}

/** Selection history entry */
export interface SelectionHistoryItem {
  id: string;
  debate_id: string;
  selected_agents: string[];
  strategy: string;
  scores: SelectionScore[];
  created_at: string;
}

/** Team selection request */
export interface SelectTeamRequest {
  task: string;
  team_size?: number;
  required_roles?: string[];
  domain?: string;
  strategy?: string;
}

/** Team selection result */
export interface SelectTeamResult {
  members: SelectedTeamMember[];
  strategy_used: string;
  selection_time_ms: number;
}

/**
 * Selection namespace for agent team composition.
 *
 * @example
 * ```typescript
 * const team = await client.selection.selectTeam({
 *   task: 'API design review',
 *   team_size: 3,
 * });
 * console.log(team.members.map(m => m.agent_name));
 * ```
 */
export class SelectionNamespace {
  constructor(private client: AragoraClient) {}

  /** Score agents for a given task. */
  async scoreAgents(options: {
    task: string;
    domain?: string;
    agent_ids?: string[];
  }): Promise<SelectionScore[]> {
    const response = await this.client.request<{ scores: SelectionScore[] }>(
      'POST',
      '/api/v1/selection/score',
      { body: options }
    );
    return response.scores;
  }

  /** Select a team for a task. */
  async selectTeam(request: SelectTeamRequest): Promise<SelectTeamResult> {
    return this.client.request<SelectTeamResult>(
      'POST',
      '/api/v1/selection/team',
      { body: request }
    );
  }

  /** Get the best agent for a task. */
  async getBestAgent(options: {
    task: string;
    domain?: string;
  }): Promise<SelectionScore> {
    return this.client.request<SelectionScore>(
      'POST',
      '/api/v1/agent-selection/best',
      { body: options }
    );
  }

  /** Get selection history. */
  async getHistory(options?: {
    limit?: number;
    offset?: number;
  }): Promise<SelectionHistoryItem[]> {
    const response = await this.client.request<{ history: SelectionHistoryItem[] }>(
      'GET',
      '/api/v1/agent-selection/history',
      { params: options }
    );
    return response.history;
  }

  /**
   * Get team selection.
   */
  async teamSelection(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/team-selection', { params }) as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Plugin Listings
  // ===========================================================================

  /**
   * List all available scorer plugins.
   *
   * @route GET /api/v1/selection/scorers
   *
   * @example
   * ```typescript
   * const scorers = await client.selection.listScorers();
   * console.log(scorers);
   * ```
   */
  async listScorers(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/selection/scorers') as Promise<Record<string, unknown>>;
  }

  /**
   * List all available team selector plugins.
   *
   * @route GET /api/v1/selection/team-selectors
   *
   * @example
   * ```typescript
   * const selectors = await client.selection.listTeamSelectors();
   * console.log(selectors);
   * ```
   */
  async listTeamSelectors(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/selection/team-selectors') as Promise<Record<string, unknown>>;
  }

  /**
   * List all available role assigner plugins.
   *
   * @route GET /api/v1/selection/role-assigners
   *
   * @example
   * ```typescript
   * const assigners = await client.selection.listRoleAssigners();
   * console.log(assigners);
   * ```
   */
  async listRoleAssigners(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/selection/role-assigners') as Promise<Record<string, unknown>>;
  }
}
