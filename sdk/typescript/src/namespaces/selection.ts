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
      '/api/v1/selection/best',
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
      '/api/v1/selection/history',
      { params: options }
    );
    return response.history;
  }
}
