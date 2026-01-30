/**
 * Agent Selection Namespace API
 *
 * Provides a namespaced interface for agent team selection operations.
 * This wraps the flat client methods for scoring, team composition, and plugin management.
 */

/**
 * Selection plugin information.
 */
export interface SelectionPlugin {
  name: string;
  type: 'scorer' | 'team_selector' | 'role_assigner';
  description: string;
  version: string;
  enabled: boolean;
  config_schema: Record<string, unknown>;
  default_config: Record<string, unknown>;
}

/**
 * Scorer plugin details.
 */
export interface ScorerPlugin extends SelectionPlugin {
  type: 'scorer';
  metrics: string[];
  weight_range: { min: number; max: number };
}

/**
 * Team selector plugin details.
 */
export interface TeamSelectorPlugin extends SelectionPlugin {
  type: 'team_selector';
  strategies: string[];
  supports_constraints: boolean;
}

/**
 * Role assigner plugin details.
 */
export interface RoleAssignerPlugin extends SelectionPlugin {
  type: 'role_assigner';
  roles: string[];
  assignment_strategies: string[];
}

/**
 * Default plugin configuration.
 */
export interface DefaultPluginConfig {
  scorer: string;
  team_selector: string;
  role_assigner: string;
  scorer_weights: Record<string, number>;
}

/**
 * Agent score request.
 */
export interface ScoreAgentsRequest {
  task: string;
  context?: string;
  candidates?: string[];
  scorer?: string;
  weights?: Record<string, number>;
  top_k?: number;
}

/**
 * Agent score.
 */
export interface AgentScore {
  agent_id: string;
  agent_name: string;
  overall_score: number;
  dimension_scores: Record<string, number>;
  confidence: number;
  reasoning: string;
}

/**
 * Score agents response.
 */
export interface ScoreAgentsResponse {
  scores: AgentScore[];
  scorer_used: string;
  scoring_time_ms: number;
}

/**
 * Team selection request.
 */
export interface SelectTeamRequest {
  task: string;
  context?: string;
  team_size?: number;
  min_team_size?: number;
  max_team_size?: number;
  required_roles?: string[];
  excluded_agents?: string[];
  diversity_weight?: number;
  selector?: string;
  role_assigner?: string;
}

/**
 * Team member with assigned role.
 */
export interface TeamMember {
  agent_id: string;
  agent_name: string;
  role: string;
  score: number;
  strengths: string[];
  selection_reasoning: string;
}

/**
 * Team selection response.
 */
export interface SelectTeamResponse {
  team: TeamMember[];
  team_score: number;
  diversity_score: number;
  coverage_score: number;
  selector_used: string;
  role_assigner_used: string;
  selection_time_ms: number;
  alternatives?: TeamMember[][];
}

/**
 * Interface for the internal client methods used by AgentSelectionAPI.
 */
interface AgentSelectionClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: unknown }
  ): Promise<T>;
}

/**
 * Agent Selection API namespace.
 *
 * Provides methods for agent team selection and scoring:
 * - Listing available selection plugins
 * - Scoring agents for specific tasks
 * - Selecting optimal teams with role assignment
 * - Managing plugin configurations
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // List available plugins
 * const { plugins } = await client.agentSelection.listPlugins();
 *
 * // Score agents for a task
 * const scores = await client.agentSelection.scoreAgents({
 *   task: 'Code review for security vulnerabilities',
 *   top_k: 5,
 * });
 *
 * // Select a team
 * const team = await client.agentSelection.selectTeam({
 *   task: 'Design system architecture',
 *   team_size: 4,
 *   required_roles: ['architect', 'security_expert', 'domain_expert'],
 * });
 * ```
 */
export class AgentSelectionAPI {
  constructor(private client: AgentSelectionClientInterface) {}

  /**
   * List all available selection plugins.
   */
  async listPlugins(): Promise<{ plugins: SelectionPlugin[] }> {
    return this.client.request<{ plugins: SelectionPlugin[] }>('GET', '/api/v1/selection/plugins');
  }

  /**
   * Get default plugin configuration.
   */
  async getDefaults(): Promise<DefaultPluginConfig> {
    return this.client.request<DefaultPluginConfig>('GET', '/api/v1/selection/defaults');
  }

  /**
   * Get details for a specific scorer plugin.
   */
  async getScorer(name: string): Promise<ScorerPlugin> {
    return this.client.request<ScorerPlugin>('GET', `/api/v1/selection/scorers/${name}`);
  }

  /**
   * Get details for a specific team selector plugin.
   */
  async getTeamSelector(name: string): Promise<TeamSelectorPlugin> {
    return this.client.request<TeamSelectorPlugin>('GET', `/api/v1/selection/team-selectors/${name}`);
  }

  /**
   * Get details for a specific role assigner plugin.
   */
  async getRoleAssigner(name: string): Promise<RoleAssignerPlugin> {
    return this.client.request<RoleAssignerPlugin>('GET', `/api/v1/selection/role-assigners/${name}`);
  }

  /**
   * Score agents for a specific task.
   */
  async scoreAgents(request: ScoreAgentsRequest): Promise<ScoreAgentsResponse> {
    return this.client.request<ScoreAgentsResponse>('POST', '/api/v1/selection/score', {
      json: request,
    });
  }

  /**
   * Select an optimal team for a task.
   */
  async selectTeam(request: SelectTeamRequest): Promise<SelectTeamResponse> {
    return this.client.request<SelectTeamResponse>('POST', '/api/v1/selection/team', {
      json: request,
    });
  }

  /**
   * Convenience method to get the best agent for a task.
   */
  async getBestAgent(task: string, context?: string): Promise<AgentScore | null> {
    const response = await this.scoreAgents({ task, context, top_k: 1 });
    return response.scores[0] || null;
  }

  /**
   * Convenience method to select a team with alternatives.
   */
  async selectTeamWithAlternatives(
    request: SelectTeamRequest,
    alternativeCount: number = 2
  ): Promise<{
    primary: SelectTeamResponse;
    alternatives: SelectTeamResponse[];
  }> {
    // Select primary team
    const primary = await this.selectTeam(request);

    // Select alternatives by excluding primary team members
    const alternatives: SelectTeamResponse[] = [];
    let excludedAgents = request.excluded_agents || [];

    for (let i = 0; i < alternativeCount; i++) {
      // Add previous team members to exclusion list
      const previousTeamIds = primary.team.map(m => m.agent_id);
      excludedAgents = [...new Set([...excludedAgents, ...previousTeamIds])];

      try {
        const altTeam = await this.selectTeam({
          ...request,
          excluded_agents: excludedAgents,
        });
        alternatives.push(altTeam);

        // Add this team's members to exclusion for next iteration
        excludedAgents = [...excludedAgents, ...altTeam.team.map(m => m.agent_id)];
      } catch {
        // Not enough agents for more alternatives
        break;
      }
    }

    return { primary, alternatives };
  }

  /**
   * List all scorer plugins.
   */
  async listScorers(): Promise<{ scorers: ScorerPlugin[] }> {
    const { plugins } = await this.listPlugins();
    return {
      scorers: plugins.filter((p): p is ScorerPlugin => p.type === 'scorer'),
    };
  }

  /**
   * List all team selector plugins.
   */
  async listTeamSelectors(): Promise<{ selectors: TeamSelectorPlugin[] }> {
    const { plugins } = await this.listPlugins();
    return {
      selectors: plugins.filter((p): p is TeamSelectorPlugin => p.type === 'team_selector'),
    };
  }

  /**
   * List all role assigner plugins.
   */
  async listRoleAssigners(): Promise<{ assigners: RoleAssignerPlugin[] }> {
    const { plugins } = await this.listPlugins();
    return {
      assigners: plugins.filter((p): p is RoleAssignerPlugin => p.type === 'role_assigner'),
    };
  }
}

export default AgentSelectionAPI;
