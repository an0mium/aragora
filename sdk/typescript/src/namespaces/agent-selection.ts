/**
 * Agent Selection Namespace API
 *
 * Provides methods for agent team selection operations:
 * - Plugin discovery and configuration
 * - Agent scoring for specific tasks
 * - Team selection with role assignment
 * - Selection history tracking
 */

// =============================================================================
// Plugin Types
// =============================================================================

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
  default_config?: Record<string, unknown>;
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
 * List plugins response.
 */
export interface ListPluginsResponse {
  plugins: SelectionPlugin[];
}

// =============================================================================
// Agent Scoring Types
// =============================================================================

/**
 * Agent score request (matches Python SDK).
 */
export interface ScoreAgentsRequest {
  /** List of agent identifiers to score */
  agents: string[];
  /** Task context or description for scoring */
  context?: string;
  /** Scoring dimensions (e.g., ["accuracy", "speed", "cost"]) */
  dimensions?: string[];
  /** Specific scorer plugin to use */
  scorer?: string;
  /** Custom weights for each dimension */
  weights?: Record<string, number>;
  /** Return only top K agents */
  top_k?: number;
}

/**
 * Agent score result.
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
 * Best agent request.
 */
export interface GetBestAgentRequest {
  /** List of candidate agent identifiers */
  pool: string[];
  /** Type of task (e.g., "code_review", "analysis", "creative") */
  task_type: string;
  /** Additional context for selection */
  context?: string;
}

/**
 * Best agent response.
 */
export interface GetBestAgentResponse {
  agent_id: string;
  agent_name: string;
  score: number;
  reasoning: string;
  task_type: string;
}

// =============================================================================
// Team Selection Types
// =============================================================================

/**
 * Team selection request (matches Python SDK).
 */
export interface SelectTeamRequest {
  /** List of candidate agent identifiers */
  pool: string[];
  /** Dict describing task needs (domain, complexity, etc.) */
  task_requirements?: Record<string, unknown>;
  /** Exact team size */
  team_size?: number;
  /** Additional selection constraints */
  constraints?: Record<string, unknown>;
  /** Minimum team size */
  min_team_size?: number;
  /** Maximum team size */
  max_team_size?: number;
  /** Roles that must be filled */
  required_roles?: string[];
  /** Agents to exclude from selection */
  excluded_agents?: string[];
  /** Weight for team diversity (0.0-1.0) */
  diversity_weight?: number;
  /** Specific team selector plugin to use */
  selector?: string;
  /** Specific role assigner plugin to use */
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

// =============================================================================
// Role Assignment Types
// =============================================================================

/**
 * Role assignment request.
 */
export interface AssignRolesRequest {
  /** List of agent identifiers to assign roles to */
  members: string[];
  /** List of roles to assign */
  roles: string[];
  /** Context for role assignment decisions */
  task_context?: string;
  /** Specific role assigner plugin to use */
  assigner?: string;
}

/**
 * Role assignment entry.
 */
export interface RoleAssignment {
  agent_id: string;
  agent_name: string;
  role: string;
  confidence: number;
  reasoning: string;
}

/**
 * Role assignment response.
 */
export interface AssignRolesResponse {
  assignments: RoleAssignment[];
  assigner_used: string;
  assignment_time_ms: number;
}

// =============================================================================
// History Types
// =============================================================================

/**
 * Selection history entry.
 */
export interface SelectionHistoryEntry {
  id: string;
  timestamp: string;
  selection_type: 'score' | 'team' | 'role';
  request: Record<string, unknown>;
  result: Record<string, unknown>;
  duration_ms: number;
}

/**
 * Selection history response.
 */
export interface SelectionHistoryResponse {
  history: SelectionHistoryEntry[];
  total: number;
  has_more: boolean;
}

/**
 * Interface for the internal client methods used by AgentSelectionAPI.
 */
export interface AgentSelectionClientInterface {
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
 * - List available selection plugins
 * - Get default plugin configurations
 * - Score agents for specific tasks
 * - Select optimal teams with role assignment
 * - Track selection history
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
 *   agents: ['claude', 'gpt-4', 'gemini'],
 *   context: 'security code review',
 *   dimensions: ['accuracy', 'speed', 'cost'],
 * });
 *
 * // Get the best agent
 * const best = await client.agentSelection.getBestAgent({
 *   pool: ['claude', 'gpt-4', 'gemini'],
 *   task_type: 'code_review',
 * });
 *
 * // Select a team
 * const team = await client.agentSelection.selectTeam({
 *   pool: ['claude', 'gpt-4', 'gemini', 'mistral'],
 *   task_requirements: { domain: 'security', complexity: 'high' },
 *   team_size: 3,
 * });
 *
 * // Assign roles to team members
 * const roles = await client.agentSelection.assignRoles({
 *   members: ['claude', 'gpt-4'],
 *   roles: ['lead', 'reviewer'],
 * });
 *
 * // Get selection history
 * const history = await client.agentSelection.getHistory({ limit: 10 });
 * ```
 */
export class AgentSelectionAPI {
  constructor(private client: AgentSelectionClientInterface) {}

  // ===========================================================================
  // Plugin Discovery
  // ===========================================================================

  /**
   * List all available selection plugins.
   */
  async listPlugins(): Promise<ListPluginsResponse> {
    return this.client.request<ListPluginsResponse>('GET', '/api/v1/agent-selection/plugins');
  }

  /**
   * Get default plugin configuration.
   */
  async getDefaults(): Promise<DefaultPluginConfig> {
    return this.client.request<DefaultPluginConfig>('GET', '/api/v1/agent-selection/defaults');
  }

  // ===========================================================================
  // Agent Scoring
  // ===========================================================================

  /**
   * Score agents for a specific task or context.
   */
  async scoreAgents(options: ScoreAgentsRequest): Promise<ScoreAgentsResponse> {
    const data: Record<string, unknown> = { agents: options.agents };
    if (options.context !== undefined) data.context = options.context;
    if (options.dimensions !== undefined) data.dimensions = options.dimensions;
    if (options.scorer !== undefined) data.scorer = options.scorer;
    if (options.weights !== undefined) data.weights = options.weights;
    if (options.top_k !== undefined) data.top_k = options.top_k;

    return this.client.request<ScoreAgentsResponse>('POST', '/api/v1/agent-selection/score', {
      json: data,
    });
  }

  /**
   * Get the best agent for a specific task from a pool.
   */
  async getBestAgent(options: GetBestAgentRequest): Promise<GetBestAgentResponse> {
    const data: Record<string, unknown> = {
      pool: options.pool,
      task_type: options.task_type,
    };
    if (options.context !== undefined) data.context = options.context;

    return this.client.request<GetBestAgentResponse>('POST', '/api/v1/agent-selection/best', {
      json: data,
    });
  }

  // ===========================================================================
  // Team Selection
  // ===========================================================================

  /**
   * Select an optimal team of agents for a task.
   */
  async selectTeam(options: SelectTeamRequest): Promise<SelectTeamResponse> {
    const data: Record<string, unknown> = { pool: options.pool };
    if (options.task_requirements !== undefined) data.task_requirements = options.task_requirements;
    if (options.team_size !== undefined) data.team_size = options.team_size;
    if (options.constraints !== undefined) data.constraints = options.constraints;
    if (options.min_team_size !== undefined) data.min_team_size = options.min_team_size;
    if (options.max_team_size !== undefined) data.max_team_size = options.max_team_size;
    if (options.required_roles !== undefined) data.required_roles = options.required_roles;
    if (options.excluded_agents !== undefined) data.excluded_agents = options.excluded_agents;
    if (options.diversity_weight !== undefined) data.diversity_weight = options.diversity_weight;
    if (options.selector !== undefined) data.selector = options.selector;
    if (options.role_assigner !== undefined) data.role_assigner = options.role_assigner;

    return this.client.request<SelectTeamResponse>('POST', '/api/v1/agent-selection/select-team', {
      json: data,
    });
  }

  /**
   * Assign roles to a set of team members.
   */
  async assignRoles(options: AssignRolesRequest): Promise<AssignRolesResponse> {
    const data: Record<string, unknown> = {
      members: options.members,
      roles: options.roles,
    };
    if (options.task_context !== undefined) data.task_context = options.task_context;
    if (options.assigner !== undefined) data.assigner = options.assigner;

    return this.client.request<AssignRolesResponse>('POST', '/api/v1/agent-selection/assign-roles', {
      json: data,
    });
  }

  // ===========================================================================
  // History
  // ===========================================================================

  /**
   * Get agent selection history.
   */
  async getHistory(options?: { limit?: number; since?: string }): Promise<SelectionHistoryResponse> {
    const params: Record<string, unknown> = {};
    if (options?.limit !== undefined) params.limit = options.limit;
    if (options?.since !== undefined) params.since = options.since;

    return this.client.request<SelectionHistoryResponse>('GET', '/api/v1/agent-selection/history', {
      params,
    });
  }

  // ===========================================================================
  // Plugin Details
  // ===========================================================================

  /**
   * Get details for a specific scorer plugin.
   */
  async getScorer(name: string): Promise<ScorerPlugin> {
    return this.client.request<ScorerPlugin>(
      'GET',
      `/api/v1/selection/scorers/${encodeURIComponent(name)}`
    );
  }

  /**
   * Get details for a specific team selector plugin.
   */
  async getTeamSelector(name: string): Promise<TeamSelectorPlugin> {
    return this.client.request<TeamSelectorPlugin>(
      'GET',
      `/api/v1/selection/team-selectors/${encodeURIComponent(name)}`
    );
  }

  /**
   * Get details for a specific role assigner plugin.
   */
  async getRoleAssigner(name: string): Promise<RoleAssignerPlugin> {
    return this.client.request<RoleAssignerPlugin>(
      'GET',
      `/api/v1/selection/role-assigners/${encodeURIComponent(name)}`
    );
  }

  // ===========================================================================
  // Convenience Methods
  // ===========================================================================

  /**
   * List all scorer plugins via the plugins endpoint.
   */
  async listScorers(): Promise<{ scorers: ScorerPlugin[] }> {
    return this.client.request<{ scorers: ScorerPlugin[] }>(
      'GET',
      '/api/v1/selection/plugins',
      { params: { type: 'scorers' } }
    );
  }

  /**
   * List all team selector plugins via the plugins endpoint.
   */
  async listTeamSelectors(): Promise<{ selectors: TeamSelectorPlugin[] }> {
    return this.client.request<{ selectors: TeamSelectorPlugin[] }>(
      'GET',
      '/api/v1/selection/plugins',
      { params: { type: 'team-selectors' } }
    );
  }

  /**
   * List all role assigner plugins via the plugins endpoint.
   */
  async listRoleAssigners(): Promise<{ assigners: RoleAssignerPlugin[] }> {
    return this.client.request<{ assigners: RoleAssignerPlugin[] }>(
      'GET',
      '/api/v1/selection/plugins',
      { params: { type: 'role-assigners' } }
    );
  }

  /**
   * Select a team with alternative team suggestions.
   */
  async selectTeamWithAlternatives(
    options: SelectTeamRequest,
    alternativeCount: number = 2
  ): Promise<{
    primary: SelectTeamResponse;
    alternatives: SelectTeamResponse[];
  }> {
    const primary = await this.selectTeam(options);
    const alternatives: SelectTeamResponse[] = [];
    let excludedAgents = options.excluded_agents || [];

    for (let i = 0; i < alternativeCount; i++) {
      const previousTeamIds = primary.team.map(m => m.agent_id);
      excludedAgents = Array.from(new Set([...excludedAgents, ...previousTeamIds]));

      try {
        const altTeam = await this.selectTeam({
          ...options,
          excluded_agents: excludedAgents,
        });
        alternatives.push(altTeam);
        excludedAgents = [...excludedAgents, ...altTeam.team.map(m => m.agent_id)];
      } catch {
        break;
      }
    }

    return { primary, alternatives };
  }
}

export default AgentSelectionAPI;
