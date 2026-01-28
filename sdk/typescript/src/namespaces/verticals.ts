/**
 * Verticals Namespace API
 *
 * Provides domain-specific vertical management:
 * - List and configure industry verticals
 * - Get vertical-specific tools and compliance
 * - Create specialized agents and debates
 */

/**
 * Vertical identifier.
 */
export type VerticalId =
  | 'software'
  | 'legal'
  | 'healthcare'
  | 'accounting'
  | 'finance'
  | 'research'
  | 'education'
  | 'marketing'
  | 'operations'
  | 'general';

/**
 * Compliance enforcement level.
 */
export type ComplianceLevel = 'advisory' | 'warning' | 'enforced';

/**
 * Model configuration.
 */
export interface ModelConfig {
  default_model: string;
  fallback_models: string[];
  temperature: number;
  max_tokens: number;
  top_p?: number;
}

/**
 * Tool configuration.
 */
export interface ToolConfig {
  name: string;
  description: string;
  enabled: boolean;
  parameters: Record<string, unknown>;
  requires_auth: boolean;
  rate_limit_per_minute?: number;
}

/**
 * Compliance rule.
 */
export interface ComplianceRule {
  id: string;
  name: string;
  description: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  check_type: 'regex' | 'keyword' | 'semantic' | 'custom';
  pattern?: string;
  action: 'block' | 'warn' | 'log' | 'redact';
}

/**
 * Compliance framework.
 */
export interface ComplianceFramework {
  framework: string;
  version: string;
  level: ComplianceLevel;
  rules: ComplianceRule[];
  exemptions: string[];
  last_updated: string;
}

/**
 * Vertical summary (list view).
 */
export interface VerticalSummary {
  vertical_id: VerticalId;
  display_name: string;
  description: string;
  expertise_areas: string[];
  tools: string[];
  agent_count: number;
  enabled: boolean;
}

/**
 * Vertical details (full view).
 */
export interface VerticalDetails extends VerticalSummary {
  compliance_frameworks: string[];
  model_config: ModelConfig;
  default_traits: string[];
  created_at: string;
  updated_at: string;
}

/**
 * Vertical suggestion.
 */
export interface VerticalSuggestion {
  vertical_id: VerticalId;
  display_name: string;
  confidence: number;
  reasoning: string;
  matched_keywords: string[];
}

/**
 * Create agent request.
 */
export interface CreateVerticalAgentRequest {
  name: string;
  model?: string;
  role?: string;
  traits?: string[];
  description?: string;
}

/**
 * Create agent response.
 */
export interface CreateVerticalAgentResponse {
  agent_name: string;
  vertical_id: VerticalId;
  model: string;
  role: string;
  created_at: string;
}

/**
 * Create debate request.
 */
export interface CreateVerticalDebateRequest {
  topic: string;
  agent_name?: string;
  rounds?: number;
  consensus?: 'majority' | 'unanimous' | 'weighted';
  additional_agents?: string[];
  context?: string;
}

/**
 * Create debate response.
 */
export interface CreateVerticalDebateResponse {
  debate_id: string;
  vertical_id: VerticalId;
  topic: string;
  agents: string[];
  rounds: number;
  status: 'created' | 'running' | 'completed';
  created_at: string;
}

/**
 * Update vertical config request.
 */
export interface UpdateVerticalConfigRequest {
  enabled?: boolean;
  model_config?: Partial<ModelConfig>;
  default_traits?: string[];
  description?: string;
}

/**
 * Client interface for verticals operations.
 */
interface VerticalsClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Verticals API namespace.
 *
 * Provides methods for managing industry verticals:
 * - List and configure verticals
 * - Get vertical-specific tools and compliance frameworks
 * - Create specialized agents and debates
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // List all verticals
 * const { verticals } = await client.verticals.list();
 *
 * // Get vertical-specific tools
 * const { tools } = await client.verticals.getTools('legal');
 *
 * // Suggest best vertical for a task
 * const suggestions = await client.verticals.suggest('Review this contract');
 *
 * // Create a vertical-specialized debate
 * const debate = await client.verticals.createDebate('healthcare', {
 *   topic: 'Evaluate treatment options',
 *   rounds: 3,
 * });
 * ```
 */
export class VerticalsAPI {
  constructor(private client: VerticalsClientInterface) {}

  /**
   * List all available verticals.
   */
  async list(options?: { keyword?: string }): Promise<{ verticals: VerticalSummary[] }> {
    return this.client.request('GET', '/api/verticals', {
      params: options as unknown as Record<string, unknown>,
    });
  }

  /**
   * Get vertical details.
   */
  async get(verticalId: VerticalId): Promise<VerticalDetails> {
    return this.client.request('GET', `/api/verticals/${encodeURIComponent(verticalId)}`);
  }

  /**
   * Get tools available for a vertical.
   */
  async getTools(verticalId: VerticalId): Promise<{ tools: ToolConfig[] }> {
    return this.client.request(
      'GET',
      `/api/verticals/${encodeURIComponent(verticalId)}/tools`
    );
  }

  /**
   * Get compliance frameworks for a vertical.
   */
  async getCompliance(
    verticalId: VerticalId,
    options?: { level?: ComplianceLevel }
  ): Promise<{ frameworks: ComplianceFramework[] }> {
    return this.client.request(
      'GET',
      `/api/verticals/${encodeURIComponent(verticalId)}/compliance`,
      {
        params: options as unknown as Record<string, unknown>,
      }
    );
  }

  /**
   * Suggest the best vertical for a task.
   */
  async suggest(task: string): Promise<{ suggestions: VerticalSuggestion[] }> {
    return this.client.request('GET', '/api/verticals/suggest', {
      params: { task },
    });
  }

  /**
   * Create a vertical-specialized agent.
   */
  async createAgent(
    verticalId: VerticalId,
    request: CreateVerticalAgentRequest
  ): Promise<CreateVerticalAgentResponse> {
    return this.client.request(
      'POST',
      `/api/verticals/${encodeURIComponent(verticalId)}/agent`,
      {
        json: request as unknown as Record<string, unknown>,
      }
    );
  }

  /**
   * Create a vertical-specialized debate.
   */
  async createDebate(
    verticalId: VerticalId,
    request: CreateVerticalDebateRequest
  ): Promise<CreateVerticalDebateResponse> {
    return this.client.request(
      'POST',
      `/api/verticals/${encodeURIComponent(verticalId)}/debate`,
      {
        json: request as unknown as Record<string, unknown>,
      }
    );
  }

  /**
   * Update vertical configuration.
   */
  async updateConfig(
    verticalId: VerticalId,
    updates: UpdateVerticalConfigRequest
  ): Promise<VerticalDetails> {
    return this.client.request(
      'PUT',
      `/api/verticals/${encodeURIComponent(verticalId)}/config`,
      {
        json: updates as unknown as Record<string, unknown>,
      }
    );
  }
}
