/**
 * Persona Namespace API
 *
 * Provides agent identity and persona management:
 * - List and manage agent personas
 * - Get grounded persona with ELO ratings
 * - Track performance and domain expertise
 * - Generate identity prompts
 */

/**
 * Persona trait.
 */
export type PersonaTrait =
  | 'analytical'
  | 'creative'
  | 'cautious'
  | 'bold'
  | 'collaborative'
  | 'independent'
  | 'detail_oriented'
  | 'big_picture'
  | 'evidence_focused'
  | 'intuitive';

/**
 * Expertise domain.
 */
export type ExpertiseDomain =
  | 'software'
  | 'legal'
  | 'healthcare'
  | 'finance'
  | 'research'
  | 'education'
  | 'marketing'
  | 'operations'
  | 'strategy'
  | 'general';

/**
 * Identity prompt section.
 */
export type IdentitySection =
  | 'core'
  | 'traits'
  | 'expertise'
  | 'performance'
  | 'guidelines'
  | 'constraints';

/**
 * Persona definition.
 */
export interface Persona {
  agent_name: string;
  description?: string;
  traits: PersonaTrait[];
  expertise: ExpertiseDomain[];
  model?: string;
  temperature?: number;
  created_at: string;
  updated_at: string;
}

/**
 * Grounded persona with performance metrics.
 */
export interface GroundedPersona {
  agent_name: string;
  elo: number;
  domain_elos: Record<string, number>;
  win_rate: number;
  calibration_score: number;
  position_accuracy: number;
  debates_count: number;
  last_active?: string;
}

/**
 * Available persona options.
 */
export interface PersonaOptions {
  traits: PersonaTrait[];
  expertise_domains: ExpertiseDomain[];
  models: string[];
  temperature_range: {
    min: number;
    max: number;
    default: number;
  };
}

/**
 * Performance summary.
 */
export interface PerformanceSummary {
  agent_name: string;
  debates_participated: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  avg_confidence: number;
  avg_calibration: number;
  streak: {
    type: 'win' | 'loss' | 'none';
    count: number;
  };
}

/**
 * Domain expertise entry.
 */
export interface DomainExpertise {
  domain: string;
  elo: number;
  calibration_score: number;
  debates_count: number;
  win_rate: number;
  last_debate?: string;
}

/**
 * Position accuracy metrics.
 */
export interface PositionAccuracy {
  agent_name: string;
  total_positions: number;
  verified_positions: number;
  correct_positions: number;
  accuracy_rate: number;
  by_domain: Record<string, {
    total: number;
    correct: number;
    accuracy: number;
  }>;
}

/**
 * Identity prompt response.
 */
export interface IdentityPrompt {
  agent_name: string;
  prompt: string;
  sections_included: IdentitySection[];
  token_count: number;
}

/**
 * Create persona request.
 */
export interface CreatePersonaRequest {
  agent_name: string;
  description?: string;
  traits?: PersonaTrait[];
  expertise?: ExpertiseDomain[];
  model?: string;
  temperature?: number;
}

/**
 * Update persona request.
 */
export interface UpdatePersonaRequest {
  description?: string;
  traits?: PersonaTrait[];
  expertise?: ExpertiseDomain[];
  model?: string;
  temperature?: number;
}

/**
 * Client interface for persona operations.
 */
interface PersonaClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Persona API namespace.
 *
 * Provides methods for managing agent personas:
 * - List and create personas
 * - Get grounded persona with ELO ratings
 * - Track performance and domain expertise
 * - Generate identity prompts
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // List all personas
 * const { personas } = await client.persona.list();
 *
 * // Get grounded persona with metrics
 * const grounded = await client.persona.getGrounded('claude-analyst');
 *
 * // Generate identity prompt
 * const { prompt } = await client.persona.getIdentityPrompt('claude-analyst', {
 *   sections: ['core', 'traits', 'expertise'],
 * });
 * ```
 */
export class PersonaAPI {
  constructor(private client: PersonaClientInterface) {}

  /**
   * List all personas.
   */
  async list(): Promise<{ personas: Persona[]; total: number }> {
    return this.client.request('GET', '/api/personas');
  }

  /**
   * Get available persona options (traits, domains, models).
   */
  async getOptions(): Promise<PersonaOptions> {
    return this.client.request('GET', '/api/personas/options');
  }

  /**
   * Get a persona by agent name.
   */
  async get(agentName: string): Promise<Persona> {
    return this.client.request('GET', `/api/agent/${encodeURIComponent(agentName)}/persona`);
  }

  /**
   * Get grounded persona with performance metrics.
   */
  async getGrounded(agentName: string): Promise<GroundedPersona> {
    return this.client.request(
      'GET',
      `/api/agent/${encodeURIComponent(agentName)}/grounded-persona`
    );
  }

  /**
   * Generate identity prompt for an agent.
   */
  async getIdentityPrompt(
    agentName: string,
    options?: { sections?: IdentitySection[] }
  ): Promise<IdentityPrompt> {
    return this.client.request(
      'GET',
      `/api/agent/${encodeURIComponent(agentName)}/identity-prompt`,
      {
        params: options?.sections ? { sections: options.sections.join(',') } : undefined,
      }
    );
  }

  /**
   * Get performance summary for an agent.
   */
  async getPerformance(agentName: string): Promise<PerformanceSummary> {
    return this.client.request(
      'GET',
      `/api/agent/${encodeURIComponent(agentName)}/performance`
    );
  }

  /**
   * Get domain expertise for an agent.
   */
  async getDomains(
    agentName: string,
    options?: { limit?: number }
  ): Promise<{ domains: DomainExpertise[] }> {
    return this.client.request(
      'GET',
      `/api/agent/${encodeURIComponent(agentName)}/domains`,
      {
        params: options as Record<string, unknown>,
      }
    );
  }

  /**
   * Get position accuracy metrics for an agent.
   */
  async getAccuracy(agentName: string): Promise<PositionAccuracy> {
    return this.client.request(
      'GET',
      `/api/agent/${encodeURIComponent(agentName)}/accuracy`
    );
  }

  /**
   * Create a new persona.
   */
  async create(request: CreatePersonaRequest): Promise<Persona> {
    return this.client.request('POST', '/api/personas', {
      json: request as unknown as Record<string, unknown>,
    });
  }

  /**
   * Update an existing persona.
   */
  async update(agentName: string, updates: UpdatePersonaRequest): Promise<Persona> {
    return this.client.request(
      'PUT',
      `/api/agent/${encodeURIComponent(agentName)}/persona`,
      {
        json: updates as unknown as Record<string, unknown>,
      }
    );
  }

  /**
   * Delete a persona.
   */
  async delete(agentName: string): Promise<{ success: boolean; message: string }> {
    return this.client.request(
      'DELETE',
      `/api/agent/${encodeURIComponent(agentName)}/persona`
    );
  }
}
