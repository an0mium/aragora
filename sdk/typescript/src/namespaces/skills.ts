/**
 * Skills Namespace API
 *
 * Provides skill management and invocation:
 * - List available skills
 * - Get skill details and metrics
 * - Invoke skills with input data
 */

/**
 * Skill capability types.
 */
export type SkillCapability =
  | 'READ_LOCAL'
  | 'WRITE_LOCAL'
  | 'READ_DATABASE'
  | 'WRITE_DATABASE'
  | 'EXTERNAL_API'
  | 'WEB_SEARCH'
  | 'WEB_FETCH'
  | 'CODE_EXECUTION'
  | 'SHELL_EXECUTION'
  | 'LLM_INFERENCE'
  | 'EMBEDDING'
  | 'DEBATE_CONTEXT'
  | 'EVIDENCE_COLLECTION'
  | 'KNOWLEDGE_QUERY'
  | 'SYSTEM_INFO'
  | 'NETWORK';

/**
 * Skill execution status.
 */
export type SkillStatus =
  | 'SUCCESS'
  | 'FAILURE'
  | 'PARTIAL'
  | 'TIMEOUT'
  | 'RATE_LIMITED'
  | 'PERMISSION_DENIED'
  | 'INVALID_INPUT'
  | 'NOT_IMPLEMENTED';

/**
 * Skill manifest.
 */
export interface SkillManifest {
  name: string;
  version: string;
  description?: string;
  capabilities: SkillCapability[];
  input_schema: Record<string, unknown>;
  output_schema?: Record<string, unknown>;
  tags?: string[];
  rate_limit_per_minute?: number;
  timeout_seconds?: number;
}

/**
 * Skill details (extended manifest).
 */
export interface SkillDetails extends SkillManifest {
  author?: string;
  required_permissions?: string[];
  required_env_vars?: string[];
  max_execution_time_seconds?: number;
  debate_compatible?: boolean;
  requires_debate_context?: boolean;
}

/**
 * Skill metrics.
 */
export interface SkillMetrics {
  skill: string;
  total_invocations: number;
  successful_invocations: number;
  failed_invocations: number;
  average_latency_ms: number;
  last_invoked?: string;
}

/**
 * Invoke skill request.
 */
export interface InvokeSkillRequest {
  /** Skill name to invoke */
  skill: string;
  /** Input data for the skill */
  input?: Record<string, unknown>;
  /** User ID for RBAC */
  user_id?: string;
  /** User permissions */
  permissions?: string[];
  /** Additional metadata/config */
  metadata?: Record<string, unknown>;
  /** Timeout in seconds (default: 30, max: 60) */
  timeout?: number;
}

/**
 * Invoke skill result.
 */
export interface InvokeSkillResult {
  status: 'success' | 'error';
  output?: Record<string, unknown>;
  error?: string;
  execution_time_ms?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Client interface for skills operations.
 */
interface SkillsClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Skills API namespace.
 *
 * Provides methods for managing and invoking skills:
 * - List all available skills
 * - Get skill details and metrics
 * - Invoke skills with custom input
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // List all skills
 * const { skills } = await client.skills.list();
 *
 * // Get skill details
 * const skill = await client.skills.get('web-search');
 *
 * // Invoke a skill
 * const result = await client.skills.invoke({
 *   skill: 'web-search',
 *   input: { query: 'TypeScript best practices' },
 * });
 * ```
 */
export class SkillsAPI {
  constructor(private client: SkillsClientInterface) {}

  /**
   * List all available skills.
   */
  async list(): Promise<{ skills: SkillManifest[]; total: number }> {
    return this.client.request('GET', '/api/skills');
  }

  /**
   * Get skill details.
   */
  async get(name: string): Promise<SkillDetails> {
    return this.client.request('GET', `/api/skills/${encodeURIComponent(name)}`);
  }

  /**
   * Get skill metrics.
   */
  async getMetrics(name: string): Promise<SkillMetrics> {
    return this.client.request('GET', `/api/skills/${encodeURIComponent(name)}/metrics`);
  }

  /**
   * Invoke a skill.
   */
  async invoke(request: InvokeSkillRequest): Promise<InvokeSkillResult> {
    return this.client.request('POST', '/api/skills/invoke', {
      json: request as unknown as Record<string, unknown>,
    });
  }
}
