/**
 * Critiques Namespace API
 *
 * Provides access to debate critiques, patterns, and archives.
 */

/**
 * Critique entry
 */
export interface Critique {
  id: string;
  debate_id: string;
  agent_name: string;
  target_proposal_id: string;
  content: string;
  critique_type: 'logical' | 'factual' | 'ethical' | 'practical' | 'stylistic';
  severity: 'minor' | 'moderate' | 'major' | 'critical';
  addressed: boolean;
  created_at: string;
}

/**
 * Critique pattern - recurring critique themes
 */
export interface CritiquePattern {
  id: string;
  name: string;
  description: string;
  critique_type: string;
  frequency: number;
  examples: string[];
  agents: string[];
  first_seen: string;
  last_seen: string;
}

/**
 * Archived critique with debate context
 */
export interface ArchivedCritique extends Critique {
  debate_task: string;
  debate_result: string;
  resolution?: string;
  archived_at: string;
}

/**
 * Interface for the internal client used by CritiquesAPI.
 */
interface CritiquesClientInterface {
  get<T>(path: string): Promise<T>;
  request<T>(method: string, path: string, options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }): Promise<T>;
}

/**
 * Critiques API namespace.
 *
 * Provides methods for working with debate critiques:
 * - Browse critique archives
 * - Analyze critique patterns
 * - Track critique resolution
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Get critique patterns
 * const { patterns } = await client.critiques.getPatterns();
 *
 * // Browse archived critiques
 * const { critiques } = await client.critiques.getArchive({ limit: 50 });
 * ```
 */
export class CritiquesAPI {
  constructor(private client: CritiquesClientInterface) {}

  /**
   * Get recurring critique patterns across debates.
   */
  async getPatterns(options?: { limit?: number; agent?: string }): Promise<{ patterns: CritiquePattern[]; total: number }> {
    return this.client.request('GET', '/api/v1/critiques/patterns', { params: options });
  }

  /**
   * Get archived critiques from past debates.
   */
  async getArchive(options?: {
    limit?: number;
    offset?: number;
    critique_type?: string;
    severity?: string;
    agent?: string;
    addressed?: boolean;
  }): Promise<{ critiques: ArchivedCritique[]; total: number }> {
    return this.client.request('GET', '/api/v1/critiques/archive', { params: options });
  }
}
