/**
 * Introspection Namespace API
 *
 * Provides access to system introspection and agent information.
 * Useful for debugging, monitoring, and understanding system state.
 */

/**
 * Agent information from introspection
 */
export interface AgentInfo {
  name: string;
  type: string;
  provider: string;
  model: string;
  available: boolean;
  capabilities: string[];
  rate_limits?: {
    requests_per_minute: number;
    tokens_per_minute: number;
  };
  metadata?: Record<string, unknown>;
}

/**
 * Agent leaderboard entry
 */
export interface LeaderboardEntry {
  agent: string;
  elo_rating: number;
  wins: number;
  losses: number;
  draws: number;
  total_debates: number;
  win_rate: number;
  rank: number;
}

/**
 * Full system introspection
 */
export interface SystemIntrospection {
  version: string;
  agents: AgentInfo[];
  features: Record<string, boolean>;
  capabilities: string[];
  limits: Record<string, number>;
  uptime_seconds: number;
}

/**
 * Internal client interface
 */
interface IntrospectionClientInterface {
  get<T>(path: string): Promise<T>;
}

/**
 * Introspection API namespace.
 *
 * Provides methods for system introspection:
 * - Get full system state
 * - View agent leaderboards
 * - List available agents
 * - Get specific agent details
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Get full system introspection
 * const system = await client.introspection.getAll();
 *
 * // Get agent leaderboard
 * const leaderboard = await client.introspection.getLeaderboard();
 *
 * // List available agents
 * const agents = await client.introspection.listAgents();
 * ```
 */
export class IntrospectionAPI {
  constructor(private client: IntrospectionClientInterface) {}

  /**
   * Get full system introspection.
   */
  async getAll(): Promise<SystemIntrospection> {
    return this.client.get('/api/v1/introspection/all');
  }

  /**
   * Get agent leaderboard.
   */
  async getLeaderboard(options?: { limit?: number }): Promise<{ leaderboard: LeaderboardEntry[] }> {
    const params = options?.limit ? `?limit=${options.limit}` : '';
    return this.client.get(`/api/v1/introspection/leaderboard${params}`);
  }

  /**
   * List available agents.
   */
  async listAgents(): Promise<{ agents: AgentInfo[] }> {
    return this.client.get('/api/v1/introspection/agents');
  }

  /**
   * Get specific agent details.
   */
  async getAgent(name: string): Promise<AgentInfo> {
    return this.client.get(`/api/v1/introspection/agents/${name}`);
  }

  /**
   * Check agent availability.
   */
  async checkAvailability(): Promise<{ available: string[]; unavailable: string[] }> {
    return this.client.get('/api/v1/introspection/agents/availability');
  }
}
