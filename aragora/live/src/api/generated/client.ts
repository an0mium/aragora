/**
 * Auto-generated Aragora API client
 * Based on aragora/server/openapi.yaml
 */

import type * as Types from './types';

export interface ApiClientConfig {
  baseUrl: string;
  getAuthToken?: () => string | null;
  onError?: (error: Error) => void;
}

export interface RequestOptions {
  signal?: AbortSignal;
  headers?: Record<string, string>;
}

export class AragoraApiClient {
  private config: ApiClientConfig;

  constructor(config: ApiClientConfig) {
    this.config = config;
  }

  private async request<T>(
    method: string,
    path: string,
    options: RequestOptions & { body?: unknown; query?: Record<string, string | number | boolean | undefined> } = {}
  ): Promise<T> {
    let url = `${this.config.baseUrl}${path}`;

    // Add query params
    if (options.query) {
      const params = new URLSearchParams();
      for (const [key, value] of Object.entries(options.query)) {
        if (value !== undefined) {
          params.append(key, String(value));
        }
      }
      const queryString = params.toString();
      if (queryString) {
        url += `?${queryString}`;
      }
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    // Add auth token if available
    const token = this.config.getAuthToken?.();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(url, {
      method,
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: options.signal,
    });

    if (!response.ok) {
      const errorBody = await response.text();
      const error = new Error(`API error: ${response.status} ${response.statusText} - ${errorBody}`);
      this.config.onError?.(error);
      throw error;
    }

    return response.json();
  }

  // =============================================================================
  // Health & System
  // =============================================================================

  /** Get health status */
  async getHealth(options?: RequestOptions): Promise<Types.HealthStatus> {
    return this.request<Types.HealthStatus>('GET', '/api/health', options);
  }

  /** Get detailed health status */
  async getHealthDetailed(options?: RequestOptions): Promise<Types.DetailedHealthStatus> {
    return this.request<Types.DetailedHealthStatus>('GET', '/api/health/detailed', options);
  }

  /** List available agents */
  async getAgents(options?: RequestOptions): Promise<Types.AvailableAgent[]> {
    return this.request<Types.AvailableAgent[]>('GET', '/api/agents', options);
  }

  // =============================================================================
  // Debates - Core
  // =============================================================================

  /** List debates with pagination */
  async getDebates(
    query?: { limit?: number; offset?: number },
    options?: RequestOptions
  ): Promise<Types.PaginatedResponse<Types.DebateSummary>> {
    return this.request<Types.PaginatedResponse<Types.DebateSummary>>('GET', '/api/debates', { ...options, query });
  }

  /** Get a single debate by ID */
  async getDebate(debateId: string, options?: RequestOptions): Promise<Types.Debate> {
    return this.request<Types.Debate>('GET', `/api/debates/${debateId}`, options);
  }

  /** Start a new debate */
  async startDebate(
    body: {
      task: string;
      agents?: string[];
      rounds?: number;
      consensus?: 'majority' | 'unanimous' | 'supermajority';
      context?: string;
    },
    options?: RequestOptions
  ): Promise<{ debate_id: string; status: string }> {
    return this.request<{ debate_id: string; status: string }>('POST', '/api/debates', { ...options, body });
  }

  /** Delete a debate */
  async deleteDebate(debateId: string, options?: RequestOptions): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>('DELETE', `/api/debates/${debateId}`, options);
  }

  // =============================================================================
  // Debates - Analysis
  // =============================================================================

  /** Get impasse analysis for a debate */
  async getImpasseAnalysis(debateId: string, options?: RequestOptions): Promise<Types.ImpasseAnalysis> {
    return this.request<Types.ImpasseAnalysis>('GET', `/api/debates/${debateId}/impasse`, options);
  }

  /** Get convergence status for a debate */
  async getConvergenceStatus(debateId: string, options?: RequestOptions): Promise<Types.ConvergenceStatus> {
    return this.request<Types.ConvergenceStatus>('GET', `/api/debates/${debateId}/convergence`, options);
  }

  /** Get citations for a debate */
  async getCitations(debateId: string, options?: RequestOptions): Promise<Types.Citations> {
    return this.request<Types.Citations>('GET', `/api/debates/${debateId}/citations`, options);
  }

  /** Get votes for a debate */
  async getVotes(debateId: string, options?: RequestOptions): Promise<{ votes: Types.Vote[] }> {
    return this.request<{ votes: Types.Vote[] }>('GET', `/api/debates/${debateId}/votes`, options);
  }

  /** Submit a user vote */
  async submitVote(
    debateId: string,
    body: { choice: string; confidence?: number; reasoning?: string },
    options?: RequestOptions
  ): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>('POST', `/api/debates/${debateId}/votes`, { ...options, body });
  }

  /** Verify debate outcome */
  async verifyOutcome(
    debateId: string,
    body: Types.VerifyOutcomeRequest,
    options?: RequestOptions
  ): Promise<{ success: boolean; calibration_updated: boolean }> {
    return this.request<{ success: boolean; calibration_updated: boolean }>(
      'POST',
      `/api/debates/${debateId}/verify`,
      { ...options, body }
    );
  }

  // =============================================================================
  // Debates - Fork & Follow-up
  // =============================================================================

  /** Fork a debate at a specific point */
  async forkDebate(
    debateId: string,
    body: Types.ForkRequest,
    options?: RequestOptions
  ): Promise<Types.ForkResult> {
    return this.request<Types.ForkResult>('POST', `/api/debates/${debateId}/fork`, { ...options, body });
  }

  /** Get suggested follow-up questions */
  async getFollowupSuggestions(
    debateId: string,
    options?: RequestOptions
  ): Promise<{ suggestions: Types.FollowupSuggestion[] }> {
    return this.request<{ suggestions: Types.FollowupSuggestion[] }>(
      'GET',
      `/api/debates/${debateId}/followup`,
      options
    );
  }

  /** Start a follow-up debate */
  async startFollowup(
    debateId: string,
    body: Types.FollowupRequest,
    options?: RequestOptions
  ): Promise<Types.FollowupResult> {
    return this.request<Types.FollowupResult>('POST', `/api/debates/${debateId}/followup`, { ...options, body });
  }

  // =============================================================================
  // Batch Debates
  // =============================================================================

  /** Submit a batch of debates */
  async submitBatch(
    body: Types.BatchRequest,
    options?: RequestOptions
  ): Promise<Types.BatchSubmitResponse> {
    return this.request<Types.BatchSubmitResponse>('POST', '/api/debates/batch', { ...options, body });
  }

  /** Get batch status */
  async getBatchStatus(
    batchId: string,
    query?: { detailed?: boolean },
    options?: RequestOptions
  ): Promise<Types.BatchStatus | Types.BatchDetailedStatus> {
    return this.request<Types.BatchStatus | Types.BatchDetailedStatus>(
      'GET',
      `/api/debates/batch/${batchId}`,
      { ...options, query }
    );
  }

  /** Cancel a batch */
  async cancelBatch(batchId: string, options?: RequestOptions): Promise<{ success: boolean; cancelled: number }> {
    return this.request<{ success: boolean; cancelled: number }>(
      'DELETE',
      `/api/debates/batch/${batchId}`,
      options
    );
  }

  /** Get batch queue status */
  async getBatchQueueStatus(options?: RequestOptions): Promise<Types.QueueStatus> {
    return this.request<Types.QueueStatus>('GET', '/api/debates/batch/queue', options);
  }

  // =============================================================================
  // Graph Debates
  // =============================================================================

  /** Run a graph debate */
  async runGraphDebate(
    body: Types.GraphDebateRequest,
    options?: RequestOptions
  ): Promise<Types.GraphDebateResponse> {
    return this.request<Types.GraphDebateResponse>('POST', '/api/debates/graph', { ...options, body });
  }

  /** Get a graph debate */
  async getGraphDebate(
    debateId: string,
    options?: RequestOptions
  ): Promise<Types.GraphDebateResponse> {
    return this.request<Types.GraphDebateResponse>('GET', `/api/debates/graph/${debateId}`, options);
  }

  /** Get nodes for a graph debate */
  async getGraphNodes(
    debateId: string,
    query?: { branch_id?: string },
    options?: RequestOptions
  ): Promise<{ nodes: Types.DebateNode[] }> {
    return this.request<{ nodes: Types.DebateNode[] }>(
      'GET',
      `/api/debates/graph/${debateId}/nodes`,
      { ...options, query }
    );
  }

  /** Get branches for a graph debate */
  async getGraphBranches(
    debateId: string,
    options?: RequestOptions
  ): Promise<{ branches: Types.DebateBranch[] }> {
    return this.request<{ branches: Types.DebateBranch[] }>(
      'GET',
      `/api/debates/graph/${debateId}/branches`,
      options
    );
  }

  /** Create a new branch */
  async createGraphBranch(
    debateId: string,
    body: { from_node: string; reason?: string },
    options?: RequestOptions
  ): Promise<Types.DebateBranch> {
    return this.request<Types.DebateBranch>(
      'POST',
      `/api/debates/graph/${debateId}/branches`,
      { ...options, body }
    );
  }

  /** Merge branches */
  async mergeGraphBranches(
    debateId: string,
    body: { branch_ids: string[]; strategy?: 'synthesis' | 'vote' | 'best' },
    options?: RequestOptions
  ): Promise<{ merged_branch_id: string; conclusion: string }> {
    return this.request<{ merged_branch_id: string; conclusion: string }>(
      'POST',
      `/api/debates/graph/${debateId}/merge`,
      { ...options, body }
    );
  }

  // =============================================================================
  // Matrix Debates
  // =============================================================================

  /** Run a matrix debate */
  async runMatrixDebate(
    body: Types.MatrixDebateRequest,
    options?: RequestOptions
  ): Promise<Types.MatrixDebateResponse> {
    return this.request<Types.MatrixDebateResponse>('POST', '/api/debates/matrix', { ...options, body });
  }

  /** Get a matrix debate */
  async getMatrixDebate(
    matrixId: string,
    options?: RequestOptions
  ): Promise<Types.MatrixDebateResponse> {
    return this.request<Types.MatrixDebateResponse>('GET', `/api/debates/matrix/${matrixId}`, options);
  }

  /** Get scenario result */
  async getScenarioResult(
    matrixId: string,
    scenarioName: string,
    options?: RequestOptions
  ): Promise<Types.ScenarioResult> {
    return this.request<Types.ScenarioResult>(
      'GET',
      `/api/debates/matrix/${matrixId}/scenarios/${scenarioName}`,
      options
    );
  }

  /** Compare scenarios */
  async compareScenarios(
    matrixId: string,
    body: { scenarios: string[] },
    options?: RequestOptions
  ): Promise<{ comparison: Types.ConditionalConclusion[] }> {
    return this.request<{ comparison: Types.ConditionalConclusion[] }>(
      'POST',
      `/api/debates/matrix/${matrixId}/compare`,
      { ...options, body }
    );
  }

  // =============================================================================
  // Agents - Leaderboard & Profiles
  // =============================================================================

  /** Get agent leaderboard */
  async getLeaderboard(
    query?: { limit?: number; domain?: string },
    options?: RequestOptions
  ): Promise<{ leaderboard: Types.LeaderboardEntry[] }> {
    return this.request<{ leaderboard: Types.LeaderboardEntry[] }>(
      'GET',
      '/api/ranking/leaderboard',
      { ...options, query }
    );
  }

  /** Get ranking statistics */
  async getRankingStats(options?: RequestOptions): Promise<Types.RankingStats> {
    return this.request<Types.RankingStats>('GET', '/api/ranking/stats', options);
  }

  /** Get agent profile */
  async getAgentProfile(agentName: string, options?: RequestOptions): Promise<Types.AgentProfile> {
    return this.request<Types.AgentProfile>('GET', `/api/agent/${agentName}/profile`, options);
  }

  /** Get agent match history */
  async getAgentHistory(
    agentName: string,
    query?: { limit?: number },
    options?: RequestOptions
  ): Promise<{ history: Types.MatchHistoryEntry[] }> {
    return this.request<{ history: Types.MatchHistoryEntry[] }>(
      'GET',
      `/api/agent/${agentName}/history`,
      { ...options, query }
    );
  }

  /** Get agent consistency score */
  async getAgentConsistency(agentName: string, options?: RequestOptions): Promise<Types.ConsistencyScore> {
    return this.request<Types.ConsistencyScore>('GET', `/api/agent/${agentName}/consistency`, options);
  }

  /** Get agent relationship network */
  async getAgentNetwork(agentName: string, options?: RequestOptions): Promise<Types.RelationshipNetwork> {
    return this.request<Types.RelationshipNetwork>('GET', `/api/agent/${agentName}/network`, options);
  }

  // =============================================================================
  // Memory
  // =============================================================================

  /** Get memory entry */
  async getMemory(key: string, options?: RequestOptions): Promise<Types.MemoryEntry> {
    return this.request<Types.MemoryEntry>('GET', `/api/memory/${key}`, options);
  }

  /** Set memory entry */
  async setMemory(
    body: { key: string; content: string; tier?: 'fast' | 'medium' | 'slow' | 'glacial'; ttl?: number },
    options?: RequestOptions
  ): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>('POST', '/api/memory', { ...options, body });
  }

  /** Delete memory entry */
  async deleteMemory(key: string, options?: RequestOptions): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>('DELETE', `/api/memory/${key}`, options);
  }

  /** Query memory entries */
  async queryMemory(
    query?: { tier?: string; prefix?: string; limit?: number },
    options?: RequestOptions
  ): Promise<Types.MemoryQueryResult> {
    return this.request<Types.MemoryQueryResult>('GET', '/api/memory', { ...options, query });
  }

  /** Get memory statistics */
  async getMemoryStats(options?: RequestOptions): Promise<Types.MemoryStats> {
    return this.request<Types.MemoryStats>('GET', '/api/memory/stats', options);
  }

  // =============================================================================
  // Analytics
  // =============================================================================

  /** Get dashboard metrics */
  async getDashboardMetrics(options?: RequestOptions): Promise<Types.DashboardMetrics> {
    return this.request<Types.DashboardMetrics>('GET', '/api/analytics/dashboard', options);
  }

  /** Get disagreement statistics */
  async getDisagreementStats(options?: RequestOptions): Promise<Types.DisagreementStats> {
    return this.request<Types.DisagreementStats>('GET', '/api/analytics/disagreements', options);
  }
}

// =============================================================================
// Factory & Hooks
// =============================================================================

/** Create a new API client instance */
export function createApiClient(config: Partial<ApiClientConfig> = {}): AragoraApiClient {
  return new AragoraApiClient({
    baseUrl: config.baseUrl || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080',
    ...config,
  });
}

/** Default client instance */
let defaultClient: AragoraApiClient | null = null;

/** Get or create the default API client */
export function getApiClient(): AragoraApiClient {
  if (!defaultClient) {
    defaultClient = createApiClient();
  }
  return defaultClient;
}

/** Configure the default client */
export function configureApiClient(config: Partial<ApiClientConfig>): void {
  defaultClient = createApiClient(config);
}
