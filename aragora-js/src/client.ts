/**
 * Aragora SDK Client
 *
 * Main client class for interacting with the Aragora API.
 */

import {
  AragoraClientOptions,
  RequestOptions,
  HealthCheck,
  DeepHealthCheck,
  // Debate types
  Debate,
  DebateCreateRequest,
  DebateCreateResponse,
  DebateListResponse,
  // Graph debate types
  GraphDebate,
  GraphDebateCreateRequest,
  GraphDebateCreateResponse,
  GraphDebateBranch,
  // Matrix debate types
  MatrixDebate,
  MatrixDebateCreateRequest,
  MatrixDebateCreateResponse,
  MatrixConclusion,
  // Verification types
  VerifyClaimRequest,
  VerifyClaimResponse,
  VerifyStatusResponse,
  // Memory types
  MemoryAnalyticsResponse,
  MemorySnapshotResponse,
  // Agent types
  AgentProfile,
  LeaderboardEntry,
  LeaderboardResponse,
  // Gauntlet types
  GauntletRunRequest,
  GauntletRunResponse,
  GauntletReceipt,
  // Replay types
  ReplaySummary,
  Replay,
  // Pulse types
  PulseTrendingResponse,
  PulseSuggestResponse,
  PulseAnalyticsResponse,
  PulseDebateRequest,
  PulseSchedulerStatus,
  PulseSchedulerConfig,
  PulseSchedulerHistoryEntry,
  // Document types
  Document,
  DocumentListResponse,
  DocumentUploadResponse,
  DocumentFormatsResponse,
  // Breakpoint types
  Breakpoint,
  BreakpointResolveRequest,
  BreakpointResolveResponse,
  BreakpointStatusResponse,
} from './types';

// =============================================================================
// HTTP Client
// =============================================================================

class HttpClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;
  private defaultHeaders: Record<string, string>;

  constructor(options: AragoraClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, '');
    this.apiKey = options.apiKey;
    this.timeout = options.timeout ?? 30000;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.apiKey) {
      this.defaultHeaders['Authorization'] = `Bearer ${this.apiKey}`;
    }
  }

  private async request<T>(
    method: string,
    path: string,
    data?: unknown,
    options?: RequestOptions
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      options?.timeout ?? this.timeout
    );

    try {
      const response = await fetch(url, {
        method,
        headers: {
          ...this.defaultHeaders,
          ...options?.headers,
        },
        body: data ? JSON.stringify(data) : undefined,
        signal: options?.signal ?? controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new AragoraError(
          errorData.error || `HTTP ${response.status}`,
          errorData.code || 'HTTP_ERROR',
          response.status,
          errorData
        );
      }

      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof AragoraError) {
        throw error;
      }

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new AragoraError('Request timed out', 'TIMEOUT', 408);
        }
        throw new AragoraError(error.message, 'NETWORK_ERROR', 0);
      }

      throw new AragoraError('Unknown error', 'UNKNOWN_ERROR', 0);
    }
  }

  async get<T>(path: string, options?: RequestOptions): Promise<T> {
    return this.request<T>('GET', path, undefined, options);
  }

  async post<T>(path: string, data?: unknown, options?: RequestOptions): Promise<T> {
    return this.request<T>('POST', path, data, options);
  }

  async put<T>(path: string, data?: unknown, options?: RequestOptions): Promise<T> {
    return this.request<T>('PUT', path, data, options);
  }

  async delete<T>(path: string, options?: RequestOptions): Promise<T> {
    return this.request<T>('DELETE', path, undefined, options);
  }
}

// =============================================================================
// Error Class
// =============================================================================

export class AragoraError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details?: Record<string, unknown>;

  constructor(
    message: string,
    code: string,
    status: number,
    details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'AragoraError';
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

// =============================================================================
// API Classes
// =============================================================================

class DebatesAPI {
  constructor(private http: HttpClient) {}

  async create(request: DebateCreateRequest): Promise<DebateCreateResponse> {
    return this.http.post<DebateCreateResponse>('/api/debates', request);
  }

  async get(debateId: string): Promise<Debate> {
    return this.http.get<Debate>(`/api/debates/${debateId}`);
  }

  async list(options?: {
    limit?: number;
    offset?: number;
    status?: string;
  }): Promise<Debate[]> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.status) params.set('status', options.status);

    const query = params.toString();
    const path = query ? `/api/debates?${query}` : '/api/debates';
    const response = await this.http.get<DebateListResponse>(path);
    return response.debates;
  }

  async run(request: DebateCreateRequest): Promise<Debate> {
    const created = await this.create(request);
    return this.waitForCompletion(created.debate_id);
  }

  async waitForCompletion(
    debateId: string,
    options?: { timeout?: number; pollInterval?: number }
  ): Promise<Debate> {
    const timeout = options?.timeout ?? 300000; // 5 minutes
    const pollInterval = options?.pollInterval ?? 2000; // 2 seconds
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const debate = await this.get(debateId);

      if (debate.status === 'completed' || debate.status === 'failed') {
        return debate;
      }

      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    throw new AragoraError(
      `Debate ${debateId} did not complete within timeout`,
      'TIMEOUT',
      408
    );
  }
}

class GraphDebatesAPI {
  constructor(private http: HttpClient) {}

  async create(request: GraphDebateCreateRequest): Promise<GraphDebateCreateResponse> {
    return this.http.post<GraphDebateCreateResponse>('/api/debates/graph', request);
  }

  async get(debateId: string): Promise<GraphDebate> {
    return this.http.get<GraphDebate>(`/api/debates/graph/${debateId}`);
  }

  async getBranches(debateId: string): Promise<GraphDebateBranch[]> {
    const response = await this.http.get<{ branches: GraphDebateBranch[] }>(
      `/api/debates/graph/${debateId}/branches`
    );
    return response.branches;
  }
}

class MatrixDebatesAPI {
  constructor(private http: HttpClient) {}

  async create(request: MatrixDebateCreateRequest): Promise<MatrixDebateCreateResponse> {
    return this.http.post<MatrixDebateCreateResponse>('/api/debates/matrix', request);
  }

  async get(matrixId: string): Promise<MatrixDebate> {
    return this.http.get<MatrixDebate>(`/api/debates/matrix/${matrixId}`);
  }

  async getConclusions(matrixId: string): Promise<MatrixConclusion> {
    return this.http.get<MatrixConclusion>(`/api/debates/matrix/${matrixId}/conclusions`);
  }
}

class VerificationAPI {
  constructor(private http: HttpClient) {}

  async verify(request: VerifyClaimRequest): Promise<VerifyClaimResponse> {
    return this.http.post<VerifyClaimResponse>('/api/verify/claim', request);
  }

  async status(): Promise<VerifyStatusResponse> {
    return this.http.get<VerifyStatusResponse>('/api/verify/status');
  }

  async verifyBatch(claims: string[]): Promise<VerifyClaimResponse[]> {
    const response = await this.http.post<{ results: VerifyClaimResponse[] }>(
      '/api/verify/batch',
      { claims }
    );
    return response.results;
  }
}

class MemoryAPI {
  constructor(private http: HttpClient) {}

  async analytics(days = 30): Promise<MemoryAnalyticsResponse> {
    return this.http.get<MemoryAnalyticsResponse>(`/api/memory/analytics?days=${days}`);
  }

  async tierStats(tierName: string, days = 30): Promise<Record<string, unknown>> {
    return this.http.get<Record<string, unknown>>(
      `/api/memory/analytics/tier/${tierName}?days=${days}`
    );
  }

  async snapshot(): Promise<MemorySnapshotResponse> {
    return this.http.post<MemorySnapshotResponse>('/api/memory/analytics/snapshot', {});
  }
}

class AgentsAPI {
  constructor(private http: HttpClient) {}

  async list(): Promise<AgentProfile[]> {
    const response = await this.http.get<{ agents: AgentProfile[] }>('/api/agents');
    return response.agents;
  }

  async get(agentId: string): Promise<AgentProfile> {
    return this.http.get<AgentProfile>(`/api/agents/${agentId}`);
  }

  async history(agentId: string, limit = 20): Promise<Record<string, unknown>[]> {
    const response = await this.http.get<{ matches: Record<string, unknown>[] }>(
      `/api/agent/${agentId}/history?limit=${limit}`
    );
    return response.matches;
  }

  async rivals(agentId: string, limit = 10): Promise<Record<string, unknown>[]> {
    const response = await this.http.get<{ rivals: Record<string, unknown>[] }>(
      `/api/agent/${agentId}/rivals?limit=${limit}`
    );
    return response.rivals;
  }

  async allies(agentId: string, limit = 10): Promise<Record<string, unknown>[]> {
    const response = await this.http.get<{ allies: Record<string, unknown>[] }>(
      `/api/agent/${agentId}/allies?limit=${limit}`
    );
    return response.allies;
  }
}

class LeaderboardAPI {
  constructor(private http: HttpClient) {}

  async get(options?: { limit?: number }): Promise<LeaderboardEntry[]> {
    const params = options?.limit ? `?limit=${options.limit}` : '';
    const response = await this.http.get<LeaderboardResponse>(`/api/leaderboard${params}`);
    return response.entries;
  }
}

class GauntletAPI {
  constructor(private http: HttpClient) {}

  async run(request: GauntletRunRequest): Promise<GauntletRunResponse> {
    return this.http.post<GauntletRunResponse>('/api/gauntlet', request);
  }

  async getReceipt(gauntletId: string): Promise<GauntletReceipt> {
    return this.http.get<GauntletReceipt>(`/api/gauntlet/${gauntletId}/receipt`);
  }

  async runAndWait(
    request: GauntletRunRequest,
    options?: { timeout?: number; pollInterval?: number }
  ): Promise<GauntletReceipt> {
    const { gauntlet_id } = await this.run(request);
    return this.waitForCompletion(gauntlet_id, options);
  }

  async waitForCompletion(
    gauntletId: string,
    options?: { timeout?: number; pollInterval?: number }
  ): Promise<GauntletReceipt> {
    const timeout = options?.timeout ?? 600000; // 10 minutes
    const pollInterval = options?.pollInterval ?? 5000; // 5 seconds
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const receipt = await this.getReceipt(gauntletId);

      if (receipt.status === 'completed' || receipt.status === 'failed') {
        return receipt;
      }

      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    throw new AragoraError(
      `Gauntlet ${gauntletId} did not complete within timeout`,
      'TIMEOUT',
      408
    );
  }
}

class ReplayAPI {
  constructor(private http: HttpClient) {}

  async list(options?: { limit?: number; debateId?: string }): Promise<ReplaySummary[]> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.debateId) params.set('debate_id', options.debateId);

    const query = params.toString();
    const path = query ? `/api/replays?${query}` : '/api/replays';
    const response = await this.http.get<{ replays: ReplaySummary[] }>(path);
    return response.replays;
  }

  async get(replayId: string): Promise<Replay> {
    return this.http.get<Replay>(`/api/replays/${replayId}`);
  }

  async delete(replayId: string): Promise<boolean> {
    await this.http.delete(`/api/replays/${replayId}`);
    return true;
  }

  async export(replayId: string, format: 'json' | 'csv' = 'json'): Promise<string> {
    const response = await this.http.get<{ data: string }>(
      `/api/replays/${replayId}/export?format=${format}`
    );
    return response.data;
  }
}

class PulseAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get trending topics from multiple sources.
   */
  async trending(options?: { sources?: string[]; limit?: number }): Promise<PulseTrendingResponse> {
    const params = new URLSearchParams();
    if (options?.sources) params.set('sources', options.sources.join(','));
    if (options?.limit) params.set('limit', String(options.limit));

    const query = params.toString();
    const path = query ? `/api/pulse/trending?${query}` : '/api/pulse/trending';
    return this.http.get<PulseTrendingResponse>(path);
  }

  /**
   * Get a suggested topic for debate.
   */
  async suggest(): Promise<PulseSuggestResponse> {
    return this.http.get<PulseSuggestResponse>('/api/pulse/suggest');
  }

  /**
   * Get analytics on trending topic debates.
   */
  async analytics(days = 30): Promise<PulseAnalyticsResponse> {
    return this.http.get<PulseAnalyticsResponse>(`/api/pulse/analytics?days=${days}`);
  }

  /**
   * Start a debate on a trending topic.
   */
  async debateTopic(request: PulseDebateRequest): Promise<DebateCreateResponse> {
    return this.http.post<DebateCreateResponse>('/api/pulse/debate-topic', request);
  }

  /**
   * Get scheduler status.
   */
  async schedulerStatus(): Promise<PulseSchedulerStatus> {
    return this.http.get<PulseSchedulerStatus>('/api/pulse/scheduler/status');
  }

  /**
   * Start the scheduler.
   */
  async schedulerStart(): Promise<{ status: string }> {
    return this.http.post<{ status: string }>('/api/pulse/scheduler/start', {});
  }

  /**
   * Stop the scheduler.
   */
  async schedulerStop(): Promise<{ status: string }> {
    return this.http.post<{ status: string }>('/api/pulse/scheduler/stop', {});
  }

  /**
   * Pause the scheduler.
   */
  async schedulerPause(): Promise<{ status: string }> {
    return this.http.post<{ status: string }>('/api/pulse/scheduler/pause', {});
  }

  /**
   * Resume the scheduler.
   */
  async schedulerResume(): Promise<{ status: string }> {
    return this.http.post<{ status: string }>('/api/pulse/scheduler/resume', {});
  }

  /**
   * Update scheduler configuration.
   */
  async schedulerConfig(config: PulseSchedulerConfig): Promise<PulseSchedulerConfig> {
    return this.http.put<PulseSchedulerConfig>('/api/pulse/scheduler/config', config);
  }

  /**
   * Get scheduled debate history.
   */
  async schedulerHistory(options?: {
    limit?: number;
    offset?: number;
    platform?: string;
  }): Promise<PulseSchedulerHistoryEntry[]> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.platform) params.set('platform', options.platform);

    const query = params.toString();
    const path = query ? `/api/pulse/scheduler/history?${query}` : '/api/pulse/scheduler/history';
    const response = await this.http.get<{ history: PulseSchedulerHistoryEntry[] }>(path);
    return response.history;
  }
}

class DocumentsAPI {
  constructor(private http: HttpClient) {}

  /**
   * List all uploaded documents.
   */
  async list(): Promise<Document[]> {
    const response = await this.http.get<DocumentListResponse>('/api/documents');
    return response.documents;
  }

  /**
   * Get supported file formats.
   */
  async formats(): Promise<DocumentFormatsResponse> {
    return this.http.get<DocumentFormatsResponse>('/api/documents/formats');
  }

  /**
   * Get a document by ID.
   */
  async get(docId: string): Promise<Document> {
    return this.http.get<Document>(`/api/documents/${docId}`);
  }

  /**
   * Delete a document by ID.
   */
  async delete(docId: string): Promise<boolean> {
    await this.http.delete(`/api/documents/${docId}`);
    return true;
  }

  /**
   * Upload a document (base64 encoded content).
   * Note: For actual file uploads, use FormData with fetch directly.
   */
  async upload(options: {
    filename: string;
    content: string;
    contentType?: string;
    metadata?: Record<string, unknown>;
  }): Promise<DocumentUploadResponse> {
    return this.http.post<DocumentUploadResponse>('/api/documents/upload', options);
  }
}

class BreakpointsAPI {
  constructor(private http: HttpClient) {}

  /**
   * List pending breakpoints awaiting resolution.
   */
  async pending(): Promise<Breakpoint[]> {
    const response = await this.http.get<{ breakpoints: Breakpoint[] }>('/api/breakpoints/pending');
    return response.breakpoints;
  }

  /**
   * Get status of a specific breakpoint.
   */
  async status(breakpointId: string): Promise<BreakpointStatusResponse> {
    return this.http.get<BreakpointStatusResponse>(`/api/breakpoints/${breakpointId}/status`);
  }

  /**
   * Resolve a pending breakpoint.
   */
  async resolve(
    breakpointId: string,
    request: BreakpointResolveRequest
  ): Promise<BreakpointResolveResponse> {
    return this.http.post<BreakpointResolveResponse>(
      `/api/breakpoints/${breakpointId}/resolve`,
      request
    );
  }
}

// =============================================================================
// Main Client
// =============================================================================

export class AragoraClient {
  private http: HttpClient;

  readonly debates: DebatesAPI;
  readonly graphDebates: GraphDebatesAPI;
  readonly matrixDebates: MatrixDebatesAPI;
  readonly verification: VerificationAPI;
  readonly memory: MemoryAPI;
  readonly agents: AgentsAPI;
  readonly leaderboard: LeaderboardAPI;
  readonly gauntlet: GauntletAPI;
  readonly replays: ReplayAPI;
  readonly pulse: PulseAPI;
  readonly documents: DocumentsAPI;
  readonly breakpoints: BreakpointsAPI;

  /**
   * Create a new Aragora client.
   *
   * @param options - Client configuration options
   *
   * @example
   * ```typescript
   * const client = new AragoraClient({
   *   baseUrl: 'http://localhost:8080',
   *   apiKey: 'your-api-key',
   * });
   *
   * // Run a debate
   * const debate = await client.debates.run({
   *   task: 'Should we use microservices?',
   *   agents: ['anthropic-api', 'openai-api'],
   * });
   *
   * // Verify a claim
   * const result = await client.verification.verify({
   *   claim: 'All primes > 2 are odd',
   * });
   * ```
   */
  constructor(options: AragoraClientOptions) {
    this.http = new HttpClient(options);

    this.debates = new DebatesAPI(this.http);
    this.graphDebates = new GraphDebatesAPI(this.http);
    this.matrixDebates = new MatrixDebatesAPI(this.http);
    this.verification = new VerificationAPI(this.http);
    this.memory = new MemoryAPI(this.http);
    this.agents = new AgentsAPI(this.http);
    this.leaderboard = new LeaderboardAPI(this.http);
    this.gauntlet = new GauntletAPI(this.http);
    this.replays = new ReplayAPI(this.http);
    this.pulse = new PulseAPI(this.http);
    this.documents = new DocumentsAPI(this.http);
    this.breakpoints = new BreakpointsAPI(this.http);
  }

  /**
   * Check server health.
   */
  async health(): Promise<HealthCheck> {
    return this.http.get<HealthCheck>('/api/health');
  }

  /**
   * Deep health check - verifies all external dependencies.
   * Use this for pre-deployment validation or debugging connectivity issues.
   */
  async healthDeep(): Promise<DeepHealthCheck> {
    return this.http.get<DeepHealthCheck>('/api/health/deep');
  }
}

export default AragoraClient;
