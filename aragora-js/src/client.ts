/**
 * Aragora SDK Client
 *
 * Main client class for interacting with the Aragora API.
 */

import {
  AragoraClientOptions,
  RequestOptions,
  RetryOptions,
  HealthCheck,
  DeepHealthCheck,
  // Debate types
  Debate,
  DebateStatus,
  normalizeStatus,
  normalizeConsensusResult,
  DebateCreateRequest,
  DebateCreateResponse,
  DebateListResponse,
  // Extended debate types
  ImpasseInfo,
  ConvergenceInfo,
  DebateCitationsResponse,
  DebateMessagesResponse,
  DebateEvidenceResponse,
  DebateSummary,
  FollowupSuggestionsResponse,
  ForkRequest,
  ForkResponse,
  FollowupRequest,
  FollowupResponse,
  DebateExportResponse,
  // Batch debate types
  BatchDebateRequest,
  BatchDebateResponse,
  BatchStatusResponse,
  QueueStatusResponse,
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
  ContinuumRetrieveResponse,
  ContinuumConsolidateResponse,
  ContinuumCleanupResponse,
  TierStatsResponse,
  ArchiveStatsResponse,
  MemoryPressureResponse,
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
  // Tournament types
  TournamentSummary,
  TournamentStanding,
  TournamentListResponse,
  TournamentStandingsResponse,
  // Organization types
  Organization,
  OrganizationMember,
  OrganizationInvitation,
  InviteRequest,
  InviteResponse,
  OrganizationUpdateRequest,
  // Analytics types
  AnalyticsResponse,
  // Auth types
  RegisterRequest,
  RegisterResponse,
  LoginRequest,
  LoginResponse,
  AuthUser,
  RefreshRequest,
  RefreshResponse,
  RevokeTokenRequest,
  UpdateMeRequest,
  ChangePasswordRequest,
  ChangePasswordResponse,
  ApiKeyCreateRequest,
  ApiKeyResponse,
  MfaSetupResponse,
  MfaEnableRequest,
  MfaEnableResponse,
  MfaDisableRequest,
  MfaVerifyRequest,
  MfaVerifyResponse,
  MfaBackupCodesResponse,
  // Billing types
  BillingPlan,
  BillingPlansResponse,
  BillingUsage,
  BillingSubscription,
  CheckoutRequest,
  CheckoutResponse,
  PortalResponse,
  CancelSubscriptionResponse,
  ResumeSubscriptionResponse,
  BillingAuditLogResponse,
  UsageForecast,
  Invoice,
  InvoicesResponse,
} from './types';

// =============================================================================
// HTTP Client
// =============================================================================

/** Default retry configuration */
const DEFAULT_RETRY: Required<RetryOptions> = {
  maxRetries: 3,
  initialDelay: 1000,
  maxDelay: 30000,
  backoffMultiplier: 2,
  jitter: true,
};

/** HTTP status codes that are retryable */
const RETRYABLE_STATUS_CODES = new Set([
  408, // Request Timeout
  429, // Too Many Requests
  500, // Internal Server Error
  502, // Bad Gateway
  503, // Service Unavailable
  504, // Gateway Timeout
]);

/** Error codes that are retryable */
const RETRYABLE_ERROR_CODES = new Set([
  'TIMEOUT',
  'NETWORK_ERROR',
  'RATE_LIMITED',
  'SERVICE_UNAVAILABLE',
]);

/** Calculate delay for exponential backoff with optional jitter */
function calculateDelay(
  attempt: number,
  options: Required<RetryOptions>
): number {
  const exponentialDelay =
    options.initialDelay * Math.pow(options.backoffMultiplier, attempt);
  const cappedDelay = Math.min(exponentialDelay, options.maxDelay);

  if (options.jitter) {
    // Add random jitter between 0-25% of the delay
    const jitterFactor = 1 + Math.random() * 0.25;
    return Math.floor(cappedDelay * jitterFactor);
  }

  return cappedDelay;
}

/** Sleep for a specified duration */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class HttpClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;
  private defaultHeaders: Record<string, string>;
  private retryOptions: Required<RetryOptions>;

  constructor(options: AragoraClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, '');
    this.apiKey = options.apiKey;
    this.timeout = options.timeout ?? 30000;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      ...options.headers,
    };
    this.retryOptions = {
      ...DEFAULT_RETRY,
      ...options.retry,
    };

    if (this.apiKey) {
      this.defaultHeaders['Authorization'] = `Bearer ${this.apiKey}`;
    }
  }

  private async requestOnce<T>(
    method: string,
    url: string,
    data?: unknown,
    options?: RequestOptions
  ): Promise<T> {
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
        const isRetryable = RETRYABLE_STATUS_CODES.has(response.status);
        throw new AragoraError(
          errorData.error || `HTTP ${response.status}`,
          errorData.code || 'HTTP_ERROR',
          response.status,
          errorData,
          isRetryable
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
          throw new AragoraError('Request timed out', 'TIMEOUT', 408, undefined, true);
        }
        throw new AragoraError(error.message, 'NETWORK_ERROR', 0, undefined, true);
      }

      throw new AragoraError('Unknown error', 'UNKNOWN_ERROR', 0);
    }
  }

  private async request<T>(
    method: string,
    path: string,
    data?: unknown,
    options?: RequestOptions
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;

    // Determine retry settings
    const retryDisabled = options?.retry === false;
    const retryOptions: Required<RetryOptions> = retryDisabled
      ? { ...this.retryOptions, maxRetries: 0 }
      : {
          ...this.retryOptions,
          ...(typeof options?.retry === 'object' ? options.retry : {}),
        };

    let lastError: AragoraError | undefined;

    for (let attempt = 0; attempt <= retryOptions.maxRetries; attempt++) {
      try {
        return await this.requestOnce<T>(method, url, data, options);
      } catch (error) {
        if (!(error instanceof AragoraError)) {
          throw error;
        }

        lastError = error;

        // Check if we should retry
        const isLastAttempt = attempt >= retryOptions.maxRetries;
        const isRetryable = error.retryable || RETRYABLE_ERROR_CODES.has(error.code);

        if (isLastAttempt || !isRetryable) {
          throw error;
        }

        // Calculate delay and wait before retrying
        const delay = calculateDelay(attempt, retryOptions);
        await sleep(delay);
      }
    }

    // Should never reach here, but throw last error if we do
    throw lastError ?? new AragoraError('Retry failed', 'RETRY_EXHAUSTED', 0);
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
  /** Whether this error is retryable (e.g., transient network issues, rate limits) */
  readonly retryable: boolean;

  constructor(
    message: string,
    code: string,
    status: number,
    details?: Record<string, unknown>,
    retryable: boolean = false
  ) {
    super(message);
    this.name = 'AragoraError';
    this.code = code;
    this.status = status;
    this.details = details;
    this.retryable = retryable;
  }

  /** Create a user-friendly error message with suggestions */
  toUserMessage(): string {
    switch (this.code) {
      case 'TIMEOUT':
        return 'Request timed out. Please try again or check your network connection.';
      case 'NETWORK_ERROR':
        return 'Network error. Please check your internet connection and try again.';
      case 'RATE_LIMITED':
        return 'Too many requests. Please wait a moment before trying again.';
      case 'UNAUTHORIZED':
        return 'Authentication failed. Please check your API key.';
      case 'FORBIDDEN':
        return 'Access denied. You do not have permission to perform this action.';
      case 'NOT_FOUND':
        return 'The requested resource was not found.';
      case 'VALIDATION_ERROR':
        return `Invalid request: ${this.message}`;
      default:
        return this.message;
    }
  }
}

// =============================================================================
// API Classes
// =============================================================================

/**
 * API client for managing debates.
 *
 * @example
 * ```typescript
 * // Create and run a debate
 * const debate = await client.debates.run({
 *   task: 'Should we adopt a microservices architecture?',
 *   agents: ['anthropic-api', 'openai-api'],
 *   rounds: 3,
 * });
 *
 * // Check consensus
 * if (debate.consensus) {
 *   console.log('Consensus reached:', debate.consensus.position);
 * }
 * ```
 */

/**
 * Normalize a debate response from the server for SDK compatibility.
 * Ensures status values and consensus fields use canonical values.
 */
function normalizeDebateResponse(debate: Debate): Debate {
  return {
    ...debate,
    status: normalizeStatus(debate.status as string) as DebateStatus,
    consensus: normalizeConsensusResult(debate.consensus),
  };
}

class DebatesAPI {
  constructor(private http: HttpClient) {}

  /**
   * Create a new debate asynchronously.
   *
   * This starts a debate in the background. Use {@link waitForCompletion}
   * or {@link run} if you want to wait for the result.
   *
   * @param request - Debate configuration including task, agents, and options
   * @returns Promise resolving to debate creation response with debate_id
   * @throws {AragoraError} When the request fails or validation errors occur
   *
   * @example
   * ```typescript
   * const response = await client.debates.create({
   *   task: 'What are the pros and cons of GraphQL vs REST?',
   *   agents: ['anthropic-api', 'openai-api', 'gemini-api'],
   *   rounds: 5,
   *   consensus: 'majority',
   * });
   * console.log('Debate started:', response.debate_id);
   * ```
   */
  async create(request: DebateCreateRequest): Promise<DebateCreateResponse> {
    return this.http.post<DebateCreateResponse>('/api/debates', request);
  }

  /**
   * Get a debate by ID.
   *
   * Retrieves the full debate object including all messages, consensus
   * information, and metadata.
   *
   * @param debateId - The unique debate identifier
   * @returns Promise resolving to the debate object
   * @throws {AragoraError} When debate not found (404) or other errors
   *
   * @example
   * ```typescript
   * const debate = await client.debates.get('debate-123');
   * console.log('Status:', debate.status);
   * console.log('Rounds completed:', debate.rounds_completed);
   * ```
   */
  async get(debateId: string): Promise<Debate> {
    const response = await this.http.get<Debate>(`/api/debates/${debateId}`);
    return normalizeDebateResponse(response);
  }

  /**
   * List debates with optional filtering and pagination.
   *
   * @param options - Optional filtering and pagination parameters
   * @param options.limit - Maximum number of debates to return (default: 20)
   * @param options.offset - Number of debates to skip for pagination
   * @param options.status - Filter by status ('pending', 'running', 'completed', 'failed')
   * @returns Promise resolving to array of debates
   *
   * @example
   * ```typescript
   * // Get recent completed debates
   * const completed = await client.debates.list({
   *   limit: 10,
   *   status: 'completed',
   * });
   *
   * // Paginate through all debates
   * const page2 = await client.debates.list({
   *   limit: 20,
   *   offset: 20,
   * });
   * ```
   */
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
    // Normalize all debate responses
    return response.debates.map(normalizeDebateResponse);
  }

  /**
   * Create a debate and wait for it to complete.
   *
   * This is the most common way to run a debate. It creates the debate
   * and polls until completion or timeout.
   *
   * @param request - Debate configuration including task, agents, and options
   * @returns Promise resolving to the completed debate
   * @throws {AragoraError} When debate fails or times out
   *
   * @example
   * ```typescript
   * const debate = await client.debates.run({
   *   task: 'Is functional programming better than OOP?',
   *   agents: ['anthropic-api', 'openai-api'],
   *   rounds: 3,
   *   environment: {
   *     context: 'For a large-scale enterprise application',
   *   },
   * });
   *
   * console.log('Final result:', debate.consensus?.position);
   * for (const message of debate.messages) {
   *   console.log(`${message.agent}: ${message.content.slice(0, 100)}...`);
   * }
   * ```
   */
  async run(request: DebateCreateRequest): Promise<Debate> {
    const created = await this.create(request);
    return this.waitForCompletion(created.debate_id);
  }

  /**
   * Wait for a debate to complete.
   *
   * Polls the debate status at regular intervals until the debate
   * reaches 'completed' or 'failed' status, or the timeout is reached.
   *
   * @param debateId - The debate ID to wait for
   * @param options - Polling configuration
   * @param options.timeout - Maximum time to wait in milliseconds (default: 300000 / 5 min)
   * @param options.pollInterval - Time between status checks in ms (default: 2000)
   * @returns Promise resolving to the completed debate
   * @throws {AragoraError} With code 'TIMEOUT' if debate doesn't complete in time
   *
   * @example
   * ```typescript
   * // Wait with custom timeout
   * const debate = await client.debates.waitForCompletion('debate-123', {
   *   timeout: 600000, // 10 minutes
   *   pollInterval: 5000, // Check every 5 seconds
   * });
   * ```
   */
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

  /**
   * Get impasse information for a debate.
   */
  async impasse(debateId: string): Promise<ImpasseInfo> {
    return this.http.get<ImpasseInfo>(`/api/debates/${debateId}/impasse`);
  }

  /**
   * Get convergence information for a debate.
   */
  async convergence(debateId: string): Promise<ConvergenceInfo> {
    return this.http.get<ConvergenceInfo>(`/api/debates/${debateId}/convergence`);
  }

  /**
   * Get citations used in a debate.
   */
  async citations(debateId: string): Promise<DebateCitationsResponse> {
    return this.http.get<DebateCitationsResponse>(`/api/debates/${debateId}/citations`);
  }

  /**
   * Get paginated messages from a debate.
   */
  async messages(
    debateId: string,
    options?: { limit?: number; offset?: number }
  ): Promise<DebateMessagesResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));

    const query = params.toString();
    const path = query
      ? `/api/debates/${debateId}/messages?${query}`
      : `/api/debates/${debateId}/messages`;
    return this.http.get<DebateMessagesResponse>(path);
  }

  /**
   * Get evidence collected during a debate.
   */
  async evidence(debateId: string): Promise<DebateEvidenceResponse> {
    return this.http.get<DebateEvidenceResponse>(`/api/debates/${debateId}/evidence`);
  }

  /**
   * Get a human-readable summary of a debate.
   */
  async summary(debateId: string): Promise<DebateSummary> {
    return this.http.get<DebateSummary>(`/api/debates/${debateId}/summary`);
  }

  /**
   * Get follow-up debate suggestions based on identified cruxes.
   */
  async followupSuggestions(debateId: string): Promise<FollowupSuggestionsResponse> {
    return this.http.get<FollowupSuggestionsResponse>(`/api/debates/${debateId}/followups`);
  }

  /**
   * Create a counterfactual fork of a debate at a specific branch point.
   */
  async fork(debateId: string, request: ForkRequest): Promise<ForkResponse> {
    return this.http.post<ForkResponse>(`/api/debates/${debateId}/fork`, request);
  }

  /**
   * Create a follow-up debate based on an identified crux.
   */
  async followup(debateId: string, request: FollowupRequest): Promise<FollowupResponse> {
    return this.http.post<FollowupResponse>(`/api/debates/${debateId}/followup`, request);
  }

  /**
   * Export a debate in the specified format.
   */
  async export(debateId: string, format: 'json' | 'markdown' | 'pdf' = 'json'): Promise<DebateExportResponse> {
    return this.http.get<DebateExportResponse>(`/api/debates/${debateId}/export/${format}`);
  }
}

class BatchDebatesAPI {
  constructor(private http: HttpClient) {}

  /**
   * Submit a batch of debates for processing.
   */
  async submit(request: BatchDebateRequest): Promise<BatchDebateResponse> {
    return this.http.post<BatchDebateResponse>('/api/debates/batch', request);
  }

  /**
   * Get the status of a batch submission.
   */
  async status(batchId: string): Promise<BatchStatusResponse> {
    return this.http.get<BatchStatusResponse>(`/api/debates/batch/${batchId}/status`);
  }

  /**
   * Get the current queue status.
   */
  async queueStatus(): Promise<QueueStatusResponse> {
    return this.http.get<QueueStatusResponse>('/api/debates/queue/status');
  }

  /**
   * Submit a batch and wait for all debates to complete.
   */
  async submitAndWait(
    request: BatchDebateRequest,
    options?: { timeout?: number; pollInterval?: number }
  ): Promise<BatchStatusResponse> {
    const { batch_id } = await this.submit(request);
    return this.waitForCompletion(batch_id, options);
  }

  /**
   * Wait for a batch to complete.
   */
  async waitForCompletion(
    batchId: string,
    options?: { timeout?: number; pollInterval?: number }
  ): Promise<BatchStatusResponse> {
    const timeout = options?.timeout ?? 600000; // 10 minutes
    const pollInterval = options?.pollInterval ?? 5000; // 5 seconds
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const batch = await this.status(batchId);

      if (batch.status === 'completed' || batch.status === 'partial_failure') {
        return batch;
      }

      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    throw new AragoraError(
      `Batch ${batchId} did not complete within timeout`,
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

  /**
   * Create a graph debate and wait for completion.
   */
  async run(request: GraphDebateCreateRequest): Promise<GraphDebate> {
    const created = await this.create(request);
    return this.waitForCompletion(created.debate_id);
  }

  /**
   * Wait for a graph debate to complete.
   */
  async waitForCompletion(
    debateId: string,
    options?: { timeout?: number; pollInterval?: number }
  ): Promise<GraphDebate> {
    const timeout = options?.timeout ?? 600000; // 10 minutes (graph debates take longer)
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
      `Graph debate ${debateId} did not complete within timeout`,
      'TIMEOUT',
      408
    );
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

  /**
   * Create a matrix debate and wait for completion.
   */
  async run(request: MatrixDebateCreateRequest): Promise<MatrixDebate> {
    const created = await this.create(request);
    return this.waitForCompletion(created.matrix_id);
  }

  /**
   * Wait for a matrix debate to complete.
   */
  async waitForCompletion(
    matrixId: string,
    options?: { timeout?: number; pollInterval?: number }
  ): Promise<MatrixDebate> {
    const timeout = options?.timeout ?? 900000; // 15 minutes (matrix debates run multiple sub-debates)
    const pollInterval = options?.pollInterval ?? 3000; // 3 seconds
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const debate = await this.get(matrixId);

      if (debate.status === 'completed' || debate.status === 'failed') {
        return debate;
      }

      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    throw new AragoraError(
      `Matrix debate ${matrixId} did not complete within timeout`,
      'TIMEOUT',
      408
    );
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

  // =========================================================================
  // Continuum Memory Operations
  // =========================================================================

  /**
   * Retrieve memories from the continuum.
   */
  async retrieve(options?: {
    tier?: string;
    limit?: number;
    offset?: number;
    query?: string;
  }): Promise<ContinuumRetrieveResponse> {
    const params = new URLSearchParams();
    if (options?.tier) params.set('tier', options.tier);
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.query) params.set('query', options.query);

    const query = params.toString();
    const path = query
      ? `/api/memory/continuum/retrieve?${query}`
      : '/api/memory/continuum/retrieve';
    return this.http.get<ContinuumRetrieveResponse>(path);
  }

  /**
   * Trigger memory consolidation across tiers.
   */
  async consolidate(): Promise<ContinuumConsolidateResponse> {
    return this.http.post<ContinuumConsolidateResponse>(
      '/api/memory/continuum/consolidate',
      {}
    );
  }

  /**
   * Cleanup expired memories.
   */
  async cleanup(options?: { tier?: string; max_age_days?: number }): Promise<ContinuumCleanupResponse> {
    return this.http.post<ContinuumCleanupResponse>(
      '/api/memory/continuum/cleanup',
      options || {}
    );
  }

  /**
   * Get tier statistics for all memory tiers.
   */
  async tiers(): Promise<TierStatsResponse> {
    return this.http.get<TierStatsResponse>('/api/memory/tier-stats');
  }

  /**
   * Get archive statistics.
   */
  async archiveStats(): Promise<ArchiveStatsResponse> {
    return this.http.get<ArchiveStatsResponse>('/api/memory/archive-stats');
  }

  /**
   * Get current memory pressure and utilization.
   */
  async pressure(): Promise<MemoryPressureResponse> {
    return this.http.get<MemoryPressureResponse>('/api/memory/pressure');
  }

  /**
   * Delete a specific memory by ID.
   */
  async delete(memoryId: string): Promise<boolean> {
    await this.http.delete(`/api/memory/continuum/${memoryId}`);
    return true;
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

class TournamentsAPI {
  constructor(private http: HttpClient) {}

  /**
   * List all available tournaments.
   */
  async list(): Promise<TournamentSummary[]> {
    const response = await this.http.get<TournamentListResponse>('/api/tournaments');
    return response.tournaments;
  }

  /**
   * Get standings for a specific tournament.
   */
  async standings(tournamentId: string): Promise<TournamentStanding[]> {
    const response = await this.http.get<TournamentStandingsResponse>(
      `/api/tournaments/${tournamentId}/standings`
    );
    return response.standings;
  }
}

class OrganizationsAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get organization details.
   */
  async get(orgId: string): Promise<Organization> {
    const response = await this.http.get<{ organization: Organization }>(`/api/org/${orgId}`);
    return response.organization;
  }

  /**
   * Update organization settings.
   */
  async update(orgId: string, data: OrganizationUpdateRequest): Promise<Organization> {
    const response = await this.http.put<{ organization: Organization }>(`/api/org/${orgId}`, data);
    return response.organization;
  }

  /**
   * List organization members.
   */
  async members(orgId: string): Promise<OrganizationMember[]> {
    const response = await this.http.get<{ members: OrganizationMember[] }>(
      `/api/org/${orgId}/members`
    );
    return response.members;
  }

  /**
   * Invite a user to the organization.
   */
  async invite(orgId: string, request: InviteRequest): Promise<InviteResponse> {
    return this.http.post<InviteResponse>(`/api/org/${orgId}/invite`, request);
  }

  /**
   * List pending invitations for the organization.
   */
  async invitations(orgId: string): Promise<OrganizationInvitation[]> {
    const response = await this.http.get<{ invitations: OrganizationInvitation[] }>(
      `/api/org/${orgId}/invitations`
    );
    return response.invitations;
  }

  /**
   * Revoke a pending invitation.
   */
  async revokeInvitation(orgId: string, invitationId: string): Promise<boolean> {
    await this.http.delete(`/api/org/${orgId}/invitations/${invitationId}`);
    return true;
  }

  /**
   * Remove a member from the organization.
   */
  async removeMember(orgId: string, userId: string): Promise<boolean> {
    await this.http.delete(`/api/org/${orgId}/members/${userId}`);
    return true;
  }

  /**
   * Update a member's role.
   */
  async updateMemberRole(
    orgId: string,
    userId: string,
    role: 'member' | 'admin'
  ): Promise<{ message: string; user_id: string; role: string }> {
    return this.http.put(`/api/org/${orgId}/members/${userId}/role`, { role });
  }

  /**
   * Get pending invitations for the current user.
   */
  async myPendingInvitations(): Promise<OrganizationInvitation[]> {
    const response = await this.http.get<{ invitations: OrganizationInvitation[] }>(
      '/api/invitations/pending'
    );
    return response.invitations;
  }

  /**
   * Accept an organization invitation.
   */
  async acceptInvitation(
    token: string
  ): Promise<{ message: string; organization: { id: string; name: string; slug: string }; role: string }> {
    return this.http.post(`/api/invitations/${token}/accept`, {});
  }
}

class AnalyticsAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get analytics overview.
   */
  async overview(days = 30): Promise<AnalyticsResponse> {
    return this.http.get<AnalyticsResponse>(`/api/analytics?days=${days}`);
  }

  /**
   * Get agent-specific analytics.
   */
  async agent(agentId: string, days = 30): Promise<Record<string, unknown>> {
    return this.http.get<Record<string, unknown>>(`/api/analytics/agent/${agentId}?days=${days}`);
  }

  /**
   * Get debate analytics.
   */
  async debates(options?: {
    days?: number;
    status?: string;
  }): Promise<Record<string, unknown>> {
    const params = new URLSearchParams();
    if (options?.days) params.set('days', String(options.days));
    if (options?.status) params.set('status', options.status);

    const query = params.toString();
    const path = query ? `/api/analytics/debates?${query}` : '/api/analytics/debates';
    return this.http.get<Record<string, unknown>>(path);
  }
}

class AuthAPI {
  constructor(private http: HttpClient) {}

  /**
   * Register a new user account.
   */
  async register(request: RegisterRequest): Promise<RegisterResponse> {
    return this.http.post<RegisterResponse>('/api/auth/register', request);
  }

  /**
   * Authenticate and get tokens.
   * If MFA is enabled, include mfa_code in the request.
   */
  async login(request: LoginRequest): Promise<LoginResponse> {
    return this.http.post<LoginResponse>('/api/auth/login', request);
  }

  /**
   * Invalidate current token (logout from current device).
   */
  async logout(): Promise<{ message: string }> {
    return this.http.post<{ message: string }>('/api/auth/logout', {});
  }

  /**
   * Invalidate all tokens for user (logout from all devices).
   */
  async logoutAll(): Promise<{ message: string }> {
    return this.http.post<{ message: string }>('/api/auth/logout-all', {});
  }

  /**
   * Refresh access token using refresh token.
   */
  async refresh(request: RefreshRequest): Promise<RefreshResponse> {
    return this.http.post<RefreshResponse>('/api/auth/refresh', request);
  }

  /**
   * Explicitly revoke a specific token.
   */
  async revoke(request: RevokeTokenRequest): Promise<{ message: string }> {
    return this.http.post<{ message: string }>('/api/auth/revoke', request);
  }

  /**
   * Get current user information.
   */
  async me(): Promise<AuthUser> {
    return this.http.get<AuthUser>('/api/auth/me');
  }

  /**
   * Update current user information.
   */
  async updateMe(request: UpdateMeRequest): Promise<AuthUser> {
    return this.http.put<AuthUser>('/api/auth/me', request);
  }

  /**
   * Change password.
   */
  async changePassword(request: ChangePasswordRequest): Promise<ChangePasswordResponse> {
    return this.http.post<ChangePasswordResponse>('/api/auth/password', request);
  }

  /**
   * Generate an API key.
   */
  async createApiKey(request?: ApiKeyCreateRequest): Promise<ApiKeyResponse> {
    return this.http.post<ApiKeyResponse>('/api/auth/api-key', request || {});
  }

  /**
   * Revoke an API key.
   */
  async revokeApiKey(): Promise<{ message: string }> {
    return this.http.delete<{ message: string }>('/api/auth/api-key');
  }

  // =========================================================================
  // MFA (Multi-Factor Authentication)
  // =========================================================================

  /**
   * Set up MFA - returns secret and QR code for authenticator app.
   */
  async mfaSetup(): Promise<MfaSetupResponse> {
    return this.http.post<MfaSetupResponse>('/api/auth/mfa/setup', {});
  }

  /**
   * Enable MFA after setup - requires verification code from authenticator.
   */
  async mfaEnable(request: MfaEnableRequest): Promise<MfaEnableResponse> {
    return this.http.post<MfaEnableResponse>('/api/auth/mfa/enable', request);
  }

  /**
   * Disable MFA - requires verification code and password.
   */
  async mfaDisable(request: MfaDisableRequest): Promise<{ message: string }> {
    return this.http.post<{ message: string }>('/api/auth/mfa/disable', request);
  }

  /**
   * Verify a TOTP code (for testing during setup or re-verification).
   */
  async mfaVerify(request: MfaVerifyRequest): Promise<MfaVerifyResponse> {
    return this.http.post<MfaVerifyResponse>('/api/auth/mfa/verify', request);
  }

  /**
   * Generate new backup codes for MFA recovery.
   */
  async mfaBackupCodes(): Promise<MfaBackupCodesResponse> {
    return this.http.post<MfaBackupCodesResponse>('/api/auth/mfa/backup-codes', {});
  }
}

class BillingAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get available subscription plans.
   */
  async plans(): Promise<BillingPlan[]> {
    const response = await this.http.get<BillingPlansResponse>('/api/billing/plans');
    return response.plans;
  }

  /**
   * Get current usage for authenticated user.
   */
  async usage(): Promise<BillingUsage> {
    return this.http.get<BillingUsage>('/api/billing/usage');
  }

  /**
   * Get current subscription.
   */
  async subscription(): Promise<BillingSubscription> {
    return this.http.get<BillingSubscription>('/api/billing/subscription');
  }

  /**
   * Create a checkout session for a subscription.
   */
  async checkout(request: CheckoutRequest): Promise<CheckoutResponse> {
    return this.http.post<CheckoutResponse>('/api/billing/checkout', request);
  }

  /**
   * Create a billing portal session (for managing subscription in Stripe).
   */
  async portal(): Promise<PortalResponse> {
    return this.http.post<PortalResponse>('/api/billing/portal', {});
  }

  /**
   * Cancel subscription (at period end).
   */
  async cancel(): Promise<CancelSubscriptionResponse> {
    return this.http.post<CancelSubscriptionResponse>('/api/billing/cancel', {});
  }

  /**
   * Resume a canceled subscription.
   */
  async resume(): Promise<ResumeSubscriptionResponse> {
    return this.http.post<ResumeSubscriptionResponse>('/api/billing/resume', {});
  }

  /**
   * Get billing audit log.
   */
  async auditLog(options?: {
    limit?: number;
    offset?: number;
  }): Promise<BillingAuditLogResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));

    const query = params.toString();
    const path = query ? `/api/billing/audit-log?${query}` : '/api/billing/audit-log';
    return this.http.get<BillingAuditLogResponse>(path);
  }

  /**
   * Export usage data as CSV.
   */
  async exportUsage(): Promise<string> {
    const response = await this.http.get<{ csv: string }>('/api/billing/usage/export');
    return response.csv;
  }

  /**
   * Get usage forecast for remaining billing period.
   */
  async forecast(): Promise<UsageForecast> {
    return this.http.get<UsageForecast>('/api/billing/usage/forecast');
  }

  /**
   * Get invoice history.
   */
  async invoices(options?: { limit?: number }): Promise<Invoice[]> {
    const params = options?.limit ? `?limit=${options.limit}` : '';
    const response = await this.http.get<InvoicesResponse>(`/api/billing/invoices${params}`);
    return response.invoices;
  }
}

// =============================================================================
// Main Client
// =============================================================================

export class AragoraClient {
  private http: HttpClient;

  readonly debates: DebatesAPI;
  readonly batchDebates: BatchDebatesAPI;
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
  readonly tournaments: TournamentsAPI;
  readonly organizations: OrganizationsAPI;
  readonly analytics: AnalyticsAPI;
  readonly auth: AuthAPI;
  readonly billing: BillingAPI;

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
    this.batchDebates = new BatchDebatesAPI(this.http);
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
    this.tournaments = new TournamentsAPI(this.http);
    this.organizations = new OrganizationsAPI(this.http);
    this.analytics = new AnalyticsAPI(this.http);
    this.auth = new AuthAPI(this.http);
    this.billing = new BillingAPI(this.http);
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
