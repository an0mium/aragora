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
  DebateSearchResponse,
  DebateVerificationReport,
  DebateShareResponse,
  DebateBroadcastResponse,
  DebatePublishResponse,
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
  AgentDetailedProfile,
  AgentCalibrationData,
  AgentPerformanceMetrics,
  AgentConsistencyData,
  AgentFlipEvent,
  AgentNetworkGraph,
  AgentMoment,
  AgentPosition,
  AgentDomainExpertise,
  AgentHeadToHead,
  AgentOpponentBriefing,
  AgentComparisonResult,
  AgentGroundedPersona,
  LeaderboardEntry,
  LeaderboardResponse,
  // Gauntlet types
  GauntletRunRequest,
  GauntletRunResponse,
  GauntletReceipt,
  GauntletResultsResponse,
  GauntletPersonaInfo,
  GauntletHeatmap,
  GauntletComparisonResult,
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
  DisagreementAnalysis,
  RoleRotationStats,
  EarlyStopAnalysis,
  ConsensusQualityMetrics,
  RankingStatistics,
  MemoryStatistics,
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
  OAuthProvidersResponse,
  OAuthInitResponse,
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
  // Evidence types
  EvidenceSnippet,
  EvidenceListResponse,
  EvidenceSearchRequest,
  EvidenceSearchResponse,
  EvidenceCollectRequest,
  EvidenceCollectResponse,
  EvidenceStatistics,
  EvidenceDebateResponse,
  // Calibration types
  CalibrationCurveResponse,
  CalibrationSummary,
  CalibrationLeaderboardEntry,
  CalibrationLeaderboardResponse,
  CalibrationVisualizationResponse,
  // Insights types
  Insight,
  InsightsRecentResponse,
  FlipEvent,
  FlipsRecentResponse,
  ExtractInsightsRequest,
  ExtractInsightsResponse,
  // Belief Network types
  Crux,
  CruxesResponse,
  LoadBearingClaim,
  LoadBearingClaimsResponse,
  ClaimSupportResponse,
  GraphStatsResponse,
  // Consensus types
  SimilarDebate,
  ConsensusSimilarResponse,
  SettledTopic,
  ConsensusSettledResponse,
  ConsensusStatsResponse,
  Dissent,
  DissentsResponse,
  ContrarianView,
  ContrarianViewsResponse,
  RiskWarning,
  RiskWarningsResponse,
  DomainHistoryResponse,
  // Admin types
  AdminUser,
  AdminUsersResponse,
  AdminUserUpdateRequest,
  AdminOrganizationsResponse,
  AdminSystemStats,
  // Dashboard types
  DashboardAnalyticsParams,
  DashboardAnalyticsResponse,
  DashboardDebateMetricsParams,
  DashboardDebateMetricsResponse,
  DashboardAgentPerformanceParams,
  DashboardAgentPerformanceResponse,
  // System types
  SystemHealthResponse,
  SystemInfoResponse,
  SystemStatusResponse,
  // Features types
  FeatureFlag,
  FeaturesListResponse,
  FeatureStatusResponse,
  // Checkpoint types
  Checkpoint,
  CheckpointListResponse,
  ResumableDebate,
  // Webhook types
  Webhook,
  WebhookEventType,
  WebhookCreateRequest,
  WebhookUpdateRequest,
  WebhookTestResult,
  // Training types
  TrainingExportOptions,
  TrainingExportResponse,
  TrainingStats,
  TrainingFormat,
  // Gallery types
  GalleryListResponse,
  GalleryListOptions,
  GalleryDebateDetail,
  GalleryEmbed,
  // Persona extended types
  AgentPerformance,
  AgentDomainsResponse,
  AgentAccuracy,
  // Metrics types
  SystemMetrics,
  HealthMetrics,
  CacheMetrics,
  VerificationMetrics,
  SystemResourceMetrics,
  BackgroundJobMetrics,
  // Routing types
  TeamRecommendation,
  RoutingRecommendation,
  AutoRouteRequest,
  AutoRouteResponse,
  DomainDetectionResult,
  DomainLeaderboardEntry,
  // Plugin types
  Plugin,
  PluginListResponse,
  PluginDetails,
  PluginRunRequest,
  PluginRunResponse,
  InstalledPlugin,
  InstalledPluginsResponse,
  PluginInstallRequest,
  PluginInstallResponse,
  PluginSubmitRequest,
  PluginSubmitResponse,
  PluginSubmissionsResponse,
  PluginMarketplace,
  // Persona types
  AgentPersona,
  GroundedPersona,
  PersonaListResponse,
  IdentityPrompt,
  // Extended system types
  HistoryCycle,
  HistoryCyclesResponse,
  HistoryEvent,
  HistoryEventsResponse,
  HistorySummary,
  CircuitBreakerStatus,
  CircuitBreakersResponse,
  MaintenanceResult,
  // Extended admin types
  NomicStatus,
  ImpersonateResponse,
  // Selection plugin types
  ScorerInfo,
  TeamSelectorInfo,
  RoleAssignerInfo,
  SelectionPluginsResponse,
  ScoreAgentsRequest,
  ScoreAgentsResponse,
  SelectTeamRequest,
  SelectTeamResponse,
  // Capability probe types
  ProbeType,
  ProbeRunRequest,
  ProbeReport,
  // Formal verification extended types
  TranslateRequest,
  TranslateResponse,
  VerificationHistoryEntry,
  VerificationHistoryResponse,
  ProofTreeResponse,
  // Nomic loop types
  NomicState,
  NomicHealth,
  NomicMetrics,
  NomicLog,
  RiskRegister,
  ModesResponse,
  // Learning analytics types
  CycleSummariesResponse,
  LearnedPatternsResponse,
  AgentEvolutionResponse,
  AggregatedInsightsResponse,
  // Genesis types
  GenesisStats,
  GenesisEventType,
  GenesisEventsResponse,
  GenomesResponse,
  TopGenomesResponse,
  PopulationResponse,
  GenomeDetails,
  LineageResponse,
  DebateTreeResponse,
  // Evolution types
  EvolutionPatternsResponse,
  EvolutionSummaryResponse,
  AgentHistoryResponse,
  AgentPromptResponse,
  // Broadcast types
  BroadcastResult,
  BroadcastOptions,
  // Relationship types
  RelationshipSummaryResponse,
  RelationshipGraphResponse,
  RelationshipStatsResponse,
  PairDetailResponse,
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

  /**
   * Search across debates using full-text search.
   *
   * @param query - Search query string
   * @param options - Optional search parameters
   * @returns Promise resolving to search results
   */
  async search(
    query: string,
    options?: { limit?: number; offset?: number; status?: string }
  ): Promise<DebateSearchResponse> {
    const params = new URLSearchParams({ q: query });
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.status) params.set('status', options.status);

    return this.http.get<DebateSearchResponse>(`/api/search?${params}`);
  }

  /**
   * Get a debate by its URL slug.
   *
   * @param slug - The debate's URL-friendly slug
   * @returns Promise resolving to the debate
   */
  async getBySlug(slug: string): Promise<Debate> {
    const response = await this.http.get<Debate>(`/api/debates/slug/${slug}`);
    return normalizeDebateResponse(response);
  }

  /**
   * Get verification feedback report for a debate.
   *
   * @param debateId - The debate ID
   * @returns Promise resolving to the verification report
   */
  async verificationReport(debateId: string): Promise<DebateVerificationReport> {
    return this.http.get<DebateVerificationReport>(`/api/debates/${debateId}/verification-report`);
  }

  /**
   * List all forks created from a debate.
   *
   * @param debateId - The parent debate ID
   * @returns Promise resolving to list of forked debates
   */
  async listForks(debateId: string): Promise<Debate[]> {
    const response = await this.http.get<{ forks: Debate[] }>(`/api/debates/${debateId}/forks`);
    return response.forks.map(normalizeDebateResponse);
  }

  /**
   * Share a debate with a shareable link.
   *
   * @param debateId - The debate ID to share
   * @param options - Sharing options
   * @returns Promise resolving to share response with link
   */
  async share(
    debateId: string,
    options?: { expires_in_days?: number }
  ): Promise<DebateShareResponse> {
    return this.http.post<DebateShareResponse>(`/api/debates/${debateId}/share`, options || {});
  }

  /**
   * Revoke a shared debate link.
   *
   * @param debateId - The debate ID
   * @returns Promise resolving to success status
   */
  async revokeShare(debateId: string): Promise<{ message: string }> {
    return this.http.post<{ message: string }>(`/api/debates/${debateId}/share/revoke`, {});
  }

  /**
   * Get broadcast-ready content for a debate.
   *
   * @param debateId - The debate ID
   * @returns Promise resolving to broadcast content
   */
  async broadcast(debateId: string): Promise<DebateBroadcastResponse> {
    return this.http.get<DebateBroadcastResponse>(`/api/debates/${debateId}/broadcast`);
  }

  /**
   * Publish a debate to a social platform.
   *
   * @param debateId - The debate ID
   * @param platform - Target platform ('twitter' | 'youtube')
   * @returns Promise resolving to publish result
   */
  async publish(
    debateId: string,
    platform: 'twitter' | 'youtube',
    options?: { message?: string }
  ): Promise<DebatePublishResponse> {
    return this.http.post<DebatePublishResponse>(
      `/api/debates/${debateId}/publish/${platform}`,
      options || {}
    );
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

  /**
   * Translate a claim to formal language without verification.
   */
  async translate(request: TranslateRequest): Promise<TranslateResponse> {
    return this.http.post<TranslateResponse>('/api/verify/translate', request);
  }

  /**
   * Get verification history.
   */
  async history(options?: {
    limit?: number;
    offset?: number;
    status?: string;
  }): Promise<VerificationHistoryResponse> {
    const params = new URLSearchParams();
    if (options?.limit !== undefined) params.set('limit', String(options.limit));
    if (options?.offset !== undefined) params.set('offset', String(options.offset));
    if (options?.status) params.set('status', options.status);

    const query = params.toString();
    const url = query ? `/api/verify/history?${query}` : '/api/verify/history';
    return this.http.get<VerificationHistoryResponse>(url);
  }

  /**
   * Get a specific verification history entry.
   */
  async getHistoryEntry(entryId: string): Promise<VerificationHistoryEntry & { result: Record<string, unknown>; proof_tree?: unknown[] }> {
    return this.http.get(`/api/verify/history/${encodeURIComponent(entryId)}`);
  }

  /**
   * Get proof tree for a verification entry.
   */
  async getProofTree(entryId: string): Promise<ProofTreeResponse> {
    return this.http.get<ProofTreeResponse>(`/api/verify/history/${encodeURIComponent(entryId)}/tree`);
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

  /**
   * Get detailed profile for an agent.
   */
  async profile(agentId: string): Promise<AgentDetailedProfile> {
    return this.http.get<AgentDetailedProfile>(`/api/agent/${agentId}/profile`);
  }

  /**
   * Get calibration data for an agent.
   */
  async calibration(agentId: string): Promise<AgentCalibrationData> {
    return this.http.get<AgentCalibrationData>(`/api/agent/${agentId}/calibration`);
  }

  /**
   * Get performance metrics for an agent.
   */
  async performance(agentId: string, days = 30): Promise<AgentPerformanceMetrics> {
    return this.http.get<AgentPerformanceMetrics>(`/api/agent/${agentId}/performance?days=${days}`);
  }

  /**
   * Get consistency analysis for an agent.
   */
  async consistency(agentId: string): Promise<AgentConsistencyData> {
    return this.http.get<AgentConsistencyData>(`/api/agent/${agentId}/consistency`);
  }

  /**
   * Get position flips/changes for an agent.
   */
  async flips(agentId: string, limit = 20): Promise<AgentFlipEvent[]> {
    const response = await this.http.get<{ flips: AgentFlipEvent[] }>(
      `/api/agent/${agentId}/flips?limit=${limit}`
    );
    return response.flips;
  }

  /**
   * Get relationship network graph for an agent.
   */
  async network(agentId: string): Promise<AgentNetworkGraph> {
    return this.http.get<AgentNetworkGraph>(`/api/agent/${agentId}/network`);
  }

  /**
   * Get key moments for an agent across debates.
   */
  async moments(agentId: string, limit = 10): Promise<AgentMoment[]> {
    const response = await this.http.get<{ moments: AgentMoment[] }>(
      `/api/agent/${agentId}/moments?limit=${limit}`
    );
    return response.moments;
  }

  /**
   * Get positions held by an agent.
   */
  async positions(agentId: string, options?: { domain?: string; limit?: number }): Promise<AgentPosition[]> {
    const params = new URLSearchParams();
    if (options?.domain) params.set('domain', options.domain);
    if (options?.limit) params.set('limit', String(options.limit));

    const query = params.toString();
    const path = query ? `/api/agent/${agentId}/positions?${query}` : `/api/agent/${agentId}/positions`;
    const response = await this.http.get<{ positions: AgentPosition[] }>(path);
    return response.positions;
  }

  /**
   * Get domain expertise for an agent.
   */
  async domains(agentId: string): Promise<AgentDomainExpertise[]> {
    const response = await this.http.get<{ domains: AgentDomainExpertise[] }>(
      `/api/agent/${agentId}/domains`
    );
    return response.domains;
  }

  /**
   * Get head-to-head comparison between two agents.
   */
  async headToHead(agentId: string, opponentId: string): Promise<AgentHeadToHead> {
    return this.http.get<AgentHeadToHead>(`/api/agent/${agentId}/head-to-head/${opponentId}`);
  }

  /**
   * Get tactical briefing about an opponent.
   */
  async opponentBriefing(agentId: string, opponentId: string): Promise<AgentOpponentBriefing> {
    return this.http.get<AgentOpponentBriefing>(`/api/agent/${agentId}/opponent-briefing/${opponentId}`);
  }

  /**
   * Compare multiple agents.
   */
  async compare(agentIds: string[]): Promise<AgentComparisonResult> {
    return this.http.post<AgentComparisonResult>('/api/agent/compare', { agents: agentIds });
  }

  /**
   * Get grounded persona for an agent.
   */
  async groundedPersona(agentId: string): Promise<AgentGroundedPersona> {
    return this.http.get<AgentGroundedPersona>(`/api/agent/${agentId}/grounded-persona`);
  }

  /**
   * Get identity prompt for an agent.
   */
  async identityPrompt(agentId: string): Promise<{ prompt: string }> {
    return this.http.get<{ prompt: string }>(`/api/agent/${agentId}/identity-prompt`);
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

  /**
   * Get a specific gauntlet by ID.
   */
  async get(gauntletId: string): Promise<GauntletReceipt> {
    return this.http.get<GauntletReceipt>(`/api/gauntlet/${gauntletId}`);
  }

  /**
   * List all gauntlet results.
   */
  async results(options?: { limit?: number; offset?: number; persona?: string }): Promise<GauntletResultsResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.persona) params.set('persona', options.persona);

    const query = params.toString();
    const path = query ? `/api/gauntlet/results?${query}` : '/api/gauntlet/results';
    return this.http.get<GauntletResultsResponse>(path);
  }

  /**
   * Get available gauntlet personas.
   */
  async personas(): Promise<GauntletPersonaInfo[]> {
    const response = await this.http.get<{ personas: GauntletPersonaInfo[] }>('/api/gauntlet/personas');
    return response.personas;
  }

  /**
   * Get performance heatmap for a gauntlet run.
   */
  async heatmap(gauntletId: string): Promise<GauntletHeatmap> {
    return this.http.get<GauntletHeatmap>(`/api/gauntlet/${gauntletId}/heatmap`);
  }

  /**
   * Compare two gauntlet runs.
   */
  async compare(gauntletId1: string, gauntletId2: string): Promise<GauntletComparisonResult> {
    return this.http.get<GauntletComparisonResult>(`/api/gauntlet/${gauntletId1}/compare/${gauntletId2}`);
  }

  /**
   * Delete a gauntlet run.
   */
  async delete(gauntletId: string): Promise<{ message: string }> {
    return this.http.delete<{ message: string }>(`/api/gauntlet/${gauntletId}`);
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

  /**
   * Get disagreement analysis across debates.
   */
  async disagreements(options?: { days?: number; limit?: number }): Promise<DisagreementAnalysis> {
    const params = new URLSearchParams();
    if (options?.days) params.set('days', String(options.days));
    if (options?.limit) params.set('limit', String(options.limit));

    const query = params.toString();
    const path = query ? `/api/analytics/disagreements?${query}` : '/api/analytics/disagreements';
    return this.http.get<DisagreementAnalysis>(path);
  }

  /**
   * Get role rotation statistics.
   */
  async roleRotation(days = 30): Promise<RoleRotationStats> {
    return this.http.get<RoleRotationStats>(`/api/analytics/role-rotation?days=${days}`);
  }

  /**
   * Get early stop analysis.
   */
  async earlyStops(days = 30): Promise<EarlyStopAnalysis> {
    return this.http.get<EarlyStopAnalysis>(`/api/analytics/early-stops?days=${days}`);
  }

  /**
   * Get consensus quality metrics.
   */
  async consensusQuality(days = 30): Promise<ConsensusQualityMetrics> {
    return this.http.get<ConsensusQualityMetrics>(`/api/analytics/consensus-quality?days=${days}`);
  }

  /**
   * Get ranking statistics.
   */
  async rankingStats(): Promise<RankingStatistics> {
    return this.http.get<RankingStatistics>('/api/ranking/stats');
  }

  /**
   * Get memory statistics.
   */
  async memoryStats(): Promise<MemoryStatistics> {
    return this.http.get<MemoryStatistics>('/api/memory/stats');
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

  // =========================================================================
  // OAuth Methods
  // =========================================================================

  /**
   * Get available OAuth providers.
   */
  async oauthProviders(): Promise<OAuthProvidersResponse> {
    return this.http.get<OAuthProvidersResponse>('/api/auth/oauth/providers');
  }

  /**
   * Initiate OAuth flow with Google.
   * Returns a redirect URL to Google's OAuth consent screen.
   */
  async oauthGoogle(returnUrl?: string): Promise<OAuthInitResponse> {
    const params = returnUrl ? { return_url: returnUrl } : {};
    return this.http.get<OAuthInitResponse>(`/api/auth/oauth/google?${new URLSearchParams(params as Record<string, string>)}`);
  }

  /**
   * Complete OAuth callback from Google.
   */
  async oauthGoogleCallback(code: string, state?: string): Promise<LoginResponse> {
    return this.http.post<LoginResponse>('/api/auth/oauth/google/callback', { code, state });
  }

  /**
   * Link an OAuth provider to the current account.
   */
  async linkOauth(provider: string, code: string): Promise<{ message: string }> {
    return this.http.post<{ message: string }>('/api/auth/oauth/link', { provider, code });
  }

  /**
   * Unlink an OAuth provider from the current account.
   */
  async unlinkOauth(provider: string): Promise<{ message: string }> {
    return this.http.post<{ message: string }>('/api/auth/oauth/unlink', { provider });
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
// Evidence API
// =============================================================================

class EvidenceAPI {
  constructor(private http: HttpClient) {}

  /**
   * List all evidence with optional filtering and pagination.
   */
  async list(options?: {
    limit?: number;
    offset?: number;
    source?: string;
    min_reliability?: number;
  }): Promise<EvidenceListResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.source) params.set('source', options.source);
    if (options?.min_reliability) params.set('min_reliability', String(options.min_reliability));

    const query = params.toString();
    const path = query ? `/api/evidence?${query}` : '/api/evidence';
    return this.http.get<EvidenceListResponse>(path);
  }

  /**
   * Get a specific evidence item by ID.
   */
  async get(evidenceId: string): Promise<EvidenceSnippet> {
    const response = await this.http.get<{ evidence: EvidenceSnippet }>(
      `/api/evidence/${evidenceId}`
    );
    return response.evidence;
  }

  /**
   * Search evidence with full-text query.
   */
  async search(request: EvidenceSearchRequest): Promise<EvidenceSearchResponse> {
    return this.http.post<EvidenceSearchResponse>('/api/evidence/search', request);
  }

  /**
   * Collect evidence for a topic/task.
   * This triggers active evidence collection from configured connectors.
   */
  async collect(request: EvidenceCollectRequest): Promise<EvidenceCollectResponse> {
    return this.http.post<EvidenceCollectResponse>('/api/evidence/collect', request);
  }

  /**
   * Get evidence associated with a specific debate.
   */
  async forDebate(debateId: string, round?: number): Promise<EvidenceDebateResponse> {
    const params = round !== undefined ? `?round=${round}` : '';
    return this.http.get<EvidenceDebateResponse>(
      `/api/evidence/debate/${debateId}${params}`
    );
  }

  /**
   * Associate evidence with a debate.
   */
  async associateWithDebate(
    debateId: string,
    evidenceIds: string[],
    options?: { round?: number; relevance_score?: number }
  ): Promise<{ debate_id: string; associated: string[]; count: number }> {
    return this.http.post(`/api/evidence/debate/${debateId}`, {
      evidence_ids: evidenceIds,
      round: options?.round,
      relevance_score: options?.relevance_score,
    });
  }

  /**
   * Get evidence store statistics.
   */
  async statistics(): Promise<EvidenceStatistics> {
    const response = await this.http.get<{ statistics: EvidenceStatistics }>(
      '/api/evidence/statistics'
    );
    return response.statistics;
  }

  /**
   * Delete evidence by ID.
   */
  async delete(evidenceId: string): Promise<boolean> {
    const response = await this.http.delete<{ deleted: boolean }>(
      `/api/evidence/${evidenceId}`
    );
    return response.deleted;
  }
}

// =============================================================================
// Calibration API
// =============================================================================

class CalibrationAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get calibration leaderboard showing top calibrated agents.
   */
  async leaderboard(options?: {
    limit?: number;
    metric?: 'brier' | 'ece' | 'accuracy';
    min_predictions?: number;
  }): Promise<CalibrationLeaderboardEntry[]> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.metric) params.set('metric', options.metric);
    if (options?.min_predictions) params.set('min_predictions', String(options.min_predictions));

    const query = params.toString();
    const path = query ? `/api/calibration/leaderboard?${query}` : '/api/calibration/leaderboard';
    const response = await this.http.get<CalibrationLeaderboardResponse>(path);
    return response.agents;
  }

  /**
   * Get calibration curve for a specific agent.
   */
  async curve(
    agentName: string,
    options?: { buckets?: number; domain?: string }
  ): Promise<CalibrationCurveResponse> {
    const params = new URLSearchParams();
    if (options?.buckets) params.set('buckets', String(options.buckets));
    if (options?.domain) params.set('domain', options.domain);

    const query = params.toString();
    const path = query
      ? `/api/agent/${agentName}/calibration-curve?${query}`
      : `/api/agent/${agentName}/calibration-curve`;
    return this.http.get<CalibrationCurveResponse>(path);
  }

  /**
   * Get calibration summary metrics for a specific agent.
   */
  async summary(agentName: string, domain?: string): Promise<CalibrationSummary> {
    const params = domain ? `?domain=${encodeURIComponent(domain)}` : '';
    return this.http.get<CalibrationSummary>(
      `/api/agent/${agentName}/calibration-summary${params}`
    );
  }

  /**
   * Get calibration visualization data for multiple agents.
   */
  async visualization(limit = 5): Promise<CalibrationVisualizationResponse> {
    return this.http.get<CalibrationVisualizationResponse>(
      `/api/calibration/visualization?limit=${limit}`
    );
  }
}

// =============================================================================
// Insights API
// =============================================================================

class InsightsAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get recent insights from debate analysis.
   */
  async recent(limit = 10): Promise<Insight[]> {
    const response = await this.http.get<InsightsRecentResponse>(
      `/api/insights/recent?limit=${limit}`
    );
    return response.insights;
  }

  /**
   * Get recent position flips detected across agents.
   */
  async flips(options?: { limit?: number; agent?: string }): Promise<FlipEvent[]> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.agent) params.set('agent', options.agent);

    const query = params.toString();
    const path = query ? `/api/flips/recent?${query}` : '/api/flips/recent';
    const response = await this.http.get<FlipsRecentResponse>(path);
    return response.flips;
  }

  /**
   * Extract detailed insights from content using AI analysis.
   */
  async extractDetailed(request: ExtractInsightsRequest): Promise<ExtractInsightsResponse> {
    return this.http.post<ExtractInsightsResponse>('/api/insights/extract-detailed', request);
  }
}

// =============================================================================
// Belief Network API
// =============================================================================

class BeliefNetworkAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get cruxes (key claims) for a debate that most impact the outcome.
   */
  async cruxes(debateId: string, topK = 3): Promise<Crux[]> {
    const response = await this.http.get<CruxesResponse>(
      `/api/belief-network/${debateId}/cruxes?top_k=${topK}`
    );
    return response.cruxes;
  }

  /**
   * Get load-bearing claims with high centrality in the argument graph.
   */
  async loadBearingClaims(debateId: string, limit = 5): Promise<LoadBearingClaim[]> {
    const response = await this.http.get<LoadBearingClaimsResponse>(
      `/api/belief-network/${debateId}/load-bearing-claims?limit=${limit}`
    );
    return response.claims;
  }

  /**
   * Get support/contradiction evidence for a specific claim.
   */
  async claimSupport(debateId: string, claimId: string): Promise<ClaimSupportResponse> {
    return this.http.get<ClaimSupportResponse>(
      `/api/provenance/${debateId}/claims/${claimId}/support`
    );
  }

  /**
   * Get argument graph statistics for a debate.
   */
  async graphStats(debateId: string): Promise<GraphStatsResponse> {
    return this.http.get<GraphStatsResponse>(`/api/debate/${debateId}/graph-stats`);
  }
}

// =============================================================================
// Consensus API
// =============================================================================

class ConsensusAPI {
  constructor(private http: HttpClient) {}

  /**
   * Find debates similar to a given topic.
   */
  async similar(
    topic: string,
    options?: { limit?: number; min_similarity?: number }
  ): Promise<SimilarDebate[]> {
    const params = new URLSearchParams({ topic });
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.min_similarity) params.set('min_similarity', String(options.min_similarity));

    const response = await this.http.get<ConsensusSimilarResponse>(
      `/api/consensus/similar?${params}`
    );
    return response.similar_debates;
  }

  /**
   * Get high-confidence settled topics from consensus memory.
   */
  async settled(options?: {
    limit?: number;
    min_confidence?: number;
    domain?: string;
  }): Promise<SettledTopic[]> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.min_confidence) params.set('min_confidence', String(options.min_confidence));
    if (options?.domain) params.set('domain', options.domain);

    const query = params.toString();
    const path = query ? `/api/consensus/settled?${query}` : '/api/consensus/settled';
    const response = await this.http.get<ConsensusSettledResponse>(path);
    return response.topics;
  }

  /**
   * Get consensus memory statistics.
   */
  async stats(): Promise<ConsensusStatsResponse> {
    return this.http.get<ConsensusStatsResponse>('/api/consensus/stats');
  }

  /**
   * Get recent dissenting views from debates.
   */
  async dissents(limit = 10): Promise<Dissent[]> {
    const response = await this.http.get<DissentsResponse>(
      `/api/consensus/dissents?limit=${limit}`
    );
    return response.dissents;
  }

  /**
   * Get contrarian perspectives worth considering.
   */
  async contrarianViews(limit = 10): Promise<ContrarianView[]> {
    const response = await this.http.get<ContrarianViewsResponse>(
      `/api/consensus/contrarian-views?limit=${limit}`
    );
    return response.views;
  }

  /**
   * Get risk warnings and edge cases from debate analysis.
   */
  async riskWarnings(limit = 10): Promise<RiskWarning[]> {
    const response = await this.http.get<RiskWarningsResponse>(
      `/api/consensus/risk-warnings?limit=${limit}`
    );
    return response.warnings;
  }

  /**
   * Get debate history for a specific domain.
   */
  async domainHistory(domain: string, limit = 20): Promise<DomainHistoryResponse> {
    return this.http.get<DomainHistoryResponse>(
      `/api/consensus/domain/${encodeURIComponent(domain)}?limit=${limit}`
    );
  }
}

// =============================================================================
// Admin API
// =============================================================================

/**
 * API client for admin operations.
 *
 * Requires admin privileges to use these endpoints.
 *
 * @example
 * ```typescript
 * // Get all users
 * const users = await client.admin.getUsers({ limit: 50 });
 *
 * // Get system statistics
 * const stats = await client.admin.getSystemStats();
 * ```
 */
class AdminAPI {
  constructor(private http: HttpClient) {}

  /**
   * List all users with pagination.
   */
  async getUsers(options?: {
    limit?: number;
    offset?: number;
    role?: string;
    is_active?: boolean;
  }): Promise<AdminUsersResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.role) params.set('role', options.role);
    if (options?.is_active !== undefined) params.set('is_active', String(options.is_active));

    const query = params.toString();
    const path = query ? `/api/admin/users?${query}` : '/api/admin/users';
    return this.http.get<AdminUsersResponse>(path);
  }

  /**
   * Get a specific user by ID.
   */
  async getUser(userId: string): Promise<AdminUser> {
    const response = await this.http.get<{ user: AdminUser }>(`/api/admin/users/${userId}`);
    return response.user;
  }

  /**
   * Update a user's information.
   */
  async updateUser(userId: string, data: AdminUserUpdateRequest): Promise<AdminUser> {
    const response = await this.http.put<{ user: AdminUser }>(`/api/admin/users/${userId}`, data);
    return response.user;
  }

  /**
   * Delete a user by ID.
   */
  async deleteUser(userId: string): Promise<boolean> {
    await this.http.delete(`/api/admin/users/${userId}`);
    return true;
  }

  /**
   * List all organizations with pagination.
   */
  async getOrganizations(options?: {
    limit?: number;
    offset?: number;
    tier?: string;
    is_active?: boolean;
  }): Promise<AdminOrganizationsResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.tier) params.set('tier', options.tier);
    if (options?.is_active !== undefined) params.set('is_active', String(options.is_active));

    const query = params.toString();
    const path = query ? `/api/admin/organizations?${query}` : '/api/admin/organizations';
    return this.http.get<AdminOrganizationsResponse>(path);
  }

  /**
   * Get system-wide statistics.
   */
  async getSystemStats(): Promise<AdminSystemStats> {
    return this.http.get<AdminSystemStats>('/api/admin/stats');
  }

  /**
   * Impersonate a user (admin only).
   */
  async impersonateUser(userId: string): Promise<ImpersonateResponse> {
    return this.http.post<ImpersonateResponse>(`/api/admin/impersonate/${userId}`, {});
  }

  /**
   * Deactivate a user account.
   */
  async deactivateUser(userId: string): Promise<{ message: string }> {
    return this.http.post<{ message: string }>(`/api/admin/users/${userId}/deactivate`, {});
  }

  /**
   * Activate a user account.
   */
  async activateUser(userId: string): Promise<{ message: string }> {
    return this.http.post<{ message: string }>(`/api/admin/users/${userId}/activate`, {});
  }

  /**
   * Unlock a locked user account.
   */
  async unlockUser(userId: string): Promise<{ message: string }> {
    return this.http.post<{ message: string }>(`/api/admin/users/${userId}/unlock`, {});
  }

  /**
   * Get nomic loop status.
   */
  async getNomicStatus(): Promise<NomicStatus> {
    return this.http.get<NomicStatus>('/api/admin/nomic/status');
  }

  /**
   * Pause the nomic loop.
   */
  async pauseNomic(): Promise<{ message: string }> {
    return this.http.post<{ message: string }>('/api/admin/nomic/pause', {});
  }

  /**
   * Resume the nomic loop.
   */
  async resumeNomic(): Promise<{ message: string }> {
    return this.http.post<{ message: string }>('/api/admin/nomic/resume', {});
  }

  /**
   * Reset nomic phase.
   */
  async resetNomic(phase?: string): Promise<{ message: string }> {
    return this.http.post<{ message: string }>('/api/admin/nomic/reset', { phase });
  }
}

// =============================================================================
// Dashboard API
// =============================================================================

/**
 * API client for dashboard analytics.
 *
 * @example
 * ```typescript
 * // Get analytics data
 * const analytics = await client.dashboard.getAnalytics({
 *   start_date: '2024-01-01',
 *   granularity: 'day',
 * });
 *
 * // Get agent performance metrics
 * const performance = await client.dashboard.getAgentPerformance({ days: 30 });
 * ```
 */
class DashboardAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get analytics data for a time period.
   */
  async getAnalytics(params?: DashboardAnalyticsParams): Promise<DashboardAnalyticsResponse> {
    const searchParams = new URLSearchParams();
    if (params?.start_date) searchParams.set('start_date', params.start_date);
    if (params?.end_date) searchParams.set('end_date', params.end_date);
    if (params?.granularity) searchParams.set('granularity', params.granularity);
    if (params?.metrics) searchParams.set('metrics', params.metrics.join(','));

    const query = searchParams.toString();
    const path = query ? `/api/dashboard/analytics?${query}` : '/api/dashboard/analytics';
    return this.http.get<DashboardAnalyticsResponse>(path);
  }

  /**
   * Get debate-specific metrics.
   */
  async getDebateMetrics(params?: DashboardDebateMetricsParams): Promise<DashboardDebateMetricsResponse> {
    const searchParams = new URLSearchParams();
    if (params?.days) searchParams.set('days', String(params.days));
    if (params?.agent) searchParams.set('agent', params.agent);
    if (params?.status) searchParams.set('status', params.status);

    const query = searchParams.toString();
    const path = query ? `/api/dashboard/debates?${query}` : '/api/dashboard/debates';
    return this.http.get<DashboardDebateMetricsResponse>(path);
  }

  /**
   * Get agent performance statistics.
   */
  async getAgentPerformance(params?: DashboardAgentPerformanceParams): Promise<DashboardAgentPerformanceResponse> {
    const searchParams = new URLSearchParams();
    if (params?.days) searchParams.set('days', String(params.days));
    if (params?.min_debates) searchParams.set('min_debates', String(params.min_debates));

    const query = searchParams.toString();
    const path = query ? `/api/dashboard/agents?${query}` : '/api/dashboard/agents';
    return this.http.get<DashboardAgentPerformanceResponse>(path);
  }
}

// =============================================================================
// System API
// =============================================================================

/**
 * API client for system information and health.
 *
 * @example
 * ```typescript
 * // Check system health
 * const health = await client.system.health();
 *
 * // Get system information
 * const info = await client.system.info();
 *
 * // Get detailed status
 * const status = await client.system.status();
 * ```
 */
class SystemAPI {
  constructor(private http: HttpClient) {}

  /**
   * Check system health status.
   */
  async health(): Promise<SystemHealthResponse> {
    return this.http.get<SystemHealthResponse>('/api/system/health');
  }

  /**
   * Get system information including version and features.
   */
  async info(): Promise<SystemInfoResponse> {
    return this.http.get<SystemInfoResponse>('/api/system/info');
  }

  /**
   * Get detailed system status including service health.
   */
  async status(): Promise<SystemStatusResponse> {
    return this.http.get<SystemStatusResponse>('/api/system/status');
  }

  /**
   * Get history cycles.
   */
  async historyCycles(options?: { limit?: number }): Promise<HistoryCycle[]> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    const query = params.toString() ? `?${params}` : '';
    const response = await this.http.get<HistoryCyclesResponse>(`/api/history/cycles${query}`);
    return response.cycles;
  }

  /**
   * Get history events.
   */
  async historyEvents(options?: { limit?: number; type?: string }): Promise<HistoryEvent[]> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.type) params.set('type', options.type);
    const query = params.toString() ? `?${params}` : '';
    const response = await this.http.get<HistoryEventsResponse>(`/api/history/events${query}`);
    return response.events;
  }

  /**
   * Get history summary.
   */
  async historySummary(): Promise<HistorySummary> {
    return this.http.get<HistorySummary>('/api/history/summary');
  }

  /**
   * Get circuit breaker status.
   */
  async circuitBreakers(): Promise<CircuitBreakerStatus[]> {
    const response = await this.http.get<CircuitBreakersResponse>('/api/circuit-breakers');
    return response.breakers;
  }

  /**
   * Perform database maintenance (admin only).
   */
  async maintenance(action: string): Promise<MaintenanceResult> {
    return this.http.post<MaintenanceResult>('/api/system/maintenance', { action });
  }
}

// =============================================================================
// Features API
// =============================================================================

/**
 * API client for feature flags.
 *
 * @example
 * ```typescript
 * // List all feature flags
 * const features = await client.features.list();
 *
 * // Check if a specific feature is enabled
 * const enabled = await client.features.isEnabled('new_debate_ui');
 * ```
 */
class FeaturesAPI {
  constructor(private http: HttpClient) {}

  /**
   * List all feature flags.
   */
  async list(): Promise<FeatureFlag[]> {
    const response = await this.http.get<FeaturesListResponse>('/api/features');
    return response.features;
  }

  /**
   * Get a specific feature flag by name.
   */
  async get(name: string): Promise<FeatureStatusResponse> {
    return this.http.get<FeatureStatusResponse>(`/api/features/${encodeURIComponent(name)}`);
  }

  /**
   * Check if a feature is enabled.
   */
  async isEnabled(name: string): Promise<boolean> {
    const feature = await this.get(name);
    return feature.enabled;
  }
}

// =============================================================================
// Checkpoints API
// =============================================================================

class CheckpointsAPI {
  constructor(private http: HttpClient) {}

  /**
   * List all checkpoints.
   */
  async list(options?: { limit?: number; offset?: number }): Promise<CheckpointListResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));

    const query = params.toString();
    const path = query ? `/api/checkpoints?${query}` : '/api/checkpoints';
    return this.http.get<CheckpointListResponse>(path);
  }

  /**
   * Get resumable debates.
   */
  async resumable(): Promise<ResumableDebate[]> {
    const response = await this.http.get<{ debates: ResumableDebate[] }>('/api/checkpoints/resumable');
    return response.debates;
  }

  /**
   * Create a checkpoint for a debate.
   */
  async create(debateId: string, options?: { name?: string }): Promise<Checkpoint> {
    return this.http.post<Checkpoint>('/api/checkpoints', { debate_id: debateId, ...options });
  }

  /**
   * Resume a debate from a checkpoint.
   */
  async resume(checkpointId: string): Promise<DebateCreateResponse> {
    return this.http.post<DebateCreateResponse>(`/api/checkpoints/${checkpointId}/resume`, {});
  }

  /**
   * Delete a checkpoint.
   */
  async delete(checkpointId: string): Promise<{ message: string }> {
    return this.http.delete<{ message: string }>(`/api/checkpoints/${checkpointId}`);
  }
}

// =============================================================================
// Webhooks API
// =============================================================================

class WebhooksAPI {
  constructor(private http: HttpClient) {}

  /**
   * List all webhooks.
   */
  async list(): Promise<Webhook[]> {
    const response = await this.http.get<{ webhooks: Webhook[] }>('/api/webhooks');
    return response.webhooks;
  }

  /**
   * Get webhook event types.
   */
  async eventTypes(): Promise<WebhookEventType[]> {
    const response = await this.http.get<{ events: WebhookEventType[] }>('/api/webhooks/events');
    return response.events;
  }

  /**
   * Create a new webhook.
   */
  async create(request: WebhookCreateRequest): Promise<Webhook> {
    return this.http.post<Webhook>('/api/webhooks', request);
  }

  /**
   * Update a webhook.
   */
  async update(webhookId: string, request: WebhookUpdateRequest): Promise<Webhook> {
    return this.http.put<Webhook>(`/api/webhooks/${webhookId}`, request);
  }

  /**
   * Delete a webhook.
   */
  async delete(webhookId: string): Promise<{ message: string }> {
    return this.http.delete<{ message: string }>(`/api/webhooks/${webhookId}`);
  }

  /**
   * Test a webhook.
   */
  async test(webhookId: string): Promise<WebhookTestResult> {
    return this.http.post<WebhookTestResult>(`/api/webhooks/${webhookId}/test`, {});
  }
}

// =============================================================================
// Training Export API
// =============================================================================

class TrainingAPI {
  constructor(private http: HttpClient) {}

  /**
   * Export training data in SFT format.
   */
  async exportSft(options?: TrainingExportOptions): Promise<TrainingExportResponse> {
    return this.http.post<TrainingExportResponse>('/api/training/export/sft', options || {});
  }

  /**
   * Export training data in DPO format.
   */
  async exportDpo(options?: TrainingExportOptions): Promise<TrainingExportResponse> {
    return this.http.post<TrainingExportResponse>('/api/training/export/dpo', options || {});
  }

  /**
   * Export adversarial training data from gauntlet runs.
   */
  async exportGauntlet(options?: TrainingExportOptions): Promise<TrainingExportResponse> {
    return this.http.post<TrainingExportResponse>('/api/training/export/gauntlet', options || {});
  }

  /**
   * Get training data statistics.
   */
  async stats(): Promise<TrainingStats> {
    return this.http.get<TrainingStats>('/api/training/stats');
  }

  /**
   * Get supported export formats.
   */
  async formats(): Promise<TrainingFormat[]> {
    const response = await this.http.get<{ formats: TrainingFormat[] }>('/api/training/formats');
    return response.formats;
  }
}

// =============================================================================
// Gallery API
// =============================================================================

class GalleryAPI {
  constructor(private http: HttpClient) {}

  /**
   * List public debates from the gallery.
   */
  async list(options?: GalleryListOptions): Promise<GalleryListResponse> {
    const params = new URLSearchParams();
    if (options?.limit !== undefined) params.set('limit', String(options.limit));
    if (options?.offset !== undefined) params.set('offset', String(options.offset));
    if (options?.agent) params.set('agent', options.agent);

    const query = params.toString();
    const url = query ? `/api/gallery?${query}` : '/api/gallery';
    return this.http.get<GalleryListResponse>(url);
  }

  /**
   * Get a specific debate from the gallery by its stable ID.
   */
  async get(debateId: string): Promise<GalleryDebateDetail> {
    return this.http.get<GalleryDebateDetail>(`/api/gallery/${debateId}`);
  }

  /**
   * Get an embeddable summary of a debate for sharing.
   */
  async embed(debateId: string): Promise<GalleryEmbed> {
    return this.http.get<GalleryEmbed>(`/api/gallery/${debateId}/embed`);
  }
}

// =============================================================================
// Metrics API
// =============================================================================

class MetricsAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get overall system metrics.
   */
  async get(): Promise<SystemMetrics> {
    return this.http.get<SystemMetrics>('/api/metrics');
  }

  /**
   * Get health metrics.
   */
  async health(): Promise<HealthMetrics> {
    return this.http.get<HealthMetrics>('/api/metrics/health');
  }

  /**
   * Get cache metrics.
   */
  async cache(): Promise<CacheMetrics> {
    return this.http.get<CacheMetrics>('/api/metrics/cache');
  }

  /**
   * Get verification metrics.
   */
  async verification(): Promise<VerificationMetrics> {
    return this.http.get<VerificationMetrics>('/api/metrics/verification');
  }

  /**
   * Get system resource metrics.
   */
  async system(): Promise<SystemResourceMetrics> {
    return this.http.get<SystemResourceMetrics>('/api/metrics/system');
  }

  /**
   * Get background job metrics.
   */
  async background(): Promise<BackgroundJobMetrics> {
    return this.http.get<BackgroundJobMetrics>('/api/metrics/background');
  }

  /**
   * Get Prometheus-formatted metrics.
   */
  async prometheus(): Promise<string> {
    const response = await this.http.get<{ metrics: string }>('/metrics');
    return response.metrics;
  }
}

// =============================================================================
// Routing API
// =============================================================================

class RoutingAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get recommended agent teams for a task.
   */
  async bestTeams(task: string, options?: { limit?: number }): Promise<TeamRecommendation[]> {
    const params = new URLSearchParams({ task });
    if (options?.limit) params.set('limit', String(options.limit));
    const response = await this.http.get<{ teams: TeamRecommendation[] }>(`/api/routing/best-teams?${params}`);
    return response.teams;
  }

  /**
   * Get routing recommendations.
   */
  async recommendations(task: string): Promise<RoutingRecommendation> {
    return this.http.get<RoutingRecommendation>(`/api/routing/recommendations?task=${encodeURIComponent(task)}`);
  }

  /**
   * Auto-route a task to the best agents.
   */
  async autoRoute(request: AutoRouteRequest): Promise<AutoRouteResponse> {
    return this.http.post<AutoRouteResponse>('/api/routing/auto-route', request);
  }

  /**
   * Detect domain for a task.
   */
  async detectDomain(task: string): Promise<DomainDetectionResult> {
    return this.http.post<DomainDetectionResult>('/api/routing/detect-domain', { task });
  }

  /**
   * Get domain leaderboard.
   */
  async domainLeaderboard(domain?: string): Promise<DomainLeaderboardEntry[]> {
    const params = domain ? `?domain=${encodeURIComponent(domain)}` : '';
    const response = await this.http.get<{ entries: DomainLeaderboardEntry[] }>(`/api/routing/domain-leaderboard${params}`);
    return response.entries;
  }
}

// =============================================================================
// Selection Plugins API
// =============================================================================

/**
 * API for agent selection plugins.
 * Provides access to scorers, team selectors, and role assigners.
 */
class SelectionAPI {
  constructor(private http: HttpClient) {}

  /**
   * List all available selection plugins.
   */
  async listPlugins(): Promise<SelectionPluginsResponse> {
    return this.http.get<SelectionPluginsResponse>('/api/selection/plugins');
  }

  /**
   * Get default plugin configuration.
   */
  async getDefaults(): Promise<{ scorer: string; team_selector: string; role_assigner: string }> {
    return this.http.get<{ scorer: string; team_selector: string; role_assigner: string }>('/api/selection/defaults');
  }

  /**
   * Get information about a specific scorer.
   */
  async getScorer(name: string): Promise<ScorerInfo> {
    return this.http.get<ScorerInfo>(`/api/selection/scorers/${encodeURIComponent(name)}`);
  }

  /**
   * Get information about a specific team selector.
   */
  async getTeamSelector(name: string): Promise<TeamSelectorInfo> {
    return this.http.get<TeamSelectorInfo>(`/api/selection/team-selectors/${encodeURIComponent(name)}`);
  }

  /**
   * Get information about a specific role assigner.
   */
  async getRoleAssigner(name: string): Promise<RoleAssignerInfo> {
    return this.http.get<RoleAssignerInfo>(`/api/selection/role-assigners/${encodeURIComponent(name)}`);
  }

  /**
   * Score agents for a task.
   * Returns agents ranked by their suitability for the task.
   */
  async scoreAgents(request: ScoreAgentsRequest): Promise<ScoreAgentsResponse> {
    return this.http.post<ScoreAgentsResponse>('/api/selection/score', request);
  }

  /**
   * Select an optimal team for a task.
   * Uses configured scorer, team selector, and role assigner plugins.
   */
  async selectTeam(request: SelectTeamRequest): Promise<SelectTeamResponse> {
    return this.http.post<SelectTeamResponse>('/api/selection/team', request);
  }
}

// =============================================================================
// Capability Probes API
// =============================================================================

/**
 * API for running capability probes against agents.
 *
 * Capability probes test agents for common vulnerabilities like:
 * - Contradiction (inconsistent reasoning)
 * - Hallucination (fabricated information)
 * - Sycophancy (agreeing without evidence)
 * - Persistence (abandoning positions too easily)
 * - Confidence calibration (overconfidence/underconfidence)
 * - Reasoning depth (shallow analysis)
 * - Edge cases (unusual inputs)
 *
 * @example
 * ```typescript
 * const report = await client.probes.run({
 *   agent_name: 'claude',
 *   probe_types: ['contradiction', 'hallucination'],
 *   probes_per_type: 3
 * });
 * console.log(`Vulnerability rate: ${report.vulnerability_rate}`);
 * ```
 */
class ProbesAPI {
  constructor(private http: HttpClient) {}

  /**
   * Run capability probes on an agent.
   *
   * @param request - Probe configuration
   * @returns Detailed probe report with vulnerabilities found
   */
  async run(request: ProbeRunRequest): Promise<ProbeReport> {
    return this.http.post<ProbeReport>('/api/probes/capability', request);
  }

  /**
   * Get available probe types.
   */
  getProbeTypes(): ProbeType[] {
    return [
      'contradiction',
      'hallucination',
      'sycophancy',
      'persistence',
      'confidence_calibration',
      'reasoning_depth',
      'edge_case',
    ];
  }
}

// =============================================================================
// Nomic Loop API
// =============================================================================

/**
 * API for monitoring and managing the Nomic self-improvement loop.
 *
 * The Nomic loop is an autonomous cycle where agents:
 * 1. Context - Gather codebase understanding
 * 2. Debate - Propose improvements
 * 3. Design - Architecture planning
 * 4. Implement - Code generation
 * 5. Verify - Tests and checks
 *
 * @example
 * ```typescript
 * const health = await client.nomic.health();
 * if (health.status === 'stalled') {
 *   console.log(`Stalled for ${health.stall_duration_seconds}s`);
 * }
 * ```
 */
class NomicAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get current nomic loop state.
   */
  async state(): Promise<NomicState> {
    return this.http.get<NomicState>('/api/nomic/state');
  }

  /**
   * Get nomic loop health with stall detection.
   */
  async health(): Promise<NomicHealth> {
    return this.http.get<NomicHealth>('/api/nomic/health');
  }

  /**
   * Get nomic loop Prometheus metrics summary.
   */
  async metrics(): Promise<NomicMetrics> {
    return this.http.get<NomicMetrics>('/api/nomic/metrics');
  }

  /**
   * Get recent nomic loop log lines.
   * @param lines - Number of lines to retrieve (default 100, max 1000)
   */
  async logs(lines = 100): Promise<NomicLog> {
    return this.http.get<NomicLog>(`/api/nomic/log?lines=${Math.min(lines, 1000)}`);
  }

  /**
   * Get risk register entries.
   * @param limit - Maximum entries to return (default 50, max 200)
   */
  async riskRegister(limit = 50): Promise<RiskRegister> {
    return this.http.get<RiskRegister>(`/api/nomic/risk-register?limit=${Math.min(limit, 200)}`);
  }

  /**
   * Get available operational modes.
   */
  async modes(): Promise<ModesResponse> {
    return this.http.get<ModesResponse>('/api/modes');
  }
}

// =============================================================================
// Learning Analytics API
// =============================================================================

/**
 * API for cross-cycle learning analytics.
 *
 * Provides insights into patterns learned across nomic loop cycles,
 * agent evolution over time, and aggregated insights.
 *
 * @example
 * ```typescript
 * const evolution = await client.learning.agentEvolution();
 * for (const agent of evolution.agents) {
 *   console.log(`${agent.agent}: ${agent.overall_trend}`);
 * }
 * ```
 */
class LearningAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get summaries of all nomic loop cycles.
   * @param limit - Maximum cycles to return (default 20, max 100)
   */
  async cycles(limit = 20): Promise<CycleSummariesResponse> {
    return this.http.get<CycleSummariesResponse>(`/api/learning/cycles?limit=${Math.min(limit, 100)}`);
  }

  /**
   * Get learned patterns across cycles.
   */
  async patterns(): Promise<LearnedPatternsResponse> {
    return this.http.get<LearnedPatternsResponse>('/api/learning/patterns');
  }

  /**
   * Get agent performance evolution over time.
   */
  async agentEvolution(): Promise<AgentEvolutionResponse> {
    return this.http.get<AgentEvolutionResponse>('/api/learning/agent-evolution');
  }

  /**
   * Get aggregated insights from cycles.
   * @param limit - Maximum insights to return (default 50, max 200)
   */
  async insights(limit = 50): Promise<AggregatedInsightsResponse> {
    return this.http.get<AggregatedInsightsResponse>(`/api/learning/insights?limit=${Math.min(limit, 200)}`);
  }
}

// =============================================================================
// Genesis API (Evolutionary Population)
// =============================================================================

/**
 * API for evolutionary genome tracking.
 *
 * Genesis manages a population of agent "genomes" that evolve through
 * debates. Each genome represents a prompt configuration that can be
 * mutated, crossed over, and selected based on fitness.
 *
 * @example
 * ```typescript
 * const stats = await client.genesis.stats();
 * console.log(`Population: ${stats.total_genomes} genomes`);
 *
 * const top = await client.genesis.topGenomes(5);
 * for (const genome of top.genomes) {
 *   console.log(`#${genome.rank}: ${genome.name} (${genome.win_rate}%)`);
 * }
 * ```
 */
class GenesisAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get overall genesis population statistics.
   */
  async stats(): Promise<GenesisStats> {
    return this.http.get<GenesisStats>('/api/genesis/stats');
  }

  /**
   * Get genesis events with optional filtering.
   * @param options - Filter options
   */
  async events(options?: {
    limit?: number;
    event_type?: GenesisEventType;
  }): Promise<GenesisEventsResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(Math.min(options.limit, 200)));
    if (options?.event_type) params.set('event_type', options.event_type);

    const query = params.toString();
    const path = query ? `/api/genesis/events?${query}` : '/api/genesis/events';
    return this.http.get<GenesisEventsResponse>(path);
  }

  /**
   * List genomes with pagination.
   * @param options - Pagination options
   */
  async genomes(options?: {
    limit?: number;
    offset?: number;
  }): Promise<GenomesResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(Math.min(options.limit, 100)));
    if (options?.offset) params.set('offset', String(options.offset));

    const query = params.toString();
    const path = query ? `/api/genesis/genomes?${query}` : '/api/genesis/genomes';
    return this.http.get<GenomesResponse>(path);
  }

  /**
   * Get top performing genomes by fitness.
   * @param limit - Number of top genomes to return (default 10, max 50)
   */
  async topGenomes(limit = 10): Promise<TopGenomesResponse> {
    return this.http.get<TopGenomesResponse>(`/api/genesis/genomes/top?limit=${Math.min(limit, 50)}`);
  }

  /**
   * Get current population statistics.
   */
  async population(): Promise<PopulationResponse> {
    return this.http.get<PopulationResponse>('/api/genesis/population');
  }

  /**
   * Get detailed information for a specific genome.
   * @param genomeId - The genome ID
   */
  async getGenome(genomeId: string): Promise<GenomeDetails> {
    return this.http.get<GenomeDetails>(`/api/genesis/genomes/${encodeURIComponent(genomeId)}`);
  }

  /**
   * Get lineage (ancestry) for a genome.
   * @param genomeId - The genome ID
   * @param maxDepth - Maximum ancestry depth to traverse (default 5)
   */
  async lineage(genomeId: string, maxDepth = 5): Promise<LineageResponse> {
    return this.http.get<LineageResponse>(
      `/api/genesis/lineage/${encodeURIComponent(genomeId)}?max_depth=${Math.min(maxDepth, 20)}`
    );
  }

  /**
   * Get genome participation tree for a debate.
   * @param debateId - The debate ID
   */
  async debateTree(debateId: string): Promise<DebateTreeResponse> {
    return this.http.get<DebateTreeResponse>(`/api/genesis/tree/${encodeURIComponent(debateId)}`);
  }
}

// =============================================================================
// Evolution API (Prompt Optimization)
// =============================================================================

/**
 * API for prompt evolution and A/B testing.
 *
 * Evolution tracks prompt optimization experiments, managing different
 * prompt versions for agents and measuring their performance over time.
 *
 * @example
 * ```typescript
 * const summary = await client.evolution.summary();
 * console.log(`Active A/B tests: ${summary.active_ab_tests}`);
 *
 * const history = await client.evolution.agentHistory('claude');
 * for (const version of history.versions) {
 *   console.log(`v${version.version}: ${version.performance_score}`);
 * }
 * ```
 */
class EvolutionAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get discovered evolution patterns across agents.
   */
  async patterns(): Promise<EvolutionPatternsResponse> {
    return this.http.get<EvolutionPatternsResponse>('/api/evolution/patterns');
  }

  /**
   * Get evolution summary for all agents.
   */
  async summary(): Promise<EvolutionSummaryResponse> {
    return this.http.get<EvolutionSummaryResponse>('/api/evolution/summary');
  }

  /**
   * Get prompt version history for a specific agent.
   * @param agent - Agent name
   * @param limit - Maximum versions to return (default 20, max 100)
   */
  async agentHistory(agent: string, limit = 20): Promise<AgentHistoryResponse> {
    return this.http.get<AgentHistoryResponse>(
      `/api/evolution/${encodeURIComponent(agent)}/history?limit=${Math.min(limit, 100)}`
    );
  }

  /**
   * Get current prompt for a specific agent.
   * @param agent - Agent name
   */
  async agentPrompt(agent: string): Promise<AgentPromptResponse> {
    return this.http.get<AgentPromptResponse>(`/api/evolution/${encodeURIComponent(agent)}/prompt`);
  }
}

// =============================================================================
// Plugins API
// =============================================================================

class PluginsAPI {
  constructor(private http: HttpClient) {}

  /**
   * List all available plugins.
   */
  async list(): Promise<Plugin[]> {
    const response = await this.http.get<PluginListResponse>('/api/plugins');
    return response.plugins;
  }

  /**
   * Get details for a specific plugin.
   */
  async get(name: string): Promise<PluginDetails> {
    return this.http.get<PluginDetails>(`/api/plugins/${encodeURIComponent(name)}`);
  }

  /**
   * Run a plugin with provided input.
   */
  async run(name: string, request: PluginRunRequest): Promise<PluginRunResponse> {
    return this.http.post<PluginRunResponse>(`/api/plugins/${encodeURIComponent(name)}/run`, request);
  }

  /**
   * List installed plugins for the current user/org.
   */
  async listInstalled(): Promise<InstalledPlugin[]> {
    const response = await this.http.get<InstalledPluginsResponse>('/api/plugins/installed');
    return response.plugins;
  }

  /**
   * Install a plugin.
   */
  async install(name: string, request?: PluginInstallRequest): Promise<PluginInstallResponse> {
    return this.http.post<PluginInstallResponse>(`/api/plugins/${encodeURIComponent(name)}/install`, request || {});
  }

  /**
   * Uninstall a plugin.
   */
  async uninstall(name: string): Promise<{ message: string }> {
    return this.http.delete<{ message: string }>(`/api/plugins/${encodeURIComponent(name)}/install`);
  }

  /**
   * Submit a new plugin for review.
   */
  async submit(request: PluginSubmitRequest): Promise<PluginSubmitResponse> {
    return this.http.post<PluginSubmitResponse>('/api/plugins/submit', request);
  }

  /**
   * List user's plugin submissions.
   */
  async listSubmissions(): Promise<PluginSubmissionsResponse> {
    return this.http.get<PluginSubmissionsResponse>('/api/plugins/submissions');
  }

  /**
   * Get the plugin marketplace.
   */
  async marketplace(): Promise<PluginMarketplace> {
    return this.http.get<PluginMarketplace>('/api/plugins/marketplace');
  }
}

// =============================================================================
// Personas API
// =============================================================================

class PersonasAPI {
  constructor(private http: HttpClient) {}

  /**
   * List all available personas.
   */
  async list(): Promise<AgentPersona[]> {
    const response = await this.http.get<PersonaListResponse>('/api/personas');
    return response.personas;
  }

  /**
   * Get persona for a specific agent.
   */
  async getForAgent(agentName: string): Promise<AgentPersona> {
    return this.http.get<AgentPersona>(`/api/agent/${encodeURIComponent(agentName)}/persona`);
  }

  /**
   * Get grounded persona for an agent.
   */
  async getGrounded(agentName: string): Promise<GroundedPersona> {
    return this.http.get<GroundedPersona>(`/api/agent/${encodeURIComponent(agentName)}/grounded-persona`);
  }

  /**
   * Get identity prompt for an agent.
   */
  async getIdentityPrompt(agentName: string, sections?: string[]): Promise<IdentityPrompt> {
    const params = sections ? `?sections=${sections.join(',')}` : '';
    return this.http.get<IdentityPrompt>(`/api/agent/${encodeURIComponent(agentName)}/identity-prompt${params}`);
  }

  /**
   * Get performance summary for an agent.
   */
  async getPerformance(agentName: string): Promise<AgentPerformance> {
    return this.http.get<AgentPerformance>(`/api/agent/${encodeURIComponent(agentName)}/performance`);
  }

  /**
   * Get agent's best expertise domains by calibration.
   */
  async getDomains(agentName: string, limit?: number): Promise<AgentDomainsResponse> {
    const params = limit !== undefined ? `?limit=${limit}` : '';
    return this.http.get<AgentDomainsResponse>(`/api/agent/${encodeURIComponent(agentName)}/domains${params}`);
  }

  /**
   * Get position accuracy stats for an agent.
   */
  async getAccuracy(agentName: string): Promise<AgentAccuracy> {
    return this.http.get<AgentAccuracy>(`/api/agent/${encodeURIComponent(agentName)}/accuracy`);
  }
}

// =============================================================================
// Broadcast API (Podcast Generation)
// =============================================================================

/**
 * API for generating podcast-style audio/video from debates.
 *
 * The broadcast system converts debate transcripts into engaging
 * audio/video content suitable for podcasts or social media.
 *
 * @example
 * ```typescript
 * // Generate audio podcast for a debate
 * const result = await client.broadcast.generate('debate-123');
 * console.log(`Audio URL: ${result.audio_url}`);
 *
 * // Generate full broadcast with video
 * const full = await client.broadcast.generateFull('debate-123', {
 *   video: true,
 *   title: 'AI Debate: Microservices vs Monoliths',
 * });
 * ```
 */
class BroadcastAPI {
  constructor(private http: HttpClient) {}

  /**
   * Generate basic broadcast (audio only) for a debate.
   * @param debateId - Debate ID to generate broadcast from
   */
  async generate(debateId: string): Promise<BroadcastResult> {
    return this.http.post<BroadcastResult>(
      `/api/debates/${encodeURIComponent(debateId)}/broadcast`,
      {}
    );
  }

  /**
   * Generate full broadcast with all options.
   * @param debateId - Debate ID to generate broadcast from
   * @param options - Broadcast generation options
   */
  async generateFull(debateId: string, options: BroadcastOptions = {}): Promise<BroadcastResult> {
    const params = new URLSearchParams();
    if (options.video !== undefined) params.set('video', String(options.video));
    if (options.title) params.set('title', options.title);
    if (options.description) params.set('description', options.description);
    if (options.episode_number !== undefined) params.set('episode_number', String(options.episode_number));
    if (options.rss !== undefined) params.set('rss', String(options.rss));

    const queryString = params.toString();
    const url = `/api/debates/${encodeURIComponent(debateId)}/broadcast/full${queryString ? `?${queryString}` : ''}`;
    return this.http.post<BroadcastResult>(url, {});
  }

  /**
   * Get RSS podcast feed XML.
   * Returns raw XML string suitable for podcast clients.
   */
  async getRssFeed(): Promise<string> {
    return this.http.get<string>('/api/podcast/feed.xml');
  }
}

// =============================================================================
// Relationship API (Agent Social Dynamics)
// =============================================================================

/**
 * API for analyzing relationships between agents.
 *
 * Tracks how agents interact over time, including agreement rates,
 * win/loss records, and rivalry dynamics.
 *
 * @example
 * ```typescript
 * // Get relationship overview
 * const summary = await client.relationships.summary();
 * console.log(`Most active pair: ${summary.most_active_pair?.agent_a} vs ${summary.most_active_pair?.agent_b}`);
 *
 * // Get relationship graph for visualization
 * const graph = await client.relationships.graph({ minDebates: 5 });
 * console.log(`Nodes: ${graph.nodes.length}, Edges: ${graph.edges.length}`);
 *
 * // Get detailed relationship between two agents
 * const detail = await client.relationships.pairDetail('claude', 'gpt-4');
 * console.log(`Agreement rate: ${detail.agreement_rate}%`);
 * ```
 */
class RelationshipAPI {
  constructor(private http: HttpClient) {}

  /**
   * Get global relationship summary.
   */
  async summary(): Promise<RelationshipSummaryResponse> {
    return this.http.get<RelationshipSummaryResponse>('/api/relationships/summary');
  }

  /**
   * Get relationship graph for visualization.
   * @param options - Graph filtering options
   */
  async graph(options: { minDebates?: number; minScore?: number } = {}): Promise<RelationshipGraphResponse> {
    const params = new URLSearchParams();
    if (options.minDebates !== undefined) params.set('min_debates', String(options.minDebates));
    if (options.minScore !== undefined) params.set('min_score', String(options.minScore));

    const queryString = params.toString();
    return this.http.get<RelationshipGraphResponse>(
      `/api/relationships/graph${queryString ? `?${queryString}` : ''}`
    );
  }

  /**
   * Get relationship statistics.
   */
  async stats(): Promise<RelationshipStatsResponse> {
    return this.http.get<RelationshipStatsResponse>('/api/relationships/stats');
  }

  /**
   * Get detailed relationship between two specific agents.
   * @param agentA - First agent name
   * @param agentB - Second agent name
   */
  async pairDetail(agentA: string, agentB: string): Promise<PairDetailResponse> {
    return this.http.get<PairDetailResponse>(
      `/api/relationship/${encodeURIComponent(agentA)}/${encodeURIComponent(agentB)}`
    );
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
  readonly evidence: EvidenceAPI;
  readonly calibration: CalibrationAPI;
  readonly insights: InsightsAPI;
  readonly beliefNetwork: BeliefNetworkAPI;
  readonly consensus: ConsensusAPI;
  readonly admin: AdminAPI;
  readonly dashboard: DashboardAPI;
  readonly system: SystemAPI;
  readonly features: FeaturesAPI;
  readonly checkpoints: CheckpointsAPI;
  readonly webhooks: WebhooksAPI;
  readonly training: TrainingAPI;
  readonly gallery: GalleryAPI;
  readonly metrics: MetricsAPI;
  readonly routing: RoutingAPI;
  readonly selection: SelectionAPI;
  readonly plugins: PluginsAPI;
  readonly personas: PersonasAPI;
  readonly probes: ProbesAPI;
  readonly nomic: NomicAPI;
  readonly learning: LearningAPI;
  readonly genesis: GenesisAPI;
  readonly evolution: EvolutionAPI;
  readonly broadcast: BroadcastAPI;
  readonly relationships: RelationshipAPI;

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
    this.evidence = new EvidenceAPI(this.http);
    this.calibration = new CalibrationAPI(this.http);
    this.insights = new InsightsAPI(this.http);
    this.beliefNetwork = new BeliefNetworkAPI(this.http);
    this.consensus = new ConsensusAPI(this.http);
    this.admin = new AdminAPI(this.http);
    this.dashboard = new DashboardAPI(this.http);
    this.system = new SystemAPI(this.http);
    this.features = new FeaturesAPI(this.http);
    this.checkpoints = new CheckpointsAPI(this.http);
    this.webhooks = new WebhooksAPI(this.http);
    this.training = new TrainingAPI(this.http);
    this.gallery = new GalleryAPI(this.http);
    this.metrics = new MetricsAPI(this.http);
    this.routing = new RoutingAPI(this.http);
    this.selection = new SelectionAPI(this.http);
    this.plugins = new PluginsAPI(this.http);
    this.personas = new PersonasAPI(this.http);
    this.probes = new ProbesAPI(this.http);
    this.nomic = new NomicAPI(this.http);
    this.learning = new LearningAPI(this.http);
    this.genesis = new GenesisAPI(this.http);
    this.evolution = new EvolutionAPI(this.http);
    this.broadcast = new BroadcastAPI(this.http);
    this.relationships = new RelationshipAPI(this.http);
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
