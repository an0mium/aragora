/**
 * Aragora API Client
 *
 * Main client for interacting with the Aragora multi-agent debate platform.
 */

import type {
  AragoraConfig,
  Agent,
  AgentCalibration,
  AgentComparison,
  AgentConsistency,
  AgentFlip,
  AgentMoment,
  AgentNetwork,
  AgentPerformance,
  AgentPosition,
  AgentProfile,
  AgentRelationship,
  AgentScore,
  AuditEvent,
  AuditFinding,
  AuditSession,
  AuditStats,
  CodebaseAudit,
  CodebaseCallgraph,
  CodebaseDeadcode,
  CodebaseImpact,
  CodebaseMetrics,
  CodebaseScan,
  CodebaseSymbols,
  CodebaseUnderstanding,
  ConsensusQualityAnalytics,
  ConsensusStats,
  ContinuumRetrieveOptions,
  ContinuumStoreOptions,
  ContinuumStoreResult,
  Counterfactual,
  CounterfactualGeneration,
  CounterfactualList,
  CreateAuditSessionRequest,
  CreateTournamentRequest,
  CritiqueEntry,
  CveDetails,
  Debate,
  DebateCitations,
  DebateConvergence,
  DebateCreateRequest,
  DebateCreateResponse,
  DebateEvidence,
  DebateExport,
  DebateUpdateRequest,
  DecisionReceipt,
  DependencyAnalysis,
  DisagreementAnalytics,
  DomainRating,
  EarlyStopAnalytics,
  ExplainabilityResult,
  ExplanationFactors,
  GauntletComparison,
  GauntletHeatmapExtended,
  GauntletPersona,
  GauntletReceiptExport,
  GauntletResult,
  GauntletRun,
  GauntletRunRequest,
  GauntletRunResponse,
  GmailDraft,
  GmailFilter,
  GmailLabel,
  GmailMessage,
  GmailThread,
  GraphDebate,
  GraphDebateCreateRequest,
  GraphNode,
  GraphStats,
  HeadToHeadStats,
  HealthCheck,
  KnowledgeEntry,
  KnowledgeSearchResult,
  KnowledgeStats,
  LicenseCheck,
  MarketplaceTemplate,
  MatrixConclusion,
  MatrixDebate,
  MatrixDebateCreateRequest,
  MatrixScenarioResult,
  MemoryAnalytics,
  MemoryEntry,
  MemorySearchParams,
  MemoryStats,
  MemoryTier,
  MemoryTierStats,
  Message,
  Narrative,
  OnboardingStatus,
  OpponentBriefing,
  Organization,
  OrganizationInvitation,
  OrganizationMember,
  OwaspSummary,
  PaginationParams,
  Provenance,
  QuickScan,
  RankingStats,
  RelationshipGraph,
  RelationshipSummary,
  Replay,
  ReplayFormat,
  RiskHeatmap,
  RoleRotationAnalytics,
  SastFinding,
  SastScan,
  Sbom,
  ScoreAgentsRequest,
  SearchResponse,
  SecretFinding,
  SecretsScan,
  SelectionPlugin,
  SettledConsensus,
  SimilarConsensus,
  TeamSelection,
  TeamSelectionRequest,
  Tenant,
  Tournament,
  TournamentBracket,
  TournamentMatch,
  TournamentStandings,
  TrendingTopic,
  TrendingTopicsList,
  UserOrganization,
  VerificationBackend,
  VerificationReport,
  VerificationResult,
  VerificationStatus,
  VerifyClaimRequest,
  VulnerabilityScan,
  WebSocketEvent,
  Workflow,
  WorkflowApproval,
  WorkflowExecution,
  WorkflowSimulationResult,
  WorkflowTemplate,
  WorkflowTemplatePackage,
  WorkflowTemplateRunResult,
  WorkflowVersion,
} from './types';
import { AragoraError } from './types';
import { AragoraWebSocket, createWebSocket, streamDebate, type WebSocketOptions, type StreamOptions } from './websocket';
import {
  DebatesAPI,
  AgentsAPI,
  WorkflowsAPI,
  SMEAPI,
  BillingAPI,
  BudgetsAPI,
  ReceiptsAPI,
  ExplainabilityAPI,
  ControlPlaneAPI,
  GauntletAPI,
  AnalyticsAPI,
  MemoryAPI,
  RBACAPI,
  KnowledgeAPI,
  TournamentsAPI,
  AuthAPI,
  VerificationAPI,
  AuditAPI,
  TenantsAPI,
  OrganizationsAPI,
  WebhooksAPI,
  PluginsAPI,
  WorkspacesAPI,
  IntegrationsAPI,
  MarketplaceAPI,
  CodebaseAPI,
} from './namespaces';

interface RequestOptions {
  body?: unknown;
  params?: Record<string, string | number | boolean | undefined>;
  headers?: Record<string, string>;
  timeout?: number;
}

/**
 * Main Aragora API client.
 *
 * Provides both flat and namespaced APIs for interacting with Aragora:
 *
 * @example Namespaced API (recommended)
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Debates namespace
 * const { debate_id } = await client.debates.create({ task: 'Should we use microservices?' });
 * const debate = await client.debates.get(debate_id);
 *
 * // Agents namespace
 * const { agents } = await client.agents.list();
 * const performance = await client.agents.getPerformance('claude');
 *
 * // Workflows namespace
 * const { execution_id } = await client.workflows.execute('workflow-123', { input: 'value' });
 * ```
 *
 * @example Flat API (legacy, still supported)
 * ```typescript
 * const debate = await client.createDebate({ task: 'Should we use microservices?' });
 * const agents = await client.listAgents();
 * ```
 */
export class AragoraClient {
  private config: Required<Omit<AragoraConfig, 'apiKey' | 'wsUrl'>> & Pick<AragoraConfig, 'apiKey' | 'wsUrl'>;

  /**
   * Debates API namespace.
   * Provides methods for creating, managing, and analyzing debates.
   */
  readonly debates: DebatesAPI;

  /**
   * Agents API namespace.
   * Provides methods for listing agents and viewing their performance.
   */
  readonly agents: AgentsAPI;

  /**
   * Workflows API namespace.
   * Provides methods for creating and executing automated workflows.
   */
  readonly workflows: WorkflowsAPI;

  /**
   * SME API namespace.
   * Provides pre-built workflows and onboarding for small/medium enterprises.
   */
  readonly sme: SMEAPI;

  /**
   * Billing API namespace.
   * Provides methods for subscription and billing management.
   */
  readonly billing: BillingAPI;

  /**
   * Budgets API namespace.
   * Provides methods for budget management and cost control.
   */
  readonly budgets: BudgetsAPI;

  /**
   * Receipts API namespace.
   * Provides methods for decision receipt management and compliance.
   */
  readonly receipts: ReceiptsAPI;

  /**
   * Explainability API namespace.
   * Provides methods for understanding AI decisions.
   */
  readonly explainability: ExplainabilityAPI;

  /**
   * Control Plane API namespace.
   * Provides methods for agent registry, scheduling, and health monitoring.
   */
  readonly controlPlane: ControlPlaneAPI;

  /**
   * Gauntlet API namespace.
   * Provides methods for security auditing and compliance testing.
   */
  readonly gauntlet: GauntletAPI;

  /**
   * Analytics API namespace.
   * Provides methods for debate metrics and performance analytics.
   */
  readonly analytics: AnalyticsAPI;

  /**
   * Memory API namespace.
   * Provides methods for multi-tier memory management.
   */
  readonly memory: MemoryAPI;

  /**
   * RBAC API namespace.
   * Provides methods for role-based access control management.
   */
  readonly rbac: RBACAPI;

  /**
   * Knowledge API namespace.
   * Provides methods for knowledge management and retrieval.
   */
  readonly knowledge: KnowledgeAPI;

  /**
   * Tournaments API namespace.
   * Provides methods for agent tournaments and rankings.
   */
  readonly tournaments: TournamentsAPI;

  /**
   * Auth API namespace.
   * Provides methods for authentication and session management.
   */
  readonly auth: AuthAPI;

  /**
   * Verification API namespace.
   * Provides methods for formal verification of debate conclusions.
   */
  readonly verification: VerificationAPI;

  /**
   * Audit API namespace.
   * Provides methods for audit logging and compliance.
   */
  readonly audit: AuditAPI;

  /**
   * Tenants API namespace.
   * Provides methods for multi-tenant management.
   */
  readonly tenants: TenantsAPI;

  /**
   * Organizations API namespace.
   * Provides methods for organization management.
   */
  readonly organizations: OrganizationsAPI;

  /**
   * Webhooks API namespace.
   * Provides methods for webhook management and event subscriptions.
   */
  readonly webhooks: WebhooksAPI;

  /**
   * Plugins API namespace.
   * Provides methods for plugin marketplace and management.
   */
  readonly plugins: PluginsAPI;

  /**
   * Workspaces API namespace.
   * Provides methods for workspace and team management.
   */
  readonly workspaces: WorkspacesAPI;

  /**
   * Integrations API namespace.
   * Provides methods for external service integrations.
   */
  readonly integrations: IntegrationsAPI;

  /**
   * Marketplace API namespace.
   * Provides methods for template publishing, discovery, and deployment.
   */
  readonly marketplace: MarketplaceAPI;

  /**
   * Codebase Analysis API namespace.
   * Provides methods for security scanning, dependency analysis, metrics, and code intelligence.
   */
  readonly codebase: CodebaseAPI;

  constructor(config: AragoraConfig) {
    this.config = {
      baseUrl: config.baseUrl.replace(/\/+$/, ''), // Remove trailing slashes
      apiKey: config.apiKey,
      headers: config.headers ?? {},
      timeout: config.timeout ?? 30000,
      retryEnabled: config.retryEnabled ?? true,
      maxRetries: config.maxRetries ?? 3,
      wsUrl: config.wsUrl,
    };

    // Initialize namespace APIs
    this.debates = new DebatesAPI(this);
    this.agents = new AgentsAPI(this);
    this.workflows = new WorkflowsAPI(this);
    this.sme = new SMEAPI(this);
    this.billing = new BillingAPI(this);
    this.budgets = new BudgetsAPI(this);
    this.receipts = new ReceiptsAPI(this);
    this.explainability = new ExplainabilityAPI(this);
    this.controlPlane = new ControlPlaneAPI(this);
    this.gauntlet = new GauntletAPI(this);
    this.analytics = new AnalyticsAPI(this);
    this.memory = new MemoryAPI(this);
    this.rbac = new RBACAPI(this);
    this.knowledge = new KnowledgeAPI(this);
    this.tournaments = new TournamentsAPI(this);
    this.auth = new AuthAPI(this);
    this.verification = new VerificationAPI(this);
    this.audit = new AuditAPI(this);
    this.tenants = new TenantsAPI(this);
    this.organizations = new OrganizationsAPI(this);
    this.webhooks = new WebhooksAPI(this);
    this.plugins = new PluginsAPI(this);
    this.workspaces = new WorkspacesAPI(this);
    this.integrations = new IntegrationsAPI(this);
    this.marketplace = new MarketplaceAPI(this);
    this.codebase = new CodebaseAPI(this);
  }

  // ===========================================================================
  // Core Request Method
  // ===========================================================================

  public async request<T>(
    method: string,
    path: string,
    options: RequestOptions = {}
  ): Promise<T> {
    const url = new URL(path, this.config.baseUrl);

    // Add query parameters
    if (options.params) {
      Object.entries(options.params).forEach(([key, value]) => {
        if (value !== undefined) {
          url.searchParams.append(key, String(value));
        }
      });
    }

    // Build headers
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...this.config.headers,
      ...options.headers,
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    // Execute request with retry logic
    let lastError: Error | null = null;
    const maxAttempts = this.config.retryEnabled ? this.config.maxRetries : 1;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const controller = new AbortController();
        const timeout = options.timeout ?? this.config.timeout;
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        const response = await fetch(url.toString(), {
          method,
          headers,
          body: options.body ? JSON.stringify(options.body) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const body = await response.json().catch(() => ({ error: response.statusText }));
          throw AragoraError.fromResponse(response.status, body);
        }

        // Handle empty responses
        const text = await response.text();
        if (!text) {
          return {} as T;
        }

        return JSON.parse(text) as T;
      } catch (error) {
        lastError = error as Error;

        // Don't retry on client errors (4xx) or abort
        if (error instanceof AragoraError && error.status && error.status < 500) {
          throw error;
        }
        if ((error as Error).name === 'AbortError') {
          throw new AragoraError('Request timeout', 'SERVICE_UNAVAILABLE');
        }

        // Retry on server errors and network failures
        if (attempt < maxAttempts) {
          await this.sleep(Math.pow(2, attempt - 1) * 1000);
        }
      }
    }

    throw lastError ?? new AragoraError('Request failed', 'INTERNAL_ERROR');
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // ===========================================================================
  // HTTP Helper Methods (used by namespace APIs)
  // ===========================================================================

  /**
   * Perform a GET request.
   */
  get<T>(path: string): Promise<T> {
    return this.request<T>('GET', path);
  }

  /**
   * Perform a POST request.
   */
  post<T>(path: string, body?: unknown): Promise<T> {
    return this.request<T>('POST', path, { body });
  }

  /**
   * Perform a PUT request.
   */
  put<T>(path: string, body?: unknown): Promise<T> {
    return this.request<T>('PUT', path, { body });
  }

  /**
   * Perform a DELETE request.
   */
  delete<T>(path: string): Promise<T> {
    return this.request<T>('DELETE', path);
  }

  // ===========================================================================
  // WebSocket
  // ===========================================================================

  /**
   * Create a WebSocket connection for real-time debate streaming.
   */
  createWebSocket(options?: WebSocketOptions): AragoraWebSocket {
    return createWebSocket(this.config, options);
  }

  // ===========================================================================
  // Streaming Convenience Methods
  // ===========================================================================

  /**
   * Create a debate and return an async generator for streaming its events.
   *
   * This is a convenience method that combines debate creation with streaming.
   * It creates the debate via the API and returns both the debate response and
   * an async generator that yields WebSocket events for the debate.
   *
   * @param request - Debate creation parameters
   * @param streamOptions - Optional WebSocket/streaming options
   * @returns Object with debate response and stream generator
   *
   * @example
   * ```typescript
   * const { debate, stream } = await client.createDebateAndStream({
   *   task: 'Should we use microservices?',
   *   agents: ['claude', 'gpt-4'],
   *   rounds: 3
   * });
   *
   * console.log(`Started debate: ${debate.debate_id}`);
   *
   * for await (const event of stream) {
   *   if (event.type === 'agent_message') {
   *     console.log(`${event.data.agent}: ${event.data.content}`);
   *   }
   *   if (event.type === 'debate_end') {
   *     break;
   *   }
   * }
   * ```
   */
  async createDebateAndStream(
    request: DebateCreateRequest,
    streamOptions?: Omit<StreamOptions, 'debateId'>
  ): Promise<{
    debate: DebateCreateResponse;
    stream: AsyncGenerator<WebSocketEvent, void, unknown>;
  }> {
    // Create the debate first
    const debate = await this.createDebate(request);

    // Create a stream for the debate
    const stream = streamDebate(this.config, {
      ...streamOptions,
      debateId: debate.debate_id,
    });

    return { debate, stream };
  }

  /**
   * Create a stream for an existing debate.
   *
   * @param debateId - The ID of the debate to stream
   * @param options - Optional WebSocket/streaming options
   * @returns AsyncGenerator that yields debate events
   *
   * @example
   * ```typescript
   * const stream = client.streamDebate('debate-123');
   *
   * for await (const event of stream) {
   *   console.log(event.type, event.data);
   * }
   * ```
   */
  streamDebate(
    debateId: string,
    options?: Omit<StreamOptions, 'debateId'>
  ): AsyncGenerator<WebSocketEvent, void, unknown> {
    return streamDebate(this.config, { ...options, debateId });
  }

  /**
   * Create a stream for all debate events (no filter).
   *
   * @param options - Optional WebSocket/streaming options
   * @returns AsyncGenerator that yields all debate events
   *
   * @example
   * ```typescript
   * const stream = client.streamAllDebates();
   *
   * for await (const event of stream) {
   *   console.log(`[${event.debate_id}] ${event.type}`);
   * }
   * ```
   */
  streamAllDebates(options?: StreamOptions): AsyncGenerator<WebSocketEvent, void, unknown> {
    return streamDebate(this.config, options);
  }

  // ===========================================================================
  // Health
  // ===========================================================================

  async getHealth(): Promise<HealthCheck> {
    return this.request<HealthCheck>('GET', '/api/health');
  }

  async getDetailedHealth(): Promise<HealthCheck> {
    return this.request<HealthCheck>('GET', '/api/health/detailed');
  }

  // ===========================================================================
  // Agents
  // ===========================================================================

  async listAgents(): Promise<{ agents: Agent[] }> {
    return this.request<{ agents: Agent[] }>('GET', '/api/agents');
  }

  async getAgent(name: string): Promise<Agent> {
    return this.request<Agent>('GET', `/api/agents/${encodeURIComponent(name)}`);
  }

  async getAgentProfile(name: string): Promise<AgentProfile> {
    return this.request<AgentProfile>('GET', `/api/agent/${encodeURIComponent(name)}/profile`);
  }

  async getAgentHistory(name: string, params?: PaginationParams): Promise<{ matches: unknown[] }> {
    return this.request<{ matches: unknown[] }>('GET', `/api/agent/${encodeURIComponent(name)}/history`, { params });
  }

  async getAgentCalibration(name: string): Promise<AgentCalibration> {
    return this.request<AgentCalibration>('GET', `/api/v1/agent/${encodeURIComponent(name)}/calibration`);
  }

  async getAgentPerformance(name: string): Promise<AgentPerformance> {
    return this.request<AgentPerformance>('GET', `/api/v1/agent/${encodeURIComponent(name)}/performance`);
  }

  async getAgentHeadToHead(name: string, opponent: string): Promise<HeadToHeadStats> {
    return this.request<HeadToHeadStats>(
      'GET',
      `/api/v1/agent/${encodeURIComponent(name)}/head-to-head/${encodeURIComponent(opponent)}`
    );
  }

  async getAgentOpponentBriefing(name: string, opponent: string): Promise<OpponentBriefing> {
    return this.request<OpponentBriefing>(
      'GET',
      `/api/v1/agent/${encodeURIComponent(name)}/opponent-briefing/${encodeURIComponent(opponent)}`
    );
  }

  async getAgentConsistency(name: string): Promise<AgentConsistency> {
    return this.request<AgentConsistency>('GET', `/api/v1/agent/${encodeURIComponent(name)}/consistency`);
  }

  async getAgentFlips(name: string, params?: { limit?: number } & PaginationParams): Promise<{ flips: AgentFlip[] }> {
    return this.request<{ flips: AgentFlip[] }>('GET', `/api/v1/agent/${encodeURIComponent(name)}/flips`, { params });
  }

  async getAgentNetwork(name: string): Promise<AgentNetwork> {
    return this.request<AgentNetwork>('GET', `/api/v1/agent/${encodeURIComponent(name)}/network`);
  }

  async getAgentMoments(name: string, params?: { type?: string; limit?: number } & PaginationParams): Promise<{ moments: AgentMoment[] }> {
    return this.request<{ moments: AgentMoment[] }>('GET', `/api/v1/agent/${encodeURIComponent(name)}/moments`, { params });
  }

  async getAgentPositions(name: string, params?: { topic?: string; limit?: number } & PaginationParams): Promise<{ positions: AgentPosition[] }> {
    return this.request<{ positions: AgentPosition[] }>('GET', `/api/v1/agent/${encodeURIComponent(name)}/positions`, { params });
  }

  async getAgentDomains(name: string): Promise<{ domains: DomainRating[] }> {
    return this.request<{ domains: DomainRating[] }>('GET', `/api/v1/agent/${encodeURIComponent(name)}/domains`);
  }

  async getLeaderboard(): Promise<{ agents: Agent[] }> {
    return this.request<{ agents: Agent[] }>('GET', '/api/leaderboard');
  }

  async compareAgents(agents: string[]): Promise<AgentComparison> {
    return this.request<AgentComparison>('GET', '/api/agent/compare', {
      params: { agents: agents.join(',') },
    });
  }

  // ===========================================================================
  // Debates
  // ===========================================================================

  async listDebates(params?: PaginationParams & { status?: string }): Promise<{ debates: Debate[] }> {
    return this.request<{ debates: Debate[] }>('GET', '/api/debates', { params });
  }

  async getDebate(debateId: string): Promise<Debate> {
    return this.request<Debate>('GET', `/api/debates/${encodeURIComponent(debateId)}`);
  }

  async getDebateBySlug(slug: string): Promise<Debate> {
    return this.request<Debate>('GET', `/api/debates/slug/${encodeURIComponent(slug)}`);
  }

  async createDebate(request: DebateCreateRequest): Promise<DebateCreateResponse> {
    return this.request<DebateCreateResponse>('POST', '/api/debate', { body: request });
  }

  async getDebateMessages(debateId: string): Promise<{ messages: Message[] }> {
    return this.request<{ messages: Message[] }>('GET', `/api/debates/${encodeURIComponent(debateId)}/messages`);
  }

  async getDebateConvergence(debateId: string): Promise<DebateConvergence> {
    return this.request<DebateConvergence>('GET', `/api/debates/${encodeURIComponent(debateId)}/convergence`);
  }

  async getDebateCitations(debateId: string): Promise<DebateCitations> {
    return this.request<DebateCitations>('GET', `/api/debates/${encodeURIComponent(debateId)}/citations`);
  }

  async getDebateEvidence(debateId: string): Promise<DebateEvidence> {
    return this.request<DebateEvidence>('GET', `/api/debates/${encodeURIComponent(debateId)}/evidence`);
  }

  async forkDebate(debateId: string, options?: { branch_point?: number }): Promise<{ debate_id: string }> {
    return this.request<{ debate_id: string }>('POST', `/api/debates/${encodeURIComponent(debateId)}/fork`, { body: options });
  }

  async exportDebate(debateId: string, format: 'json' | 'markdown' | 'html' | 'pdf'): Promise<DebateExport> {
    return this.request<DebateExport>('GET', `/api/debates/${encodeURIComponent(debateId)}/export/${format}`);
  }

  async searchDebates(query: string, params?: PaginationParams): Promise<SearchResponse> {
    return this.request<SearchResponse>('GET', '/api/v1/search', { params: { q: query, ...params } });
  }

  async updateDebate(debateId: string, updates: DebateUpdateRequest): Promise<Debate> {
    return this.request<Debate>('PATCH', `/api/v1/debates/${encodeURIComponent(debateId)}`, { body: updates });
  }

  async getVerificationReport(debateId: string): Promise<VerificationReport> {
    return this.request<VerificationReport>('GET', `/api/v1/debates/${encodeURIComponent(debateId)}/verification-report`);
  }

  // ===========================================================================
  // Explainability
  // ===========================================================================

  async getExplanation(debateId: string, options?: {
    include_factors?: boolean;
    include_counterfactuals?: boolean;
    include_provenance?: boolean;
  }): Promise<ExplainabilityResult> {
    return this.request<ExplainabilityResult>('GET', `/api/debates/${encodeURIComponent(debateId)}/explainability`, { params: options });
  }

  async getExplanationFactors(debateId: string, options?: { min_contribution?: number }): Promise<ExplanationFactors> {
    return this.request<ExplanationFactors>('GET', `/api/debates/${encodeURIComponent(debateId)}/explainability/factors`, { params: options });
  }

  async getCounterfactuals(debateId: string, options?: { max_scenarios?: number }): Promise<CounterfactualList> {
    return this.request<CounterfactualList>('GET', `/api/debates/${encodeURIComponent(debateId)}/explainability/counterfactual`, { params: options });
  }

  async generateCounterfactual(debateId: string, body: {
    hypothesis: string;
    affected_agents?: string[];
  }): Promise<CounterfactualGeneration> {
    return this.request<CounterfactualGeneration>('POST', `/api/debates/${encodeURIComponent(debateId)}/explainability/counterfactual`, { body });
  }

  async getProvenance(debateId: string): Promise<Provenance> {
    return this.request<Provenance>('GET', `/api/debates/${encodeURIComponent(debateId)}/explainability/provenance`);
  }

  async getNarrative(debateId: string, options?: { format?: 'brief' | 'detailed' | 'executive_summary' }): Promise<Narrative> {
    return this.request<Narrative>('GET', `/api/debates/${encodeURIComponent(debateId)}/explainability/narrative`, { params: options });
  }

  // Batch explainability
  async createBatchExplanation(body: {
    debate_ids: string[];
    options?: {
      include_evidence?: boolean;
      include_counterfactuals?: boolean;
      format?: 'full' | 'summary' | 'minimal';
    };
  }): Promise<{ batch_id: string; status_url: string }> {
    return this.request<{ batch_id: string; status_url: string }>('POST', '/api/v1/explainability/batch', { body });
  }

  async getBatchExplanationStatus(batchId: string): Promise<{
    batch_id: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress_pct: number;
  }> {
    return this.request<{ batch_id: string; status: 'pending' | 'processing' | 'completed' | 'failed'; progress_pct: number }>('GET', `/api/v1/explainability/batch/${encodeURIComponent(batchId)}/status`);
  }

  async getBatchExplanationResults(batchId: string, params?: PaginationParams): Promise<{
    results: Array<{ debate_id: string; status: string; explanation?: unknown }>;
  }> {
    return this.request<{ results: Array<{ debate_id: string; status: string; explanation?: unknown }> }>('GET', `/api/v1/explainability/batch/${encodeURIComponent(batchId)}/results`, { params });
  }

  // ===========================================================================
  // Workflows
  // ===========================================================================

  async listWorkflows(params?: PaginationParams): Promise<{ workflows: Workflow[] }> {
    return this.request<{ workflows: Workflow[] }>('GET', '/api/workflows', { params });
  }

  async getWorkflow(workflowId: string): Promise<Workflow> {
    return this.request<Workflow>('GET', `/api/workflows/${encodeURIComponent(workflowId)}`);
  }

  async createWorkflow(workflow: Partial<Workflow>): Promise<Workflow> {
    return this.request<Workflow>('POST', '/api/workflows', { body: workflow });
  }

  async updateWorkflow(workflowId: string, updates: Partial<Workflow>): Promise<Workflow> {
    return this.request<Workflow>('PUT', `/api/workflows/${encodeURIComponent(workflowId)}`, { body: updates });
  }

  async deleteWorkflow(workflowId: string): Promise<void> {
    await this.request<void>('DELETE', `/api/workflows/${encodeURIComponent(workflowId)}`);
  }

  async executeWorkflow(workflowId: string, inputs?: Record<string, unknown>): Promise<{ execution_id: string }> {
    return this.request<{ execution_id: string }>('POST', `/api/workflows/${encodeURIComponent(workflowId)}/execute`, { body: { inputs } });
  }

  // Workflow templates
  async listWorkflowTemplates(params?: {
    category?: string;
    pattern?: string;
    search?: string;
    tags?: string;
  } & PaginationParams): Promise<{ templates: WorkflowTemplate[] }> {
    return this.request<{ templates: WorkflowTemplate[] }>('GET', '/api/workflow/templates', { params });
  }

  async getWorkflowTemplate(templateId: string): Promise<WorkflowTemplate> {
    return this.request<WorkflowTemplate>('GET', `/api/workflow/templates/${encodeURIComponent(templateId)}`);
  }

  async runWorkflowTemplate(templateId: string, body?: {
    inputs?: Record<string, unknown>;
    config?: { timeout?: number; async?: boolean };
  }): Promise<WorkflowTemplateRunResult> {
    return this.request<WorkflowTemplateRunResult>('POST', `/api/workflow/templates/${encodeURIComponent(templateId)}/run`, { body });
  }

  async listWorkflowCategories(): Promise<{ categories: string[] }> {
    return this.request<{ categories: string[] }>('GET', '/api/workflow/categories');
  }

  async listWorkflowPatterns(): Promise<{ patterns: string[] }> {
    return this.request<{ patterns: string[] }>('GET', '/api/workflow/patterns');
  }

  async getWorkflowTemplatePackage(templateId: string, options?: {
    include_examples?: boolean;
  }): Promise<WorkflowTemplatePackage> {
    return this.request<WorkflowTemplatePackage>(
      'GET',
      `/api/workflow/templates/${encodeURIComponent(templateId)}/package`,
      { params: options }
    );
  }

  async instantiatePattern(patternId: string, body: {
    name: string;
    description: string;
    category?: string;
    config?: Record<string, unknown>;
    agents?: string[];
  }): Promise<{ template_id: string; workflow: Workflow }> {
    return this.request<{ template_id: string; workflow: Workflow }>(
      'POST',
      `/api/workflow/patterns/${encodeURIComponent(patternId)}/instantiate`,
      { body }
    );
  }

  // ---------------------------------------------------------------------------
  // Workflow Execution Tracking
  // ---------------------------------------------------------------------------

  /**
   * List all workflow executions (for runtime dashboard).
   */
  async listWorkflowExecutions(params?: {
    workflow_id?: string;
    status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  } & PaginationParams): Promise<{ executions: WorkflowExecution[] }> {
    return this.request<{ executions: WorkflowExecution[] }>(
      'GET',
      '/api/v1/workflow-executions',
      { params }
    );
  }

  /**
   * Get workflow execution details.
   */
  async getWorkflowExecution(executionId: string): Promise<WorkflowExecution> {
    return this.request<WorkflowExecution>(
      'GET',
      `/api/v1/workflow-executions/${encodeURIComponent(executionId)}`
    );
  }

  /**
   * Get execution status for a workflow.
   */
  async getWorkflowStatus(workflowId: string): Promise<WorkflowExecution> {
    return this.request<WorkflowExecution>(
      'GET',
      `/api/v1/workflows/${encodeURIComponent(workflowId)}/status`
    );
  }

  /**
   * Get workflow version history.
   */
  async getWorkflowVersions(workflowId: string): Promise<{ versions: WorkflowVersion[] }> {
    return this.request<{ versions: WorkflowVersion[] }>(
      'GET',
      `/api/v1/workflows/${encodeURIComponent(workflowId)}/versions`
    );
  }

  /**
   * Simulate (dry-run) a workflow without executing.
   */
  async simulateWorkflow(workflowId: string, inputs?: Record<string, unknown>): Promise<WorkflowSimulationResult> {
    return this.request<WorkflowSimulationResult>(
      'POST',
      `/api/v1/workflows/${encodeURIComponent(workflowId)}/simulate`,
      { body: { inputs } }
    );
  }

  // ---------------------------------------------------------------------------
  // Workflow Approvals
  // ---------------------------------------------------------------------------

  /**
   * List pending workflow approvals.
   */
  async listWorkflowApprovals(params?: {
    workflow_id?: string;
    status?: 'pending' | 'approved' | 'rejected';
  } & PaginationParams): Promise<{ approvals: WorkflowApproval[] }> {
    return this.request<{ approvals: WorkflowApproval[] }>(
      'GET',
      '/api/v1/workflow-approvals',
      { params }
    );
  }

  /**
   * Resolve a workflow approval (approve or reject).
   */
  async resolveWorkflowApproval(approvalId: string, body: {
    approved: boolean;
    reason?: string;
  }): Promise<WorkflowApproval> {
    return this.request<WorkflowApproval>(
      'POST',
      `/api/v1/workflow-approvals/${encodeURIComponent(approvalId)}/resolve`,
      { body }
    );
  }

  // ---------------------------------------------------------------------------
  // SME Workflows
  // ---------------------------------------------------------------------------

  /**
   * List SME-specific workflow templates.
   */
  async listSMEWorkflows(params?: {
    category?: string;
    industry?: string;
  } & PaginationParams): Promise<{ workflows: WorkflowTemplate[] }> {
    return this.request<{ workflows: WorkflowTemplate[] }>(
      'GET',
      '/api/v1/sme/workflows',
      { params }
    );
  }

  /**
   * Get SME workflow template details.
   */
  async getSMEWorkflow(workflowId: string): Promise<WorkflowTemplate> {
    return this.request<WorkflowTemplate>(
      'GET',
      `/api/v1/sme/workflows/${encodeURIComponent(workflowId)}`
    );
  }

  /**
   * Execute an SME workflow template.
   */
  async executeSMEWorkflow(workflowId: string, body: {
    inputs?: Record<string, unknown>;
    context?: Record<string, unknown>;
    execute?: boolean;
    tenant_id?: string;
  }): Promise<{ execution_id: string }> {
    const payload: Record<string, unknown> = { ...(body?.inputs ?? {}) };
    if (body?.context) {
      payload.context = body.context;
    }
    if (body?.tenant_id) {
      payload.tenant_id = body.tenant_id;
    }
    if (body?.execute !== undefined) {
      payload.execute = body.execute;
    } else if (payload.execute === undefined) {
      payload.execute = true;
    }

    return this.request<{ execution_id: string }>(
      'POST',
      `/api/v1/sme/workflows/${encodeURIComponent(workflowId)}`,
      { body: payload }
    );
  }

  // ===========================================================================
  // Gauntlet
  // ===========================================================================

  async listGauntletReceipts(params?: { verdict?: string } & PaginationParams): Promise<{ receipts: DecisionReceipt[] }> {
    return this.request<{ receipts: DecisionReceipt[] }>('GET', '/api/gauntlet/receipts', { params });
  }

  async getGauntletReceipt(receiptId: string): Promise<DecisionReceipt> {
    return this.request<DecisionReceipt>('GET', `/api/gauntlet/receipts/${encodeURIComponent(receiptId)}`);
  }

  async verifyGauntletReceipt(receiptId: string): Promise<{ valid: boolean; hash: string }> {
    return this.request<{ valid: boolean; hash: string }>('GET', `/api/gauntlet/receipts/${encodeURIComponent(receiptId)}/verify`);
  }

  async exportGauntletReceipt(receiptId: string, format: 'json' | 'html' | 'markdown' | 'sarif'): Promise<GauntletReceiptExport> {
    return this.request<GauntletReceiptExport>('GET', `/api/gauntlet/receipts/${encodeURIComponent(receiptId)}/export`, { params: { format } });
  }

  async listRiskHeatmaps(params?: PaginationParams): Promise<{ heatmaps: RiskHeatmap[] }> {
    return this.request<{ heatmaps: RiskHeatmap[] }>('GET', '/api/gauntlet/heatmaps', { params });
  }

  async getRiskHeatmap(heatmapId: string): Promise<RiskHeatmap> {
    return this.request<RiskHeatmap>('GET', `/api/gauntlet/heatmaps/${encodeURIComponent(heatmapId)}`);
  }

  async getGauntlet(gauntletId: string): Promise<GauntletRun> {
    return this.request<GauntletRun>('GET', `/api/v1/gauntlet/${encodeURIComponent(gauntletId)}`);
  }

  async deleteGauntlet(gauntletId: string): Promise<{ deleted: boolean }> {
    return this.request<{ deleted: boolean }>('DELETE', `/api/v1/gauntlet/${encodeURIComponent(gauntletId)}`);
  }

  async listGauntletPersonas(params?: { category?: string; enabled?: boolean }): Promise<{ personas: GauntletPersona[] }> {
    return this.request<{ personas: GauntletPersona[] }>('GET', '/api/v1/gauntlet/personas', { params });
  }

  async listGauntletResults(params?: { gauntlet_id?: string; status?: string } & PaginationParams): Promise<{ results: GauntletResult[] }> {
    return this.request<{ results: GauntletResult[] }>('GET', '/api/v1/gauntlet/results', { params });
  }

  async getGauntletHeatmap(gauntletId: string, format?: 'json' | 'svg'): Promise<GauntletHeatmapExtended> {
    return this.request<GauntletHeatmapExtended>('GET', `/api/v1/gauntlet/${encodeURIComponent(gauntletId)}/heatmap`, { params: { format } });
  }

  async compareGauntlets(gauntletId1: string, gauntletId2: string): Promise<GauntletComparison> {
    return this.request<GauntletComparison>(
      'GET',
      `/api/v1/gauntlet/${encodeURIComponent(gauntletId1)}/compare/${encodeURIComponent(gauntletId2)}`
    );
  }

  /**
   * Run a gauntlet stress-test on input content.
   *
   * @param request - Gauntlet run configuration
   * @returns Gauntlet run response with ID for status polling
   *
   * @example
   * ```typescript
   * const run = await client.runGauntlet({
   *   input: 'AI should replace all human workers',
   *   profile: 'comprehensive',
   *   personas: ['adversarial', 'compliance'],
   * });
   *
   * // Poll for completion
   * let status = await client.getGauntlet(run.gauntlet_id);
   * while (status.status === 'running') {
   *   await new Promise(r => setTimeout(r, 2000));
   *   status = await client.getGauntlet(run.gauntlet_id);
   * }
   * ```
   */
  async runGauntlet(request: GauntletRunRequest): Promise<GauntletRunResponse> {
    return this.request<GauntletRunResponse>('POST', '/api/v1/gauntlet/run', { body: request });
  }

  /**
   * Run a gauntlet and wait for completion.
   * Convenience method that handles polling.
   *
   * @param request - Gauntlet run configuration
   * @param options - Polling options
   * @returns Completed gauntlet run
   */
  async runGauntletAndWait(
    request: GauntletRunRequest,
    options?: {
      pollIntervalMs?: number;
      timeoutMs?: number;
    }
  ): Promise<GauntletRun> {
    const response = await this.runGauntlet(request);
    const gauntletId = response.gauntlet_id;
    const pollInterval = options?.pollIntervalMs ?? 2000;
    const timeout = options?.timeoutMs ?? 300000;

    const startTime = Date.now();
    while (Date.now() - startTime < timeout) {
      const run = await this.getGauntlet(gauntletId);
      if (['completed', 'failed', 'cancelled'].includes(run.status)) {
        return run;
      }
      await this.sleep(pollInterval);
    }

    throw new AragoraError(
      `Gauntlet ${gauntletId} did not complete within ${timeout}ms`,
      'AGENT_TIMEOUT'
    );
  }

  // ===========================================================================
  // Knowledge
  // ===========================================================================

  /**
   * Search the knowledge base.
   *
   * @param query - Search query
   * @param options - Search options
   * @returns Matching knowledge entries
   */
  async searchKnowledge(query: string, options?: {
    limit?: number;
    source?: string;
    tags?: string[];
    min_score?: number;
  }): Promise<{ results: KnowledgeSearchResult[] }> {
    return this.request<{ results: KnowledgeSearchResult[] }>('GET', '/api/v1/knowledge/search', {
      params: {
        q: query,
        limit: options?.limit,
        source: options?.source,
        tags: options?.tags?.join(','),
        min_score: options?.min_score,
      },
    });
  }

  /**
   * Add an entry to the knowledge base.
   *
   * @param entry - Knowledge entry to add
   * @returns Created entry with ID
   */
  async addKnowledge(entry: KnowledgeEntry): Promise<{ id: string; created_at: string }> {
    return this.request<{ id: string; created_at: string }>('POST', '/api/v1/knowledge', { body: entry });
  }

  /**
   * Get a knowledge entry by ID.
   *
   * @param entryId - Knowledge entry ID
   * @returns Knowledge entry
   */
  async getKnowledgeEntry(entryId: string): Promise<KnowledgeEntry> {
    return this.request<KnowledgeEntry>('GET', `/api/v1/knowledge/${encodeURIComponent(entryId)}`);
  }

  /**
   * Update a knowledge entry.
   *
   * @param entryId - Knowledge entry ID
   * @param updates - Fields to update
   * @returns Updated entry
   */
  async updateKnowledge(entryId: string, updates: Partial<KnowledgeEntry>): Promise<KnowledgeEntry> {
    return this.request<KnowledgeEntry>('PATCH', `/api/v1/knowledge/${encodeURIComponent(entryId)}`, { body: updates });
  }

  /**
   * Delete a knowledge entry.
   *
   * @param entryId - Knowledge entry ID
   */
  async deleteKnowledge(entryId: string): Promise<{ deleted: boolean }> {
    return this.request<{ deleted: boolean }>('DELETE', `/api/v1/knowledge/${encodeURIComponent(entryId)}`);
  }

  /**
   * Get knowledge base statistics.
   */
  async getKnowledgeStats(): Promise<KnowledgeStats> {
    return this.request<KnowledgeStats>('GET', '/api/v1/knowledge/stats');
  }

  /**
   * Bulk import knowledge entries.
   *
   * @param entries - Entries to import
   * @returns Import results
   */
  async bulkImportKnowledge(entries: KnowledgeEntry[]): Promise<{
    imported: number;
    failed: number;
    errors?: Array<{ index: number; error: string }>;
  }> {
    return this.request<{ imported: number; failed: number; errors?: Array<{ index: number; error: string }> }>(
      'POST',
      '/api/v1/knowledge/bulk',
      { body: { entries } }
    );
  }

  // ===========================================================================
  // Knowledge Mound (Advanced Knowledge Management)
  // ===========================================================================

  /**
   * Query the Knowledge Mound with semantic search.
   *
   * @param query - Search query
   * @param options - Query options
   * @returns Query results with nodes and relationships
   */
  async queryKnowledgeMound(query: string, options?: {
    limit?: number;
    node_types?: string[];
    min_confidence?: number;
    include_relationships?: boolean;
  }): Promise<{ nodes: any[]; relationships: any[]; total: number; query_time_ms: number }> {
    return this.request<{ nodes: any[]; relationships: any[]; total: number; query_time_ms: number }>(
      'POST',
      '/api/v1/knowledge/mound/query',
      { body: { query, ...options } }
    );
  }

  /**
   * List Knowledge Mound nodes with filtering.
   *
   * @param options - Filter options
   * @returns List of nodes
   */
  async listKnowledgeMoundNodes(options?: {
    node_type?: string;
    min_confidence?: number;
    visibility?: 'private' | 'team' | 'global';
    limit?: number;
    offset?: number;
  }): Promise<{ nodes: any[]; total: number }> {
    return this.request<{ nodes: any[]; total: number }>('GET', '/api/v1/knowledge/mound/nodes', {
      params: options as Record<string, string | number | boolean | undefined>,
    });
  }

  /**
   * Create a new Knowledge Mound node.
   *
   * @param node - Node to create
   * @returns Created node
   */
  async createKnowledgeMoundNode(node: {
    content: string;
    node_type: 'fact' | 'concept' | 'claim' | 'evidence' | 'insight';
    confidence?: number;
    source?: string;
    tags?: string[];
    visibility?: 'private' | 'team' | 'global';
    metadata?: Record<string, unknown>;
  }): Promise<{ id: string; created_at: string }> {
    return this.request<{ id: string; created_at: string }>('POST', '/api/v1/knowledge/mound/nodes', { body: node });
  }

  /**
   * Get Knowledge Mound relationships.
   *
   * @param options - Filter options
   * @returns List of relationships
   */
  async listKnowledgeMoundRelationships(options?: {
    node_id?: string;
    relationship_type?: string;
    limit?: number;
  }): Promise<{ relationships: any[]; total: number }> {
    return this.request<{ relationships: any[]; total: number }>('GET', '/api/v1/knowledge/mound/relationships', {
      params: options as Record<string, string | number | boolean | undefined>,
    });
  }

  /**
   * Create a relationship between Knowledge Mound nodes.
   *
   * @param relationship - Relationship to create
   * @returns Created relationship
   */
  async createKnowledgeMoundRelationship(relationship: {
    source_id: string;
    target_id: string;
    relationship_type: 'supports' | 'contradicts' | 'elaborates' | 'derived_from' | 'related_to';
    strength?: number;
    confidence?: number;
    metadata?: Record<string, unknown>;
  }): Promise<{ id: string; created_at: string }> {
    return this.request<{ id: string; created_at: string }>(
      'POST',
      '/api/v1/knowledge/mound/relationships',
      { body: relationship }
    );
  }

  /**
   * Get Knowledge Mound statistics.
   *
   * @returns Knowledge Mound statistics
   */
  async getKnowledgeMoundStats(): Promise<any> {
    return this.request<any>('GET', '/api/v1/knowledge/mound/stats');
  }

  /**
   * Get stale knowledge items that need revalidation.
   *
   * @param options - Filter options
   * @returns Stale items
   */
  async getStaleKnowledge(options?: {
    max_age_days?: number;
    limit?: number;
  }): Promise<{ items: any[]; total: number }> {
    return this.request<{ items: any[]; total: number }>('GET', '/api/v1/knowledge/mound/stale', {
      params: options as Record<string, string | number | boolean | undefined>,
    });
  }

  /**
   * Revalidate a knowledge node.
   *
   * @param nodeId - Node ID to revalidate
   * @param validation - Validation result
   * @returns Updated node
   */
  async revalidateKnowledge(nodeId: string, validation: {
    valid: boolean;
    new_confidence?: number;
    notes?: string;
  }): Promise<{ updated: boolean }> {
    return this.request<{ updated: boolean }>(
      'POST',
      `/api/v1/knowledge/mound/revalidate/${encodeURIComponent(nodeId)}`,
      { body: validation }
    );
  }

  // ---------------------------------------------------------------------------
  // Knowledge Mound - Graph Operations
  // ---------------------------------------------------------------------------

  /**
   * Get lineage (ancestry/descendants) of a knowledge node.
   *
   * @param nodeId - Node ID
   * @param options - Traversal options
   * @returns Lineage graph
   */
  async getKnowledgeLineage(nodeId: string, options?: {
    direction?: 'ancestors' | 'descendants' | 'both';
    max_depth?: number;
  }): Promise<{ nodes: any[]; relationships: any[] }> {
    return this.request<{ nodes: any[]; relationships: any[] }>(
      'GET',
      `/api/v1/knowledge/mound/graph/${encodeURIComponent(nodeId)}/lineage`,
      { params: options as Record<string, string | number | boolean | undefined> }
    );
  }

  /**
   * Get nodes related to a specific knowledge node.
   *
   * @param nodeId - Node ID
   * @param options - Filter options
   * @returns Related nodes
   */
  async getRelatedKnowledge(nodeId: string, options?: {
    relationship_types?: string[];
    limit?: number;
  }): Promise<{ nodes: any[]; relationships: any[] }> {
    return this.request<{ nodes: any[]; relationships: any[] }>(
      'GET',
      `/api/v1/knowledge/mound/graph/${encodeURIComponent(nodeId)}/related`,
      { params: options as Record<string, string | number | boolean | undefined> }
    );
  }

  /**
   * Export knowledge graph in D3 format.
   *
   * @param options - Export options
   * @returns D3-compatible graph data
   */
  async exportKnowledgeGraphD3(options?: {
    node_types?: string[];
    max_nodes?: number;
  }): Promise<{ nodes: any[]; links: any[] }> {
    return this.request<{ nodes: any[]; links: any[] }>('GET', '/api/v1/knowledge/mound/export/d3', {
      params: options as Record<string, string | number | boolean | undefined>,
    });
  }

  /**
   * Export knowledge graph in GraphML format.
   *
   * @param options - Export options
   * @returns GraphML XML string
   */
  async exportKnowledgeGraphML(options?: {
    node_types?: string[];
  }): Promise<{ graphml: string }> {
    return this.request<{ graphml: string }>('GET', '/api/v1/knowledge/mound/export/graphml', {
      params: options as Record<string, string | number | boolean | undefined>,
    });
  }

  // ---------------------------------------------------------------------------
  // Knowledge Mound - Contradiction Detection (Phase A2)
  // ---------------------------------------------------------------------------

  /**
   * Detect contradictions in the knowledge base.
   *
   * @param options - Detection options
   * @returns Detected contradictions
   */
  async detectKnowledgeContradictions(options?: {
    scope?: 'all' | 'recent' | 'high_confidence';
    min_confidence?: number;
  }): Promise<{ contradictions: any[]; detected_at: string }> {
    return this.request<{ contradictions: any[]; detected_at: string }>(
      'POST',
      '/api/v1/knowledge/mound/contradictions/detect',
      { body: options || {} }
    );
  }

  /**
   * List existing contradictions.
   *
   * @param options - Filter options
   * @returns List of contradictions
   */
  async listKnowledgeContradictions(options?: {
    status?: 'pending' | 'resolved' | 'all';
    limit?: number;
    offset?: number;
  }): Promise<{ contradictions: any[]; total: number }> {
    return this.request<{ contradictions: any[]; total: number }>('GET', '/api/v1/knowledge/mound/contradictions', {
      params: options as Record<string, string | number | boolean | undefined>,
    });
  }

  /**
   * Resolve a knowledge contradiction.
   *
   * @param contradictionId - Contradiction ID
   * @param resolution - Resolution details
   * @returns Resolution result
   */
  async resolveKnowledgeContradiction(contradictionId: string, resolution: {
    resolution: string;
    resolution_method: 'manual' | 'automated' | 'debate';
    keep_node_id?: string;
    notes?: string;
  }): Promise<{ resolved: boolean; resolved_at: string }> {
    return this.request<{ resolved: boolean; resolved_at: string }>(
      'POST',
      `/api/v1/knowledge/mound/contradictions/${encodeURIComponent(contradictionId)}/resolve`,
      { body: resolution }
    );
  }

  /**
   * Get contradiction detection statistics.
   *
   * @returns Contradiction stats
   */
  async getKnowledgeContradictionStats(): Promise<any> {
    return this.request<any>('GET', '/api/v1/knowledge/mound/contradictions/stats');
  }

  // ---------------------------------------------------------------------------
  // Knowledge Mound - Governance (Phase A2)
  // ---------------------------------------------------------------------------

  /**
   * List Knowledge Mound governance roles.
   *
   * @returns List of roles
   */
  async listKnowledgeGovernanceRoles(): Promise<{ roles: any[] }> {
    return this.request<{ roles: any[] }>('GET', '/api/v1/knowledge/mound/governance/roles');
  }

  /**
   * Assign a governance role to a user.
   *
   * @param assignment - Role assignment details
   * @returns Assignment result
   */
  async assignKnowledgeGovernanceRole(assignment: {
    user_id: string;
    role_id: string;
    scope?: string;
  }): Promise<{ assigned: boolean }> {
    return this.request<{ assigned: boolean }>(
      'POST',
      '/api/v1/knowledge/mound/governance/roles/assign',
      { body: assignment }
    );
  }

  /**
   * Revoke a governance role from a user.
   *
   * @param revocation - Role revocation details
   * @returns Revocation result
   */
  async revokeKnowledgeGovernanceRole(revocation: {
    user_id: string;
    role_id: string;
  }): Promise<{ revoked: boolean }> {
    return this.request<{ revoked: boolean }>(
      'POST',
      '/api/v1/knowledge/mound/governance/roles/revoke',
      { body: revocation }
    );
  }

  /**
   * Check if a user has permission for a knowledge operation.
   *
   * @param check - Permission check request
   * @returns Permission check result
   */
  async checkKnowledgePermission(check: {
    user_id: string;
    permission: string;
    resource_id?: string;
  }): Promise<{ allowed: boolean; reason?: string }> {
    return this.request<{ allowed: boolean; reason?: string }>(
      'POST',
      '/api/v1/knowledge/mound/governance/permissions/check',
      { body: check }
    );
  }

  /**
   * Get Knowledge Mound audit log.
   *
   * @param options - Filter options
   * @returns Audit events
   */
  async getKnowledgeAuditLog(options?: {
    user_id?: string;
    action?: string;
    start_date?: string;
    end_date?: string;
    limit?: number;
  }): Promise<{ events: any[]; total: number }> {
    return this.request<{ events: any[]; total: number }>('GET', '/api/v1/knowledge/mound/governance/audit', {
      params: options as Record<string, string | number | boolean | undefined>,
    });
  }

  /**
   * Get governance statistics.
   *
   * @returns Governance stats
   */
  async getKnowledgeGovernanceStats(): Promise<any> {
    return this.request<any>('GET', '/api/v1/knowledge/mound/governance/stats');
  }

  // ---------------------------------------------------------------------------
  // Knowledge Mound - Analytics (Phase A2)
  // ---------------------------------------------------------------------------

  /**
   * Get knowledge coverage analytics.
   *
   * @returns Coverage metrics
   */
  async getKnowledgeCoverageAnalytics(): Promise<any> {
    return this.request<any>('GET', '/api/v1/knowledge/mound/analytics/coverage');
  }

  /**
   * Get knowledge usage analytics.
   *
   * @param options - Time range options
   * @returns Usage metrics
   */
  async getKnowledgeUsageAnalytics(options?: {
    start_date?: string;
    end_date?: string;
  }): Promise<any> {
    return this.request<any>('GET', '/api/v1/knowledge/mound/analytics/usage', {
      params: options as Record<string, string | number | boolean | undefined>,
    });
  }

  /**
   * Record a knowledge usage event.
   *
   * @param event - Usage event details
   * @returns Recording result
   */
  async recordKnowledgeUsage(event: {
    node_id: string;
    user_id?: string;
    action: 'view' | 'cite' | 'share' | 'export';
    context?: string;
  }): Promise<{ recorded: boolean }> {
    return this.request<{ recorded: boolean }>(
      'POST',
      '/api/v1/knowledge/mound/analytics/usage/record',
      { body: event }
    );
  }

  /**
   * Get a quality snapshot of the knowledge base.
   *
   * @returns Quality metrics snapshot
   */
  async getKnowledgeQualitySnapshot(): Promise<any> {
    return this.request<any>('GET', '/api/v1/knowledge/mound/analytics/quality/snapshot');
  }

  /**
   * Get quality trend over time.
   *
   * @param options - Time range options
   * @returns Quality trend data
   */
  async getKnowledgeQualityTrend(options?: {
    days?: number;
  }): Promise<{ trend: Array<{ date: string; score: number }> }> {
    return this.request<{ trend: Array<{ date: string; score: number }> }>(
      'GET',
      '/api/v1/knowledge/mound/analytics/quality/trend',
      { params: options as Record<string, string | number | boolean | undefined> }
    );
  }

  // ---------------------------------------------------------------------------
  // Knowledge Mound - Dashboard
  // ---------------------------------------------------------------------------

  /**
   * Get Knowledge Mound health status.
   *
   * @returns Health status
   */
  async getKnowledgeMoundHealth(): Promise<any> {
    return this.request<any>('GET', '/api/v1/knowledge/mound/dashboard/health');
  }

  /**
   * Get Knowledge Mound metrics.
   *
   * @returns Current metrics
   */
  async getKnowledgeMoundMetrics(): Promise<any> {
    return this.request<any>('GET', '/api/v1/knowledge/mound/dashboard/metrics');
  }

  /**
   * Get Knowledge Mound adapter status.
   *
   * @returns List of adapters and their status
   */
  async getKnowledgeMoundAdapters(): Promise<{ adapters: Array<{ name: string; status: string; last_sync?: string }> }> {
    return this.request<{ adapters: Array<{ name: string; status: string; last_sync?: string }> }>(
      'GET',
      '/api/v1/knowledge/mound/dashboard/adapters'
    );
  }

  // ---------------------------------------------------------------------------
  // Knowledge Mound - Deduplication
  // ---------------------------------------------------------------------------

  /**
   * Get duplicate knowledge clusters.
   *
   * @param options - Detection options
   * @returns Duplicate clusters
   */
  async getKnowledgeDeduplicationClusters(options?: {
    min_similarity?: number;
    limit?: number;
  }): Promise<{ clusters: any[]; total: number }> {
    return this.request<{ clusters: any[]; total: number }>('GET', '/api/v1/knowledge/mound/dedup/clusters', {
      params: options as Record<string, string | number | boolean | undefined>,
    });
  }

  /**
   * Merge duplicate knowledge nodes.
   *
   * @param merge - Merge request
   * @returns Merge result
   */
  async mergeKnowledgeDuplicates(merge: {
    cluster_id?: string;
    node_ids: string[];
    keep_node_id: string;
  }): Promise<{ merged: number; kept_id: string }> {
    return this.request<{ merged: number; kept_id: string }>(
      'POST',
      '/api/v1/knowledge/mound/dedup/merge',
      { body: merge }
    );
  }

  // ---------------------------------------------------------------------------
  // Knowledge Mound - Pruning
  // ---------------------------------------------------------------------------

  /**
   * Get items eligible for pruning.
   *
   * @param options - Filter options
   * @returns Pruning candidates
   */
  async getKnowledgePruningItems(options?: {
    reason?: 'stale' | 'low_confidence' | 'unused' | 'duplicate';
    limit?: number;
  }): Promise<{ items: any[]; total: number }> {
    return this.request<{ items: any[]; total: number }>('GET', '/api/v1/knowledge/mound/pruning/items', {
      params: options as Record<string, string | number | boolean | undefined>,
    });
  }

  /**
   * Execute pruning on selected items.
   *
   * @param pruning - Pruning request
   * @returns Pruning result
   */
  async executeKnowledgePruning(pruning: {
    node_ids?: string[];
    reason?: string;
    archive?: boolean;
  }): Promise<{ pruned: number; archived: number }> {
    return this.request<{ pruned: number; archived: number }>(
      'POST',
      '/api/v1/knowledge/mound/pruning/execute',
      { body: pruning }
    );
  }

  /**
   * Get pruning history.
   *
   * @param options - Filter options
   * @returns Pruning history
   */
  async getKnowledgePruningHistory(options?: {
    limit?: number;
  }): Promise<{ history: any[] }> {
    return this.request<{ history: any[] }>('GET', '/api/v1/knowledge/mound/pruning/history', {
      params: options as Record<string, string | number | boolean | undefined>,
    });
  }

  // ---------------------------------------------------------------------------
  // Knowledge Mound - Extraction
  // ---------------------------------------------------------------------------

  /**
   * Extract knowledge from a completed debate.
   *
   * @param debateId - Debate ID to extract from
   * @param options - Extraction options
   * @returns Extraction result
   */
  async extractKnowledgeFromDebate(debateId: string, options?: {
    min_confidence?: number;
    node_types?: string[];
  }): Promise<{ extracted_nodes: number; node_ids: string[] }> {
    return this.request<{ extracted_nodes: number; node_ids: string[] }>(
      'POST',
      '/api/v1/knowledge/mound/extraction/debate',
      { body: { debate_id: debateId, ...options } }
    );
  }

  /**
   * Promote extracted knowledge to permanent storage.
   *
   * @param nodeIds - Node IDs to promote
   * @returns Promotion result
   */
  async promoteExtractedKnowledge(nodeIds: string[]): Promise<{ promoted: number }> {
    return this.request<{ promoted: number }>(
      'POST',
      '/api/v1/knowledge/mound/extraction/promote',
      { body: { node_ids: nodeIds } }
    );
  }

  // ---------------------------------------------------------------------------
  // Knowledge Mound - Confidence Decay
  // ---------------------------------------------------------------------------

  /**
   * Get confidence decay information for a node.
   *
   * @param nodeId - Node ID
   * @returns Decay information
   */
  async getKnowledgeConfidenceDecay(nodeId: string): Promise<any> {
    return this.request<any>('GET', `/api/v1/knowledge/mound/confidence/decay`, {
      params: { node_id: nodeId },
    });
  }

  /**
   * Record a confidence event (validation, citation, etc.).
   *
   * @param event - Confidence event
   * @returns Event recording result
   */
  async recordKnowledgeConfidenceEvent(event: {
    node_id: string;
    event_type: 'validation' | 'citation' | 'contradiction' | 'update';
    confidence_delta: number;
    notes?: string;
  }): Promise<{ recorded: boolean; new_confidence: number }> {
    return this.request<{ recorded: boolean; new_confidence: number }>(
      'POST',
      '/api/v1/knowledge/mound/confidence/event',
      { body: event }
    );
  }

  /**
   * Get confidence history for a node.
   *
   * @param nodeId - Node ID
   * @param options - History options
   * @returns Confidence history
   */
  async getKnowledgeConfidenceHistory(nodeId: string, options?: {
    limit?: number;
  }): Promise<{ history: Array<{ timestamp: string; confidence: number; event_type: string }> }> {
    return this.request<{ history: Array<{ timestamp: string; confidence: number; event_type: string }> }>(
      'GET',
      '/api/v1/knowledge/mound/confidence/history',
      { params: { node_id: nodeId, ...options } as Record<string, string | number | boolean | undefined> }
    );
  }

  // ===========================================================================
  // Memory Store/Retrieve
  // ===========================================================================

  /**
   * Store a value in memory.
   *
   * @param key - Memory key
   * @param value - Value to store
   * @param options - Storage options
   */
  async storeMemory(key: string, value: unknown, options?: {
    tier?: 'fast' | 'medium' | 'slow' | 'glacial';
    importance?: number;
    tags?: string[];
    ttl_seconds?: number;
  }): Promise<{ stored: boolean; tier: string }> {
    return this.request<{ stored: boolean; tier: string }>('POST', '/api/v1/memory/store', {
      body: {
        key,
        value,
        ...options,
      },
    });
  }

  /**
   * Retrieve a value from memory by key.
   *
   * @param key - Memory key
   * @param options - Retrieval options
   * @returns Stored value or null if not found
   */
  async retrieveMemory(key: string, options?: {
    tier?: 'fast' | 'medium' | 'slow' | 'glacial';
  }): Promise<{ value: unknown; tier: string; metadata?: Record<string, unknown> } | null> {
    try {
      return await this.request<{ value: unknown; tier: string; metadata?: Record<string, unknown> }>(
        'GET',
        '/api/v1/memory/retrieve',
        { params: { key, tier: options?.tier } }
      );
    } catch (error) {
      if (error instanceof AragoraError && error.status === 404) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Delete a memory entry.
   *
   * @param key - Memory key
   * @param tier - Optional tier to delete from
   */
  async deleteMemory(key: string, tier?: 'fast' | 'medium' | 'slow' | 'glacial'): Promise<{ deleted: boolean }> {
    return this.request<{ deleted: boolean }>('DELETE', '/api/v1/memory/delete', {
      params: { key, tier },
    });
  }

  // ===========================================================================
  // Marketplace
  // ===========================================================================

  async browseMarketplace(params?: {
    category?: string;
    search?: string;
    sort_by?: 'downloads' | 'rating' | 'newest';
    min_rating?: number;
  } & PaginationParams): Promise<{ templates: MarketplaceTemplate[] }> {
    return this.request<{ templates: MarketplaceTemplate[] }>('GET', '/api/marketplace/templates', { params });
  }

  async getMarketplaceTemplate(templateId: string): Promise<MarketplaceTemplate> {
    return this.request<MarketplaceTemplate>('GET', `/api/marketplace/templates/${encodeURIComponent(templateId)}`);
  }

  async publishTemplate(body: {
    template_id: string;
    name: string;
    description: string;
    category: string;
    tags?: string[];
    documentation?: string;
  }): Promise<{ marketplace_id: string }> {
    return this.request<{ marketplace_id: string }>('POST', '/api/marketplace/templates', { body });
  }

  async importTemplate(templateId: string, workspaceId?: string): Promise<{ imported_id: string }> {
    return this.request<{ imported_id: string }>('POST', `/api/marketplace/templates/${encodeURIComponent(templateId)}/import`, {
      body: { workspace_id: workspaceId },
    });
  }

  async rateTemplate(templateId: string, rating: number): Promise<{ new_rating: number }> {
    return this.request<{ new_rating: number }>('POST', `/api/marketplace/templates/${encodeURIComponent(templateId)}/rate`, {
      body: { rating },
    });
  }

  async getFeaturedTemplates(): Promise<{ templates: MarketplaceTemplate[] }> {
    return this.request<{ templates: MarketplaceTemplate[] }>('GET', '/api/marketplace/featured');
  }

  async getTrendingTemplates(): Promise<{ templates: MarketplaceTemplate[] }> {
    return this.request<{ templates: MarketplaceTemplate[] }>('GET', '/api/marketplace/trending');
  }

  async reviewTemplate(templateId: string, body: {
    rating: number;
    title: string;
    content: string;
  }): Promise<{ review_id: string }> {
    return this.request<{ review_id: string }>(
      'POST',
      `/api/marketplace/templates/${encodeURIComponent(templateId)}/review`,
      { body }
    );
  }

  async getMarketplaceCategories(): Promise<{ categories: string[] }> {
    return this.request<{ categories: string[] }>('GET', '/api/marketplace/categories');
  }

  async getMarketplaceIndustries(): Promise<{ industries: string[] }> {
    return this.request<{ industries: string[] }>('GET', '/api/marketplace/industries');
  }

  async getNewMarketplaceReleases(params?: { limit?: number }): Promise<{ templates: MarketplaceTemplate[] }> {
    return this.request<{ templates: MarketplaceTemplate[] }>('GET', '/api/marketplace/new', { params });
  }

  async getMarketplaceReviews(templateId: string, params?: PaginationParams): Promise<{ reviews: TemplateReview[] }> {
    return this.request<{ reviews: TemplateReview[] }>(
      'GET',
      `/api/marketplace/templates/${encodeURIComponent(templateId)}/reviews`,
      { params }
    );
  }

  async purchaseTemplate(templateId: string, licenseType?: string): Promise<{ purchase_id: string; license_key?: string }> {
    return this.request<{ purchase_id: string; license_key?: string }>(
      'POST',
      `/api/marketplace/templates/${encodeURIComponent(templateId)}/purchase`,
      { body: { license_type: licenseType ?? 'standard' } }
    );
  }

  async downloadTemplate(templateId: string): Promise<{ content: Record<string, unknown>; version: string }> {
    return this.request<{ content: Record<string, unknown>; version: string }>(
      'GET',
      `/api/marketplace/templates/${encodeURIComponent(templateId)}/download`
    );
  }

  async getMyMarketplacePurchases(params?: PaginationParams): Promise<{ purchases: Array<{ purchase_id: string; template_id: string; purchased_at: string }> }> {
    return this.request<{ purchases: Array<{ purchase_id: string; template_id: string; purchased_at: string }> }>(
      'GET',
      '/api/marketplace/purchases',
      { params }
    );
  }

  async updateMarketplaceTemplate(templateId: string, body: {
    name?: string;
    description?: string;
    tags?: string[];
    documentation?: string;
    price?: number;
  }): Promise<MarketplaceTemplate> {
    return this.request<MarketplaceTemplate>(
      'PUT',
      `/api/marketplace/templates/${encodeURIComponent(templateId)}`,
      { body }
    );
  }

  async unpublishTemplate(templateId: string): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(
      'DELETE',
      `/api/marketplace/templates/${encodeURIComponent(templateId)}`
    );
  }

  // ===========================================================================
  // Memory & Consensus
  // ===========================================================================

  async getMemoryStats(): Promise<MemoryStats> {
    return this.request<MemoryStats>('GET', '/api/v1/memory/stats');
  }

  async searchMemory(params: MemorySearchParams): Promise<{ entries: MemoryEntry[] }> {
    return this.request<{ entries: MemoryEntry[] }>('GET', '/api/v1/memory/search', {
      params: {
        q: params.query,
        tiers: params.tiers?.join(','),
        agent: params.agent,
        limit: params.limit,
        min_importance: params.min_importance,
        include_expired: params.include_expired,
      }
    });
  }

  async getMemoryTiers(): Promise<{ tiers: MemoryTierStats[] }> {
    return this.request<{ tiers: MemoryTierStats[] }>('GET', '/api/v1/memory/tiers');
  }

  async getMemoryCritiques(params?: { limit?: number } & PaginationParams): Promise<{ critiques: CritiqueEntry[] }> {
    return this.request<{ critiques: CritiqueEntry[] }>('GET', '/api/v1/memory/critiques', { params });
  }

  /**
   * Store content in the continuum memory system.
   *
   * The continuum memory system organizes data across tiers based on
   * access patterns and importance, automatically promoting or demoting
   * entries over time.
   *
   * @param content - Content to store in memory
   * @param options - Storage options
   * @returns Storage confirmation with entry ID and tier
   *
   * @example
   * ```typescript
   * const result = await client.storeToContinuum('Important insight from debate', {
   *   tier: 'medium',
   *   tags: ['debate', 'insight'],
   *   metadata: { debate_id: 'deb-123' }
   * });
   * console.log(`Stored with ID: ${result.id}`);
   * ```
   */
  async storeToContinuum(content: string, options?: ContinuumStoreOptions): Promise<ContinuumStoreResult> {
    return this.request<ContinuumStoreResult>('POST', '/api/memory/continuum/store', {
      body: {
        content,
        tier: options?.tier ?? 'medium',
        tags: options?.tags,
        metadata: options?.metadata,
      },
    });
  }

  /**
   * Retrieve memories from the continuum by semantic query.
   *
   * Searches across memory tiers for entries matching the query,
   * returning relevant memories ranked by relevance and importance.
   *
   * @param query - Search query string
   * @param options - Retrieval options
   * @returns Matching memory entries
   *
   * @example
   * ```typescript
   * const { entries } = await client.retrieveFromContinuum('database optimization', {
   *   tier: 'slow',
   *   limit: 5
   * });
   * entries.forEach(e => console.log(e.content));
   * ```
   */
  async retrieveFromContinuum(query: string, options?: ContinuumRetrieveOptions): Promise<{ entries: MemoryEntry[] }> {
    return this.request<{ entries: MemoryEntry[] }>('GET', '/api/memory/continuum/retrieve', {
      params: { q: query, ...options },
    });
  }

  /**
   * Get statistics for the continuum memory system.
   *
   * Returns detailed metrics about memory usage across all tiers,
   * including entry counts, consolidation rates, and health status.
   *
   * @returns Continuum memory statistics
   *
   * @example
   * ```typescript
   * const stats = await client.getContinuumStats();
   * console.log(`Total entries: ${stats.total_entries}`);
   * console.log(`Health: ${stats.health_status}`);
   * ```
   */
  async getContinuumStats(): Promise<MemoryStats> {
    return this.request<MemoryStats>('GET', '/api/memory/continuum/stats');
  }

  /**
   * Consolidate memory across tiers.
   *
   * Triggers the memory consolidation process which promotes frequently
   * accessed entries to faster tiers and demotes infrequently accessed
   * entries to slower tiers.
   *
   * @returns Consolidation result
   */
  async consolidateMemory(): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>('POST', '/api/memory/continuum/consolidate');
  }

  async getConsensusStats(): Promise<ConsensusStats> {
    return this.request<ConsensusStats>('GET', '/api/consensus/stats');
  }

  async getSettledConsensus(params?: { domain?: string } & PaginationParams): Promise<{ settled: SettledConsensus[]; total: number }> {
    return this.request<{ settled: SettledConsensus[]; total: number }>('GET', '/api/consensus/settled', { params });
  }

  async getSimilarConsensus(topic: string): Promise<{ similar: SimilarConsensus[] }> {
    return this.request<{ similar: SimilarConsensus[] }>('GET', '/api/consensus/similar', { params: { topic } });
  }

  // ===========================================================================
  // Analytics
  // ===========================================================================

  async getDisagreementAnalytics(params?: { period?: string }): Promise<DisagreementAnalytics> {
    return this.request<DisagreementAnalytics>('GET', '/api/v1/analytics/disagreements', { params });
  }

  async getRoleRotationAnalytics(params?: { period?: string }): Promise<RoleRotationAnalytics> {
    return this.request<RoleRotationAnalytics>('GET', '/api/v1/analytics/role-rotation', { params });
  }

  async getEarlyStopAnalytics(params?: { period?: string }): Promise<EarlyStopAnalytics> {
    return this.request<EarlyStopAnalytics>('GET', '/api/v1/analytics/early-stops', { params });
  }

  async getConsensusQualityAnalytics(params?: { period?: string }): Promise<ConsensusQualityAnalytics> {
    return this.request<ConsensusQualityAnalytics>('GET', '/api/v1/analytics/consensus-quality', { params });
  }

  async getRankingStats(): Promise<RankingStats> {
    return this.request<RankingStats>('GET', '/api/v1/ranking/stats');
  }

  // ===========================================================================
  // ELO Rankings
  // ===========================================================================

  /**
   * Get ELO rankings for all agents.
   */
  async getEloRankings(params?: {
    domain?: string;
    limit?: number;
  } & PaginationParams): Promise<{ rankings: Array<{ agent: string; elo: number; rank: number }> }> {
    return this.request<{ rankings: Array<{ agent: string; elo: number; rank: number }> }>(
      'GET',
      '/api/v1/ranking/elo',
      { params }
    );
  }

  /**
   * Get ELO rating for a specific agent.
   */
  async getAgentElo(agentName: string): Promise<{ agent: string; elo: number; history: Array<{ date: string; elo: number }> }> {
    return this.request<{ agent: string; elo: number; history: Array<{ date: string; elo: number }> }>(
      'GET',
      `/api/v1/ranking/elo/${encodeURIComponent(agentName)}`
    );
  }

  /**
   * Get ELO history for an agent.
   */
  async getEloHistory(agentName: string, params?: {
    period?: '7d' | '30d' | '90d' | 'all';
  }): Promise<{ agent: string; history: Array<{ date: string; elo: number; change: number; opponent?: string }> }> {
    return this.request<{ agent: string; history: Array<{ date: string; elo: number; change: number; opponent?: string }> }>(
      'GET',
      `/api/v1/ranking/elo/${encodeURIComponent(agentName)}/history`,
      { params }
    );
  }

  // ===========================================================================
  // Control Plane
  // ===========================================================================

  /**
   * Register an agent with the control plane.
   */
  async registerAgent(body: {
    agent_id: string;
    name?: string;
    capabilities?: string[];
    metadata?: Record<string, unknown>;
  }): Promise<{ registered: boolean; agent_id: string }> {
    return this.request<{ registered: boolean; agent_id: string }>(
      'POST',
      '/api/control-plane/agents/register',
      { body }
    );
  }

  /**
   * Unregister an agent from the control plane.
   */
  async unregisterAgent(agentId: string): Promise<{ unregistered: boolean }> {
    return this.request<{ unregistered: boolean }>(
      'POST',
      `/api/control-plane/agents/${encodeURIComponent(agentId)}/unregister`
    );
  }

  /**
   * Send a heartbeat for an agent.
   */
  async sendHeartbeat(body: {
    agent_id: string;
    status?: 'idle' | 'busy' | 'offline' | 'draining';
    current_task?: string;
    metrics?: Record<string, number>;
  }): Promise<{ acknowledged: boolean }> {
    return this.request<{ acknowledged: boolean }>(
      'POST',
      '/api/control-plane/heartbeat',
      { body }
    );
  }

  /**
   * Get the status of an agent.
   */
  async getAgentStatus(agentId: string): Promise<{
    agent_id: string;
    status: string;
    last_heartbeat?: string;
    current_task?: string;
  }> {
    return this.request<{
      agent_id: string;
      status: string;
      last_heartbeat?: string;
      current_task?: string;
    }>('GET', `/api/control-plane/agents/${encodeURIComponent(agentId)}/status`);
  }

  /**
   * List all registered agents.
   */
  async listRegisteredAgents(params?: {
    status?: 'idle' | 'busy' | 'offline' | 'draining';
    capability?: string;
  } & PaginationParams): Promise<{ agents: Array<{
    agent_id: string;
    name?: string;
    status: string;
    capabilities?: string[];
    last_heartbeat?: string;
  }> }> {
    return this.request<{ agents: Array<{
      agent_id: string;
      name?: string;
      status: string;
      capabilities?: string[];
      last_heartbeat?: string;
    }> }>('GET', '/api/control-plane/agents', { params });
  }

  /**
   * Submit a task to the control plane.
   */
  async submitTask(body: {
    task_type: string;
    payload: Record<string, unknown>;
    priority?: 'low' | 'normal' | 'high' | 'critical';
    agent_hint?: string;
    timeout_seconds?: number;
    metadata?: Record<string, unknown>;
  }): Promise<{ task_id: string; status: string }> {
    return this.request<{ task_id: string; status: string }>(
      'POST',
      '/api/control-plane/tasks',
      { body }
    );
  }

  /**
   * Get the status of a task.
   */
  async getTaskStatus(taskId: string): Promise<{
    task_id: string;
    status: string;
    assigned_agent?: string;
    result?: unknown;
    error?: string;
    submitted_at: string;
    completed_at?: string;
  }> {
    return this.request<{
      task_id: string;
      status: string;
      assigned_agent?: string;
      result?: unknown;
      error?: string;
      submitted_at: string;
      completed_at?: string;
    }>('GET', `/api/control-plane/tasks/${encodeURIComponent(taskId)}`);
  }

  /**
   * List tasks with optional filtering.
   */
  async listTasks(params?: {
    status?: 'pending' | 'claimed' | 'running' | 'completed' | 'failed' | 'cancelled';
    task_type?: string;
    agent_id?: string;
  } & PaginationParams): Promise<{ tasks: Array<{
    task_id: string;
    task_type: string;
    status: string;
    priority: string;
    assigned_agent?: string;
    submitted_at: string;
  }> }> {
    return this.request<{ tasks: Array<{
      task_id: string;
      task_type: string;
      status: string;
      priority: string;
      assigned_agent?: string;
      submitted_at: string;
    }> }>('GET', '/api/control-plane/tasks', { params });
  }

  /**
   * Claim a pending task for an agent.
   */
  async claimTask(body: {
    agent_id: string;
    task_type?: string;
    capabilities?: string[];
  }): Promise<{ task_id: string; payload: Record<string, unknown> } | null> {
    return this.request<{ task_id: string; payload: Record<string, unknown> } | null>(
      'POST',
      '/api/control-plane/tasks/claim',
      { body }
    );
  }

  /**
   * Complete a task with a result.
   */
  async completeTask(taskId: string, body: {
    result: unknown;
    metrics?: Record<string, number>;
  }): Promise<{ completed: boolean }> {
    return this.request<{ completed: boolean }>(
      'POST',
      `/api/control-plane/tasks/${encodeURIComponent(taskId)}/complete`,
      { body }
    );
  }

  /**
   * Fail a task with an error.
   */
  async failTask(taskId: string, body: {
    error: string;
    retry?: boolean;
  }): Promise<{ failed: boolean }> {
    return this.request<{ failed: boolean }>(
      'POST',
      `/api/control-plane/tasks/${encodeURIComponent(taskId)}/fail`,
      { body }
    );
  }

  /**
   * Cancel a pending or running task.
   */
  async cancelTask(taskId: string): Promise<{ cancelled: boolean }> {
    return this.request<{ cancelled: boolean }>(
      'POST',
      `/api/control-plane/tasks/${encodeURIComponent(taskId)}/cancel`
    );
  }

  /**
   * Get control plane health status.
   */
  async getControlPlaneHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    agents_total: number;
    agents_active: number;
    tasks_pending: number;
    tasks_running: number;
  }> {
    return this.request<{
      status: 'healthy' | 'degraded' | 'unhealthy';
      agents_total: number;
      agents_active: number;
      tasks_pending: number;
      tasks_running: number;
    }>('GET', '/api/control-plane/health');
  }

  // ===========================================================================
  // Policies
  // ===========================================================================

  /**
   * Create a new policy.
   */
  async createPolicy(body: {
    name: string;
    description?: string;
    rules: Array<{
      condition: string;
      action: 'allow' | 'deny' | 'require_approval';
      priority?: number;
    }>;
    enabled?: boolean;
    scope?: 'global' | 'tenant' | 'user';
  }): Promise<{ policy_id: string; created: boolean }> {
    return this.request<{ policy_id: string; created: boolean }>(
      'POST',
      '/api/control-plane/policies',
      { body }
    );
  }

  /**
   * Get a policy by ID.
   */
  async getPolicy(policyId: string): Promise<{
    policy_id: string;
    name: string;
    description?: string;
    rules: Array<{
      condition: string;
      action: string;
      priority: number;
    }>;
    enabled: boolean;
    scope: string;
    created_at: string;
    updated_at: string;
  }> {
    return this.request<{
      policy_id: string;
      name: string;
      description?: string;
      rules: Array<{
        condition: string;
        action: string;
        priority: number;
      }>;
      enabled: boolean;
      scope: string;
      created_at: string;
      updated_at: string;
    }>('GET', `/api/control-plane/policies/${encodeURIComponent(policyId)}`);
  }

  /**
   * Update a policy.
   */
  async updatePolicy(policyId: string, body: {
    name?: string;
    description?: string;
    rules?: Array<{
      condition: string;
      action: 'allow' | 'deny' | 'require_approval';
      priority?: number;
    }>;
    enabled?: boolean;
  }): Promise<{ updated: boolean }> {
    return this.request<{ updated: boolean }>(
      'PATCH',
      `/api/control-plane/policies/${encodeURIComponent(policyId)}`,
      { body }
    );
  }

  /**
   * Delete a policy.
   */
  async deletePolicy(policyId: string): Promise<{ deleted: boolean }> {
    return this.request<{ deleted: boolean }>(
      'DELETE',
      `/api/control-plane/policies/${encodeURIComponent(policyId)}`
    );
  }

  /**
   * List all policies.
   */
  async listPolicies(params?: {
    scope?: 'global' | 'tenant' | 'user';
    enabled?: boolean;
  } & PaginationParams): Promise<{ policies: Array<{
    policy_id: string;
    name: string;
    enabled: boolean;
    scope: string;
    rules_count: number;
  }> }> {
    return this.request<{ policies: Array<{
      policy_id: string;
      name: string;
      enabled: boolean;
      scope: string;
      rules_count: number;
    }> }>('GET', '/api/control-plane/policies', { params });
  }

  // ===========================================================================
  // Task Scheduler
  // ===========================================================================

  /**
   * Schedule a task for future execution.
   */
  async scheduleTask(body: {
    task_type: string;
    payload: Record<string, unknown>;
    schedule_at?: string;  // ISO 8601 datetime
    cron?: string;         // Cron expression for recurring tasks
    priority?: 'low' | 'normal' | 'high' | 'critical';
    max_retries?: number;
    timeout_seconds?: number;
  }): Promise<{ schedule_id: string; next_run_at: string }> {
    return this.request<{ schedule_id: string; next_run_at: string }>(
      'POST',
      '/api/control-plane/schedule',
      { body }
    );
  }

  /**
   * Get scheduled task details.
   */
  async getScheduledTask(scheduleId: string): Promise<{
    schedule_id: string;
    task_type: string;
    status: 'active' | 'paused' | 'completed' | 'failed';
    schedule_at?: string;
    cron?: string;
    next_run_at?: string;
    last_run_at?: string;
    run_count: number;
  }> {
    return this.request<{
      schedule_id: string;
      task_type: string;
      status: 'active' | 'paused' | 'completed' | 'failed';
      schedule_at?: string;
      cron?: string;
      next_run_at?: string;
      last_run_at?: string;
      run_count: number;
    }>('GET', `/api/control-plane/schedule/${encodeURIComponent(scheduleId)}`);
  }

  /**
   * List scheduled tasks.
   */
  async listScheduledTasks(params?: {
    status?: 'active' | 'paused' | 'completed' | 'failed';
    task_type?: string;
  } & PaginationParams): Promise<{ schedules: Array<{
    schedule_id: string;
    task_type: string;
    status: string;
    next_run_at?: string;
    cron?: string;
  }> }> {
    return this.request<{ schedules: Array<{
      schedule_id: string;
      task_type: string;
      status: string;
      next_run_at?: string;
      cron?: string;
    }> }>('GET', '/api/control-plane/schedule', { params });
  }

  /**
   * Cancel a scheduled task.
   */
  async cancelScheduledTask(scheduleId: string): Promise<{ cancelled: boolean }> {
    return this.request<{ cancelled: boolean }>(
      'DELETE',
      `/api/control-plane/schedule/${encodeURIComponent(scheduleId)}`
    );
  }

  // ===========================================================================
  // Relationships
  // ===========================================================================

  async getRelationshipSummary(): Promise<RelationshipSummary> {
    return this.request<RelationshipSummary>('GET', '/api/relationships/summary');
  }

  async getRelationshipGraph(): Promise<RelationshipGraph> {
    return this.request<RelationshipGraph>('GET', '/api/relationships/graph');
  }

  async getAgentRelationship(agentA: string, agentB: string): Promise<AgentRelationship> {
    return this.request<AgentRelationship>('GET', `/api/relationship/${encodeURIComponent(agentA)}/${encodeURIComponent(agentB)}`);
  }

  // ===========================================================================
  // Pulse (Trending)
  // ===========================================================================

  async getTrendingTopics(category?: string): Promise<TrendingTopicsList> {
    return this.request<TrendingTopicsList>('GET', '/api/pulse/trending', { params: { category } });
  }

  async getSuggestedTopics(): Promise<{ topics: TrendingTopic[] }> {
    return this.request<{ topics: TrendingTopic[] }>('GET', '/api/pulse/suggest');
  }

  // ===========================================================================
  // Metrics
  // ===========================================================================

  async getMetrics(): Promise<string> {
    // Prometheus format
    return this.request<string>('GET', '/metrics');
  }

  async getMetricsHealth(): Promise<{ healthy: boolean; checks: Record<string, boolean> }> {
    return this.request<{ healthy: boolean; checks: Record<string, boolean> }>('GET', '/api/metrics/health');
  }

  async getSystemMetrics(): Promise<{ cpu: number; memory: number; uptime: number; requests_per_second: number }> {
    return this.request<{ cpu: number; memory: number; uptime: number; requests_per_second: number }>('GET', '/api/metrics/system');
  }

  // ===========================================================================
  // Codebase
  // ===========================================================================

  /**
   * Trigger a dependency vulnerability scan.
   */
  async startCodebaseScan(
    repo: string,
    body: { repo_path: string; branch?: string; commit_sha?: string; workspace_id?: string }
  ): Promise<CodebaseScan> {
    return this.request<CodebaseScan>(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan`,
      { body }
    );
  }

  /**
   * Get the latest dependency scan for a repo.
   */
  async getLatestCodebaseScan(repo: string): Promise<CodebaseScan> {
    return this.request<CodebaseScan>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/latest`
    );
  }

  /**
   * Get a dependency scan by ID.
   */
  async getCodebaseScan(repo: string, scanId: string): Promise<CodebaseScan> {
    return this.request<CodebaseScan>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/${encodeURIComponent(scanId)}`
    );
  }

  /**
   * List dependency scans for a repo.
   */
  async listCodebaseScans(
    repo: string,
    params?: { status?: string; limit?: number; offset?: number }
  ): Promise<{ scans: CodebaseScan[]; total: number }> {
    return this.request<{ scans: CodebaseScan[]; total: number }>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scans`,
      { params }
    );
  }

  /**
   * List vulnerabilities from the latest scan.
   */
  async listCodebaseVulnerabilities(
    repo: string,
    params?: { severity?: string; package?: string; ecosystem?: string; limit?: number; offset?: number }
  ): Promise<VulnerabilityScan> {
    return this.request<VulnerabilityScan>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/vulnerabilities`,
      { params }
    );
  }

  /**
   * Query package vulnerabilities by ecosystem.
   */
  async getPackageVulnerabilities(
    ecosystem: string,
    packageName: string,
    params?: { version?: string }
  ): Promise<VulnerabilityScan> {
    return this.request<VulnerabilityScan>(
      'GET',
      `/api/v1/codebase/package/${encodeURIComponent(ecosystem)}/${encodeURIComponent(packageName)}/vulnerabilities`,
      { params }
    );
  }

  /**
   * Get CVE details.
   */
  async getCveDetails(cveId: string): Promise<CveDetails> {
    return this.request<CveDetails>(
      'GET',
      `/api/v1/cve/${encodeURIComponent(cveId)}`
    );
  }

  /**
   * Analyze dependencies for a repository.
   */
  async analyzeDependencies(body: { repo_path: string; ecosystem?: string }): Promise<DependencyAnalysis> {
    return this.request<DependencyAnalysis>('POST', '/api/v1/codebase/analyze-dependencies', { body });
  }

  /**
   * Run a vulnerability scan for a repository.
   */
  async scanVulnerabilities(body: { repo_path: string; severity_threshold?: string }): Promise<VulnerabilityScan> {
    return this.request<VulnerabilityScan>('POST', '/api/v1/codebase/scan-vulnerabilities', { body });
  }

  /**
   * Check license compatibility.
   */
  async checkCodebaseLicenses(body: { repo_path: string; allowed_licenses?: string[] }): Promise<LicenseCheck> {
    return this.request<LicenseCheck>('POST', '/api/v1/codebase/check-licenses', { body });
  }

  /**
   * Generate a software bill of materials (SBOM).
   */
  async generateCodebaseSbom(body: { repo_path: string; format?: 'spdx' | 'cyclonedx' }): Promise<Sbom> {
    return this.request<Sbom>('POST', '/api/v1/codebase/sbom', { body });
  }

  /**
   * Clear dependency analysis cache.
   */
  async clearCodebaseCache(): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>('POST', '/api/v1/codebase/clear-cache');
  }

  /**
   * Trigger a secrets scan.
   */
  async startSecretsScan(repo: string, body: { repo_path: string; scan_history?: boolean }): Promise<SecretsScan> {
    return this.request<SecretsScan>(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/secrets`,
      { body }
    );
  }

  /**
   * Get the latest secrets scan.
   */
  async getLatestSecretsScan(repo: string): Promise<SecretsScan> {
    return this.request<SecretsScan>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/secrets/latest`
    );
  }

  /**
   * Get a secrets scan by ID.
   */
  async getSecretsScan(repo: string, scanId: string): Promise<SecretsScan> {
    return this.request<SecretsScan>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/secrets/${encodeURIComponent(scanId)}`
    );
  }

  /**
   * List secrets from the latest scan.
   */
  async listSecrets(repo: string): Promise<{ secrets: SecretFinding[]; total: number }> {
    return this.request<{ secrets: SecretFinding[]; total: number }>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/secrets`
    );
  }

  /**
   * List secrets scans.
   */
  async listSecretsScans(repo: string): Promise<{ scans: SecretsScan[]; total: number }> {
    return this.request<{ scans: SecretsScan[]; total: number }>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scans/secrets`
    );
  }

  /**
   * Trigger a SAST scan.
   */
  async startSastScan(repo: string, body: { repo_path: string; rule_sets?: string[]; workspace_id?: string }): Promise<SastScan> {
    return this.request<SastScan>(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/sast`,
      { body }
    );
  }

  /**
   * Get a SAST scan by ID.
   */
  async getSastScan(repo: string, scanId: string): Promise<SastScan> {
    return this.request<SastScan>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/sast/${encodeURIComponent(scanId)}`
    );
  }

  /**
   * List SAST findings.
   */
  async listSastFindings(
    repo: string,
    params?: { severity?: string; owasp_category?: string; limit?: number; offset?: number }
  ): Promise<{ findings: SastFinding[]; total: number }> {
    return this.request<{ findings: SastFinding[]; total: number }>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/sast/findings`,
      { params }
    );
  }

  /**
   * Get OWASP summary for SAST findings.
   */
  async getSastOwaspSummary(repo: string): Promise<OwaspSummary> {
    return this.request<OwaspSummary>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/sast/owasp-summary`
    );
  }

  /**
   * Run codebase metrics analysis.
   */
  async runCodebaseMetricsAnalysis(repo: string, body: { repo_path: string; include_coverage?: boolean }): Promise<CodebaseMetrics> {
    return this.request<CodebaseMetrics>(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/metrics/analyze`,
      { body }
    );
  }

  /**
   * Get latest metrics report.
   */
  async getLatestCodebaseMetrics(repo: string): Promise<CodebaseMetrics> {
    return this.request<CodebaseMetrics>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/metrics`
    );
  }

  /**
   * Get metrics report by ID.
   */
  async getCodebaseMetrics(repo: string, analysisId: string): Promise<CodebaseMetrics> {
    return this.request<CodebaseMetrics>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/metrics/${encodeURIComponent(analysisId)}`
    );
  }

  /**
   * List metrics history.
   */
  async listCodebaseMetricsHistory(
    repo: string,
    params?: { limit?: number; offset?: number }
  ): Promise<{ metrics: CodebaseMetrics[]; total: number }> {
    return this.request<{ metrics: CodebaseMetrics[]; total: number }>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/metrics/history`,
      { params }
    );
  }

  /**
   * Get hotspots for a repository.
   */
  async getCodebaseHotspots(
    repo: string,
    params?: { limit?: number; offset?: number }
  ): Promise<{ hotspots: Array<{ file_path: string; complexity: number; churn: number; risk_score: number }>; total: number }> {
    return this.request<{ hotspots: Array<{ file_path: string; complexity: number; churn: number; risk_score: number }>; total: number }>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/hotspots`,
      { params }
    );
  }

  /**
   * Get duplicate blocks for a repository.
   */
  async getCodebaseDuplicates(
    repo: string,
    params?: { limit?: number; offset?: number }
  ): Promise<{ duplicates: Array<{ files: string[]; lines: number; tokens: number }>; total: number; total_duplicated_lines: number }> {
    return this.request<{ duplicates: Array<{ files: string[]; lines: number; tokens: number }>; total: number; total_duplicated_lines: number }>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/duplicates`,
      { params }
    );
  }

  /**
   * Get metrics for a specific file.
   */
  async getCodebaseFileMetrics(
    repo: string,
    filePath: string
  ): Promise<{ file_path: string; lines: number; complexity: number; maintainability: number }> {
    return this.request<{ file_path: string; lines: number; complexity: number; maintainability: number }>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/metrics/file/${encodeURIComponent(filePath)}`
    );
  }

  /**
   * Run code intelligence analysis.
   */
  async analyzeCodebase(repo: string, body: { repo_path: string; depth?: number }): Promise<CodebaseUnderstanding> {
    return this.request<CodebaseUnderstanding>(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/analyze`,
      { body }
    );
  }

  /**
   * Get codebase symbols.
   */
  async getCodebaseSymbols(repo: string, params?: { type?: string; limit?: number; offset?: number }): Promise<CodebaseSymbols> {
    return this.request<CodebaseSymbols>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/symbols`,
      { params }
    );
  }

  /**
   * Get codebase call graph.
   */
  async getCodebaseCallgraph(repo: string, params?: { entry_point?: string; depth?: number }): Promise<CodebaseCallgraph> {
    return this.request<CodebaseCallgraph>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/callgraph`,
      { params }
    );
  }

  /**
   * Get dead code report.
   */
  async getCodebaseDeadcode(repo: string, params?: { confidence_threshold?: number; limit?: number }): Promise<CodebaseDeadcode> {
    return this.request<CodebaseDeadcode>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/deadcode`,
      { params }
    );
  }

  /**
   * Analyze impact for a change.
   */
  async analyzeCodebaseImpact(repo: string, body: { file_path: string; change_type?: string }): Promise<CodebaseImpact> {
    return this.request<CodebaseImpact>(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/impact`,
      { body }
    );
  }

  /**
   * Explain codebase components.
   */
  async understandCodebase(repo: string, body: { query?: string; include_architecture?: boolean }): Promise<CodebaseUnderstanding> {
    return this.request<CodebaseUnderstanding>(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/understand`,
      { body }
    );
  }

  /**
   * Start a codebase audit.
   */
  async startCodebaseAudit(repo: string, body: { repo_path: string; categories?: string[] }): Promise<CodebaseAudit> {
    return this.request<CodebaseAudit>(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/audit`,
      { body }
    );
  }

  /**
   * Get codebase audit results.
   */
  async getCodebaseAudit(repo: string, auditId: string): Promise<CodebaseAudit> {
    return this.request<CodebaseAudit>(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/audit/${encodeURIComponent(auditId)}`
    );
  }

  /**
   * Run a quick security scan.
   */
  async startQuickScan(body: { repo_path: string; scan_types?: string[] }): Promise<QuickScan> {
    return this.request<QuickScan>('POST', '/api/v1/codebase/quick-scan', { body });
  }

  /**
   * Get quick scan result.
   */
  async getQuickScan(scanId: string): Promise<QuickScan> {
    return this.request<QuickScan>(
      'GET',
      `/api/v1/codebase/quick-scan/${encodeURIComponent(scanId)}`
    );
  }

  /**
   * List recent quick scans.
   */
  async listQuickScans(): Promise<{ scans: QuickScan[]; total: number }> {
    return this.request<{ scans: QuickScan[]; total: number }>('GET', '/api/v1/codebase/quick-scans');
  }

  // ===========================================================================
  // Gmail
  // ===========================================================================

  /**
   * List Gmail labels.
   */
  async listGmailLabels(params?: { user_id?: string }): Promise<{ labels: GmailLabel[] }> {
    return this.request<{ labels: GmailLabel[] }>('GET', '/api/v1/gmail/labels', { params });
  }

  /**
   * Create a Gmail label.
   */
  async createGmailLabel(body: {
    name: string;
    user_id?: string;
    message_list_visibility?: string;
    label_list_visibility?: string;
  }): Promise<GmailLabel> {
    return this.request<GmailLabel>('POST', '/api/v1/gmail/labels', { body });
  }

  /**
   * Update a Gmail label.
   */
  async updateGmailLabel(
    labelId: string,
    body: {
      name?: string;
      user_id?: string;
      message_list_visibility?: string;
      label_list_visibility?: string;
    }
  ): Promise<GmailLabel> {
    return this.request<GmailLabel>(
      'PATCH',
      `/api/v1/gmail/labels/${encodeURIComponent(labelId)}`,
      { body }
    );
  }

  /**
   * Delete a Gmail label.
   */
  async deleteGmailLabel(labelId: string, params?: { user_id?: string }): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(
      'DELETE',
      `/api/v1/gmail/labels/${encodeURIComponent(labelId)}`,
      { params }
    );
  }

  /**
   * List Gmail filters.
   */
  async listGmailFilters(params?: { user_id?: string }): Promise<{ filters: GmailFilter[] }> {
    return this.request<{ filters: GmailFilter[] }>('GET', '/api/v1/gmail/filters', { params });
  }

  /**
   * Create a Gmail filter.
   */
  async createGmailFilter(body: { criteria: GmailFilter['criteria']; action: GmailFilter['action']; user_id?: string }): Promise<GmailFilter> {
    return this.request<GmailFilter>('POST', '/api/v1/gmail/filters', { body });
  }

  /**
   * Delete a Gmail filter.
   */
  async deleteGmailFilter(filterId: string, params?: { user_id?: string }): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(
      'DELETE',
      `/api/v1/gmail/filters/${encodeURIComponent(filterId)}`,
      { params }
    );
  }

  /**
   * Modify labels for a Gmail message.
   */
  async modifyGmailMessageLabels(messageId: string, body: { add_labels?: string[]; remove_labels?: string[]; user_id?: string }): Promise<GmailMessage> {
    return this.request<GmailMessage>(
      'POST',
      `/api/v1/gmail/messages/${encodeURIComponent(messageId)}/labels`,
      { body }
    );
  }

  /**
   * Mark a Gmail message read/unread.
   */
  async setGmailMessageReadState(messageId: string, body?: { read?: boolean; user_id?: string }): Promise<GmailMessage> {
    return this.request<GmailMessage>(
      'POST',
      `/api/v1/gmail/messages/${encodeURIComponent(messageId)}/read`,
      { body }
    );
  }

  /**
   * Star or unstar a Gmail message.
   */
  async setGmailMessageStarState(messageId: string, body?: { starred?: boolean; user_id?: string }): Promise<GmailMessage> {
    return this.request<GmailMessage>(
      'POST',
      `/api/v1/gmail/messages/${encodeURIComponent(messageId)}/star`,
      { body }
    );
  }

  /**
   * Archive a Gmail message.
   */
  async archiveGmailMessage(messageId: string): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(
      'POST',
      `/api/v1/gmail/messages/${encodeURIComponent(messageId)}/archive`
    );
  }

  /**
   * Trash or untrash a Gmail message.
   */
  async trashGmailMessage(messageId: string, body?: { trash?: boolean; user_id?: string }): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(
      'POST',
      `/api/v1/gmail/messages/${encodeURIComponent(messageId)}/trash`,
      { body }
    );
  }

  /**
   * Get a Gmail attachment.
   */
  async getGmailAttachment(
    messageId: string,
    attachmentId: string,
    params?: { user_id?: string }
  ): Promise<{ attachment_id: string; filename: string; mime_type: string; size: number; data: string }> {
    return this.request<{ attachment_id: string; filename: string; mime_type: string; size: number; data: string }>(
      'GET',
      `/api/v1/gmail/messages/${encodeURIComponent(messageId)}/attachments/${encodeURIComponent(attachmentId)}`,
      { params }
    );
  }

  /**
   * List Gmail threads.
   */
  async listGmailThreads(params?: {
    user_id?: string;
    q?: string;
    label_ids?: string;
    limit?: number;
    page_token?: string;
  }): Promise<{ threads: GmailThread[]; next_page_token?: string }> {
    return this.request<{ threads: GmailThread[]; next_page_token?: string }>('GET', '/api/v1/gmail/threads', { params });
  }

  /**
   * Get a Gmail thread.
   */
  async getGmailThread(threadId: string, params?: { user_id?: string }): Promise<GmailThread> {
    return this.request<GmailThread>(
      'GET',
      `/api/v1/gmail/threads/${encodeURIComponent(threadId)}`,
      { params }
    );
  }

  /**
   * Archive a Gmail thread.
   */
  async archiveGmailThread(threadId: string): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(
      'POST',
      `/api/v1/gmail/threads/${encodeURIComponent(threadId)}/archive`
    );
  }

  /**
   * Trash or untrash a Gmail thread.
   */
  async trashGmailThread(threadId: string, body?: { trash?: boolean; user_id?: string }): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(
      'POST',
      `/api/v1/gmail/threads/${encodeURIComponent(threadId)}/trash`,
      { body }
    );
  }

  /**
   * Modify labels for a Gmail thread.
   */
  async modifyGmailThreadLabels(threadId: string, body: { add_labels?: string[]; remove_labels?: string[]; user_id?: string }): Promise<GmailThread> {
    return this.request<GmailThread>(
      'POST',
      `/api/v1/gmail/threads/${encodeURIComponent(threadId)}/labels`,
      { body }
    );
  }

  /**
   * List Gmail drafts.
   */
  async listGmailDrafts(params?: { user_id?: string; limit?: number; page_token?: string }): Promise<{ drafts: GmailDraft[]; next_page_token?: string }> {
    return this.request<{ drafts: GmailDraft[]; next_page_token?: string }>('GET', '/api/v1/gmail/drafts', { params });
  }

  /**
   * Create a Gmail draft.
   */
  async createGmailDraft(body: { message: GmailDraft['message']; user_id?: string }): Promise<GmailDraft> {
    return this.request<GmailDraft>('POST', '/api/v1/gmail/drafts', { body });
  }

  /**
   * Get a Gmail draft.
   */
  async getGmailDraft(draftId: string, params?: { user_id?: string }): Promise<GmailDraft> {
    return this.request<GmailDraft>(
      'GET',
      `/api/v1/gmail/drafts/${encodeURIComponent(draftId)}`,
      { params }
    );
  }

  /**
   * Update a Gmail draft.
   */
  async updateGmailDraft(draftId: string, body: { message: Partial<GmailDraft['message']>; user_id?: string }): Promise<GmailDraft> {
    return this.request<GmailDraft>(
      'PUT',
      `/api/v1/gmail/drafts/${encodeURIComponent(draftId)}`,
      { body }
    );
  }

  /**
   * Delete a Gmail draft.
   */
  async deleteGmailDraft(draftId: string, params?: { user_id?: string }): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(
      'DELETE',
      `/api/v1/gmail/drafts/${encodeURIComponent(draftId)}`,
      { params }
    );
  }

  /**
   * Send a Gmail draft.
   */
  async sendGmailDraft(draftId: string): Promise<GmailMessage> {
    return this.request<GmailMessage>(
      'POST',
      `/api/v1/gmail/drafts/${encodeURIComponent(draftId)}/send`
    );
  }

  // ===========================================================================
  // Graph Debates
  // ===========================================================================

  /**
   * Create a new graph (branching) debate.
   */
  async createGraphDebate(request: GraphDebateCreateRequest): Promise<{ id: string }> {
    return this.request<{ id: string }>('POST', '/api/v1/graph-debates', { body: request });
  }

  /**
   * Get a graph debate by ID.
   */
  async getGraphDebate(debateId: string): Promise<GraphDebate> {
    return this.request<GraphDebate>('GET', `/api/v1/graph-debates/${encodeURIComponent(debateId)}`);
  }

  /**
   * List all branches in a graph debate.
   */
  async getGraphDebateBranches(debateId: string): Promise<{ branches: GraphDebate['branches'] }> {
    return this.request<{ branches: GraphDebate['branches'] }>(
      'GET',
      `/api/v1/graph-debates/${encodeURIComponent(debateId)}/branches`
    );
  }

  /**
   * Create a new branch in a graph debate.
   */
  async createGraphDebateBranch(debateId: string, body: {
    parent_branch_id?: string;
    divergence_point: number;
    hypothesis: string;
  }): Promise<{ branch_id: string }> {
    return this.request<{ branch_id: string }>(
      'POST',
      `/api/v1/graph-debates/${encodeURIComponent(debateId)}/branches`,
      { body }
    );
  }

  // ===========================================================================
  // Matrix Debates
  // ===========================================================================

  /**
   * Create a new matrix debate across multiple scenarios.
   */
  async createMatrixDebate(request: MatrixDebateCreateRequest): Promise<{ id: string }> {
    return this.request<{ id: string }>('POST', '/api/v1/matrix-debates', { body: request });
  }

  /**
   * Get a matrix debate by ID.
   */
  async getMatrixDebate(debateId: string): Promise<MatrixDebate> {
    return this.request<MatrixDebate>('GET', `/api/v1/matrix-debates/${encodeURIComponent(debateId)}`);
  }

  /**
   * Get conclusions for all scenarios in a matrix debate.
   */
  async getMatrixDebateConclusions(debateId: string): Promise<{ conclusions: Record<string, MatrixConclusion> }> {
    return this.request<{ conclusions: Record<string, MatrixConclusion> }>(
      'GET',
      `/api/v1/matrix-debates/${encodeURIComponent(debateId)}/conclusions`
    );
  }

  /**
   * List matrix debates with optional filtering.
   */
  async listMatrixDebates(params?: {
    status?: string;
  } & PaginationParams): Promise<{ debates: MatrixDebate[] }> {
    return this.request<{ debates: MatrixDebate[] }>('GET', '/api/v1/matrix-debates', { params });
  }

  /**
   * Get all scenario results for a matrix debate.
   */
  async getMatrixDebateScenarios(debateId: string): Promise<{ scenarios: MatrixScenarioResult[] }> {
    return this.request<{ scenarios: MatrixScenarioResult[] }>(
      'GET',
      `/api/v1/debates/matrix/${encodeURIComponent(debateId)}/scenarios`
    );
  }

  // ---------------------------------------------------------------------------
  // Graph Debate Extended APIs
  // ---------------------------------------------------------------------------

  /**
   * List all graph debates.
   */
  async listGraphDebates(params?: {
    status?: string;
  } & PaginationParams): Promise<{ debates: GraphDebate[] }> {
    return this.request<{ debates: GraphDebate[] }>('GET', '/api/v1/graph-debates', { params });
  }

  /**
   * Get all nodes in a graph debate.
   */
  async getGraphDebateNodes(debateId: string): Promise<{ nodes: GraphNode[] }> {
    return this.request<{ nodes: GraphNode[] }>(
      'GET',
      `/api/v1/debates/graph/${encodeURIComponent(debateId)}/nodes`
    );
  }

  /**
   * Get graph statistics for a debate.
   */
  async getDebateGraphStats(debateId: string): Promise<GraphStats> {
    return this.request<GraphStats>(
      'GET',
      `/api/v1/debate/${encodeURIComponent(debateId)}/graph-stats`
    );
  }

  // ---------------------------------------------------------------------------
  // Belief Network Graph APIs
  // ---------------------------------------------------------------------------

  /**
   * Get belief network graph visualization data.
   */
  async getBeliefNetworkGraph(networkId: string): Promise<{
    nodes: Array<{
      id: string;
      type: string;
      label: string;
      confidence: number;
    }>;
    edges: Array<{
      source: string;
      target: string;
      type: string;
      weight: number;
    }>;
  }> {
    return this.request<{
      nodes: Array<{ id: string; type: string; label: string; confidence: number }>;
      edges: Array<{ source: string; target: string; type: string; weight: number }>;
    }>(
      'GET',
      `/api/v1/belief-network/${encodeURIComponent(networkId)}/graph`
    );
  }

  // ===========================================================================
  // Verification (Z3/Lean)
  // ===========================================================================

  /**
   * Verify a claim using formal verification (Z3 or Lean).
   */
  async verifyClaim(request: VerifyClaimRequest): Promise<VerificationResult> {
    return this.request<VerificationResult>('POST', '/api/v1/verification/verify', { body: request });
  }

  /**
   * Get verification backend status.
   */
  async getVerificationStatus(): Promise<VerificationStatus> {
    return this.request<VerificationStatus>('GET', '/api/v1/verification/status');
  }

  /**
   * Verify a debate conclusion.
   */
  async verifyDebateConclusion(debateId: string, options?: {
    include_evidence?: boolean;
    include_counterfactuals?: boolean;
    depth?: 'shallow' | 'standard' | 'deep';
  }): Promise<VerificationResult> {
    return this.request<VerificationResult>(
      'POST',
      `/api/v1/verification/debate/${encodeURIComponent(debateId)}`,
      { body: options }
    );
  }

  // ===========================================================================
  // Agent Selection & Scoring
  // ===========================================================================

  /**
   * List available selection plugins.
   */
  async listSelectionPlugins(): Promise<{ plugins: SelectionPlugin[] }> {
    return this.request<{ plugins: SelectionPlugin[] }>('GET', '/api/v1/selection/plugins');
  }

  /**
   * Get default selection configuration.
   */
  async getSelectionDefaults(): Promise<{
    default_plugin: string;
    team_size: number;
    diversity_weight: number;
    quality_weight: number;
  }> {
    return this.request<{
      default_plugin: string;
      team_size: number;
      diversity_weight: number;
      quality_weight: number;
    }>('GET', '/api/v1/selection/defaults');
  }

  /**
   * Score agents for a specific task.
   */
  async scoreAgents(request: ScoreAgentsRequest): Promise<{ scores: AgentScore[] }> {
    return this.request<{ scores: AgentScore[] }>('POST', '/api/v1/selection/score', { body: request });
  }

  /**
   * Select an optimal team of agents for a task.
   */
  async selectTeam(request: TeamSelectionRequest): Promise<TeamSelection> {
    return this.request<TeamSelection>('POST', '/api/v1/selection/team', { body: request });
  }

  // ===========================================================================
  // Replays
  // ===========================================================================

  /**
   * List debate replays.
   */
  async listReplays(params?: {
    debate_id?: string;
  } & PaginationParams): Promise<{ replays: Replay[] }> {
    return this.request<{ replays: Replay[] }>('GET', '/api/v1/replays', { params });
  }

  /**
   * Get a specific replay.
   */
  async getReplay(replayId: string): Promise<Replay> {
    return this.request<Replay>('GET', `/api/v1/replays/${encodeURIComponent(replayId)}`);
  }

  /**
   * Export a replay in a specific format.
   */
  async exportReplay(replayId: string, format: ReplayFormat): Promise<{ content: string; filename: string }> {
    return this.request<{ content: string; filename: string }>(
      'GET',
      `/api/v1/replays/${encodeURIComponent(replayId)}/export`,
      { params: { format } }
    );
  }

  /**
   * Delete a replay.
   */
  async deleteReplay(replayId: string): Promise<{ deleted: boolean }> {
    return this.request<{ deleted: boolean }>('DELETE', `/api/v1/replays/${encodeURIComponent(replayId)}`);
  }

  /**
   * Create a replay from a debate.
   */
  async createReplay(debateId: string, options?: {
    name?: string;
    include_metadata?: boolean;
  }): Promise<{ replay_id: string }> {
    return this.request<{ replay_id: string }>(
      'POST',
      `/api/v1/replays`,
      { body: { debate_id: debateId, ...options } }
    );
  }

  // ===========================================================================
  // Memory Analytics
  // ===========================================================================

  /**
   * Get memory system analytics for a time period.
   */
  async getMemoryAnalytics(options?: {
    period_hours?: number;
    start_time?: string;
    end_time?: string;
  }): Promise<MemoryAnalytics> {
    return this.request<MemoryAnalytics>('GET', '/api/v1/memory/analytics', { params: options });
  }

  /**
   * Get statistics for a specific memory tier.
   */
  async getMemoryTierStats(tier: MemoryTier): Promise<MemoryTierStats> {
    return this.request<MemoryTierStats>('GET', `/api/v1/memory/tiers/${tier}/stats`);
  }

  /**
   * Create a manual memory snapshot.
   */
  async createMemorySnapshot(options?: {
    tiers?: MemoryTier[];
    include_entries?: boolean;
  }): Promise<{ snapshot_id: string; created_at: string }> {
    return this.request<{ snapshot_id: string; created_at: string }>(
      'POST',
      '/api/v1/memory/snapshot',
      { body: options }
    );
  }

  /**
   * Get glacial tier insights for cross-session learning.
   */
  async getGlacialInsights(options?: {
    topic?: string;
    min_importance?: number;
    limit?: number;
  }): Promise<{ insights: Array<{ content: string; importance: number; tags: string[] }> }> {
    return this.request<{ insights: Array<{ content: string; importance: number; tags: string[] }> }>(
      'GET',
      '/api/v1/memory/glacial/insights',
      { params: options }
    );
  }

  // ===========================================================================
  // Authentication
  // ===========================================================================

  /**
   * Register a new user account.
   */
  async registerUser(body: import('./types').RegisterRequest): Promise<import('./types').RegisterResponse> {
    return this.request<import('./types').RegisterResponse>('POST', '/api/v1/auth/register', { body });
  }

  /**
   * Login with email and password.
   */
  async login(body: import('./types').LoginRequest): Promise<import('./types').AuthToken> {
    return this.request<import('./types').AuthToken>('POST', '/api/v1/auth/login', { body });
  }

  /**
   * Refresh an access token using a refresh token.
   */
  async refreshToken(body: import('./types').RefreshRequest): Promise<import('./types').AuthToken> {
    return this.request<import('./types').AuthToken>('POST', '/api/v1/auth/refresh', { body });
  }

  /**
   * Logout and invalidate the current session.
   */
  async logout(): Promise<void> {
    await this.request<void>('POST', '/api/v1/auth/logout');
  }

  /**
   * Verify email address with a verification token.
   */
  async verifyEmail(body: import('./types').VerifyEmailRequest): Promise<import('./types').VerifyResponse> {
    return this.request<import('./types').VerifyResponse>('POST', '/api/v1/auth/verify-email', { body });
  }

  /**
   * Get the current authenticated user's profile.
   */
  async getCurrentUser(): Promise<import('./types').User> {
    return this.request<import('./types').User>('GET', '/api/v1/auth/me');
  }

  /**
   * Update the current user's profile.
   */
  async updateProfile(body: import('./types').UpdateProfileRequest): Promise<import('./types').UpdateProfileResponse> {
    return this.request<import('./types').UpdateProfileResponse>('PATCH', '/api/v1/auth/me', { body });
  }

  /**
   * Change the current user's password.
   */
  async changePassword(body: import('./types').ChangePasswordRequest): Promise<void> {
    await this.request<void>('POST', '/api/v1/auth/change-password', { body });
  }

  /**
   * Request a password reset email.
   */
  async requestPasswordReset(body: import('./types').ForgotPasswordRequest): Promise<void> {
    await this.request<void>('POST', '/api/v1/auth/forgot-password', { body });
  }

  /**
   * Reset password using a reset token.
   */
  async resetPassword(body: import('./types').ResetPasswordRequest): Promise<void> {
    await this.request<void>('POST', '/api/v1/auth/reset-password', { body });
  }

  /**
   * Get an OAuth authorization URL for a provider.
   */
  async getOAuthUrl(params: import('./types').OAuthUrlParams): Promise<import('./types').OAuthUrl> {
    return this.request<import('./types').OAuthUrl>('GET', '/api/v1/auth/oauth/authorize', {
      params: {
        provider: params.provider,
        redirect_uri: params.redirect_uri,
        state: params.state,
        scope: params.scope,
      },
    });
  }

  /**
   * Complete OAuth flow with authorization code.
   */
  async completeOAuth(body: import('./types').OAuthCallbackRequest): Promise<import('./types').AuthToken> {
    return this.request<import('./types').AuthToken>('POST', '/api/v1/auth/oauth/callback', { body });
  }

  /**
   * Setup multi-factor authentication.
   */
  async setupMFA(body: import('./types').MFASetupRequest): Promise<import('./types').MFASetupResponse> {
    return this.request<import('./types').MFASetupResponse>('POST', '/api/v1/auth/mfa/setup', { body });
  }

  /**
   * Verify MFA setup with a code.
   */
  async verifyMFASetup(body: import('./types').MFAVerifyRequest): Promise<import('./types').MFAVerifyResponse> {
    return this.request<import('./types').MFAVerifyResponse>('POST', '/api/v1/auth/mfa/verify', { body });
  }

  /**
   * Disable multi-factor authentication.
   */
  async disableMFA(): Promise<void> {
    await this.request<void>('DELETE', '/api/v1/auth/mfa');
  }

  // ===========================================================================
  // Tenancy
  // ===========================================================================

  /**
   * List all tenants the current user has access to.
   */
  async listTenants(params?: PaginationParams): Promise<import('./types').TenantList> {
    return this.request<import('./types').TenantList>('GET', '/api/v1/tenants', { params });
  }

  /**
   * Get a tenant by ID.
   */
  async getTenant(tenantId: string): Promise<import('./types').Tenant> {
    return this.request<import('./types').Tenant>('GET', `/api/v1/tenants/${encodeURIComponent(tenantId)}`);
  }

  /**
   * Create a new tenant.
   */
  async createTenant(body: import('./types').CreateTenantRequest): Promise<import('./types').Tenant> {
    return this.request<import('./types').Tenant>('POST', '/api/v1/tenants', { body });
  }

  /**
   * Update a tenant.
   */
  async updateTenant(tenantId: string, body: import('./types').UpdateTenantRequest): Promise<import('./types').Tenant> {
    return this.request<import('./types').Tenant>('PATCH', `/api/v1/tenants/${encodeURIComponent(tenantId)}`, { body });
  }

  /**
   * Delete a tenant.
   */
  async deleteTenant(tenantId: string): Promise<void> {
    await this.request<void>('DELETE', `/api/v1/tenants/${encodeURIComponent(tenantId)}`);
  }

  /**
   * Get quota status for a tenant.
   */
  async getTenantQuotas(tenantId: string): Promise<import('./types').QuotaStatus> {
    return this.request<import('./types').QuotaStatus>('GET', `/api/v1/tenants/${encodeURIComponent(tenantId)}/quotas`);
  }

  /**
   * Update quota limits for a tenant.
   */
  async updateTenantQuotas(tenantId: string, body: import('./types').QuotaUpdate): Promise<import('./types').QuotaStatus> {
    return this.request<import('./types').QuotaStatus>('PATCH', `/api/v1/tenants/${encodeURIComponent(tenantId)}/quotas`, { body });
  }

  /**
   * List members of a tenant.
   */
  async listTenantMembers(tenantId: string, params?: PaginationParams): Promise<import('./types').MemberList> {
    return this.request<import('./types').MemberList>('GET', `/api/v1/tenants/${encodeURIComponent(tenantId)}/members`, { params });
  }

  /**
   * Add a member to a tenant.
   */
  async addTenantMember(tenantId: string, body: import('./types').AddMemberRequest): Promise<import('./types').TenantMember> {
    return this.request<import('./types').TenantMember>('POST', `/api/v1/tenants/${encodeURIComponent(tenantId)}/members`, { body });
  }

  /**
   * Remove a member from a tenant.
   */
  async removeTenantMember(tenantId: string, userId: string): Promise<void> {
    await this.request<void>('DELETE', `/api/v1/tenants/${encodeURIComponent(tenantId)}/members/${encodeURIComponent(userId)}`);
  }

  // ===========================================================================
  // Organizations
  // ===========================================================================

  /**
   * Get organization details.
   */
  async getOrganization(orgId: string): Promise<Organization> {
    return this.request<Organization>('GET', `/api/v1/org/${encodeURIComponent(orgId)}`);
  }

  /**
   * Update organization settings.
   */
  async updateOrganization(orgId: string, body: {
    name?: string;
    settings?: Record<string, unknown>;
  }): Promise<Organization> {
    return this.request<Organization>('PUT', `/api/v1/org/${encodeURIComponent(orgId)}`, { body });
  }

  /**
   * List organization members.
   */
  async listOrganizationMembers(orgId: string, params?: PaginationParams): Promise<{ members: OrganizationMember[] }> {
    return this.request<{ members: OrganizationMember[] }>(
      'GET',
      `/api/v1/org/${encodeURIComponent(orgId)}/members`,
      { params }
    );
  }

  /**
   * Invite a user to an organization.
   */
  async inviteToOrganization(orgId: string, body: {
    email: string;
    role: 'admin' | 'member';
    message?: string;
  }): Promise<OrganizationInvitation> {
    return this.request<OrganizationInvitation>(
      'POST',
      `/api/v1/org/${encodeURIComponent(orgId)}/invite`,
      { body }
    );
  }

  /**
   * List pending invitations for an organization.
   */
  async listOrganizationInvitations(orgId: string, params?: PaginationParams): Promise<{ invitations: OrganizationInvitation[] }> {
    return this.request<{ invitations: OrganizationInvitation[] }>(
      'GET',
      `/api/v1/org/${encodeURIComponent(orgId)}/invitations`,
      { params }
    );
  }

  /**
   * Revoke an invitation.
   */
  async revokeOrganizationInvitation(orgId: string, invitationId: string): Promise<void> {
    await this.request<void>(
      'DELETE',
      `/api/v1/org/${encodeURIComponent(orgId)}/invitations/${encodeURIComponent(invitationId)}`
    );
  }

  /**
   * Remove a member from an organization.
   */
  async removeOrganizationMember(orgId: string, userId: string): Promise<void> {
    await this.request<void>(
      'DELETE',
      `/api/v1/org/${encodeURIComponent(orgId)}/members/${encodeURIComponent(userId)}`
    );
  }

  /**
   * Update a member's role in an organization.
   */
  async updateOrganizationMemberRole(orgId: string, userId: string, role: 'admin' | 'member'): Promise<OrganizationMember> {
    return this.request<OrganizationMember>(
      'PUT',
      `/api/v1/org/${encodeURIComponent(orgId)}/members/${encodeURIComponent(userId)}/role`,
      { body: { role } }
    );
  }

  // ---------------------------------------------------------------------------
  // User Organization Management
  // ---------------------------------------------------------------------------

  /**
   * List organizations for the current user.
   */
  async listUserOrganizations(): Promise<{ organizations: UserOrganization[] }> {
    return this.request<{ organizations: UserOrganization[] }>('GET', '/api/v1/user/organizations');
  }

  /**
   * Switch active organization for the current user.
   */
  async switchOrganization(orgId: string): Promise<{ organization_id: string }> {
    return this.request<{ organization_id: string }>(
      'POST',
      '/api/v1/user/organizations/switch',
      { body: { organization_id: orgId } }
    );
  }

  /**
   * Set default organization for the current user.
   */
  async setDefaultOrganization(orgId: string): Promise<{ organization_id: string }> {
    return this.request<{ organization_id: string }>(
      'POST',
      '/api/v1/user/organizations/default',
      { body: { organization_id: orgId } }
    );
  }

  /**
   * Leave an organization.
   */
  async leaveOrganization(orgId: string): Promise<void> {
    await this.request<void>('DELETE', `/api/v1/user/organizations/${encodeURIComponent(orgId)}`);
  }

  // ---------------------------------------------------------------------------
  // User Invitation Management
  // ---------------------------------------------------------------------------

  /**
   * List pending invitations for the current user.
   */
  async listPendingInvitations(): Promise<{ invitations: OrganizationInvitation[] }> {
    return this.request<{ invitations: OrganizationInvitation[] }>('GET', '/api/v1/invitations/pending');
  }

  /**
   * Accept an invitation.
   */
  async acceptInvitation(token: string): Promise<{ organization_id: string; role: string }> {
    return this.request<{ organization_id: string; role: string }>(
      'POST',
      `/api/v1/invitations/${encodeURIComponent(token)}/accept`
    );
  }

  // ===========================================================================
  // RBAC (Role-Based Access Control)
  // ===========================================================================

  /**
   * List all roles.
   */
  async listRoles(params?: PaginationParams): Promise<import('./types').RoleList> {
    return this.request<import('./types').RoleList>('GET', '/api/v1/rbac/roles', { params });
  }

  /**
   * Get a role by ID.
   */
  async getRole(roleId: string): Promise<import('./types').Role> {
    return this.request<import('./types').Role>('GET', `/api/v1/rbac/roles/${encodeURIComponent(roleId)}`);
  }

  /**
   * Create a new role.
   */
  async createRole(body: import('./types').CreateRoleRequest): Promise<import('./types').Role> {
    return this.request<import('./types').Role>('POST', '/api/v1/rbac/roles', { body });
  }

  /**
   * Update a role.
   */
  async updateRole(roleId: string, body: import('./types').UpdateRoleRequest): Promise<import('./types').Role> {
    return this.request<import('./types').Role>('PATCH', `/api/v1/rbac/roles/${encodeURIComponent(roleId)}`, { body });
  }

  /**
   * Delete a role.
   */
  async deleteRole(roleId: string): Promise<{ deleted: boolean }> {
    return this.request<{ deleted: boolean }>('DELETE', `/api/v1/rbac/roles/${encodeURIComponent(roleId)}`);
  }

  /**
   * List all available permissions.
   */
  async listPermissions(): Promise<import('./types').PermissionList> {
    return this.request<import('./types').PermissionList>('GET', '/api/v1/rbac/permissions');
  }

  /**
   * Assign a role to a user.
   */
  async assignRole(userId: string, roleId: string, tenantId?: string): Promise<void> {
    await this.request<void>('POST', '/api/v1/rbac/assignments', {
      body: { user_id: userId, role_id: roleId, tenant_id: tenantId },
    });
  }

  /**
   * Revoke a role from a user.
   */
  async revokeRole(userId: string, roleId: string, tenantId?: string): Promise<void> {
    await this.request<void>('DELETE', '/api/v1/rbac/assignments', {
      body: { user_id: userId, role_id: roleId, tenant_id: tenantId },
    });
  }

  /**
   * Get all roles assigned to a user.
   */
  async getUserRoles(userId: string): Promise<import('./types').RoleList> {
    return this.request<import('./types').RoleList>('GET', `/api/v1/rbac/users/${encodeURIComponent(userId)}/roles`);
  }

  /**
   * Check if a user has a specific permission.
   */
  async checkPermission(userId: string, permission: string, resource?: string): Promise<import('./types').PermissionCheck> {
    return this.request<import('./types').PermissionCheck>('GET', '/api/v1/rbac/check', {
      params: { user_id: userId, permission, resource },
    });
  }

  /**
   * List all users assigned to a role.
   */
  async listRoleAssignments(roleId: string, params?: PaginationParams): Promise<import('./types').AssignmentList> {
    return this.request<import('./types').AssignmentList>('GET', `/api/v1/rbac/roles/${encodeURIComponent(roleId)}/assignments`, { params });
  }

  /**
   * Bulk assign roles to multiple users.
   */
  async bulkAssignRoles(body: import('./types').BulkAssignRequest): Promise<import('./types').BulkAssignResponse> {
    return this.request<import('./types').BulkAssignResponse>('POST', '/api/v1/rbac/assignments/bulk', { body });
  }

  // ===========================================================================
  // Tournaments
  // ===========================================================================

  /**
   * List tournaments.
   */
  async listTournaments(params?: { status?: string } & PaginationParams): Promise<{ tournaments: Tournament[] }> {
    return this.request<{ tournaments: Tournament[] }>('GET', '/api/tournaments', { params });
  }

  /**
   * Get a tournament by ID.
   */
  async getTournament(tournamentId: string): Promise<Tournament> {
    return this.request<Tournament>('GET', `/api/tournaments/${encodeURIComponent(tournamentId)}`);
  }

  /**
   * Create a new tournament.
   */
  async createTournament(request: CreateTournamentRequest): Promise<Tournament> {
    return this.request<Tournament>('POST', '/api/tournaments', { body: request });
  }

  /**
   * Get tournament standings.
   */
  async getTournamentStandings(tournamentId: string): Promise<TournamentStandings> {
    return this.request<TournamentStandings>('GET', `/api/tournaments/${encodeURIComponent(tournamentId)}/standings`);
  }

  /**
   * Get tournament bracket.
   */
  async getTournamentBracket(tournamentId: string): Promise<TournamentBracket> {
    return this.request<TournamentBracket>('GET', `/api/tournaments/${encodeURIComponent(tournamentId)}/bracket`);
  }

  /**
   * List tournament matches.
   */
  async listTournamentMatches(tournamentId: string, params?: { round?: number; status?: string }): Promise<{ matches: TournamentMatch[] }> {
    return this.request<{ matches: TournamentMatch[] }>(
      'GET',
      `/api/tournaments/${encodeURIComponent(tournamentId)}/matches`,
      { params }
    );
  }

  /**
   * Submit match result.
   */
  async submitMatchResult(tournamentId: string, matchId: string, result: {
    winner: string;
    loser: string;
    score?: { winner: number; loser: number };
    notes?: string;
  }): Promise<TournamentMatch> {
    return this.request<TournamentMatch>(
      'POST',
      `/api/tournaments/${encodeURIComponent(tournamentId)}/matches/${encodeURIComponent(matchId)}/result`,
      { body: result }
    );
  }

  /**
   * Advance tournament to next round.
   */
  async advanceTournament(tournamentId: string): Promise<{ advanced: boolean; next_round: number }> {
    return this.request<{ advanced: boolean; next_round: number }>(
      'POST',
      `/api/tournaments/${encodeURIComponent(tournamentId)}/advance`
    );
  }

  // ===========================================================================
  // Audit
  // ===========================================================================

  /**
   * List audit events.
   */
  async listAuditEvents(params?: {
    start_date?: string;
    end_date?: string;
    actor_id?: string;
    resource_type?: string;
    action?: string;
  } & PaginationParams): Promise<{ events: AuditEvent[]; total: number }> {
    return this.request<{ events: AuditEvent[]; total: number }>('GET', '/api/v1/audit/events', { params });
  }

  /**
   * Get audit statistics.
   */
  async getAuditStats(params?: { period?: string }): Promise<AuditStats> {
    return this.request<AuditStats>('GET', '/api/v1/audit/stats', { params });
  }

  /**
   * Export audit logs.
   */
  async exportAuditLogs(request: {
    start_date: string;
    end_date: string;
    format: 'json' | 'csv' | 'pdf';
    filters?: Record<string, string>;
  }): Promise<{ url: string; expires_at: string }> {
    return this.request<{ url: string; expires_at: string }>('POST', '/api/v1/audit/export', { body: request });
  }

  /**
   * Verify audit log integrity.
   */
  async verifyAuditIntegrity(params?: { start_date?: string; end_date?: string }): Promise<{
    verified: boolean;
    entries_checked: number;
    tampered_entries: number;
  }> {
    return this.request<{ verified: boolean; entries_checked: number; tampered_entries: number }>(
      'GET',
      '/api/v1/audit/verify',
      { params }
    );
  }

  /**
   * List audit sessions.
   */
  async listAuditSessions(params?: { status?: string } & PaginationParams): Promise<{ sessions: AuditSession[]; total: number }> {
    return this.request<{ sessions: AuditSession[]; total: number }>('GET', '/api/v1/audit/sessions', { params });
  }

  /**
   * Get an audit session.
   */
  async getAuditSession(sessionId: string): Promise<AuditSession> {
    return this.request<AuditSession>('GET', `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}`);
  }

  /**
   * Create an audit session.
   */
  async createAuditSession(request: CreateAuditSessionRequest): Promise<AuditSession> {
    return this.request<AuditSession>('POST', '/api/v1/audit/sessions', { body: request });
  }

  /**
   * Start an audit session.
   */
  async startAuditSession(sessionId: string): Promise<{ started: boolean }> {
    return this.request<{ started: boolean }>('POST', `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/start`);
  }

  /**
   * Pause an audit session.
   */
  async pauseAuditSession(sessionId: string): Promise<{ paused: boolean }> {
    return this.request<{ paused: boolean }>('POST', `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/pause`);
  }

  /**
   * Resume an audit session.
   */
  async resumeAuditSession(sessionId: string): Promise<{ resumed: boolean }> {
    return this.request<{ resumed: boolean }>('POST', `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/resume`);
  }

  /**
   * Cancel an audit session.
   */
  async cancelAuditSession(sessionId: string): Promise<{ cancelled: boolean }> {
    return this.request<{ cancelled: boolean }>('POST', `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/cancel`);
  }

  /**
   * Get audit session findings.
   */
  async getAuditSessionFindings(sessionId: string, params?: PaginationParams): Promise<{ findings: AuditFinding[] }> {
    return this.request<{ findings: AuditFinding[] }>(
      'GET',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/findings`,
      { params }
    );
  }

  /**
   * Generate audit session report.
   */
  async generateAuditReport(sessionId: string, format?: 'json' | 'pdf' | 'markdown'): Promise<{ report_url: string }> {
    return this.request<{ report_url: string }>(
      'GET',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/report`,
      { params: { format } }
    );
  }

  // ===========================================================================
  // Extended Auth (Sessions, API Keys, SSO)
  // ===========================================================================

  /**
   * Logout from all sessions.
   */
  async logoutAll(): Promise<{ logged_out: boolean; sessions_revoked: number }> {
    return this.request<{ logged_out: boolean; sessions_revoked: number }>('POST', '/api/v1/auth/logout-all');
  }

  /**
   * Resend email verification.
   */
  async resendVerification(email: string): Promise<{ sent: boolean }> {
    return this.request<{ sent: boolean }>('POST', '/api/v1/auth/resend-verification', { body: { email } });
  }

  /**
   * List active sessions.
   */
  async listSessions(): Promise<{ sessions: Array<{
    id: string;
    device: string;
    ip_address: string;
    created_at: string;
    last_active: string;
    current: boolean;
  }> }> {
    return this.request<{ sessions: Array<{
      id: string;
      device: string;
      ip_address: string;
      created_at: string;
      last_active: string;
      current: boolean;
    }> }>('GET', '/api/v1/auth/sessions');
  }

  /**
   * Revoke a session.
   */
  async revokeSession(sessionId: string): Promise<{ revoked: boolean }> {
    return this.request<{ revoked: boolean }>('DELETE', `/api/v1/auth/sessions/${encodeURIComponent(sessionId)}`);
  }

  /**
   * List API keys.
   */
  async listApiKeys(): Promise<{ keys: Array<{ id: string; name: string; prefix: string; created_at: string; last_used?: string }> }> {
    return this.request<{ keys: Array<{ id: string; name: string; prefix: string; created_at: string; last_used?: string }> }>(
      'GET',
      '/api/v1/auth/api-keys'
    );
  }

  /**
   * Create a new API key.
   */
  async createApiKey(name: string, expiresIn?: number): Promise<{ id: string; key: string; prefix: string }> {
    return this.request<{ id: string; key: string; prefix: string }>('POST', '/api/v1/auth/api-keys', {
      body: { name, expires_in: expiresIn },
    });
  }

  /**
   * Revoke an API key.
   */
  async revokeApiKey(keyId: string): Promise<{ revoked: boolean }> {
    return this.request<{ revoked: boolean }>('DELETE', `/api/v1/auth/api-keys/${encodeURIComponent(keyId)}`);
  }

  /**
   * List available OAuth providers.
   */
  async listOAuthProviders(): Promise<{ providers: Array<{ type: string; name: string; enabled: boolean; auth_url: string }> }> {
    return this.request<{ providers: Array<{ type: string; name: string; enabled: boolean; auth_url: string }> }>(
      'GET',
      '/api/v1/auth/oauth/providers'
    );
  }

  /**
   * Link OAuth provider to account.
   */
  async linkOAuthProvider(provider: string, code: string): Promise<{ linked: boolean }> {
    return this.request<{ linked: boolean }>('POST', '/api/v1/auth/oauth/link', {
      body: { provider, code },
    });
  }

  /**
   * Unlink OAuth provider from account.
   */
  async unlinkOAuthProvider(provider: string): Promise<{ unlinked: boolean }> {
    return this.request<{ unlinked: boolean }>('DELETE', '/api/v1/auth/oauth/unlink', {
      params: { provider },
    });
  }

  /**
   * Initiate SSO login.
   */
  async initiateSSOLogin(provider?: string, redirectUrl?: string): Promise<{
    authorization_url: string;
    state: string;
    provider: string;
    expires_in: number;
  }> {
    return this.request<{
      authorization_url: string;
      state: string;
      provider: string;
      expires_in: number;
    }>('GET', '/api/v1/auth/sso/login', {
      params: { provider, redirect_url: redirectUrl },
    });
  }

  /**
   * List available SSO providers.
   */
  async listSSOProviders(): Promise<{
    providers: Array<{ type: string; name: string; enabled: boolean }>;
    sso_enabled: boolean;
  }> {
    return this.request<{
      providers: Array<{ type: string; name: string; enabled: boolean }>;
      sso_enabled: boolean;
    }>('GET', '/api/v1/auth/sso/providers');
  }

  /**
   * Enable MFA after setup.
   */
  async enableMFA(code: string): Promise<{ enabled: boolean }> {
    return this.request<{ enabled: boolean }>('POST', '/api/v1/auth/mfa/enable', { body: { code } });
  }

  /**
   * Generate new backup codes.
   */
  async generateBackupCodes(): Promise<{ codes: string[] }> {
    return this.request<{ codes: string[] }>('POST', '/api/v1/auth/mfa/backup-codes');
  }

  // ===========================================================================
  // Onboarding
  // ===========================================================================

  /**
   * Get onboarding status.
   */
  async getOnboardingStatus(): Promise<OnboardingStatus> {
    return this.request<OnboardingStatus>('GET', '/api/v1/onboarding/status');
  }

  /**
   * Complete onboarding.
   */
  async completeOnboarding(request?: { first_debate_id?: string; template_used?: string }): Promise<{
    completed: boolean;
    organization_id: string;
    completed_at: string;
  }> {
    return this.request<{ completed: boolean; organization_id: string; completed_at: string }>(
      'POST',
      '/api/v1/onboarding/complete',
      { body: request }
    );
  }

  /**
   * Setup organization after signup.
   */
  async setupOrganization(request: {
    name: string;
    slug?: string;
    plan?: string;
    billing_email?: string;
  }): Promise<{ organization: Tenant }> {
    return this.request<{ organization: Tenant }>('POST', '/api/v1/auth/setup-organization', { body: request });
  }

  /**
   * Send team invitation.
   */
  async inviteTeamMember(request: {
    email: string;
    organization_id: string;
    role?: string;
  }): Promise<{ invite_token: string; invite_url: string; expires_in: number }> {
    return this.request<{ invite_token: string; invite_url: string; expires_in: number }>(
      'POST',
      '/api/v1/auth/invite',
      { body: request }
    );
  }

  /**
   * Check invitation validity.
   */
  async checkInvite(token: string): Promise<{
    valid: boolean;
    email: string;
    organization_id: string;
    role: string;
    expires_at: number;
  }> {
    return this.request<{
      valid: boolean;
      email: string;
      organization_id: string;
      role: string;
      expires_at: number;
    }>('GET', '/api/v1/auth/check-invite', { params: { token } });
  }

  /**
   * Accept team invitation.
   */
  async acceptInvite(token: string): Promise<{ organization_id: string; role: string }> {
    return this.request<{ organization_id: string; role: string }>('POST', '/api/v1/auth/accept-invite', {
      body: { token },
    });
  }

  // ===========================================================================
  // Billing
  // ===========================================================================

  /**
   * List available subscription plans.
   */
  async listBillingPlans(): Promise<import('./types').BillingPlanList> {
    return this.request<import('./types').BillingPlanList>('GET', '/api/v1/billing/plans');
  }

  /**
   * Get current usage statistics.
   */
  async getBillingUsage(params?: { period?: string }): Promise<import('./types').BillingUsage> {
    return this.request<import('./types').BillingUsage>('GET', '/api/v1/billing/usage', { params });
  }

  /**
   * Get current subscription status.
   */
  async getSubscription(): Promise<import('./types').Subscription> {
    return this.request<import('./types').Subscription>('GET', '/api/v1/billing/subscription');
  }

  /**
   * Create a Stripe checkout session.
   */
  async createCheckoutSession(body: {
    plan_id: string;
    success_url: string;
    cancel_url: string;
  }): Promise<{ session_id: string; checkout_url: string }> {
    return this.request<{ session_id: string; checkout_url: string }>('POST', '/api/v1/billing/checkout', { body });
  }

  /**
   * Create a billing portal session.
   */
  async createBillingPortalSession(returnUrl?: string): Promise<{ url: string }> {
    return this.request<{ url: string }>('POST', '/api/v1/billing/portal', {
      body: { return_url: returnUrl },
    });
  }

  /**
   * Cancel subscription at period end.
   */
  async cancelSubscription(): Promise<{ cancelled: boolean; effective_date: string }> {
    return this.request<{ cancelled: boolean; effective_date: string }>('POST', '/api/v1/billing/cancel');
  }

  /**
   * Resume a cancelled subscription.
   */
  async resumeSubscription(): Promise<{ resumed: boolean }> {
    return this.request<{ resumed: boolean }>('POST', '/api/v1/billing/resume');
  }

  /**
   * Get invoice history.
   */
  async getInvoiceHistory(params?: PaginationParams): Promise<import('./types').InvoiceList> {
    return this.request<import('./types').InvoiceList>('GET', '/api/v1/billing/invoices', { params });
  }

  /**
   * Get usage forecast and tier recommendation.
   */
  async getUsageForecast(): Promise<import('./types').UsageForecast> {
    return this.request<import('./types').UsageForecast>('GET', '/api/v1/billing/usage/forecast');
  }

  /**
   * Export usage data as CSV.
   */
  async exportUsageData(params?: { start_date?: string; end_date?: string }): Promise<{ download_url: string }> {
    return this.request<{ download_url: string }>('GET', '/api/v1/billing/usage/export', { params });
  }

  // ===========================================================================
  // Notifications
  // ===========================================================================

  /**
   * Get notification integration status.
   */
  async getNotificationStatus(): Promise<import('./types').NotificationStatus> {
    return this.request<import('./types').NotificationStatus>('GET', '/api/v1/notifications/status');
  }

  /**
   * Configure email notifications.
   */
  async configureEmailNotifications(body: import('./types').EmailNotificationConfig): Promise<{ configured: boolean }> {
    return this.request<{ configured: boolean }>('POST', '/api/v1/notifications/email/config', { body });
  }

  /**
   * Configure Telegram notifications.
   */
  async configureTelegramNotifications(body: import('./types').TelegramNotificationConfig): Promise<{ configured: boolean }> {
    return this.request<{ configured: boolean }>('POST', '/api/v1/notifications/telegram/config', { body });
  }

  /**
   * Get email recipients.
   */
  async getEmailRecipients(): Promise<{ recipients: import('./types').NotificationRecipient[] }> {
    return this.request<{ recipients: import('./types').NotificationRecipient[] }>('GET', '/api/v1/notifications/email/recipients');
  }

  /**
   * Add email recipient.
   */
  async addEmailRecipient(body: { email: string; name?: string; preferences?: Record<string, boolean> }): Promise<{ added: boolean }> {
    return this.request<{ added: boolean }>('POST', '/api/v1/notifications/email/recipient', { body });
  }

  /**
   * Remove email recipient.
   */
  async removeEmailRecipient(email: string): Promise<{ removed: boolean }> {
    return this.request<{ removed: boolean }>('DELETE', '/api/v1/notifications/email/recipient', {
      params: { email },
    });
  }

  /**
   * Send a test notification.
   */
  async sendTestNotification(channel: 'email' | 'telegram' | 'slack'): Promise<{ sent: boolean }> {
    return this.request<{ sent: boolean }>('POST', '/api/v1/notifications/test', {
      body: { channel },
    });
  }

  /**
   * Send a custom notification.
   */
  async sendNotification(body: {
    channel: 'email' | 'telegram' | 'slack';
    subject?: string;
    message: string;
    recipients?: string[];
  }): Promise<{ sent: boolean; delivered_to: number }> {
    return this.request<{ sent: boolean; delivered_to: number }>('POST', '/api/v1/notifications/send', { body });
  }

  // ===========================================================================
  // Budgets
  // ===========================================================================

  /**
   * List budgets for the organization.
   */
  async listBudgets(params?: PaginationParams): Promise<import('./types').BudgetList> {
    return this.request<import('./types').BudgetList>('GET', '/api/v1/budgets', { params });
  }

  /**
   * Create a new budget.
   */
  async createBudget(body: import('./types').CreateBudgetRequest): Promise<import('./types').Budget> {
    return this.request<import('./types').Budget>('POST', '/api/v1/budgets', { body });
  }

  /**
   * Get a budget by ID.
   */
  async getBudget(budgetId: string): Promise<import('./types').Budget> {
    return this.request<import('./types').Budget>('GET', `/api/v1/budgets/${encodeURIComponent(budgetId)}`);
  }

  /**
   * Update a budget.
   */
  async updateBudget(budgetId: string, body: import('./types').UpdateBudgetRequest): Promise<import('./types').Budget> {
    return this.request<import('./types').Budget>('PATCH', `/api/v1/budgets/${encodeURIComponent(budgetId)}`, { body });
  }

  /**
   * Delete a budget.
   */
  async deleteBudget(budgetId: string): Promise<{ deleted: boolean }> {
    return this.request<{ deleted: boolean }>('DELETE', `/api/v1/budgets/${encodeURIComponent(budgetId)}`);
  }

  /**
   * Get budget alerts.
   */
  async getBudgetAlerts(budgetId: string, params?: PaginationParams): Promise<import('./types').BudgetAlertList> {
    return this.request<import('./types').BudgetAlertList>('GET', `/api/v1/budgets/${encodeURIComponent(budgetId)}/alerts`, { params });
  }

  /**
   * Acknowledge a budget alert.
   */
  async acknowledgeBudgetAlert(budgetId: string, alertId: string): Promise<{ acknowledged: boolean }> {
    return this.request<{ acknowledged: boolean }>(
      'POST',
      `/api/v1/budgets/${encodeURIComponent(budgetId)}/alerts/${encodeURIComponent(alertId)}/acknowledge`
    );
  }

  /**
   * Add a user override to a budget.
   */
  async addBudgetOverride(budgetId: string, body: { user_id: string; limit: number; reason?: string }): Promise<{ added: boolean }> {
    return this.request<{ added: boolean }>('POST', `/api/v1/budgets/${encodeURIComponent(budgetId)}/override`, { body });
  }

  /**
   * Remove a user override from a budget.
   */
  async removeBudgetOverride(budgetId: string, userId: string): Promise<{ removed: boolean }> {
    return this.request<{ removed: boolean }>(
      'DELETE',
      `/api/v1/budgets/${encodeURIComponent(budgetId)}/override/${encodeURIComponent(userId)}`
    );
  }

  /**
   * Reset a budget period.
   */
  async resetBudget(budgetId: string): Promise<{ reset: boolean; new_period_start: string }> {
    return this.request<{ reset: boolean; new_period_start: string }>(
      'POST',
      `/api/v1/budgets/${encodeURIComponent(budgetId)}/reset`
    );
  }

  /**
   * Get organization budget summary.
   */
  async getBudgetSummary(): Promise<import('./types').BudgetSummary> {
    return this.request<import('./types').BudgetSummary>('GET', '/api/v1/budgets/summary');
  }

  /**
   * Pre-flight cost check.
   */
  async checkBudget(body: {
    operation: string;
    estimated_cost: number;
    user_id?: string;
  }): Promise<{ allowed: boolean; remaining_budget: number; warnings?: string[] }> {
    return this.request<{ allowed: boolean; remaining_budget: number; warnings?: string[] }>(
      'POST',
      '/api/v1/budgets/check',
      { body }
    );
  }

  // ===========================================================================
  // Costs
  // ===========================================================================

  /**
   * Get cost dashboard data.
   */
  async getCostDashboard(params?: { period?: string }): Promise<import('./types').CostDashboard> {
    return this.request<import('./types').CostDashboard>('GET', '/api/v1/costs', { params });
  }

  /**
   * Get detailed cost breakdown.
   */
  async getCostBreakdown(params?: { period?: string; group_by?: string }): Promise<import('./types').CostBreakdown> {
    return this.request<import('./types').CostBreakdown>('GET', '/api/v1/costs/breakdown', { params });
  }

  /**
   * Get cost timeline.
   */
  async getCostTimeline(params?: { period?: string; granularity?: string }): Promise<import('./types').CostTimeline> {
    return this.request<import('./types').CostTimeline>('GET', '/api/v1/costs/timeline', { params });
  }

  /**
   * Get cost alerts.
   */
  async getCostAlerts(): Promise<{ alerts: import('./types').CostAlert[] }> {
    return this.request<{ alerts: import('./types').CostAlert[] }>('GET', '/api/v1/costs/alerts');
  }

  /**
   * Set budget limits.
   */
  async setCostBudget(body: {
    daily_limit?: number;
    monthly_limit?: number;
    alert_threshold?: number;
  }): Promise<{ set: boolean }> {
    return this.request<{ set: boolean }>('POST', '/api/v1/costs/budget', { body });
  }

  /**
   * Dismiss a cost alert.
   */
  async dismissCostAlert(alertId: string): Promise<{ dismissed: boolean }> {
    return this.request<{ dismissed: boolean }>('POST', `/api/v1/costs/alerts/${encodeURIComponent(alertId)}/dismiss`);
  }

  // ===========================================================================
  // Audit Trails
  // ===========================================================================

  /**
   * List audit trails.
   */
  async listAuditTrails(params?: {
    verdict?: string;
    risk_level?: string;
  } & PaginationParams): Promise<import('./types').AuditTrailList> {
    return this.request<import('./types').AuditTrailList>('GET', '/api/v1/audit-trails', { params });
  }

  /**
   * Get an audit trail by ID.
   */
  async getAuditTrail(trailId: string): Promise<import('./types').AuditTrail> {
    return this.request<import('./types').AuditTrail>('GET', `/api/v1/audit-trails/${encodeURIComponent(trailId)}`);
  }

  /**
   * Export an audit trail.
   */
  async exportAuditTrail(trailId: string, format: 'json' | 'csv' | 'markdown'): Promise<{ content: string; filename: string }> {
    return this.request<{ content: string; filename: string }>(
      'GET',
      `/api/v1/audit-trails/${encodeURIComponent(trailId)}/export`,
      { params: { format } }
    );
  }

  /**
   * Verify audit trail integrity.
   */
  async verifyAuditTrail(trailId: string): Promise<{ valid: boolean; checksum: string; verified_at: string }> {
    return this.request<{ valid: boolean; checksum: string; verified_at: string }>(
      'POST',
      `/api/v1/audit-trails/${encodeURIComponent(trailId)}/verify`
    );
  }

  // ===========================================================================
  // Decision Receipts (extended)
  // ===========================================================================

  /**
   * List decision receipts.
   */
  async listDecisionReceipts(params?: { verdict?: string } & PaginationParams): Promise<{ receipts: DecisionReceipt[] }> {
    return this.request<{ receipts: DecisionReceipt[] }>('GET', '/api/v1/receipts', { params });
  }

  /**
   * Get a decision receipt.
   */
  async getDecisionReceipt(receiptId: string): Promise<DecisionReceipt> {
    return this.request<DecisionReceipt>('GET', `/api/v1/receipts/${encodeURIComponent(receiptId)}`);
  }

  /**
   * Verify a decision receipt's integrity.
   */
  async verifyDecisionReceipt(receiptId: string): Promise<{ valid: boolean; hash: string; verified_at: string }> {
    return this.request<{ valid: boolean; hash: string; verified_at: string }>(
      'POST',
      `/api/v1/receipts/${encodeURIComponent(receiptId)}/verify`
    );
  }

  /**
   * Run a debate and wait for completion.
   * Convenience method that creates a debate and polls until complete.
   */
  async runDebate(
    request: DebateCreateRequest,
    options?: {
      pollIntervalMs?: number;
      timeoutMs?: number;
    }
  ): Promise<Debate> {
    const response = await this.createDebate(request);
    const debateId = response.debate_id;
    const pollInterval = options?.pollIntervalMs ?? 1000;
    const timeout = options?.timeoutMs ?? 300000;

    const startTime = Date.now();
    while (Date.now() - startTime < timeout) {
      const debate = await this.getDebate(debateId);
      if (['completed', 'failed', 'cancelled'].includes(debate.status)) {
        return debate;
      }
      await this.sleep(pollInterval);
    }

    throw new AragoraError(
      `Debate ${debateId} did not complete within ${timeout}ms`,
      'AGENT_TIMEOUT'
    );
  }
}

/**
 * Create a new Aragora client instance.
 */
export function createClient(config: AragoraConfig): AragoraClient {
  return new AragoraClient(config);
}
