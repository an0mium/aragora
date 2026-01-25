/**
 * Aragora API Client
 *
 * Main client for interacting with the Aragora multi-agent debate platform.
 */

import type {
  AragoraConfig,
  Agent,
  AgentCalibration,
  AgentConsistency,
  AgentFlip,
  AgentMoment,
  AgentNetwork,
  AgentPerformance,
  AgentPosition,
  AgentProfile,
  AgentScore,
  AuditEvent,
  AuditFinding,
  AuditSession,
  AuditStats,
  ConsensusQualityAnalytics,
  CreateAuditSessionRequest,
  CreateTournamentRequest,
  CritiqueEntry,
  Debate,
  DebateCreateRequest,
  DebateCreateResponse,
  DebateUpdateRequest,
  DisagreementAnalytics,
  DomainRating,
  EarlyStopAnalytics,
  ExplainabilityResult,
  GauntletComparison,
  GauntletHeatmapExtended,
  GauntletPersona,
  GauntletResult,
  GauntletRun,
  GauntletRunRequest,
  GauntletRunResponse,
  GraphDebate,
  GraphDebateCreateRequest,
  HeadToHeadStats,
  HealthCheck,
  KnowledgeEntry,
  KnowledgeSearchResult,
  KnowledgeStats,
  MarketplaceTemplate,
  MatrixConclusion,
  MatrixDebate,
  MatrixDebateCreateRequest,
  MemoryAnalytics,
  MemoryEntry,
  MemorySearchParams,
  MemoryStats,
  MemoryTier,
  MemoryTierStats,
  OnboardingStatus,
  OpponentBriefing,
  PaginationParams,
  RankingStats,
  Replay,
  ReplayFormat,
  DecisionReceipt,
  RiskHeatmap,
  RoleRotationAnalytics,
  ScoreAgentsRequest,
  SearchResponse,
  SelectionPlugin,
  Tenant,
  TeamSelection,
  TeamSelectionRequest,
  Tournament,
  TournamentBracket,
  TournamentMatch,
  TournamentStandings,
  VerificationBackend,
  VerificationReport,
  VerificationResult,
  VerificationStatus,
  VerifyClaimRequest,
  WebSocketEvent,
  Workflow,
  WorkflowTemplate,
} from './types';
import { AragoraError } from './types';
import { AragoraWebSocket, createWebSocket, streamDebate, type WebSocketOptions, type StreamOptions } from './websocket';

interface RequestOptions {
  body?: unknown;
  params?: Record<string, string | number | boolean | undefined>;
  headers?: Record<string, string>;
  timeout?: number;
}

/**
 * Main Aragora API client.
 */
export class AragoraClient {
  private config: Required<Omit<AragoraConfig, 'apiKey' | 'wsUrl'>> & Pick<AragoraConfig, 'apiKey' | 'wsUrl'>;

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
  }

  // ===========================================================================
  // Core Request Method
  // ===========================================================================

  private async request<T>(
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

  async compareAgents(agents: string[]): Promise<unknown> {
    return this.request<unknown>('GET', '/api/agent/compare', {
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

  async getDebateMessages(debateId: string): Promise<{ messages: unknown[] }> {
    return this.request<{ messages: unknown[] }>('GET', `/api/debates/${encodeURIComponent(debateId)}/messages`);
  }

  async getDebateConvergence(debateId: string): Promise<unknown> {
    return this.request<unknown>('GET', `/api/debates/${encodeURIComponent(debateId)}/convergence`);
  }

  async getDebateCitations(debateId: string): Promise<unknown> {
    return this.request<unknown>('GET', `/api/debates/${encodeURIComponent(debateId)}/citations`);
  }

  async getDebateEvidence(debateId: string): Promise<unknown> {
    return this.request<unknown>('GET', `/api/debates/${encodeURIComponent(debateId)}/evidence`);
  }

  async forkDebate(debateId: string, options?: { branch_point?: number }): Promise<{ debate_id: string }> {
    return this.request<{ debate_id: string }>('POST', `/api/debates/${encodeURIComponent(debateId)}/fork`, { body: options });
  }

  async exportDebate(debateId: string, format: 'json' | 'markdown' | 'html' | 'pdf'): Promise<unknown> {
    return this.request<unknown>('GET', `/api/debates/${encodeURIComponent(debateId)}/export/${format}`);
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

  async getExplanationFactors(debateId: string, options?: { min_contribution?: number }): Promise<unknown> {
    return this.request<unknown>('GET', `/api/debates/${encodeURIComponent(debateId)}/explainability/factors`, { params: options });
  }

  async getCounterfactuals(debateId: string, options?: { max_scenarios?: number }): Promise<unknown> {
    return this.request<unknown>('GET', `/api/debates/${encodeURIComponent(debateId)}/explainability/counterfactual`, { params: options });
  }

  async generateCounterfactual(debateId: string, body: {
    hypothesis: string;
    affected_agents?: string[];
  }): Promise<unknown> {
    return this.request<unknown>('POST', `/api/debates/${encodeURIComponent(debateId)}/explainability/counterfactual`, { body });
  }

  async getProvenance(debateId: string): Promise<unknown> {
    return this.request<unknown>('GET', `/api/debates/${encodeURIComponent(debateId)}/explainability/provenance`);
  }

  async getNarrative(debateId: string, options?: { format?: 'brief' | 'detailed' | 'executive_summary' }): Promise<unknown> {
    return this.request<unknown>('GET', `/api/debates/${encodeURIComponent(debateId)}/explainability/narrative`, { params: options });
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
  }): Promise<unknown> {
    return this.request<unknown>('POST', `/api/workflow/templates/${encodeURIComponent(templateId)}/run`, { body });
  }

  async listWorkflowCategories(): Promise<{ categories: string[] }> {
    return this.request<{ categories: string[] }>('GET', '/api/workflow/categories');
  }

  async listWorkflowPatterns(): Promise<{ patterns: string[] }> {
    return this.request<{ patterns: string[] }>('GET', '/api/workflow/patterns');
  }

  async getWorkflowTemplatePackage(templateId: string, options?: {
    include_examples?: boolean;
  }): Promise<{ template: WorkflowTemplate; examples?: unknown[] }> {
    return this.request<{ template: WorkflowTemplate; examples?: unknown[] }>(
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

  async exportGauntletReceipt(receiptId: string, format: 'json' | 'html' | 'markdown' | 'sarif'): Promise<unknown> {
    return this.request<unknown>('GET', `/api/gauntlet/receipts/${encodeURIComponent(receiptId)}/export`, { params: { format } });
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

  async retrieveFromContinuum(query: string, options?: {
    tier?: 'fast' | 'medium' | 'slow' | 'glacial';
    limit?: number;
  }): Promise<{ entries: MemoryEntry[] }> {
    return this.request<{ entries: MemoryEntry[] }>('GET', '/api/memory/continuum/retrieve', {
      params: { q: query, ...options },
    });
  }

  async consolidateMemory(): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>('POST', '/api/memory/continuum/consolidate');
  }

  async getConsensusStats(): Promise<unknown> {
    return this.request<unknown>('GET', '/api/consensus/stats');
  }

  async getSettledConsensus(params?: { domain?: string } & PaginationParams): Promise<unknown> {
    return this.request<unknown>('GET', '/api/consensus/settled', { params });
  }

  async getSimilarConsensus(topic: string): Promise<unknown> {
    return this.request<unknown>('GET', '/api/consensus/similar', { params: { topic } });
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
  // Relationships
  // ===========================================================================

  async getRelationshipSummary(): Promise<unknown> {
    return this.request<unknown>('GET', '/api/relationships/summary');
  }

  async getRelationshipGraph(): Promise<unknown> {
    return this.request<unknown>('GET', '/api/relationships/graph');
  }

  async getAgentRelationship(agentA: string, agentB: string): Promise<unknown> {
    return this.request<unknown>('GET', `/api/relationship/${encodeURIComponent(agentA)}/${encodeURIComponent(agentB)}`);
  }

  // ===========================================================================
  // Pulse (Trending)
  // ===========================================================================

  async getTrendingTopics(category?: string): Promise<unknown> {
    return this.request<unknown>('GET', '/api/pulse/trending', { params: { category } });
  }

  async getSuggestedTopics(): Promise<unknown> {
    return this.request<unknown>('GET', '/api/pulse/suggest');
  }

  // ===========================================================================
  // Metrics
  // ===========================================================================

  async getMetrics(): Promise<string> {
    // Prometheus format
    return this.request<string>('GET', '/metrics');
  }

  async getMetricsHealth(): Promise<unknown> {
    return this.request<unknown>('GET', '/api/metrics/health');
  }

  async getSystemMetrics(): Promise<unknown> {
    return this.request<unknown>('GET', '/api/metrics/system');
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
    backend?: VerificationBackend;
    include_assumptions?: boolean;
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
  async deleteRole(roleId: string): Promise<void> {
    await this.request<void>('DELETE', `/api/v1/rbac/roles/${encodeURIComponent(roleId)}`);
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
    debate_id: string;
    notes?: string;
  }): Promise<{ recorded: boolean }> {
    return this.request<{ recorded: boolean }>(
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
  }): Promise<{ export_id: string; download_url?: string }> {
    return this.request<{ export_id: string; download_url?: string }>('POST', '/api/v1/audit/export', { body: request });
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
    last_active: string;
    is_current: boolean;
  }> }> {
    return this.request<{ sessions: Array<{
      id: string;
      device: string;
      ip_address: string;
      last_active: string;
      is_current: boolean;
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
