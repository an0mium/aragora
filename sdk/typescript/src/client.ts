/**
 * Aragora API Client
 *
 * Main client for interacting with the Aragora multi-agent debate platform.
 */

import type {
  AragoraConfig,
  Agent,
  AgentProfile,
  AgentScore,
  Debate,
  DebateCreateRequest,
  DebateCreateResponse,
  GraphDebate,
  GraphDebateCreateRequest,
  HealthCheck,
  MatrixConclusion,
  MatrixDebate,
  MatrixDebateCreateRequest,
  MemoryAnalytics,
  MemoryTier,
  MemoryTierStats,
  PaginationParams,
  Replay,
  ReplayFormat,
  ScoreAgentsRequest,
  SelectionPlugin,
  TeamSelection,
  TeamSelectionRequest,
  VerificationBackend,
  VerificationResult,
  VerificationStatus,
  VerifyClaimRequest,
  WebSocketEvent,
  Workflow,
  WorkflowTemplate,
  DecisionReceipt,
  RiskHeatmap,
  ExplainabilityResult,
  MarketplaceTemplate,
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

  async getAgentCalibration(name: string): Promise<unknown> {
    return this.request<unknown>('GET', `/api/agent/${encodeURIComponent(name)}/calibration`);
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

  async searchDebates(query: string, params?: PaginationParams): Promise<{ debates: Debate[] }> {
    return this.request<{ debates: Debate[] }>('GET', '/api/search', { params: { q: query, ...params } });
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

  async getMemoryStats(): Promise<unknown> {
    return this.request<unknown>('GET', '/api/memory/stats');
  }

  async retrieveFromContinuum(query: string, options?: {
    tier?: 'fast' | 'medium' | 'slow' | 'glacial';
    limit?: number;
  }): Promise<unknown> {
    return this.request<unknown>('GET', '/api/memory/continuum/retrieve', {
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

  async getDisagreementAnalytics(): Promise<unknown> {
    return this.request<unknown>('GET', '/api/analytics/disagreements');
  }

  async getRoleRotationAnalytics(): Promise<unknown> {
    return this.request<unknown>('GET', '/api/analytics/role-rotation');
  }

  async getEarlyStopAnalytics(): Promise<unknown> {
    return this.request<unknown>('GET', '/api/analytics/early-stops');
  }

  async getRankingStats(): Promise<unknown> {
    return this.request<unknown>('GET', '/api/ranking/stats');
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
