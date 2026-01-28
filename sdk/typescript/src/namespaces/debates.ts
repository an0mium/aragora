/**
 * Debates Namespace API
 *
 * Provides a namespaced interface for debate-related operations.
 * This wraps the flat client methods for a more intuitive API.
 */

import type {
  Debate,
  DebateCreateRequest,
  DebateCreateResponse,
  DebateConvergence,
  DebateCitations,
  DebateEvidence,
  DebateExport,
  DebateUpdateRequest,
  GraphStats,
  Message,
  PaginationParams,
  WebSocketEvent,
} from '../types';
import type { StreamOptions } from '../websocket';

/**
 * Debate impasse analysis result
 */
export interface DebateImpasse {
  is_impasse: boolean;
  confidence: number;
  reason: string | null;
  stuck_since_round: number | null;
  suggested_intervention: string | null;
}

/**
 * Rhetorical pattern observation
 */
export interface RhetoricalObservation {
  agent: string;
  pattern: string;
  description: string;
  severity: number;
  round: number;
  timestamp: string;
}

/**
 * Rhetorical analysis response
 */
export interface RhetoricalAnalysis {
  debate_id: string;
  observations: RhetoricalObservation[];
  summary: {
    total_observations: number;
    patterns_detected: string[];
    agents_flagged: string[];
  };
}

/**
 * Trickster hollow consensus status
 */
export interface TricksterStatus {
  debate_id: string;
  hollow_consensus_detected: boolean;
  confidence: number;
  indicators: Array<{
    type: string;
    description: string;
    severity: number;
  }>;
  recommendation: string | null;
}

/**
 * Meta-critique analysis
 */
export interface MetaCritique {
  debate_id: string;
  quality_score: number;
  critique: string;
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
  agent_performance: Array<{
    agent: string;
    contribution_score: number;
    critique: string;
  }>;
}

/**
 * Debate summary with key points
 */
export interface DebateSummary {
  debate_id: string;
  verdict: string;
  confidence: number;
  key_points: string[];
  dissenting_views: string[];
  evidence_quality: number;
  generated_at: string;
}

/**
 * Verification report for debate conclusions
 */
export interface VerificationReport {
  debate_id: string;
  verified: boolean;
  confidence: number;
  claims_verified: number;
  claims_total: number;
  verification_details: Array<{
    claim: string;
    status: 'verified' | 'unverified' | 'disputed';
    evidence: string[];
    confidence: number;
  }>;
  bonuses: Array<{
    type: string;
    amount: number;
    reason: string;
  }>;
  generated_at: string;
}

/**
 * Claim verification result
 */
export interface ClaimVerification {
  claim_id: string;
  verified: boolean;
  confidence: number;
  supporting_evidence: string[];
  counter_evidence: string[];
  status: 'verified' | 'unverified' | 'disputed' | 'pending';
}

/**
 * Follow-up debate suggestion
 */
export interface FollowupSuggestion {
  id: string;
  topic: string;
  question: string;
  crux_id?: string;
  rationale: string;
  priority: 'high' | 'medium' | 'low';
  estimated_value: number;
}

/**
 * Fork information
 */
export interface ForkInfo {
  fork_id: string;
  parent_debate_id: string;
  branch_point: number;
  created_at: string;
  status: string;
  divergence_reason?: string;
}

/**
 * Debate search options
 */
export interface DebateSearchOptions {
  query: string;
  limit?: number;
  offset?: number;
  status?: string;
  domain?: string;
  since?: string;
  until?: string;
}

/**
 * Batch job status
 */
export interface BatchJob {
  job_id: string;
  debate_id?: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  error?: string;
  created_at: string;
}

/**
 * Batch submission response
 */
export interface BatchSubmission {
  batch_id: string;
  jobs: BatchJob[];
  total_jobs: number;
  submitted_at: string;
}

/**
 * Batch status response
 */
export interface BatchStatus {
  batch_id: string;
  status: 'pending' | 'running' | 'completed' | 'partially_completed' | 'failed';
  total_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  jobs: BatchJob[];
}

/**
 * Queue status information
 */
export interface QueueStatus {
  pending_count: number;
  running_count: number;
  completed_today: number;
  average_wait_time_ms: number;
  estimated_completion_time?: string;
}

/**
 * Graph visualization data
 */
export interface DebateGraph {
  nodes: Array<{
    id: string;
    type: 'claim' | 'evidence' | 'argument' | 'counter';
    content: string;
    agent: string;
    round: number;
    confidence?: number;
  }>;
  edges: Array<{
    source: string;
    target: string;
    type: 'supports' | 'attacks' | 'responds_to';
    weight?: number;
  }>;
  metadata: {
    total_nodes: number;
    total_edges: number;
    depth: number;
    branching_factor: number;
  };
}

/**
 * Graph branch information
 */
export interface GraphBranch {
  branch_id: string;
  root_node: string;
  depth: number;
  node_count: number;
  conclusion?: string;
}

/**
 * Matrix debate comparison
 */
export interface MatrixComparison {
  debate_id: string;
  scenarios: Array<{
    scenario_id: string;
    name: string;
    parameters: Record<string, unknown>;
    outcome: string;
    confidence: number;
  }>;
  comparison_matrix: Array<Array<number>>;
  dominant_scenario?: string;
  sensitivity_analysis?: Record<string, number>;
}

/**
 * Interface for the internal client methods used by DebatesAPI.
 * This allows the namespace to work without circular imports.
 */
interface DebatesClientInterface {
  listDebates(params?: PaginationParams & { status?: string }): Promise<{ debates: Debate[] }>;
  getDebate(debateId: string): Promise<Debate>;
  getDebateBySlug(slug: string): Promise<Debate>;
  createDebate(request: DebateCreateRequest): Promise<DebateCreateResponse>;
  getDebateMessages(debateId: string): Promise<{ messages: Message[] }>;
  getDebateConvergence(debateId: string): Promise<DebateConvergence>;
  getDebateCitations(debateId: string): Promise<DebateCitations>;
  getDebateEvidence(debateId: string): Promise<DebateEvidence>;
  forkDebate(debateId: string, options?: { branch_point?: number }): Promise<{ debate_id: string }>;
  exportDebate(debateId: string, format: 'json' | 'markdown' | 'html' | 'pdf'): Promise<DebateExport>;
  updateDebate(debateId: string, updates: DebateUpdateRequest): Promise<Debate>;
  getDebateGraphStats(debateId: string): Promise<GraphStats>;
  createDebateAndStream(
    request: DebateCreateRequest,
    streamOptions?: Omit<StreamOptions, 'debateId'>
  ): Promise<{ debate: DebateCreateResponse; stream: AsyncGenerator<WebSocketEvent, void, unknown> }>;
  runDebate(
    request: DebateCreateRequest,
    options?: { pollIntervalMs?: number; timeoutMs?: number }
  ): Promise<Debate>;
  request<T>(method: string, path: string, options?: { params?: Record<string, unknown> }): Promise<T>;
}

/**
 * Debates API namespace.
 *
 * Provides methods for managing debates:
 * - Creating and running debates
 * - Retrieving debate details and messages
 * - Exporting debates in various formats
 * - Analyzing debate convergence and evidence
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Create a debate
 * const response = await client.debates.create({
 *   task: 'Should we use microservices?',
 *   agents: ['claude', 'gpt-4'],
 * });
 *
 * // Get debate details
 * const debate = await client.debates.get(response.debate_id);
 *
 * // List all debates
 * const { debates } = await client.debates.list({ limit: 10 });
 *
 * // Export as markdown
 * const export = await client.debates.export(debateId, 'markdown');
 * ```
 */
export class DebatesAPI {
  constructor(private client: DebatesClientInterface) {}

  /**
   * List all debates with optional filtering and pagination.
   */
  async list(params?: PaginationParams & { status?: string }): Promise<{ debates: Debate[] }> {
    return this.client.listDebates(params);
  }

  /**
   * Get a debate by ID.
   */
  async get(debateId: string): Promise<Debate> {
    return this.client.getDebate(debateId);
  }

  /**
   * Get a debate by its URL slug.
   */
  async getBySlug(slug: string): Promise<Debate> {
    return this.client.getDebateBySlug(slug);
  }

  /**
   * Create a new debate.
   */
  async create(request: DebateCreateRequest): Promise<DebateCreateResponse> {
    return this.client.createDebate(request);
  }

  /**
   * Update an existing debate.
   */
  async update(debateId: string, updates: DebateUpdateRequest): Promise<Debate> {
    return this.client.updateDebate(debateId, updates);
  }

  /**
   * Get all messages from a debate.
   */
  async getMessages(debateId: string): Promise<{ messages: Message[] }> {
    return this.client.getDebateMessages(debateId);
  }

  /**
   * Get convergence analysis for a debate.
   */
  async getConvergence(debateId: string): Promise<DebateConvergence> {
    return this.client.getDebateConvergence(debateId);
  }

  /**
   * Get citations from a debate.
   */
  async getCitations(debateId: string): Promise<DebateCitations> {
    return this.client.getDebateCitations(debateId);
  }

  /**
   * Get evidence gathered during a debate.
   */
  async getEvidence(debateId: string): Promise<DebateEvidence> {
    return this.client.getDebateEvidence(debateId);
  }

  /**
   * Fork a debate from a specific point.
   */
  async fork(debateId: string, options?: { branch_point?: number }): Promise<{ debate_id: string }> {
    return this.client.forkDebate(debateId, options);
  }

  /**
   * Export a debate in a specific format.
   */
  async export(debateId: string, format: 'json' | 'markdown' | 'html' | 'pdf'): Promise<DebateExport> {
    return this.client.exportDebate(debateId, format);
  }

  /**
   * Get graph statistics for a debate.
   */
  async getGraphStats(debateId: string): Promise<GraphStats> {
    return this.client.getDebateGraphStats(debateId);
  }

  /**
   * Create a debate and return a stream of events.
   */
  async createAndStream(
    request: DebateCreateRequest,
    streamOptions?: Omit<StreamOptions, 'debateId'>
  ): Promise<{ debate: DebateCreateResponse; stream: AsyncGenerator<WebSocketEvent, void, unknown> }> {
    return this.client.createDebateAndStream(request, streamOptions);
  }

  /**
   * Create a debate and wait for it to complete.
   * Polls the debate status until it reaches a terminal state.
   */
  async run(
    request: DebateCreateRequest,
    options?: { pollIntervalMs?: number; timeoutMs?: number }
  ): Promise<Debate> {
    return this.client.runDebate(request, options);
  }

  /**
   * Alias for run() - creates a debate and waits for completion.
   */
  async waitForCompletion(debateId: string, options?: { pollIntervalMs?: number; timeoutMs?: number }): Promise<Debate> {
    const pollInterval = options?.pollIntervalMs ?? 1000;
    const timeout = options?.timeoutMs ?? 300000;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const debate = await this.client.getDebate(debateId);
      if (['completed', 'failed', 'cancelled'].includes(debate.status)) {
        return debate;
      }
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new Error(`Debate ${debateId} did not complete within ${timeout}ms`);
  }

  // ===========================================================================
  // Analysis
  // ===========================================================================

  /**
   * Detect if a debate has reached an impasse.
   *
   * An impasse occurs when the debate is stuck and not making progress
   * toward consensus. This helps identify when intervention may be needed.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const impasse = await client.debates.getImpasse('debate-123');
   * if (impasse.is_impasse) {
   *   console.log(`Impasse detected: ${impasse.reason}`);
   *   console.log(`Suggested: ${impasse.suggested_intervention}`);
   * }
   * ```
   */
  async getImpasse(debateId: string): Promise<DebateImpasse> {
    return this.client.request('GET', `/api/v1/debates/${debateId}/impasse`);
  }

  /**
   * Get rhetorical pattern observations for a debate.
   *
   * Analyzes the debate for rhetorical patterns that may indicate
   * manipulation, circular reasoning, or other issues.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const analysis = await client.debates.getRhetorical('debate-123');
   * console.log(`Found ${analysis.observations.length} rhetorical patterns`);
   * for (const obs of analysis.observations) {
   *   console.log(`${obs.agent}: ${obs.pattern} (severity: ${obs.severity})`);
   * }
   * ```
   */
  async getRhetorical(debateId: string): Promise<RhetoricalAnalysis> {
    return this.client.request('GET', `/api/v1/debates/${debateId}/rhetorical`);
  }

  /**
   * Get trickster hollow consensus detection status.
   *
   * The Trickster detects "hollow consensus" - apparent agreement that
   * masks underlying disagreement or manipulation.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const status = await client.debates.getTrickster('debate-123');
   * if (status.hollow_consensus_detected) {
   *   console.log(`Warning: Hollow consensus detected (${status.confidence})`);
   *   for (const ind of status.indicators) {
   *     console.log(`- ${ind.type}: ${ind.description}`);
   *   }
   * }
   * ```
   */
  async getTrickster(debateId: string): Promise<TricksterStatus> {
    return this.client.request('GET', `/api/v1/debates/${debateId}/trickster`);
  }

  /**
   * Get meta-level critique of the debate.
   *
   * Provides an overall analysis of debate quality, including
   * strengths, weaknesses, and recommendations.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const meta = await client.debates.getMetaCritique('debate-123');
   * console.log(`Overall quality: ${meta.quality_score}/100`);
   * console.log(`Strengths: ${meta.strengths.join(', ')}`);
   * console.log(`Weaknesses: ${meta.weaknesses.join(', ')}`);
   * ```
   */
  async getMetaCritique(debateId: string): Promise<MetaCritique> {
    return this.client.request('GET', `/api/v1/debate/${debateId}/meta-critique`);
  }

  // ===========================================================================
  // Summary & Verification
  // ===========================================================================

  /**
   * Get a human-readable summary of the debate.
   *
   * Provides a condensed verdict with key points, confidence level,
   * and any dissenting views that emerged.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const summary = await client.debates.getSummary('debate-123');
   * console.log(`Verdict: ${summary.verdict}`);
   * console.log(`Confidence: ${summary.confidence}%`);
   * console.log('Key points:', summary.key_points);
   * ```
   */
  async getSummary(debateId: string): Promise<DebateSummary> {
    return this.client.request('GET', `/api/v1/debates/${debateId}/summary`);
  }

  /**
   * Get the verification report for debate conclusions.
   *
   * Shows which claims were verified, evidence quality, and any
   * bonuses awarded for verified conclusions.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const report = await client.debates.getVerificationReport('debate-123');
   * console.log(`Verified: ${report.claims_verified}/${report.claims_total}`);
   * for (const detail of report.verification_details) {
   *   console.log(`${detail.claim}: ${detail.status}`);
   * }
   * ```
   */
  async getVerificationReport(debateId: string): Promise<VerificationReport> {
    return this.client.request('GET', `/api/v1/debates/${debateId}/verification-report`);
  }

  /**
   * Verify a specific claim from the debate.
   *
   * @param debateId - The debate ID
   * @param claimId - The claim to verify
   * @param evidence - Optional additional evidence to consider
   *
   * @example
   * ```typescript
   * const result = await client.debates.verifyClaim('debate-123', 'claim-456');
   * if (result.verified) {
   *   console.log(`Claim verified with ${result.confidence}% confidence`);
   * }
   * ```
   */
  async verifyClaim(
    debateId: string,
    claimId: string,
    evidence?: string
  ): Promise<ClaimVerification> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/verify`, {
      body: { claim_id: claimId, evidence },
    });
  }

  // ===========================================================================
  // Follow-up & Continuation
  // ===========================================================================

  /**
   * Get suggestions for follow-up debates.
   *
   * Based on unresolved cruxes, dissenting views, or areas
   * that warrant deeper exploration.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const suggestions = await client.debates.getFollowupSuggestions('debate-123');
   * for (const s of suggestions) {
   *   console.log(`${s.priority}: ${s.question}`);
   *   console.log(`  Rationale: ${s.rationale}`);
   * }
   * ```
   */
  async getFollowupSuggestions(debateId: string): Promise<FollowupSuggestion[]> {
    const response = await this.client.request<{ suggestions: FollowupSuggestion[] }>(
      'GET',
      `/api/v1/debates/${debateId}/followups`
    );
    return response.suggestions;
  }

  /**
   * Create a follow-up debate from an existing one.
   *
   * Can optionally target a specific crux for deeper exploration.
   *
   * @param debateId - The parent debate ID
   * @param options - Follow-up options
   *
   * @example
   * ```typescript
   * const followup = await client.debates.followUp('debate-123', {
   *   cruxId: 'crux-456',
   *   context: 'Focus on the security implications',
   * });
   * console.log(`New debate created: ${followup.debate_id}`);
   * ```
   */
  async followUp(
    debateId: string,
    options?: { cruxId?: string; context?: string }
  ): Promise<{ debate_id: string }> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/followup`, {
      body: options,
    });
  }

  /**
   * List all forks created from a debate.
   *
   * @param debateId - The parent debate ID
   *
   * @example
   * ```typescript
   * const forks = await client.debates.listForks('debate-123');
   * console.log(`${forks.length} forks created`);
   * for (const fork of forks) {
   *   console.log(`Fork at round ${fork.branch_point}: ${fork.fork_id}`);
   * }
   * ```
   */
  async listForks(debateId: string): Promise<ForkInfo[]> {
    const response = await this.client.request<{ forks: ForkInfo[] }>(
      'GET',
      `/api/v1/debates/${debateId}/forks`
    );
    return response.forks;
  }

  // ===========================================================================
  // Debate Lifecycle
  // ===========================================================================

  /**
   * Cancel a running debate.
   *
   * @param debateId - The debate ID to cancel
   *
   * @example
   * ```typescript
   * const result = await client.debates.cancel('debate-123');
   * if (result.success) {
   *   console.log(`Debate cancelled: ${result.status}`);
   * }
   * ```
   */
  async cancel(debateId: string): Promise<{ success: boolean; status: string }> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/cancel`);
  }

  // ===========================================================================
  // Search & Discovery
  // ===========================================================================

  /**
   * Search across all debates.
   *
   * @param options - Search options including query, filters, and pagination
   *
   * @example
   * ```typescript
   * const results = await client.debates.search({
   *   query: 'microservices architecture',
   *   limit: 10,
   *   status: 'completed',
   * });
   * console.log(`Found ${results.debates.length} matching debates`);
   * ```
   */
  async search(options: DebateSearchOptions): Promise<{ debates: Debate[]; total: number }> {
    return this.client.request('GET', '/api/v1/search', {
      params: options as Record<string, unknown>,
    });
  }

  // ===========================================================================
  // Batch Operations
  // ===========================================================================

  /**
   * Submit multiple debates for batch processing.
   *
   * @param requests - Array of debate creation requests
   *
   * @example
   * ```typescript
   * const batch = await client.debates.submitBatch([
   *   { task: 'Should we use Redis?' },
   *   { task: 'Should we use PostgreSQL?' },
   * ]);
   * console.log(`Batch ${batch.batch_id} submitted with ${batch.total_jobs} jobs`);
   * ```
   */
  async submitBatch(requests: DebateCreateRequest[]): Promise<BatchSubmission> {
    return this.client.request('POST', '/api/v1/debates/batch', {
      body: { requests },
    });
  }

  /**
   * Get the status of a batch job.
   *
   * @param batchId - The batch ID
   *
   * @example
   * ```typescript
   * const status = await client.debates.getBatchStatus('batch-123');
   * console.log(`Progress: ${status.completed_jobs}/${status.total_jobs}`);
   * ```
   */
  async getBatchStatus(batchId: string): Promise<BatchStatus> {
    return this.client.request('GET', `/api/v1/debates/batch/${batchId}/status`);
  }

  /**
   * List all batch jobs.
   *
   * @param options - Filter options
   *
   * @example
   * ```typescript
   * const batches = await client.debates.listBatches({ limit: 10 });
   * for (const batch of batches) {
   *   console.log(`${batch.batch_id}: ${batch.status}`);
   * }
   * ```
   */
  async listBatches(options?: {
    limit?: number;
    offset?: number;
    status?: string;
  }): Promise<BatchStatus[]> {
    const response = await this.client.request<{ batches: BatchStatus[] }>(
      'GET',
      '/api/v1/debates/batch',
      { params: options as Record<string, unknown> }
    );
    return response.batches;
  }

  /**
   * Get the current queue status.
   *
   * @example
   * ```typescript
   * const queue = await client.debates.getQueueStatus();
   * console.log(`${queue.pending_count} pending, ${queue.running_count} running`);
   * console.log(`Avg wait: ${queue.average_wait_time_ms}ms`);
   * ```
   */
  async getQueueStatus(): Promise<QueueStatus> {
    return this.client.request('GET', '/api/v1/debates/queue/status');
  }

  // ===========================================================================
  // Graph & Visualization
  // ===========================================================================

  /**
   * Get the argument graph for a debate.
   *
   * Returns nodes (claims, evidence, arguments) and edges (relationships)
   * for visualization.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const graph = await client.debates.getGraph('debate-123');
   * console.log(`${graph.nodes.length} nodes, ${graph.edges.length} edges`);
   * console.log(`Max depth: ${graph.metadata.depth}`);
   * ```
   */
  async getGraph(debateId: string): Promise<DebateGraph> {
    return this.client.request('GET', `/api/v1/debates/graph/${debateId}`);
  }

  /**
   * Get branches in the argument graph.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const branches = await client.debates.getGraphBranches('debate-123');
   * for (const branch of branches) {
   *   console.log(`Branch ${branch.branch_id}: ${branch.node_count} nodes`);
   * }
   * ```
   */
  async getGraphBranches(debateId: string): Promise<GraphBranch[]> {
    const response = await this.client.request<{ branches: GraphBranch[] }>(
      'GET',
      `/api/v1/debates/graph/${debateId}/branches`
    );
    return response.branches;
  }

  /**
   * Get matrix comparison for a multi-scenario debate.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const matrix = await client.debates.getMatrixComparison('debate-123');
   * console.log(`${matrix.scenarios.length} scenarios compared`);
   * if (matrix.dominant_scenario) {
   *   console.log(`Dominant: ${matrix.dominant_scenario}`);
   * }
   * ```
   */
  async getMatrixComparison(debateId: string): Promise<MatrixComparison> {
    return this.client.request('GET', `/api/v1/debates/matrix/${debateId}`);
  }
}
