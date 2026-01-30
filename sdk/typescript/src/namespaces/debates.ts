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
  request<T>(method: string, path: string, options?: { params?: Record<string, unknown>; body?: unknown }): Promise<T>;
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

  // ===========================================================================
  // Core CRUD Operations
  // ===========================================================================

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
   * Delete a debate.
   *
   * @param debateId - The debate ID to delete
   *
   * @example
   * ```typescript
   * const result = await client.debates.delete('debate-123');
   * if (result.success) {
   *   console.log('Debate deleted');
   * }
   * ```
   */
  async delete(debateId: string): Promise<{ success: boolean }> {
    return this.client.request('DELETE', `/api/v1/debates/${debateId}`);
  }

  /**
   * Get all messages from a debate.
   */
  async getMessages(debateId: string): Promise<{ messages: Message[] }> {
    return this.client.getDebateMessages(debateId);
  }

  /**
   * Add a message to a debate.
   *
   * @param debateId - The debate ID
   * @param content - Message content
   * @param role - Message role (user, system, etc.)
   *
   * @example
   * ```typescript
   * const message = await client.debates.addMessage('debate-123', 'What about security?', 'user');
   * console.log(`Message added: ${message.id}`);
   * ```
   */
  async addMessage(
    debateId: string,
    content: string,
    role: string = 'user'
  ): Promise<Message> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/messages`, {
      body: { content, role },
    });
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
   * Add evidence to a debate.
   *
   * @param debateId - The debate ID
   * @param evidence - Evidence content
   * @param source - Optional source of the evidence
   * @param metadata - Optional additional metadata
   *
   * @example
   * ```typescript
   * const result = await client.debates.addEvidence('debate-123', 'Studies show...', 'research-paper');
   * console.log(`Evidence added: ${result.evidence_id}`);
   * ```
   */
  async addEvidence(
    debateId: string,
    evidence: string,
    source?: string,
    metadata?: Record<string, unknown>
  ): Promise<{ evidence_id: string; success: boolean }> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/evidence`, {
      body: { evidence, source, metadata },
    });
  }

  /**
   * Get consensus information for a debate.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const consensus = await client.debates.getConsensus('debate-123');
   * if (consensus.reached) {
   *   console.log(`Consensus: ${consensus.conclusion}`);
   * }
   * ```
   */
  async getConsensus(debateId: string): Promise<{
    reached: boolean;
    conclusion: string | null;
    confidence: number;
    dissent: string[];
  }> {
    return this.client.request('GET', `/api/v1/debates/${debateId}/consensus`);
  }

  /**
   * Fork a debate from a specific point.
   */
  async fork(debateId: string, options?: { branch_point?: number }): Promise<{ debate_id: string }> {
    return this.client.forkDebate(debateId, options);
  }

  /**
   * Clone a debate (create a copy with fresh state).
   *
   * @param debateId - The debate ID to clone
   * @param options - Optional clone options
   *
   * @example
   * ```typescript
   * const cloned = await client.debates.clone('debate-123', { preserveAgents: true });
   * console.log(`Cloned debate: ${cloned.debate_id}`);
   * ```
   */
  async clone(
    debateId: string,
    options?: { preserveAgents?: boolean; preserveContext?: boolean }
  ): Promise<{ debate_id: string }> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/clone`, {
      body: options,
    });
  }

  /**
   * Archive a debate.
   *
   * @param debateId - The debate ID to archive
   *
   * @example
   * ```typescript
   * const result = await client.debates.archive('debate-123');
   * if (result.success) {
   *   console.log('Debate archived');
   * }
   * ```
   */
  async archive(debateId: string): Promise<{ success: boolean }> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/archive`);
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
   * Start a debate.
   *
   * @param debateId - The debate ID to start
   *
   * @example
   * ```typescript
   * const result = await client.debates.start('debate-123');
   * console.log(`Debate started: ${result.status}`);
   * ```
   */
  async start(debateId: string): Promise<{ success: boolean; status: string }> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/start`);
  }

  /**
   * Stop a running debate.
   *
   * @param debateId - The debate ID to stop
   *
   * @example
   * ```typescript
   * const result = await client.debates.stop('debate-123');
   * console.log(`Debate stopped: ${result.status}`);
   * ```
   */
  async stop(debateId: string): Promise<{ success: boolean; status: string }> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/stop`);
  }

  /**
   * Pause a running debate.
   *
   * @param debateId - The debate ID to pause
   *
   * @example
   * ```typescript
   * const result = await client.debates.pause('debate-123');
   * console.log(`Debate paused: ${result.status}`);
   * ```
   */
  async pause(debateId: string): Promise<{ success: boolean; status: string }> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/pause`);
  }

  /**
   * Resume a paused debate.
   *
   * @param debateId - The debate ID to resume
   *
   * @example
   * ```typescript
   * const result = await client.debates.resume('debate-123');
   * console.log(`Debate resumed: ${result.status}`);
   * ```
   */
  async resume(debateId: string): Promise<{ success: boolean; status: string }> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/resume`);
  }

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
  // Rounds, Agents, and Votes
  // ===========================================================================

  /**
   * Get rounds from a debate.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const rounds = await client.debates.getRounds('debate-123');
   * console.log(`${rounds.length} rounds completed`);
   * for (const round of rounds) {
   *   console.log(`Round ${round.number}: ${round.proposals.length} proposals`);
   * }
   * ```
   */
  async getRounds(debateId: string): Promise<Array<{
    number: number;
    proposals: Array<{ agent: string; content: string }>;
    critiques: Array<{ agent: string; target: string; content: string }>;
    status: string;
    started_at: string;
    ended_at?: string;
  }>> {
    const response = await this.client.request<{ rounds: Array<{
      number: number;
      proposals: Array<{ agent: string; content: string }>;
      critiques: Array<{ agent: string; target: string; content: string }>;
      status: string;
      started_at: string;
      ended_at?: string;
    }> }>('GET', `/api/v1/debates/${debateId}/rounds`);
    return response.rounds;
  }

  /**
   * Get agents participating in a debate.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const agents = await client.debates.getAgents('debate-123');
   * for (const agent of agents) {
   *   console.log(`${agent.name}: ${agent.role} (ELO: ${agent.elo})`);
   * }
   * ```
   */
  async getAgents(debateId: string): Promise<Array<{
    name: string;
    role: string;
    model: string;
    elo?: number;
    contributions: number;
  }>> {
    const response = await this.client.request<{ agents: Array<{
      name: string;
      role: string;
      model: string;
      elo?: number;
      contributions: number;
    }> }>('GET', `/api/v1/debates/${debateId}/agents`);
    return response.agents;
  }

  /**
   * Get votes from a debate.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const votes = await client.debates.getVotes('debate-123');
   * console.log(`${votes.length} votes cast`);
   * for (const vote of votes) {
   *   console.log(`${vote.agent} voted for: ${vote.position}`);
   * }
   * ```
   */
  async getVotes(debateId: string): Promise<Array<{
    agent: string;
    position: string;
    confidence: number;
    round: number;
    reasoning?: string;
  }>> {
    const response = await this.client.request<{ votes: Array<{
      agent: string;
      position: string;
      confidence: number;
      round: number;
      reasoning?: string;
    }> }>('GET', `/api/v1/debates/${debateId}/votes`);
    return response.votes;
  }

  /**
   * Add user input to a debate.
   *
   * @param debateId - The debate ID
   * @param input - User input content
   * @param inputType - Type of input (suggestion, vote, question, etc.)
   *
   * @example
   * ```typescript
   * const result = await client.debates.addUserInput('debate-123', 'Consider the scalability aspect', 'suggestion');
   * console.log(`Input added: ${result.input_id}`);
   * ```
   */
  async addUserInput(
    debateId: string,
    input: string,
    inputType: 'suggestion' | 'vote' | 'question' | 'context' = 'suggestion'
  ): Promise<{ input_id: string; success: boolean }> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/user-input`, {
      body: { input, type: inputType },
    });
  }

  /**
   * Get the timeline of events in a debate.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const timeline = await client.debates.getTimeline('debate-123');
   * for (const event of timeline) {
   *   console.log(`${event.timestamp}: ${event.type} - ${event.description}`);
   * }
   * ```
   */
  async getTimeline(debateId: string): Promise<Array<{
    timestamp: string;
    type: string;
    agent?: string;
    description: string;
    round?: number;
    metadata?: Record<string, unknown>;
  }>> {
    const response = await this.client.request<{ timeline: Array<{
      timestamp: string;
      type: string;
      agent?: string;
      description: string;
      round?: number;
      metadata?: Record<string, unknown>;
    }> }>('GET', `/api/v1/debates/${debateId}/timeline`);
    return response.timeline;
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
      params: options as unknown as Record<string, unknown>,
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

  // ===========================================================================
  // Explainability
  // ===========================================================================

  /**
   * Get explainability data for a debate.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const explainability = await client.debates.getExplainability('debate-123');
   * console.log(`Decision explanation: ${explainability.narrative}`);
   * ```
   */
  async getExplainability(debateId: string): Promise<{
    debate_id: string;
    narrative: string;
    factors: Array<{ name: string; weight: number; description: string }>;
    confidence: number;
  }> {
    return this.client.request('GET', `/api/v1/debates/${debateId}/explainability`);
  }

  /**
   * Get factor decomposition for a debate decision.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const factors = await client.debates.getExplainabilityFactors('debate-123');
   * for (const factor of factors.factors) {
   *   console.log(`${factor.name}: ${factor.weight} - ${factor.description}`);
   * }
   * ```
   */
  async getExplainabilityFactors(debateId: string): Promise<{
    factors: Array<{ name: string; weight: number; description: string; evidence: string[] }>;
  }> {
    return this.client.request('GET', `/api/v1/debates/${debateId}/explainability/factors`);
  }

  /**
   * Get natural language narrative explanation.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const narrative = await client.debates.getExplainabilityNarrative('debate-123');
   * console.log(narrative.text);
   * ```
   */
  async getExplainabilityNarrative(debateId: string): Promise<{
    text: string;
    key_points: string[];
    audience_level: string;
  }> {
    return this.client.request('GET', `/api/v1/debates/${debateId}/explainability/narrative`);
  }

  /**
   * Get provenance chain for debate claims.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const provenance = await client.debates.getExplainabilityProvenance('debate-123');
   * for (const claim of provenance.claims) {
   *   console.log(`${claim.text}: ${claim.sources.join(', ')}`);
   * }
   * ```
   */
  async getExplainabilityProvenance(debateId: string): Promise<{
    claims: Array<{ text: string; sources: string[]; confidence: number; agent: string }>;
  }> {
    return this.client.request('GET', `/api/v1/debates/${debateId}/explainability/provenance`);
  }

  /**
   * Get counterfactual analysis.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const counterfactual = await client.debates.getExplainabilityCounterfactual('debate-123');
   * for (const scenario of counterfactual.scenarios) {
   *   console.log(`If ${scenario.condition}, then ${scenario.outcome}`);
   * }
   * ```
   */
  async getExplainabilityCounterfactual(debateId: string): Promise<{
    scenarios: Array<{ condition: string; outcome: string; probability: number }>;
  }> {
    return this.client.request('GET', `/api/v1/debates/${debateId}/explainability/counterfactual`);
  }

  /**
   * Create a counterfactual scenario.
   *
   * @param debateId - The debate ID
   * @param changes - The hypothetical changes to consider
   *
   * @example
   * ```typescript
   * const result = await client.debates.createCounterfactual('debate-123', {
   *   agents: ['claude', 'gpt-4', 'gemini'],
   *   rounds: 5,
   * });
   * console.log(`Counterfactual outcome: ${result.predicted_outcome}`);
   * ```
   */
  async createCounterfactual(
    debateId: string,
    changes: Record<string, unknown>
  ): Promise<{
    predicted_outcome: string;
    confidence: number;
    impact_analysis: Array<{ factor: string; original: unknown; modified: unknown; impact: number }>;
  }> {
    return this.client.request('POST', `/api/v1/debates/${debateId}/explainability/counterfactual`, {
      body: changes,
    });
  }

  // ===========================================================================
  // Red Team & Specialized Debates
  // ===========================================================================

  /**
   * Get red team analysis for a debate.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const redTeam = await client.debates.getRedTeam('debate-123');
   * for (const vulnerability of redTeam.vulnerabilities) {
   *   console.log(`${vulnerability.severity}: ${vulnerability.description}`);
   * }
   * ```
   */
  async getRedTeam(debateId: string): Promise<{
    debate_id: string;
    vulnerabilities: Array<{
      severity: 'low' | 'medium' | 'high' | 'critical';
      description: string;
      recommendation: string;
    }>;
    overall_risk: number;
  }> {
    return this.client.request('GET', `/api/v1/debates/${debateId}/red-team`);
  }

  /**
   * Run a capability probe debate.
   *
   * @param task - The task to probe
   * @param agents - Optional list of agents to use
   *
   * @example
   * ```typescript
   * const probe = await client.debates.capabilityProbe('Can this system handle real-time data?', ['claude', 'gpt-4']);
   * console.log(`Capability assessment: ${probe.assessment}`);
   * ```
   */
  async capabilityProbe(
    task: string,
    agents?: string[]
  ): Promise<{
    debate_id: string;
    assessment: string;
    capabilities: Array<{ name: string; level: number; evidence: string }>;
    gaps: string[];
  }> {
    const data: { task: string; agents?: string[] } = { task };
    if (agents) {
      data.agents = agents;
    }
    return this.client.request('POST', '/api/v1/debates/capability-probe', {
      body: data,
    });
  }

  /**
   * Run a deep audit debate.
   *
   * @param task - The task to audit
   * @param agents - Optional list of agents to use
   *
   * @example
   * ```typescript
   * const audit = await client.debates.deepAudit('Review the security implementation', ['claude', 'gemini']);
   * console.log(`Audit findings: ${audit.findings.length}`);
   * ```
   */
  async deepAudit(
    task: string,
    agents?: string[]
  ): Promise<{
    debate_id: string;
    findings: Array<{ severity: string; description: string; recommendation: string }>;
    compliance_score: number;
    summary: string;
  }> {
    const data: { task: string; agents?: string[] } = { task };
    if (agents) {
      data.agents = agents;
    }
    return this.client.request('POST', '/api/v1/debates/deep-audit', {
      body: data,
    });
  }

  // ===========================================================================
  // Broadcasting & Publishing
  // ===========================================================================

  /**
   * Broadcast debate to channels.
   *
   * @param debateId - The debate ID
   * @param channels - Optional list of channels to broadcast to
   *
   * @example
   * ```typescript
   * const result = await client.debates.broadcast('debate-123', ['slack', 'discord']);
   * console.log(`Broadcasted to ${result.channels_notified.length} channels`);
   * ```
   */
  async broadcast(
    debateId: string,
    channels?: string[]
  ): Promise<{
    success: boolean;
    channels_notified: string[];
  }> {
    const data: { channels?: string[] } = {};
    if (channels) {
      data.channels = channels;
    }
    return this.client.request('POST', `/api/v1/debates/${debateId}/broadcast`, {
      body: data,
    });
  }

  /**
   * Publish debate summary to Twitter.
   *
   * @param debateId - The debate ID
   * @param message - Optional custom message
   *
   * @example
   * ```typescript
   * const result = await client.debates.publishTwitter('debate-123', 'Check out this debate!');
   * console.log(`Tweet posted: ${result.tweet_id}`);
   * ```
   */
  async publishTwitter(
    debateId: string,
    message?: string
  ): Promise<{
    success: boolean;
    tweet_id?: string;
    url?: string;
  }> {
    const data: { message?: string } = {};
    if (message) {
      data.message = message;
    }
    return this.client.request('POST', `/api/v1/debates/${debateId}/publish/twitter`, {
      body: data,
    });
  }

  /**
   * Publish debate to YouTube.
   *
   * @param debateId - The debate ID
   * @param title - Optional custom title
   *
   * @example
   * ```typescript
   * const result = await client.debates.publishYouTube('debate-123', 'AI Debate: Microservices vs Monolith');
   * console.log(`Video published: ${result.video_id}`);
   * ```
   */
  async publishYouTube(
    debateId: string,
    title?: string
  ): Promise<{
    success: boolean;
    video_id?: string;
    url?: string;
  }> {
    const data: { title?: string } = {};
    if (title) {
      data.title = title;
    }
    return this.client.request('POST', `/api/v1/debates/${debateId}/publish/youtube`, {
      body: data,
    });
  }

  // ===========================================================================
  // Dashboard & History
  // ===========================================================================

  /**
   * Get debates dashboard view.
   *
   * @example
   * ```typescript
   * const dashboard = await client.debates.getDashboard();
   * console.log(`Active debates: ${dashboard.active_count}`);
   * console.log(`Completed today: ${dashboard.completed_today}`);
   * ```
   */
  async getDashboard(): Promise<{
    active_count: number;
    completed_today: number;
    pending_count: number;
    recent_debates: Debate[];
    trending_topics: string[];
  }> {
    return this.client.request('GET', '/api/v1/dashboard/debates');
  }

  /**
   * Get graph stats via the alternate debate endpoint.
   *
   * @param debateId - The debate ID
   *
   * @example
   * ```typescript
   * const stats = await client.debates.getDebateGraphStats('debate-123');
   * console.log(`Nodes: ${stats.node_count}, Edges: ${stats.edge_count}`);
   * ```
   */
  async getDebateGraphStats(debateId: string): Promise<GraphStats> {
    return this.client.request('GET', `/api/v1/debate/${debateId}/graph-stats`);
  }

  /**
   * Get debate history.
   *
   * @param limit - Maximum number of debates to return (default 20)
   * @param offset - Number of debates to skip (default 0)
   *
   * @example
   * ```typescript
   * const history = await client.debates.getHistory(10, 0);
   * for (const debate of history.debates) {
   *   console.log(`${debate.task}: ${debate.status}`);
   * }
   * ```
   */
  async getHistory(
    limit: number = 20,
    offset: number = 0
  ): Promise<{ debates: Debate[]; total: number }> {
    return this.client.request('GET', '/api/v1/history/debates', {
      params: { limit, offset },
    });
  }
}
