/**
 * Main client for the Aragora TypeScript SDK.
 *
 * @example
 * ```typescript
 * import { AragoraClient } from '@aragora/client';
 *
 * const client = new AragoraClient('http://localhost:8080');
 *
 * // Run a debate
 * const debate = await client.debates.run('Should we use microservices?');
 * console.log(debate.consensus?.conclusion);
 *
 * // Use control plane
 * const status = await client.controlPlane.getStatus();
 * console.log(status);
 * ```
 */

import { ControlPlaneAPI } from './control-plane';
import type {
  Debate,
  HealthStatus,
  AgentProfile,
  GraphDebate,
  GraphBranch,
  MatrixDebate,
  MatrixConclusion,
  VerificationResult,
  GauntletReceipt,
  TeamSelection,
  SelectionPlugins,
  // Extended Agent types
  AgentRating,
  AgentHistory,
  AgentCalibration,
  AgentConsistency,
  AgentFlipsResponse,
  AgentNetwork,
  AgentMoment,
  AgentPosition,
  AgentDomains,
  AgentPerformance,
  AgentMetadata,
  AgentIntrospection,
  HeadToHeadStats,
  OpponentBriefing,
  AgentComparison,
  Leaderboard,
  AgentHealthStatus,
  RecentFlipsResponse,
  FlipSummary,
  // Calibration types
  CalibrationCurve,
  CalibrationSummary,
  CalibrationLeaderboard,
  CalibrationVisualization,
  // Analytics types
  DisagreementStats,
  RoleRotationStats,
  EarlyStopStats,
  ConsensusQuality,
  CrossPollinationStats,
  LearningEfficiency,
  VotingAccuracy,
  // Memory types
  MemoryItem,
  MemoryRetrieveResponse,
  MemoryTierInfo,
  MemoryTiersResponse,
  MemoryPressure,
  MemorySearchResult,
  Critique,
  CritiqueListResponse,
  ConsolidationResult,
  CleanupResult,
  ArchiveStats,
  // Knowledge types
  KnowledgeEntry,
  KnowledgeSearchResult,
  KnowledgeStats,
  Fact,
  KnowledgeQueryResponse,
  // Workflow types
  Workflow,
  WorkflowStatus,
  WorkflowStep,
  WorkflowTrigger,
  WorkflowExecution,
  ExecutionStatus,
  WorkflowTemplate,
  // Tournament types
  Tournament,
  TournamentStatus,
  TournamentFormat,
  TournamentStandings,
  TournamentBracket,
  TournamentMatch,
  // RBAC types
  Role,
  Permission,
  RoleAssignment,
  // Auth types
  AuthToken,
  User,
  MFASetupResponse,
  MFAVerifyResponse,
  // Codebase types
  CodebaseScanResult,
  CodebaseVulnerability,
  CodebaseDependencyAnalysis,
  CodebaseMetrics,
  // Decisions types
  DecisionType,
  DecisionPriority,
  DecisionConfig,
  DecisionContext,
  ResponseChannel,
  DecisionResult,
  DecisionStatusResponse,
  // Documents types
  Document,
  DocumentUploadResponse,
  SupportedFormats,
  BatchUploadResponse,
  BatchJobStatus,
  BatchJobResults,
  ProcessingStats,
  DocumentChunk,
  DocumentContext,
  AuditSession,
  AuditSessionCreateResponse,
  AuditFinding,
  AuditReport,
  // Policies types
  Policy,
  PolicyRule,
  PolicyViolation,
  ComplianceCheckResult,
  ComplianceStats,
} from './types';

export interface AragoraClientOptions {
  apiKey?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

export class AragoraError extends Error {
  code?: string;
  status?: number;
  details?: Record<string, unknown>;

  constructor(
    message: string,
    options: { code?: string; status?: number; details?: Record<string, unknown> } = {}
  ) {
    super(message);
    this.name = 'AragoraError';
    this.code = options.code;
    this.status = options.status;
    this.details = options.details;
  }
}

export class DebatesAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Create a new debate.
   */
  async create(options: {
    task: string;
    agents?: string[];
    maxRounds?: number;
    consensusThreshold?: number;
    metadata?: Record<string, unknown>;
  }): Promise<{ id: string }> {
    return this.client.post<{ id: string }>('/api/v1/debates', {
      task: options.task,
      agents: options.agents,
      max_rounds: options.maxRounds ?? 5,
      consensus_threshold: options.consensusThreshold ?? 0.8,
      metadata: options.metadata ?? {},
    });
  }

  /**
   * Get a debate by ID.
   */
  async get(debateId: string): Promise<Debate> {
    return this.client.get<Debate>(`/api/v1/debates/${debateId}`);
  }

  /**
   * List debates.
   */
  async list(options: {
    limit?: number;
    offset?: number;
    status?: string;
  } = {}): Promise<Debate[]> {
    const data = await this.client.get<{ debates: Debate[] }>('/api/v1/debates', {
      limit: options.limit ?? 50,
      offset: options.offset ?? 0,
      ...(options.status && { status: options.status }),
    });
    return data.debates ?? [];
  }

  /**
   * Run a debate and wait for completion.
   */
  async run(
    task: string,
    options: {
      agents?: string[];
      maxRounds?: number;
      consensusThreshold?: number;
      pollInterval?: number;
      timeout?: number;
    } = {}
  ): Promise<Debate> {
    const response = await this.create({
      task,
      agents: options.agents,
      maxRounds: options.maxRounds ?? 5,
      consensusThreshold: options.consensusThreshold ?? 0.8,
    });

    const debateId = response.id;
    const pollInterval = options.pollInterval ?? 1000;
    const timeout = options.timeout ?? 300000;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const debate = await this.get(debateId);
      if (['completed', 'failed', 'cancelled'].includes(debate.status)) {
        return debate;
      }
      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    throw new AragoraError(`Debate ${debateId} did not complete within ${timeout}ms`);
  }

  /**
   * Update a debate's metadata or configuration.
   */
  async update(debateId: string, options: {
    metadata?: Record<string, unknown>;
    status?: string;
    tags?: string[];
  }): Promise<Debate> {
    return this.client.post<Debate>(`/api/v1/debates/${debateId}`, {
      ...(options.metadata && { metadata: options.metadata }),
      ...(options.status && { status: options.status }),
      ...(options.tags && { tags: options.tags }),
    });
  }

  /**
   * Verify a debate's integrity and consensus.
   */
  async verify(debateId: string): Promise<{ valid: boolean; issues: string[]; hash: string }> {
    return this.client.get(`/api/v1/debates/${debateId}/verify`);
  }

  /**
   * Get a detailed verification report for a debate.
   */
  async verificationReport(debateId: string): Promise<{
    debate_id: string;
    verified_at: string;
    consensus_valid: boolean;
    message_integrity: boolean;
    agent_signatures: Array<{ agent: string; valid: boolean }>;
    provenance_chain: Array<{ step: string; hash: string; valid: boolean }>;
    overall_valid: boolean;
  }> {
    return this.client.get(`/api/v1/debates/${debateId}/verification-report`);
  }

  /**
   * Search debates by query string.
   */
  async search(query: string, options: {
    limit?: number;
    offset?: number;
    status?: string;
    agents?: string[];
    dateFrom?: string;
    dateTo?: string;
  } = {}): Promise<{ debates: Debate[]; total: number }> {
    return this.client.get('/api/v1/debates/search', {
      q: query,
      limit: options.limit ?? 20,
      offset: options.offset ?? 0,
      ...(options.status && { status: options.status }),
      ...(options.agents && { agents: options.agents.join(',') }),
      ...(options.dateFrom && { date_from: options.dateFrom }),
      ...(options.dateTo && { date_to: options.dateTo }),
    });
  }

  /**
   * Fork a debate to create a new branch with different parameters.
   */
  async fork(debateId: string, options: {
    fromRound?: number;
    newTask?: string;
    agents?: string[];
    maxRounds?: number;
  } = {}): Promise<{ id: string; forked_from: string; fork_point: number }> {
    return this.client.post(`/api/v1/debates/${debateId}/fork`, {
      ...(options.fromRound !== undefined && { from_round: options.fromRound }),
      ...(options.newTask && { new_task: options.newTask }),
      ...(options.agents && { agents: options.agents }),
      ...(options.maxRounds !== undefined && { max_rounds: options.maxRounds }),
    });
  }

  /**
   * Cancel a running debate.
   */
  async cancel(debateId: string, options: { reason?: string } = {}): Promise<{ cancelled: boolean }> {
    return this.client.post(`/api/v1/debates/${debateId}/cancel`, {
      ...(options.reason && { reason: options.reason }),
    });
  }

  /**
   * Get debate statistics.
   */
  async stats(debateId: string): Promise<{
    debate_id: string;
    total_rounds: number;
    total_messages: number;
    consensus_reached: boolean;
    avg_message_length: number;
    agent_participation: Record<string, number>;
    duration_seconds: number;
  }> {
    return this.client.get(`/api/v1/debates/${debateId}/stats`);
  }
}

export class AgentsAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * List all available agents.
   */
  async list(options: { includeStats?: boolean } = {}): Promise<AgentProfile[]> {
    const data = await this.client.get<{ agents: AgentProfile[] }>('/api/v1/agents', {
      include_stats: options.includeStats ?? false,
    });
    return data.agents ?? [];
  }

  /**
   * Get an agent by name/ID.
   */
  async get(agentId: string): Promise<AgentProfile> {
    return this.client.get<AgentProfile>(`/api/v1/agents/${agentId}`);
  }

  /**
   * Get agent profile with rating data.
   */
  async profile(agentName: string): Promise<AgentRating> {
    return this.client.get<AgentRating>(`/api/v1/agent/${agentName}/profile`);
  }

  /**
   * Get agent ELO rating history.
   */
  async history(agentName: string, options: { limit?: number } = {}): Promise<AgentHistory> {
    return this.client.get<AgentHistory>(`/api/v1/agent/${agentName}/history`, {
      limit: options.limit ?? 30,
    });
  }

  /**
   * Get agent calibration scores.
   */
  async calibration(agentName: string, options: { domain?: string } = {}): Promise<AgentCalibration> {
    return this.client.get<AgentCalibration>(`/api/v1/agent/${agentName}/calibration`, {
      ...(options.domain && { domain: options.domain }),
    });
  }

  /**
   * Get agent consistency score.
   */
  async consistency(agentName: string): Promise<AgentConsistency> {
    return this.client.get<AgentConsistency>(`/api/v1/agent/${agentName}/consistency`);
  }

  /**
   * Get agent position flip history.
   */
  async flips(agentName: string, options: { limit?: number } = {}): Promise<AgentFlipsResponse> {
    return this.client.get<AgentFlipsResponse>(`/api/v1/agent/${agentName}/flips`, {
      limit: options.limit ?? 20,
    });
  }

  /**
   * Get agent relationship network.
   */
  async network(agentName: string): Promise<AgentNetwork> {
    return this.client.get<AgentNetwork>(`/api/v1/agent/${agentName}/network`);
  }

  /**
   * Get agent's top rivals.
   */
  async rivals(agentName: string, options: { limit?: number } = {}): Promise<{ agent: string; rivals: Array<{ agent: string; matches: number; win_rate: number }> }> {
    return this.client.get(`/api/v1/agent/${agentName}/rivals`, {
      limit: options.limit ?? 5,
    });
  }

  /**
   * Get agent's top allies.
   */
  async allies(agentName: string, options: { limit?: number } = {}): Promise<{ agent: string; allies: Array<{ agent: string; matches: number; agreement_rate: number }> }> {
    return this.client.get(`/api/v1/agent/${agentName}/allies`, {
      limit: options.limit ?? 5,
    });
  }

  /**
   * Get agent's significant moments.
   */
  async moments(agentName: string, options: { limit?: number } = {}): Promise<{ agent: string; moments: AgentMoment[] }> {
    return this.client.get(`/api/v1/agent/${agentName}/moments`, {
      limit: options.limit ?? 10,
    });
  }

  /**
   * Get agent's position history.
   */
  async positions(agentName: string, options: { limit?: number } = {}): Promise<{ agent: string; positions: AgentPosition[] }> {
    return this.client.get(`/api/v1/agent/${agentName}/positions`, {
      limit: options.limit ?? 20,
    });
  }

  /**
   * Get agent's domain-specific ELO ratings.
   */
  async domains(agentName: string): Promise<AgentDomains> {
    return this.client.get<AgentDomains>(`/api/v1/agent/${agentName}/domains`);
  }

  /**
   * Get detailed agent performance statistics.
   */
  async performance(agentName: string): Promise<AgentPerformance> {
    return this.client.get<AgentPerformance>(`/api/v1/agent/${agentName}/performance`);
  }

  /**
   * Get agent metadata (provider, model, capabilities).
   */
  async metadata(agentName: string): Promise<AgentMetadata> {
    return this.client.get<AgentMetadata>(`/api/v1/agent/${agentName}/metadata`);
  }

  /**
   * Get agent introspection data for self-awareness and debugging.
   */
  async introspect(agentName: string, options: { debateId?: string } = {}): Promise<AgentIntrospection> {
    return this.client.get<AgentIntrospection>(`/api/v1/agent/${agentName}/introspect`, {
      ...(options.debateId && { debate_id: options.debateId }),
    });
  }

  /**
   * Get head-to-head statistics between two agents.
   */
  async headToHead(agentName: string, opponentName: string): Promise<HeadToHeadStats> {
    return this.client.get<HeadToHeadStats>(`/api/v1/agent/${agentName}/head-to-head/${opponentName}`);
  }

  /**
   * Get strategic briefing about an opponent for an agent.
   */
  async opponentBriefing(agentName: string, opponentName: string): Promise<OpponentBriefing> {
    return this.client.get<OpponentBriefing>(`/api/v1/agent/${agentName}/opponent-briefing/${opponentName}`);
  }

  /**
   * Compare multiple agents.
   */
  async compare(agents: string[]): Promise<AgentComparison> {
    return this.client.get<AgentComparison>('/api/v1/agent/compare', {
      agents: agents.join(','),
    });
  }

  /**
   * Get agent leaderboard/rankings.
   */
  async leaderboard(options: { limit?: number; domain?: string } = {}): Promise<Leaderboard> {
    return this.client.get<Leaderboard>('/api/v1/leaderboard', {
      limit: options.limit ?? 20,
      ...(options.domain && { domain: options.domain }),
    });
  }

  /**
   * Get rankings (alias for leaderboard).
   */
  async rankings(options: { limit?: number; domain?: string } = {}): Promise<Leaderboard> {
    return this.leaderboard(options);
  }

  /**
   * Get runtime health status for all agents.
   */
  async health(): Promise<AgentHealthStatus> {
    return this.client.get<AgentHealthStatus>('/api/v1/agents/health');
  }

  /**
   * Get recent matches.
   */
  async recentMatches(options: { limit?: number; loopId?: string } = {}): Promise<{ matches: Array<Record<string, unknown>> }> {
    return this.client.get('/api/v1/matches/recent', {
      limit: options.limit ?? 10,
      ...(options.loopId && { loop_id: options.loopId }),
    });
  }

  /**
   * Get recent flips across all agents.
   */
  async recentFlips(options: { limit?: number } = {}): Promise<RecentFlipsResponse> {
    return this.client.get<RecentFlipsResponse>('/api/v1/flips/recent', {
      limit: options.limit ?? 20,
    });
  }

  /**
   * Get flip summary for dashboard.
   */
  async flipSummary(): Promise<FlipSummary> {
    return this.client.get<FlipSummary>('/api/v1/flips/summary');
  }

  // ==========================================================================
  // Calibration methods
  // ==========================================================================

  /**
   * Get calibration curve (confidence vs accuracy per bucket).
   */
  async calibrationCurve(agentName: string, options: { buckets?: number; domain?: string } = {}): Promise<CalibrationCurve> {
    return this.client.get<CalibrationCurve>(`/api/v1/agent/${agentName}/calibration-curve`, {
      buckets: options.buckets ?? 10,
      ...(options.domain && { domain: options.domain }),
    });
  }

  /**
   * Get comprehensive calibration summary for an agent.
   */
  async calibrationSummary(agentName: string, options: { domain?: string } = {}): Promise<CalibrationSummary> {
    return this.client.get<CalibrationSummary>(`/api/v1/agent/${agentName}/calibration-summary`, {
      ...(options.domain && { domain: options.domain }),
    });
  }

  /**
   * Get calibration leaderboard.
   */
  async calibrationLeaderboard(options: { limit?: number; metric?: 'brier' | 'ece' | 'accuracy' | 'composite'; minPredictions?: number } = {}): Promise<CalibrationLeaderboard> {
    return this.client.get<CalibrationLeaderboard>('/api/v1/calibration/leaderboard', {
      limit: options.limit ?? 20,
      metric: options.metric ?? 'brier',
      min_predictions: options.minPredictions ?? 5,
    });
  }

  /**
   * Get comprehensive calibration visualization data.
   */
  async calibrationVisualization(options: { limit?: number } = {}): Promise<CalibrationVisualization> {
    return this.client.get<CalibrationVisualization>('/api/v1/calibration/visualization', {
      limit: options.limit ?? 5,
    });
  }
}

export class GraphDebatesAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Create a new graph debate.
   */
  async create(options: {
    task: string;
    agents?: string[];
    maxRounds?: number;
    branchThreshold?: number;
    maxBranches?: number;
  }): Promise<{ id: string }> {
    return this.client.post<{ id: string }>('/api/v1/graph-debates', {
      task: options.task,
      agents: options.agents,
      max_rounds: options.maxRounds ?? 5,
      branch_threshold: options.branchThreshold ?? 0.5,
      max_branches: options.maxBranches ?? 10,
    });
  }

  /**
   * Get a graph debate by ID.
   */
  async get(debateId: string): Promise<GraphDebate> {
    return this.client.get<GraphDebate>(`/api/v1/graph-debates/${debateId}`);
  }

  /**
   * Get branches for a graph debate.
   */
  async getBranches(debateId: string): Promise<GraphBranch[]> {
    const data = await this.client.get<{ branches: GraphBranch[] }>(
      `/api/v1/graph-debates/${debateId}/branches`
    );
    return data.branches ?? [];
  }
}

export class MatrixDebatesAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Create a new matrix debate.
   */
  async create(options: {
    task: string;
    scenarios: Array<{ name: string; parameters: Record<string, unknown>; weight?: number }>;
    agents?: string[];
    maxRounds?: number;
  }): Promise<{ id: string }> {
    return this.client.post<{ id: string }>('/api/v1/matrix-debates', {
      task: options.task,
      scenarios: options.scenarios,
      agents: options.agents,
      max_rounds: options.maxRounds ?? 3,
    });
  }

  /**
   * Get a matrix debate by ID.
   */
  async get(debateId: string): Promise<MatrixDebate> {
    return this.client.get<MatrixDebate>(`/api/v1/matrix-debates/${debateId}`);
  }

  /**
   * Get conclusions for a matrix debate.
   */
  async getConclusions(debateId: string): Promise<MatrixConclusion> {
    return this.client.get<MatrixConclusion>(
      `/api/v1/matrix-debates/${debateId}/conclusions`
    );
  }
}

export class VerificationAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Verify a claim using multi-agent deliberation.
   */
  async verifyClaim(options: {
    claim: string;
    context?: string;
    evidenceSources?: string[];
    minConfidence?: number;
  }): Promise<VerificationResult> {
    return this.client.post<VerificationResult>('/api/v1/verification/claim', {
      claim: options.claim,
      context: options.context,
      evidence_sources: options.evidenceSources,
      min_confidence: options.minConfidence ?? 0.7,
    });
  }

  /**
   * Get a verification result by claim ID.
   */
  async get(claimId: string): Promise<VerificationResult> {
    return this.client.get<VerificationResult>(`/api/v1/verification/${claimId}`);
  }
}

export class GauntletAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Run an agent through the gauntlet certification process.
   */
  async run(options: {
    agentId: string;
    challengeTypes?: string[];
    minPassScore?: number;
  }): Promise<GauntletReceipt> {
    return this.client.post<GauntletReceipt>('/api/v1/gauntlet/run', {
      agent_id: options.agentId,
      challenge_types: options.challengeTypes,
      min_pass_score: options.minPassScore ?? 0.7,
    });
  }

  /**
   * Get a gauntlet receipt by ID.
   */
  async get(receiptId: string): Promise<GauntletReceipt> {
    return this.client.get<GauntletReceipt>(`/api/v1/gauntlet/${receiptId}`);
  }

  /**
   * List gauntlet receipts.
   */
  async list(options: { limit?: number; agentId?: string } = {}): Promise<{ receipts: GauntletReceipt[]; count: number }> {
    return this.client.get('/api/v1/gauntlet', {
      limit: options.limit ?? 50,
      ...(options.agentId && { agent_id: options.agentId }),
    });
  }

  /**
   * Delete a gauntlet receipt.
   */
  async delete(receiptId: string): Promise<void> {
    return this.client.delete(`/api/v1/gauntlet/${receiptId}`);
  }
}

export class AnalyticsAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Get statistics about debate disagreements.
   */
  async disagreements(): Promise<DisagreementStats> {
    return this.client.get<DisagreementStats>('/api/v1/analytics/disagreements');
  }

  /**
   * Get statistics about cognitive role rotation.
   */
  async roleRotation(): Promise<RoleRotationStats> {
    return this.client.get<RoleRotationStats>('/api/v1/analytics/role-rotation');
  }

  /**
   * Get statistics about early debate stopping.
   */
  async earlyStops(): Promise<EarlyStopStats> {
    return this.client.get<EarlyStopStats>('/api/v1/analytics/early-stops');
  }

  /**
   * Get consensus quality monitoring metrics.
   */
  async consensusQuality(): Promise<ConsensusQuality> {
    return this.client.get<ConsensusQuality>('/api/v1/analytics/consensus-quality');
  }

  /**
   * Get ranking system statistics.
   */
  async rankingStats(): Promise<{ stats: Record<string, unknown> }> {
    return this.client.get('/api/v1/ranking/stats');
  }

  /**
   * Get memory system statistics.
   */
  async memoryStats(): Promise<{ stats: Record<string, unknown> }> {
    return this.client.get('/api/v1/memory/stats');
  }

  /**
   * Get aggregate cross-pollination statistics.
   */
  async crossPollination(): Promise<CrossPollinationStats> {
    return this.client.get<CrossPollinationStats>('/api/v1/analytics/cross-pollination');
  }

  /**
   * Get learning efficiency statistics for agents.
   */
  async learningEfficiency(options: { agent?: string; domain?: string; limit?: number } = {}): Promise<LearningEfficiency> {
    return this.client.get<LearningEfficiency>('/api/v1/analytics/learning-efficiency', {
      ...(options.agent && { agent: options.agent }),
      ...(options.domain && { domain: options.domain }),
      limit: options.limit ?? 20,
    });
  }

  /**
   * Get voting accuracy statistics for agents.
   */
  async votingAccuracy(options: { agent?: string; limit?: number } = {}): Promise<VotingAccuracy> {
    return this.client.get<VotingAccuracy>('/api/v1/analytics/voting-accuracy', {
      ...(options.agent && { agent: options.agent }),
      limit: options.limit ?? 20,
    });
  }

  /**
   * Get calibration statistics for agents.
   */
  async calibration(options: { agent?: string; limit?: number } = {}): Promise<{ agent?: string; agents?: Array<{ agent: string; calibration: Record<string, unknown> | null }> }> {
    return this.client.get('/api/v1/analytics/calibration', {
      ...(options.agent && { agent: options.agent }),
      limit: options.limit ?? 20,
    });
  }
}

export class TeamSelectionAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Select an optimal team of agents for a task.
   */
  async selectTeam(options: {
    task: string;
    teamSize?: number;
    requiredCapabilities?: string[];
    excludeAgents?: string[];
    plugins?: SelectionPlugins;
  }): Promise<TeamSelection> {
    return this.client.post<TeamSelection>('/api/v1/team-selection', {
      task: options.task,
      team_size: options.teamSize ?? 3,
      required_capabilities: options.requiredCapabilities,
      exclude_agents: options.excludeAgents,
      plugins: options.plugins,
    });
  }
}

export class MemoryAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Retrieve memories from the continuum memory system.
   */
  async retrieve(options: {
    query?: string;
    tiers?: ('fast' | 'medium' | 'slow' | 'glacial')[];
    limit?: number;
    minImportance?: number;
  } = {}): Promise<MemoryRetrieveResponse> {
    return this.client.get<MemoryRetrieveResponse>('/api/v1/memory/continuum/retrieve', {
      ...(options.query && { query: options.query }),
      ...(options.tiers && { tiers: options.tiers.join(',') }),
      limit: options.limit ?? 10,
      min_importance: options.minImportance ?? 0,
    });
  }

  /**
   * Get all memory tiers with detailed statistics.
   */
  async tiers(): Promise<MemoryTiersResponse> {
    return this.client.get<MemoryTiersResponse>('/api/v1/memory/tiers');
  }

  /**
   * Get tier statistics (summary).
   */
  async tierStats(): Promise<{ tiers: Record<string, { count: number; total_size: number; avg_age: number }> }> {
    return this.client.get('/api/v1/memory/tier-stats');
  }

  /**
   * Get archive statistics.
   */
  async archiveStats(): Promise<ArchiveStats> {
    return this.client.get<ArchiveStats>('/api/v1/memory/archive-stats');
  }

  /**
   * Get memory pressure and utilization.
   */
  async pressure(): Promise<MemoryPressure> {
    return this.client.get<MemoryPressure>('/api/v1/memory/pressure');
  }

  /**
   * Search memories across tiers.
   */
  async search(query: string, options: {
    tiers?: ('fast' | 'medium' | 'slow' | 'glacial')[];
    limit?: number;
    minScore?: number;
  } = {}): Promise<MemorySearchResult> {
    return this.client.get<MemorySearchResult>('/api/v1/memory/search', {
      query,
      ...(options.tiers && { tiers: options.tiers.join(',') }),
      limit: options.limit ?? 20,
      ...(options.minScore !== undefined && { min_score: options.minScore }),
    });
  }

  /**
   * Browse critique store entries.
   */
  async critiques(options: {
    debateId?: string;
    agent?: string;
    limit?: number;
    offset?: number;
  } = {}): Promise<CritiqueListResponse> {
    return this.client.get<CritiqueListResponse>('/api/v1/memory/critiques', {
      ...(options.debateId && { debate_id: options.debateId }),
      ...(options.agent && { agent: options.agent }),
      limit: options.limit ?? 50,
      offset: options.offset ?? 0,
    });
  }

  /**
   * Trigger memory consolidation (requires auth).
   */
  async consolidate(): Promise<ConsolidationResult> {
    return this.client.post<ConsolidationResult>('/api/v1/memory/continuum/consolidate', {});
  }

  /**
   * Cleanup expired memories (requires auth).
   */
  async cleanup(options: { maxAge?: number } = {}): Promise<CleanupResult> {
    return this.client.post<CleanupResult>('/api/v1/memory/continuum/cleanup', {
      ...(options.maxAge !== undefined && { max_age: options.maxAge }),
    });
  }
}

// =============================================================================
// Knowledge API
// =============================================================================

export class KnowledgeAPI {
  constructor(private client: AragoraClient) {}

  /**
   * Search the knowledge base.
   */
  async search(
    query: string,
    options: {
      limit?: number;
      minScore?: number;
      source?: string;
      tags?: string[];
    } = {}
  ): Promise<{ results: KnowledgeSearchResult[] }> {
    return this.client.get<{ results: KnowledgeSearchResult[] }>('/api/v1/knowledge/search', {
      query,
      limit: options.limit ?? 10,
      ...(options.minScore !== undefined && { min_score: options.minScore }),
      ...(options.source && { source: options.source }),
      ...(options.tags && { tags: options.tags.join(',') }),
    });
  }

  /**
   * Query the knowledge base with natural language.
   */
  async query(
    question: string,
    options: { context?: string; includeSources?: boolean } = {}
  ): Promise<KnowledgeQueryResponse> {
    return this.client.post<KnowledgeQueryResponse>('/api/v1/knowledge/query', {
      question,
      include_sources: options.includeSources ?? true,
      ...(options.context && { context: options.context }),
    });
  }

  /**
   * Add an entry to the knowledge base.
   */
  async add(options: {
    content: string;
    source?: string;
    sourceType?: string;
    metadata?: Record<string, unknown>;
    tags?: string[];
    confidence?: number;
  }): Promise<{ id: string; created_at: string }> {
    return this.client.post<{ id: string; created_at: string }>('/api/v1/knowledge', {
      content: options.content,
      ...(options.source && { source: options.source }),
      ...(options.sourceType && { source_type: options.sourceType }),
      ...(options.metadata && { metadata: options.metadata }),
      ...(options.tags && { tags: options.tags }),
      ...(options.confidence !== undefined && { confidence: options.confidence }),
    });
  }

  /**
   * Get a knowledge entry by ID.
   */
  async get(entryId: string): Promise<KnowledgeEntry> {
    return this.client.get<KnowledgeEntry>(`/api/v1/knowledge/${entryId}`);
  }

  /**
   * Delete a knowledge entry.
   */
  async delete(entryId: string): Promise<void> {
    return this.client.delete(`/api/v1/knowledge/${entryId}`);
  }

  /**
   * List facts from the knowledge base.
   */
  async listFacts(options: {
    limit?: number;
    offset?: number;
    verified?: boolean;
    source?: string;
  } = {}): Promise<{ facts: Fact[]; count: number }> {
    return this.client.get<{ facts: Fact[]; count: number }>('/api/v1/knowledge/facts', {
      limit: options.limit ?? 50,
      offset: options.offset ?? 0,
      ...(options.verified !== undefined && { verified: options.verified }),
      ...(options.source && { source: options.source }),
    });
  }

  /**
   * Add a fact to the knowledge base.
   */
  async addFact(options: {
    content: string;
    source?: string;
    confidence?: number;
    metadata?: Record<string, unknown>;
  }): Promise<Fact> {
    return this.client.post<Fact>('/api/v1/knowledge/facts', {
      content: options.content,
      ...(options.source && { source: options.source }),
      ...(options.confidence !== undefined && { confidence: options.confidence }),
      ...(options.metadata && { metadata: options.metadata }),
    });
  }

  /**
   * Verify a fact using agents.
   */
  async verifyFact(
    factId: string,
    options: { agents?: string[] } = {}
  ): Promise<{ verified: boolean; confidence: number; reasoning: string }> {
    return this.client.post<{ verified: boolean; confidence: number; reasoning: string }>(
      `/api/v1/knowledge/facts/${factId}/verify`,
      {
        ...(options.agents && { agents: options.agents }),
      }
    );
  }

  /**
   * Get facts that contradict the given fact.
   */
  async getContradictions(factId: string): Promise<{ contradictions: Fact[] }> {
    return this.client.get<{ contradictions: Fact[] }>(`/api/v1/knowledge/facts/${factId}/contradictions`);
  }

  /**
   * Get knowledge base statistics.
   */
  async stats(): Promise<KnowledgeStats> {
    return this.client.get<KnowledgeStats>('/api/v1/knowledge/stats');
  }

  /**
   * Bulk import knowledge entries.
   */
  async bulkImport(
    entries: Array<{
      content: string;
      source?: string;
      tags?: string[];
      metadata?: Record<string, unknown>;
    }>,
    options: { skipDuplicates?: boolean } = {}
  ): Promise<{ imported: number; skipped: number; errors: number }> {
    return this.client.post<{ imported: number; skipped: number; errors: number }>('/api/v1/knowledge/bulk-import', {
      entries,
      skip_duplicates: options.skipDuplicates ?? true,
    });
  }
}

// =============================================================================
// Workflows API
// =============================================================================

export class WorkflowsAPI {
  constructor(private client: AragoraClient) {}

  /**
   * List workflows.
   */
  async list(options: { limit?: number; offset?: number; status?: WorkflowStatus } = {}): Promise<{
    workflows: Workflow[];
    count: number;
  }> {
    return this.client.get<{ workflows: Workflow[]; count: number }>('/api/v1/workflows', {
      limit: options.limit ?? 50,
      offset: options.offset ?? 0,
      ...(options.status && { status: options.status }),
    });
  }

  /**
   * Get a workflow by ID.
   */
  async get(workflowId: string): Promise<Workflow> {
    return this.client.get<Workflow>(`/api/v1/workflows/${workflowId}`);
  }

  /**
   * Create a new workflow.
   */
  async create(options: {
    name: string;
    description?: string;
    steps: WorkflowStep[];
    triggers?: WorkflowTrigger[];
    metadata?: Record<string, unknown>;
  }): Promise<Workflow> {
    return this.client.post<Workflow>('/api/v1/workflows', {
      name: options.name,
      ...(options.description && { description: options.description }),
      steps: options.steps,
      ...(options.triggers && { triggers: options.triggers }),
      ...(options.metadata && { metadata: options.metadata }),
    });
  }

  /**
   * Delete a workflow.
   */
  async delete(workflowId: string): Promise<void> {
    return this.client.delete(`/api/v1/workflows/${workflowId}`);
  }

  /**
   * Execute a workflow.
   */
  async execute(
    workflowId: string,
    options: { inputs?: Record<string, unknown> } = {}
  ): Promise<WorkflowExecution> {
    return this.client.post<WorkflowExecution>(`/api/v1/workflows/${workflowId}/execute`, {
      ...(options.inputs && { inputs: options.inputs }),
    });
  }

  /**
   * Get a workflow execution.
   */
  async getExecution(executionId: string): Promise<WorkflowExecution> {
    return this.client.get<WorkflowExecution>(`/api/v1/workflows/executions/${executionId}`);
  }

  /**
   * List workflow executions.
   */
  async listExecutions(options: {
    workflowId?: string;
    status?: ExecutionStatus;
    limit?: number;
    offset?: number;
  } = {}): Promise<{ executions: WorkflowExecution[]; count: number }> {
    return this.client.get<{ executions: WorkflowExecution[]; count: number }>('/api/v1/workflows/executions', {
      ...(options.workflowId && { workflow_id: options.workflowId }),
      ...(options.status && { status: options.status }),
      limit: options.limit ?? 50,
      offset: options.offset ?? 0,
    });
  }

  /**
   * Cancel a workflow execution.
   */
  async cancelExecution(executionId: string): Promise<WorkflowExecution> {
    return this.client.post<WorkflowExecution>(`/api/v1/workflows/executions/${executionId}/cancel`, {});
  }

  /**
   * List workflow templates.
   */
  async listTemplates(options: { category?: string } = {}): Promise<{ templates: WorkflowTemplate[] }> {
    return this.client.get<{ templates: WorkflowTemplate[] }>('/api/v1/workflows/templates', {
      ...(options.category && { category: options.category }),
    });
  }

  /**
   * Get a workflow template.
   */
  async getTemplate(templateId: string): Promise<WorkflowTemplate> {
    return this.client.get<WorkflowTemplate>(`/api/v1/workflows/templates/${templateId}`);
  }

  /**
   * Run a workflow from a template.
   */
  async runTemplate(
    templateId: string,
    options: { parameters?: Record<string, unknown> } = {}
  ): Promise<WorkflowExecution> {
    return this.client.post<WorkflowExecution>(`/api/v1/workflows/templates/${templateId}/run`, {
      ...(options.parameters && { parameters: options.parameters }),
    });
  }
}

// =============================================================================
// Tournaments API
// =============================================================================

export class TournamentsAPI {
  constructor(private client: AragoraClient) {}

  /**
   * List tournaments.
   */
  async list(options: { limit?: number; offset?: number; status?: TournamentStatus } = {}): Promise<{
    tournaments: Tournament[];
    count: number;
  }> {
    return this.client.get<{ tournaments: Tournament[]; count: number }>('/api/v1/tournaments', {
      limit: options.limit ?? 50,
      offset: options.offset ?? 0,
      ...(options.status && { status: options.status }),
    });
  }

  /**
   * Get a tournament by ID.
   */
  async get(tournamentId: string): Promise<Tournament> {
    return this.client.get<Tournament>(`/api/v1/tournaments/${tournamentId}`);
  }

  /**
   * Create a new tournament.
   */
  async create(options: {
    name: string;
    description?: string;
    format: TournamentFormat;
    topic: string;
    participants: string[];
    metadata?: Record<string, unknown>;
  }): Promise<Tournament> {
    return this.client.post<Tournament>('/api/v1/tournaments', {
      name: options.name,
      ...(options.description && { description: options.description }),
      format: options.format,
      topic: options.topic,
      participants: options.participants,
      ...(options.metadata && { metadata: options.metadata }),
    });
  }

  /**
   * Start a tournament.
   */
  async start(tournamentId: string): Promise<Tournament> {
    return this.client.post<Tournament>(`/api/v1/tournaments/${tournamentId}/start`, {});
  }

  /**
   * Cancel a tournament.
   */
  async cancel(tournamentId: string): Promise<Tournament> {
    return this.client.post<Tournament>(`/api/v1/tournaments/${tournamentId}/cancel`, {});
  }

  /**
   * Get tournament standings.
   */
  async standings(tournamentId: string): Promise<TournamentStandings> {
    return this.client.get<TournamentStandings>(`/api/v1/tournaments/${tournamentId}/standings`);
  }

  /**
   * Get tournament bracket.
   */
  async bracket(tournamentId: string): Promise<TournamentBracket> {
    return this.client.get<TournamentBracket>(`/api/v1/tournaments/${tournamentId}/bracket`);
  }

  /**
   * List tournament matches.
   */
  async listMatches(
    tournamentId: string,
    options: { round?: number } = {}
  ): Promise<{ matches: TournamentMatch[] }> {
    return this.client.get<{ matches: TournamentMatch[] }>(`/api/v1/tournaments/${tournamentId}/matches`, {
      ...(options.round !== undefined && { round: options.round }),
    });
  }

  /**
   * Get a specific match.
   */
  async getMatch(tournamentId: string, matchId: string): Promise<TournamentMatch> {
    return this.client.get<TournamentMatch>(`/api/v1/tournaments/${tournamentId}/matches/${matchId}`);
  }

  /**
   * Delete a tournament.
   */
  async delete(tournamentId: string): Promise<void> {
    return this.client.delete(`/api/v1/tournaments/${tournamentId}`);
  }
}

// =============================================================================
// RBAC API
// =============================================================================

export class RBACAPI {
  constructor(private client: AragoraClient) {}

  /**
   * List roles.
   */
  async listRoles(options: { limit?: number; offset?: number } = {}): Promise<{ roles: Role[]; count: number }> {
    return this.client.get<{ roles: Role[]; count: number }>('/api/v1/rbac/roles', {
      limit: options.limit ?? 50,
      offset: options.offset ?? 0,
    });
  }

  /**
   * Get a role by ID.
   */
  async getRole(roleId: string): Promise<Role> {
    return this.client.get<Role>(`/api/v1/rbac/roles/${roleId}`);
  }

  /**
   * Create a new role.
   */
  async createRole(options: {
    name: string;
    description?: string;
    permissions: string[];
  }): Promise<Role> {
    return this.client.post<Role>('/api/v1/rbac/roles', {
      name: options.name,
      ...(options.description && { description: options.description }),
      permissions: options.permissions,
    });
  }

  /**
   * Delete a role.
   */
  async deleteRole(roleId: string): Promise<void> {
    return this.client.delete(`/api/v1/rbac/roles/${roleId}`);
  }

  /**
   * List all permissions.
   */
  async listPermissions(): Promise<{ permissions: Permission[] }> {
    return this.client.get<{ permissions: Permission[] }>('/api/v1/rbac/permissions');
  }

  /**
   * Assign a role to a user.
   */
  async assignRole(userId: string, roleId: string): Promise<RoleAssignment> {
    return this.client.post<RoleAssignment>('/api/v1/rbac/assignments', {
      user_id: userId,
      role_id: roleId,
    });
  }

  /**
   * Revoke a role from a user.
   */
  async revokeRole(userId: string, roleId: string): Promise<void> {
    return this.client.delete(`/api/v1/rbac/assignments/${userId}/${roleId}`);
  }

  /**
   * Get user's roles.
   */
  async getUserRoles(userId: string): Promise<{ roles: Role[] }> {
    return this.client.get<{ roles: Role[] }>(`/api/v1/rbac/users/${userId}/roles`);
  }

  /**
   * Check if a user has a permission.
   */
  async checkPermission(
    userId: string,
    permission: string
  ): Promise<{ allowed: boolean; source?: string }> {
    return this.client.get<{ allowed: boolean; source?: string }>('/api/v1/rbac/check', {
      user_id: userId,
      permission,
    });
  }

  /**
   * Get effective permissions for a user.
   */
  async getEffectivePermissions(userId: string): Promise<{ permissions: string[] }> {
    return this.client.get<{ permissions: string[] }>(`/api/v1/rbac/users/${userId}/permissions`);
  }
}

// =============================================================================
// Auth API
// =============================================================================

export class AuthAPI {
  constructor(private client: AragoraClient) {}

  /**
   * Register a new user.
   */
  async register(options: {
    email: string;
    password: string;
    name?: string;
  }): Promise<{ user: User; token: AuthToken }> {
    return this.client.post<{ user: User; token: AuthToken }>('/api/v1/auth/register', {
      email: options.email,
      password: options.password,
      ...(options.name && { name: options.name }),
    });
  }

  /**
   * Login with email and password.
   */
  async login(email: string, password: string): Promise<AuthToken> {
    return this.client.post<AuthToken>('/api/v1/auth/login', { email, password });
  }

  /**
   * Refresh an access token.
   */
  async refreshToken(refreshToken: string): Promise<AuthToken> {
    return this.client.post<AuthToken>('/api/v1/auth/refresh', { refresh_token: refreshToken });
  }

  /**
   * Logout (invalidate token).
   */
  async logout(): Promise<void> {
    return this.client.post<void>('/api/v1/auth/logout', {});
  }

  /**
   * Get current authenticated user.
   */
  async getCurrentUser(): Promise<User> {
    return this.client.get<User>('/api/v1/auth/me');
  }

  /**
   * Update user profile.
   */
  async updateProfile(options: { name?: string; avatarUrl?: string }): Promise<User> {
    return this.client.post<User>('/api/v1/auth/profile', {
      ...(options.name && { name: options.name }),
      ...(options.avatarUrl && { avatar_url: options.avatarUrl }),
    });
  }

  /**
   * Change password.
   */
  async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    return this.client.post<void>('/api/v1/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
  }

  /**
   * Request password reset email.
   */
  async requestPasswordReset(email: string): Promise<void> {
    return this.client.post<void>('/api/v1/auth/forgot-password', { email });
  }

  /**
   * Reset password with token.
   */
  async resetPassword(token: string, newPassword: string): Promise<void> {
    return this.client.post<void>('/api/v1/auth/reset-password', {
      token,
      new_password: newPassword,
    });
  }

  /**
   * Get OAuth provider URL.
   */
  async getOAuthUrl(provider: string, options: { redirectUri?: string } = {}): Promise<{ url: string }> {
    return this.client.get<{ url: string }>(`/api/v1/auth/oauth/${provider}`, {
      ...(options.redirectUri && { redirect_uri: options.redirectUri }),
    });
  }

  /**
   * Setup MFA (TOTP).
   */
  async setupMFA(method: string = 'totp'): Promise<MFASetupResponse> {
    return this.client.post<MFASetupResponse>('/api/v1/auth/mfa/setup', { method });
  }

  /**
   * Verify MFA setup.
   */
  async verifyMFASetup(code: string): Promise<MFAVerifyResponse> {
    return this.client.post<MFAVerifyResponse>('/api/v1/auth/mfa/verify', { code });
  }
}

// =============================================================================
// Codebase API
// =============================================================================

export class CodebaseAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Start a security scan on a repository.
   */
  async startScan(repoId: string, options: {
    repoPath: string;
    includePatterns?: string[];
    excludePatterns?: string[];
    severityThreshold?: string;
    includeSecrets?: boolean;
    includeDependencies?: boolean;
  }): Promise<{ scan_id: string; status: string }> {
    return this.client.post<{ scan_id: string; status: string }>(
      `/api/v1/codebase/${repoId}/scan`,
      {
        repo_path: options.repoPath,
        include_patterns: options.includePatterns,
        exclude_patterns: options.excludePatterns,
        severity_threshold: options.severityThreshold,
        include_secrets: options.includeSecrets,
        include_dependencies: options.includeDependencies,
      }
    );
  }

  /**
   * Get the latest scan result for a repository.
   */
  async getLatestScan(repoId: string): Promise<CodebaseScanResult> {
    return this.client.get<CodebaseScanResult>(`/api/v1/codebase/${repoId}/scan/latest`);
  }

  /**
   * Get a specific scan result.
   */
  async getScan(repoId: string, scanId: string): Promise<CodebaseScanResult> {
    return this.client.get<CodebaseScanResult>(`/api/v1/codebase/${repoId}/scan/${scanId}`);
  }

  /**
   * List all scans for a repository.
   */
  async listScans(repoId: string, options: {
    limit?: number;
    offset?: number;
  } = {}): Promise<{ scans: CodebaseScanResult[]; total: number }> {
    return this.client.get<{ scans: CodebaseScanResult[]; total: number }>(
      `/api/v1/codebase/${repoId}/scans`,
      {
        ...(options.limit !== undefined && { limit: options.limit }),
        ...(options.offset !== undefined && { offset: options.offset }),
      }
    );
  }

  /**
   * Get vulnerabilities found in a repository.
   */
  async getVulnerabilities(repoId: string, options: {
    scanId?: string;
    severity?: string;
    limit?: number;
    offset?: number;
  } = {}): Promise<{ vulnerabilities: CodebaseVulnerability[]; total: number }> {
    return this.client.get<{ vulnerabilities: CodebaseVulnerability[]; total: number }>(
      `/api/v1/codebase/${repoId}/vulnerabilities`,
      {
        ...(options.scanId && { scan_id: options.scanId }),
        ...(options.severity && { severity: options.severity }),
        ...(options.limit !== undefined && { limit: options.limit }),
        ...(options.offset !== undefined && { offset: options.offset }),
      }
    );
  }

  /**
   * Analyze dependencies for security vulnerabilities.
   */
  async analyzeDependencies(options: {
    repoPath: string;
    includeTransitive?: boolean;
    checkVulnerabilities?: boolean;
  }): Promise<CodebaseDependencyAnalysis> {
    return this.client.post<CodebaseDependencyAnalysis>(
      '/api/v1/codebase/analyze-dependencies',
      {
        repo_path: options.repoPath,
        include_transitive: options.includeTransitive ?? true,
        check_vulnerabilities: options.checkVulnerabilities ?? true,
      }
    );
  }

  /**
   * Run a quick vulnerability scan (combined patterns + secrets).
   */
  async quickScan(options: {
    repoPath: string;
    severityThreshold?: string;
    includeSecrets?: boolean;
  }): Promise<CodebaseScanResult & { findings: CodebaseVulnerability[] }> {
    return this.client.post<CodebaseScanResult & { findings: CodebaseVulnerability[] }>(
      '/api/v1/codebase/scan-vulnerabilities',
      {
        repo_path: options.repoPath,
        severity_threshold: options.severityThreshold ?? 'medium',
        include_secrets: options.includeSecrets ?? true,
      }
    );
  }

  /**
   * Get code metrics for a repository.
   */
  async getMetrics(repoId: string): Promise<CodebaseMetrics> {
    return this.client.get<CodebaseMetrics>(`/api/v1/codebase/${repoId}/metrics`);
  }

  /**
   * Analyze code metrics (complexity, duplication, etc.).
   */
  async analyzeMetrics(repoId: string, options: {
    repoPath: string;
    includePatterns?: string[];
    excludePatterns?: string[];
    complexityWarning?: number;
    complexityError?: number;
  }): Promise<{ analysis_id: string; status: string }> {
    return this.client.post<{ analysis_id: string; status: string }>(
      `/api/v1/codebase/${repoId}/metrics/analyze`,
      {
        repo_path: options.repoPath,
        include_patterns: options.includePatterns,
        exclude_patterns: options.excludePatterns,
        complexity_warning: options.complexityWarning ?? 10,
        complexity_error: options.complexityError ?? 20,
      }
    );
  }

  /**
   * Get complexity hotspots in a repository.
   */
  async getHotspots(repoId: string, options: {
    minComplexity?: number;
    limit?: number;
  } = {}): Promise<{ hotspots: Array<{ file_path: string; complexity: number; risk_score: number }>; total: number }> {
    return this.client.get<{ hotspots: Array<{ file_path: string; complexity: number; risk_score: number }>; total: number }>(
      `/api/v1/codebase/${repoId}/hotspots`,
      {
        ...(options.minComplexity !== undefined && { min_complexity: options.minComplexity }),
        ...(options.limit !== undefined && { limit: options.limit }),
      }
    );
  }

  /**
   * Get code duplicates in a repository.
   */
  async getDuplicates(repoId: string, options: {
    minLines?: number;
    limit?: number;
  } = {}): Promise<{ duplicates: Array<{ hash: string; lines: number; occurrences: Array<{ file: string; start: number; end: number }> }>; total: number }> {
    return this.client.get<{ duplicates: Array<{ hash: string; lines: number; occurrences: Array<{ file: string; start: number; end: number }> }>; total: number }>(
      `/api/v1/codebase/${repoId}/duplicates`,
      {
        ...(options.minLines !== undefined && { min_lines: options.minLines }),
        ...(options.limit !== undefined && { limit: options.limit }),
      }
    );
  }
}

// =============================================================================
// Decisions API
// =============================================================================

export class DecisionsAPI {
  constructor(private client: AragoraClient) {}

  /**
   * Create a new decision request.
   */
  async create(options: {
    content: string;
    decisionType?: DecisionType;
    config?: Partial<DecisionConfig>;
    context?: DecisionContext;
    priority?: DecisionPriority;
    responseChannels?: ResponseChannel[];
  }): Promise<DecisionResult> {
    return this.client.post<DecisionResult>('/api/v1/decisions', {
      content: options.content,
      decision_type: options.decisionType ?? 'auto',
      priority: options.priority ?? 'normal',
      ...(options.config && {
        config: {
          agents: options.config.agents,
          rounds: options.config.rounds,
          consensus: options.config.consensus,
          timeout_seconds: options.config.timeout_seconds,
        },
      }),
      ...(options.context && {
        context: {
          user_id: options.context.user_id,
          workspace_id: options.context.workspace_id,
          ...options.context.metadata,
        },
      }),
      ...(options.responseChannels && {
        response_channels: options.responseChannels.map((ch) => ({
          platform: ch.platform,
          target: ch.target,
          ...ch.options,
        })),
      }),
    });
  }

  /**
   * Get a decision by request ID.
   */
  async get(requestId: string): Promise<DecisionResult> {
    return this.client.get<DecisionResult>(`/api/v1/decisions/${requestId}`);
  }

  /**
   * Get decision status for polling.
   */
  async getStatus(requestId: string): Promise<DecisionStatusResponse> {
    return this.client.get<DecisionStatusResponse>(`/api/v1/decisions/${requestId}/status`);
  }

  /**
   * List recent decisions.
   */
  async list(options: {
    status?: string;
    decisionType?: DecisionType;
    limit?: number;
    offset?: number;
  } = {}): Promise<{ decisions: DecisionResult[]; total: number }> {
    return this.client.get<{ decisions: DecisionResult[]; total: number }>('/api/v1/decisions', {
      limit: options.limit ?? 50,
      offset: options.offset ?? 0,
      ...(options.status && { status: options.status }),
      ...(options.decisionType && { decision_type: options.decisionType }),
    });
  }

  /**
   * Make a quick decision with minimal configuration.
   */
  async quickDecision(question: string, agents?: string[]): Promise<DecisionResult> {
    return this.create({
      content: question,
      decisionType: 'quick',
      config: {
        agents: agents ?? ['anthropic-api', 'openai-api'],
        rounds: 2,
        consensus: 'majority',
        timeout_seconds: 60,
      },
    });
  }

  /**
   * Start a full debate on a topic.
   */
  async startDebate(
    topic: string,
    options: { agents?: string[]; rounds?: number } = {}
  ): Promise<DecisionResult> {
    return this.create({
      content: topic,
      decisionType: 'debate',
      config: {
        agents: options.agents ?? ['anthropic-api', 'openai-api', 'gemini-api'],
        rounds: options.rounds ?? 3,
        consensus: 'majority',
        timeout_seconds: 300,
      },
    });
  }
}

// =============================================================================
// Documents API
// =============================================================================

export class DocumentsAPI {
  constructor(private client: AragoraClient) {}

  /**
   * List uploaded documents.
   */
  async list(options: {
    limit?: number;
    offset?: number;
    status?: string;
  } = {}): Promise<Document[]> {
    const data = await this.client.get<{ documents: Document[] }>('/api/documents', {
      limit: options.limit ?? 50,
      offset: options.offset ?? 0,
      ...(options.status && { status: options.status }),
    });
    return data.documents ?? [];
  }

  /**
   * Get a document by ID.
   */
  async get(documentId: string): Promise<Document> {
    return this.client.get<Document>(`/api/documents/${documentId}`);
  }

  /**
   * Upload a document for processing.
   */
  async upload(options: {
    filename: string;
    content: string;
    contentType: string;
    metadata?: Record<string, unknown>;
  }): Promise<DocumentUploadResponse> {
    return this.client.post<DocumentUploadResponse>('/api/documents/upload', {
      filename: options.filename,
      content: options.content,
      content_type: options.contentType,
      ...(options.metadata && { metadata: options.metadata }),
    });
  }

  /**
   * Delete a document by ID.
   */
  async delete(documentId: string): Promise<void> {
    return this.client.delete(`/api/documents/${documentId}`);
  }

  /**
   * Get supported document formats.
   */
  async formats(): Promise<SupportedFormats> {
    return this.client.get<SupportedFormats>('/api/documents/formats');
  }

  /**
   * Upload multiple documents as a batch.
   */
  async batchUpload(
    files: Array<{ filename: string; content: string; contentType: string }>,
    metadata?: Record<string, unknown>
  ): Promise<BatchUploadResponse> {
    return this.client.post<BatchUploadResponse>('/api/documents/batch', {
      files: files.map((f) => ({
        filename: f.filename,
        content: f.content,
        content_type: f.contentType,
      })),
      ...(metadata && { metadata }),
    });
  }

  /**
   * Get the status of a batch processing job.
   */
  async batchStatus(jobId: string): Promise<BatchJobStatus> {
    return this.client.get<BatchJobStatus>(`/api/documents/batch/${jobId}`);
  }

  /**
   * Get the results of a completed batch job.
   */
  async batchResults(jobId: string): Promise<BatchJobResults> {
    return this.client.get<BatchJobResults>(`/api/documents/batch/${jobId}/results`);
  }

  /**
   * Cancel a batch processing job.
   */
  async batchCancel(jobId: string): Promise<void> {
    return this.client.delete(`/api/documents/batch/${jobId}`);
  }

  /**
   * Get document processing statistics.
   */
  async processingStats(): Promise<ProcessingStats> {
    return this.client.get<ProcessingStats>('/api/documents/processing/stats');
  }

  /**
   * Get chunks for a document.
   */
  async chunks(
    documentId: string,
    options: { limit?: number; offset?: number } = {}
  ): Promise<DocumentChunk[]> {
    const data = await this.client.get<{ chunks: DocumentChunk[] }>(
      `/api/documents/${documentId}/chunks`,
      {
        limit: options.limit ?? 100,
        offset: options.offset ?? 0,
      }
    );
    return data.chunks ?? [];
  }

  /**
   * Get LLM-ready context from a document.
   */
  async context(
    documentId: string,
    options: { maxTokens?: number; model?: string } = {}
  ): Promise<DocumentContext> {
    return this.client.get<DocumentContext>(`/api/documents/${documentId}/context`, {
      max_tokens: options.maxTokens ?? 100000,
      model: options.model ?? 'gemini-1.5-flash',
    });
  }

  /**
   * Create a new audit session for documents.
   */
  async createAudit(
    documentIds: string[],
    options: { auditTypes?: string[]; model?: string } = {}
  ): Promise<AuditSessionCreateResponse> {
    return this.client.post<AuditSessionCreateResponse>('/api/audit/sessions', {
      document_ids: documentIds,
      audit_types: options.auditTypes ?? ['security', 'compliance', 'consistency', 'quality'],
      model: options.model ?? 'gemini-1.5-flash',
    });
  }

  /**
   * List audit sessions.
   */
  async listAudits(options: {
    limit?: number;
    offset?: number;
    status?: string;
  } = {}): Promise<AuditSession[]> {
    const data = await this.client.get<{ sessions: AuditSession[] }>('/api/audit/sessions', {
      limit: options.limit ?? 20,
      offset: options.offset ?? 0,
      ...(options.status && { status: options.status }),
    });
    return data.sessions ?? [];
  }

  /**
   * Get audit session details.
   */
  async getAudit(sessionId: string): Promise<AuditSession> {
    return this.client.get<AuditSession>(`/api/audit/sessions/${sessionId}`);
  }

  /**
   * Start an audit session.
   */
  async startAudit(sessionId: string): Promise<AuditSession> {
    return this.client.post<AuditSession>(`/api/audit/sessions/${sessionId}/start`, {});
  }

  /**
   * Pause an audit session.
   */
  async pauseAudit(sessionId: string): Promise<AuditSession> {
    return this.client.post<AuditSession>(`/api/audit/sessions/${sessionId}/pause`, {});
  }

  /**
   * Resume a paused audit session.
   */
  async resumeAudit(sessionId: string): Promise<AuditSession> {
    return this.client.post<AuditSession>(`/api/audit/sessions/${sessionId}/resume`, {});
  }

  /**
   * Cancel an audit session.
   */
  async cancelAudit(sessionId: string): Promise<AuditSession> {
    return this.client.post<AuditSession>(`/api/audit/sessions/${sessionId}/cancel`, {});
  }

  /**
   * Get findings from an audit session.
   */
  async auditFindings(
    sessionId: string,
    options: { severity?: string; auditType?: string } = {}
  ): Promise<AuditFinding[]> {
    const data = await this.client.get<{ findings: AuditFinding[] }>(
      `/api/audit/sessions/${sessionId}/findings`,
      {
        ...(options.severity && { severity: options.severity }),
        ...(options.auditType && { audit_type: options.auditType }),
      }
    );
    return data.findings ?? [];
  }

  /**
   * Generate an audit report.
   */
  async auditReport(sessionId: string, format: string = 'json'): Promise<AuditReport> {
    return this.client.get<AuditReport>(`/api/audit/sessions/${sessionId}/report`, { format });
  }
}

// =============================================================================
// Policies API
// =============================================================================

export class PoliciesAPI {
  constructor(private client: AragoraClient) {}

  /**
   * List policies with optional filters.
   */
  async list(options: {
    workspaceId?: string;
    verticalId?: string;
    frameworkId?: string;
    enabledOnly?: boolean;
    limit?: number;
    offset?: number;
  } = {}): Promise<{ policies: Policy[]; total: number }> {
    return this.client.get<{ policies: Policy[]; total: number }>('/api/v1/policies', {
      limit: options.limit ?? 100,
      offset: options.offset ?? 0,
      ...(options.workspaceId && { workspace_id: options.workspaceId }),
      ...(options.verticalId && { vertical_id: options.verticalId }),
      ...(options.frameworkId && { framework_id: options.frameworkId }),
      ...(options.enabledOnly && { enabled_only: options.enabledOnly }),
    });
  }

  /**
   * Get policy details.
   */
  async get(policyId: string): Promise<Policy> {
    const data = await this.client.get<{ policy: Policy }>(`/api/v1/policies/${policyId}`);
    return data.policy ?? (data as unknown as Policy);
  }

  /**
   * Create a new policy.
   */
  async create(options: {
    name: string;
    frameworkId: string;
    verticalId: string;
    description?: string;
    workspaceId?: string;
    level?: string;
    enabled?: boolean;
    rules?: Array<Partial<PolicyRule>>;
    metadata?: Record<string, unknown>;
  }): Promise<Policy> {
    const data = await this.client.post<{ policy: Policy }>('/api/v1/policies', {
      name: options.name,
      framework_id: options.frameworkId,
      vertical_id: options.verticalId,
      description: options.description ?? '',
      workspace_id: options.workspaceId ?? 'default',
      level: options.level ?? 'recommended',
      enabled: options.enabled ?? true,
      ...(options.rules && { rules: options.rules }),
      ...(options.metadata && { metadata: options.metadata }),
    });
    return data.policy ?? (data as unknown as Policy);
  }

  /**
   * Update a policy.
   */
  async update(
    policyId: string,
    updates: {
      name?: string;
      description?: string;
      level?: string;
      enabled?: boolean;
      rules?: Array<Partial<PolicyRule>>;
      metadata?: Record<string, unknown>;
    }
  ): Promise<Policy> {
    const data = await this.client.post<{ policy: Policy }>(`/api/v1/policies/${policyId}`, {
      ...(updates.name !== undefined && { name: updates.name }),
      ...(updates.description !== undefined && { description: updates.description }),
      ...(updates.level !== undefined && { level: updates.level }),
      ...(updates.enabled !== undefined && { enabled: updates.enabled }),
      ...(updates.rules !== undefined && { rules: updates.rules }),
      ...(updates.metadata !== undefined && { metadata: updates.metadata }),
    });
    return data.policy ?? (data as unknown as Policy);
  }

  /**
   * Delete a policy.
   */
  async delete(policyId: string): Promise<void> {
    return this.client.delete(`/api/v1/policies/${policyId}`);
  }

  /**
   * Toggle a policy's enabled status.
   */
  async toggle(policyId: string, enabled?: boolean): Promise<Policy> {
    const data = await this.client.post<{ policy: Policy }>(`/api/v1/policies/${policyId}/toggle`, {
      ...(enabled !== undefined && { enabled }),
    });
    return data.policy ?? (data as unknown as Policy);
  }

  /**
   * List policy violations.
   */
  async listViolations(options: {
    policyId?: string;
    workspaceId?: string;
    status?: string;
    severity?: string;
    limit?: number;
    offset?: number;
  } = {}): Promise<{ violations: PolicyViolation[]; total: number }> {
    return this.client.get<{ violations: PolicyViolation[]; total: number }>(
      '/api/v1/compliance/violations',
      {
        limit: options.limit ?? 100,
        offset: options.offset ?? 0,
        ...(options.policyId && { policy_id: options.policyId }),
        ...(options.workspaceId && { workspace_id: options.workspaceId }),
        ...(options.status && { status: options.status }),
        ...(options.severity && { severity: options.severity }),
      }
    );
  }

  /**
   * Get violation details.
   */
  async getViolation(violationId: string): Promise<PolicyViolation> {
    const data = await this.client.get<{ violation: PolicyViolation }>(
      `/api/v1/compliance/violations/${violationId}`
    );
    return data.violation ?? (data as unknown as PolicyViolation);
  }

  /**
   * Update violation status.
   */
  async updateViolation(
    violationId: string,
    status: string,
    resolutionNotes?: string
  ): Promise<PolicyViolation> {
    const data = await this.client.post<{ violation: PolicyViolation }>(
      `/api/v1/compliance/violations/${violationId}`,
      {
        status,
        ...(resolutionNotes && { resolution_notes: resolutionNotes }),
      }
    );
    return data.violation ?? (data as unknown as PolicyViolation);
  }

  /**
   * Run compliance check on content.
   */
  async check(
    content: string,
    options: {
      frameworks?: string[];
      minSeverity?: string;
      storeViolations?: boolean;
      workspaceId?: string;
    } = {}
  ): Promise<ComplianceCheckResult> {
    return this.client.post<ComplianceCheckResult>('/api/v1/compliance/check', {
      content,
      min_severity: options.minSeverity ?? 'low',
      store_violations: options.storeViolations ?? false,
      workspace_id: options.workspaceId ?? 'default',
      ...(options.frameworks && { frameworks: options.frameworks }),
    });
  }

  /**
   * Get compliance statistics.
   */
  async getStats(workspaceId?: string): Promise<ComplianceStats> {
    return this.client.get<ComplianceStats>('/api/v1/compliance/stats', {
      ...(workspaceId && { workspace_id: workspaceId }),
    });
  }
}

export class AragoraClient {
  public readonly baseUrl: string;
  private apiKey?: string;
  private timeout: number;
  private headers: Record<string, string>;

  // API namespaces
  public readonly debates: DebatesAPI;
  public readonly graphDebates: GraphDebatesAPI;
  public readonly matrixDebates: MatrixDebatesAPI;
  public readonly agents: AgentsAPI;
  public readonly verification: VerificationAPI;
  public readonly gauntlet: GauntletAPI;
  public readonly teamSelection: TeamSelectionAPI;
  public readonly controlPlane: ControlPlaneAPI;
  public readonly analytics: AnalyticsAPI;
  public readonly memory: MemoryAPI;
  public readonly knowledge: KnowledgeAPI;
  public readonly workflows: WorkflowsAPI;
  public readonly tournaments: TournamentsAPI;
  public readonly rbac: RBACAPI;
  public readonly auth: AuthAPI;
  public readonly codebase: CodebaseAPI;
  public readonly decisions: DecisionsAPI;
  public readonly documents: DocumentsAPI;
  public readonly policies: PoliciesAPI;

  constructor(baseUrl: string = 'http://localhost:8080', options: AragoraClientOptions = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = options.apiKey;
    this.timeout = options.timeout ?? 30000;
    this.headers = {
      'Content-Type': 'application/json',
      'User-Agent': 'aragora-client-js/2.1.13',
      ...(options.apiKey && { Authorization: `Bearer ${options.apiKey}` }),
      ...options.headers,
    };

    // Initialize API namespaces
    this.debates = new DebatesAPI(this);
    this.graphDebates = new GraphDebatesAPI(this);
    this.matrixDebates = new MatrixDebatesAPI(this);
    this.agents = new AgentsAPI(this);
    this.verification = new VerificationAPI(this);
    this.gauntlet = new GauntletAPI(this);
    this.teamSelection = new TeamSelectionAPI(this);
    this.controlPlane = new ControlPlaneAPI(this);
    this.analytics = new AnalyticsAPI(this);
    this.memory = new MemoryAPI(this);
    this.knowledge = new KnowledgeAPI(this);
    this.workflows = new WorkflowsAPI(this);
    this.tournaments = new TournamentsAPI(this);
    this.rbac = new RBACAPI(this);
    this.auth = new AuthAPI(this);
    this.codebase = new CodebaseAPI(this);
  }

  /**
   * Get server health status.
   */
  async health(): Promise<HealthStatus> {
    return this.get<HealthStatus>('/api/v1/health');
  }

  /**
   * Make a GET request.
   */
  async get<T>(
    path: string,
    params?: Record<string, string | number | boolean>
  ): Promise<T> {
    const url = new URL(path, this.baseUrl);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        url.searchParams.set(key, String(value));
      });
    }

    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: this.headers,
      signal: AbortSignal.timeout(this.timeout),
    });

    return this.handleResponse<T>(response);
  }

  /**
   * Make a POST request.
   */
  async post<T>(
    path: string,
    data: Record<string, unknown>
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(data),
      signal: AbortSignal.timeout(this.timeout),
    });

    return this.handleResponse<T>(response);
  }

  /**
   * Make a DELETE request.
   */
  async delete(path: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: 'DELETE',
      headers: this.headers,
      signal: AbortSignal.timeout(this.timeout),
    });

    if (!response.ok) {
      await this.handleError(response);
    }
  }

  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      await this.handleError(response);
    }

    const data = await response.json();
    return data as T;
  }

  private async handleError(response: Response): Promise<never> {
    let errorData: Record<string, unknown> = {};
    try {
      errorData = (await response.json()) as Record<string, unknown>;
    } catch {
      // Ignore JSON parse errors
    }

    if (response.status === 401) {
      throw new AragoraError('Authentication failed', { status: 401 });
    }
    if (response.status === 404) {
      throw new AragoraError('Resource not found', { status: 404 });
    }
    if (response.status === 400) {
      throw new AragoraError(
        (errorData.error as string) ?? 'Validation error',
        { status: 400, details: errorData.details as Record<string, unknown> }
      );
    }

    throw new AragoraError(
      (errorData.error as string) ?? `Request failed with status ${response.status}`,
      {
        status: response.status,
        code: errorData.code as string,
        details: errorData.details as Record<string, unknown>,
      }
    );
  }
}
