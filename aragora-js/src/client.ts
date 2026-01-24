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
