/**
 * Aragora SDK Type Definitions
 *
 * Core types for interacting with the Aragora API.
 */

// =============================================================================
// Enums
// =============================================================================

/**
 * Debate status values.
 *
 * Canonical values: pending, running, completed, failed, cancelled, paused
 * Legacy values (still supported): created, in_progress, active, concluded
 */
export enum DebateStatus {
  // Canonical SDK values
  PENDING = 'pending',
  RUNNING = 'running',
  STARTING = 'starting',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  PAUSED = 'paused',

  // Legacy values (kept for backwards compatibility)
  CREATED = 'created',
  IN_PROGRESS = 'in_progress',
}

/**
 * Map of legacy server status values to canonical SDK statuses.
 * Use normalizeStatus() to convert server responses.
 */
export const LEGACY_STATUS_MAP: Record<string, DebateStatus> = {
  active: DebateStatus.RUNNING,
  concluded: DebateStatus.COMPLETED,
  archived: DebateStatus.COMPLETED,
  created: DebateStatus.PENDING,
  in_progress: DebateStatus.RUNNING,
  starting: DebateStatus.PENDING,
};

/**
 * Normalize a status value from the server to a canonical SDK status.
 *
 * @param status - Status string from server response
 * @returns Canonical DebateStatus value
 *
 * @example
 * ```typescript
 * const status = normalizeStatus('active'); // Returns DebateStatus.RUNNING
 * const status2 = normalizeStatus('running'); // Returns DebateStatus.RUNNING
 * ```
 */
export function normalizeStatus(status: string | DebateStatus): DebateStatus {
  const statusStr = String(status);
  // Check if it's a legacy value that needs mapping
  const mapped = LEGACY_STATUS_MAP[statusStr];
  if (mapped !== undefined) {
    return mapped;
  }
  // Return as-is if it's already a valid enum value
  return statusStr as DebateStatus;
}

export enum VerificationStatus {
  VALID = 'valid',
  INVALID = 'invalid',
  UNKNOWN = 'unknown',
  ERROR = 'error',
}

export enum GauntletPersona {
  SECURITY = 'security',
  PERFORMANCE = 'performance',
  USABILITY = 'usability',
  ACCESSIBILITY = 'accessibility',
  COMPLIANCE = 'compliance',
  DEVIL_ADVOCATE = 'devil_advocate',
}

// =============================================================================
// Base Types
// =============================================================================

export interface ConsensusResult {
  reached: boolean;
  /** Final answer/conclusion from the debate */
  conclusion?: string;
  /** Alias for conclusion (for SDK compatibility) */
  final_answer?: string;
  /** Agreement level (0-1), alias: agreement */
  confidence: number;
  /** Alias for confidence (for SDK compatibility) */
  agreement?: number;
  supporting_agents: string[];
  dissenting_agents?: string[];
}

/**
 * Normalize a consensus result from the server, ensuring both field variants are present.
 *
 * @param consensus - Raw consensus result from server
 * @returns Normalized consensus with both field aliases populated
 */
export function normalizeConsensusResult(
  consensus: ConsensusResult | undefined
): ConsensusResult | undefined {
  if (!consensus) return undefined;

  return {
    ...consensus,
    // Ensure both confidence and agreement are present
    confidence: consensus.confidence ?? consensus.agreement ?? 0,
    agreement: consensus.agreement ?? consensus.confidence ?? 0,
    // Ensure both conclusion and final_answer are present
    conclusion: consensus.conclusion ?? consensus.final_answer,
    final_answer: consensus.final_answer ?? consensus.conclusion,
  };
}

export interface DebateMessage {
  agent_id: string;
  content: string;
  round: number;
  message_type?: 'proposal' | 'critique' | 'revision' | 'synthesis';
  timestamp?: string;
}

export interface DebateRound {
  round_number: number;
  messages: DebateMessage[];
  critiques?: DebateCritique[];
}

export interface DebateCritique {
  critic_id: string;
  target_agent_id: string;
  content: string;
  severity?: 'low' | 'medium' | 'high';
}

// =============================================================================
// Debate Types
// =============================================================================

/**
 * Request to create a new debate.
 *
 * @example
 * ```typescript
 * const request: DebateCreateRequest = {
 *   task: 'Should we adopt TypeScript for our backend?',
 *   agents: ['anthropic-api', 'openai-api'],
 *   max_rounds: 5,
 *   consensus_threshold: 0.7,
 * };
 * ```
 */
export interface DebateCreateRequest {
  /**
   * The topic or question to debate.
   * Should be a clear, specific question or task for the agents to discuss.
   */
  task: string;

  /**
   * Agent identifiers to participate in the debate.
   * Available agents: 'anthropic-api', 'openai-api', 'gemini-api', 'mistral-api', 'grok-api'
   * If not specified, default agents will be selected.
   */
  agents?: string[];

  /**
   * Maximum number of debate rounds (default: 5).
   * Each round consists of proposals, critiques, and revisions.
   */
  max_rounds?: number;

  /**
   * Threshold for consensus detection (0.0-1.0, default: 0.7).
   * Higher values require stronger agreement between agents.
   */
  consensus_threshold?: number;

  /**
   * Enable voting mechanism for human participation.
   * When true, breakpoints may be created for user input.
   */
  enable_voting?: boolean;

  /**
   * Additional context to provide to all agents.
   * Useful for domain-specific knowledge or constraints.
   */
  context?: string;
}

/**
 * Response from creating a new debate.
 */
export interface DebateCreateResponse {
  /** Unique identifier for the created debate */
  debate_id: string;

  /** Initial status of the debate (typically 'created' or 'in_progress') */
  status: string;

  /** Echo of the task that was submitted */
  task: string;
}

/**
 * Complete debate object with all rounds, messages, and results.
 *
 * @example
 * ```typescript
 * const debate = await client.debates.get('debate-123');
 * if (debate.status === DebateStatus.COMPLETED) {
 *   console.log('Consensus:', debate.consensus?.conclusion);
 *   for (const round of debate.rounds) {
 *     console.log(`Round ${round.round_number}:`, round.messages.length, 'messages');
 *   }
 * }
 * ```
 */
export interface Debate {
  /** Optional alias for debate_id used in some API responses */
  id?: string;

  /** Unique debate identifier */
  debate_id: string;

  /** The topic or question being debated */
  task: string;

  /** Current status of the debate */
  status: DebateStatus;

  /** List of agent identifiers participating in the debate */
  agents: string[];

  /** Completed debate rounds with messages and critiques */
  rounds: DebateRound[];

  /** Consensus result if debate reached agreement */
  consensus?: ConsensusResult;

  /** ISO 8601 timestamp when the debate was created */
  created_at?: string;

  /** ISO 8601 timestamp when the debate completed (if finished) */
  completed_at?: string;

  /** Additional metadata attached to the debate */
  metadata?: Record<string, unknown>;
}

export interface DebateListResponse {
  debates: Debate[];
  total: number;
  offset?: number;
  limit?: number;
}

// =============================================================================
// Graph Debate Types
// =============================================================================

export interface GraphDebateNode {
  node_id: string;
  content: string;
  agent_id: string;
  node_type: 'proposal' | 'critique' | 'synthesis';
  parent_id?: string;
  round: number;
}

export interface GraphDebateBranch {
  branch_id: string;
  name: string;
  nodes: GraphDebateNode[];
  created_at?: string;
  is_main: boolean;
}

export interface GraphDebateCreateRequest {
  task: string;
  agents?: string[];
  max_rounds?: number;
  branch_threshold?: number;
  max_branches?: number;
}

export interface GraphDebateCreateResponse {
  debate_id: string;
  status: string;
  task: string;
}

export interface GraphDebate {
  debate_id: string;
  task: string;
  status: DebateStatus;
  agents: string[];
  branches: GraphDebateBranch[];
  consensus?: ConsensusResult;
  created_at?: string;
  completed_at?: string;
}

// =============================================================================
// Matrix Debate Types
// =============================================================================

export interface MatrixScenario {
  name: string;
  parameters: Record<string, unknown>;
  constraints?: string[];
  is_baseline?: boolean;
}

export interface MatrixScenarioResult {
  scenario_name: string;
  consensus?: ConsensusResult;
  key_findings: string[];
  differences_from_baseline?: string[];
}

export interface MatrixConclusion {
  universal: string[];
  conditional: Record<string, string[]>;
  contradictions: string[];
}

export interface MatrixDebateCreateRequest {
  task: string;
  agents?: string[];
  scenarios?: MatrixScenario[];
  max_rounds?: number;
}

export interface MatrixDebateCreateResponse {
  matrix_id: string;
  status: string;
  task: string;
  scenario_count: number;
}

export interface MatrixDebate {
  matrix_id: string;
  task: string;
  status: DebateStatus;
  agents: string[];
  scenarios: MatrixScenarioResult[];
  conclusions?: MatrixConclusion;
  created_at?: string;
  completed_at?: string;
}

// =============================================================================
// Verification Types
// =============================================================================

export interface VerifyClaimRequest {
  claim: string;
  context?: string;
  backend?: 'z3' | 'lean' | 'coq';
  timeout?: number;
}

export interface VerifyClaimResponse {
  status: VerificationStatus;
  claim: string;
  formal_translation?: string;
  proof?: string;
  counterexample?: string;
  error_message?: string;
  duration_ms: number;
}

export interface VerificationBackendStatus {
  name: string;
  available: boolean;
  version?: string;
}

export interface VerifyStatusResponse {
  available: boolean;
  backends: VerificationBackendStatus[];
}

// =============================================================================
// Memory Types
// =============================================================================

export interface MemoryTierStats {
  tier_name: string;
  entry_count: number;
  avg_access_frequency: number;
  promotion_rate: number;
  demotion_rate: number;
  hit_rate: number;
}

export interface MemoryRecommendation {
  type: 'promotion' | 'cleanup' | 'rebalance';
  description: string;
  impact: 'high' | 'medium' | 'low';
}

export interface MemoryAnalyticsResponse {
  tiers: MemoryTierStats[];
  total_entries: number;
  learning_velocity: number;
  promotion_effectiveness: number;
  recommendations: MemoryRecommendation[];
  period_days: number;
}

export interface MemorySnapshotResponse {
  snapshot_id: string;
  timestamp: string;
  success: boolean;
}

// =============================================================================
// Agent Types
// =============================================================================

export interface AgentProfile {
  agent_id: string;
  name: string;
  provider: string;
  elo_rating?: number;
  wins?: number;
  losses?: number;
  draws?: number;
  specializations?: string[];
}

export interface LeaderboardEntry {
  agent_id: string;
  elo_rating: number;
  rank: number;
  wins?: number;
  losses?: number;
  win_rate?: number;
}

// =============================================================================
// Extended Agent Types
// =============================================================================

export interface AgentDetailedProfile extends AgentProfile {
  description?: string;
  model?: string;
  created_at?: string;
  last_active?: string;
  total_debates: number;
  total_rounds: number;
  consensus_rate: number;
  calibration_score?: number;
  traits?: string[];
  strengths?: string[];
  weaknesses?: string[];
}

export interface AgentCalibrationData {
  agent: string;
  total_predictions: number;
  brier_score: number;
  log_loss: number;
  expected_calibration_error: number;
  accuracy: number;
  overconfidence_rate: number;
  underconfidence_rate: number;
  buckets: Array<{
    range: string;
    predicted: number;
    actual: number;
    count: number;
  }>;
}

export interface AgentPerformanceMetrics {
  agent: string;
  period_days: number;
  debates_participated: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  avg_contribution_score: number;
  avg_response_time_ms: number;
  consensus_influence: number;
  position_flip_rate: number;
  domains_active: string[];
}

export interface AgentConsistencyData {
  agent: string;
  overall_consistency: number;
  domain_consistency: Record<string, number>;
  temporal_consistency: number;
  self_contradiction_rate: number;
  position_stability: number;
  recent_inconsistencies: Array<{
    debate_id: string;
    claim_a: string;
    claim_b: string;
    similarity: number;
    timestamp: string;
  }>;
}

export interface AgentFlipEvent {
  id: string;
  agent: string;
  debate_id: string;
  round: number;
  type: 'contradiction' | 'retraction' | 'qualification' | 'refinement';
  old_position: string;
  new_position: string;
  reason?: string;
  confidence_before: number;
  confidence_after: number;
  timestamp: string;
}

export interface AgentNetworkNode {
  id: string;
  label: string;
  type: 'agent' | 'topic';
  size?: number;
}

export interface AgentNetworkEdge {
  source: string;
  target: string;
  weight: number;
  type: 'ally' | 'rival' | 'neutral';
}

export interface AgentNetworkGraph {
  agent: string;
  nodes: AgentNetworkNode[];
  edges: AgentNetworkEdge[];
  clusters?: Array<{
    id: string;
    agents: string[];
    common_positions: string[];
  }>;
}

export interface AgentMoment {
  id: string;
  debate_id: string;
  round: number;
  type: 'breakthrough' | 'pivot' | 'concession' | 'challenge' | 'synthesis';
  description: string;
  impact_score: number;
  timestamp: string;
  context?: string;
}

export interface AgentPosition {
  id: string;
  claim: string;
  confidence: number;
  domain?: string;
  debate_id: string;
  round: number;
  timestamp: string;
  supporting_evidence?: string[];
  status: 'active' | 'revised' | 'retracted';
}

export interface AgentDomainExpertise {
  domain: string;
  debates_count: number;
  win_rate: number;
  avg_contribution_score: number;
  calibration_score?: number;
  expertise_level: 'novice' | 'intermediate' | 'expert' | 'master';
  key_topics: string[];
}

export interface AgentHeadToHead {
  agent_a: string;
  agent_b: string;
  total_encounters: number;
  agent_a_wins: number;
  agent_b_wins: number;
  draws: number;
  agent_a_win_rate: number;
  recent_matches: Array<{
    debate_id: string;
    winner?: string;
    topic: string;
    timestamp: string;
  }>;
  domain_breakdown: Record<string, {
    agent_a_wins: number;
    agent_b_wins: number;
  }>;
}

export interface AgentOpponentBriefing {
  opponent: string;
  strengths: string[];
  weaknesses: string[];
  common_arguments: string[];
  typical_strategies: string[];
  vulnerabilities: string[];
  recommended_approaches: string[];
  historical_wins_against: number;
  historical_losses_against: number;
}

export interface AgentComparisonResult {
  agents: string[];
  comparison_matrix: Record<string, Record<string, {
    wins: number;
    losses: number;
    draws: number;
  }>>;
  rankings: Array<{
    agent: string;
    rank: number;
    elo: number;
    win_rate: number;
  }>;
  strongest_domains: Record<string, string>;
  notable_matchups: Array<{
    agents: [string, string];
    rivalry_score: number;
    total_matches: number;
  }>;
}

export interface AgentGroundedPersona {
  agent: string;
  persona_summary: string;
  core_beliefs: string[];
  argumentation_style: string;
  strengths: string[];
  weaknesses: string[];
  domain_expertise: string[];
  typical_positions: Record<string, string>;
  evolution_notes?: string;
}

export interface LeaderboardResponse {
  entries: LeaderboardEntry[];
  updated_at?: string;
}

// =============================================================================
// Gauntlet Types
// =============================================================================

export interface GauntletRunRequest {
  input_content: string;
  input_type?: 'spec' | 'code' | 'document';
  persona?: GauntletPersona;
  max_rounds?: number;
}

export interface GauntletRunResponse {
  gauntlet_id: string;
  status: string;
  persona: string;
}

export interface GauntletFinding {
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  category: string;
  description: string;
  location?: string;
  suggestion?: string;
}

export interface GauntletReceipt {
  gauntlet_id: string;
  status: string;
  score: number;
  findings: GauntletFinding[];
  duration_seconds: number;
  persona: string;
  summary?: string;
}

export interface GauntletResultsResponse {
  results: GauntletReceipt[];
  total: number;
  limit: number;
  offset: number;
}

export interface GauntletPersonaInfo {
  name: string;
  description: string;
  focus_areas: string[];
  severity_weights: Record<string, number>;
  example_findings: string[];
}

export interface GauntletHeatmapCell {
  category: string;
  severity: string;
  count: number;
  score: number;
}

export interface GauntletHeatmap {
  gauntlet_id: string;
  cells: GauntletHeatmapCell[];
  categories: string[];
  severities: string[];
  max_score: number;
}

export interface GauntletComparisonResult {
  gauntlet_a: string;
  gauntlet_b: string;
  score_diff: number;
  findings_diff: {
    added: GauntletFinding[];
    removed: GauntletFinding[];
    unchanged: number;
  };
  category_comparison: Record<string, {
    a_count: number;
    b_count: number;
    diff: number;
  }>;
  improvement_rate: number;
}

// =============================================================================
// Replay Types
// =============================================================================

export interface ReplaySummary {
  replay_id: string;
  debate_id: string;
  created_at: string;
  duration_ms: number;
  event_count: number;
}

export interface ReplayEvent {
  event_type: string;
  timestamp: string;
  data: Record<string, unknown>;
}

export interface Replay {
  replay_id: string;
  debate_id: string;
  events: ReplayEvent[];
  created_at: string;
  metadata?: Record<string, unknown>;
}

// =============================================================================
// WebSocket Types
// =============================================================================

// Stream event types (matches server StreamEventType plus control messages).
export type DebateEventType =
  // Control messages
  | 'connection_info'
  | 'loop_list'
  | 'sync'
  | 'ack'
  | 'auth_revoked'
  // Debate lifecycle
  | 'debate_start'
  | 'round_start'
  | 'agent_message'
  | 'critique'
  | 'vote'
  | 'consensus'
  | 'debate_end'
  // Token streaming
  | 'token_start'
  | 'token_delta'
  | 'token_end'
  // Nomic loop
  | 'cycle_start'
  | 'cycle_end'
  | 'phase_start'
  | 'phase_end'
  | 'task_start'
  | 'task_complete'
  | 'task_retry'
  | 'verification_start'
  | 'verification_result'
  | 'commit'
  | 'backup_created'
  | 'backup_restored'
  | 'error'
  | 'log_message'
  // Multi-loop management
  | 'loop_register'
  | 'loop_unregister'
  // Audience participation
  | 'user_vote'
  | 'user_suggestion'
  | 'audience_summary'
  | 'audience_metrics'
  | 'audience_drain'
  // Memory & learning
  | 'memory_recall'
  | 'insight_extracted'
  // Rankings & leaderboard
  | 'match_recorded'
  | 'leaderboard_update'
  | 'grounded_verdict'
  | 'moment_detected'
  | 'agent_elo_updated'
  // Claim verification
  | 'claim_verification_result'
  | 'formal_verification_result'
  // Memory tiers
  | 'memory_tier_promotion'
  | 'memory_tier_demotion'
  // Graph debates
  | 'graph_node_added'
  | 'graph_branch_created'
  | 'graph_branch_merged'
  // Position tracking
  | 'flip_detected'
  // Feature integration
  | 'trait_emerged'
  | 'risk_warning'
  | 'evidence_found'
  | 'calibration_update'
  | 'genesis_evolution'
  | 'training_data_exported'
  // Rhetorical analysis
  | 'rhetorical_observation'
  // Trickster events
  | 'hollow_consensus'
  | 'trickster_intervention'
  // Breakpoints
  | 'breakpoint'
  | 'breakpoint_resolved'
  // Mood/sentiment
  | 'mood_detected'
  | 'mood_shift'
  | 'debate_energy'
  // Capability probes
  | 'probe_start'
  | 'probe_result'
  | 'probe_complete'
  // Deep audit
  | 'audit_start'
  | 'audit_round'
  | 'audit_finding'
  | 'audit_cross_exam'
  | 'audit_verdict'
  // Telemetry
  | 'telemetry_thought'
  | 'telemetry_capability'
  | 'telemetry_redaction'
  | 'telemetry_diagnostic'
  // Gauntlet
  | 'gauntlet_start'
  | 'gauntlet_phase'
  | 'gauntlet_agent_active'
  | 'gauntlet_attack'
  | 'gauntlet_finding'
  | 'gauntlet_probe'
  | 'gauntlet_verification'
  | 'gauntlet_risk'
  | 'gauntlet_progress'
  | 'gauntlet_verdict'
  | 'gauntlet_complete'
  // Analytics
  | 'uncertainty_analysis'
  // Client keepalive / legacy
  | 'ping'
  | 'pong'
  | 'round_end';

export interface DebateEvent {
  type: DebateEventType;
  data: Record<string, unknown>;
  timestamp?: number;
  round?: number;
  agent?: string;
  seq?: number;
  agent_seq?: number;
  loop_id?: string;
  debate_id?: string;
}

export interface WebSocketOptions {
  reconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
}

// =============================================================================
// Client Types
// =============================================================================

export interface RetryOptions {
  /** Maximum number of retry attempts (default: 3) */
  maxRetries?: number;
  /** Initial delay in milliseconds (default: 1000) */
  initialDelay?: number;
  /** Maximum delay in milliseconds (default: 30000) */
  maxDelay?: number;
  /** Multiplier for exponential backoff (default: 2) */
  backoffMultiplier?: number;
  /** Whether to add jitter to delays (default: true) */
  jitter?: boolean;
}

/**
 * Configuration options for the Aragora client.
 *
 * @example
 * ```typescript
 * // Basic configuration
 * const client = new AragoraClient({
 *   baseUrl: 'https://api.aragora.ai',
 *   apiKey: process.env.ARAGORA_API_KEY,
 * });
 *
 * // Advanced configuration with retry options
 * const client = new AragoraClient({
 *   baseUrl: 'https://api.aragora.ai',
 *   apiKey: process.env.ARAGORA_API_KEY,
 *   timeout: 60000,
 *   retry: {
 *     maxRetries: 5,
 *     initialDelay: 2000,
 *   },
 * });
 * ```
 */
export interface AragoraClientOptions {
  /**
   * Base URL of the Aragora API server.
   * For local development: 'http://localhost:8080'
   * For production: 'https://api.aragora.ai'
   */
  baseUrl: string;

  /**
   * API key for authentication.
   * Obtain from your Aragora dashboard or via client.auth.createApiKey()
   */
  apiKey?: string;

  /**
   * Default request timeout in milliseconds (default: 30000).
   * Can be overridden per-request via RequestOptions.
   */
  timeout?: number;

  /**
   * Additional headers to include with every request.
   * Useful for custom tracking or proxy authentication.
   */
  headers?: Record<string, string>;

  /**
   * Retry configuration for failed requests.
   * By default, retries 3 times with exponential backoff.
   */
  retry?: RetryOptions;
}

export interface RequestOptions {
  timeout?: number;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  /** Override retry settings for this request */
  retry?: RetryOptions | false;
}

export interface ApiError {
  error: string;
  code: string;
  details?: string;
  suggestion?: string;
}

// =============================================================================
// Pulse Types (Trending Topics)
// =============================================================================

export interface TrendingTopic {
  topic: string;
  source: string;
  score: number;
  url?: string;
  timestamp?: string;
  category?: string;
}

export interface PulseTrendingResponse {
  topics: TrendingTopic[];
  sources: string[];
  updated_at: string;
}

export interface PulseSuggestResponse {
  suggestion: TrendingTopic;
  reason: string;
}

export interface PulseAnalyticsResponse {
  total_debates: number;
  topics_debated: number;
  avg_consensus_rate: number;
  top_categories: Array<{ category: string; count: number }>;
  period_days: number;
}

export interface PulseDebateRequest {
  topic: TrendingTopic;
  agents?: string[];
  max_rounds?: number;
}

export interface PulseSchedulerStatus {
  running: boolean;
  paused: boolean;
  next_run?: string;
  interval_minutes: number;
  debates_today: number;
  last_error?: string;
}

export interface PulseSchedulerConfig {
  enabled?: boolean;
  interval_minutes?: number;
  max_debates_per_day?: number;
  sources?: string[];
}

export interface PulseSchedulerHistoryEntry {
  debate_id: string;
  topic: string;
  platform: string;
  started_at: string;
  completed_at?: string;
  status: string;
}

// =============================================================================
// Documents Types
// =============================================================================

export interface Document {
  doc_id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  uploaded_at: string;
  metadata?: Record<string, unknown>;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
}

export interface DocumentUploadResponse {
  doc_id: string;
  filename: string;
  size_bytes: number;
  content_type: string;
}

export interface DocumentFormatsResponse {
  formats: Array<{
    extension: string;
    content_type: string;
    max_size_mb: number;
  }>;
}

// =============================================================================
// Breakpoints Types (Human-in-the-Loop)
// =============================================================================

export interface Breakpoint {
  breakpoint_id: string;
  debate_id: string;
  round: number;
  reason: string;
  agent_id?: string;
  created_at: string;
  status: 'pending' | 'resolved' | 'skipped';
  context?: Record<string, unknown>;
}

export interface BreakpointResolveRequest {
  action: 'continue' | 'modify' | 'stop';
  modification?: string;
  feedback?: string;
}

export interface BreakpointResolveResponse {
  breakpoint_id: string;
  status: string;
  action_taken: string;
}

export interface BreakpointStatusResponse {
  breakpoint_id: string;
  status: 'pending' | 'resolved' | 'skipped';
  resolved_at?: string;
  action_taken?: string;
}

// =============================================================================
// Health Types
// =============================================================================

export interface HealthCheck {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime_seconds: number;
  services?: Record<string, boolean>;
}

export interface DeepHealthCheck {
  status: 'healthy' | 'degraded' | 'healthy_with_warnings';
  healthy: boolean;
  checks: Record<string, {
    healthy: boolean;
    status?: string;
    error?: string;
    warning?: string;
    latency_ms?: number;
  }>;
  warnings: string[];
  response_time_ms: number;
  timestamp: string;
  version: string;
}

// =============================================================================
// Tournament Types
// =============================================================================

export interface TournamentSummary {
  tournament_id: string;
  participants: number;
  total_matches: number;
  top_agent: string | null;
}

export interface TournamentStanding {
  agent: string;
  wins: number;
  losses: number;
  draws: number;
  points: number;
  total_score: number;
  win_rate: number;
}

export interface TournamentListResponse {
  tournaments: TournamentSummary[];
  count: number;
}

export interface TournamentStandingsResponse {
  tournament_id: string;
  standings: TournamentStanding[];
  count: number;
}

// =============================================================================
// Organization Types
// =============================================================================

export interface Organization {
  id: string;
  name: string;
  slug: string;
  tier: string;
  owner_id: string;
  member_count: number;
  debates_used: number;
  debates_limit: number;
  settings: Record<string, unknown>;
  created_at: string;
}

export interface OrganizationMember {
  id: string;
  email: string;
  name: string;
  role: 'member' | 'admin' | 'owner';
  is_active: boolean;
  created_at: string;
  last_login_at: string | null;
}

export interface OrganizationInvitation {
  id: string;
  org_id: string;
  email: string;
  role: 'member' | 'admin';
  invited_by: string;
  status: 'pending' | 'accepted' | 'revoked' | 'expired';
  expires_at: string;
  created_at: string;
  org_name?: string;
}

export interface InviteRequest {
  email: string;
  role?: 'member' | 'admin';
}

export interface InviteResponse {
  message: string;
  invitation_id?: string;
  user_id?: string;
  role: string;
  expires_at?: string;
  invite_link?: string;
}

export interface OrganizationUpdateRequest {
  name?: string;
  settings?: Record<string, unknown>;
}

// =============================================================================
// Analytics Types
// =============================================================================

export interface AnalyticsOverview {
  total_debates: number;
  active_debates: number;
  completed_debates: number;
  failed_debates: number;
  avg_debate_duration_seconds: number;
  consensus_rate: number;
  period_days: number;
}

export interface AnalyticsAgentStats {
  agent_id: string;
  debates_participated: number;
  wins: number;
  losses: number;
  draws: number;
  avg_contribution_score: number;
}

export interface AnalyticsResponse {
  overview: AnalyticsOverview;
  top_agents: AnalyticsAgentStats[];
  debates_by_day: Array<{ date: string; count: number }>;
}

// =============================================================================
// Extended Analytics Types
// =============================================================================

export interface DisagreementPoint {
  topic: string;
  agents: string[];
  positions: Record<string, string>;
  severity: 'minor' | 'moderate' | 'major';
  debate_ids: string[];
}

export interface DisagreementAnalysis {
  total_disagreements: number;
  period_days: number;
  top_disagreements: DisagreementPoint[];
  by_domain: Record<string, number>;
  by_agent_pair: Array<{
    agents: [string, string];
    count: number;
    topics: string[];
  }>;
  resolution_rate: number;
}

export interface RoleRotationStats {
  period_days: number;
  total_role_changes: number;
  agents: Array<{
    agent: string;
    roles_held: string[];
    role_distribution: Record<string, number>;
    avg_role_duration: number;
  }>;
  most_versatile_agents: string[];
  role_effectiveness: Record<string, {
    win_rate: number;
    consensus_contribution: number;
  }>;
}

export interface EarlyStopAnalysis {
  period_days: number;
  total_debates: number;
  early_stops: number;
  early_stop_rate: number;
  reasons: Record<string, number>;
  avg_rounds_before_stop: number;
  consensus_in_early_stops: number;
  by_agent: Record<string, {
    involved_in: number;
    caused_by: number;
  }>;
}

export interface ConsensusQualityMetrics {
  period_days: number;
  total_consensus: number;
  avg_confidence: number;
  confidence_distribution: Array<{
    range: string;
    count: number;
    percentage: number;
  }>;
  stability_rate: number;
  reversal_rate: number;
  by_domain: Record<string, {
    consensus_rate: number;
    avg_confidence: number;
    avg_rounds: number;
  }>;
  quality_factors: {
    evidence_backing: number;
    participant_agreement: number;
    logical_coherence: number;
  };
}

export interface RankingStatistics {
  total_agents: number;
  total_matches: number;
  elo_range: { min: number; max: number; median: number };
  rating_changes_today: number;
  most_active_agents: Array<{
    agent: string;
    matches_30d: number;
    elo_change_30d: number;
  }>;
  biggest_upsets: Array<{
    winner: string;
    loser: string;
    elo_diff: number;
    debate_id: string;
    timestamp: string;
  }>;
}

export interface MemoryStatistics {
  total_memories: number;
  total_size_bytes: number;
  by_tier: Record<string, {
    count: number;
    size_bytes: number;
    avg_access_count: number;
  }>;
  hit_rate: number;
  promotion_rate: number;
  demotion_rate: number;
  cleanup_pending: number;
  last_consolidation?: string;
}

// =============================================================================
// Batch Debate Types
// =============================================================================

export interface BatchDebateItem {
  question: string;
  agents?: string;
  rounds?: number;
  consensus?: string;
  priority?: number;
  metadata?: Record<string, unknown>;
}

export interface BatchDebateRequest {
  items: BatchDebateItem[];
  webhook_url?: string;
  webhook_headers?: Record<string, string>;
  max_parallel?: number;
}

export interface BatchDebateResponse {
  success: boolean;
  batch_id: string;
  items_queued: number;
  estimated_completion_minutes?: number;
}

export interface BatchStatusItem {
  question: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  debate_id?: string;
  error?: string;
}

export interface BatchStatusResponse {
  batch_id: string;
  status: 'pending' | 'processing' | 'completed' | 'partial_failure';
  total_items: number;
  completed_items: number;
  failed_items: number;
  items: BatchStatusItem[];
  estimated_completion?: string;
}

export interface QueueStatusResponse {
  queue_length: number;
  processing: number;
  completed_today: number;
  failed_today: number;
  avg_wait_seconds: number;
  estimated_wait_seconds?: number;
}

// =============================================================================
// Extended Debate Types
// =============================================================================

export interface ImpasseInfo {
  has_impasse: boolean;
  impasse_round?: number;
  disagreement_points?: string[];
  suggested_resolution?: string;
}

export interface ConvergenceInfo {
  convergence_reached: boolean;
  convergence_round?: number;
  similarity_score: number;
  key_agreements: string[];
  remaining_disagreements?: string[];
}

export interface Citation {
  citation_id: string;
  agent_id: string;
  text: string;
  source?: string;
  url?: string;
  round: number;
  verified?: boolean;
}

export interface DebateCitationsResponse {
  debate_id: string;
  citations: Citation[];
  total: number;
}

export interface DebateMessagesResponse {
  debate_id: string;
  messages: DebateMessage[];
  total: number;
  limit: number;
  offset: number;
}

export interface Evidence {
  evidence_id: string;
  content: string;
  source: string;
  agent_id: string;
  round: number;
  confidence?: number;
  verified?: boolean;
}

export interface DebateEvidenceResponse {
  debate_id: string;
  evidence: Evidence[];
  total: number;
}

export interface DebateSummary {
  debate_id: string;
  task: string;
  summary: string;
  key_points: string[];
  consensus_reached: boolean;
  conclusion?: string;
  duration_seconds: number;
}

export interface FollowupSuggestion {
  crux: string;
  suggested_question: string;
  priority: 'high' | 'medium' | 'low';
  reasoning: string;
}

export interface FollowupSuggestionsResponse {
  debate_id: string;
  suggestions: FollowupSuggestion[];
}

export interface ForkRequest {
  branch_point: number;
  modified_context?: string;
}

export interface ForkResponse {
  debate_id: string;
  parent_debate_id: string;
  branch_point: number;
  status: string;
}

export interface FollowupRequest {
  crux: string;
  agents?: string[];
  max_rounds?: number;
}

export interface FollowupResponse {
  debate_id: string;
  parent_debate_id: string;
  crux: string;
  status: string;
}

export interface DebateExportResponse {
  debate_id: string;
  format: string;
  data: string;
  filename: string;
}

export interface DebateSearchResult {
  debate_id: string;
  task: string;
  status: string;
  consensus_reached: boolean;
  score: number;
  snippet: string;
  created_at: string;
}

export interface DebateSearchResponse {
  query: string;
  results: DebateSearchResult[];
  total: number;
  limit: number;
  offset: number;
}

export interface DebateVerificationReport {
  debate_id: string;
  claims_verified: number;
  claims_disputed: number;
  claims_unverified: number;
  verification_results: Array<{
    claim: string;
    status: 'verified' | 'disputed' | 'unverified';
    evidence?: string;
    confidence: number;
  }>;
  generated_at: string;
}

export interface DebateShareResponse {
  debate_id: string;
  share_url: string;
  share_token: string;
  expires_at?: string;
  message: string;
}

export interface DebateBroadcastResponse {
  debate_id: string;
  title: string;
  summary: string;
  key_points: string[];
  conclusion?: string;
  audio_url?: string;
  transcript?: string;
}

export interface DebatePublishResponse {
  debate_id: string;
  platform: string;
  status: 'published' | 'pending' | 'failed';
  url?: string;
  message: string;
}

// =============================================================================
// Extended Memory Types (Continuum)
// =============================================================================

export interface ContinuumMemory {
  memory_id: string;
  content: string;
  tier: string;
  created_at: string;
  last_accessed?: string;
  access_count: number;
  metadata?: Record<string, unknown>;
}

export interface ContinuumRetrieveResponse {
  memories: ContinuumMemory[];
  tier: string;
  total: number;
  limit: number;
  offset: number;
}

export interface ContinuumConsolidateResponse {
  success: boolean;
  memories_consolidated: number;
  promotions: number;
  demotions: number;
}

export interface ContinuumCleanupResponse {
  success: boolean;
  memories_removed: number;
  bytes_freed?: number;
}

export interface TierStatsResponse {
  tiers: MemoryTierStats[];
  total_entries: number;
}

export interface ArchiveStatsResponse {
  total_archived: number;
  size_bytes: number;
  oldest_entry?: string;
  newest_entry?: string;
}

export interface MemoryPressureResponse {
  pressure_level: 'low' | 'medium' | 'high' | 'critical';
  utilization_percent: number;
  recommended_action?: string;
}

// =============================================================================
// Auth Types
// =============================================================================

export interface RegisterRequest {
  email: string;
  password: string;
  name?: string;
}

export interface RegisterResponse {
  user_id: string;
  email: string;
  message: string;
}

export interface LoginRequest {
  email: string;
  password: string;
  mfa_code?: string;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: AuthUser;
}

export interface AuthUser {
  user_id: string;
  email: string;
  name?: string;
  role: string;
  org_id?: string;
  mfa_enabled: boolean;
  created_at: string;
  last_login_at?: string;
}

export interface RefreshRequest {
  refresh_token: string;
}

export interface RefreshResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface RevokeTokenRequest {
  token: string;
  token_type?: 'access' | 'refresh';
}

export interface UpdateMeRequest {
  name?: string;
  email?: string;
}

export interface ChangePasswordRequest {
  current_password: string;
  new_password: string;
}

export interface ChangePasswordResponse {
  message: string;
}

export interface ApiKeyCreateRequest {
  name?: string;
  expires_in_days?: number;
}

export interface ApiKeyResponse {
  api_key: string;
  key_id: string;
  name?: string;
  expires_at?: string;
  message: string;
}

export interface MfaSetupResponse {
  secret: string;
  qr_code: string;
  provisioning_uri: string;
  message: string;
}

export interface MfaEnableRequest {
  code: string;
}

export interface MfaEnableResponse {
  message: string;
  backup_codes: string[];
}

export interface MfaDisableRequest {
  code: string;
  password: string;
}

export interface MfaVerifyRequest {
  code: string;
}

export interface MfaVerifyResponse {
  valid: boolean;
  message: string;
}

export interface MfaBackupCodesResponse {
  backup_codes: string[];
  message: string;
}

// =============================================================================
// OAuth Types
// =============================================================================

export interface OAuthProvider {
  name: string;
  enabled: boolean;
  auth_url?: string;
}

export interface OAuthProvidersResponse {
  providers: OAuthProvider[];
}

export interface OAuthInitResponse {
  redirect_url: string;
  state: string;
}

// =============================================================================
// Billing Types
// =============================================================================

export interface BillingPlan {
  id: string;
  name: string;
  tier: 'free' | 'starter' | 'professional' | 'enterprise';
  price_monthly: number;
  price_yearly: number;
  currency: string;
  features: string[];
  limits: {
    debates_per_month: number;
    api_calls_per_minute: number;
    storage_gb: number;
    team_members: number;
  };
  stripe_price_id_monthly?: string;
  stripe_price_id_yearly?: string;
}

export interface BillingPlansResponse {
  plans: BillingPlan[];
}

export interface BillingUsage {
  debates_used: number;
  debates_limit: number;
  api_calls_used: number;
  api_calls_limit: number;
  storage_used_bytes: number;
  storage_limit_bytes: number;
  period_start: string;
  period_end: string;
  usage_percent: number;
}

export interface BillingSubscription {
  id: string;
  status: 'active' | 'canceled' | 'past_due' | 'trialing' | 'incomplete';
  plan: BillingPlan;
  current_period_start: string;
  current_period_end: string;
  cancel_at_period_end: boolean;
  trial_end?: string;
  stripe_subscription_id?: string;
}

export interface CheckoutRequest {
  plan_id: string;
  billing_period: 'monthly' | 'yearly';
  success_url?: string;
  cancel_url?: string;
}

export interface CheckoutResponse {
  checkout_url: string;
  session_id: string;
}

export interface PortalResponse {
  portal_url: string;
}

export interface CancelSubscriptionResponse {
  message: string;
  cancel_at: string;
}

export interface ResumeSubscriptionResponse {
  message: string;
  subscription: BillingSubscription;
}

export interface BillingAuditEntry {
  id: string;
  timestamp: string;
  action: string;
  user_id: string;
  details: Record<string, unknown>;
}

export interface BillingAuditLogResponse {
  entries: BillingAuditEntry[];
  total: number;
  limit: number;
  offset: number;
}

export interface UsageForecast {
  projected_debates: number;
  projected_api_calls: number;
  projected_cost: number;
  days_remaining: number;
  will_exceed_limit: boolean;
  exceed_date?: string;
}

export interface Invoice {
  id: string;
  number: string;
  status: 'draft' | 'open' | 'paid' | 'void' | 'uncollectible';
  amount_due: number;
  amount_paid: number;
  currency: string;
  created_at: string;
  due_date?: string;
  paid_at?: string;
  pdf_url?: string;
  hosted_invoice_url?: string;
}

export interface InvoicesResponse {
  invoices: Invoice[];
  total: number;
}

// =============================================================================
// Evidence Types
// =============================================================================

export interface EvidenceSnippet {
  id: string;
  source: string;
  title: string;
  snippet: string;
  url?: string;
  reliability_score: number;
  freshness_score?: number;
  quality_score?: number;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

export interface EvidenceListResponse {
  evidence: EvidenceSnippet[];
  total: number;
  limit: number;
  offset: number;
}

export interface EvidenceSearchRequest {
  query: string;
  limit?: number;
  source?: string;
  min_reliability?: number;
  context?: {
    topic?: string;
    required_sources?: string[];
    recency_weight?: number;
  };
}

export interface EvidenceSearchResponse {
  query: string;
  results: EvidenceSnippet[];
  count: number;
}

export interface EvidenceCollectRequest {
  task: string;
  connectors?: string[];
  debate_id?: string;
  round?: number;
}

export interface EvidenceCollectResponse {
  task: string;
  keywords: string[];
  snippets: EvidenceSnippet[];
  count: number;
  total_searched: number;
  average_reliability: number;
  average_freshness: number;
  saved_ids: string[];
  debate_id?: string;
}

export interface EvidenceStatistics {
  total_evidence: number;
  total_debates: number;
  avg_reliability: number;
  sources_distribution: Record<string, number>;
}

export interface EvidenceDebateResponse {
  debate_id: string;
  round?: number;
  evidence: EvidenceSnippet[];
  count: number;
}

// =============================================================================
// Calibration Types
// =============================================================================

export interface CalibrationBucket {
  bucket: string;
  predicted: number;
  actual: number;
  count: number;
}

export interface CalibrationCurveResponse {
  agent: string;
  buckets: CalibrationBucket[];
  expected_calibration_error: number;
  domain?: string;
}

export interface CalibrationSummary {
  agent: string;
  brier_score: number;
  log_loss: number;
  ece: number;
  total_predictions: number;
  accuracy: number;
  overconfidence_rate: number;
  underconfidence_rate: number;
  domain?: string;
}

export interface CalibrationLeaderboardEntry {
  agent: string;
  elo: number;
  calibration_score: number;
  brier_score: number;
  accuracy: number;
  games: number;
}

export interface CalibrationLeaderboardResponse {
  agents: CalibrationLeaderboardEntry[];
  metric: string;
  min_predictions: number;
}

export interface CalibrationVisualizationResponse {
  agents: Array<{
    name: string;
    buckets: CalibrationBucket[];
    ece: number;
  }>;
}

// =============================================================================
// Insights Types
// =============================================================================

export interface Insight {
  id: string;
  type: string;
  title: string;
  description: string;
  confidence: number;
  agents_involved: string[];
  evidence: string[];
  timestamp?: string;
}

export interface InsightsRecentResponse {
  insights: Insight[];
  count: number;
}

export interface FlipEvent {
  id: string;
  agent: string;
  type: 'contradiction' | 'retraction' | 'qualification' | 'refinement';
  type_emoji: string;
  before: { claim: string; confidence: string };
  after: { claim: string; confidence: string };
  similarity: string;
  domain?: string;
  timestamp: string;
}

export interface FlipsRecentResponse {
  flips: FlipEvent[];
  count: number;
}

export interface ExtractInsightsRequest {
  content: string;
  context?: string;
}

export interface ExtractInsightsResponse {
  insights: Insight[];
  extraction_time_ms: number;
}

// =============================================================================
// Belief Network Types
// =============================================================================

export interface Crux {
  claim_id: string;
  statement: string;
  author: string;
  crux_score: number;
  centrality: number;
  entropy: number;
  current_belief: {
    true_prob: number;
    false_prob: number;
    uncertain_prob: number;
    confidence: number;
  };
}

export interface CruxesResponse {
  debate_id: string;
  cruxes: Crux[];
  count: number;
}

export interface LoadBearingClaim {
  claim_id: string;
  statement: string;
  author: string;
  centrality: number;
  degree: number;
  downstream_claims: number;
  belief_state: {
    true_prob: number;
    false_prob: number;
    uncertain_prob: number;
  };
}

export interface LoadBearingClaimsResponse {
  debate_id: string;
  claims: LoadBearingClaim[];
  count: number;
}

export interface ClaimSupportResponse {
  claim_id: string;
  statement: string;
  verification_status: 'verified' | 'disputed' | 'unverified';
  supporting_evidence: Array<{
    evidence_id: string;
    text: string;
    source?: string;
    reliability: number;
  }>;
  contradicting_evidence: Array<{
    evidence_id: string;
    text: string;
    source?: string;
    reliability: number;
  }>;
}

export interface GraphStatsResponse {
  debate_id: string;
  total_claims: number;
  total_edges: number;
  average_degree: number;
  max_depth: number;
  connected_components: number;
  claim_types: Record<string, number>;
}

// =============================================================================
// Consensus Types
// =============================================================================

export interface SimilarDebate {
  debate_id: string;
  topic: string;
  similarity: number;
  consensus_reached: boolean;
  conclusion?: string;
  timestamp: string;
}

export interface ConsensusSimilarResponse {
  query: string;
  similar_debates: SimilarDebate[];
  count: number;
}

export interface SettledTopic {
  topic: string;
  conclusion: string;
  confidence: number;
  supporting_debates: number;
  last_updated: string;
  domain?: string;
}

export interface ConsensusSettledResponse {
  topics: SettledTopic[];
  count: number;
  min_confidence: number;
}

export interface ConsensusStatsResponse {
  total_debates: number;
  consensus_rate: number;
  average_confidence: number;
  topics_by_domain: Record<string, number>;
  debates_by_day: Array<{ date: string; count: number }>;
}

export interface Dissent {
  id: string;
  topic: string;
  agent: string;
  dissenting_view: string;
  majority_view: string;
  confidence: number;
  timestamp: string;
}

export interface DissentsResponse {
  dissents: Dissent[];
  count: number;
}

export interface ContrarianView {
  id: string;
  topic: string;
  view: string;
  reasoning: string;
  strength: number;
  timestamp: string;
}

export interface ContrarianViewsResponse {
  views: ContrarianView[];
  count: number;
}

export interface RiskWarning {
  id: string;
  topic: string;
  warning: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  edge_cases: string[];
  timestamp: string;
}

export interface RiskWarningsResponse {
  warnings: RiskWarning[];
  count: number;
}

export interface DomainHistoryResponse {
  domain: string;
  debates: Array<{
    debate_id: string;
    topic: string;
    consensus_reached: boolean;
    conclusion?: string;
    timestamp: string;
  }>;
  count: number;
}

// =============================================================================
// Admin Types
// =============================================================================

export interface AdminUser {
  id: string;
  email: string;
  name?: string;
  role: string;
  org_id?: string;
  is_active: boolean;
  mfa_enabled: boolean;
  created_at: string;
  last_login_at?: string;
  metadata?: Record<string, unknown>;
}

export interface AdminUsersResponse {
  users: AdminUser[];
  total: number;
  limit: number;
  offset: number;
}

export interface AdminUserUpdateRequest {
  name?: string;
  email?: string;
  role?: string;
  is_active?: boolean;
  metadata?: Record<string, unknown>;
}

export interface AdminOrganization {
  id: string;
  name: string;
  slug: string;
  tier: string;
  owner_id: string;
  member_count: number;
  debates_used: number;
  debates_limit: number;
  is_active: boolean;
  settings: Record<string, unknown>;
  created_at: string;
}

export interface AdminOrganizationsResponse {
  organizations: AdminOrganization[];
  total: number;
  limit: number;
  offset: number;
}

export interface AdminSystemStats {
  total_users: number;
  active_users: number;
  total_organizations: number;
  total_debates: number;
  debates_today: number;
  active_debates: number;
  total_api_calls: number;
  api_calls_today: number;
  storage_used_bytes: number;
  uptime_seconds: number;
  version: string;
}

// =============================================================================
// Dashboard Types
// =============================================================================

export interface DashboardAnalyticsParams {
  start_date?: string;
  end_date?: string;
  granularity?: 'hour' | 'day' | 'week' | 'month';
  metrics?: string[];
}

export interface DashboardAnalyticsResponse {
  period: {
    start: string;
    end: string;
    granularity: string;
  };
  metrics: {
    debates_created: number;
    debates_completed: number;
    consensus_rate: number;
    avg_duration_seconds: number;
    unique_users: number;
    api_calls: number;
  };
  timeseries: Array<{
    timestamp: string;
    debates: number;
    api_calls: number;
    users: number;
  }>;
}

export interface DashboardDebateMetricsParams {
  days?: number;
  agent?: string;
  status?: string;
}

export interface DashboardDebateMetricsResponse {
  total_debates: number;
  completed_debates: number;
  failed_debates: number;
  avg_rounds: number;
  avg_duration_seconds: number;
  consensus_rate: number;
  by_status: Record<string, number>;
  by_agent: Record<string, number>;
  top_topics: Array<{ topic: string; count: number }>;
}

export interface DashboardAgentPerformanceParams {
  days?: number;
  min_debates?: number;
}

export interface DashboardAgentPerformanceResponse {
  agents: Array<{
    agent_id: string;
    name: string;
    debates_participated: number;
    wins: number;
    losses: number;
    draws: number;
    win_rate: number;
    elo_rating: number;
    avg_response_time_ms: number;
    calibration_score?: number;
  }>;
  period_days: number;
}

// =============================================================================
// System Types
// =============================================================================

export interface SystemHealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  checks: Record<string, {
    healthy: boolean;
    latency_ms?: number;
    error?: string;
  }>;
}

export interface SystemInfoResponse {
  version: string;
  environment: string;
  build_date?: string;
  commit_sha?: string;
  python_version: string;
  platform: string;
  features: string[];
  api_version: string;
}

export interface SystemStatusResponse {
  status: 'operational' | 'degraded' | 'partial_outage' | 'major_outage';
  services: Record<string, {
    status: 'operational' | 'degraded' | 'down';
    latency_ms?: number;
    last_checked: string;
    error?: string;
  }>;
  active_incidents: Array<{
    id: string;
    title: string;
    status: string;
    started_at: string;
    updated_at: string;
  }>;
  uptime_percent_30d: number;
  last_incident?: string;
}

// =============================================================================
// Features Types
// =============================================================================

export interface FeatureFlag {
  name: string;
  enabled: boolean;
  description?: string;
  rollout_percent?: number;
  target_groups?: string[];
  metadata?: Record<string, unknown>;
}

export interface FeaturesListResponse {
  features: FeatureFlag[];
  count: number;
}

export interface FeatureStatusResponse {
  name: string;
  enabled: boolean;
  description?: string;
  rollout_percent?: number;
}

// =============================================================================
// Checkpoint Types
// =============================================================================

export interface Checkpoint {
  checkpoint_id: string;
  debate_id: string;
  name?: string;
  round: number;
  state: Record<string, unknown>;
  created_at: string;
  size_bytes: number;
}

export interface CheckpointListResponse {
  checkpoints: Checkpoint[];
  total: number;
  limit: number;
  offset: number;
}

export interface ResumableDebate {
  debate_id: string;
  task: string;
  checkpoint_id: string;
  round: number;
  status: string;
  created_at: string;
  checkpoint_created_at: string;
}

// =============================================================================
// Webhook Types
// =============================================================================

export interface Webhook {
  webhook_id: string;
  url: string;
  events: string[];
  secret?: string;
  active: boolean;
  created_at: string;
  last_triggered?: string;
  failure_count: number;
}

export interface WebhookEventType {
  name: string;
  description: string;
  payload_schema?: Record<string, unknown>;
}

export interface WebhookCreateRequest {
  url: string;
  events: string[];
  secret?: string;
}

export interface WebhookUpdateRequest {
  url?: string;
  events?: string[];
  secret?: string;
  active?: boolean;
}

export interface WebhookTestResult {
  success: boolean;
  status_code?: number;
  response_time_ms: number;
  error?: string;
}

// =============================================================================
// Training Export Types
// =============================================================================

export interface TrainingExportOptions {
  format?: string;
  start_date?: string;
  end_date?: string;
  min_confidence?: number;
  include_metadata?: boolean;
  limit?: number;
}

export interface TrainingExportResponse {
  export_id: string;
  format: string;
  records_count: number;
  size_bytes: number;
  download_url: string;
  expires_at: string;
}

export interface TrainingStats {
  total_debates: number;
  total_rounds: number;
  total_messages: number;
  consensus_rate: number;
  available_formats: string[];
  estimated_tokens: number;
}

export interface TrainingFormat {
  name: string;
  description: string;
  extension: string;
  supports_metadata: boolean;
}

// =============================================================================
// Metrics Types
// =============================================================================

export interface SystemMetrics {
  debates_total: number;
  debates_active: number;
  agents_total: number;
  requests_per_minute: number;
  avg_response_time_ms: number;
  error_rate: number;
  uptime_seconds: number;
}

export interface HealthMetrics {
  status: 'healthy' | 'degraded' | 'unhealthy';
  checks: Record<string, {
    healthy: boolean;
    latency_ms?: number;
    last_check: string;
  }>;
  warnings: string[];
}

export interface CacheMetrics {
  hit_rate: number;
  miss_rate: number;
  entries: number;
  size_bytes: number;
  evictions: number;
  ttl_expires: number;
}

export interface VerificationMetrics {
  total_verifications: number;
  success_rate: number;
  avg_verification_time_ms: number;
  by_backend: Record<string, {
    count: number;
    success_rate: number;
    avg_time_ms: number;
  }>;
}

export interface SystemResourceMetrics {
  cpu_percent: number;
  memory_percent: number;
  memory_used_mb: number;
  memory_available_mb: number;
  disk_percent: number;
  open_connections: number;
  active_threads: number;
}

export interface BackgroundJobMetrics {
  queued: number;
  processing: number;
  completed_today: number;
  failed_today: number;
  avg_processing_time_ms: number;
  by_type: Record<string, {
    queued: number;
    processing: number;
    completed: number;
    failed: number;
  }>;
}

// =============================================================================
// Routing Types
// =============================================================================

export interface TeamRecommendation {
  agents: string[];
  score: number;
  strengths: string[];
  suggested_roles?: Record<string, string>;
}

export interface RoutingRecommendation {
  task: string;
  detected_domain: string;
  recommended_agents: string[];
  alternative_agents: string[];
  confidence: number;
  reasoning: string;
}

export interface AutoRouteRequest {
  task: string;
  constraints?: {
    required_agents?: string[];
    excluded_agents?: string[];
    max_agents?: number;
    min_agents?: number;
  };
  preferences?: {
    diversity?: number;
    expertise_weight?: number;
  };
}

export interface AutoRouteResponse {
  task: string;
  selected_agents: string[];
  domain: string;
  confidence: number;
  debate_id?: string;
}

export interface DomainDetectionResult {
  task: string;
  detected_domain: string;
  confidence: number;
  alternative_domains: Array<{
    domain: string;
    confidence: number;
  }>;
}

export interface DomainLeaderboardEntry {
  agent: string;
  domain: string;
  elo: number;
  wins: number;
  losses: number;
  win_rate: number;
  calibration_score?: number;
}
