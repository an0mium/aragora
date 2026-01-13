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

export type DebateEventType =
  | 'debate_start'
  | 'round_start'
  | 'round_end'
  | 'agent_message'
  | 'critique'
  | 'vote'
  | 'consensus'
  | 'debate_end'
  | 'error';

export interface DebateEvent {
  type: DebateEventType;
  debate_id: string;
  timestamp: string;
  data: Record<string, unknown>;
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
