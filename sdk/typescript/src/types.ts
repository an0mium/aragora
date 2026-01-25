/**
 * Aragora SDK Type Definitions
 *
 * Auto-generated types from OpenAPI spec with additional SDK-specific types.
 */

// =============================================================================
// Configuration Types
// =============================================================================

export interface AragoraConfig {
  /** Base URL of the Aragora API (e.g., "https://api.aragora.ai") */
  baseUrl: string;
  /** API key for authentication */
  apiKey?: string;
  /** Additional headers to include in all requests */
  headers?: Record<string, string>;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Enable automatic retries on transient failures */
  retryEnabled?: boolean;
  /** Maximum number of retry attempts (default: 3) */
  maxRetries?: number;
  /** WebSocket URL override (defaults to ws:// or wss:// version of baseUrl) */
  wsUrl?: string;
}

// =============================================================================
// Error Types
// =============================================================================

export type ErrorCode =
  | 'INVALID_JSON'
  | 'MISSING_FIELD'
  | 'INVALID_VALUE'
  | 'AUTH_REQUIRED'
  | 'INVALID_TOKEN'
  | 'FORBIDDEN'
  | 'NOT_OWNER'
  | 'NOT_FOUND'
  | 'QUOTA_EXCEEDED'
  | 'RATE_LIMITED'
  | 'INTERNAL_ERROR'
  | 'SERVICE_UNAVAILABLE'
  | 'AGENT_TIMEOUT'
  | 'CONSENSUS_FAILED';

export interface ApiError {
  error: string;
  code?: ErrorCode;
  trace_id?: string;
  field?: string;
  resource_type?: string;
  resource_id?: string;
  limit?: number;
  retry_after?: number;
  resets_at?: string;
  upgrade_url?: string;
  support_url?: string;
}

export class AragoraError extends Error {
  constructor(
    message: string,
    public code?: ErrorCode,
    public status?: number,
    public traceId?: string,
    public details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'AragoraError';
  }

  static fromResponse(status: number, body: ApiError): AragoraError {
    return new AragoraError(body.error, body.code, status, body.trace_id, body as unknown as Record<string, unknown>);
  }
}

// =============================================================================
// Common Types
// =============================================================================

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  offset: number;
  limit: number;
  has_more: boolean;
}

export interface PaginationParams {
  [key: string]: string | number | boolean | undefined;
  offset?: number;
  limit?: number;
}

// =============================================================================
// Agent Types
// =============================================================================

export interface Agent {
  name: string;
  elo?: number;
  matches?: number;
  wins?: number;
  losses?: number;
  calibration_score?: number;
  persona?: string;
  domains?: string[];
}

export interface AgentProfile extends Agent {
  reputation?: number;
  consistency_score?: number;
  flip_rate?: number;
  allies?: string[];
  rivals?: string[];
}

// =============================================================================
// Debate Types
// =============================================================================

export type DebateStatus =
  | 'created'
  | 'starting'
  | 'pending'
  | 'running'
  | 'in_progress'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'paused'
  | 'active'
  | 'concluded'
  | 'archived';

export type ConsensusType = 'majority' | 'unanimous' | 'weighted' | 'semantic';

export interface ConsensusResult {
  reached: boolean;
  agreement?: number;
  confidence?: number;
  final_answer?: string;
  conclusion?: string;
  supporting_agents?: string[];
  dissenting_agents?: string[];
}

export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
  agent?: string;
  agent_id?: string;
  round?: number;
  timestamp?: string;
}

export interface Round {
  round_number: number;
  messages: Message[];
  votes?: Record<string, unknown>;
  summary?: string;
}

export interface Debate {
  debate_id: string;
  id?: string;
  slug?: string;
  task: string;
  topic?: string;
  context?: string;
  status: DebateStatus;
  outcome?: string;
  final_answer?: string;
  consensus?: ConsensusResult;
  consensus_proof?: Record<string, unknown>;
  consensus_reached?: boolean;
  confidence?: number;
  rounds_used?: number;
  duration_seconds?: number;
  agents: string[];
  rounds?: Round[];
  created_at: string;
  completed_at?: string;
  metadata?: Record<string, unknown>;
}

export interface DebateCreateRequest {
  task: string;
  question?: string;
  agents?: string[];
  rounds?: number;
  consensus?: ConsensusType;
  context?: string;
  auto_select?: boolean;
  auto_select_config?: Record<string, unknown>;
  use_trending?: boolean;
  trending_category?: 'tech' | 'science' | 'politics' | 'business' | 'health';
}

export interface DebateCreateResponse {
  success: boolean;
  debate_id: string;
  status: DebateStatus;
  task: string;
  agents: string[];
  websocket_url?: string;
  estimated_duration?: number;
}

// =============================================================================
// Workflow Types
// =============================================================================

export interface StepDefinition {
  id: string;
  name: string;
  type: string;
  config?: Record<string, unknown>;
  depends_on?: string[];
}

export interface TransitionRule {
  from_step: string;
  to_step: string;
  condition?: string;
}

export interface Workflow {
  id: string;
  name: string;
  description?: string;
  version?: string;
  status?: string;
  steps?: StepDefinition[];
  transitions?: TransitionRule[];
  created_at?: string;
  updated_at?: string;
}

export interface WorkflowTemplate {
  id: string;
  name: string;
  description?: string;
  category?: string;
  pattern?: string;
  steps?: StepDefinition[];
  parameters?: Record<string, unknown>;
}

// =============================================================================
// Gauntlet Types
// =============================================================================

export interface DecisionReceipt {
  id: string;
  debate_id: string;
  verdict: string;
  confidence?: number;
  consensus_reached?: boolean;
  participating_agents?: string[];
  dissenting_agents?: string[];
  evidence?: Record<string, unknown>[];
  reasoning?: string;
  hash?: string;
  created_at?: string;
  metadata?: Record<string, unknown>;
}

export interface RiskHeatmap {
  id: string;
  gauntlet_id?: string;
  categories: string[];
  scores: number[];
  matrix?: number[][];
  overall_risk?: number;
  created_at?: string;
  metadata?: Record<string, unknown>;
}

// =============================================================================
// Explainability Types
// =============================================================================

export interface ExplanationFactor {
  name: string;
  contribution: number;
  description?: string;
  evidence?: string[];
}

export interface CounterfactualScenario {
  hypothesis: string;
  predicted_outcome: string;
  confidence: number;
  affected_agents?: string[];
}

export interface ExplainabilityResult {
  debate_id: string;
  factors?: ExplanationFactor[];
  counterfactuals?: CounterfactualScenario[];
  provenance?: Record<string, unknown>;
  narrative?: string;
}

// =============================================================================
// WebSocket Event Types
// =============================================================================

export type WebSocketEventType =
  | 'connected'
  | 'debate_start'
  | 'round_start'
  | 'round_end'
  | 'agent_message'
  | 'propose'
  | 'critique'
  | 'revision'
  | 'synthesis'
  | 'vote'
  | 'consensus'
  | 'consensus_reached'
  | 'debate_end'
  | 'phase_change'
  | 'audience_suggestion'
  | 'user_vote'
  | 'error'
  | 'warning'
  | 'heartbeat';

export interface WebSocketEvent<T = unknown> {
  type: WebSocketEventType;
  debate_id?: string;
  loop_id?: string;
  timestamp: string;
  data: T;
}

export interface DebateStartEvent {
  debate_id: string;
  task: string;
  agents: string[];
  total_rounds: number;
}

export interface RoundStartEvent {
  debate_id: string;
  round_number: number;
}

export interface AgentMessageEvent {
  debate_id: string;
  round_number: number;
  agent: string;
  content: string;
  confidence?: number;
}

export interface CritiqueEvent {
  debate_id: string;
  round_number: number;
  critic: string;
  target: string;
  critique: string;
  severity?: 'minor' | 'moderate' | 'major';
}

export interface VoteEvent {
  debate_id: string;
  round_number: number;
  agent: string;
  vote: string;
  confidence?: number;
}

export interface ConsensusEvent {
  debate_id: string;
  consensus: ConsensusResult;
}

export interface DebateEndEvent {
  debate_id: string;
  status: DebateStatus;
  result?: Debate;
}

export interface SynthesisEvent {
  debate_id: string;
  round_number: number;
  synthesis: string;
  confidence?: number;
}

export interface RevisionEvent {
  debate_id: string;
  round_number: number;
  agent: string;
  original: string;
  revised: string;
}

export interface PhaseChangeEvent {
  debate_id: string;
  from_phase: string;
  to_phase: string;
}

export interface AudienceSuggestionEvent {
  debate_id: string;
  user_id: string;
  suggestion: string;
  round_number?: number;
}

export interface UserVoteEvent {
  debate_id: string;
  user_id: string;
  vote: string;
  weight?: number;
}

export interface ErrorEvent {
  debate_id?: string;
  code: string;
  message: string;
  recoverable?: boolean;
}

export interface WarningEvent {
  debate_id?: string;
  code: string;
  message: string;
}

// =============================================================================
// Health Types
// =============================================================================

export interface HealthCheck {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version?: string;
  timestamp: string;
  checks?: Record<string, { status: string; latency_ms?: number }>;
  response_time_ms?: number;
}

// =============================================================================
// Control Plane Types
// =============================================================================

export type AgentStatus = 'idle' | 'busy' | 'offline' | 'draining';
export type TaskStatus = 'pending' | 'claimed' | 'running' | 'completed' | 'failed' | 'cancelled' | 'timeout';
export type TaskPriority = 'low' | 'normal' | 'high' | 'critical';

export interface RegisteredAgent {
  agent_id: string;
  name?: string;
  capabilities?: string[];
  status: AgentStatus;
  registered_at: string;
  last_heartbeat?: string;
  current_task?: string;
  metadata?: Record<string, unknown>;
}

export interface Task {
  task_id: string;
  task_type: string;
  payload: Record<string, unknown>;
  priority: TaskPriority;
  status: TaskStatus;
  submitted_at: string;
  claimed_at?: string;
  completed_at?: string;
  assigned_agent?: string;
  result?: unknown;
  error?: string;
  metadata?: Record<string, unknown>;
}

export interface TaskSubmitRequest {
  task_type: string;
  payload: Record<string, unknown>;
  priority?: TaskPriority;
  agent_hint?: string;
  timeout_seconds?: number;
  metadata?: Record<string, unknown>;
}

export interface AgentRegisterRequest {
  agent_id: string;
  name?: string;
  capabilities?: string[];
  metadata?: Record<string, unknown>;
}

export interface HeartbeatRequest {
  agent_id: string;
  status?: AgentStatus;
  current_task?: string;
  metrics?: Record<string, number>;
}

export interface ControlPlaneHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  agents_total: number;
  agents_active: number;
  tasks_pending: number;
  tasks_running: number;
  uptime_seconds?: number;
}

// =============================================================================
// Marketplace Types
// =============================================================================

export interface MarketplaceTemplate {
  id: string;
  name: string;
  description: string;
  author: string;
  category: string;
  downloads: number;
  rating: number;
  tags: string[];
  created_at?: string;
  updated_at?: string;
}

export interface TemplateReview {
  review_id: string;
  template_id: string;
  user_id: string;
  rating: number;
  title: string;
  content: string;
  created_at: string;
}

// =============================================================================
// Matrix Debate Types
// =============================================================================

export interface MatrixDebateCreateRequest {
  task: string;
  scenarios: string[];
  agents?: string[];
  max_rounds?: number;
  consensus_threshold?: number;
  parallel?: boolean;
  metadata?: Record<string, unknown>;
}

export interface MatrixDebate {
  id: string;
  task: string;
  scenarios: string[];
  status: DebateStatus;
  created_at: string;
  completed_at?: string;
  conclusions?: Record<string, MatrixConclusion>;
  metadata?: Record<string, unknown>;
}

export interface MatrixConclusion {
  scenario: string;
  conclusion: string;
  confidence: number;
  supporting_agents: string[];
  dissenting_agents?: string[];
}

// =============================================================================
// Graph Debate Types
// =============================================================================

export interface GraphDebateCreateRequest {
  task: string;
  agents?: string[];
  max_rounds?: number;
  branch_threshold?: number;
  max_branches?: number;
  metadata?: Record<string, unknown>;
}

export interface GraphDebate {
  id: string;
  task: string;
  status: DebateStatus;
  branches: GraphBranch[];
  created_at: string;
  completed_at?: string;
  metadata?: Record<string, unknown>;
}

export interface GraphBranch {
  branch_id: string;
  parent_id?: string;
  divergence_point: number;
  hypothesis: string;
  messages: Message[];
  conclusion?: string;
  confidence?: number;
}

// =============================================================================
// Verification Types
// =============================================================================

export type VerificationBackend = 'z3' | 'lean' | 'auto';

export interface VerifyClaimRequest {
  claim: string;
  context?: string;
  backend?: VerificationBackend;
  timeout_seconds?: number;
}

export interface VerificationResult {
  verified: boolean;
  backend: VerificationBackend;
  proof?: string;
  counterexample?: string;
  assumptions?: string[];
  duration_ms: number;
  trace_id?: string;
}

export interface VerificationStatus {
  z3_available: boolean;
  lean_available: boolean;
  default_backend: VerificationBackend;
}

// =============================================================================
// Selection API Types
// =============================================================================

export interface SelectionPlugin {
  name: string;
  description: string;
  version: string;
  capabilities: string[];
}

export interface AgentScore {
  agent_id: string;
  score: number;
  factors: Record<string, number>;
  confidence: number;
}

export interface ScoreAgentsRequest {
  task_type: string;
  required_capabilities?: string[];
  context?: string;
  limit?: number;
}

export interface TeamSelectionRequest {
  task_type: string;
  team_size: number;
  required_capabilities?: string[];
  diversity_weight?: number;
  quality_weight?: number;
  context?: string;
}

export interface TeamSelection {
  agents: AgentScore[];
  total_score: number;
  diversity_score: number;
  coverage: Record<string, string[]>;
}

// =============================================================================
// Replay Types
// =============================================================================

export type ReplayFormat = 'json' | 'markdown' | 'html';

export interface Replay {
  id: string;
  debate_id: string;
  name?: string;
  duration_ms: number;
  message_count: number;
  created_at: string;
  metadata?: Record<string, unknown>;
}

export interface ReplayExport {
  format: ReplayFormat;
  content: string;
  filename: string;
}

// =============================================================================
// Memory Analytics Types
// =============================================================================

export type MemoryTier = 'fast' | 'medium' | 'slow' | 'glacial';

export interface MemoryAnalytics {
  period_start: string;
  period_end: string;
  total_entries: number;
  entries_by_tier: Record<MemoryTier, number>;
  consolidation_events: number;
  promotions: number;
  demotions: number;
  avg_importance: number;
  storage_bytes: number;
}

export interface MemoryTierStats {
  tier: MemoryTier;
  entry_count: number;
  avg_importance: number;
  avg_consolidation: number;
  oldest_entry?: string;
  newest_entry?: string;
  utilization_pct: number;
}

// =============================================================================
// Extended Agent Types (Profile, Performance, etc.)
// =============================================================================

export interface AgentCalibration {
  agent: string;
  overall_score: number;
  domain_scores: Record<string, number>;
  confidence_accuracy: number;
  last_calibrated?: string;
  sample_size: number;
}

export interface AgentPerformance {
  agent: string;
  win_rate: number;
  loss_rate: number;
  draw_rate: number;
  elo_trend: number[];
  elo_change_30d: number;
  avg_confidence: number;
  avg_round_duration_ms: number;
  total_debates: number;
  recent_results: Array<{ debate_id: string; outcome: string; date: string }>;
}

export interface HeadToHeadStats {
  agent: string;
  opponent: string;
  total_matchups: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  avg_margin: number;
  recent_matchups: Array<{
    debate_id: string;
    outcome: string;
    margin: number;
    date: string;
  }>;
  domain_breakdown?: Record<string, { wins: number; losses: number }>;
}

export interface OpponentBriefing {
  agent: string;
  opponent: string;
  opponent_profile: {
    elo: number;
    strengths: string[];
    weaknesses: string[];
    preferred_domains: string[];
  };
  historical_summary: string;
  recommended_strategy: string;
  key_insights: string[];
  confidence: number;
}

export interface AgentConsistency {
  agent: string;
  overall_consistency: number;
  position_stability: number;
  flip_rate: number;
  consistency_by_domain: Record<string, number>;
  volatility_index: number;
  sample_size: number;
}

export interface AgentFlip {
  flip_id: string;
  agent: string;
  debate_id: string;
  topic: string;
  original_position: string;
  new_position: string;
  flip_reason?: string;
  round_number: number;
  timestamp: string;
  was_justified: boolean;
}

export interface AgentNetwork {
  agent: string;
  allies: Array<{
    agent: string;
    agreement_rate: number;
    shared_debates: number;
  }>;
  rivals: Array<{
    agent: string;
    disagreement_rate: number;
    shared_debates: number;
  }>;
  neutrals: string[];
  cluster_id?: string;
  network_position: 'central' | 'peripheral' | 'bridge';
}

export interface AgentMoment {
  moment_id: string;
  agent: string;
  debate_id: string;
  type: 'breakthrough' | 'comeback' | 'decisive_argument' | 'consensus_catalyst' | 'upset';
  description: string;
  impact_score: number;
  timestamp: string;
  context: {
    round: number;
    opponent?: string;
    topic?: string;
  };
}

export interface AgentPosition {
  position_id: string;
  agent: string;
  debate_id: string;
  topic: string;
  stance: string;
  confidence: number;
  supporting_evidence: string[];
  round_number: number;
  timestamp: string;
  was_final: boolean;
}

export interface DomainRating {
  domain: string;
  elo: number;
  matches: number;
  win_rate: number;
  avg_confidence: number;
  last_active?: string;
  trend: 'rising' | 'stable' | 'falling';
}

// =============================================================================
// Gauntlet Types (Extended)
// =============================================================================

export interface GauntletRun {
  id: string;
  name?: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
  started_at?: string;
  completed_at?: string;
  config: {
    scenarios: string[];
    personas?: string[];
    max_rounds?: number;
    parallel?: boolean;
  };
  progress: {
    total: number;
    completed: number;
    failed: number;
  };
  results_summary?: {
    pass_rate: number;
    avg_confidence: number;
    risk_score: number;
  };
  metadata?: Record<string, unknown>;
}

export interface GauntletPersona {
  id: string;
  name: string;
  description: string;
  category: 'adversarial' | 'edge_case' | 'stress' | 'compliance' | 'custom';
  severity: 'low' | 'medium' | 'high' | 'critical';
  tags: string[];
  example_prompts?: string[];
  enabled: boolean;
}

export interface GauntletResult {
  id: string;
  gauntlet_id: string;
  scenario: string;
  persona?: string;
  status: 'pass' | 'fail' | 'error' | 'skip';
  verdict: string;
  confidence: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  duration_ms: number;
  debate_id?: string;
  findings: Array<{
    type: string;
    description: string;
    severity: string;
  }>;
  timestamp: string;
}

export interface GauntletHeatmapExtended {
  gauntlet_id: string;
  dimensions: {
    categories: string[];
    personas: string[];
  };
  matrix: number[][];
  overall_risk: number;
  hotspots: Array<{
    category: string;
    persona: string;
    risk: number;
  }>;
  generated_at: string;
}

export interface GauntletComparison {
  gauntlet_a: string;
  gauntlet_b: string;
  comparison: {
    pass_rate_delta: number;
    risk_score_delta: number;
    new_failures: string[];
    fixed_failures: string[];
    regression_count: number;
    improvement_count: number;
  };
  scenario_diffs: Array<{
    scenario: string;
    a_status: string;
    b_status: string;
    delta: string;
  }>;
  recommendation: 'promote' | 'investigate' | 'block';
  generated_at: string;
}

export interface GauntletRunRequest {
  /** The input text to run through the gauntlet */
  input: string;
  /** Optional gauntlet profile to use */
  profile?: string;
  /** List of persona categories to include */
  personas?: string[];
  /** Maximum rounds per scenario */
  max_rounds?: number;
  /** Run scenarios in parallel */
  parallel?: boolean;
  /** Optional metadata */
  metadata?: Record<string, unknown>;
}

export interface GauntletRunResponse {
  gauntlet_id: string;
  status: 'pending' | 'running';
  status_url: string;
  estimated_duration_seconds?: number;
}

// =============================================================================
// Knowledge Types
// =============================================================================

export interface KnowledgeEntry {
  id?: string;
  content: string;
  source?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
  importance?: number;
  visibility?: 'private' | 'team' | 'global';
  expires_at?: string;
}

export interface KnowledgeSearchResult {
  id: string;
  content: string;
  score: number;
  source?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface KnowledgeStats {
  total_entries: number;
  by_visibility: Record<string, number>;
  by_source: Record<string, number>;
  storage_bytes: number;
}

// =============================================================================
// Analytics Types
// =============================================================================

export interface DisagreementAnalytics {
  period: string;
  total_debates: number;
  disagreement_rate: number;
  avg_dissent_count: number;
  top_disagreement_topics: Array<{
    topic: string;
    count: number;
    avg_resolution_rounds: number;
  }>;
  agent_disagreement_matrix: Record<string, Record<string, number>>;
  persistent_disagreements: Array<{
    agents: string[];
    topic_pattern: string;
    occurrence_count: number;
  }>;
}

export interface RoleRotationAnalytics {
  period: string;
  total_assignments: number;
  role_distribution: Record<string, number>;
  agent_role_frequency: Record<string, Record<string, number>>;
  rotation_fairness_index: number;
  stuck_agents: Array<{
    agent: string;
    dominant_role: string;
    frequency: number;
  }>;
  recommendations: string[];
}

export interface EarlyStopAnalytics {
  period: string;
  total_debates: number;
  early_stop_rate: number;
  avg_rounds_saved: number;
  early_stop_reasons: Record<string, number>;
  confidence_at_stop: {
    avg: number;
    min: number;
    max: number;
    distribution: number[];
  };
  false_early_stops: number;
  missed_early_stops: number;
}

export interface ConsensusQualityAnalytics {
  period: string;
  total_consensuses: number;
  quality_distribution: Record<'high' | 'medium' | 'low', number>;
  avg_confidence: number;
  avg_agreement_level: number;
  hollow_consensus_rate: number;
  contested_consensus_rate: number;
  consensus_durability: {
    stable: number;
    challenged: number;
    overturned: number;
  };
  quality_by_topic: Record<string, number>;
}

export interface RankingStats {
  total_agents: number;
  elo_distribution: {
    min: number;
    max: number;
    mean: number;
    median: number;
    std_dev: number;
  };
  tier_distribution: Record<string, number>;
  top_performers: Array<{
    agent: string;
    elo: number;
    trend: string;
  }>;
  most_improved: Array<{
    agent: string;
    elo_change: number;
    period: string;
  }>;
  last_updated: string;
}

export interface MemoryStats {
  total_entries: number;
  storage_bytes: number;
  tier_counts: Record<MemoryTier, number>;
  consolidation_rate: number;
  avg_importance: number;
  cache_hit_rate: number;
  oldest_entry?: string;
  newest_entry?: string;
  health_status: 'healthy' | 'degraded' | 'critical';
}

// =============================================================================
// Debate Update Types
// =============================================================================

export interface DebateUpdateRequest {
  status?: DebateStatus;
  metadata?: Record<string, unknown>;
  tags?: string[];
  archived?: boolean;
  notes?: string;
}

export interface VerificationReport {
  debate_id: string;
  verified: boolean;
  verification_method: string;
  claims_verified: number;
  claims_failed: number;
  claims_skipped: number;
  claim_details: Array<{
    claim: string;
    verified: boolean;
    confidence: number;
    evidence?: string;
    counterevidence?: string;
  }>;
  overall_confidence: number;
  verification_duration_ms: number;
  generated_at: string;
}

export interface SearchResult {
  type: 'debate' | 'agent' | 'memory' | 'claim';
  id: string;
  title?: string;
  snippet: string;
  score: number;
  highlights: string[];
  metadata?: Record<string, unknown>;
  created_at?: string;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total_count: number;
  facets?: Record<string, Record<string, number>>;
  suggestions?: string[];
  took_ms: number;
}

// =============================================================================
// Memory Search Types
// =============================================================================

export interface MemoryEntry {
  id: string;
  tier: MemoryTier;
  content: string;
  importance: number;
  consolidation_count: number;
  source_debate_id?: string;
  agent?: string;
  tags?: string[];
  created_at: string;
  accessed_at?: string;
  expires_at?: string;
  metadata?: Record<string, unknown>;
}

export interface MemorySearchParams {
  query: string;
  tiers?: MemoryTier[];
  agent?: string;
  limit?: number;
  min_importance?: number;
  include_expired?: boolean;
}

export interface CritiqueEntry {
  id: string;
  debate_id: string;
  critic_agent: string;
  target_agent: string;
  critique: string;
  severity: 'minor' | 'moderate' | 'major';
  was_addressed: boolean;
  resolution?: string;
  round_number: number;
  timestamp: string;
  impact_score?: number;
}

// =============================================================================
// Auth Types
// =============================================================================

export interface RegisterRequest {
  email: string;
  password: string;
  name?: string;
  organization?: string;
  metadata?: Record<string, unknown>;
}

export interface RegisterResponse {
  user_id: string;
  email: string;
  name?: string;
  requires_verification: boolean;
  created_at: string;
}

export interface LoginRequest {
  email: string;
  password: string;
  mfa_code?: string;
}

export interface AuthToken {
  access_token: string;
  refresh_token?: string;
  token_type: string;
  expires_in: number;
  expires_at?: string;
  scope?: string;
}

export interface RefreshRequest {
  refresh_token: string;
}

export interface VerifyEmailRequest {
  token: string;
}

export interface VerifyResponse {
  verified: boolean;
  email: string;
  user_id: string;
}

export interface User {
  id: string;
  email: string;
  name?: string;
  organization?: string;
  avatar_url?: string;
  roles?: string[];
  permissions?: string[];
  mfa_enabled: boolean;
  email_verified: boolean;
  created_at: string;
  updated_at?: string;
  last_login?: string;
  metadata?: Record<string, unknown>;
}

export interface UpdateProfileRequest {
  name?: string;
  organization?: string;
  avatar_url?: string;
  metadata?: Record<string, unknown>;
}

export interface UpdateProfileResponse {
  updated: boolean;
  user: User;
}

export interface ChangePasswordRequest {
  current_password: string;
  new_password: string;
}

export interface ForgotPasswordRequest {
  email: string;
}

export interface ResetPasswordRequest {
  token: string;
  new_password: string;
}

export interface OAuthUrlParams {
  provider: 'google' | 'github' | 'microsoft' | 'okta' | 'saml';
  redirect_uri?: string;
  state?: string;
  scope?: string;
}

export interface OAuthUrl {
  url: string;
  state: string;
  provider: string;
}

export interface OAuthCallbackRequest {
  code: string;
  state: string;
  provider: string;
}

export interface MFASetupRequest {
  type: 'totp' | 'sms' | 'email';
  phone_number?: string;
}

export interface MFASetupResponse {
  secret?: string;
  qr_code_url?: string;
  backup_codes?: string[];
  verification_required: boolean;
}

export interface MFAVerifyRequest {
  code: string;
  type: 'totp' | 'sms' | 'email';
}

export interface MFAVerifyResponse {
  verified: boolean;
  mfa_enabled: boolean;
}

// =============================================================================
// Tenancy Types
// =============================================================================

export interface Tenant {
  id: string;
  name: string;
  slug?: string;
  description?: string;
  owner_id: string;
  plan?: string;
  status: 'active' | 'suspended' | 'pending' | 'deleted';
  settings?: Record<string, unknown>;
  created_at: string;
  updated_at?: string;
  metadata?: Record<string, unknown>;
}

export interface CreateTenantRequest {
  name: string;
  slug?: string;
  description?: string;
  plan?: string;
  settings?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface UpdateTenantRequest {
  name?: string;
  description?: string;
  plan?: string;
  status?: 'active' | 'suspended';
  settings?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface TenantList {
  tenants: Tenant[];
  total: number;
  offset: number;
  limit: number;
}

export interface QuotaStatus {
  tenant_id: string;
  quotas: {
    debates_per_month: { used: number; limit: number };
    agents_per_debate: { used: number; limit: number };
    storage_bytes: { used: number; limit: number };
    api_calls_per_minute: { used: number; limit: number };
    members: { used: number; limit: number };
  };
  period_start: string;
  period_end: string;
  overage_allowed: boolean;
}

export interface QuotaUpdate {
  debates_per_month?: number;
  agents_per_debate?: number;
  storage_bytes?: number;
  api_calls_per_minute?: number;
  members?: number;
  overage_allowed?: boolean;
}

export interface TenantMember {
  user_id: string;
  email: string;
  name?: string;
  role: string;
  joined_at: string;
  invited_by?: string;
  status: 'active' | 'invited' | 'suspended';
}

export interface MemberList {
  members: TenantMember[];
  total: number;
  offset: number;
  limit: number;
}

export interface AddMemberRequest {
  email: string;
  role: string;
  send_invitation?: boolean;
}

// =============================================================================
// RBAC Types
// =============================================================================

export interface Permission {
  id: string;
  name: string;
  description?: string;
  resource: string;
  action: string;
  conditions?: Record<string, unknown>;
}

export interface Role {
  id: string;
  name: string;
  description?: string;
  permissions: string[];
  inherits_from?: string[];
  is_system: boolean;
  tenant_id?: string;
  created_at: string;
  updated_at?: string;
}

export interface RoleList {
  roles: Role[];
  total: number;
  offset: number;
  limit: number;
}

export interface CreateRoleRequest {
  name: string;
  description?: string;
  permissions: string[];
  inherits_from?: string[];
}

export interface UpdateRoleRequest {
  name?: string;
  description?: string;
  permissions?: string[];
  inherits_from?: string[];
}

export interface PermissionList {
  permissions: Permission[];
  total: number;
}

export interface PermissionCheck {
  allowed: boolean;
  permission: string;
  user_id: string;
  matched_role?: string;
  conditions_met?: boolean;
  reason?: string;
}

export interface RoleAssignment {
  user_id: string;
  role_id: string;
  tenant_id?: string;
  assigned_at: string;
  assigned_by?: string;
  expires_at?: string;
}

export interface AssignmentList {
  assignments: RoleAssignment[];
  total: number;
  offset: number;
  limit: number;
}

export interface BulkAssignRequest {
  assignments: Array<{
    user_id: string;
    role_id: string;
    tenant_id?: string;
    expires_at?: string;
  }>;
}

export interface BulkAssignResponse {
  assigned: number;
  failed: number;
  errors?: Array<{ user_id: string; error: string }>;
}

// =============================================================================
// Tournament Types
// =============================================================================

export interface Tournament {
  id: string;
  name: string;
  description?: string;
  status: 'pending' | 'active' | 'completed' | 'cancelled';
  format: 'single_elimination' | 'double_elimination' | 'round_robin' | 'swiss';
  participants: string[];
  current_round: number;
  total_rounds: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  winner?: string;
  metadata?: Record<string, unknown>;
}

export interface CreateTournamentRequest {
  name: string;
  description?: string;
  format?: 'single_elimination' | 'double_elimination' | 'round_robin' | 'swiss';
  participants: string[];
  config?: {
    debate_rounds?: number;
    consensus_type?: string;
    auto_advance?: boolean;
  };
  metadata?: Record<string, unknown>;
}

export interface TournamentStandings {
  tournament_id: string;
  standings: Array<{
    rank: number;
    agent: string;
    wins: number;
    losses: number;
    draws: number;
    points: number;
    elo_change: number;
  }>;
  updated_at: string;
}

export interface TournamentBracket {
  tournament_id: string;
  rounds: Array<{
    round_number: number;
    matches: TournamentMatch[];
  }>;
}

export interface TournamentMatch {
  id: string;
  tournament_id: string;
  round: number;
  participant_a: string;
  participant_b: string;
  status: 'pending' | 'in_progress' | 'completed' | 'bye';
  winner?: string;
  debate_id?: string;
  scheduled_at?: string;
  completed_at?: string;
  notes?: string;
}

// =============================================================================
// Audit Types
// =============================================================================

export interface AuditEvent {
  id: string;
  timestamp: string;
  actor_id: string;
  actor_type: 'user' | 'system' | 'api_key' | 'service';
  action: string;
  resource_type: string;
  resource_id: string;
  tenant_id?: string;
  ip_address?: string;
  user_agent?: string;
  changes?: Record<string, { old: unknown; new: unknown }>;
  metadata?: Record<string, unknown>;
  success: boolean;
  error_message?: string;
}

export interface AuditStats {
  period: string;
  total_events: number;
  events_by_action: Record<string, number>;
  events_by_resource: Record<string, number>;
  events_by_actor: Record<string, number>;
  failed_events: number;
  unique_actors: number;
}

export interface AuditSession {
  id: string;
  name: string;
  description?: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'cancelled' | 'failed';
  target_type: 'codebase' | 'debate' | 'system' | 'custom';
  target_id?: string;
  config?: Record<string, unknown>;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  created_by: string;
  findings_count?: number;
  severity_breakdown?: {
    critical: number;
    high: number;
    medium: number;
    low: number;
    info: number;
  };
  metadata?: Record<string, unknown>;
}

export interface CreateAuditSessionRequest {
  name: string;
  description?: string;
  target_type: 'codebase' | 'debate' | 'system' | 'custom';
  target_id?: string;
  config?: {
    probes?: string[];
    depth?: 'shallow' | 'standard' | 'deep';
    timeout_seconds?: number;
    parallel?: boolean;
  };
  metadata?: Record<string, unknown>;
}

export interface AuditFinding {
  id: string;
  session_id: string;
  type: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  title: string;
  description: string;
  location?: {
    file?: string;
    line?: number;
    column?: number;
    context?: string;
  };
  evidence?: string[];
  recommendations?: string[];
  status: 'open' | 'acknowledged' | 'resolved' | 'false_positive';
  assigned_to?: string;
  created_at: string;
  resolved_at?: string;
  metadata?: Record<string, unknown>;
}

// =============================================================================
// Onboarding Types
// =============================================================================

export interface OnboardingStatus {
  completed: boolean;
  organization_id: string;
  completed_at?: string;
  first_debate_id?: string;
  template_used?: string;
  steps: {
    signup: boolean;
    organization_created: boolean;
    first_debate: boolean;
    first_receipt: boolean;
  };
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
  features: string[];
  limits: {
    debates_per_month: number;
    agents_per_debate: number;
    storage_gb: number;
    api_calls_per_minute: number;
  };
}

export interface BillingPlanList {
  plans: BillingPlan[];
}

export interface BillingUsage {
  period_start: string;
  period_end: string;
  debates_used: number;
  debates_limit: number;
  tokens_used: number;
  cost_usd: number;
  by_provider: Record<string, { tokens: number; cost: number }>;
  by_feature: Record<string, { count: number; cost: number }>;
}

export interface Subscription {
  id: string;
  plan_id: string;
  plan_name: string;
  status: 'active' | 'cancelled' | 'past_due' | 'trialing' | 'incomplete';
  current_period_start: string;
  current_period_end: string;
  cancel_at_period_end: boolean;
  created_at: string;
  stripe_subscription_id?: string;
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

export interface InvoiceList {
  invoices: Invoice[];
  total: number;
  has_more: boolean;
}

export interface UsageForecast {
  projected_monthly_cost: number;
  projected_debates: number;
  projected_tokens: number;
  recommended_tier: string;
  savings_with_upgrade?: number;
  days_until_limit?: number;
}

// =============================================================================
// Notification Types
// =============================================================================

export interface NotificationStatus {
  email_configured: boolean;
  telegram_configured: boolean;
  slack_configured: boolean;
  last_notification_sent?: string;
  notifications_sent_today: number;
}

export interface EmailNotificationConfig {
  provider: 'smtp' | 'sendgrid' | 'ses';
  smtp_host?: string;
  smtp_port?: number;
  smtp_username?: string;
  smtp_password?: string;
  use_tls?: boolean;
  from_email: string;
  from_name?: string;
  notify_on_consensus?: boolean;
  notify_on_debate_end?: boolean;
  notify_on_error?: boolean;
  enable_digest?: boolean;
  digest_frequency?: 'daily' | 'weekly';
}

export interface TelegramNotificationConfig {
  bot_token: string;
  chat_id: string;
  notify_on_consensus?: boolean;
  notify_on_debate_end?: boolean;
  notify_on_error?: boolean;
}

export interface NotificationRecipient {
  email: string;
  name?: string;
  preferences: {
    consensus: boolean;
    debate_end: boolean;
    errors: boolean;
    digest: boolean;
  };
  added_at: string;
}

// =============================================================================
// Budget Types
// =============================================================================

export interface Budget {
  id: string;
  name: string;
  description?: string;
  limit_amount: number;
  spent_amount: number;
  currency: string;
  period: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'yearly';
  period_start: string;
  period_end: string;
  status: 'active' | 'exceeded' | 'warning' | 'inactive';
  alert_threshold: number;
  created_at: string;
  updated_at?: string;
  metadata?: Record<string, unknown>;
}

export interface BudgetList {
  budgets: Budget[];
  total: number;
  offset: number;
  limit: number;
}

export interface CreateBudgetRequest {
  name: string;
  description?: string;
  limit_amount: number;
  currency?: string;
  period: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'yearly';
  alert_threshold?: number;
  metadata?: Record<string, unknown>;
}

export interface UpdateBudgetRequest {
  name?: string;
  description?: string;
  limit_amount?: number;
  alert_threshold?: number;
  status?: 'active' | 'inactive';
  metadata?: Record<string, unknown>;
}

export interface BudgetAlert {
  id: string;
  budget_id: string;
  type: 'budget_warning' | 'spike_detected' | 'limit_reached' | 'anomaly';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  threshold_pct: number;
  current_pct: number;
  acknowledged: boolean;
  acknowledged_at?: string;
  acknowledged_by?: string;
  created_at: string;
}

export interface BudgetAlertList {
  alerts: BudgetAlert[];
  total: number;
}

export interface BudgetSummary {
  total_budget: number;
  total_spent: number;
  total_remaining: number;
  active_budgets: number;
  exceeded_budgets: number;
  warning_budgets: number;
  currency: string;
}

// =============================================================================
// Cost Types
// =============================================================================

export interface CostDashboard {
  period: string;
  total_cost: number;
  cost_change_pct: number;
  debates_run: number;
  tokens_used: number;
  avg_cost_per_debate: number;
  top_providers: Array<{ provider: string; cost: number; percentage: number }>;
  budget_status: {
    limit: number;
    spent: number;
    remaining: number;
    pct_used: number;
  };
}

export interface CostBreakdown {
  period: string;
  by_provider: Record<string, { cost: number; tokens: number; debates: number }>;
  by_feature: Record<string, { cost: number; count: number }>;
  by_model: Record<string, { cost: number; tokens: number }>;
  by_user?: Record<string, { cost: number; debates: number }>;
}

export interface CostTimeline {
  period: string;
  granularity: 'hourly' | 'daily' | 'weekly' | 'monthly';
  data_points: Array<{
    timestamp: string;
    cost: number;
    tokens: number;
    debates: number;
  }>;
}

export interface CostAlert {
  id: string;
  type: 'budget_exceeded' | 'spike_detected' | 'anomaly' | 'forecast_warning';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  details?: Record<string, unknown>;
  dismissed: boolean;
  created_at: string;
}

// =============================================================================
// Audit Trail Types
// =============================================================================

export interface AuditTrail {
  id: string;
  debate_id?: string;
  gauntlet_id?: string;
  created_at: string;
  verdict: string;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  checksum: string;
  findings: Array<{
    type: string;
    severity: string;
    description: string;
  }>;
  metadata?: Record<string, unknown>;
}

export interface AuditTrailList {
  trails: AuditTrail[];
  total: number;
  offset: number;
  limit: number;
}
