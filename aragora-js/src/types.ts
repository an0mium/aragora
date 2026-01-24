/**
 * Type definitions for the Aragora TypeScript SDK.
 */

export type DebateStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
export type AgentStatus = 'starting' | 'ready' | 'busy' | 'draining' | 'offline' | 'failed';
export type TaskStatus = 'pending' | 'assigned' | 'running' | 'completed' | 'failed' | 'cancelled' | 'timeout';
export type TaskPriority = 'low' | 'normal' | 'high' | 'urgent';

export interface ConsensusResult {
  reached: boolean;
  conclusion?: string;
  confidence: number;
  supporting_agents: string[];
  dissenting_agents: string[];
  reasoning?: string;
}

export interface AgentMessage {
  agent_id: string;
  content: string;
  round_number: number;
  timestamp: string;
  metadata: Record<string, unknown>;
}

export interface Debate {
  id: string;
  task: string;
  status: DebateStatus;
  agents: string[];
  rounds: AgentMessage[][];
  consensus?: ConsensusResult;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface AgentProfile {
  id: string;
  name: string;
  provider: string;
  elo_rating: number;
  matches_played: number;
  win_rate: number;
  specialties: string[];
  metadata: Record<string, unknown>;
}

export interface HealthStatus {
  status: string;
  version: string;
  uptime_seconds: number;
  components: Record<string, string>;
}

// Control Plane Types
export interface RegisteredAgent {
  agent_id: string;
  capabilities: string[];
  status: AgentStatus;
  model: string;
  provider: string;
  is_available: boolean;
  tasks_completed: number;
  tasks_failed: number;
  avg_latency_ms: number;
  region_id: string;
}

export interface AgentHealth {
  agent_id: string;
  status: AgentStatus;
  is_available: boolean;
  last_heartbeat: number;
  heartbeat_age_seconds: number;
  tasks_completed: number;
  tasks_failed: number;
  avg_latency_ms: number;
  current_task_id?: string;
  health_check?: {
    status: string;
    last_check?: number;
    consecutive_failures: number;
  };
}

export interface Task {
  task_id: string;
  task_type: string;
  status: TaskStatus;
  priority: TaskPriority;
  created_at: number;
  assigned_at?: number;
  started_at?: number;
  completed_at?: number;
  assigned_agent?: string;
  retries: number;
  max_retries: number;
  timeout_seconds: number;
  result?: Record<string, unknown>;
  error?: string;
  is_timed_out: boolean;
}

export interface ControlPlaneStatus {
  status: string;
  registry: Record<string, unknown>;
  scheduler: Record<string, unknown>;
  health_monitor: Record<string, unknown>;
  config: Record<string, unknown>;
  knowledge_mound?: Record<string, unknown>;
}

export interface ResourceUtilization {
  queue_depths: Record<string, number>;
  agents: Record<string, unknown>;
  tasks_by_type: Record<string, number>;
  tasks_by_priority: Record<string, number>;
}

// Request/Response types
export interface CreateDebateRequest {
  task: string;
  agents?: string[];
  max_rounds?: number;
  consensus_threshold?: number;
  metadata?: Record<string, unknown>;
}

export interface SubmitTaskRequest {
  task_type: string;
  payload: Record<string, unknown>;
  required_capabilities?: string[];
  priority?: TaskPriority;
  timeout_seconds?: number;
}

export interface RegisterAgentRequest {
  agent_id: string;
  capabilities: string[];
  model?: string;
  provider?: string;
}

// WebSocket types
export type DebateEventType =
  | 'debate_start'
  | 'debate_end'
  | 'round_start'
  | 'round_end'
  | 'agent_message'
  | 'consensus_update'
  | 'vote'
  | 'critique'
  | 'error'
  | 'ping'
  | 'pong';

export interface DebateEvent {
  type: DebateEventType;
  data: Record<string, unknown>;
  timestamp: number;
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
// Graph Debate Types (parity with Python SDK)
// =============================================================================

export interface GraphBranch {
  branch_id: string;
  parent_branch_id?: string;
  path: string[];
  conclusion?: string;
  confidence: number;
  depth: number;
  is_terminal: boolean;
  agents: string[];
  messages: AgentMessage[];
  created_at: string;
}

export interface GraphDebate {
  id: string;
  task: string;
  status: DebateStatus;
  agents: string[];
  branches: GraphBranch[];
  root_branch_id: string;
  max_branches: number;
  branch_threshold: number;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface CreateGraphDebateRequest {
  task: string;
  agents?: string[];
  max_rounds?: number;
  branch_threshold?: number;
  max_branches?: number;
}

// =============================================================================
// Matrix Debate Types (parity with Python SDK)
// =============================================================================

export interface MatrixScenario {
  scenario_id: string;
  name: string;
  parameters: Record<string, unknown>;
  weight: number;
}

export interface MatrixCell {
  scenario_id: string;
  conclusion?: string;
  confidence: number;
  supporting_agents: string[];
  reasoning?: string;
}

export interface MatrixConclusion {
  overall_conclusion?: string;
  overall_confidence: number;
  cells: MatrixCell[];
  pattern_analysis?: string;
  recommendations?: string[];
}

export interface MatrixDebate {
  id: string;
  task: string;
  status: DebateStatus;
  agents: string[];
  scenarios: MatrixScenario[];
  cells: MatrixCell[];
  conclusion?: MatrixConclusion;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface CreateMatrixDebateRequest {
  task: string;
  scenarios: MatrixScenario[];
  agents?: string[];
  max_rounds?: number;
}

// =============================================================================
// Verification Types (parity with Python SDK)
// =============================================================================

export type VerificationStatus = 'pending' | 'verified' | 'refuted' | 'uncertain' | 'failed';

export interface VerificationResult {
  claim_id: string;
  claim: string;
  status: VerificationStatus;
  confidence: number;
  supporting_evidence: string[];
  refuting_evidence: string[];
  reasoning?: string;
  verified_at: string;
  verifier_agents: string[];
}

export interface VerifyClaimRequest {
  claim: string;
  context?: string;
  evidence_sources?: string[];
  min_confidence?: number;
}

// =============================================================================
// Gauntlet Types (parity with Python SDK)
// =============================================================================

export interface GauntletChallenge {
  challenge_id: string;
  challenge_type: string;
  prompt: string;
  expected_capabilities: string[];
  difficulty: number;
  time_limit_seconds?: number;
}

export interface GauntletResult {
  challenge_id: string;
  passed: boolean;
  score: number;
  response?: string;
  reasoning?: string;
  time_taken_ms: number;
}

export interface GauntletReceipt {
  receipt_id: string;
  agent_id: string;
  challenges: GauntletChallenge[];
  results: GauntletResult[];
  overall_score: number;
  passed: boolean;
  capabilities_verified: string[];
  capabilities_failed: string[];
  completed_at: string;
}

export interface RunGauntletRequest {
  agent_id: string;
  challenges?: GauntletChallenge[];
  challenge_types?: string[];
  min_pass_score?: number;
}

// =============================================================================
// Memory Analytics Types (parity with Python SDK)
// =============================================================================

export interface MemoryTierStats {
  tier: string;
  item_count: number;
  total_size_bytes: number;
  avg_age_seconds: number;
  hit_rate: number;
}

export interface MemoryAnalytics {
  total_items: number;
  total_size_bytes: number;
  tiers: MemoryTierStats[];
  cache_hit_rate: number;
  avg_retrieval_ms: number;
  most_accessed_topics: string[];
}

// =============================================================================
// Team Selection Types (parity with Python SDK)
// =============================================================================

export interface AgentScore {
  agent_id: string;
  elo_rating: number;
  specialty_match: number;
  recent_performance: number;
  availability_score: number;
  composite_score: number;
}

export interface TeamSelection {
  selected_agents: string[];
  scores: AgentScore[];
  selection_reasoning?: string;
  diversity_score: number;
  predicted_performance: number;
}

export interface SelectionPlugins {
  elo_weight?: number;
  specialty_weight?: number;
  diversity_weight?: number;
  recency_weight?: number;
  custom_scorer?: string;
}

export interface SelectTeamRequest {
  task: string;
  team_size?: number;
  required_capabilities?: string[];
  exclude_agents?: string[];
  plugins?: SelectionPlugins;
}

// =============================================================================
// Extended Agent Types (parity with Python SDK v2.1.13)
// =============================================================================

export interface AgentRating {
  name: string;
  rating: number;
  rank?: number;
  wins: number;
  losses: number;
  win_rate: number;
}

export interface AgentHistory {
  agent: string;
  history: Array<{
    timestamp: string;
    elo: number;
  }>;
}

export interface AgentCalibration {
  agent: string;
  score: number;
  domain?: string;
}

export interface AgentConsistency {
  agent: string;
  consistency_score: number;
}

export interface AgentFlip {
  id: string;
  agent_name: string;
  topic: string;
  old_stance: string;
  new_stance: string;
  flip_type: string;
  timestamp: string;
  debate_id?: string;
  reasoning?: string;
}

export interface AgentFlipsResponse {
  agent: string;
  flips: AgentFlip[];
  consistency: {
    agent_name: string;
    total_positions: number;
    total_flips: number;
    consistency_score: number;
  };
  count: number;
}

export interface AgentNetwork {
  agent: string;
  rivals: Array<{ agent: string; matches: number; win_rate: number }>;
  allies: Array<{ agent: string; matches: number; agreement_rate: number }>;
}

export interface AgentMoment {
  id: string;
  moment_type: string;
  agent_name: string;
  description: string;
  significance_score: number;
  timestamp?: string;
  debate_id?: string;
}

export interface AgentPosition {
  topic: string;
  stance: string;
  confidence: number;
  timestamp: string;
  debate_id?: string;
}

export interface AgentDomains {
  agent: string;
  overall_elo: number;
  domains: Array<{
    domain: string;
    elo: number;
    relative: number;
  }>;
  domain_count: number;
}

export interface AgentPerformance {
  agent: string;
  elo: number;
  total_games: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  recent_win_rate: number;
  elo_trend: number;
  critiques_accepted: number;
  critiques_total: number;
  critique_acceptance_rate: number;
  calibration: {
    accuracy: number;
    brier_score: number;
    prediction_count: number;
  };
}

export interface AgentMetadata {
  agent: string;
  metadata: {
    provider: string;
    model_id: string;
    context_window: number;
    specialties: string[];
    strengths: string[];
    release_date?: string;
    updated_at?: string;
  } | null;
  message?: string;
}

export interface AgentIntrospection {
  agent_id: string;
  timestamp: string;
  identity: {
    name: string;
    persona?: {
      style?: string;
      temperature?: number;
      system_prompt_preview?: string;
    };
  };
  calibration: {
    accuracy?: number;
    brier_score?: number;
    prediction_count?: number;
    confidence_level?: string;
  };
  positions: Array<{
    topic: string;
    stance: string;
    confidence: number;
    timestamp: string;
  }>;
  performance: {
    elo?: number;
    total_games?: number;
    wins?: number;
    losses?: number;
    win_rate?: number;
  };
  memory_summary: {
    tier_counts?: Record<string, number>;
    total_memories?: number;
    red_line_count?: number;
  };
  fatigue_indicators: Record<string, unknown> | null;
  debate_context: {
    debate_id: string;
    messages_sent: number;
    current_round: number;
    debate_status: string;
  } | null;
}

export interface HeadToHeadStats {
  agent1: string;
  agent2: string;
  matches: number;
  agent1_wins: number;
  agent2_wins: number;
  draws?: number;
}

export interface OpponentBriefing {
  agent: string;
  opponent: string;
  briefing: {
    strengths: string[];
    weaknesses: string[];
    common_strategies: string[];
    recommendations: string[];
    historical_summary?: string;
  } | null;
  message?: string;
}

export interface AgentComparison {
  agents: Array<{
    name: string;
    rating: number;
    wins?: number;
    losses?: number;
    win_rate?: number;
  }>;
  head_to_head?: HeadToHeadStats | null;
}

export interface LeaderboardEntry {
  name: string;
  elo: number;
  matches?: number;
  wins?: number;
  losses?: number;
  consistency?: number;
  consistency_class?: string;
}

export interface Leaderboard {
  rankings: LeaderboardEntry[];
  agents: LeaderboardEntry[];
}

export interface AgentHealthStatus {
  timestamp: number;
  overall_status: 'healthy' | 'degraded' | 'unhealthy';
  agents: Record<string, {
    type: string;
    requires_api_key: boolean;
    api_key_configured: boolean;
    available: boolean;
    circuit_breaker_open?: boolean;
  }>;
  circuit_breakers: Record<string, {
    state: string;
    failure_count: number;
    last_failure?: number;
    available: boolean;
  }>;
  fallback: {
    openrouter_available: boolean;
    local_llm_available: boolean;
    local_providers: string[];
  };
  summary: {
    available_agents: number;
    total_agents: number;
    availability_rate: number;
  };
  cross_pollination?: {
    total_subscribers: number;
    healthy_subscribers: number;
    health_rate: number;
    total_events_processed: number;
    total_events_failed: number;
  };
}

export interface FlipSummary {
  total_flips: number;
  by_type: Record<string, number>;
  by_agent: Record<string, number>;
  recent_24h: number;
}

export interface RecentFlipsResponse {
  flips: AgentFlip[];
  summary: FlipSummary;
  count: number;
}

// =============================================================================
// Calibration Types (parity with Python SDK v2.1.13)
// =============================================================================

export interface CalibrationBucket {
  range_start: number;
  range_end: number;
  total_predictions: number;
  correct_predictions: number;
  accuracy: number;
  expected_accuracy: number;
  brier_score: number;
}

export interface CalibrationCurve {
  agent: string;
  domain?: string;
  buckets: CalibrationBucket[];
  count: number;
}

export interface CalibrationSummary {
  agent: string;
  domain?: string;
  total_predictions: number;
  total_correct: number;
  accuracy: number;
  brier_score: number;
  ece: number;
  is_overconfident: boolean;
  is_underconfident: boolean;
}

export interface CalibrationLeaderboardEntry {
  agent: string;
  calibration_score: number;
  brier_score: number;
  accuracy: number;
  ece: number;
  predictions_count: number;
  correct_count: number;
  elo: number;
}

export interface CalibrationLeaderboard {
  metric: string;
  min_predictions: number;
  agents: CalibrationLeaderboardEntry[];
  count: number;
}

export interface CalibrationVisualization {
  calibration_curves: Record<string, {
    buckets: Array<{
      x: number;
      expected: number;
      actual: number;
      count: number;
    }>;
    perfect_line: Array<{ x: number; y: number }>;
  }>;
  scatter_data: Array<{
    agent: string;
    accuracy: number;
    brier_score: number;
    ece: number;
    predictions: number;
    is_overconfident: boolean;
    is_underconfident: boolean;
  }>;
  confidence_histogram: Array<{
    range: string;
    count: number;
  }>;
  domain_heatmap: Record<string, Record<string, {
    accuracy: number;
    brier: number;
    count: number;
  }>>;
  summary: {
    total_agents: number;
    avg_brier: number;
    avg_ece: number;
    best_calibrated?: string;
    worst_calibrated?: string;
  };
}

// =============================================================================
// Analytics Types (parity with Python SDK v2.1.13)
// =============================================================================

export interface DisagreementStats {
  stats: {
    total_debates: number;
    with_disagreements: number;
    unanimous: number;
    disagreement_types: Record<string, number>;
  };
}

export interface RoleRotationStats {
  stats: {
    total_debates: number;
    with_rotation: number;
    role_assignments: Record<string, number>;
  };
}

export interface EarlyStopStats {
  stats: {
    total_debates: number;
    early_stopped: number;
    full_rounds: number;
    average_rounds: number;
  };
}

export interface ConsensusQuality {
  stats: {
    total_debates: number;
    confidence_history: Array<{
      debate_id: string;
      confidence: number;
      consensus_reached: boolean;
      timestamp: string;
    }>;
    trend: 'improving' | 'stable' | 'declining' | 'insufficient_data';
    average_confidence: number;
    consensus_rate: number;
    consensus_reached_count: number;
  };
  quality_score: number;
  alert: {
    level: 'critical' | 'warning' | 'info';
    message: string;
  } | null;
}

export interface CrossPollinationStats {
  stats: {
    calibration: { enabled: boolean; adjustments: number };
    learning: { enabled: boolean; bonuses_applied: number };
    voting_accuracy: { enabled: boolean; updates: number };
    adaptive_rounds: { enabled: boolean; changes: number };
    rlm_cache: { enabled: boolean; hits: number; misses: number; hit_rate: number };
  };
  version: string;
}

export interface LearningEfficiency {
  agent?: string;
  domain: string;
  efficiency?: number;
  agents?: Array<{ agent: string; efficiency: number }>;
}

export interface VotingAccuracy {
  agent?: string;
  accuracy?: number;
  agents?: Array<{ agent: string; accuracy: number }>;
}
