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

// =============================================================================
// Memory API Types (parity with Python SDK)
// =============================================================================

export interface MemoryItem {
  id: string;
  tier: 'fast' | 'medium' | 'slow' | 'glacial';
  content: string;
  importance: number;
  surprise_score?: number;
  consolidation_score?: number;
  update_count?: number;
  created_at?: string;
  updated_at?: string;
}

export interface MemoryRetrieveResponse {
  memories: MemoryItem[];
  count: number;
  query: string;
  tiers: string[];
}

export interface MemoryTierInfo {
  name: string;
  ttl_seconds: number;
  max_items: number;
  current_items: number;
  utilization_percent: number;
  avg_importance: number;
  oldest_entry_age_seconds?: number;
  newest_entry_age_seconds?: number;
}

export interface MemoryTiersResponse {
  tiers: MemoryTierInfo[];
  total_items: number;
  total_capacity: number;
}

export interface MemoryPressure {
  overall_pressure: number;
  tier_pressures: Record<string, number>;
  recommendation: 'ok' | 'consolidate' | 'cleanup';
  items_pending_promotion: number;
  items_pending_eviction: number;
}

export interface MemorySearchResult {
  memories: MemoryItem[];
  total_count: number;
  search_time_ms: number;
  tiers_searched: string[];
}

export interface Critique {
  id: string;
  debate_id: string;
  critic_agent: string;
  target_agent: string;
  content: string;
  score?: number;
  critique_type: 'constructive' | 'challenging' | 'supportive';
  created_at: string;
}

export interface CritiqueListResponse {
  critiques: Critique[];
  count: number;
  has_more: boolean;
}

export interface ConsolidationResult {
  promoted: number;
  evicted: number;
  merged: number;
  duration_ms: number;
}

export interface CleanupResult {
  removed: number;
  recovered_bytes: number;
  duration_ms: number;
}

export interface ArchiveStats {
  total_archived: number;
  total_bytes: number;
  oldest_entry?: string;
  newest_entry?: string;
  compression_ratio?: number;
}

// =============================================================================
// Knowledge API Types
// =============================================================================

export interface KnowledgeEntry {
  id?: string;
  content: string;
  source?: string;
  source_type?: string;
  metadata?: Record<string, unknown>;
  confidence?: number;
  created_at?: string;
  updated_at?: string;
  tags?: string[];
}

export interface KnowledgeSearchResult {
  id: string;
  content: string;
  score: number;
  source?: string;
  metadata?: Record<string, unknown>;
}

export interface KnowledgeStats {
  total_entries: number;
  total_facts: number;
  sources?: Record<string, number>;
  categories?: Record<string, number>;
  avg_confidence?: number;
  last_updated?: string;
}

export interface Fact {
  id: string;
  content: string;
  source?: string;
  confidence?: number;
  verified: boolean;
  metadata?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

export interface KnowledgeQueryResponse {
  answer: string;
  sources?: Array<{ id: string; content: string; score: number }>;
  confidence?: number;
}

// =============================================================================
// Workflow API Types
// =============================================================================

export type WorkflowStatus = 'draft' | 'active' | 'paused' | 'archived';
export type ExecutionStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface Workflow {
  id: string;
  name: string;
  description?: string;
  status: WorkflowStatus;
  steps: WorkflowStep[];
  triggers?: WorkflowTrigger[];
  created_at: string;
  updated_at: string;
  metadata?: Record<string, unknown>;
}

export interface WorkflowStep {
  id: string;
  name: string;
  type: string;
  config: Record<string, unknown>;
  depends_on?: string[];
}

export interface WorkflowTrigger {
  type: 'schedule' | 'event' | 'webhook' | 'manual';
  config: Record<string, unknown>;
}

export interface WorkflowExecution {
  id: string;
  workflow_id: string;
  status: ExecutionStatus;
  started_at: string;
  completed_at?: string;
  current_step?: string;
  results?: Record<string, unknown>;
  error?: string;
}

export interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  steps: WorkflowStep[];
  parameters?: Array<{
    name: string;
    type: string;
    required: boolean;
    default?: unknown;
  }>;
}

// =============================================================================
// Tournament API Types
// =============================================================================

export type TournamentStatus = 'draft' | 'registration' | 'running' | 'completed' | 'cancelled';
export type TournamentFormat = 'single_elimination' | 'double_elimination' | 'round_robin' | 'swiss';

export interface Tournament {
  id: string;
  name: string;
  description?: string;
  status: TournamentStatus;
  format: TournamentFormat;
  topic: string;
  participants: string[];
  created_at: string;
  started_at?: string;
  completed_at?: string;
  winner?: string;
  metadata?: Record<string, unknown>;
}

export interface TournamentStandings {
  tournament_id: string;
  standings: Array<{
    agent: string;
    wins: number;
    losses: number;
    points: number;
    rank: number;
  }>;
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
  round: number;
  agent_a: string;
  agent_b: string;
  winner?: string;
  debate_id?: string;
  status: 'pending' | 'running' | 'completed';
}

// =============================================================================
// RBAC API Types
// =============================================================================

export interface Role {
  id: string;
  name: string;
  description?: string;
  permissions: string[];
  is_system: boolean;
  created_at: string;
  updated_at: string;
}

export interface Permission {
  id: string;
  name: string;
  description?: string;
  resource: string;
  action: string;
}

export interface RoleAssignment {
  id: string;
  user_id: string;
  role_id: string;
  assigned_at: string;
  assigned_by?: string;
}

// =============================================================================
// Auth API Types
// =============================================================================

export interface AuthToken {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface User {
  id: string;
  email: string;
  name?: string;
  avatar_url?: string;
  roles: string[];
  created_at: string;
  updated_at: string;
  email_verified: boolean;
  mfa_enabled: boolean;
}

export interface MFASetupResponse {
  secret: string;
  qr_code_url: string;
  backup_codes: string[];
}

export interface MFAVerifyResponse {
  success: boolean;
  backup_codes_remaining: number;
}

// =============================================================================
// Codebase API Types
// =============================================================================

export type CodebaseScanStatus = 'pending' | 'running' | 'completed' | 'failed';
export type VulnerabilitySeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';

export interface CodebaseScanResult {
  scan_id: string;
  repo_id: string;
  status: CodebaseScanStatus;
  started_at: string;
  completed_at?: string;
  files_scanned: number;
  lines_scanned: number;
  vulnerabilities_found: number;
  risk_score: number;
  summary: {
    critical: number;
    high: number;
    medium: number;
    low: number;
    info: number;
  };
}

export interface CodebaseVulnerability {
  id: string;
  scan_id: string;
  severity: VulnerabilitySeverity;
  title: string;
  description: string;
  file_path: string;
  line_number?: number;
  code_snippet?: string;
  cwe_id?: string;
  cvss_score?: number;
  recommendation?: string;
  category: string;
  confidence: number;
  created_at: string;
}

export interface CodebaseScanRequest {
  repo_path: string;
  include_patterns?: string[];
  exclude_patterns?: string[];
  severity_threshold?: VulnerabilitySeverity;
  include_secrets?: boolean;
  include_dependencies?: boolean;
}

export interface CodebaseDependency {
  name: string;
  version: string;
  license?: string;
  vulnerabilities: number;
  outdated: boolean;
  latest_version?: string;
  ecosystem: string;
}

export interface CodebaseDependencyAnalysis {
  total_dependencies: number;
  direct_dependencies: number;
  transitive_dependencies: number;
  vulnerable_dependencies: number;
  outdated_dependencies: number;
  dependencies: CodebaseDependency[];
  risk_score: number;
}

export interface CodebaseMetrics {
  repo_id: string;
  total_files: number;
  total_lines: number;
  languages: Record<string, number>;
  avg_complexity: number;
  max_complexity: number;
  maintainability_index: number;
  test_coverage?: number;
  hotspots: Array<{
    file_path: string;
    complexity: number;
    risk_score: number;
  }>;
  duplicates: Array<{
    hash: string;
    lines: number;
    occurrences: Array<{
      file: string;
      start: number;
      end: number;
    }>;
  }>;
}

export interface CodebaseAnalysisRequest {
  repo_path: string;
  include_patterns?: string[];
  exclude_patterns?: string[];
  complexity_warning?: number;
  complexity_error?: number;
}

// =============================================================================
// Decisions Types
// =============================================================================

export type DecisionType = 'debate' | 'workflow' | 'gauntlet' | 'quick' | 'auto';
export type DecisionPriority = 'high' | 'normal' | 'low';
export type DecisionStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface DecisionConfig {
  agents: string[];
  rounds: number;
  consensus: string;
  timeout_seconds: number;
}

export interface DecisionContext {
  user_id?: string;
  workspace_id?: string;
  metadata?: Record<string, unknown>;
}

export interface ResponseChannel {
  platform: string;
  target?: string;
  options?: Record<string, unknown>;
}

export interface DecisionResult {
  request_id: string;
  status: DecisionStatus;
  decision_type: DecisionType;
  content: string;
  result?: Record<string, unknown>;
  error?: string;
  created_at?: string;
  completed_at?: string;
  metadata?: Record<string, unknown>;
}

export interface DecisionStatusResponse {
  request_id: string;
  status: DecisionStatus;
  progress: number;
  current_stage?: string;
  estimated_remaining_seconds?: number;
}

// =============================================================================
// Documents Types
// =============================================================================

export type DocumentStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface Document {
  id: string;
  filename: string;
  content_type: string;
  status: DocumentStatus;
  size_bytes?: number;
  metadata?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

export interface DocumentUploadResponse {
  document_id: string;
  status: DocumentStatus;
  message?: string;
}

export interface SupportedFormats {
  formats: string[];
  mime_types: string[];
}

export interface BatchUploadResponse {
  job_id: string;
  status: string;
  files_count: number;
}

export interface BatchJobStatus {
  job_id: string;
  status: string;
  progress: number;
  total: number;
  completed: number;
  failed: number;
}

export interface BatchJobResults {
  job_id: string;
  documents: Document[];
  errors: Array<{ filename: string; error: string }>;
}

export interface ProcessingStats {
  total_documents: number;
  pending: number;
  processing: number;
  completed: number;
  failed: number;
}

export interface DocumentChunk {
  id: string;
  document_id: string;
  content: string;
  position: number;
  metadata?: Record<string, unknown>;
}

export interface DocumentContext {
  content: string;
  token_count: number;
  truncated: boolean;
  chunks_included: number;
}

export interface AuditSession {
  id: string;
  status: string;
  document_ids: string[];
  audit_types: string[];
  model: string;
  progress: number;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
}

export interface AuditSessionCreateResponse {
  session_id: string;
  status: string;
}

export interface AuditFinding {
  id: string;
  session_id: string;
  document_id: string;
  audit_type: string;
  severity: string;
  title: string;
  description: string;
  location?: string;
  recommendation?: string;
}

export interface AuditReport {
  session_id: string;
  format: string;
  content: string;
  generated_at: string;
}

// =============================================================================
// Policies Types
// =============================================================================

export type PolicyLevel = 'required' | 'recommended' | 'optional';
export type ViolationStatus = 'open' | 'investigating' | 'resolved' | 'false_positive';

export interface PolicyRule {
  id: string;
  name: string;
  description: string;
  condition: string;
  severity: string;
  enabled: boolean;
  metadata?: Record<string, unknown>;
}

export interface Policy {
  id: string;
  name: string;
  description: string;
  framework_id: string;
  workspace_id: string;
  vertical_id: string;
  level: PolicyLevel;
  enabled: boolean;
  rules: PolicyRule[];
  created_at?: string;
  updated_at?: string;
  metadata?: Record<string, unknown>;
}

export interface PolicyViolation {
  id: string;
  policy_id: string;
  rule_id: string;
  rule_name: string;
  framework_id: string;
  vertical_id: string;
  workspace_id: string;
  severity: string;
  status: ViolationStatus;
  description: string;
  source: string;
  created_at?: string;
  resolved_at?: string;
  resolved_by?: string;
  resolution_notes?: string;
  metadata?: Record<string, unknown>;
}

export interface ComplianceCheckResult {
  compliant: boolean;
  score: number;
  issues: Array<Record<string, unknown>>;
  checked_at?: string;
}

export interface ComplianceStats {
  policies_total: number;
  policies_enabled: number;
  policies_disabled: number;
  violations_total: number;
  violations_open: number;
  violations_by_severity: Record<string, number>;
  risk_score: number;
}
