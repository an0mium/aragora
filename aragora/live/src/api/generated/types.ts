/**
 * Auto-generated types from Aragora OpenAPI spec
 * Based on aragora/server/openapi.yaml
 */

// =============================================================================
// Core Types
// =============================================================================

export interface DebateSummary {
  id: string;
  slug?: string;
  task: string;
  outcome?: string;
  created_at?: string;
  rounds?: number;
  agents?: string[];
}

export interface Debate {
  id: string;
  slug?: string;
  task: string;
  context?: string;
  outcome?: string;
  final_answer?: string;
  rounds?: Round[];
  created_at?: string;
}

export interface Round {
  round_number: number;
  messages?: Message[];
  critiques?: Critique[];
}

export interface Message {
  agent: string;
  content: string;
  timestamp?: string;
}

export interface Critique {
  author: string;
  target: string;
  content: string;
  severity?: number;
}

export interface Vote {
  agent: string;
  choice: string;
  confidence?: number;
  reasoning?: string;
}

// =============================================================================
// Analysis Types
// =============================================================================

export interface ImpasseAnalysis {
  impasse_detected: boolean;
  indicators?: string[];
  circular_arguments?: Record<string, unknown>[];
  repetition_score?: number;
}

export interface ConvergenceStatus {
  converged: boolean;
  similarity_score?: number;
  rounds_to_convergence?: number;
}

export interface Citations {
  citations?: Array<{
    claim: string;
    source: string;
    confidence?: number;
  }>;
  grounded_verdict?: Record<string, unknown>;
}

// =============================================================================
// Agent Types
// =============================================================================

export interface LeaderboardEntry {
  rank: number;
  agent: string;
  elo: number;
  wins: number;
  losses: number;
  win_rate?: number;
}

export interface AgentProfile {
  name: string;
  elo: number;
  matches: number;
  win_rate?: number;
  domains?: string[];
}

export interface ConsistencyScore {
  agent: string;
  consistency_score: number;
  flip_count: number;
  recent_flips?: Array<{
    debate_id: string;
    from_position: string;
    to_position: string;
  }>;
}

export interface RelationshipNetwork {
  agent: string;
  rivals?: string[];
  allies?: string[];
  neutral?: string[];
}

export interface MatchHistoryEntry {
  debate_id: string;
  opponent: string;
  outcome: 'win' | 'loss' | 'draw';
  elo_change: number;
  timestamp?: string;
}

// =============================================================================
// Batch Debate Types
// =============================================================================

export interface BatchItem {
  question: string;
  agents?: string;
  rounds?: number;
  consensus?: 'majority' | 'unanimous' | 'supermajority';
  priority?: number;
  metadata?: Record<string, unknown>;
}

export interface BatchRequest {
  items: BatchItem[];
  webhook_url?: string;
  webhook_headers?: Record<string, string>;
  max_parallel?: number;
}

export interface BatchSubmitResponse {
  success: boolean;
  batch_id: string;
  items_queued: number;
  status_url: string;
}

export interface BatchStatus {
  batch_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  items_total: number;
  items_completed: number;
  items_failed: number;
  created_at?: string;
}

export interface BatchDetailedStatus extends BatchStatus {
  items: Array<{
    index: number;
    question: string;
    status: string;
    debate_id?: string;
    error?: string;
  }>;
  progress_percentage?: number;
  estimated_completion?: string;
}

export interface QueueStatus {
  active: boolean;
  max_concurrent: number;
  active_count: number;
  total_batches: number;
  status_counts: Record<string, number>;
}

// =============================================================================
// Graph Debate Types
// =============================================================================

export interface BranchPolicy {
  min_disagreement?: number;
  max_branches?: number;
  auto_merge?: boolean;
  merge_strategy?: 'synthesis' | 'vote' | 'best';
}

export interface GraphDebateRequest {
  task: string;
  agents: string[];
  max_rounds?: number;
  branch_policy?: BranchPolicy;
}

export interface GraphDebateResponse {
  debate_id: string;
  task: string;
  graph?: Record<string, unknown>;
  branches?: DebateBranch[];
  merge_results?: Record<string, unknown>[];
  node_count: number;
  branch_count: number;
}

export interface DebateBranch {
  branch_id: string;
  parent_branch?: string;
  branch_point: number;
  reason?: string;
  messages?: Message[];
  conclusion?: string;
  merged?: boolean;
}

export interface DebateNode {
  node_id: string;
  branch_id: string;
  round: number;
  agent: string;
  content: string;
  parent_nodes?: string[];
  child_nodes?: string[];
}

// =============================================================================
// Matrix Debate Types
// =============================================================================

export interface ScenarioConfig {
  name?: string;
  parameters?: Record<string, unknown>;
  constraints?: string[];
  is_baseline?: boolean;
}

export interface MatrixDebateRequest {
  task: string;
  agents?: string[];
  scenarios: ScenarioConfig[];
  max_rounds?: number;
}

export interface MatrixDebateResponse {
  matrix_id: string;
  task: string;
  scenario_count: number;
  results: ScenarioResult[];
  universal_conclusions?: string[];
  conditional_conclusions?: ConditionalConclusion[];
  comparison_matrix?: {
    scenarios: string[];
    consensus_rate: number;
    avg_confidence: number;
    avg_rounds: number;
  };
}

export interface ScenarioResult {
  scenario_name: string;
  parameters?: Record<string, unknown>;
  constraints?: string[];
  is_baseline?: boolean;
  winner?: string;
  final_answer?: string;
  confidence?: number;
  consensus_reached?: boolean;
  rounds_used?: number;
}

export interface ConditionalConclusion {
  condition: string;
  parameters?: Record<string, unknown>;
  conclusion: string;
  confidence?: number;
}

// =============================================================================
// Fork and Follow-up Types
// =============================================================================

export interface ForkRequest {
  branch_point: number;
  modified_context?: string;
}

export interface ForkResult {
  success: boolean;
  branch_id: string;
  parent_debate_id: string;
  branch_point: number;
  messages_inherited: number;
  modified_context?: string;
  status: string;
  message?: string;
}

export interface VerifyOutcomeRequest {
  correct: boolean;
  source?: string;
}

export interface FollowupSuggestion {
  id: string;
  crux_description: string;
  suggested_task: string;
  priority: number;
  suggested_agents?: string[];
  evidence_needed?: string;
}

export interface FollowupRequest {
  crux_id?: string;
  task?: string;
  agents?: string[];
}

export interface FollowupResult {
  success: boolean;
  followup_id: string;
  parent_debate_id: string;
  task: string;
  agents?: string[];
  crux_id?: string;
  status: string;
  message?: string;
}

// =============================================================================
// Memory Types
// =============================================================================

export interface MemoryEntry {
  key: string;
  content: string;
  tier: 'fast' | 'medium' | 'slow' | 'glacial';
  ttl?: number;
  metadata?: Record<string, unknown>;
}

export interface MemoryQueryResult {
  entries: MemoryEntry[];
  total: number;
}

export interface MemoryStats {
  total_entries: number;
  by_tier: Record<string, number>;
  storage_bytes?: number;
}

// =============================================================================
// System Types
// =============================================================================

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version?: string;
  uptime_seconds?: number;
}

export interface DetailedHealthStatus extends HealthStatus {
  components: Record<string, {
    status: string;
    latency_ms?: number;
    message?: string;
  }>;
}

export interface AvailableAgent {
  name: string;
  type: 'cli' | 'api';
  available: boolean;
  description?: string;
}

// =============================================================================
// Analytics Types
// =============================================================================

export interface DashboardMetrics {
  debates_count: number;
  consensus_rate: number;
  avg_rounds: number;
  top_agents?: LeaderboardEntry[];
  recent_debates?: DebateSummary[];
}

export interface DisagreementStats {
  total_disagreements: number;
  avg_severity: number;
}

export interface RankingStats {
  total_agents: number;
  total_matches: number;
  avg_elo: number;
}

// =============================================================================
// API Response Wrappers
// =============================================================================

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface ApiError {
  error: string;
  code?: string;
  details?: string[];
}
