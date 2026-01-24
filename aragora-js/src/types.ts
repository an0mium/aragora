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
