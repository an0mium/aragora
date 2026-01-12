/**
 * Aragora SDK Type Definitions
 *
 * Core types for interacting with the Aragora API.
 */

// =============================================================================
// Enums
// =============================================================================

export enum DebateStatus {
  CREATED = 'created',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
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
  conclusion?: string;
  confidence: number;
  supporting_agents: string[];
  dissenting_agents?: string[];
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

export interface DebateCreateRequest {
  task: string;
  agents?: string[];
  max_rounds?: number;
  consensus_threshold?: number;
  enable_voting?: boolean;
  context?: string;
}

export interface DebateCreateResponse {
  debate_id: string;
  status: string;
  task: string;
}

export interface Debate {
  debate_id: string;
  task: string;
  status: DebateStatus;
  agents: string[];
  rounds: DebateRound[];
  consensus?: ConsensusResult;
  created_at?: string;
  completed_at?: string;
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

export interface AragoraClientOptions {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

export interface RequestOptions {
  timeout?: number;
  headers?: Record<string, string>;
  signal?: AbortSignal;
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
