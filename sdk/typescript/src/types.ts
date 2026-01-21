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
    return new AragoraError(body.error, body.code, status, body.trace_id, body);
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
  | 'critique'
  | 'vote'
  | 'consensus'
  | 'debate_end'
  | 'error'
  | 'heartbeat';

export interface WebSocketEvent<T = unknown> {
  type: WebSocketEventType;
  debate_id?: string;
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
