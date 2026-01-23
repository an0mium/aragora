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
