export type StreamEventType =
  // Debate events
  | 'debate_start'
  | 'round_start'
  | 'agent_message'
  | 'critique'
  | 'vote'
  | 'consensus'
  | 'debate_end'
  // Nomic loop events
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
  | 'sync'
  // Multi-loop events
  | 'loop_register'
  | 'loop_unregister'
  | 'loop_list';

export interface StreamEvent {
  type: StreamEventType;
  data: Record<string, unknown>;
  timestamp: number;
  round?: number;
  agent?: string;
}

export interface NomicState {
  phase?: string;
  stage?: string;
  cycle?: number;
  total_tasks?: number;
  completed_tasks?: number;
  last_task?: string;
  last_success?: boolean;
  saved_at?: string;
  status?: string;
  message?: string;
}

export interface AgentMessage {
  id: string;
  agent: string;
  content: string;
  role: string;
  round: number;
  timestamp: number;
  isExpanded: boolean;
}

export interface CritiqueMessage {
  id: string;
  agent: string;
  target: string;
  issues: string[];
  severity: number;
  round: number;
  timestamp: number;
}

export type Phase = 'idle' | 'debate' | 'design' | 'implement' | 'verify' | 'commit';

export interface PhaseStatus {
  phase: Phase;
  started?: number;
  ended?: number;
  success?: boolean;
  details?: Record<string, unknown>;
}

// Multi-loop support
export interface LoopInstance {
  loop_id: string;
  name: string;
  started_at: number;
  cycle: number;
  phase: string;
  path: string;
}

export interface LoopListData {
  loops: LoopInstance[];
  count: number;
}
