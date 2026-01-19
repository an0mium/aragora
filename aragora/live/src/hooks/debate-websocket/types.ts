/**
 * Types for Debate WebSocket hook
 */

import type { StreamEvent } from '@/types/events';

export interface TranscriptMessage {
  agent: string;
  role?: string;
  content: string;
  round?: number;
  phase?: number;  // Current debate phase (0-8 for 9-round format)
  timestamp?: number;
}

export interface StreamingMessage {
  agent: string;
  taskId: string;  // Task ID for distinguishing concurrent outputs from same agent
  content: string;
  isComplete: boolean;
  startTime: number;
  expectedSeq: number;  // Next expected agent_seq for ordering
  pendingTokens: Map<number, string>;  // Buffer for out-of-order tokens
}

export type DebateConnectionStatus = 'connecting' | 'streaming' | 'complete' | 'error';

export interface UseDebateWebSocketOptions {
  debateId: string;
  wsUrl?: string;
  enabled?: boolean;
}

export interface UseDebateWebSocketReturn {
  // Connection state
  status: DebateConnectionStatus;
  error: string | null;
  errorDetails: string | null;  // Detailed error message from server
  isConnected: boolean;
  reconnectAttempt: number;  // Expose for UI feedback

  // Debate data
  task: string;
  agents: string[];
  messages: TranscriptMessage[];
  streamingMessages: Map<string, StreamingMessage>;
  streamEvents: StreamEvent[];
  hasCitations: boolean;

  // Actions
  sendVote: (choice: string, intensity?: number) => void;
  sendSuggestion: (suggestion: string) => void;
  registerAckCallback: (callback: (msgType: string) => void) => () => void;
  registerErrorCallback: (callback: (message: string) => void) => () => void;
  reconnect: () => void;  // Manual reconnect trigger
}

// Debate status from server API
export interface DebateStatus {
  status: string;
  error?: string;
  task?: string;
  agents?: string[];
}

// Event data types for type-safe event processing
export interface EventData {
  debate_id?: string;
  loop_id?: string;
  task?: string;
  agents?: string[];
  agent?: string;
  content?: string;
  token?: string;
  issues?: string[];
  target?: string;
  confidence?: number;
  round?: number;
  phase?: number;
  timestamp?: number;
  error_type?: string;
  message?: string;
  dropped_count?: number;
  ended?: boolean;
  messages?: Array<Record<string, unknown>>;
}

export interface ParsedEvent {
  type: string;
  agent?: string;
  seq?: number;
  loop_id?: string;
  task_id?: string;
  round?: number;
  timestamp?: number;
  data?: EventData;
}
