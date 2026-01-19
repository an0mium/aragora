/**
 * Types for event handler context and shared interfaces
 */

import type { StreamEvent } from '@/types/events';
import type { TranscriptMessage, StreamingMessage } from '../types';

/**
 * Context provided to all event handlers for state access
 */
export interface EventHandlerContext {
  debateId: string;

  // State setters
  setTask: React.Dispatch<React.SetStateAction<string>>;
  setAgents: React.Dispatch<React.SetStateAction<string[]>>;
  setStatus: React.Dispatch<React.SetStateAction<'connecting' | 'streaming' | 'complete' | 'error'>>;
  setError: React.Dispatch<React.SetStateAction<string | null>>;
  setErrorDetails: React.Dispatch<React.SetStateAction<string | null>>;
  setHasCitations: React.Dispatch<React.SetStateAction<boolean>>;
  setHasReceivedDebateStart: React.Dispatch<React.SetStateAction<boolean>>;
  setStreamingMessages: React.Dispatch<React.SetStateAction<Map<string, StreamingMessage>>>;

  // Helper functions
  addMessageIfNew: (msg: TranscriptMessage) => boolean;
  addStreamEvent: (event: StreamEvent) => void;
  clearDebateStartTimeout: () => void;

  // Refs for callbacks
  errorCallbackRef: React.MutableRefObject<((message: string) => void) | null>;
  ackCallbackRef: React.MutableRefObject<((msgType: string) => void) | null>;
  seenMessagesRef: React.MutableRefObject<Set<string>>;
  lastSeqRef: React.MutableRefObject<number>;
}

/**
 * Parsed event from WebSocket
 */
export interface ParsedEventData {
  type: string;
  agent?: string;
  seq?: number;
  agent_seq?: number;
  loop_id?: string;
  task_id?: string;
  round?: number;
  timestamp?: number;
  data?: Record<string, unknown>;
}

/**
 * Event handler function signature
 */
export type EventHandler = (
  data: ParsedEventData,
  ctx: EventHandlerContext
) => void;

/**
 * Event handler registry mapping event types to handlers
 */
export type EventHandlerRegistry = Record<string, EventHandler>;
