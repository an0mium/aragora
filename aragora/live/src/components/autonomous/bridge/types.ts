'use client';

export type BridgeSchemaVersion = 1;
export type BridgeParseStatus = 'ok' | 'missing' | 'malformed';
export type BridgeRunStatus = 'running' | 'awaiting_human' | 'completed' | 'failed';
export type BridgeSessionStatus = 'not_started' | 'active' | 'completed' | 'failed';
export type BridgeEventType =
  | 'run_started'
  | 'run_failed'
  | 'run_completed'
  | 'turn.started'
  | 'turn.result'
  | 'turn.completed'
  | 'turn.repair_requested'
  | 'footer_ok'
  | 'footer_malformed'
  | 'footer_missing';

export interface AgentBridgeParticipant {
  role: string;
  harness: string;
  model: string;
}

export interface AgentBridgeFooter {
  summary: string;
  next_actor: string | null;
  needs_human: boolean;
  done: boolean;
  artifacts: string[];
  tests_run: string[];
}

export interface AgentBridgeSessionEntry {
  role: string;
  harness: string;
  model: string;
  session_id: string | null;
  worktree_agent_slug: string | null;
  worktree_path: string | null;
  branch: string | null;
  session_status: BridgeSessionStatus;
  started_at: string | null;
  last_turn_index: number;
  last_completed_at: string | null;
  harness_options?: Record<string, unknown>;
}

export interface AgentBridgeRunSummary {
  schema_version: BridgeSchemaVersion;
  run_id: string;
  task: string;
  status: BridgeRunStatus;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  last_turn_index: number;
  next_actor: string | null;
  repair_budget_per_turn: number;
  footer_mode: string;
  worktree_cleanup_mode: string;
  participants: AgentBridgeParticipant[];
  last_event_id: string | null;
}

export interface AgentBridgeRunDetail extends AgentBridgeRunSummary {
  roles: Record<string, AgentBridgeSessionEntry>;
  worktree_path: string | null;
  worktree_agent_slug: string | null;
}

export interface AgentBridgeEvent {
  schema_version: BridgeSchemaVersion;
  event_id: string;
  run_id: string;
  ts: string;
  event_type: BridgeEventType;
  turn_index: number;
  role: string;
  harness: string;
  session_id: string | null;
  parse_status: BridgeParseStatus | null;
  payload: Record<string, unknown>;
}

export interface AgentBridgeTurnRecord {
  turn_index: number;
  author_role: string;
  started_at: string;
  completed_at: string | null;
  parse_status: BridgeParseStatus;
  footer: AgentBridgeFooter | null;
  body_markdown: string;
}

export interface AgentBridgeRunListResponse {
  schema_version: BridgeSchemaVersion;
  runs: AgentBridgeRunSummary[];
  next_cursor?: string | null;
}

export interface AgentBridgeEventsResponse {
  schema_version: BridgeSchemaVersion;
  events: AgentBridgeEvent[];
  next_cursor?: string | null;
}

export interface AgentBridgeTranscriptResponse {
  schema_version: BridgeSchemaVersion;
  turns: AgentBridgeTurnRecord[];
}

export type BridgeApiError = Error & { status?: number };

export function formatBridgeTimestamp(value: string | null): string {
  if (!value) {
    return 'n/a';
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return parsed.toLocaleString();
}

export function truncateBridgeSessionId(sessionId: string | null): string {
  if (!sessionId) {
    return 'not started';
  }

  if (sessionId.length <= 14) {
    return sessionId;
  }

  return `${sessionId.slice(0, 8)}...${sessionId.slice(-4)}`;
}

export function isBridgeRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function getBridgeStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === 'string');
}
