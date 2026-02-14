/**
 * OpenClaw Namespace API
 *
 * Provides endpoints for the OpenClaw gateway.
 * All methods were removed — no corresponding backend routes exist.
 * The class and type definitions are preserved for future use.
 */

import type { AragoraClient } from '../client';

/** OpenClaw session status */
export type SessionStatus = 'active' | 'idle' | 'closing' | 'closed' | 'error';

/** OpenClaw action status */
export type ActionStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'timeout';

/** OpenClaw session */
export interface OpenClawSession {
  id: string;
  status: SessionStatus;
  config: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at?: string;
}

/** OpenClaw action */
export interface OpenClawAction {
  id: string;
  session_id: string;
  action_type: string;
  status: ActionStatus;
  input_data?: Record<string, unknown>;
  output_data?: Record<string, unknown>;
  created_at: string;
  completed_at?: string;
  error?: string;
}

/** OpenClaw credential metadata */
export interface OpenClawCredential {
  id: string;
  name: string;
  credential_type: string;
  created_at: string;
  updated_at?: string;
  expires_at?: string;
  last_rotated_at?: string;
}

export interface CreateSessionRequest {
  config?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface ExecuteActionRequest {
  session_id: string;
  action_type: string;
  params?: Record<string, unknown>;
  input_data?: Record<string, unknown>;
}

export class OpenClawNamespace {
  constructor(private client: AragoraClient) {}
}
