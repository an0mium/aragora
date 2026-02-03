/**
 * OpenClaw Namespace API
 *
 * Provides endpoints for the OpenClaw legal analysis gateway including
 * session management, action execution, and credential management.
 */

import type { AragoraClient } from '../client';

/** OpenClaw session status */
export type SessionStatus = 'active' | 'closed' | 'expired';

/** OpenClaw action status */
export type ActionStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

/** OpenClaw session */
export interface OpenClawSession {
  id: string;
  status: SessionStatus;
  config: Record<string, unknown>;
  created_at: string;
  closed_at?: string;
}

/** OpenClaw action */
export interface OpenClawAction {
  id: string;
  session_id: string;
  action_type: string;
  status: ActionStatus;
  input: Record<string, unknown>;
  output?: Record<string, unknown>;
  created_at: string;
  completed_at?: string;
}

/** OpenClaw credential */
export interface OpenClawCredential {
  id: string;
  name: string;
  type: string;
  created_at: string;
  rotated_at?: string;
}

/** Create session request */
export interface CreateSessionRequest {
  config?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

/**
 * OpenClaw namespace for legal analysis gateway.
 *
 * @example
 * ```typescript
 * const session = await client.openclaw.createSession({});
 * const action = await client.openclaw.executeAction(session.id, {
 *   action_type: 'analyze_contract',
 *   input: { document_url: '...' },
 * });
 * ```
 */
export class OpenClawNamespace {
  constructor(private client: AragoraClient) {}

  /** List sessions. */
  async listSessions(options?: {
    status?: string;
    limit?: number;
  }): Promise<OpenClawSession[]> {
    const response = await this.client.request<{ sessions: OpenClawSession[] }>(
      'GET',
      '/api/v1/openclaw/sessions',
      { params: options }
    );
    return response.sessions;
  }

  /** Create a new session. */
  async createSession(request: CreateSessionRequest): Promise<OpenClawSession> {
    return this.client.request<OpenClawSession>(
      'POST',
      '/api/v1/openclaw/sessions',
      { body: request }
    );
  }

  /** Get session by ID. */
  async getSession(sessionId: string): Promise<OpenClawSession> {
    return this.client.request<OpenClawSession>(
      'GET',
      `/api/v1/openclaw/sessions/${encodeURIComponent(sessionId)}`
    );
  }

  /** Execute an action within a session. */
  async executeAction(
    sessionId: string,
    action: { action_type: string; input: Record<string, unknown> }
  ): Promise<OpenClawAction> {
    return this.client.request<OpenClawAction>(
      'POST',
      `/api/v1/openclaw/sessions/${encodeURIComponent(sessionId)}/actions`,
      { body: action }
    );
  }

  /** Close a session. */
  async closeSession(sessionId: string): Promise<OpenClawSession> {
    return this.client.request<OpenClawSession>(
      'POST',
      `/api/v1/openclaw/sessions/${encodeURIComponent(sessionId)}/close`
    );
  }
}
