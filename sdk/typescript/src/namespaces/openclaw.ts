/**
 * OpenClaw Namespace API
 *
 * Provides endpoints for the OpenClaw gateway:
 * - Session orchestration
 * - Action execution
 * - Policy and approvals
 * - Credential lifecycle
 * - Health, metrics, and audit
 */

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

/** Audit filter options */
export interface AuditFilterOptions {
  event_type?: string;
  user_id?: string;
  session_id?: string;
  start_time?: string;
  end_time?: string;
}

interface OpenClawClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class OpenClawNamespace {
  constructor(private client: OpenClawClientInterface) {}

  // -- Session management ---------------------------------------------------

  /** List active OpenClaw sessions. */
  async listSessions(params?: { skip?: number; limit?: number }): Promise<OpenClawSession[]> {
    return this.client.request<OpenClawSession[]>('GET', '/api/v1/openclaw/sessions', { params });
  }

  /** Create a new OpenClaw session. */
  async createSession(data: CreateSessionRequest): Promise<OpenClawSession> {
    return this.client.request<OpenClawSession>('POST', '/api/v1/openclaw/sessions', { body: data });
  }

  /** Get an OpenClaw session by ID. */
  async getSession(sessionId: string): Promise<OpenClawSession> {
    return this.client.request<OpenClawSession>('GET', `/api/v1/openclaw/sessions/${sessionId}`);
  }

  /** End an active OpenClaw session. */
  async endSession(sessionId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('POST', `/api/v1/openclaw/sessions/${sessionId}/end`);
  }

  /** Delete an OpenClaw session. */
  async deleteSession(sessionId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('DELETE', `/api/v1/openclaw/sessions/${sessionId}`);
  }

  // -- Action management ----------------------------------------------------

  /** Submit a new action for execution. */
  async executeAction(data: ExecuteActionRequest): Promise<OpenClawAction> {
    return this.client.request<OpenClawAction>('POST', '/api/v1/openclaw/actions', { body: data });
  }

  /** Get an action by ID. */
  async getAction(actionId: string): Promise<OpenClawAction> {
    return this.client.request<OpenClawAction>('GET', `/api/v1/openclaw/actions/${actionId}`);
  }

  /** Cancel a pending action. */
  async cancelAction(actionId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('POST', `/api/v1/openclaw/actions/${actionId}/cancel`);
  }

  // -- Credential lifecycle -------------------------------------------------

  /** List OpenClaw credentials. */
  async listCredentials(): Promise<OpenClawCredential[]> {
    return this.client.request<OpenClawCredential[]>('GET', '/api/v1/openclaw/credentials');
  }

  /** Store a new credential. */
  async storeCredential(data: Record<string, unknown>): Promise<OpenClawCredential> {
    return this.client.request<OpenClawCredential>('POST', '/api/v1/openclaw/credentials', { body: data });
  }

  /** Delete a credential. */
  async deleteCredential(credentialId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('DELETE', `/api/v1/openclaw/credentials/${credentialId}`);
  }

  /** Rotate a credential. */
  async rotateCredential(credentialId: string, body?: { new_value?: string }): Promise<OpenClawCredential> {
    return this.client.request<OpenClawCredential>('POST', `/api/v1/openclaw/credentials/${credentialId}/rotate`, { body });
  }

  // -- Policy rules ---------------------------------------------------------

  /** Get policy rules. */
  async getPolicyRules(): Promise<Record<string, unknown>[]> {
    return this.client.request<Record<string, unknown>[]>('GET', '/api/v1/openclaw/policy/rules');
  }

  /** Add a new policy rule. */
  async addPolicyRule(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('POST', '/api/v1/openclaw/policy/rules', { body: data });
  }

  /** Remove a policy rule. */
  async removePolicyRule(ruleId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('DELETE', `/api/v1/openclaw/policy/rules/${ruleId}`);
  }

  // -- Approvals ------------------------------------------------------------

  /** List pending approvals. */
  async listApprovals(): Promise<Record<string, unknown>[]> {
    return this.client.request<Record<string, unknown>[]>('GET', '/api/v1/openclaw/approvals');
  }

  /** Approve a pending action. */
  async approveAction(approvalId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('POST', `/api/v1/openclaw/approvals/${approvalId}/approve`);
  }

  /** Deny a pending action. */
  async denyAction(approvalId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('POST', `/api/v1/openclaw/approvals/${approvalId}/deny`);
  }

  // -- Service introspection ------------------------------------------------

  /** Get OpenClaw gateway health status. */
  async health(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('GET', '/api/v1/openclaw/health');
  }

  /** Get OpenClaw gateway metrics. */
  async metrics(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('GET', '/api/v1/openclaw/metrics');
  }

  /** Get OpenClaw audit log with optional filters. */
  async audit(options?: AuditFilterOptions): Promise<Record<string, unknown>> {
    const params: Record<string, string> = {};
    if (options?.event_type) params.event_type = options.event_type;
    if (options?.user_id) params.user_id = options.user_id;
    if (options?.session_id) params.session_id = options.session_id;
    if (options?.start_time) params.start_time = options.start_time;
    if (options?.end_time) params.end_time = options.end_time;
    return this.client.request<Record<string, unknown>>('GET', '/api/v1/openclaw/audit', { params });
  }

  /** Get OpenClaw gateway stats. */
  async stats(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('GET', '/api/v1/openclaw/stats');
  }
}
