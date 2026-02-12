/**
 * OpenClaw Namespace API
 *
 * Provides endpoints for the OpenClaw gateway:
 * - session management
 * - action execution
 * - policy and approvals
 * - credential lifecycle
 * - health, metrics, and audit
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

  /** List sessions. */
  async listSessions(options?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ sessions: OpenClawSession[]; total?: number }> {
    return this.client.request<{ sessions: OpenClawSession[]; total?: number }>(
      'GET',
      '/api/v1/openclaw/sessions',
      { params: options }
    );
  }

  /** Create a new session. */
  async createSession(request: CreateSessionRequest = {}): Promise<OpenClawSession> {
    return this.client.request<OpenClawSession>('POST', '/api/v1/openclaw/sessions', {
      body: request,
    });
  }

  /** Get session by ID. */
  async getSession(sessionId: string): Promise<OpenClawSession> {
    return this.client.request<OpenClawSession>(
      'GET',
      `/api/v1/openclaw/sessions/${encodeURIComponent(sessionId)}`
    );
  }

  /** End a session. */
  async endSession(sessionId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      `/api/v1/openclaw/sessions/${encodeURIComponent(sessionId)}/end`
    );
  }

  /** Delete a session by ID. */
  async deleteSession(sessionId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'DELETE',
      `/api/v1/openclaw/sessions/${encodeURIComponent(sessionId)}`
    );
  }

  /** Backward-compatible alias for endSession. */
  async closeSession(sessionId: string): Promise<Record<string, unknown>> {
    return this.endSession(sessionId);
  }

  /** Execute an action. */
  async executeAction(request: ExecuteActionRequest): Promise<OpenClawAction> {
    return this.client.request<OpenClawAction>('POST', '/api/v1/openclaw/actions', {
      body: request,
    });
  }

  /** Get action by ID. */
  async getAction(actionId: string): Promise<OpenClawAction> {
    return this.client.request<OpenClawAction>(
      'GET',
      `/api/v1/openclaw/actions/${encodeURIComponent(actionId)}`
    );
  }

  /** Cancel an action. */
  async cancelAction(actionId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      `/api/v1/openclaw/actions/${encodeURIComponent(actionId)}/cancel`
    );
  }

  /** Get policy rules. */
  async getPolicyRules(options?: {
    enabled?: boolean;
  }): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('GET', '/api/v1/openclaw/policy/rules', {
      params: options,
    });
  }

  /** Add a policy rule. */
  async addPolicyRule(rule: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('POST', '/api/v1/openclaw/policy/rules', {
      body: rule,
    });
  }

  /** Remove a policy rule by name. */
  async removePolicyRule(ruleName: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'DELETE',
      `/api/v1/openclaw/policy/rules/${encodeURIComponent(ruleName)}`
    );
  }

  /** List approval requests. */
  async listApprovals(options?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('GET', '/api/v1/openclaw/approvals', {
      params: options,
    });
  }

  /** Approve pending action. */
  async approveAction(
    approvalId: string,
    body?: { reason?: string }
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      `/api/v1/openclaw/approvals/${encodeURIComponent(approvalId)}/approve`,
      { body }
    );
  }

  /** Deny pending action. */
  async denyAction(
    approvalId: string,
    body: { reason: string }
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      `/api/v1/openclaw/approvals/${encodeURIComponent(approvalId)}/deny`,
      { body }
    );
  }

  /** List stored credentials (metadata only). */
  async listCredentials(options?: { credential_type?: string; limit?: number; offset?: number }): Promise<{
    credentials: OpenClawCredential[];
    total?: number;
  }> {
    return this.client.request<{ credentials: OpenClawCredential[]; total?: number }>(
      'GET',
      '/api/v1/openclaw/credentials',
      { params: options }
    );
  }

  /** Store credential. */
  async storeCredential(body: {
    name: string;
    credential_type: string;
    value: string;
    expires_at?: string;
    metadata?: Record<string, unknown>;
  }): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('POST', '/api/v1/openclaw/credentials', {
      body,
    });
  }

  /** Rotate credential value. */
  async rotateCredential(
    credentialId: string,
    body: { value: string }
  ): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      `/api/v1/openclaw/credentials/${encodeURIComponent(credentialId)}/rotate`,
      { body }
    );
  }

  /** Delete credential by ID. */
  async deleteCredential(credentialId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'DELETE',
      `/api/v1/openclaw/credentials/${encodeURIComponent(credentialId)}`
    );
  }

  /** OpenClaw gateway health. */
  async health(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('GET', '/api/v1/openclaw/health');
  }

  /** OpenClaw gateway metrics. */
  async metrics(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('GET', '/api/v1/openclaw/metrics');
  }

  /** OpenClaw audit entries. */
  async audit(options?: {
    event_type?: string;
    user_id?: string;
    session_id?: string;
    start_time?: string;
    end_time?: string;
    limit?: number;
    offset?: number;
  }): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('GET', '/api/v1/openclaw/audit', {
      params: options,
    });
  }

  /** OpenClaw aggregate stats. */
  async stats(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>('GET', '/api/v1/openclaw/stats');
  }

  /** Execute a task through the OpenClaw gateway. */
  async gatewayExecute(body: {
    content: string;
    request_type?: string;
    capabilities?: string[];
    plugins?: string[];
    priority?: string;
    timeout_seconds?: number;
    context?: Record<string, unknown>;
    metadata?: Record<string, unknown>;
  }): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/gateway/openclaw/execute',
      { body }
    );
  }

  /** Get gateway task execution status. */
  async gatewayStatus(taskId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/gateway/openclaw/status/${encodeURIComponent(taskId)}`
    );
  }

  /** Register a device with the OpenClaw gateway. */
  async registerDevice(body: {
    device_id: string;
    device_name: string;
    device_type: string;
    capabilities?: string[];
    metadata?: Record<string, unknown>;
  }): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/gateway/openclaw/devices/register',
      { body }
    );
  }

  /** Unregister a device from the OpenClaw gateway. */
  async unregisterDevice(deviceId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/gateway/openclaw/devices/unregister',
      { body: { device_id: deviceId } }
    );
  }

  /** Install a plugin on the OpenClaw gateway. */
  async installPlugin(body: {
    plugin_id: string;
    plugin_name: string;
    version: string;
    config?: Record<string, unknown>;
  }): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/gateway/openclaw/plugins/install',
      { body }
    );
  }

  /** Uninstall a plugin from the OpenClaw gateway. */
  async uninstallPlugin(pluginId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/gateway/openclaw/plugins/uninstall',
      { body: { plugin_id: pluginId } }
    );
  }

  /** Get OpenClaw gateway configuration. */
  async gatewayConfig(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      '/api/v1/gateway/openclaw/config'
    );
  }
}
