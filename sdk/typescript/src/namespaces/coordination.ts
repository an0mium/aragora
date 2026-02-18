/**
 * Coordination Namespace API
 *
 * Cross-workspace federation and coordination capabilities.
 */

interface CoordinationClientInterface {
  request<T = unknown>(method: string, path: string, options?: {
    params?: Record<string, unknown>;
    json?: Record<string, unknown>;
    body?: Record<string, unknown>;
  }): Promise<T>;
}

export interface FederatedWorkspace {
  id: string;
  name: string;
  org_id: string;
  is_federated: boolean;
  federation_mode: string;
  joined_at: string;
  supports_agent_execution: boolean;
  supports_workflow_execution: boolean;
  supports_knowledge_query: boolean;
  is_online: boolean;
  last_heartbeat: string | null;
  latency_ms: number;
}

export interface FederationPolicy {
  id: string;
  name: string;
  description: string;
  mode: string;
  sharing_scope: string;
  allowed_operations: string[];
  blocked_operations: string[];
  max_requests_per_hour: number;
  require_approval: boolean;
}

export interface DataSharingConsent {
  id: string;
  source_workspace_id: string;
  target_workspace_id: string;
  scope: string;
  data_types: string[];
  operations: string[];
  granted_by: string;
  granted_at: string;
  expires_at: string | null;
  is_valid: boolean;
  times_used: number;
}

export interface CrossWorkspaceResult {
  request_id: string;
  success: boolean;
  data: unknown;
  error: string | null;
  error_code: string | null;
  execution_time_ms: number;
}

export interface CoordinationStats {
  total_workspaces: number;
  total_consents: number;
  valid_consents: number;
  pending_requests: number;
  registered_handlers: string[];
}

export interface CoordinationHealth {
  status: string;
  total_workspaces?: number;
  pending_requests?: number;
  valid_consents?: number;
  message?: string;
}

/**
 * Coordination API namespace.
 *
 * Manages cross-workspace federation, consent, and execution.
 */
export class CoordinationAPI {
  constructor(private client: CoordinationClientInterface) {}

  // ===========================================================================
  // Workspaces
  // ===========================================================================

  async registerWorkspace(body: {
    id: string;
    name?: string;
    org_id?: string;
    federation_mode?: string;
    endpoint_url?: string;
  }): Promise<FederatedWorkspace> {
    return this.client.request('POST', '/api/v1/coordination/workspaces', { body });
  }

  async listWorkspaces(): Promise<{ workspaces: FederatedWorkspace[]; total: number }> {
    return this.client.request('GET', '/api/v1/coordination/workspaces');
  }

  async unregisterWorkspace(workspaceId: string): Promise<{ unregistered: boolean }> {
    return this.client.request('DELETE', `/api/v1/coordination/workspaces/${encodeURIComponent(workspaceId)}`);
  }

  // ===========================================================================
  // Federation Policies
  // ===========================================================================

  async createPolicy(body: {
    name: string;
    description?: string;
    mode?: string;
    sharing_scope?: string;
    allowed_operations?: string[];
    max_requests_per_hour?: number;
    require_approval?: boolean;
  }): Promise<FederationPolicy> {
    return this.client.request('POST', '/api/v1/coordination/federation', { body });
  }

  async listPolicies(): Promise<{ policies: FederationPolicy[]; total: number }> {
    return this.client.request('GET', '/api/v1/coordination/federation');
  }

  // ===========================================================================
  // Execution
  // ===========================================================================

  async execute(body: {
    operation: string;
    source_workspace_id: string;
    target_workspace_id: string;
    payload?: Record<string, unknown>;
    timeout_seconds?: number;
  }): Promise<CrossWorkspaceResult> {
    return this.client.request('POST', '/api/v1/coordination/execute', { body });
  }

  async listExecutions(workspaceId?: string): Promise<{ executions: unknown[]; total: number }> {
    const params = workspaceId ? { workspace_id: workspaceId } : undefined;
    return this.client.request('GET', '/api/v1/coordination/executions', { params });
  }

  async approveRequest(requestId: string, body?: {
    approved_by?: string;
  }): Promise<{ approved: boolean }> {
    return this.client.request('POST', `/api/v1/coordination/approve/${encodeURIComponent(requestId)}`, { body: body ?? {} });
  }

  // ===========================================================================
  // Consent
  // ===========================================================================

  async grantConsent(body: {
    source_workspace_id: string;
    target_workspace_id: string;
    scope?: string;
    data_types?: string[];
    operations?: string[];
    granted_by?: string;
    expires_in_days?: number;
  }): Promise<DataSharingConsent> {
    return this.client.request('POST', '/api/v1/coordination/consent', { body });
  }

  async revokeConsent(consentId: string): Promise<{ revoked: boolean }> {
    return this.client.request('DELETE', `/api/v1/coordination/consent/${encodeURIComponent(consentId)}`);
  }

  async listConsents(workspaceId?: string): Promise<{ consents: DataSharingConsent[]; total: number }> {
    const params = workspaceId ? { workspace_id: workspaceId } : undefined;
    return this.client.request('GET', '/api/v1/coordination/consent', { params });
  }

  // ===========================================================================
  // Stats and Health
  // ===========================================================================

  async getStats(): Promise<CoordinationStats> {
    return this.client.request('GET', '/api/v1/coordination/stats');
  }

  async getHealth(): Promise<CoordinationHealth> {
    return this.client.request('GET', '/api/v1/coordination/health');
  }
}
