/**
 * Coordination Namespace API
 *
 * Cross-workspace federation and coordination capabilities including
 * workspace registration, federation policy management, cross-workspace
 * execution, and data sharing consent.
 */

interface CoordinationClientInterface {
  request<T = unknown>(method: string, path: string, options?: {
    params?: Record<string, unknown>;
    json?: Record<string, unknown>;
    body?: Record<string, unknown>;
  }): Promise<T>;
}

export interface RegisterWorkspaceRequest {
  id: string;
  name?: string;
  org_id?: string;
  federation_mode?: string;
  endpoint_url?: string;
  supports_agent_execution?: boolean;
  supports_workflow_execution?: boolean;
  supports_knowledge_query?: boolean;
  [key: string]: unknown;
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

export interface CreateFederationPolicyRequest {
  name: string;
  description?: string;
  mode?: string;
  sharing_scope?: string;
  allowed_operations?: string[];
  max_requests_per_hour?: number;
  require_approval?: boolean;
  audit_all_requests?: boolean;
  workspace_id?: string;
  source_workspace_id?: string;
  target_workspace_id?: string;
  [key: string]: unknown;
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

export interface ExecuteCrossWorkspaceRequest {
  operation: string;
  source_workspace_id: string;
  target_workspace_id: string;
  payload?: Record<string, unknown>;
  timeout_seconds?: number;
  requester_id?: string;
  consent_id?: string;
  [key: string]: unknown;
}

export interface CrossWorkspaceResult {
  request_id: string;
  success: boolean;
  data: unknown;
  error: string | null;
  error_code: string | null;
  execution_time_ms: number;
}

export interface GrantConsentRequest {
  source_workspace_id: string;
  target_workspace_id: string;
  scope?: string;
  data_types?: string[];
  operations?: string[];
  granted_by?: string;
  expires_in_days?: number;
  [key: string]: unknown;
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

export interface ApproveRequestBody {
  approved_by?: string;
  [key: string]: unknown;
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
 *
 * @example
 * ```ts
 * const workspaces = await client.coordination.listWorkspaces();
 * await client.coordination.registerWorkspace({ id: 'ws-1', name: 'Primary' });
 * await client.coordination.createFederationPolicy({ name: 'default' });
 * ```
 */
export class CoordinationAPI {
  constructor(private client: CoordinationClientInterface) {}

  // ===========================================================================
  // Workspaces
  // ===========================================================================

  /** Register a workspace for federation. */
  async registerWorkspace(body: RegisterWorkspaceRequest): Promise<FederatedWorkspace> {
    return this.client.request('POST', '/api/v1/coordination/workspaces', { body });
  }

  /** List all registered workspaces. */
  async listWorkspaces(): Promise<{ workspaces: FederatedWorkspace[]; total: number }> {
    return this.client.request('GET', '/api/v1/coordination/workspaces');
  }

  /** Unregister a workspace from federation. */
  async unregisterWorkspace(workspaceId: string): Promise<{ unregistered: boolean }> {
    return this.client.request('DELETE', `/api/v1/coordination/workspaces/${encodeURIComponent(workspaceId)}`);
  }

  // ===========================================================================
  // Federation Policies
  // ===========================================================================

  /** Create a federation policy. */
  async createFederationPolicy(body: CreateFederationPolicyRequest): Promise<FederationPolicy> {
    return this.client.request('POST', '/api/v1/coordination/federation', { body });
  }

  /** List all federation policies. */
  async listFederationPolicies(): Promise<{ policies: FederationPolicy[]; total: number }> {
    return this.client.request('GET', '/api/v1/coordination/federation');
  }

  /** @deprecated Use createFederationPolicy instead. */
  async createPolicy(body: CreateFederationPolicyRequest): Promise<FederationPolicy> {
    return this.createFederationPolicy(body);
  }

  /** @deprecated Use listFederationPolicies instead. */
  async listPolicies(): Promise<{ policies: FederationPolicy[]; total: number }> {
    return this.listFederationPolicies();
  }

  // ===========================================================================
  // Execution
  // ===========================================================================

  /** Execute a cross-workspace operation. */
  async executeCrossWorkspace(body: ExecuteCrossWorkspaceRequest): Promise<CrossWorkspaceResult> {
    return this.client.request('POST', '/api/v1/coordination/execute', { body });
  }

  /** @deprecated Use executeCrossWorkspace instead. */
  async execute(body: ExecuteCrossWorkspaceRequest): Promise<CrossWorkspaceResult> {
    return this.executeCrossWorkspace(body);
  }

  /** List pending cross-workspace execution requests. */
  async listExecutions(workspaceId?: string): Promise<{ executions: unknown[]; total: number }> {
    const params = workspaceId ? { workspace_id: workspaceId } : undefined;
    return this.client.request('GET', '/api/v1/coordination/executions', { params });
  }

  // ===========================================================================
  // Consent
  // ===========================================================================

  /** Grant data sharing consent between workspaces. */
  async grantConsent(body: GrantConsentRequest): Promise<DataSharingConsent> {
    return this.client.request('POST', '/api/v1/coordination/consent', { body });
  }

  /** Revoke a data sharing consent. */
  async revokeConsent(consentId: string): Promise<{ revoked: boolean }> {
    return this.client.request('DELETE', `/api/v1/coordination/consent/${encodeURIComponent(consentId)}`);
  }

  /** List data sharing consents. */
  async listConsents(workspaceId?: string): Promise<{ consents: DataSharingConsent[]; total: number }> {
    const params = workspaceId ? { workspace_id: workspaceId } : undefined;
    return this.client.request('GET', '/api/v1/coordination/consent', { params });
  }

  // ===========================================================================
  // Approval
  // ===========================================================================

  /** Approve a pending cross-workspace execution request. */
  async approveRequest(requestId: string, body?: ApproveRequestBody): Promise<{ approved: boolean }> {
    return this.client.request('POST', `/api/v1/coordination/approve/${encodeURIComponent(requestId)}`, { body: body ?? {} });
  }

  // ===========================================================================
  // Stats and Health
  // ===========================================================================

  /** Get coordination statistics. */
  async getStats(): Promise<CoordinationStats> {
    return this.client.request('GET', '/api/v1/coordination/stats');
  }

  /** Get coordination health status. */
  async getHealth(): Promise<CoordinationHealth> {
    return this.client.request('GET', '/api/v1/coordination/health');
  }
}
