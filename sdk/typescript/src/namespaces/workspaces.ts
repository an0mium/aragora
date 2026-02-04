/**
 * Workspaces Namespace API
 *
 * Provides methods for workspace management, data isolation, and privacy controls.
 *
 * Features:
 * - Workspace creation and management
 * - Member access control
 * - Retention policy management
 * - Content sensitivity classification
 * - Privacy audit logging
 *
 * Essential for multi-tenancy and team collaboration.
 */

// ===========================================================================
// Core Workspace Types
// ===========================================================================

/**
 * Workspace definition
 */
export interface Workspace {
  id: string;
  name: string;
  slug: string;
  description?: string;
  owner_id: string;
  tenant_id?: string;
  tier: 'free' | 'pro' | 'enterprise';
  settings: WorkspaceSettings;
  member_count: number;
  debate_count: number;
  created_at: string;
  updated_at: string;
}

/**
 * Workspace settings
 */
export interface WorkspaceSettings {
  allow_public_debates: boolean;
  default_debate_visibility: 'private' | 'workspace' | 'public';
  require_approval_for_debates: boolean;
  max_debate_duration_minutes: number;
  enabled_features: string[];
  retention_days: number;
  custom_branding?: {
    logo_url?: string;
    primary_color?: string;
    accent_color?: string;
  };
}

/**
 * Workspace member
 */
export interface WorkspaceMember {
  user_id: string;
  workspace_id: string;
  role: 'viewer' | 'member' | 'admin' | 'owner';
  email: string;
  name?: string;
  avatar_url?: string;
  joined_at: string;
  last_active_at?: string;
}

/**
 * Workspace profile (permission set)
 */
export interface WorkspaceProfile {
  id: string;
  workspace_id: string;
  name: string;
  description?: string;
  permissions: string[];
  is_default: boolean;
}

/**
 * Create workspace request
 */
export interface CreateWorkspaceRequest {
  name: string;
  slug?: string;
  tenant_id?: string;
  description?: string;
  settings?: Partial<WorkspaceSettings>;
}

/**
 * Update workspace request
 */
export interface UpdateWorkspaceRequest {
  name?: string;
  description?: string;
  settings?: Partial<WorkspaceSettings>;
}

/**
 * Add member request
 */
export interface AddMemberRequest {
  user_id?: string;
  email?: string;
  role: 'viewer' | 'member' | 'admin';
}

/**
 * Update member request
 */
export interface UpdateMemberRequest {
  role: 'viewer' | 'member' | 'admin';
}

// ===========================================================================
// Retention Policy Types
// ===========================================================================

/**
 * Retention policy definition
 */
export interface RetentionPolicy {
  id: string;
  name: string;
  description?: string;
  retention_days: number;
  data_types: string[];
  workspace_id?: string;
  action: 'archive' | 'delete' | 'anonymize';
  enabled: boolean;
  schedule?: string;
  last_executed?: string;
  next_execution?: string;
  created_at: string;
  updated_at: string;
}

/**
 * Create retention policy request
 */
export interface CreateRetentionPolicyRequest {
  name: string;
  retention_days: number;
  data_types?: string[];
  workspace_id?: string;
  description?: string;
  action?: 'archive' | 'delete' | 'anonymize';
}

/**
 * Update retention policy request
 */
export interface UpdateRetentionPolicyRequest {
  name?: string;
  retention_days?: number;
  data_types?: string[];
  description?: string;
  action?: 'archive' | 'delete' | 'anonymize';
  enabled?: boolean;
}

/**
 * Expiring data item
 */
export interface ExpiringItem {
  id: string;
  type: string;
  name?: string;
  expires_at: string;
  policy_id: string;
  workspace_id?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Retention policy execution result
 */
export interface RetentionExecutionResult {
  policy_id: string;
  executed_at: string;
  items_processed: number;
  items_archived: number;
  items_deleted: number;
  items_anonymized: number;
  errors: string[];
  success: boolean;
}

// ===========================================================================
// Content Classification Types
// ===========================================================================

/**
 * Sensitivity level for classified content
 */
export type SensitivityLevel = 'public' | 'internal' | 'confidential' | 'restricted';

/**
 * Content classification result
 */
export interface ClassificationResult {
  level: SensitivityLevel;
  confidence: number;
  reasons?: string[];
  suggested_policies?: string[];
  metadata?: Record<string, unknown>;
}

/**
 * Classification policy details
 */
export interface ClassificationPolicy {
  level: SensitivityLevel;
  description: string;
  allowed_actions: string[];
  required_permissions: string[];
  retention_override?: number;
  encryption_required: boolean;
  audit_required: boolean;
}

// ===========================================================================
// Audit Types
// ===========================================================================

/**
 * Audit entry for workspace operations
 */
export interface WorkspaceAuditEntry {
  id: string;
  timestamp: string;
  action: string;
  actor_id: string;
  workspace_id?: string;
  resource_type?: string;
  resource_id?: string;
  details?: Record<string, unknown>;
  ip_address?: string;
  user_agent?: string;
}

/**
 * Audit query options
 */
export interface AuditQueryOptions {
  workspace_id?: string;
  action?: string;
  start_time?: string;
  end_time?: string;
  actor_id?: string;
  resource_type?: string;
  limit?: number;
  offset?: number;
}

/**
 * Audit report types
 */
export type AuditReportType = 'compliance' | 'access' | 'data' | 'security';

/**
 * Audit report result
 */
export interface AuditReport {
  report_type: AuditReportType;
  generated_at: string;
  start_date?: string;
  end_date?: string;
  summary: {
    total_events: number;
    events_by_action: Record<string, number>;
    top_actors: Array<{ actor_id: string; event_count: number }>;
    risk_score?: number;
  };
  findings?: Array<{
    severity: 'low' | 'medium' | 'high' | 'critical';
    description: string;
    recommendation?: string;
  }>;
  compliance_status?: {
    framework: string;
    compliant: boolean;
    score: number;
    gaps: string[];
  };
}

/**
 * Audit integrity verification result
 */
export interface AuditVerificationResult {
  verified: boolean;
  entries_checked: number;
  tampered_entries: number;
  last_verified?: string;
  hash_algorithm?: string;
  chain_valid: boolean;
}

// ===========================================================================
// Client Interface
// ===========================================================================

/**
 * Interface for the internal client used by WorkspacesAPI.
 */
interface WorkspacesClientInterface {
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
  put<T>(path: string, body?: unknown): Promise<T>;
  delete<T>(path: string): Promise<T>;
  request<T>(method: string, path: string, options?: { params?: Record<string, unknown>; json?: unknown }): Promise<T>;
}

/**
 * Workspaces API namespace.
 *
 * Provides methods for managing workspaces:
 * - Create and configure workspaces
 * - Manage workspace members and roles
 * - Configure workspace settings and profiles
 * - Manage retention policies
 * - Classify content sensitivity
 * - Query audit logs
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Create a workspace
 * const workspace = await client.workspaces.create({
 *   name: 'Engineering Team',
 *   description: 'Technical discussions and decision-making',
 * });
 *
 * // Add members
 * await client.workspaces.addMember(workspace.id, { email: 'dev@company.com', role: 'member' });
 *
 * // List members
 * const { members } = await client.workspaces.listMembers(workspace.id);
 *
 * // Update settings
 * await client.workspaces.update(workspace.id, {
 *   settings: { allow_public_debates: false },
 * });
 *
 * // Create retention policy
 * const policy = await client.workspaces.createRetentionPolicy({
 *   name: 'GDPR Compliance',
 *   retention_days: 365,
 *   data_types: ['debates', 'messages'],
 * });
 *
 * // Classify content
 * const classification = await client.workspaces.classifyContent('Confidential financial data');
 *
 * // Query audit logs
 * const { entries } = await client.workspaces.queryAuditEntries({
 *   workspace_id: workspace.id,
 *   action: 'member.added',
 * });
 * ```
 */
export class WorkspacesAPI {
  constructor(private client: WorkspacesClientInterface) {}

  // ===========================================================================
  // Workspace Management
  // ===========================================================================

  /**
   * List workspaces with optional filtering.
   *
   * @param options - Filter and pagination options
   * @returns List of workspaces with total count
   *
   * @example
   * ```typescript
   * // List all workspaces
   * const { workspaces, total } = await client.workspaces.list();
   *
   * // List workspaces for a specific tenant
   * const { workspaces } = await client.workspaces.list({
   *   tenant_id: 'tenant-123',
   *   limit: 10,
   * });
   * ```
   */
  async list(options?: {
    tenant_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ workspaces: Workspace[]; total: number }> {
    const params: Record<string, unknown> = {
      limit: options?.limit ?? 20,
      offset: options?.offset ?? 0,
    };
    if (options?.tenant_id) {
      params.tenant_id = options.tenant_id;
    }
    return this.client.request('GET', '/api/v1/workspaces', { params });
  }

  /**
   * Get a specific workspace by ID.
   *
   * @param workspaceId - The workspace ID
   * @returns Workspace details
   *
   * @example
   * ```typescript
   * const workspace = await client.workspaces.get('ws-123');
   * console.log(`Workspace: ${workspace.name}`);
   * ```
   */
  async get(workspaceId: string): Promise<Workspace> {
    return this.client.get(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}`);
  }

  /**
   * Create a new workspace.
   *
   * @param body - Workspace creation parameters
   * @returns Created workspace
   *
   * @example
   * ```typescript
   * const workspace = await client.workspaces.create({
   *   name: 'Engineering',
   *   tenant_id: 'tenant-123',
   *   description: 'Engineering team workspace',
   *   settings: {
   *     allow_public_debates: false,
   *     retention_days: 365,
   *   },
   * });
   * ```
   */
  async create(body: CreateWorkspaceRequest): Promise<Workspace> {
    return this.client.post('/api/v1/workspaces', body);
  }

  /**
   * Update an existing workspace.
   *
   * @param workspaceId - The workspace ID
   * @param body - Fields to update
   * @returns Updated workspace
   *
   * @example
   * ```typescript
   * const workspace = await client.workspaces.update('ws-123', {
   *   name: 'Engineering Team',
   *   settings: { allow_public_debates: false },
   * });
   * ```
   */
  async update(workspaceId: string, body: UpdateWorkspaceRequest): Promise<Workspace> {
    return this.client.put(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}`, body);
  }

  /**
   * Delete a workspace.
   *
   * @param workspaceId - The workspace ID to delete
   * @returns Deletion confirmation
   *
   * @example
   * ```typescript
   * const result = await client.workspaces.delete('ws-123');
   * if (result.success) {
   *   console.log('Workspace deleted');
   * }
   * ```
   */
  async delete(workspaceId: string): Promise<{ success: boolean; deleted?: boolean }> {
    return this.client.delete(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}`);
  }

  // ===========================================================================
  // Member Management
  // ===========================================================================

  /**
   * List workspace members.
   *
   * @param workspaceId - The workspace ID
   * @param options - Pagination options
   * @returns List of members with total count
   *
   * @example
   * ```typescript
   * const { members, total } = await client.workspaces.listMembers('ws-123');
   * for (const member of members) {
   *   console.log(`${member.email}: ${member.role}`);
   * }
   * ```
   */
  async listMembers(
    workspaceId: string,
    options?: { limit?: number; offset?: number }
  ): Promise<{ members: WorkspaceMember[]; total: number }> {
    const params: Record<string, unknown> = {
      limit: options?.limit ?? 50,
      offset: options?.offset ?? 0,
    };
    return this.client.request('GET', `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/members`, { params });
  }

  /**
   * Add a member to the workspace.
   *
   * @param workspaceId - The workspace ID
   * @param body - Member details (user_id or email, and role)
   * @returns Added member details
   *
   * @example
   * ```typescript
   * // Add by user ID
   * const member = await client.workspaces.addMember('ws-123', {
   *   user_id: 'user-456',
   *   role: 'member',
   * });
   *
   * // Add by email (will invite if not a user)
   * const member = await client.workspaces.addMember('ws-123', {
   *   email: 'dev@company.com',
   *   role: 'admin',
   * });
   * ```
   */
  async addMember(workspaceId: string, body: AddMemberRequest): Promise<WorkspaceMember> {
    return this.client.post(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}/members`, body);
  }

  /**
   * Update a member's role.
   *
   * @param workspaceId - The workspace ID
   * @param userId - The user ID to update
   * @param body - New role assignment
   * @returns Updated member details
   *
   * @example
   * ```typescript
   * const member = await client.workspaces.updateMember('ws-123', 'user-456', {
   *   role: 'admin',
   * });
   * ```
   */
  async updateMember(
    workspaceId: string,
    userId: string,
    body: UpdateMemberRequest
  ): Promise<WorkspaceMember> {
    return this.client.put(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}`,
      body
    );
  }

  /**
   * Remove a member from the workspace.
   *
   * @param workspaceId - The workspace ID
   * @param userId - The user ID to remove
   * @returns Removal confirmation
   *
   * @example
   * ```typescript
   * const result = await client.workspaces.removeMember('ws-123', 'user-456');
   * if (result.success) {
   *   console.log('Member removed');
   * }
   * ```
   */
  async removeMember(
    workspaceId: string,
    userId: string
  ): Promise<{ success: boolean; removed?: boolean }> {
    return this.client.delete(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}`
    );
  }

  /**
   * Update a member's role in a workspace.
   */
  async updateMemberRole(
    workspaceId: string,
    userId: string,
    role: 'viewer' | 'member' | 'admin'
  ): Promise<WorkspaceMember> {
    return this.client.put(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}/role`,
      { role }
    );
  }

  /**
   * Get available roles for a workspace.
   */
  async getWorkspaceRoles(workspaceId: string): Promise<{
    roles: Array<{ name: string; description: string; permissions: string[] }>;
  }> {
    return this.client.get(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/roles`
    );
  }

  // ===========================================================================
  // Invite Management
  // ===========================================================================

  /**
   * Create an invite to join a workspace.
   */
  async createInvite(
    workspaceId: string,
    body: { email: string; role?: 'viewer' | 'member' | 'admin' }
  ): Promise<{
    invite_id: string;
    token: string;
    email: string;
    expires_at: string;
  }> {
    return this.client.post(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/invites`,
      body
    );
  }

  /**
   * List pending invites for a workspace.
   */
  async listInvites(workspaceId: string): Promise<{
    invites: Array<{
      invite_id: string;
      email: string;
      role: string;
      created_at: string;
      expires_at: string;
    }>;
  }> {
    return this.client.get(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/invites`
    );
  }

  /**
   * Cancel a pending invite.
   */
  async cancelInvite(
    workspaceId: string,
    inviteId: string
  ): Promise<{ success: boolean }> {
    return this.client.delete(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/invites/${encodeURIComponent(inviteId)}`
    );
  }

  /**
   * Resend an invite email.
   */
  async resendInvite(
    workspaceId: string,
    inviteId: string
  ): Promise<{ success: boolean }> {
    return this.client.post(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/invites/${encodeURIComponent(inviteId)}/resend`
    );
  }

  /**
   * Accept a workspace invite.
   */
  async acceptInvite(
    workspaceId: string,
    inviteId: string
  ): Promise<{ success: boolean; workspace_id: string }> {
    return this.client.post(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/invites/${encodeURIComponent(inviteId)}/accept`
    );
  }

  // ===========================================================================
  // Profiles
  // ===========================================================================

  /**
   * List workspace profiles (permission sets).
   *
   * @param workspaceId - The workspace ID
   * @returns List of profiles
   *
   * @example
   * ```typescript
   * const { profiles } = await client.workspaces.listProfiles('ws-123');
   * for (const profile of profiles) {
   *   console.log(`${profile.name}: ${profile.permissions.length} permissions`);
   * }
   * ```
   */
  async listProfiles(workspaceId: string): Promise<{ profiles: WorkspaceProfile[] }> {
    return this.client.get(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}/profiles`);
  }

  // ===========================================================================
  // Retention Policies
  // ===========================================================================

  /**
   * List all retention policies.
   *
   * @returns List of retention policies
   *
   * @example
   * ```typescript
   * const { policies } = await client.workspaces.listRetentionPolicies();
   * for (const policy of policies) {
   *   console.log(`${policy.name}: ${policy.retention_days} days`);
   * }
   * ```
   */
  async listRetentionPolicies(): Promise<{ policies: RetentionPolicy[] }> {
    return this.client.get('/api/v1/retention/policies');
  }

  /**
   * Create a retention policy.
   *
   * @param body - Retention policy configuration
   * @returns Created policy
   *
   * @example
   * ```typescript
   * const policy = await client.workspaces.createRetentionPolicy({
   *   name: 'GDPR Compliance',
   *   retention_days: 365,
   *   data_types: ['debates', 'messages', 'uploads'],
   *   workspace_id: 'ws-123', // Optional: scope to workspace
   * });
   * ```
   */
  async createRetentionPolicy(body: CreateRetentionPolicyRequest): Promise<RetentionPolicy> {
    return this.client.post('/api/v1/retention/policies', body);
  }

  /**
   * Update a retention policy.
   *
   * @param policyId - The policy ID to update
   * @param body - Fields to update
   * @returns Updated policy
   *
   * @example
   * ```typescript
   * const policy = await client.workspaces.updateRetentionPolicy('policy-123', {
   *   retention_days: 180,
   *   data_types: ['debates', 'messages'],
   * });
   * ```
   */
  async updateRetentionPolicy(
    policyId: string,
    body: UpdateRetentionPolicyRequest
  ): Promise<RetentionPolicy> {
    return this.client.put(`/api/v1/retention/policies/${encodeURIComponent(policyId)}`, body);
  }

  /**
   * Delete a retention policy.
   *
   * @param policyId - The policy ID to delete
   * @returns Deletion confirmation
   *
   * @example
   * ```typescript
   * const result = await client.workspaces.deleteRetentionPolicy('policy-123');
   * if (result.success) {
   *   console.log('Policy deleted');
   * }
   * ```
   */
  async deleteRetentionPolicy(policyId: string): Promise<{ success: boolean }> {
    return this.client.delete(`/api/v1/retention/policies/${encodeURIComponent(policyId)}`);
  }

  /**
   * Execute a retention policy manually.
   *
   * This applies the retention policy immediately, archiving or deleting
   * data according to the policy rules.
   *
   * @param policyId - The policy ID to execute
   * @param options - Execution options
   * @returns Execution results
   *
   * @example
   * ```typescript
   * // Execute policy
   * const result = await client.workspaces.executeRetentionPolicy('policy-123');
   * console.log(`Processed: ${result.items_processed}`);
   * console.log(`Deleted: ${result.items_deleted}`);
   *
   * // Dry run to preview
   * const preview = await client.workspaces.executeRetentionPolicy('policy-123', {
   *   dry_run: true,
   * });
   * ```
   */
  async executeRetentionPolicy(
    policyId: string,
    options?: { dry_run?: boolean }
  ): Promise<RetentionExecutionResult> {
    return this.client.post(`/api/v1/retention/policies/${encodeURIComponent(policyId)}/execute`, options);
  }

  /**
   * Get items expiring within a specified number of days.
   *
   * @param options - Query options
   * @returns List of expiring items
   *
   * @example
   * ```typescript
   * // Get items expiring in the next 30 days
   * const { items, total } = await client.workspaces.getExpiringItems({ days: 30 });
   * for (const item of items) {
   *   console.log(`${item.type} ${item.id} expires at ${item.expires_at}`);
   * }
   * ```
   */
  async getExpiringItems(options?: {
    days?: number;
    limit?: number;
    offset?: number;
  }): Promise<{ items: ExpiringItem[]; total: number }> {
    const params: Record<string, unknown> = {
      days: options?.days ?? 30,
    };
    if (options?.limit) params.limit = options.limit;
    if (options?.offset) params.offset = options.offset;

    return this.client.request('GET', '/api/v1/retention/expiring', { params });
  }

  // ===========================================================================
  // Content Classification
  // ===========================================================================

  /**
   * Classify content sensitivity level.
   *
   * Analyzes content and returns the appropriate sensitivity classification
   * (public, internal, confidential, restricted).
   *
   * @param content - The content to classify
   * @param contentType - Type of content (default: "text")
   * @returns Classification result with confidence score
   *
   * @example
   * ```typescript
   * const result = await client.workspaces.classifyContent(
   *   'Customer SSN: 123-45-6789',
   *   'text'
   * );
   * console.log(`Level: ${result.level}`);
   * console.log(`Confidence: ${result.confidence}`);
   * // Level: restricted
   * // Confidence: 0.95
   * ```
   */
  async classifyContent(
    content: string,
    contentType: string = 'text'
  ): Promise<ClassificationResult> {
    return this.client.post('/api/v1/classify', {
      content,
      content_type: contentType,
    });
  }

  /**
   * Get classification policy for a sensitivity level.
   *
   * @param level - Sensitivity level (public, internal, confidential, restricted)
   * @returns Policy details for the level
   *
   * @example
   * ```typescript
   * const policy = await client.workspaces.getClassificationPolicy('confidential');
   * console.log(`Encryption required: ${policy.encryption_required}`);
   * console.log(`Allowed actions: ${policy.allowed_actions.join(', ')}`);
   * ```
   */
  async getClassificationPolicy(level: SensitivityLevel): Promise<ClassificationPolicy> {
    return this.client.get(`/api/v1/classify/policy/${encodeURIComponent(level)}`);
  }

  // ===========================================================================
  // Audit
  // ===========================================================================

  /**
   * Query audit log entries.
   *
   * @param options - Query filters and pagination
   * @returns Audit entries matching the query
   *
   * @example
   * ```typescript
   * // Query recent entries for a workspace
   * const { entries, total } = await client.workspaces.queryAuditEntries({
   *   workspace_id: 'ws-123',
   *   action: 'member.added',
   *   limit: 100,
   * });
   *
   * // Query by time range
   * const { entries } = await client.workspaces.queryAuditEntries({
   *   start_time: '2024-01-01T00:00:00Z',
   *   end_time: '2024-03-31T23:59:59Z',
   * });
   * ```
   */
  async queryAuditEntries(
    options?: AuditQueryOptions
  ): Promise<{ entries: WorkspaceAuditEntry[]; total: number }> {
    const params: Record<string, unknown> = {
      limit: options?.limit ?? 100,
    };
    if (options?.workspace_id) params.workspace_id = options.workspace_id;
    if (options?.action) params.action = options.action;
    if (options?.start_time) params.start_time = options.start_time;
    if (options?.end_time) params.end_time = options.end_time;
    if (options?.actor_id) params.actor_id = options.actor_id;
    if (options?.resource_type) params.resource_type = options.resource_type;
    if (options?.offset) params.offset = options.offset;

    return this.client.request('GET', '/api/v1/audit/entries', { params });
  }

  /**
   * Generate a compliance audit report.
   *
   * @param options - Report configuration
   * @returns Generated audit report
   *
   * @example
   * ```typescript
   * // Generate monthly compliance report
   * const report = await client.workspaces.generateAuditReport({
   *   report_type: 'compliance',
   *   start_date: '2024-01-01',
   *   end_date: '2024-01-31',
   * });
   * console.log(`Total events: ${report.summary.total_events}`);
   *
   * // Generate access report
   * const accessReport = await client.workspaces.generateAuditReport({
   *   report_type: 'access',
   * });
   * ```
   */
  async generateAuditReport(options?: {
    report_type?: AuditReportType;
    start_date?: string;
    end_date?: string;
  }): Promise<AuditReport> {
    const params: Record<string, unknown> = {
      report_type: options?.report_type ?? 'compliance',
    };
    if (options?.start_date) params.start_date = options.start_date;
    if (options?.end_date) params.end_date = options.end_date;

    return this.client.request('GET', '/api/v1/audit/report', { params });
  }

  /**
   * Verify audit log integrity.
   *
   * Checks the cryptographic chain of audit entries to detect tampering.
   *
   * @returns Verification result
   *
   * @example
   * ```typescript
   * const result = await client.workspaces.verifyAuditIntegrity();
   * if (result.verified) {
   *   console.log(`Verified ${result.entries_checked} entries`);
   * } else {
   *   console.log(`Found ${result.tampered_entries} tampered entries!`);
   * }
   * ```
   */
  async verifyAuditIntegrity(): Promise<AuditVerificationResult> {
    return this.client.get('/api/v1/audit/verify');
  }
}
