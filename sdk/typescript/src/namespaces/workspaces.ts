/**
 * Workspaces Namespace API
 *
 * Provides methods for workspace management, data isolation, and privacy controls.
 *
 * Features:
 * - Workspace creation and listing
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
 * Create workspace request
 */
export interface CreateWorkspaceRequest {
  name: string;
  slug?: string;
  tenant_id?: string;
  description?: string;
  settings?: Partial<WorkspaceSettings>;
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
 * - Create and list workspaces
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

  // ===========================================================================
  // Invites
  // ===========================================================================

  /**
   * List workspace invites.
   *
   * @param params - Optional query parameters
   * @returns Invite entries
   */
  async listInvites(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/invites', { params }) as Promise<Record<string, unknown>>;
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
