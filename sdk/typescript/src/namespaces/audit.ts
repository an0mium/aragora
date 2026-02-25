/**
 * Audit Namespace API
 *
 * Provides a namespaced interface for audit logging operations:
 * - List and filter audit events
 * - Export audit data for compliance
 * - Generate compliance reports
 * - Manage audit sessions
 * - Verify audit trail integrity
 */

import type {
  AuditEvent,
  AuditSession,
  AuditStats,
  AuditFinding,
  CreateAuditSessionRequest,
  PaginationParams,
} from '../types';

/**
 * Export format for audit data.
 */
export type ExportFormat = 'json' | 'csv' | 'pdf';

/**
 * Report format for audit session reports.
 */
export type ReportFormat = 'json' | 'pdf' | 'markdown';

/**
 * Compliance framework type.
 */
export type ComplianceFramework = 'soc2' | 'gdpr' | 'hipaa';

/**
 * Report period type.
 */
export type ReportPeriod = 'daily' | 'weekly' | 'monthly' | 'quarterly';

/**
 * Options for listing audit events.
 */
export interface AuditEventFilterOptions {
  /** Filter by event type (e.g., "debate.created") */
  event_type?: string;
  /** Filter by actor (user/agent) ID */
  actor_id?: string;
  /** Filter by user ID (alias for actor_id) */
  user_id?: string;
  /** Filter by resource type */
  resource_type?: string;
  /** Filter by resource ID */
  resource_id?: string;
  /** Start date (ISO string) */
  start_date?: string;
  /** End date (ISO string) */
  end_date?: string;
  /** Start date (ISO string) - alias for from_date */
  from_date?: string;
  /** End date (ISO string) - alias for to_date */
  to_date?: string;
}

/**
 * Options for exporting audit logs.
 */
export interface AuditExportOptions {
  /** Export format */
  format: ExportFormat;
  /** Start date for export range */
  start_date?: string;
  /** End date for export range */
  end_date?: string;
  /** Start date (ISO string) - alias for from_date */
  from_date?: string;
  /** End date (ISO string) - alias for to_date */
  to_date?: string;
  /** Event types to include */
  event_types?: string[];
}

/**
 * Compliance report response.
 */
export interface ComplianceReport {
  period: string;
  framework?: string;
  generated_at: string;
  metrics: Record<string, number>;
  findings: Array<{
    severity: string;
    description: string;
    recommendation: string;
  }>;
  compliance_score: number;
}

/**
 * Actor activity summary.
 */
export interface ActorActivity {
  actor_id: string;
  total_events: number;
  events_by_type: Record<string, number>;
  first_seen: string;
  last_seen: string;
  resources_accessed: string[];
}

/**
 * Resource audit history.
 */
export interface ResourceHistory {
  resource_type: string;
  resource_id: string;
  events: AuditEvent[];
  total: number;
}

/**
 * Audit entry (OpenAPI-aligned).
 */
export interface AuditEntry {
  id: string;
  timestamp: string;
  action: string;
  actor_id: string;
  resource_type?: string;
  resource_id?: string;
  details?: Record<string, unknown>;
}

/**
 * Audit verification result.
 */
export interface AuditVerification {
  verified: boolean;
  entries_checked: number;
  tampered_entries: number;
  last_verified?: string;
  hash_algorithm?: string;
}

/**
 * Session events response.
 */
export interface SessionEvents {
  session_id: string;
  events: AuditEvent[];
  total: number;
}

/**
 * Interface for the internal client methods used by AuditAPI.
 */
interface AuditClientInterface {
  listAuditEvents(params?: AuditEventFilterOptions & PaginationParams): Promise<{ events: AuditEvent[]; total: number }>;
  getAuditStats(params?: { period?: string }): Promise<AuditStats>;
  exportAuditLogs(request: AuditExportOptions): Promise<{ url: string; expires_at: string }>;
  verifyAuditIntegrity(params?: { start_date?: string; end_date?: string }): Promise<{
    verified: boolean;
    entries_checked: number;
    tampered_entries: number;
  }>;
  listAuditSessions(params?: { status?: string } & PaginationParams): Promise<{ sessions: AuditSession[]; total: number }>;
  getAuditSession(sessionId: string): Promise<AuditSession>;
  createAuditSession(request: CreateAuditSessionRequest): Promise<AuditSession>;
  startAuditSession(sessionId: string): Promise<{ started: boolean }>;
  pauseAuditSession(sessionId: string): Promise<{ paused: boolean }>;
  resumeAuditSession(sessionId: string): Promise<{ resumed: boolean }>;
  cancelAuditSession(sessionId: string): Promise<{ cancelled: boolean }>;
  getAuditSessionFindings(sessionId: string, params?: PaginationParams): Promise<{ findings: AuditFinding[] }>;
  generateAuditReport(sessionId: string, format?: ReportFormat): Promise<{ report_url: string }>;
  // Generic request method for direct API calls
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown }
  ): Promise<T>;
}

/**
 * Audit API namespace.
 *
 * Provides methods for audit logging and compliance:
 * - Listing and filtering audit events
 * - Audit session management
 * - Exporting audit logs
 * - Integrity verification
 * - Compliance reporting
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // List recent audit events
 * const { events } = await client.audit.listEvents({
 *   event_type: 'auth.login',
 *   limit: 100,
 * });
 *
 * // Get a specific audit event
 * const event = await client.audit.getEvent('event-123');
 *
 * // Get audit statistics
 * const stats = await client.audit.getStats({ from_date: '2024-01-01' });
 *
 * // Verify audit trail integrity
 * const verification = await client.audit.verify();
 *
 * // Create an audit session
 * const session = await client.audit.createSession('Q4 Security Review', {
 *   scope: ['auth', 'rbac'],
 * });
 *
 * // Start the session
 * await client.audit.startSession(session.id);
 *
 * // Pause, resume, or cancel as needed
 * await client.audit.pauseSession(session.id);
 * await client.audit.resumeSession(session.id);
 *
 * // Get compliance report
 * const report = await client.audit.getComplianceReport('monthly', 'soc2');
 *
 * // Export audit logs
 * const { url } = await client.audit.export({
 *   format: 'csv',
 *   from_date: '2024-01-01',
 *   to_date: '2024-03-31',
 * });
 * ```
 */
export class AuditAPI {
  constructor(private client: AuditClientInterface) {}

  // ===========================================================================
  // Event Operations
  // ===========================================================================

  /**
   * List audit events with optional filtering.
   *
   * @param options - Filter and pagination options
   * @returns List of audit events with total count
   *
   * @example
   * ```typescript
   * const { events, total } = await client.audit.listEvents({
   *   event_type: 'debate.created',
   *   actor_id: 'user-123',
   *   from_date: '2024-01-01',
   *   to_date: '2024-03-31',
   *   limit: 50,
   *   offset: 0,
   * });
   * ```
   */
  async listEvents(
    options?: AuditEventFilterOptions & PaginationParams
  ): Promise<{ events: AuditEvent[]; total: number }> {
    const params: Record<string, unknown> = {
      limit: options?.limit ?? 50,
      offset: options?.offset ?? 0,
    };

    // Support both naming conventions
    if (options?.event_type) params.event_type = options.event_type;
    if (options?.actor_id || options?.user_id) {
      params.actor_id = options.actor_id || options.user_id;
    }
    if (options?.resource_type) params.resource_type = options.resource_type;
    if (options?.resource_id) params.resource_id = options.resource_id;
    if (options?.from_date || options?.start_date) {
      params.from_date = options.from_date || options.start_date;
    }
    if (options?.to_date || options?.end_date) {
      params.to_date = options.to_date || options.end_date;
    }

    return this.client.request<{ events: AuditEvent[]; total: number }>(
      'GET',
      '/api/v1/audit/entries',
      { params }
    );
  }

  /**
   * Get a specific audit event by ID.
   *
   * @param eventId - The audit event ID
   * @returns The audit event details
   *
   * @example
   * ```typescript
   * const event = await client.audit.getEvent('event-123');
   * console.log(`${event.timestamp}: ${event.action} by ${event.actor_id}`);
   * ```
   */
  async getEvent(eventId: string): Promise<AuditEvent> {
    return this.client.request<AuditEvent>(
      'GET',
      `/api/v1/audit/entries/${encodeURIComponent(eventId)}`
    );
  }

  // ===========================================================================
  // Statistics & Reports
  // ===========================================================================

  /**
   * Get audit statistics.
   *
   * @param options - Optional date range for statistics
   * @returns Audit statistics including event counts by type, top actors, etc.
   *
   * @example
   * ```typescript
   * const stats = await client.audit.getStats({
   *   from_date: '2024-01-01',
   *   to_date: '2024-03-31',
   * });
   * console.log(`Total events: ${stats.total_events}`);
   * ```
   */
  async getStats(options?: {
    from_date?: string;
    to_date?: string;
  }): Promise<AuditStats> {
    const params: Record<string, unknown> = {};
    if (options?.from_date) params.from_date = options.from_date;
    if (options?.to_date) params.to_date = options.to_date;

    return this.client.request<AuditStats>(
      'GET',
      '/api/v1/audit/report',
      { params }
    );
  }

  /**
   * Get audit statistics (alias for getStats).
   * @deprecated Use getStats() instead
   */
  async stats(params?: { period?: string }): Promise<AuditStats> {
    return this.client.getAuditStats(params);
  }

  /**
   * Get the global audit report.
   *
   * @returns The audit report
   *
   * @example
   * ```typescript
   * const report = await client.audit.getReport();
   * console.log(`Report generated at: ${report.generated_at}`);
   * ```
   */
  async getReport(): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      '/api/v1/audit/report'
    );
  }

  /**
   * Generate a compliance report.
   *
   * @param period - Report period (daily, weekly, monthly, quarterly)
   * @param framework - Optional compliance framework (soc2, gdpr, hipaa)
   * @returns Compliance report with metrics and findings
   *
   * @example
   * ```typescript
   * const report = await client.audit.getComplianceReport('monthly', 'soc2');
   * console.log(`Compliance score: ${report.compliance_score}%`);
   * for (const finding of report.findings) {
   *   console.log(`${finding.severity}: ${finding.description}`);
   * }
   * ```
   */
  async getComplianceReport(
    period: ReportPeriod = 'monthly',
    framework?: ComplianceFramework
  ): Promise<ComplianceReport> {
    const params: Record<string, unknown> = { period };
    if (framework) params.framework = framework;

    return this.client.request<ComplianceReport>(
      'GET',
      '/api/v1/audit/report',
      { params }
    );
  }

  // ===========================================================================
  // Actor & Resource History
  // ===========================================================================

  /**
   * Get activity summary for an actor.
   *
   * @param actorId - The actor (user/agent) ID
   * @param options - Optional date range
   * @returns Activity summary with event counts
   *
   * @example
   * ```typescript
   * const activity = await client.audit.getActorActivity('user-123', {
   *   from_date: '2024-01-01',
   *   to_date: '2024-03-31',
   * });
   * console.log(`Total events: ${activity.total_events}`);
   * console.log(`Resources accessed: ${activity.resources_accessed.join(', ')}`);
   * ```
   */
  async getActorActivity(
    actorId: string,
    options?: { from_date?: string; to_date?: string }
  ): Promise<ActorActivity> {
    const params: Record<string, unknown> = { actor_id: actorId };
    if (options?.from_date) params.from_date = options.from_date;
    if (options?.to_date) params.to_date = options.to_date;

    return this.client.request<ActorActivity>(
      'GET',
      `/api/v1/audit/actor/${encodeURIComponent(actorId)}/history`,
      { params }
    );
  }

  /**
   * Get audit history for a specific resource.
   *
   * @param resourceType - Resource type (debate, agent, etc.)
   * @param resourceId - Resource ID
   * @returns Resource audit history
   *
   * @example
   * ```typescript
   * const history = await client.audit.getResourceHistory('debate', 'debate-123');
   * console.log(`${history.total} events for this debate`);
   * for (const event of history.events) {
   *   console.log(`${event.timestamp}: ${event.action}`);
   * }
   * ```
   */
  async getResourceHistory(
    resourceType: string,
    resourceId: string
  ): Promise<ResourceHistory> {
    return this.client.request<ResourceHistory>(
      'GET',
      `/api/audit/resources/${encodeURIComponent(resourceType)}/${encodeURIComponent(resourceId)}/history`
    );
  }

  // ===========================================================================
  // Export & Verification
  // ===========================================================================

  /**
   * Export audit events.
   *
   * @param options - Export options including format and date range
   * @returns Export result with download URL
   *
   * @example
   * ```typescript
   * const { url, expires_at } = await client.audit.export({
   *   format: 'csv',
   *   from_date: '2024-01-01',
   *   to_date: '2024-03-31',
   *   event_types: ['debate.created', 'debate.completed'],
   * });
   * console.log(`Download from: ${url}`);
   * ```
   */
  async export(options: AuditExportOptions): Promise<{ url: string; expires_at: string }> {
    const body: Record<string, unknown> = { format: options.format };

    // Support both naming conventions
    if (options.from_date || options.start_date) {
      body.from_date = options.from_date || options.start_date;
    }
    if (options.to_date || options.end_date) {
      body.to_date = options.to_date || options.end_date;
    }
    if (options.event_types) {
      body.event_types = options.event_types;
    }

    return this.client.request<{ url: string; expires_at: string }>(
      'POST',
      '/api/audit/export',
      { body }
    );
  }

  /**
   * Verify audit trail integrity.
   *
   * @returns Verification result
   *
   * @example
   * ```typescript
   * const result = await client.audit.verify();
   * if (result.verified) {
   *   console.log(`Verified ${result.entries_checked} entries`);
   * } else {
   *   console.log(`Found ${result.tampered_entries} tampered entries!`);
   * }
   * ```
   */
  async verify(): Promise<AuditVerification> {
    return this.client.request<AuditVerification>(
      'GET',
      '/api/v1/audit/verify'
    );
  }

  /**
   * Verify integrity of audit logs with optional date range.
   *
   * @param params - Optional date range for verification
   * @returns Verification result
   *
   * @example
   * ```typescript
   * const result = await client.audit.verifyIntegrity({
   *   start_date: '2024-01-01',
   *   end_date: '2024-03-31',
   * });
   * ```
   */
  async verifyIntegrity(
    params?: { start_date?: string; end_date?: string }
  ): Promise<AuditVerification> {
    return this.client.verifyAuditIntegrity(params);
  }

  // ===========================================================================
  // Entries (OpenAPI-aligned)
  // ===========================================================================

  /**
   * List audit entries (OpenAPI-aligned endpoint).
   *
   * @param options - Pagination options
   * @returns List of audit entries
   *
   * @example
   * ```typescript
   * const { entries, total } = await client.audit.listEntries({ limit: 50 });
   * for (const entry of entries) {
   *   console.log(`${entry.timestamp}: ${entry.action}`);
   * }
   * ```
   */
  async listEntries(options?: {
    limit?: number;
    offset?: number;
  }): Promise<{ entries: AuditEntry[]; total: number }> {
    const params: Record<string, unknown> = {
      limit: options?.limit ?? 50,
      offset: options?.offset ?? 0,
    };

    return this.client.request<{ entries: AuditEntry[]; total: number }>(
      'GET',
      '/api/v1/audit/entries',
      { params }
    );
  }

  // ===========================================================================
  // Session Management
  // ===========================================================================

  /**
   * List audit sessions.
   *
   * @param options - Filter and pagination options
   * @returns List of audit sessions
   *
   * @example
   * ```typescript
   * const { sessions, total } = await client.audit.listSessions({
   *   status: 'active',
   *   limit: 10,
   * });
   * ```
   */
  async listSessions(
    options?: { status?: string } & PaginationParams
  ): Promise<{ sessions: AuditSession[]; total: number }> {
    const params: Record<string, unknown> = {
      limit: options?.limit ?? 50,
      offset: options?.offset ?? 0,
    };
    if (options?.status) params.status = options.status;

    return this.client.request<{ sessions: AuditSession[]; total: number }>(
      'GET',
      '/api/v1/audit/sessions',
      { params }
    );
  }

  /**
   * Create a new audit session.
   *
   * @param name - Session name
   * @param config - Optional session configuration
   * @returns The created audit session
   *
   * @example
   * ```typescript
   * const session = await client.audit.createSession('Q4 Security Review', {
   *   description: 'Quarterly security audit',
   *   target_type: 'system',
   *   scope: ['auth', 'rbac', 'debates'],
   * });
   * console.log(`Created session: ${session.id}`);
   * ```
   */
  async createSession(
    name: string,
    config?: Partial<Omit<CreateAuditSessionRequest, 'name'>>
  ): Promise<AuditSession> {
    const body: Record<string, unknown> = { name };
    if (config) {
      Object.assign(body, config);
    }

    return this.client.request<AuditSession>(
      'POST',
      '/api/v1/audit/sessions',
      { body }
    );
  }

  /**
   * Get an audit session by ID.
   *
   * @param sessionId - The session ID
   * @returns The audit session
   *
   * @example
   * ```typescript
   * const session = await client.audit.getSession('session-123');
   * console.log(`Session status: ${session.status}`);
   * ```
   */
  async getSession(sessionId: string): Promise<AuditSession> {
    return this.client.request<AuditSession>(
      'GET',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}`
    );
  }

  /**
   * Delete an audit session.
   *
   * @param sessionId - The session ID to delete
   * @returns Deletion result
   *
   * @example
   * ```typescript
   * const result = await client.audit.deleteSession('session-123');
   * if (result.success) {
   *   console.log('Session deleted');
   * }
   * ```
   */
  async deleteSession(sessionId: string): Promise<{ success: boolean; message?: string }> {
    return this.client.request<{ success: boolean; message?: string }>(
      'DELETE',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}`
    );
  }

  /**
   * Get events for an audit session.
   *
   * @param sessionId - The session ID
   * @returns Session events
   *
   * @example
   * ```typescript
   * const { events, total } = await client.audit.getSessionEvents('session-123');
   * console.log(`Found ${total} events in this session`);
   * ```
   */
  async getSessionEvents(sessionId: string): Promise<SessionEvents> {
    return this.client.request<SessionEvents>(
      'GET',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/events`
    );
  }

  /**
   * Get findings from an audit session.
   *
   * @param sessionId - The session ID
   * @param options - Pagination options
   * @returns Session findings
   *
   * @example
   * ```typescript
   * const { findings } = await client.audit.getSessionFindings('session-123');
   * for (const finding of findings) {
   *   console.log(`${finding.severity}: ${finding.title}`);
   * }
   * ```
   */
  async getSessionFindings(
    sessionId: string,
    options?: PaginationParams
  ): Promise<{ findings: AuditFinding[] }> {
    return this.client.request<{ findings: AuditFinding[] }>(
      'GET',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/findings`,
      { params: options as Record<string, unknown> }
    );
  }

  /**
   * Get report for an audit session.
   *
   * @param sessionId - The session ID
   * @returns Session report
   *
   * @example
   * ```typescript
   * const report = await client.audit.getSessionReport('session-123');
   * console.log(`Report: ${report.summary}`);
   * ```
   */
  async getSessionReport(sessionId: string): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/report`
    );
  }

  /**
   * Generate a report for an audit session.
   *
   * @param sessionId - The session ID
   * @param format - Report format (json, pdf, markdown)
   * @returns Report URL
   *
   * @example
   * ```typescript
   * const { report_url } = await client.audit.generateReport('session-123', 'pdf');
   * console.log(`Download report: ${report_url}`);
   * ```
   */
  async generateReport(
    sessionId: string,
    format: ReportFormat = 'json'
  ): Promise<{ report_url: string }> {
    return this.client.generateAuditReport(sessionId, format);
  }

  // ===========================================================================
  // Session Lifecycle
  // ===========================================================================

  /**
   * Start an audit session.
   *
   * @param sessionId - The session ID to start
   * @returns Start result
   *
   * @example
   * ```typescript
   * const result = await client.audit.startSession('session-123');
   * if (result.started) {
   *   console.log('Session started');
   * }
   * ```
   */
  async startSession(sessionId: string): Promise<{ started: boolean }> {
    return this.client.request<{ started: boolean }>(
      'POST',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/start`
    );
  }

  /**
   * Pause an audit session.
   *
   * @param sessionId - The session ID to pause
   * @returns Pause result
   *
   * @example
   * ```typescript
   * const result = await client.audit.pauseSession('session-123');
   * if (result.paused) {
   *   console.log('Session paused');
   * }
   * ```
   */
  async pauseSession(sessionId: string): Promise<{ paused: boolean }> {
    return this.client.request<{ paused: boolean }>(
      'POST',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/pause`
    );
  }

  /**
   * Resume a paused audit session.
   *
   * @param sessionId - The session ID to resume
   * @returns Resume result
   *
   * @example
   * ```typescript
   * const result = await client.audit.resumeSession('session-123');
   * if (result.resumed) {
   *   console.log('Session resumed');
   * }
   * ```
   */
  async resumeSession(sessionId: string): Promise<{ resumed: boolean }> {
    return this.client.request<{ resumed: boolean }>(
      'POST',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/resume`
    );
  }

  /**
   * Cancel an audit session.
   *
   * @param sessionId - The session ID to cancel
   * @returns Cancel result
   *
   * @example
   * ```typescript
   * const result = await client.audit.cancelSession('session-123');
   * if (result.cancelled) {
   *   console.log('Session cancelled');
   * }
   * ```
   */
  async cancelSession(sessionId: string): Promise<{ cancelled: boolean }> {
    return this.client.request<{ cancelled: boolean }>(
      'POST',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/cancel`
    );
  }

  /**
   * Intervene in an audit session.
   *
   * @param sessionId - The session ID
   * @param action - The intervention action to perform
   * @returns Intervention result
   *
   * @example
   * ```typescript
   * const result = await client.audit.interveneSession('session-123', 'escalate');
   * console.log(`Intervention result: ${result.success}`);
   * ```
   */
  async interveneSession(
    sessionId: string,
    action: string
  ): Promise<{ success: boolean; message?: string }> {
    return this.client.request<{ success: boolean; message?: string }>(
      'POST',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/intervene`,
      { body: { action } }
    );
  }

  /**
   * End an audit session.
   *
   * @param sessionId - The session ID to end
   * @returns End result with session summary
   *
   * @example
   * ```typescript
   * const result = await client.audit.endSession('session-123');
   * console.log(`Session ended with ${result.findings_count} findings`);
   * ```
   */
  async endSession(sessionId: string): Promise<{ ended: boolean; summary?: Record<string, unknown> }> {
    return this.client.request<{ ended: boolean; summary?: Record<string, unknown> }>(
      'POST',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/end`
    );
  }

  /**
   * Export an audit session.
   *
   * @param sessionId - The session ID to export
   * @param format - Export format (json, csv, pdf)
   * @returns Export URL
   *
   * @example
   * ```typescript
   * const { url } = await client.audit.exportSession('session-123', 'pdf');
   * console.log(`Download export: ${url}`);
   * ```
   */
  async exportSession(
    sessionId: string,
    format: ExportFormat = 'json'
  ): Promise<{ url: string; expires_at: string }> {
    return this.client.request<{ url: string; expires_at: string }>(
      'POST',
      `/api/v1/audit/sessions/${encodeURIComponent(sessionId)}/export`,
      { body: { format } }
    );
  }

  // ===========================================================================
  // Finding Management
  // ===========================================================================

  /**
   * List audit findings.
   *
   * @param options - Filter and pagination options
   * @returns List of findings
   *
   * @example
   * ```typescript
   * const { findings, total } = await client.audit.listFindings({
   *   status: 'open',
   *   priority: 'critical',
   *   limit: 20,
   * });
   * ```
   */
  async listFindings(options?: {
    session_id?: string;
    status?: string;
    priority?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ findings: AuditFinding[]; total: number }> {
    const params: Record<string, unknown> = {
      limit: options?.limit ?? 50,
      offset: options?.offset ?? 0,
    };
    if (options?.session_id) params.session_id = options.session_id;
    if (options?.status) params.status = options.status;
    if (options?.priority) params.priority = options.priority;

    return this.client.request<{ findings: AuditFinding[]; total: number }>(
      'GET',
      '/api/audit/findings',
      { params }
    );
  }

  /**
   * Get a specific audit finding.
   *
   * @param findingId - The finding ID
   * @returns Finding details
   *
   * @example
   * ```typescript
   * const finding = await client.audit.getFinding('finding-123');
   * console.log(`${finding.severity}: ${finding.title}`);
   * ```
   */
  async getFinding(findingId: string): Promise<AuditFinding> {
    return this.client.request<AuditFinding>(
      'GET',
      `/api/audit/findings/${encodeURIComponent(findingId)}`
    );
  }

  /**
   * Assign a finding to a user.
   *
   * @param findingId - The finding ID
   * @param assigneeId - The user ID to assign to
   * @returns Assignment result
   *
   * @example
   * ```typescript
   * const result = await client.audit.assignFinding('finding-123', 'user-456');
   * if (result.assigned) {
   *   console.log('Finding assigned successfully');
   * }
   * ```
   */
  async assignFinding(
    findingId: string,
    assigneeId: string
  ): Promise<{ assigned: boolean }> {
    return this.client.request<{ assigned: boolean }>('GET', `/api/v1/audit/findings/${encodeURIComponent(findingId)}/assign`,
      { body: { assignee_id: assigneeId } }
    );
  }

  /**
   * Unassign a finding.
   *
   * @param findingId - The finding ID
   * @returns Unassignment result
   *
   * @example
   * ```typescript
   * const result = await client.audit.unassignFinding('finding-123');
   * if (result.unassigned) {
   *   console.log('Finding unassigned');
   * }
   * ```
   */
  async unassignFinding(findingId: string): Promise<{ unassigned: boolean }> {
    return this.client.request<{ unassigned: boolean }>('GET', `/api/v1/audit/findings/${encodeURIComponent(findingId)}/unassign`
    );
  }

  /**
   * Update finding status.
   *
   * @param findingId - The finding ID
   * @param status - New status (open, in_progress, resolved, dismissed)
   * @param resolutionNotes - Optional resolution notes
   * @returns Updated finding
   *
   * @example
   * ```typescript
   * const finding = await client.audit.updateFindingStatus(
   *   'finding-123',
   *   'resolved',
   *   'Fixed by applying security patch'
   * );
   * ```
   */
  async updateFindingStatus(
    findingId: string,
    status: string,
    resolutionNotes?: string
  ): Promise<AuditFinding> {
    const body: Record<string, unknown> = { status };
    if (resolutionNotes) body.resolution_notes = resolutionNotes;

    return this.client.request<AuditFinding>('GET', `/api/v1/audit/findings/${encodeURIComponent(findingId)}/status`,
      { body }
    );
  }

  /**
   * Update finding priority.
   *
   * @param findingId - The finding ID
   * @param priority - New priority (critical, high, medium, low)
   * @returns Updated finding
   *
   * @example
   * ```typescript
   * const finding = await client.audit.updateFindingPriority('finding-123', 'critical');
   * ```
   */
  async updateFindingPriority(
    findingId: string,
    priority: string
  ): Promise<AuditFinding> {
    return this.client.request<AuditFinding>('GET', `/api/v1/audit/findings/${encodeURIComponent(findingId)}/priority`,
      { body: { priority } }
    );
  }

  /**
   * Add a comment to a finding.
   *
   * @param findingId - The finding ID
   * @param content - Comment text
   * @returns Created comment
   *
   * @example
   * ```typescript
   * const comment = await client.audit.addFindingComment(
   *   'finding-123',
   *   'Investigating the root cause'
   * );
   * ```
   */
  async addFindingComment(
    findingId: string,
    content: string
  ): Promise<{ id: string; content: string; created_at: string; author_id: string }> {
    return this.client.request<{ id: string; content: string; created_at: string; author_id: string }>('GET', `/api/v1/audit/findings/${encodeURIComponent(findingId)}/comments`,
      { body: { content } }
    );
  }

  /**
   * List comments on a finding.
   *
   * @param findingId - The finding ID
   * @returns List of comments
   *
   * @example
   * ```typescript
   * const { comments } = await client.audit.listFindingComments('finding-123');
   * for (const comment of comments) {
   *   console.log(`${comment.author_id}: ${comment.content}`);
   * }
   * ```
   */
  async listFindingComments(
    findingId: string
  ): Promise<{ comments: Array<{ id: string; content: string; created_at: string; author_id: string }> }> {
    return this.client.request<{ comments: Array<{ id: string; content: string; created_at: string; author_id: string }> }>(
      'GET',
      `/api/v1/audit/findings/${encodeURIComponent(findingId)}/comments`
    );
  }

  /**
   * List security audit debates.
   * @route GET /api/v1/audit/security/debate
   */
  async listSecurityDebates(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/audit/security/debate', { params }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get a specific security audit debate.
   * @route GET /api/v1/audit/security/debate/:id
   */
  async getSecurityDebate(debateId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/audit/security/debate/${debateId}`) as Promise<Record<string, unknown>>;
  }

}
