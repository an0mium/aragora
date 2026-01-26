/**
 * Audit Namespace API
 *
 * Provides a namespaced interface for audit logging operations.
 * This wraps the flat client methods for a more intuitive API.
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
 * Options for listing audit events.
 */
export interface AuditEventFilterOptions {
  /** Filter by event type */
  event_type?: string;
  /** Filter by user ID */
  user_id?: string;
  /** Filter by resource type */
  resource_type?: string;
  /** Filter by resource ID */
  resource_id?: string;
  /** Start date (ISO string) */
  start_date?: string;
  /** End date (ISO string) */
  end_date?: string;
}

/**
 * Options for exporting audit logs.
 */
export interface AuditExportOptions {
  /** Export format */
  format: 'json' | 'csv' | 'pdf';
  /** Start date for export range */
  start_date?: string;
  /** End date for export range */
  end_date?: string;
  /** Event types to include */
  event_types?: string[];
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
  generateAuditReport(sessionId: string, format?: 'json' | 'pdf' | 'markdown'): Promise<{ report_url: string }>;
}

/**
 * Audit API namespace.
 *
 * Provides methods for audit logging and compliance:
 * - Listing and filtering audit events
 * - Audit session management
 * - Exporting audit logs
 * - Integrity verification
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
 * // Get audit statistics
 * const stats = await client.audit.stats({ period: '7d' });
 *
 * // Create an audit session
 * const session = await client.audit.createSession({
 *   name: 'Q4 Security Review',
 *   scope: ['auth', 'rbac'],
 * });
 *
 * // Export audit logs
 * const { url } = await client.audit.export({
 *   format: 'csv',
 *   start_date: '2024-01-01',
 *   end_date: '2024-03-31',
 * });
 * ```
 */
export class AuditAPI {
  constructor(private client: AuditClientInterface) {}

  /**
   * List audit events with optional filtering.
   */
  async listEvents(
    params?: AuditEventFilterOptions & PaginationParams
  ): Promise<{ events: AuditEvent[]; total: number }> {
    return this.client.listAuditEvents(params);
  }

  /**
   * Get audit statistics.
   */
  async stats(params?: { period?: string }): Promise<AuditStats> {
    return this.client.getAuditStats(params);
  }

  /**
   * Export audit logs.
   */
  async export(request: AuditExportOptions): Promise<{ url: string; expires_at: string }> {
    return this.client.exportAuditLogs(request);
  }

  /**
   * Verify integrity of audit logs.
   */
  async verifyIntegrity(
    params?: { start_date?: string; end_date?: string }
  ): Promise<{
    verified: boolean;
    entries_checked: number;
    tampered_entries: number;
  }> {
    return this.client.verifyAuditIntegrity(params);
  }

  /**
   * List audit sessions.
   */
  async listSessions(
    params?: { status?: string } & PaginationParams
  ): Promise<{ sessions: AuditSession[]; total: number }> {
    return this.client.listAuditSessions(params);
  }

  /**
   * Get an audit session by ID.
   */
  async getSession(sessionId: string): Promise<AuditSession> {
    return this.client.getAuditSession(sessionId);
  }

  /**
   * Create a new audit session.
   */
  async createSession(request: CreateAuditSessionRequest): Promise<AuditSession> {
    return this.client.createAuditSession(request);
  }

  /**
   * Start an audit session.
   */
  async startSession(sessionId: string): Promise<{ started: boolean }> {
    return this.client.startAuditSession(sessionId);
  }

  /**
   * Pause an audit session.
   */
  async pauseSession(sessionId: string): Promise<{ paused: boolean }> {
    return this.client.pauseAuditSession(sessionId);
  }

  /**
   * Resume a paused audit session.
   */
  async resumeSession(sessionId: string): Promise<{ resumed: boolean }> {
    return this.client.resumeAuditSession(sessionId);
  }

  /**
   * Cancel an audit session.
   */
  async cancelSession(sessionId: string): Promise<{ cancelled: boolean }> {
    return this.client.cancelAuditSession(sessionId);
  }

  /**
   * Get findings from an audit session.
   */
  async getSessionFindings(
    sessionId: string,
    params?: PaginationParams
  ): Promise<{ findings: AuditFinding[] }> {
    return this.client.getAuditSessionFindings(sessionId, params);
  }

  /**
   * Generate a report for an audit session.
   */
  async generateReport(
    sessionId: string,
    format?: 'json' | 'pdf' | 'markdown'
  ): Promise<{ report_url: string }> {
    return this.client.generateAuditReport(sessionId, format);
  }
}
