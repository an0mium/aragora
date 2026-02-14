/**
 * Compliance Namespace API
 *
 * Provides methods for compliance and audit operations including
 * SOC 2 reporting, GDPR compliance, and audit trail verification.
 *
 * Features:
 * - SOC 2 Type II report generation
 * - GDPR data export and right-to-be-forgotten
 * - Audit trail verification
 * - SIEM-compatible event export
 */

/**
 * Audit event types for compliance tracking.
 */
export type AuditEventType =
  | 'authentication'
  | 'authorization'
  | 'data_access'
  | 'data_modification'
  | 'admin_action'
  | 'compliance';

/**
 * Compliance status across frameworks.
 */
export interface ComplianceStatus {
  soc2: {
    compliant: boolean;
    last_audit: string;
    findings_count: number;
  };
  gdpr: {
    compliant: boolean;
    data_processing_agreement: boolean;
    dpo_configured: boolean;
  };
  hipaa?: {
    compliant: boolean;
    baa_signed: boolean;
  };
  overall_status: 'compliant' | 'partial' | 'non_compliant';
}

/**
 * SOC 2 report structure.
 */
export interface Soc2Report {
  report_id: string;
  period_start: string;
  period_end: string;
  controls: Soc2ControlAssessment[];
  findings: Soc2Finding[];
  overall_assessment: 'pass' | 'pass_with_exceptions' | 'fail';
  generated_at: string;
}

/**
 * SOC 2 control assessment.
 */
export interface Soc2ControlAssessment {
  control_id: string;
  control_name: string;
  category: string;
  status: 'pass' | 'fail' | 'not_applicable';
  evidence_count: number;
  notes?: string;
}

/**
 * SOC 2 finding.
 */
export interface Soc2Finding {
  finding_id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  control_id: string;
  description: string;
  remediation?: string;
  status: 'open' | 'remediated' | 'accepted';
}

/**
 * GDPR data export result.
 */
export interface GdprExportResult {
  export_id: string;
  user_id: string;
  format: string;
  data: Record<string, unknown>;
  generated_at: string;
  expires_at: string;
}

/**
 * GDPR deletion result.
 */
export interface GdprDeletionResult {
  deletion_id: string;
  user_id: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  data_deleted: string[];
  data_retained: string[];
  retention_reason?: string;
  completed_at?: string;
}

/**
 * Audit trail verification result.
 */
export interface AuditVerificationResult {
  verified: boolean;
  period_start: string;
  period_end: string;
  events_checked: number;
  anomalies: AuditAnomaly[];
  integrity_hash: string;
  verified_at: string;
}

/**
 * Audit anomaly detected during verification.
 */
export interface AuditAnomaly {
  anomaly_id: string;
  type: 'gap' | 'tampering' | 'sequence_error' | 'timestamp_error';
  severity: 'low' | 'medium' | 'high';
  description: string;
  event_ids?: string[];
  detected_at: string;
}

/**
 * Audit event for SIEM export.
 */
export interface AuditEvent {
  event_id: string;
  event_type: AuditEventType;
  timestamp: string;
  user_id?: string;
  resource_type?: string;
  resource_id?: string;
  action: string;
  outcome: 'success' | 'failure';
  ip_address?: string;
  user_agent?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Audit events export result.
 */
export interface AuditEventsExport {
  events: AuditEvent[];
  total_count: number;
  period_start: string;
  period_end: string;
  format: string;
  exported_at: string;
}

// ===========================================================================
// EU AI Act Types
// ===========================================================================

/**
 * EU AI Act risk level classification.
 */
export type EuAiActRiskLevel = 'unacceptable' | 'high' | 'limited' | 'minimal';

/**
 * EU AI Act risk classification result.
 */
export interface EuAiActRiskClassification {
  risk_level: EuAiActRiskLevel;
  annex_iii_category: string | null;
  annex_iii_number: number | null;
  rationale: string;
  matched_keywords: string[];
  applicable_articles: string[];
  obligations: string[];
}

/**
 * EU AI Act conformity report.
 */
export interface EuAiActConformityReport {
  overall_status: string;
  receipt_id: string;
  article_mappings: Array<{
    article: string;
    article_title: string;
    receipt_field: string;
    evidence: string;
    status: string;
  }>;
  generated_at: string;
  integrity_hash: string;
}

/**
 * EU AI Act compliance artifact bundle.
 */
export interface EuAiActArtifactBundle {
  bundle_id: string;
  regulation: string;
  compliance_deadline: string;
  receipt_id: string;
  generated_at: string;
  risk_classification: EuAiActRiskClassification;
  conformity_report: EuAiActConformityReport;
  article_12_record_keeping: Record<string, unknown>;
  article_13_transparency: Record<string, unknown>;
  article_14_human_oversight: Record<string, unknown>;
  integrity_hash: string;
}

/**
 * Options for artifact bundle generation.
 */
export interface EuAiActBundleOptions {
  receipt: Record<string, unknown>;
  providerName?: string;
  providerContact?: string;
  euRepresentative?: string;
  systemName?: string;
  systemVersion?: string;
}

/**
 * Client interface for compliance operations.
 */
interface ComplianceClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Compliance API for enterprise compliance and audit operations.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get compliance status
 * const status = await client.compliance.getStatus();
 *
 * // Generate SOC 2 report
 * const report = await client.compliance.generateSoc2Report({
 *   startDate: '2024-01-01',
 *   endDate: '2024-12-31',
 * });
 *
 * // GDPR data export
 * const export = await client.compliance.gdprExport('user-123');
 * ```
 */
export class ComplianceAPI {
  constructor(private client: ComplianceClientInterface) {}

  // ===========================================================================
  // Compliance Status
  // ===========================================================================

  /**
   * Get overall compliance status across frameworks.
   *
   * Returns compliance status for SOC 2, GDPR, HIPAA (if applicable),
   * and an overall assessment.
   */
  async getStatus(): Promise<ComplianceStatus> {
    return this.client.request('GET', '/api/v2/compliance/status');
  }

  // ===========================================================================
  // SOC 2 Compliance
  // ===========================================================================

  /**
   * Generate SOC 2 compliance summary report.
   *
   * @param options - Report generation options
   * @param options.startDate - Report period start (ISO date)
   * @param options.endDate - Report period end (ISO date)
   * @param options.controls - Specific controls to include (default: all)
   */
  async generateSoc2Report(options?: {
    startDate?: string;
    endDate?: string;
    controls?: string[];
  }): Promise<Soc2Report> {
    const params: Record<string, unknown> = {};
    if (options?.startDate) {
      params.start_date = options.startDate;
    }
    if (options?.endDate) {
      params.end_date = options.endDate;
    }
    if (options?.controls) {
      params.controls = options.controls.join(',');
    }

    return this.client.request('GET', '/api/v2/compliance/soc2-report', { params });
  }

  // ===========================================================================
  // GDPR Compliance
  // ===========================================================================

  /**
   * Export user data for GDPR compliance (Article 15 - Right of Access).
   *
   * Generates a complete export of all personal data held for a user,
   * including debate participation, preferences, and activity logs.
   *
   * @param userId - ID of the user whose data to export
   * @param format - Export format (json for programmatic use, csv for spreadsheets)
   * @returns Export result with data and download expiration
   *
   * @example
   * ```typescript
   * // Export user data as JSON
   * const result = await client.compliance.gdprExport('user-123', 'json');
   * console.log(`Export ID: ${result.export_id}`);
   * console.log(`Data categories: ${Object.keys(result.data).join(', ')}`);
   * console.log(`Expires: ${result.expires_at}`);
   *
   * // Export as CSV for user to download
   * const csvExport = await client.compliance.gdprExport('user-123', 'csv');
   * ```
   */
  async gdprExport(userId: string, format: 'json' | 'csv' = 'json'): Promise<GdprExportResult> {
    return this.client.request('GET', '/api/v2/compliance/gdpr-export', {
      params: { user_id: userId, format },
    });
  }

  /**
   * Execute GDPR right to erasure (Article 17 - Right to be Forgotten).
   *
   * Initiates deletion of all personal data for a user. Some data may be
   * retained for legal compliance (e.g., audit logs, financial records).
   *
   * @param userId - ID of the user to erase
   * @param options - Deletion options
   * @param options.confirm - Must be true to confirm deletion (default: true)
   * @param options.reason - Reason for deletion request (recommended for audit)
   * @returns Deletion result with status and list of deleted/retained data
   *
   * @remarks
   * - This operation is irreversible
   * - Some data may be retained for legal compliance (listed in `data_retained`)
   * - The deletion may be processed asynchronously for large datasets
   *
   * @example
   * ```typescript
   * const result = await client.compliance.gdprRightToBeForgotten('user-123', {
   *   confirm: true,
   *   reason: 'User requested account deletion via support ticket #456',
   * });
   *
   * if (result.status === 'completed') {
   *   console.log(`Deleted: ${result.data_deleted.join(', ')}`);
   *   if (result.data_retained.length > 0) {
   *     console.log(`Retained for compliance: ${result.data_retained.join(', ')}`);
   *     console.log(`Reason: ${result.retention_reason}`);
   *   }
   * }
   * ```
   */
  async gdprRightToBeForgotten(
    userId: string,
    options?: {
      confirm?: boolean;
      reason?: string;
    }
  ): Promise<GdprDeletionResult> {
    const data: Record<string, unknown> = {
      user_id: userId,
      confirm: options?.confirm ?? true,
    };
    if (options?.reason) {
      data.reason = options.reason;
    }

    return this.client.request('POST', '/api/v2/compliance/gdpr/right-to-be-forgotten', {
      json: data,
    });
  }

  // ===========================================================================
  // Audit Trail
  // ===========================================================================

  /**
   * Verify audit trail integrity.
   *
   * Checks for gaps, tampering, sequence errors, and timestamp anomalies
   * in the audit log.
   *
   * @param options - Verification options
   * @param options.startDate - Verification period start (ISO date)
   * @param options.endDate - Verification period end (ISO date)
   */
  async verifyAuditTrail(options?: {
    startDate?: string;
    endDate?: string;
  }): Promise<AuditVerificationResult> {
    const data: Record<string, unknown> = {};
    if (options?.startDate) {
      data.start_date = options.startDate;
    }
    if (options?.endDate) {
      data.end_date = options.endDate;
    }

    return this.client.request('POST', '/api/v2/compliance/audit-verify', { json: data });
  }

  /**
   * Export audit events for SIEM integration.
   *
   * @param startDate - Export period start (ISO date)
   * @param endDate - Export period end (ISO date)
   * @param options - Export options
   * @param options.eventTypes - Filter by event types
   * @param options.format - Export format (json, elasticsearch)
   * @param options.limit - Maximum events to export
   */
  async exportAuditEvents(
    startDate: string,
    endDate: string,
    options?: {
      eventTypes?: AuditEventType[];
      format?: 'json' | 'elasticsearch';
      limit?: number;
    }
  ): Promise<AuditEventsExport> {
    const params: Record<string, unknown> = {
      start_date: startDate,
      end_date: endDate,
      format: options?.format ?? 'json',
      limit: options?.limit ?? 1000,
    };
    if (options?.eventTypes) {
      params.event_types = options.eventTypes.join(',');
    }

    return this.client.request('GET', '/api/v2/compliance/audit-events', { params });
  }

  /**
   * Get compliance violation details.
   */
  async getViolation(violationId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/compliance/violations/${violationId}`) as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // EU AI Act
  // ===========================================================================

  /**
   * Classify an AI use case by EU AI Act risk level.
   *
   * @param description - Free-text description of the AI use case.
   * @returns Risk classification with level, rationale, and obligations.
   *
   * @example
   * ```typescript
   * const result = await client.compliance.euAiActClassify(
   *   'AI system for employment screening and hiring decisions'
   * );
   * console.log(`Risk: ${result.classification.risk_level}`);
   * console.log(`Obligations: ${result.classification.obligations.join(', ')}`);
   * ```
   */
  async euAiActClassify(description: string): Promise<{ classification: EuAiActRiskClassification }> {
    return this.client.request('POST', '/api/v2/compliance/eu-ai-act/classify', {
      json: { description },
    });
  }

  /**
   * Generate a conformity report from a decision receipt.
   *
   * @param receipt - Decision receipt data.
   * @returns Conformity report with article-by-article assessment.
   */
  async euAiActAudit(receipt: Record<string, unknown>): Promise<{ conformity_report: EuAiActConformityReport }> {
    return this.client.request('POST', '/api/v2/compliance/eu-ai-act/audit', {
      json: { receipt },
    });
  }

  /**
   * Generate a full EU AI Act compliance artifact bundle.
   *
   * Produces Articles 12 (Record-Keeping), 13 (Transparency), and
   * 14 (Human Oversight) artifacts bundled with a conformity report
   * and SHA-256 integrity hash.
   *
   * @param options - Bundle generation options including receipt and provider details.
   * @returns Complete artifact bundle.
   *
   * @example
   * ```typescript
   * const result = await client.compliance.euAiActGenerateBundle({
   *   receipt: myDecisionReceipt,
   *   providerName: 'Acme Corp',
   *   systemName: 'Acme Decision Engine',
   * });
   * console.log(`Bundle: ${result.bundle.bundle_id}`);
   * console.log(`Hash: ${result.bundle.integrity_hash}`);
   * ```
   */
  async euAiActGenerateBundle(
    options: EuAiActBundleOptions
  ): Promise<{ bundle: EuAiActArtifactBundle }> {
    const body: Record<string, unknown> = { receipt: options.receipt };
    if (options.providerName) body.provider_name = options.providerName;
    if (options.providerContact) body.provider_contact = options.providerContact;
    if (options.euRepresentative) body.eu_representative = options.euRepresentative;
    if (options.systemName) body.system_name = options.systemName;
    if (options.systemVersion) body.system_version = options.systemVersion;

    return this.client.request('POST', '/api/v2/compliance/eu-ai-act/generate-bundle', {
      json: body,
    });
  }
}
