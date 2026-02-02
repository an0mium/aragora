/**
 * Security Namespace API
 *
 * Provides security status, health checks, and key management.
 *
 * Features:
 * - Overall security status monitoring
 * - Security health checks and scans
 * - Encryption key management (create, rotate, revoke)
 * - Audit logging and compliance
 * - Threat detection and resolution
 */

// =============================================================================
// Types
// =============================================================================

/**
 * Security level indicating overall security posture.
 */
export type SecurityLevel = 'healthy' | 'degraded' | 'critical';

/**
 * Status of a security key.
 */
export type KeyStatus = 'active' | 'expired' | 'revoked';

/**
 * Health check status for security components.
 */
export type CheckStatus = 'ok' | 'warning' | 'error';

/**
 * Threat status.
 */
export type ThreatStatus = 'active' | 'resolved' | 'dismissed';

/**
 * Overall security status response.
 */
export interface SecurityStatus {
  overall: SecurityLevel;
  encryption_enabled: boolean;
  audit_logging_enabled: boolean;
  mfa_enabled: boolean;
  last_security_scan?: string;
  active_threats: number;
  metadata?: Record<string, unknown>;
}

/**
 * Individual security health check result.
 */
export interface SecurityHealthCheck {
  component: string;
  status: CheckStatus;
  message?: string;
  last_checked: string;
}

/**
 * Security key details.
 */
export interface SecurityKey {
  id: string;
  name: string;
  algorithm: string;
  created_at: string;
  expires_at?: string;
  status: KeyStatus;
  metadata?: Record<string, unknown>;
}

/**
 * Request body for key rotation.
 */
export interface RotateKeyRequest {
  key_id?: string;
  algorithm?: string;
  reason?: string;
}

/**
 * Result of a key rotation operation.
 */
export interface RotateKeyResult {
  success: boolean;
  new_key_id: string;
  old_key_id: string;
  rotated_at: string;
}

/**
 * Request body for creating a new key.
 */
export interface CreateKeyRequest {
  name: string;
  algorithm?: string;
  expires_in_days?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Request body for revoking a key.
 */
export interface RevokeKeyRequest {
  reason?: string;
}

/**
 * Security scan information.
 */
export interface SecurityScan {
  id: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  progress?: number;
  started_at?: string;
  completed_at?: string;
  findings?: SecurityFinding[];
}

/**
 * Security scan finding.
 */
export interface SecurityFinding {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  description: string;
  recommendation?: string;
}

/**
 * Audit log entry.
 */
export interface AuditLogEntry {
  id: string;
  event_type: string;
  user_id?: string;
  resource_type?: string;
  resource_id?: string;
  action: string;
  timestamp: string;
  ip_address?: string;
  user_agent?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Compliance status for a standard.
 */
export interface ComplianceStatus {
  standard: string;
  status: 'compliant' | 'partial' | 'non_compliant';
  last_assessment?: string;
  controls_passed: number;
  controls_total: number;
  findings?: string[];
}

/**
 * Detected security threat.
 */
export interface SecurityThreat {
  id: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: ThreatStatus;
  description: string;
  detected_at: string;
  resolved_at?: string;
  resolution?: string;
  source?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Client interface for security operations.
 */
interface SecurityClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown }
  ): Promise<T>;
}

// =============================================================================
// SecurityAPI Class
// =============================================================================

/**
 * Security API for security management operations.
 *
 * Provides methods for:
 * - Overall security status monitoring
 * - Security health checks and scans
 * - Encryption key management (create, rotate, revoke)
 * - Audit logging and compliance
 * - Threat detection and resolution
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get security status
 * const status = await client.security.getStatus();
 * if (status.overall !== 'healthy') {
 *   console.log('Security issues detected!');
 * }
 *
 * // Run a security scan
 * const { id: scanId } = await client.security.runSecurityScan();
 * const scan = await client.security.getScanStatus(scanId);
 *
 * // Manage encryption keys
 * const keys = await client.security.listKeys();
 * const newKey = await client.security.createKey({
 *   name: 'production-key',
 *   algorithm: 'AES-256-GCM',
 * });
 *
 * // Check for threats
 * const { threats } = await client.security.listThreats({ status: 'active' });
 * ```
 */
export class SecurityAPI {
  constructor(private client: SecurityClientInterface) {}

  // ===========================================================================
  // Security Status
  // ===========================================================================

  /**
   * Get overall security status.
   *
   * Returns the overall security posture including encryption status,
   * audit logging status, MFA status, and active threat count.
   */
  async getStatus(): Promise<SecurityStatus> {
    return this.client.request('GET', '/api/v1/admin/security/status');
  }

  // ===========================================================================
  // Health Checks
  // ===========================================================================

  /**
   * Get security health checks.
   *
   * Runs checks on all security components and returns their status.
   */
  async getHealthChecks(): Promise<{ checks: SecurityHealthCheck[] }> {
    return this.client.request('GET', '/api/v1/admin/security/health');
  }

  /**
   * Trigger a security scan.
   *
   * Initiates a comprehensive security scan of the system.
   *
   * @returns The scan ID and initial status
   */
  async runSecurityScan(): Promise<SecurityScan> {
    return this.client.request('POST', '/api/v1/admin/security/scan');
  }

  /**
   * Get the status of a security scan.
   *
   * @param scanId - The scan identifier
   * @returns Scan status, progress, and findings
   */
  async getScanStatus(scanId: string): Promise<SecurityScan> {
    return this.client.request('GET', `/api/v1/admin/security/scan/${scanId}`);
  }

  // ===========================================================================
  // Key Management
  // ===========================================================================

  /**
   * List all security keys.
   *
   * Returns a list of encryption keys with their status and metadata.
   */
  async listKeys(): Promise<{ keys: SecurityKey[] }> {
    return this.client.request('GET', '/api/v1/admin/security/keys');
  }

  /**
   * Get details of a specific key.
   *
   * @param keyId - The key identifier
   * @returns Key details including status and metadata
   */
  async getKey(keyId: string): Promise<SecurityKey> {
    return this.client.request('GET', `/api/v1/admin/security/keys/${keyId}`);
  }

  /**
   * Create a new encryption key.
   *
   * @param request - Key creation parameters
   * @param request.name - Name for the key
   * @param request.algorithm - Encryption algorithm (default: AES-256-GCM)
   * @param request.expires_in_days - Optional expiration in days
   * @param request.metadata - Optional metadata
   * @returns Created key details
   */
  async createKey(request: CreateKeyRequest): Promise<SecurityKey> {
    const body: Record<string, unknown> = {
      name: request.name,
      algorithm: request.algorithm ?? 'AES-256-GCM',
    };
    if (request.expires_in_days !== undefined) {
      body.expires_in_days = request.expires_in_days;
    }
    if (request.metadata !== undefined) {
      body.metadata = request.metadata;
    }
    return this.client.request('POST', '/api/v1/admin/security/keys', { body });
  }

  /**
   * Revoke an encryption key.
   *
   * @param keyId - The key identifier
   * @param options - Revocation options
   * @param options.reason - Optional reason for revocation
   * @returns Confirmation of revocation
   */
  async revokeKey(
    keyId: string,
    options?: RevokeKeyRequest
  ): Promise<{ revoked: boolean; key_id: string; revoked_at: string }> {
    const body = options?.reason ? { reason: options.reason } : undefined;
    return this.client.request('POST', `/api/v1/admin/security/keys/${keyId}/revoke`, {
      body,
    });
  }

  /**
   * Rotate an encryption key.
   *
   * Creates a new key and deprecates the old one.
   *
   * @param request - Rotation parameters
   * @param request.key_id - Optional specific key to rotate
   * @param request.algorithm - Optional new algorithm to use
   * @param request.reason - Optional reason for rotation
   * @returns Rotation result with new and old key IDs
   */
  async rotateKey(request?: RotateKeyRequest): Promise<RotateKeyResult> {
    const body = request ? { ...request } : undefined;
    return this.client.request('POST', '/api/v1/admin/security/rotate-key', { body });
  }

  // ===========================================================================
  // Audit & Compliance
  // ===========================================================================

  /**
   * Get security audit log entries.
   *
   * @param options - Filtering and pagination options
   * @param options.limit - Maximum entries to return (default: 50)
   * @param options.offset - Pagination offset (default: 0)
   * @param options.event_type - Filter by event type
   * @param options.since - Filter events after this timestamp (ISO format)
   * @param options.until - Filter events before this timestamp (ISO format)
   * @returns Audit log entries and total count
   */
  async getAuditLog(options?: {
    limit?: number;
    offset?: number;
    event_type?: string;
    since?: string;
    until?: string;
  }): Promise<{ entries: AuditLogEntry[]; total: number }> {
    const params: Record<string, unknown> = {
      limit: options?.limit ?? 50,
      offset: options?.offset ?? 0,
    };
    if (options?.event_type) {
      params.event_type = options.event_type;
    }
    if (options?.since) {
      params.since = options.since;
    }
    if (options?.until) {
      params.until = options.until;
    }
    return this.client.request('GET', '/api/v1/admin/security/audit', { params });
  }

  /**
   * Get compliance status for security standards.
   *
   * Returns compliance status for various standards (SOC2, GDPR, etc.)
   */
  async getComplianceStatus(): Promise<{ standards: ComplianceStatus[] }> {
    return this.client.request('GET', '/api/v1/admin/security/compliance');
  }

  // ===========================================================================
  // Threat Detection
  // ===========================================================================

  /**
   * List detected threats.
   *
   * @param options - Filtering options
   * @param options.limit - Maximum threats to return (default: 50)
   * @param options.status - Filter by status (active/resolved/dismissed)
   * @returns List of detected threats
   */
  async listThreats(options?: {
    limit?: number;
    status?: ThreatStatus;
  }): Promise<{ threats: SecurityThreat[]; total: number }> {
    const params: Record<string, unknown> = {
      limit: options?.limit ?? 50,
    };
    if (options?.status) {
      params.status = options.status;
    }
    return this.client.request('GET', '/api/v1/admin/security/threats', { params });
  }

  /**
   * Mark a threat as resolved.
   *
   * @param threatId - The threat identifier
   * @param resolution - Description of how the threat was resolved
   * @returns Confirmation of resolution
   */
  async resolveThreat(
    threatId: string,
    resolution: string
  ): Promise<{ resolved: boolean; threat_id: string; resolved_at: string }> {
    return this.client.request('POST', `/api/v1/admin/security/threats/${threatId}/resolve`, {
      body: { resolution },
    });
  }
}
