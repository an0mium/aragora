/**
 * Security Namespace API
 *
 * Provides security status, health checks, and key rotation management.
 *
 * Features:
 * - Overall security status monitoring
 * - Security health checks
 * - Encryption key inventory
 * - Key rotation operations
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
 * - Security health checks
 * - Encryption key inventory
 * - Key rotation operations
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
 * // Inspect encryption keys
 * const keys = await client.security.listKeys();
 *
 * // Rotate the active key when needed
 * const rotation = await client.security.rotateKey({ reason: 'scheduled-maintenance' });
 * console.log(rotation.success);
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
}
