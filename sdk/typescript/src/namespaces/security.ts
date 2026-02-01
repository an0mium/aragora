/**
 * Security Namespace API
 *
 * Provides security status, health checks, and key management.
 */

export interface SecurityStatus {
  overall: 'healthy' | 'degraded' | 'critical';
  encryption_enabled: boolean;
  audit_logging_enabled: boolean;
  mfa_enabled: boolean;
  last_security_scan?: string;
  active_threats: number;
  metadata?: Record<string, unknown>;
}

export interface SecurityHealthCheck {
  component: string;
  status: 'ok' | 'warning' | 'error';
  message?: string;
  last_checked: string;
}

export interface SecurityKey {
  id: string;
  name: string;
  algorithm: string;
  created_at: string;
  expires_at?: string;
  status: 'active' | 'expired' | 'revoked';
}

export interface RotateKeyRequest {
  key_id?: string;
  algorithm?: string;
  reason?: string;
}

export interface RotateKeyResult {
  success: boolean;
  new_key_id: string;
  old_key_id: string;
  rotated_at: string;
}

interface SecurityClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { body?: unknown }
  ): Promise<T>;
}

export class SecurityAPI {
  constructor(private client: SecurityClientInterface) {}

  /**
   * Get overall security status.
   */
  async getStatus(): Promise<SecurityStatus> {
    return this.client.request('GET', '/api/v1/admin/security/status');
  }

  /**
   * Get security health checks.
   */
  async getHealthChecks(): Promise<{ checks: SecurityHealthCheck[] }> {
    return this.client.request('GET', '/api/v1/admin/security/health');
  }

  /**
   * List security keys.
   */
  async listKeys(): Promise<{ keys: SecurityKey[] }> {
    return this.client.request('GET', '/api/v1/admin/security/keys');
  }

  /**
   * Rotate an encryption key.
   */
  async rotateKey(body?: RotateKeyRequest): Promise<RotateKeyResult> {
    return this.client.request('POST', '/api/v1/admin/security/rotate-key', { body });
  }
}
