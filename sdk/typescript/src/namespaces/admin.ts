/**
 * Admin Namespace API
 *
 * Provides a namespaced interface for platform administration.
 * Requires admin role for all operations.
 */

import type { PaginationParams } from '../types';

// Admin-specific types
export interface Organization {
  id: string;
  name: string;
  created_at: string;
  status: 'active' | 'suspended' | 'pending';
  plan: string;
  user_count: number;
}

export interface OrganizationList {
  organizations: Organization[];
  total: number;
  limit: number;
  offset: number;
}

export interface AdminUser {
  id: string;
  email: string;
  name: string;
  org_id: string;
  role: string;
  created_at: string;
  last_login?: string;
  status: 'active' | 'suspended' | 'pending';
}

export interface AdminUserList {
  users: AdminUser[];
  total: number;
  limit: number;
  offset: number;
}

export interface PlatformStats {
  total_organizations: number;
  total_users: number;
  active_debates: number;
  debates_today: number;
  debates_this_week: number;
  total_debates: number;
  agent_calls_today: number;
  consensus_rate: number;
}

export interface RevenueData {
  mrr: number;
  arr: number;
  revenue_this_month: number;
  revenue_last_month: number;
  growth_rate: number;
  churn_rate: number;
  active_subscriptions: number;
  trial_conversions: number;
}

export interface NomicStatus {
  running: boolean;
  current_phase: string | null;
  current_cycle: number;
  total_cycles: number;
  last_run: string | null;
  next_scheduled: string | null;
  health: 'healthy' | 'degraded' | 'unhealthy';
}

export interface SecurityStatus {
  encryption_enabled: boolean;
  mfa_enforcement: 'none' | 'optional' | 'required';
  audit_logging: boolean;
  key_rotation_due: boolean;
  last_security_scan: string | null;
  vulnerabilities_found: number;
}

export interface SecurityKey {
  id: string;
  type: string;
  created_at: string;
  expires_at?: string;
  last_rotated?: string;
  status: 'active' | 'rotating' | 'expired';
}

/**
 * Interface for the internal client methods used by AdminAPI.
 */
interface AdminClientInterface {
  listOrganizations(params?: PaginationParams): Promise<OrganizationList>;
  listAdminUsers(params?: PaginationParams): Promise<AdminUserList>;
  getAdminStats(): Promise<PlatformStats>;
  getRevenue(): Promise<RevenueData>;
  getAdminNomicStatus(): Promise<NomicStatus>;
  resetNomic(): Promise<{ success: boolean }>;
  pauseNomic(): Promise<{ success: boolean }>;
  resumeNomic(): Promise<{ success: boolean }>;
  // Security endpoints
  getAdminSecurityStatus(): Promise<SecurityStatus>;
  getAdminSecurityHealth(): Promise<{ healthy: boolean; checks: Record<string, boolean> }>;
  listSecurityKeys(): Promise<{ keys: SecurityKey[] }>;
}

/**
 * Admin API namespace.
 *
 * Provides methods for platform administration:
 * - Organization and user listing
 * - Platform statistics
 * - Revenue analytics
 * - Nomic loop control
 * - Security operations
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'admin-key' });
 *
 * // View platform stats
 * const stats = await client.admin.getStats();
 * console.log(`${stats.total_organizations} organizations, ${stats.active_debates} active debates`);
 *
 * // Control Nomic loop
 * await client.admin.pauseNomic();
 * const status = await client.admin.getNomicStatus();
 * console.log(`Nomic running: ${status.running}`);
 * ```
 */
export class AdminAPI {
  constructor(private client: AdminClientInterface) {}

  // ===========================================================================
  // Organizations and Users
  // ===========================================================================

  /**
   * List all organizations with pagination.
   */
  async listOrganizations(params?: PaginationParams): Promise<OrganizationList> {
    return this.client.listOrganizations(params);
  }

  /**
   * List all users with pagination.
   */
  async listUsers(params?: PaginationParams): Promise<AdminUserList> {
    return this.client.listAdminUsers(params);
  }

  // ===========================================================================
  // Platform Statistics
  // ===========================================================================

  /**
   * Get platform-wide statistics.
   */
  async getStats(): Promise<PlatformStats> {
    return this.client.getAdminStats();
  }

  /**
   * Get revenue analytics.
   */
  async getRevenue(): Promise<RevenueData> {
    return this.client.getRevenue();
  }

  // ===========================================================================
  // Nomic Loop Control
  // ===========================================================================

  /**
   * Get the current Nomic loop status.
   */
  async getNomicStatus(): Promise<NomicStatus> {
    return this.client.getAdminNomicStatus();
  }

  /**
   * Reset the Nomic loop to initial state.
   */
  async resetNomic(): Promise<{ success: boolean }> {
    return this.client.resetNomic();
  }

  /**
   * Pause the Nomic loop.
   */
  async pauseNomic(): Promise<{ success: boolean }> {
    return this.client.pauseNomic();
  }

  /**
   * Resume a paused Nomic loop.
   */
  async resumeNomic(): Promise<{ success: boolean }> {
    return this.client.resumeNomic();
  }

  // ===========================================================================
  // Security Operations
  // ===========================================================================

  /**
   * Get security status overview.
   */
  async getSecurityStatus(): Promise<SecurityStatus> {
    return this.client.getAdminSecurityStatus();
  }

  /**
   * Get security health check results.
   */
  async getSecurityHealth(): Promise<{ healthy: boolean; checks: Record<string, boolean> }> {
    return this.client.getAdminSecurityHealth();
  }

  /**
   * List all security keys.
   */
  async listSecurityKeys(): Promise<{ keys: SecurityKey[] }> {
    return this.client.listSecurityKeys();
  }
}

// Re-export types for convenience
export type {
  PaginationParams,

  /**
   * Get handler diagnostics.
   */
  async getHandlerDiagnostics(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/diagnostics/handlers', { params }) as Promise<Record<string, unknown>>;
  }
};
