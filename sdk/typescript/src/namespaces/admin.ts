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

export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  active_connections: number;
  request_rate: number;
  error_rate: number;
  avg_latency_ms: number;
  uptime_seconds: number;
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

export interface ImpersonationToken {
  token: string;
  expires_at: string;
  user_id: string;
  user_email: string;
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

export interface CircuitBreaker {
  name: string;
  state: 'closed' | 'open' | 'half_open';
  failure_count: number;
  success_count: number;
  last_failure: string | null;
  last_success: string | null;
  threshold: number;
  timeout_seconds: number;
}

export interface CircuitBreakerList {
  circuit_breakers: CircuitBreaker[];
}

export interface CreditAccount {
  org_id: string;
  balance: number;
  lifetime_issued: number;
  lifetime_used: number;
  expires_at?: string;
}

export interface CreditTransaction {
  id: string;
  org_id: string;
  amount: number;
  type: 'issue' | 'use' | 'adjust' | 'expire';
  reason: string;
  created_at: string;
  created_by?: string;
}

export interface CreditTransactionList {
  transactions: CreditTransaction[];
  total: number;
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

export interface OrganizationUpdate {
  name?: string;
  plan?: string;
  status?: 'active' | 'suspended' | 'pending';
  settings?: Record<string, unknown>;
}

export interface UserAction {
  success: boolean;
  user_id: string;
  status: string;
  message?: string;
}

/**
 * Interface for the internal client methods used by AdminAPI.
 */
interface AdminClientInterface {
  listOrganizations(params?: PaginationParams): Promise<OrganizationList>;
  getAdminOrganization(orgId: string): Promise<Organization>;
  updateAdminOrganization(orgId: string, data: OrganizationUpdate): Promise<Organization>;
  listAdminUsers(params?: PaginationParams): Promise<AdminUserList>;
  getAdminUser(userId: string): Promise<AdminUser>;
  suspendAdminUser(userId: string, reason?: string): Promise<UserAction>;
  activateAdminUser(userId: string): Promise<UserAction>;
  getAdminStats(): Promise<PlatformStats>;
  getAdminSystemMetrics(): Promise<SystemMetrics>;
  getRevenue(): Promise<RevenueData>;
  impersonateUser(userId: string): Promise<ImpersonationToken>;
  getAdminNomicStatus(): Promise<NomicStatus>;
  getAdminCircuitBreakers(): Promise<CircuitBreakerList>;
  resetNomic(): Promise<{ success: boolean }>;
  pauseNomic(): Promise<{ success: boolean }>;
  resumeNomic(): Promise<{ success: boolean }>;
  resetAdminCircuitBreakers(): Promise<{ success: boolean; reset_count: number }>;
  // Credits endpoints
  issueCredits(orgId: string, body: { amount: number; reason: string; expires_at?: string }): Promise<CreditAccount>;
  getCreditAccount(orgId: string): Promise<CreditAccount>;
  listCreditTransactions(orgId: string, params?: PaginationParams): Promise<CreditTransactionList>;
  adjustCreditBalance(orgId: string, body: { amount: number; reason: string }): Promise<CreditAccount>;
  getExpiringCredits(orgId: string): Promise<{ credits: Array<{ amount: number; expires_at: string }> }>;
  // Security endpoints
  getAdminSecurityStatus(): Promise<SecurityStatus>;
  rotateSecurityKey(keyType: string): Promise<{ success: boolean; new_key_id: string }>;
  getAdminSecurityHealth(): Promise<{ healthy: boolean; checks: Record<string, boolean> }>;
  listSecurityKeys(): Promise<{ keys: SecurityKey[] }>;
}

/**
 * Admin API namespace.
 *
 * Provides methods for platform administration:
 * - Organization and user management
 * - Platform statistics and system metrics
 * - Revenue analytics
 * - Nomic loop control
 * - Credit management
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
 * // Check system health
 * const metrics = await client.admin.getSystemMetrics();
 * console.log(`CPU: ${metrics.cpu_usage}%, Memory: ${metrics.memory_usage}%`);
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

  /**
   * Get a specific organization by ID.
   *
   * @param orgId - Organization ID
   */
  async getOrganization(orgId: string): Promise<Organization> {
    return this.client.getAdminOrganization(orgId);
  }

  /**
   * Update an organization.
   *
   * @param orgId - Organization ID
   * @param data - Fields to update
   */
  async updateOrganization(orgId: string, data: OrganizationUpdate): Promise<Organization> {
    return this.client.updateAdminOrganization(orgId, data);
  }

  /**
   * Get a specific user by ID.
   *
   * @param userId - User ID
   */
  async getUser(userId: string): Promise<AdminUser> {
    return this.client.getAdminUser(userId);
  }

  /**
   * Suspend a user account.
   *
   * @param userId - User ID to suspend
   * @param reason - Optional reason for suspension
   */
  async suspendUser(userId: string, reason?: string): Promise<UserAction> {
    return this.client.suspendAdminUser(userId, reason);
  }

  /**
   * Activate a suspended user account.
   *
   * @param userId - User ID to activate
   */
  async activateUser(userId: string): Promise<UserAction> {
    return this.client.activateAdminUser(userId);
  }

  /**
   * Impersonate a user for support/debugging purposes.
   * Returns a temporary token that can be used to act as the user.
   *
   * @param userId - The user ID to impersonate
   */
  async impersonateUser(userId: string): Promise<ImpersonationToken> {
    return this.client.impersonateUser(userId);
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
   * Get real-time system metrics.
   */
  async getSystemMetrics(): Promise<SystemMetrics> {
    return this.client.getAdminSystemMetrics();
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
   * Get circuit breaker states.
   */
  async getCircuitBreakers(): Promise<CircuitBreakerList> {
    return this.client.getAdminCircuitBreakers();
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

  /**
   * Reset all circuit breakers to closed state.
   */
  async resetCircuitBreakers(): Promise<{ success: boolean; reset_count: number }> {
    return this.client.resetAdminCircuitBreakers();
  }

  // ===========================================================================
  // Credit Management
  // ===========================================================================

  /**
   * Issue credits to an organization.
   *
   * @param orgId - Organization ID
   * @param amount - Number of credits to issue
   * @param reason - Reason for issuing credits
   * @param expiresAt - Optional expiration date
   */
  async issueCredits(
    orgId: string,
    amount: number,
    reason: string,
    expiresAt?: string
  ): Promise<CreditAccount> {
    return this.client.issueCredits(orgId, { amount, reason, expires_at: expiresAt });
  }

  /**
   * Get credit account details for an organization.
   *
   * @param orgId - Organization ID
   */
  async getCreditAccount(orgId: string): Promise<CreditAccount> {
    return this.client.getCreditAccount(orgId);
  }

  /**
   * List credit transactions for an organization.
   *
   * @param orgId - Organization ID
   * @param params - Pagination parameters
   */
  async listCreditTransactions(
    orgId: string,
    params?: PaginationParams
  ): Promise<CreditTransactionList> {
    return this.client.listCreditTransactions(orgId, params);
  }

  /**
   * Adjust credit balance for an organization.
   *
   * @param orgId - Organization ID
   * @param amount - Amount to adjust (positive or negative)
   * @param reason - Reason for adjustment
   */
  async adjustCredits(orgId: string, amount: number, reason: string): Promise<CreditAccount> {
    return this.client.adjustCreditBalance(orgId, { amount, reason });
  }

  /**
   * Get credits that are expiring soon for an organization.
   *
   * @param orgId - Organization ID
   */
  async getExpiringCredits(
    orgId: string
  ): Promise<{ credits: Array<{ amount: number; expires_at: string }> }> {
    return this.client.getExpiringCredits(orgId);
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
   * Rotate a security key.
   *
   * @param keyType - Type of key to rotate (e.g., 'api', 'encryption', 'signing')
   */
  async rotateSecurityKey(keyType: string): Promise<{ success: boolean; new_key_id: string }> {
    return this.client.rotateSecurityKey(keyType);
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
};
