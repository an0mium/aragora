/**
 * RBAC Namespace API
 *
 * Provides a namespaced interface for role-based access control operations.
 * This wraps the flat client methods for a more intuitive API.
 */

import type {
  Role,
  Permission,
  RoleAssignment,
  AssignmentList,
  BulkAssignRequest,
  BulkAssignResponse,
  PaginationParams,
} from '../types';

/**
 * Request for creating a new role.
 */
export interface CreateRoleRequest {
  /** Role name */
  name: string;
  /** Role description */
  description?: string;
  /** Permission IDs to include */
  permissions: string[];
  /** Parent role ID for inheritance */
  parent_role_id?: string;
}

/**
 * Request for updating a role.
 */
export interface UpdateRoleRequest {
  /** Updated name */
  name?: string;
  /** Updated description */
  description?: string;
  /** Updated permissions */
  permissions?: string[];
}

/**
 * Interface for the internal client methods used by RBACAPI.
 */
interface RBACClientInterface {
  // Generic request method for extended endpoints
  request<T = unknown>(method: string, path: string, options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }): Promise<T>;

  // Core RBAC methods
  listRoles(params?: PaginationParams): Promise<{ roles: Role[] }>;
  getRole(roleId: string): Promise<Role>;
  createRole(request: CreateRoleRequest): Promise<Role>;
  updateRole(roleId: string, updates: UpdateRoleRequest): Promise<Role>;
  deleteRole(roleId: string): Promise<{ deleted: boolean }>;
  listPermissions(params?: PaginationParams): Promise<{ permissions: Permission[] }>;
  assignRole(userId: string, roleId: string): Promise<{ assigned: boolean }>;
  revokeRole(userId: string, roleId: string): Promise<void>;
  getUserRoles(userId: string): Promise<{ roles: Role[] }>;
  checkPermission(userId: string, permission: string): Promise<{ allowed: boolean }>;
  listRoleAssignments(roleId: string, params?: PaginationParams): Promise<AssignmentList>;
  bulkAssignRoles(body: BulkAssignRequest): Promise<BulkAssignResponse>;
}

/**
 * RBAC API namespace.
 *
 * Provides methods for role-based access control:
 * - Managing roles and permissions
 * - Assigning and revoking roles
 * - Checking user permissions
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // List all roles
 * const { roles } = await client.rbac.listRoles();
 *
 * // Create a new role
 * const role = await client.rbac.createRole({
 *   name: 'Analyst',
 *   permissions: ['debates:read', 'analytics:read'],
 * });
 *
 * // Assign role to user
 * await client.rbac.assignRole(userId, role.id);
 *
 * // Check permission
 * const { allowed } = await client.rbac.checkPermission(userId, 'debates:create');
 * ```
 */
export class RBACAPI {
  constructor(private client: RBACClientInterface) {}

  /**
   * List all roles with optional pagination.
   */
  async listRoles(params?: PaginationParams): Promise<{ roles: Role[] }> {
    return this.client.listRoles(params);
  }

  /**
   * Get a role by ID.
   */
  async getRole(roleId: string): Promise<Role> {
    return this.client.getRole(roleId);
  }

  /**
   * Create a new role.
   */
  async createRole(request: CreateRoleRequest): Promise<Role> {
    return this.client.createRole(request);
  }

  /**
   * Update an existing role.
   */
  async updateRole(roleId: string, updates: UpdateRoleRequest): Promise<Role> {
    return this.client.updateRole(roleId, updates);
  }

  /**
   * Delete a role.
   */
  async deleteRole(roleId: string): Promise<{ deleted: boolean }> {
    return this.client.deleteRole(roleId);
  }

  /**
   * List all permissions with optional pagination.
   */
  async listPermissions(params?: PaginationParams): Promise<{ permissions: Permission[] }> {
    return this.client.listPermissions(params);
  }

  /**
   * Assign a role to a user.
   */
  async assignRole(userId: string, roleId: string): Promise<{ assigned: boolean }> {
    await this.client.assignRole(userId, roleId);
    return { assigned: true };
  }

  /**
   * Revoke a role from a user.
   */
  async revokeRole(userId: string, roleId: string): Promise<void> {
    return this.client.revokeRole(userId, roleId);
  }

  /**
   * Get all roles assigned to a user.
   */
  async getUserRoles(userId: string): Promise<{ roles: Role[] }> {
    return this.client.getUserRoles(userId);
  }

  /**
   * Check if a user has a specific permission.
   */
  async checkPermission(userId: string, permission: string): Promise<{ allowed: boolean }> {
    return this.client.checkPermission(userId, permission);
  }

  /**
   * List users assigned to a role.
   */
  async listAssignments(
    roleId: string,
    params?: PaginationParams
  ): Promise<AssignmentList> {
    return this.client.listRoleAssignments(roleId, params);
  }

  /**
   * Bulk assign roles to multiple users.
   */
  async bulkAssign(body: BulkAssignRequest): Promise<BulkAssignResponse> {
    return this.client.bulkAssignRoles(body);
  }

  // =========================================================================
  // User Management
  // =========================================================================

  /**
   * List users in organization.
   */
  async listUsers(params?: PaginationParams): Promise<{ users: unknown[]; total: number }> {
    return this.client.request('GET', '/api/users', { params });
  }

  /**
   * Invite a new user to the organization.
   */
  async inviteUser(email: string, role?: string): Promise<{ invitation_id: string; email: string }> {
    return this.client.request('POST', '/api/users/invite', { json: { email, role } });
  }

  /**
   * Remove a user from the organization.
   */
  async removeUser(userId: string): Promise<{ removed: boolean }> {
    return this.client.request('DELETE', `/api/users/${userId}`);
  }

  /**
   * Change a user's role in the organization.
   */
  async changeUserRole(userId: string, role: string): Promise<{ updated: boolean }> {
    return this.client.request('PUT', `/api/users/${userId}/role`, { json: { role } });
  }

  // =========================================================================
  // Workspace Roles
  // =========================================================================

  /**
   * Get available roles for a workspace based on RBAC profile.
   */
  async getWorkspaceRoles(workspaceId: string): Promise<{ roles: unknown[]; profile: string }> {
    return this.client.request('GET', `/api/v1/workspaces/${workspaceId}/roles`);
  }

  /**
   * Update a member's role in a workspace.
   */
  async updateMemberRole(workspaceId: string, userId: string, role: string): Promise<{ updated: boolean }> {
    return this.client.request('PUT', `/api/v1/workspaces/${workspaceId}/members/${userId}/role`, { json: { role } });
  }

  /**
   * Add a member to a workspace.
   */
  async addWorkspaceMember(workspaceId: string, userId: string, role?: string): Promise<{ added: boolean }> {
    return this.client.request('POST', `/api/v1/workspaces/${workspaceId}/members`, { json: { user_id: userId, role } });
  }

  /**
   * Remove a member from a workspace.
   */
  async removeWorkspaceMember(workspaceId: string, userId: string): Promise<{ removed: boolean }> {
    return this.client.request('DELETE', `/api/v1/workspaces/${workspaceId}/members/${userId}`);
  }

  /**
   * List available RBAC profiles.
   */
  async listProfiles(): Promise<{ profiles: unknown[] }> {
    return this.client.request('GET', '/api/v1/workspaces/profiles');
  }

  // =========================================================================
  // Audit
  // =========================================================================

  /**
   * Query privacy audit entries.
   */
  async queryAudit(options?: { action?: string; user_id?: string; since?: string; limit?: number }): Promise<{ entries: unknown[]; total: number }> {
    return this.client.request('GET', '/api/v1/audit/entries', { params: options });
  }

  /**
   * Generate compliance audit report.
   */
  async getAuditReport(options?: { framework?: string; since?: string }): Promise<unknown> {
    return this.client.request('GET', '/api/v1/audit/report', { params: options });
  }

  /**
   * Verify audit log integrity.
   */
  async verifyAuditIntegrity(): Promise<{ valid: boolean; issues: unknown[] }> {
    return this.client.request('GET', '/api/v1/audit/verify');
  }

  /**
   * Get user activity history.
   */
  async getUserActivityHistory(userId: string, options?: PaginationParams): Promise<{ activities: unknown[]; total: number }> {
    return this.client.request('GET', `/api/v1/audit/actor/${userId}/history`, { params: options });
  }

  /**
   * Get resource access history.
   */
  async getResourceHistory(resourceType: string, resourceId: string, options?: PaginationParams): Promise<{ accesses: unknown[]; total: number }> {
    return this.client.request('GET', `/api/v1/audit/resource/${resourceType}/${resourceId}/history`, { params: options });
  }

  /**
   * Get denied access attempts.
   */
  async getDeniedAccess(options?: PaginationParams): Promise<{ denied: unknown[]; total: number }> {
    return this.client.request('GET', '/api/v1/audit/denied', { params: options });
  }

  // =========================================================================
  // API Keys
  // =========================================================================

  /**
   * Generate a new API key.
   */
  async generateApiKey(name: string, permissions?: string[], expires_at?: string): Promise<{ key: string; key_id: string }> {
    return this.client.request('POST', '/api/auth/api-key', { json: { name, permissions, expires_at } });
  }

  /**
   * List API keys.
   */
  async listApiKeys(): Promise<{ keys: unknown[] }> {
    return this.client.request('GET', '/api/keys');
  }

  /**
   * Revoke an API key.
   */
  async revokeApiKey(keyId: string): Promise<{ revoked: boolean }> {
    return this.client.request('DELETE', `/api/keys/${keyId}`);
  }

  // =========================================================================
  // Sessions
  // =========================================================================

  /**
   * List active sessions for the current user.
   */
  async listSessions(): Promise<{ sessions: unknown[] }> {
    return this.client.request('GET', '/api/auth/sessions');
  }

  /**
   * Revoke a specific session.
   */
  async revokeSession(sessionId: string): Promise<{ revoked: boolean }> {
    return this.client.request('DELETE', `/api/auth/sessions/${sessionId}`);
  }

  /**
   * Logout from all devices.
   */
  async logoutAll(): Promise<{ logged_out: boolean }> {
    return this.client.request('POST', '/api/auth/logout-all');
  }

  // =========================================================================
  // MFA
  // =========================================================================

  /**
   * Setup MFA - generate secret and QR code.
   */
  async setupMfa(): Promise<{ secret: string; qr_code: string }> {
    return this.client.request('POST', '/api/auth/mfa/setup');
  }

  /**
   * Enable MFA by verifying setup code.
   */
  async enableMfa(code: string): Promise<{ enabled: boolean; backup_codes: string[] }> {
    return this.client.request('POST', '/api/auth/mfa/enable', { json: { code } });
  }

  /**
   * Disable MFA.
   */
  async disableMfa(code: string): Promise<{ disabled: boolean }> {
    return this.client.request('POST', '/api/auth/mfa/disable', { json: { code } });
  }

  /**
   * Verify MFA code during login.
   */
  async verifyMfa(code: string): Promise<{ verified: boolean; token?: string }> {
    return this.client.request('POST', '/api/auth/mfa/verify', { json: { code } });
  }

  /**
   * Regenerate MFA backup codes.
   */
  async regenerateBackupCodes(code: string): Promise<{ backup_codes: string[] }> {
    return this.client.request('POST', '/api/auth/mfa/backup-codes', { json: { code } });
  }
}
