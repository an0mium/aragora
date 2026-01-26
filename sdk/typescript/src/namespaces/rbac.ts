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
  listRoles(params?: PaginationParams): Promise<{ roles: Role[] }>;
  getRole(roleId: string): Promise<Role>;
  createRole(request: CreateRoleRequest): Promise<Role>;
  updateRole(roleId: string, updates: UpdateRoleRequest): Promise<Role>;
  deleteRole(roleId: string): Promise<{ deleted: boolean }>;
  listPermissions(params?: PaginationParams): Promise<{ permissions: Permission[] }>;
  assignRole(userId: string, roleId: string): Promise<void>;
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
  async assignRole(userId: string, roleId: string): Promise<void> {
    return this.client.assignRole(userId, roleId);
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
}
