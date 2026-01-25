/**
 * Tenants Namespace API
 *
 * Provides a namespaced interface for multi-tenancy operations.
 * This wraps the flat client methods for a more intuitive API.
 */

import type {
  Tenant,
  TenantList,
  CreateTenantRequest,
  UpdateTenantRequest,
  QuotaStatus,
  QuotaUpdate,
  MemberList,
  TenantMember,
  AddMemberRequest,
  PaginationParams,
} from '../types';

/**
 * Interface for the internal client methods used by TenantsAPI.
 */
interface TenantsClientInterface {
  listTenants(params?: PaginationParams): Promise<TenantList>;
  getTenant(tenantId: string): Promise<Tenant>;
  createTenant(body: CreateTenantRequest): Promise<Tenant>;
  updateTenant(tenantId: string, body: UpdateTenantRequest): Promise<Tenant>;
  deleteTenant(tenantId: string): Promise<void>;
  getTenantQuotas(tenantId: string): Promise<QuotaStatus>;
  updateTenantQuotas(tenantId: string, body: QuotaUpdate): Promise<QuotaStatus>;
  listTenantMembers(tenantId: string, params?: PaginationParams): Promise<MemberList>;
  addTenantMember(tenantId: string, body: AddMemberRequest): Promise<TenantMember>;
  removeTenantMember(tenantId: string, userId: string): Promise<void>;
}

/**
 * Tenants API namespace.
 *
 * Provides methods for multi-tenancy management:
 * - Creating and managing tenants
 * - Quota management
 * - Member management
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Create a new tenant
 * const tenant = await client.tenants.create({
 *   name: 'Acme Corp',
 *   plan: 'enterprise',
 * });
 *
 * // Get tenant quotas
 * const quotas = await client.tenants.getQuotas(tenant.id);
 *
 * // Add a member
 * await client.tenants.addMember(tenant.id, {
 *   email: 'user@acme.com',
 *   role: 'admin',
 * });
 *
 * // List tenant members
 * const { members } = await client.tenants.listMembers(tenant.id);
 * ```
 */
export class TenantsAPI {
  constructor(private client: TenantsClientInterface) {}

  /**
   * List all tenants.
   */
  async list(params?: PaginationParams): Promise<TenantList> {
    return this.client.listTenants(params);
  }

  /**
   * Get a tenant by ID.
   */
  async get(tenantId: string): Promise<Tenant> {
    return this.client.getTenant(tenantId);
  }

  /**
   * Create a new tenant.
   */
  async create(body: CreateTenantRequest): Promise<Tenant> {
    return this.client.createTenant(body);
  }

  /**
   * Update a tenant.
   */
  async update(tenantId: string, body: UpdateTenantRequest): Promise<Tenant> {
    return this.client.updateTenant(tenantId, body);
  }

  /**
   * Delete a tenant.
   */
  async delete(tenantId: string): Promise<void> {
    return this.client.deleteTenant(tenantId);
  }

  /**
   * Get tenant quota status.
   */
  async getQuotas(tenantId: string): Promise<QuotaStatus> {
    return this.client.getTenantQuotas(tenantId);
  }

  /**
   * Update tenant quotas.
   */
  async updateQuotas(tenantId: string, body: QuotaUpdate): Promise<QuotaStatus> {
    return this.client.updateTenantQuotas(tenantId, body);
  }

  /**
   * List tenant members.
   */
  async listMembers(tenantId: string, params?: PaginationParams): Promise<MemberList> {
    return this.client.listTenantMembers(tenantId, params);
  }

  /**
   * Add a member to a tenant.
   */
  async addMember(tenantId: string, body: AddMemberRequest): Promise<TenantMember> {
    return this.client.addTenantMember(tenantId, body);
  }

  /**
   * Remove a member from a tenant.
   */
  async removeMember(tenantId: string, userId: string): Promise<void> {
    return this.client.removeTenantMember(tenantId, userId);
  }
}
