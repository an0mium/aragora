/**
 * Tenants Namespace API
 *
 * Provides a namespaced interface for multi-tenancy operations.
 */

interface TenantsClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
  listTenants(params?: { limit?: number; offset?: number }): Promise<any>;
  getTenant(tenantId: string): Promise<any>;
  createTenant(body: CreateTenantRequest): Promise<any>;
  updateTenant(tenantId: string, body: UpdateTenantRequest): Promise<any>;
  deleteTenant(tenantId: string): Promise<void>;
  getTenantQuotas(tenantId: string): Promise<any>;
  updateTenantQuotas(tenantId: string, body: QuotaUpdate): Promise<any>;
  listTenantMembers(
    tenantId: string,
    params?: { limit?: number; offset?: number }
  ): Promise<any>;
  addTenantMember(tenantId: string, body: { email: string; role?: string }): Promise<any>;
  removeTenantMember(tenantId: string, userId: string): Promise<void>;
}

/**
 * Tenant object.
 */
export interface Tenant {
  id: string;
  name: string;
  plan?: string;
  status: 'active' | 'suspended';
  created_at: string;
  updated_at?: string;
}

/**
 * Create tenant request.
 */
export interface CreateTenantRequest {
  name: string;
  plan?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Update tenant request.
 */
export interface UpdateTenantRequest {
  name?: string;
  plan?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Tenant member.
 */
export interface TenantMember {
  user_id: string;
  email: string;
  role: string;
  joined_at: string;
}

/**
 * Quota status.
 */
export interface QuotaStatus {
  tenant_id: string;
  quotas: Record<string, { used: number; limit: number }>;
}

/**
 * Quota update request.
 */
export interface QuotaUpdate {
  debates_per_month?: number;
  agents_per_debate?: number;
  storage_bytes?: number;
  api_calls_per_minute?: number;
  members?: number;
  overage_allowed?: boolean;
}

/**
 * Tenants API namespace.
 *
 * Provides methods for multi-tenancy management:
 * - Creating and managing tenants
 * - Quota management
 * - Member management
 */
export class TenantsAPI {
  constructor(private client: TenantsClientInterface) {}

  /**
   * List all tenants.
   * @route GET /api/v1/tenants
   */
  async list(params?: { limit?: number; offset?: number }): Promise<{ tenants: Tenant[] }> {
    return this.client.listTenants(params);
  }

  /**
   * Create a new tenant.
   * @route POST /api/v1/tenants
   */
  async create(body: CreateTenantRequest): Promise<Tenant> {
    return this.client.createTenant(body);
  }

  /**
   * Get a tenant by ID.
   * @route GET /api/v1/tenants/{tenant_id}
   */
  async get(tenantId: string): Promise<Tenant> {
    if ('getTenant' in this.client && typeof this.client.getTenant === 'function') {
      return this.client.getTenant(tenantId);
    }
    return this.client.request(
      'GET',
      `/api/v1/tenants/${encodeURIComponent(tenantId)}`
    ) as Promise<Tenant>;
  }

  /**
   * Update a tenant.
   * @route PATCH /api/v1/tenants/{tenant_id}
   */
  async update(tenantId: string, body: UpdateTenantRequest): Promise<Tenant> {
    return this.client.updateTenant(tenantId, body);
  }

  /**
   * Delete a tenant.
   * @route DELETE /api/v1/tenants/{tenant_id}
   */
  async delete(tenantId: string): Promise<void> {
    return this.client.deleteTenant(tenantId);
  }

  /**
   * Get tenant quota status.
   * @route GET /api/v1/tenants/{tenant_id}/quotas
   */
  async getQuotas(tenantId: string): Promise<QuotaStatus> {
    if ('getTenantQuotas' in this.client && typeof this.client.getTenantQuotas === 'function') {
      return this.client.getTenantQuotas(tenantId);
    }
    return this.client.request(
      'GET',
      `/api/v1/tenants/${encodeURIComponent(tenantId)}/quotas`
    ) as Promise<QuotaStatus>;
  }

  /**
   * Update tenant quotas.
   * @route PUT /api/v1/tenants/{tenant_id}/quotas
   */
  async updateQuotas(tenantId: string, body: QuotaUpdate): Promise<QuotaStatus> {
    return this.client.updateTenantQuotas(tenantId, body);
  }

  /**
   * List tenant members.
   * @route GET /api/v1/tenants/{tenant_id}/members
   */
  async listMembers(tenantId: string, params?: { limit?: number; offset?: number }): Promise<{ members: TenantMember[] }> {
    if ('listTenantMembers' in this.client && typeof this.client.listTenantMembers === 'function') {
      return this.client.listTenantMembers(tenantId, params);
    }
    return this.client.request(
      'GET',
      `/api/v1/tenants/${encodeURIComponent(tenantId)}/members`,
      { params: params as Record<string, unknown> | undefined }
    ) as Promise<{ members: TenantMember[] }>;
  }

  /**
   * Invite a member to a tenant.
   * @route POST /api/v1/tenants/{tenant_id}/members/invite
   */
  async inviteMember(tenantId: string, body: { email: string; role?: string }): Promise<TenantMember> {
    if ('addTenantMember' in this.client && typeof this.client.addTenantMember === 'function') {
      return this.client.addTenantMember(tenantId, body);
    }
    return this.client.request(
      'POST',
      `/api/v1/tenants/${encodeURIComponent(tenantId)}/members/invite`,
      { body }
    ) as Promise<TenantMember>;
  }

  /**
   * Add a member to a tenant.
   * Compatibility alias for the flat client method.
   */
  async addMember(tenantId: string, body: { email: string; role?: string }): Promise<TenantMember> {
    return this.client.addTenantMember(tenantId, body);
  }

  /**
   * Remove a member from a tenant.
   * Compatibility alias for the flat client method.
   */
  async removeMember(tenantId: string, userId: string): Promise<void> {
    return this.client.removeTenantMember(tenantId, userId);
  }

  /**
   * Get tenant usage.
   * @route GET /api/v1/tenants/{tenant_id}/usage
   */
  async getUsage(tenantId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/tenants/${encodeURIComponent(tenantId)}/usage`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Suspend a tenant.
   * @route POST /api/v1/tenants/{tenant_id}/suspend
   */
  async suspend(tenantId: string): Promise<Tenant> {
    return this.client.request(
      'POST',
      `/api/v1/tenants/${encodeURIComponent(tenantId)}/suspend`
    ) as Promise<Tenant>;
  }

  /**
   * Reactivate a suspended tenant.
   * @route POST /api/v1/tenants/{tenant_id}/reactivate
   */
  async reactivate(tenantId: string): Promise<Tenant> {
    return this.client.request(
      'POST',
      `/api/v1/tenants/${encodeURIComponent(tenantId)}/reactivate`
    ) as Promise<Tenant>;
  }
}
