/**
 * Tenants Namespace API
 *
 * Provides a namespaced interface for multi-tenancy operations.
 */

import type { AragoraClient } from '../client';

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
  quotas: Record<string, { limit: number }>;
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
  constructor(private client: AragoraClient) {}

  /**
   * List all tenants.
   * @route GET /api/v1/tenants
   */
  async list(params?: { limit?: number; offset?: number }): Promise<{ tenants: Tenant[] }> {
    return this.client.request('GET', '/api/v1/tenants', {
      params: params as Record<string, unknown>,
    }) as Promise<{ tenants: Tenant[] }>;
  }

  /**
   * Create a new tenant.
   * @route POST /api/v1/tenants
   */
  async create(body: CreateTenantRequest): Promise<Tenant> {
    return this.client.request('POST', '/api/v1/tenants', {
      body,
    }) as Promise<Tenant>;
  }

  /**
   * Get a tenant by ID.
   * @route GET /api/v1/tenants/{tenant_id}
   */
  async get(tenantId: string): Promise<Tenant> {
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
    return this.client.request(
      'PATCH',
      `/api/v1/tenants/${encodeURIComponent(tenantId)}`,
      { body }
    ) as Promise<Tenant>;
  }

  /**
   * Delete a tenant.
   * @route DELETE /api/v1/tenants/{tenant_id}
   */
  async delete(tenantId: string): Promise<void> {
    return this.client.request(
      'DELETE',
      `/api/v1/tenants/${encodeURIComponent(tenantId)}`
    ) as Promise<void>;
  }

  /**
   * Get tenant quota status.
   * @route GET /api/v1/tenants/{tenant_id}/quotas
   */
  async getQuotas(tenantId: string): Promise<QuotaStatus> {
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
    return this.client.request(
      'PUT',
      `/api/v1/tenants/${encodeURIComponent(tenantId)}/quotas`,
      { body }
    ) as Promise<QuotaStatus>;
  }

  /**
   * List tenant members.
   * @route GET /api/v1/tenants/{tenant_id}/members
   */
  async listMembers(tenantId: string, params?: { limit?: number; offset?: number }): Promise<{ members: TenantMember[] }> {
    return this.client.request(
      'GET',
      `/api/v1/tenants/${encodeURIComponent(tenantId)}/members`,
      { params: params as Record<string, unknown> }
    ) as Promise<{ members: TenantMember[] }>;
  }

  /**
   * Invite a member to a tenant.
   * @route POST /api/v1/tenants/{tenant_id}/members/invite
   */
  async inviteMember(tenantId: string, body: { email: string; role?: string }): Promise<TenantMember> {
    return this.client.request(
      'POST',
      `/api/v1/tenants/${encodeURIComponent(tenantId)}/members/invite`,
      { body }
    ) as Promise<TenantMember>;
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
