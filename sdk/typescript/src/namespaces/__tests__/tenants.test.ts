/**
 * Tenants Namespace Tests
 *
 * Comprehensive tests for the tenants namespace API including:
 * - Tenant CRUD operations
 * - Quota management
 * - Member management
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { TenantsAPI } from '../tenants';

interface MockClient {
  listTenants: Mock;
  getTenant: Mock;
  createTenant: Mock;
  updateTenant: Mock;
  deleteTenant: Mock;
  getTenantQuotas: Mock;
  updateTenantQuotas: Mock;
  listTenantMembers: Mock;
  addTenantMember: Mock;
  removeTenantMember: Mock;
}

describe('TenantsAPI Namespace', () => {
  let api: TenantsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      listTenants: vi.fn(),
      getTenant: vi.fn(),
      createTenant: vi.fn(),
      updateTenant: vi.fn(),
      deleteTenant: vi.fn(),
      getTenantQuotas: vi.fn(),
      updateTenantQuotas: vi.fn(),
      listTenantMembers: vi.fn(),
      addTenantMember: vi.fn(),
      removeTenantMember: vi.fn(),
    };
    api = new TenantsAPI(mockClient as any);
  });

  // ===========================================================================
  // Tenant CRUD
  // ===========================================================================

  describe('Tenant CRUD', () => {
    it('should list tenants', async () => {
      const mockTenants = {
        tenants: [
          { id: 't1', name: 'Acme Corp', plan: 'enterprise', status: 'active' },
          { id: 't2', name: 'TechStart', plan: 'starter', status: 'active' },
        ],
        total: 2,
      };
      mockClient.listTenants.mockResolvedValue(mockTenants);

      const result = await api.list();

      expect(mockClient.listTenants).toHaveBeenCalled();
      expect(result.tenants).toHaveLength(2);
    });

    it('should list tenants with pagination', async () => {
      const mockTenants = { tenants: [{ id: 't3' }], total: 10 };
      mockClient.listTenants.mockResolvedValue(mockTenants);

      await api.list({ limit: 10, offset: 20 });

      expect(mockClient.listTenants).toHaveBeenCalledWith({ limit: 10, offset: 20 });
    });

    it('should get tenant by ID', async () => {
      const mockTenant = {
        id: 't1',
        name: 'Acme Corp',
        plan: 'enterprise',
        status: 'active',
        created_at: '2024-01-01T00:00:00Z',
        settings: { theme: 'dark' },
      };
      mockClient.getTenant.mockResolvedValue(mockTenant);

      const result = await api.get('t1');

      expect(mockClient.getTenant).toHaveBeenCalledWith('t1');
      expect(result.name).toBe('Acme Corp');
    });

    it('should create tenant', async () => {
      const mockTenant = {
        id: 't_new',
        name: 'New Tenant',
        plan: 'pro',
        status: 'active',
      };
      mockClient.createTenant.mockResolvedValue(mockTenant);

      const result = await api.create({ name: 'New Tenant', plan: 'pro' });

      expect(mockClient.createTenant).toHaveBeenCalledWith({ name: 'New Tenant', plan: 'pro' });
      expect(result.id).toBe('t_new');
    });

    it('should update tenant', async () => {
      const mockTenant = { id: 't1', name: 'Updated Name', plan: 'enterprise' };
      mockClient.updateTenant.mockResolvedValue(mockTenant);

      const result = await api.update('t1', { name: 'Updated Name' });

      expect(mockClient.updateTenant).toHaveBeenCalledWith('t1', { name: 'Updated Name' });
      expect(result.name).toBe('Updated Name');
    });

    it('should delete tenant', async () => {
      mockClient.deleteTenant.mockResolvedValue(undefined);

      await api.delete('t1');

      expect(mockClient.deleteTenant).toHaveBeenCalledWith('t1');
    });
  });

  // ===========================================================================
  // Quota Management
  // ===========================================================================

  describe('Quota Management', () => {
    it('should get tenant quotas', async () => {
      const mockQuotas = {
        debates_per_month: { used: 150, limit: 500, percentage: 30 },
        agent_calls_per_month: { used: 3000, limit: 10000, percentage: 30 },
        storage_mb: { used: 250, limit: 5000, percentage: 5 },
        members: { used: 10, limit: 50, percentage: 20 },
      };
      mockClient.getTenantQuotas.mockResolvedValue(mockQuotas);

      const result = await api.getQuotas('t1');

      expect(mockClient.getTenantQuotas).toHaveBeenCalledWith('t1');
      expect(result.debates_per_month.used).toBe(150);
    });

    it('should update tenant quotas', async () => {
      const mockQuotas = {
        debates_per_month: { used: 150, limit: 1000, percentage: 15 },
        members: { used: 10, limit: 100, percentage: 10 },
      };
      mockClient.updateTenantQuotas.mockResolvedValue(mockQuotas);

      const result = await api.updateQuotas('t1', {
        debates_per_month: 1000,
        members: 100,
      });

      expect(mockClient.updateTenantQuotas).toHaveBeenCalledWith('t1', {
        debates_per_month: 1000,
        members: 100,
      });
      expect(result.debates_per_month.limit).toBe(1000);
    });
  });

  // ===========================================================================
  // Member Management
  // ===========================================================================

  describe('Member Management', () => {
    it('should list tenant members', async () => {
      const mockMembers = {
        members: [
          { id: 'u1', email: 'admin@acme.com', role: 'admin', joined_at: '2024-01-01' },
          { id: 'u2', email: 'user@acme.com', role: 'member', joined_at: '2024-01-02' },
        ],
        total: 2,
      };
      mockClient.listTenantMembers.mockResolvedValue(mockMembers);

      const result = await api.listMembers('t1');

      expect(mockClient.listTenantMembers).toHaveBeenCalledWith('t1', undefined);
      expect(result.members).toHaveLength(2);
    });

    it('should list members with pagination', async () => {
      const mockMembers = { members: [], total: 50 };
      mockClient.listTenantMembers.mockResolvedValue(mockMembers);

      await api.listMembers('t1', { limit: 10, offset: 20 });

      expect(mockClient.listTenantMembers).toHaveBeenCalledWith('t1', { limit: 10, offset: 20 });
    });

    it('should add tenant member', async () => {
      const mockMember = {
        id: 'u_new',
        email: 'newuser@acme.com',
        role: 'member',
        joined_at: '2024-01-20T10:00:00Z',
      };
      mockClient.addTenantMember.mockResolvedValue(mockMember);

      const result = await api.addMember('t1', { email: 'newuser@acme.com', role: 'member' });

      expect(mockClient.addTenantMember).toHaveBeenCalledWith('t1', {
        email: 'newuser@acme.com',
        role: 'member',
      });
      expect(result.email).toBe('newuser@acme.com');
    });

    it('should remove tenant member', async () => {
      mockClient.removeTenantMember.mockResolvedValue(undefined);

      await api.removeMember('t1', 'u2');

      expect(mockClient.removeTenantMember).toHaveBeenCalledWith('t1', 'u2');
    });
  });
});
