/**
 * Organizations Namespace Tests
 *
 * Comprehensive tests for the organizations namespace API including:
 * - Organization management
 * - Member management
 * - Invitations
 * - User organization operations
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { OrganizationsAPI } from '../organizations';

interface MockClient {
  getOrganization: Mock;
  updateOrganization: Mock;
  listOrganizationMembers: Mock;
  inviteToOrganization: Mock;
  listOrganizationInvitations: Mock;
  revokeOrganizationInvitation: Mock;
  removeOrganizationMember: Mock;
  updateOrganizationMemberRole: Mock;
  listUserOrganizations: Mock;
  switchOrganization: Mock;
  setDefaultOrganization: Mock;
  leaveOrganization: Mock;
  listPendingInvitations: Mock;
  acceptInvitation: Mock;
}

describe('OrganizationsAPI Namespace', () => {
  let api: OrganizationsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      getOrganization: vi.fn(),
      updateOrganization: vi.fn(),
      listOrganizationMembers: vi.fn(),
      inviteToOrganization: vi.fn(),
      listOrganizationInvitations: vi.fn(),
      revokeOrganizationInvitation: vi.fn(),
      removeOrganizationMember: vi.fn(),
      updateOrganizationMemberRole: vi.fn(),
      listUserOrganizations: vi.fn(),
      switchOrganization: vi.fn(),
      setDefaultOrganization: vi.fn(),
      leaveOrganization: vi.fn(),
      listPendingInvitations: vi.fn(),
      acceptInvitation: vi.fn(),
    };
    api = new OrganizationsAPI(mockClient as any);
  });

  // ===========================================================================
  // Organization Management
  // ===========================================================================

  describe('Organization Management', () => {
    it('should get organization', async () => {
      const mockOrg = {
        id: 'org1',
        name: 'Acme Corp',
        slug: 'acme',
        created_at: '2024-01-01T00:00:00Z',
        settings: { theme: 'dark', notifications: true },
      };
      mockClient.getOrganization.mockResolvedValue(mockOrg);

      const result = await api.get('org1');

      expect(mockClient.getOrganization).toHaveBeenCalledWith('org1');
      expect(result.name).toBe('Acme Corp');
    });

    it('should update organization', async () => {
      const mockOrg = { id: 'org1', name: 'Acme Corporation', settings: { theme: 'light' } };
      mockClient.updateOrganization.mockResolvedValue(mockOrg);

      const result = await api.update('org1', {
        name: 'Acme Corporation',
        settings: { theme: 'light' },
      });

      expect(mockClient.updateOrganization).toHaveBeenCalledWith('org1', {
        name: 'Acme Corporation',
        settings: { theme: 'light' },
      });
      expect(result.name).toBe('Acme Corporation');
    });
  });

  // ===========================================================================
  // Member Management
  // ===========================================================================

  describe('Member Management', () => {
    it('should list members', async () => {
      const mockMembers = {
        members: [
          { id: 'u1', email: 'admin@acme.com', role: 'admin', name: 'Admin' },
          { id: 'u2', email: 'user@acme.com', role: 'member', name: 'User' },
        ],
      };
      mockClient.listOrganizationMembers.mockResolvedValue(mockMembers);

      const result = await api.listMembers('org1');

      expect(mockClient.listOrganizationMembers).toHaveBeenCalledWith('org1', undefined);
      expect(result.members).toHaveLength(2);
    });

    it('should list members with pagination', async () => {
      const mockMembers = { members: [] };
      mockClient.listOrganizationMembers.mockResolvedValue(mockMembers);

      await api.listMembers('org1', { limit: 10, offset: 20 });

      expect(mockClient.listOrganizationMembers).toHaveBeenCalledWith('org1', {
        limit: 10,
        offset: 20,
      });
    });

    it('should remove member', async () => {
      mockClient.removeOrganizationMember.mockResolvedValue(undefined);

      await api.removeMember('org1', 'u2');

      expect(mockClient.removeOrganizationMember).toHaveBeenCalledWith('org1', 'u2');
    });

    it('should update member role', async () => {
      const mockMember = { id: 'u2', email: 'user@acme.com', role: 'admin' };
      mockClient.updateOrganizationMemberRole.mockResolvedValue(mockMember);

      const result = await api.updateMemberRole('org1', 'u2', 'admin');

      expect(mockClient.updateOrganizationMemberRole).toHaveBeenCalledWith('org1', 'u2', 'admin');
      expect(result.role).toBe('admin');
    });
  });

  // ===========================================================================
  // Invitations
  // ===========================================================================

  describe('Invitations', () => {
    it('should invite user', async () => {
      const mockInvitation = {
        id: 'inv1',
        email: 'newuser@example.com',
        role: 'member',
        created_at: '2024-01-20T10:00:00Z',
        expires_at: '2024-01-27T10:00:00Z',
        status: 'pending',
      };
      mockClient.inviteToOrganization.mockResolvedValue(mockInvitation);

      const result = await api.invite('org1', { email: 'newuser@example.com', role: 'member' });

      expect(mockClient.inviteToOrganization).toHaveBeenCalledWith('org1', {
        email: 'newuser@example.com',
        role: 'member',
      });
      expect(result.email).toBe('newuser@example.com');
    });

    it('should invite as admin', async () => {
      const mockInvitation = { id: 'inv2', email: 'admin@example.com', role: 'admin' };
      mockClient.inviteToOrganization.mockResolvedValue(mockInvitation);

      const result = await api.invite('org1', { email: 'admin@example.com', role: 'admin' });

      expect(result.role).toBe('admin');
    });

    it('should list invitations', async () => {
      const mockInvitations = {
        invitations: [
          { id: 'inv1', email: 'user1@example.com', status: 'pending' },
          { id: 'inv2', email: 'user2@example.com', status: 'expired' },
        ],
      };
      mockClient.listOrganizationInvitations.mockResolvedValue(mockInvitations);

      const result = await api.listInvitations('org1');

      expect(mockClient.listOrganizationInvitations).toHaveBeenCalledWith('org1', undefined);
      expect(result.invitations).toHaveLength(2);
    });

    it('should revoke invitation', async () => {
      mockClient.revokeOrganizationInvitation.mockResolvedValue(undefined);

      await api.revokeInvitation('org1', 'inv1');

      expect(mockClient.revokeOrganizationInvitation).toHaveBeenCalledWith('org1', 'inv1');
    });
  });

  // ===========================================================================
  // User Organization Operations
  // ===========================================================================

  describe('User Organization Operations', () => {
    it('should list user organizations', async () => {
      const mockOrgs = {
        organizations: [
          { id: 'org1', name: 'Acme', role: 'admin', is_default: true },
          { id: 'org2', name: 'TechStart', role: 'member', is_default: false },
        ],
      };
      mockClient.listUserOrganizations.mockResolvedValue(mockOrgs);

      const result = await api.listMine();

      expect(mockClient.listUserOrganizations).toHaveBeenCalled();
      expect(result.organizations).toHaveLength(2);
    });

    it('should switch organization', async () => {
      mockClient.switchOrganization.mockResolvedValue({ organization_id: 'org2' });

      const result = await api.switch('org2');

      expect(mockClient.switchOrganization).toHaveBeenCalledWith('org2');
      expect(result.organization_id).toBe('org2');
    });

    it('should set default organization', async () => {
      mockClient.setDefaultOrganization.mockResolvedValue({ organization_id: 'org2' });

      const result = await api.setDefault('org2');

      expect(mockClient.setDefaultOrganization).toHaveBeenCalledWith('org2');
      expect(result.organization_id).toBe('org2');
    });

    it('should leave organization', async () => {
      mockClient.leaveOrganization.mockResolvedValue(undefined);

      await api.leave('org2');

      expect(mockClient.leaveOrganization).toHaveBeenCalledWith('org2');
    });

    it('should list pending invitations', async () => {
      const mockInvitations = {
        invitations: [
          { id: 'inv1', organization_name: 'New Org', role: 'member' },
        ],
      };
      mockClient.listPendingInvitations.mockResolvedValue(mockInvitations);

      const result = await api.listMyInvitations();

      expect(mockClient.listPendingInvitations).toHaveBeenCalled();
      expect(result.invitations).toHaveLength(1);
    });

    it('should accept invitation', async () => {
      mockClient.acceptInvitation.mockResolvedValue({ organization_id: 'org3', role: 'member' });

      const result = await api.acceptInvitation('token123');

      expect(mockClient.acceptInvitation).toHaveBeenCalledWith('token123');
      expect(result.organization_id).toBe('org3');
    });
  });
});
