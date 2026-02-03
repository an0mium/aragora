/**
 * SCIM 2.0 Namespace Tests
 *
 * Comprehensive tests for the SCIM namespace API including:
 * - User CRUD operations
 * - Group CRUD operations
 * - Filtering and pagination
 * - PATCH operations
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { SCIMAPI } from '../scim';

interface MockClient {
  request: Mock;
}

describe('SCIMAPI Namespace', () => {
  let api: SCIMAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new SCIMAPI(mockClient as any);
  });

  // ===========================================================================
  // User Operations
  // ===========================================================================

  describe('User Operations', () => {
    it('should list users', async () => {
      const mockResponse = {
        schemas: ['urn:ietf:params:scim:api:messages:2.0:ListResponse'],
        totalResults: 2,
        startIndex: 1,
        itemsPerPage: 100,
        Resources: [
          { id: 'u_1', userName: 'john@example.com', active: true },
          { id: 'u_2', userName: 'jane@example.com', active: true },
        ],
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.listUsers();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/scim/v2/Users', {
        params: { startIndex: 1, count: 100 },
      });
      expect(result.totalResults).toBe(2);
      expect(result.Resources).toHaveLength(2);
    });

    it('should list users with filtering', async () => {
      const mockResponse = {
        schemas: ['urn:ietf:params:scim:api:messages:2.0:ListResponse'],
        totalResults: 1,
        startIndex: 1,
        itemsPerPage: 100,
        Resources: [{ id: 'u_1', userName: 'john@example.com' }],
      };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.listUsers({
        filter: 'userName eq "john@example.com"',
        startIndex: 1,
        count: 50,
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/scim/v2/Users', {
        params: { startIndex: 1, count: 50, filter: 'userName eq "john@example.com"' },
      });
    });

    it('should get user by ID', async () => {
      const mockUser = {
        schemas: ['urn:ietf:params:scim:schemas:core:2.0:User'],
        id: 'u_123',
        userName: 'john@example.com',
        name: { givenName: 'John', familyName: 'Doe' },
        emails: [{ value: 'john@example.com', primary: true }],
        active: true,
        meta: {
          resourceType: 'User',
          created: '2024-01-15T10:00:00Z',
          lastModified: '2024-01-20T15:00:00Z',
        },
      };
      mockClient.request.mockResolvedValue(mockUser);

      const result = await api.getUser('u_123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/scim/v2/Users/u_123');
      expect(result.userName).toBe('john@example.com');
      expect(result.name?.givenName).toBe('John');
    });

    it('should create user', async () => {
      const newUser = {
        schemas: ['urn:ietf:params:scim:schemas:core:2.0:User'],
        userName: 'newuser@example.com',
        name: { givenName: 'New', familyName: 'User' },
        emails: [{ value: 'newuser@example.com', primary: true }],
        active: true,
      };
      const mockResponse = { ...newUser, id: 'u_new' };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.createUser(newUser);

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/scim/v2/Users', {
        json: newUser,
      });
      expect(result.id).toBe('u_new');
    });

    it('should replace user', async () => {
      const updatedUser = {
        schemas: ['urn:ietf:params:scim:schemas:core:2.0:User'],
        userName: 'john@example.com',
        name: { givenName: 'John', familyName: 'Smith' },
        emails: [{ value: 'john@example.com', primary: true }],
        active: true,
      };
      const mockResponse = { ...updatedUser, id: 'u_123' };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.replaceUser('u_123', updatedUser);

      expect(mockClient.request).toHaveBeenCalledWith('PUT', '/scim/v2/Users/u_123', {
        json: updatedUser,
      });
      expect(result.name?.familyName).toBe('Smith');
    });

    it('should patch user', async () => {
      const patchOp = {
        schemas: ['urn:ietf:params:scim:api:messages:2.0:PatchOp'],
        Operations: [{ op: 'replace' as const, path: 'active', value: false }],
      };
      const mockResponse = {
        id: 'u_123',
        userName: 'john@example.com',
        active: false,
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.patchUser('u_123', patchOp);

      expect(mockClient.request).toHaveBeenCalledWith('PATCH', '/scim/v2/Users/u_123', {
        json: patchOp,
      });
      expect(result.active).toBe(false);
    });

    it('should delete user', async () => {
      mockClient.request.mockResolvedValue(undefined);

      await api.deleteUser('u_123');

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/scim/v2/Users/u_123');
    });
  });

  // ===========================================================================
  // Group Operations
  // ===========================================================================

  describe('Group Operations', () => {
    it('should list groups', async () => {
      const mockResponse = {
        schemas: ['urn:ietf:params:scim:api:messages:2.0:ListResponse'],
        totalResults: 2,
        startIndex: 1,
        itemsPerPage: 100,
        Resources: [
          { id: 'g_1', displayName: 'Admins', members: [] },
          { id: 'g_2', displayName: 'Users', members: [] },
        ],
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.listGroups();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/scim/v2/Groups', {
        params: { startIndex: 1, count: 100 },
      });
      expect(result.totalResults).toBe(2);
    });

    it('should list groups with filtering', async () => {
      const mockResponse = {
        schemas: ['urn:ietf:params:scim:api:messages:2.0:ListResponse'],
        totalResults: 1,
        startIndex: 1,
        itemsPerPage: 100,
        Resources: [{ id: 'g_1', displayName: 'Admins' }],
      };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.listGroups({
        filter: 'displayName eq "Admins"',
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/scim/v2/Groups', {
        params: { startIndex: 1, count: 100, filter: 'displayName eq "Admins"' },
      });
    });

    it('should get group by ID', async () => {
      const mockGroup = {
        schemas: ['urn:ietf:params:scim:schemas:core:2.0:Group'],
        id: 'g_123',
        displayName: 'Engineering',
        members: [
          { value: 'u_1', display: 'John Doe' },
          { value: 'u_2', display: 'Jane Smith' },
        ],
        meta: {
          resourceType: 'Group',
          created: '2024-01-10T10:00:00Z',
        },
      };
      mockClient.request.mockResolvedValue(mockGroup);

      const result = await api.getGroup('g_123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/scim/v2/Groups/g_123');
      expect(result.displayName).toBe('Engineering');
      expect(result.members).toHaveLength(2);
    });

    it('should create group', async () => {
      const newGroup = {
        schemas: ['urn:ietf:params:scim:schemas:core:2.0:Group'],
        displayName: 'New Team',
        members: [{ value: 'u_1' }],
      };
      const mockResponse = { ...newGroup, id: 'g_new' };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.createGroup(newGroup);

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/scim/v2/Groups', {
        json: newGroup,
      });
      expect(result.id).toBe('g_new');
    });

    it('should replace group', async () => {
      const updatedGroup = {
        schemas: ['urn:ietf:params:scim:schemas:core:2.0:Group'],
        displayName: 'Renamed Team',
        members: [{ value: 'u_1' }, { value: 'u_2' }],
      };
      const mockResponse = { ...updatedGroup, id: 'g_123' };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.replaceGroup('g_123', updatedGroup);

      expect(mockClient.request).toHaveBeenCalledWith('PUT', '/scim/v2/Groups/g_123', {
        json: updatedGroup,
      });
      expect(result.displayName).toBe('Renamed Team');
    });

    it('should patch group to add member', async () => {
      const patchOp = {
        schemas: ['urn:ietf:params:scim:api:messages:2.0:PatchOp'],
        Operations: [
          {
            op: 'add' as const,
            path: 'members',
            value: [{ value: 'u_3' }],
          },
        ],
      };
      const mockResponse = {
        id: 'g_123',
        displayName: 'Team',
        members: [{ value: 'u_1' }, { value: 'u_2' }, { value: 'u_3' }],
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.patchGroup('g_123', patchOp);

      expect(mockClient.request).toHaveBeenCalledWith('PATCH', '/scim/v2/Groups/g_123', {
        json: patchOp,
      });
      expect(result.members).toHaveLength(3);
    });

    it('should patch group to remove member', async () => {
      const patchOp = {
        schemas: ['urn:ietf:params:scim:api:messages:2.0:PatchOp'],
        Operations: [
          {
            op: 'remove' as const,
            path: 'members[value eq "u_2"]',
          },
        ],
      };
      const mockResponse = {
        id: 'g_123',
        displayName: 'Team',
        members: [{ value: 'u_1' }],
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.patchGroup('g_123', patchOp);

      expect(mockClient.request).toHaveBeenCalledWith('PATCH', '/scim/v2/Groups/g_123', {
        json: patchOp,
      });
      expect(result.members).toHaveLength(1);
    });

    it('should delete group', async () => {
      mockClient.request.mockResolvedValue(undefined);

      await api.deleteGroup('g_123');

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/scim/v2/Groups/g_123');
    });
  });

  // ===========================================================================
  // Pagination
  // ===========================================================================

  describe('Pagination', () => {
    it('should handle pagination for users', async () => {
      const mockResponse = {
        schemas: ['urn:ietf:params:scim:api:messages:2.0:ListResponse'],
        totalResults: 150,
        startIndex: 101,
        itemsPerPage: 50,
        Resources: Array(50).fill({ id: 'u', userName: 'user@example.com' }),
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.listUsers({ startIndex: 101, count: 50 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/scim/v2/Users', {
        params: { startIndex: 101, count: 50 },
      });
      expect(result.startIndex).toBe(101);
      expect(result.itemsPerPage).toBe(50);
    });

    it('should handle pagination for groups', async () => {
      const mockResponse = {
        schemas: ['urn:ietf:params:scim:api:messages:2.0:ListResponse'],
        totalResults: 75,
        startIndex: 51,
        itemsPerPage: 25,
        Resources: Array(25).fill({ id: 'g', displayName: 'Group' }),
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.listGroups({ startIndex: 51, count: 25 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/scim/v2/Groups', {
        params: { startIndex: 51, count: 25 },
      });
      expect(result.startIndex).toBe(51);
    });
  });
});
