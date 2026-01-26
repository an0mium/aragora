/**
 * Aragora SDK Enterprise Namespace API Tests
 *
 * Tests for enterprise namespaces: rbac, audit, webhooks, plugins, workspaces,
 * integrations, organizations, tenants, controlPlane.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createClient } from '../client';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('Enterprise Namespace APIs', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rbac namespace', () => {
    it('should expose rbac namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.rbac).toBeDefined();
      expect(typeof client.rbac.listRoles).toBe('function');
      expect(typeof client.rbac.getRole).toBe('function');
      expect(typeof client.rbac.createRole).toBe('function');
    });

    it('should list roles via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          roles: [{ id: 'admin', name: 'Admin', permissions: ['*'] }]
        })),
      });

      const result = await client.rbac.listRoles();
      expect(result.roles).toHaveLength(1);
      expect(result.roles[0].id).toBe('admin');
    });

    it('should create role via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ id: 'new-role', name: 'Custom Role' })),
      });

      const result = await client.rbac.createRole({
        name: 'Custom Role',
        permissions: ['debates.read', 'debates.create']
      });
      expect(result.id).toBe('new-role');
    });

    it('should get user permissions via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          user_id: 'user-123',
          permissions: ['debates.read', 'debates.create'],
          roles: ['editor']
        })),
      });

      const result = await client.rbac.getUserPermissions('user-123');
      expect(result.permissions).toContain('debates.read');
    });

    it('should check permission via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ allowed: true })),
      });

      const result = await client.rbac.checkPermission('user-123', 'debates.create');
      expect(result.allowed).toBe(true);
    });
  });

  describe('audit namespace', () => {
    it('should expose audit namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.audit).toBeDefined();
      expect(typeof client.audit.listEvents).toBe('function');
      expect(typeof client.audit.getEvent).toBe('function');
    });

    it('should list audit events via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          events: [{ id: 'e1', action: 'debate.create', actor_id: 'user-123' }],
          total: 1
        })),
      });

      const result = await client.audit.listEvents();
      expect(result.events).toHaveLength(1);
      expect(result.events[0].action).toBe('debate.create');
    });

    it('should get audit event by ID via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'event-123',
          action: 'user.login',
          actor_id: 'user-456',
          timestamp: '2024-01-15T10:00:00Z'
        })),
      });

      const event = await client.audit.getEvent('event-123');
      expect(event.id).toBe('event-123');
      expect(event.action).toBe('user.login');
    });

    it('should export audit logs via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          export_id: 'export-123',
          status: 'pending',
          download_url: null
        })),
      });

      const result = await client.audit.exportLogs({ format: 'csv', start_date: '2024-01-01' });
      expect(result.export_id).toBe('export-123');
    });
  });

  describe('webhooks namespace', () => {
    it('should expose webhooks namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.webhooks).toBeDefined();
      expect(typeof client.webhooks.list).toBe('function');
      expect(typeof client.webhooks.create).toBe('function');
    });

    it('should list webhooks via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          webhooks: [{ id: 'wh1', url: 'https://example.com/hook', events: ['debate.completed'] }]
        })),
      });

      const result = await client.webhooks.list();
      expect(result.webhooks).toHaveLength(1);
      expect(result.webhooks[0].events).toContain('debate.completed');
    });

    it('should create webhook via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'webhook-123',
          url: 'https://example.com/hook',
          secret: 'whsec_xxx'
        })),
      });

      const result = await client.webhooks.create({
        url: 'https://example.com/hook',
        events: ['debate.completed', 'workflow.completed']
      });
      expect(result.id).toBe('webhook-123');
      expect(result.secret).toBeDefined();
    });

    it('should test webhook via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          success: true,
          response_code: 200,
          latency_ms: 150
        })),
      });

      const result = await client.webhooks.test('webhook-123');
      expect(result.success).toBe(true);
    });

    it('should list webhook deliveries via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          deliveries: [{ id: 'd1', status: 'success', response_code: 200 }]
        })),
      });

      const result = await client.webhooks.listDeliveries('webhook-123');
      expect(result.deliveries).toHaveLength(1);
      expect(result.deliveries[0].status).toBe('success');
    });
  });

  describe('plugins namespace', () => {
    it('should expose plugins namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.plugins).toBeDefined();
      expect(typeof client.plugins.list).toBe('function');
      expect(typeof client.plugins.install).toBe('function');
    });

    it('should list available plugins via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          plugins: [{ id: 'jira', name: 'Jira Integration', version: '1.0.0' }]
        })),
      });

      const result = await client.plugins.list();
      expect(result.plugins).toHaveLength(1);
      expect(result.plugins[0].id).toBe('jira');
    });

    it('should install plugin via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'install-123',
          plugin_id: 'jira',
          status: 'installed'
        })),
      });

      const result = await client.plugins.install({ plugin_id: 'jira', config: {} });
      expect(result.status).toBe('installed');
    });

    it('should configure plugin via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'jira',
          config: { domain: 'company.atlassian.net' },
          updated_at: '2024-01-15'
        })),
      });

      const result = await client.plugins.configure('jira', { domain: 'company.atlassian.net' });
      expect(result.config.domain).toBe('company.atlassian.net');
    });
  });

  describe('workspaces namespace', () => {
    it('should expose workspaces namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.workspaces).toBeDefined();
      expect(typeof client.workspaces.list).toBe('function');
      expect(typeof client.workspaces.create).toBe('function');
    });

    it('should list workspaces via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          workspaces: [{ id: 'ws1', name: 'Engineering', member_count: 10 }]
        })),
      });

      const result = await client.workspaces.list();
      expect(result.workspaces).toHaveLength(1);
      expect(result.workspaces[0].name).toBe('Engineering');
    });

    it('should create workspace via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'workspace-123',
          name: 'New Workspace',
          created_at: '2024-01-15'
        })),
      });

      const result = await client.workspaces.create({ name: 'New Workspace' });
      expect(result.id).toBe('workspace-123');
    });

    it('should add member to workspace via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          user_id: 'user-123',
          workspace_id: 'workspace-456',
          role: 'member'
        })),
      });

      const result = await client.workspaces.addMember('workspace-456', {
        user_id: 'user-123',
        role: 'member'
      });
      expect(result.user_id).toBe('user-123');
    });

    it('should list workspace members via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          members: [{ user_id: 'u1', role: 'admin' }, { user_id: 'u2', role: 'member' }]
        })),
      });

      const result = await client.workspaces.listMembers('workspace-123');
      expect(result.members).toHaveLength(2);
    });
  });

  describe('integrations namespace', () => {
    it('should expose integrations namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.integrations).toBeDefined();
      expect(typeof client.integrations.list).toBe('function');
      expect(typeof client.integrations.connect).toBe('function');
    });

    it('should list integrations via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          integrations: [{ id: 'slack', status: 'connected', workspace: 'acme' }]
        })),
      });

      const result = await client.integrations.list();
      expect(result.integrations).toHaveLength(1);
      expect(result.integrations[0].status).toBe('connected');
    });

    it('should connect integration via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'integration-123',
          type: 'github',
          status: 'connected'
        })),
      });

      const result = await client.integrations.connect({
        type: 'github',
        config: { organization: 'my-org' }
      });
      expect(result.status).toBe('connected');
    });

    it('should sync integration via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          sync_id: 'sync-123',
          status: 'in_progress',
          started_at: '2024-01-15T10:00:00Z'
        })),
      });

      const result = await client.integrations.sync('integration-123');
      expect(result.status).toBe('in_progress');
    });
  });

  describe('organizations namespace', () => {
    it('should expose organizations namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.organizations).toBeDefined();
      expect(typeof client.organizations.get).toBe('function');
      expect(typeof client.organizations.update).toBe('function');
    });

    it('should get organization via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'org-123',
          name: 'Acme Corp',
          plan: 'enterprise',
          member_count: 50
        })),
      });

      const org = await client.organizations.get();
      expect(org.name).toBe('Acme Corp');
      expect(org.plan).toBe('enterprise');
    });

    it('should update organization via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'org-123',
          name: 'Acme Industries',
          updated_at: '2024-01-15'
        })),
      });

      const result = await client.organizations.update({ name: 'Acme Industries' });
      expect(result.name).toBe('Acme Industries');
    });

    it('should list organization members via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          members: [{ id: 'u1', email: 'admin@acme.com', role: 'owner' }],
          total: 50
        })),
      });

      const result = await client.organizations.listMembers();
      expect(result.members).toHaveLength(1);
    });
  });

  describe('tenants namespace', () => {
    it('should expose tenants namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.tenants).toBeDefined();
      expect(typeof client.tenants.get).toBe('function');
    });

    it('should get current tenant via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          id: 'tenant-123',
          name: 'Acme Tenant',
          limits: { max_debates: 1000, max_users: 100 }
        })),
      });

      const tenant = await client.tenants.get();
      expect(tenant.name).toBe('Acme Tenant');
      expect(tenant.limits.max_debates).toBe(1000);
    });

    it('should get tenant usage via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          debates_used: 150,
          storage_bytes: 1024000,
          api_calls_this_month: 5000
        })),
      });

      const usage = await client.tenants.getUsage();
      expect(usage.debates_used).toBe(150);
    });
  });

  describe('controlPlane namespace', () => {
    it('should expose controlPlane namespace', () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });
      expect(client.controlPlane).toBeDefined();
      expect(typeof client.controlPlane.getHealth).toBe('function');
      expect(typeof client.controlPlane.agent).toBeDefined();
      expect(typeof client.controlPlane.task).toBeDefined();
    });

    it('should get control plane health via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          status: 'healthy',
          active_agents: 5,
          queued_tasks: 10
        })),
      });

      const health = await client.controlPlane.getHealth();
      expect(health.status).toBe('healthy');
    });

    it('should list control plane agents via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          agents: [{ id: 'a1', status: 'online', last_heartbeat: '2024-01-15T10:00:00Z' }]
        })),
      });

      const result = await client.controlPlane.agent.list();
      expect(result.agents).toHaveLength(1);
      expect(result.agents[0].status).toBe('online');
    });

    it('should submit task via namespace', async () => {
      const client = createClient({ baseUrl: 'https://api.example.com' });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({
          task_id: 'task-123',
          status: 'queued'
        })),
      });

      const result = await client.controlPlane.task.submit({
        type: 'debate',
        payload: { task: 'Review PR' }
      });
      expect(result.task_id).toBe('task-123');
      expect(result.status).toBe('queued');
    });
  });
});
