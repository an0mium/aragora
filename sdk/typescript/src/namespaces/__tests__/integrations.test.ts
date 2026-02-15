/**
 * Integrations Namespace Tests
 *
 * Comprehensive tests for the integrations namespace API including:
 * - Integration discovery
 * - Integration CRUD operations
 * - Testing and sync
 * - Bot platforms (Slack, Telegram, WhatsApp, Discord)
 * - Teams integration
 * - Zapier, Make, and n8n integrations
 * - Integration wizard
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { IntegrationsAPI } from '../integrations';

interface MockClient {
  get: Mock;
  post: Mock;
  put: Mock;
  delete: Mock;
  request: Mock;
}

describe('IntegrationsAPI Namespace', () => {
  let api: IntegrationsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      get: vi.fn(),
      post: vi.fn(),
      put: vi.fn(),
      delete: vi.fn(),
      request: vi.fn(),
    };
    api = new IntegrationsAPI(mockClient as any);
  });

  // ===========================================================================
  // Integration Discovery
  // ===========================================================================

  describe('Integration Discovery', () => {
    it('should list available integrations', async () => {
      const mockAvailable = {
        integrations: [
          { type: 'slack', name: 'Slack', category: 'communication', oauth_supported: true },
          { type: 'teams', name: 'Microsoft Teams', category: 'communication', oauth_supported: true },
          { type: 's3', name: 'Amazon S3', category: 'storage', oauth_supported: false },
        ],
      };
      mockClient.request.mockResolvedValue(mockAvailable);

      const result = await api.listAvailable();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/integrations/available');
      expect(result.integrations).toHaveLength(3);
    });

    it('should get config schema', async () => {
      const mockSchema = {
        type: 'slack',
        required_fields: [
          { name: 'workspace_id', type: 'string', description: 'Slack workspace ID' },
        ],
        optional_fields: [
          { name: 'channel', type: 'string', description: 'Default channel', default: '#general' },
        ],
      };
      mockClient.request.mockResolvedValue(mockSchema);

      const result = await api.getConfigSchema('slack');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/integrations/config/slack');
      expect(result.type).toBe('slack');
    });
  });

  // ===========================================================================
  // Integration CRUD
  // ===========================================================================

  describe('Integration CRUD', () => {
    it('should list integrations', async () => {
      const mockIntegrations = {
        integrations: [
          { id: 'i1', type: 'slack', name: 'Slack', status: 'connected' },
          { id: 'i2', type: 'teams', name: 'Teams', status: 'disconnected' },
        ],
        total: 2,
      };
      mockClient.request.mockResolvedValue(mockIntegrations);

      const result = await api.list();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/integrations');
      expect(result.integrations).toHaveLength(2);
    });

    it('should get integration by ID', async () => {
      const mockIntegration = {
        id: 'i1',
        type: 'slack',
        name: 'Slack Workspace',
        status: 'connected',
        credentials_stored: true,
      };
      mockClient.request.mockResolvedValue(mockIntegration);

      const result = await api.get('i1');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/integrations/i1');
      expect(result.status).toBe('connected');
    });

    it('should create integration', async () => {
      const mockIntegration = {
        id: 'i_new',
        type: 'slack',
        name: 'My Slack',
        status: 'pending',
      };
      mockClient.request.mockResolvedValue(mockIntegration);

      const result = await api.create({
        type: 'slack',
        name: 'My Slack',
        config: { workspace_id: 'T123456' },
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/integrations', {
        json: {
          type: 'slack',
          name: 'My Slack',
          config: { workspace_id: 'T123456' },
        },
      });
      expect(result.id).toBe('i_new');
    });

    it('should update integration', async () => {
      const mockIntegration = { id: 'i1', name: 'Updated Name', enabled: false };
      mockClient.request.mockResolvedValue(mockIntegration);

      const result = await api.update('i1', { name: 'Updated Name', enabled: false });

      expect(mockClient.request).toHaveBeenCalledWith('PUT', '/api/v1/integrations/i1', {
        json: {
          name: 'Updated Name',
          enabled: false,
        },
      });
    });

    it('should delete integration', async () => {
      mockClient.request.mockResolvedValue({ deleted: true });

      const result = await api.delete('i1');

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/api/v1/integrations/i1');
      expect(result.deleted).toBe(true);
    });
  });

  // ===========================================================================
  // Testing and Sync
  // ===========================================================================

  describe('Testing and Sync', () => {
    it('should test integration', async () => {
      mockClient.request.mockResolvedValue({ success: true, message: 'Connection successful' });

      const result = await api.test('i1');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/integrations/i1/test');
      expect(result.success).toBe(true);
    });

    it('should trigger sync', async () => {
      const mockSync = {
        integration_id: 'i1',
        status: 'syncing',
        items_synced: 0,
      };
      mockClient.request.mockResolvedValue(mockSync);

      const result = await api.sync('i1');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/integrations/i1/sync');
      expect(result.status).toBe('syncing');
    });

    it('should get sync status', async () => {
      const mockSync = {
        integration_id: 'i1',
        status: 'idle',
        items_synced: 1500,
        last_sync_at: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockSync);

      const result = await api.getSyncStatus('i1');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/integrations/i1/sync');
      expect(result.items_synced).toBe(1500);
    });
  });

  // ===========================================================================
  // Bot Platforms
  // ===========================================================================

  describe('Bot Platforms', () => {
    it('should get Slack status', async () => {
      const mockStatus = { platform: 'slack', connected: true, active_debates: 5 };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getSlackStatus();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/bots/slack/status');
      expect(result.connected).toBe(true);
    });

    it('should get Telegram status', async () => {
      const mockStatus = { platform: 'telegram', connected: true };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getTelegramStatus();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/bots/telegram/status');
    });

    it('should get WhatsApp status', async () => {
      const mockStatus = { platform: 'whatsapp', connected: false, error: 'Not configured' };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getWhatsAppStatus();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/bots/whatsapp/status');
      expect(result.connected).toBe(false);
    });

    it('should get Discord status', async () => {
      const mockStatus = { platform: 'discord', connected: true, active_debates: 2 };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getDiscordStatus();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/bots/discord/status');
    });

    it('should get email status', async () => {
      const mockStatus = {
        sendgrid: { platform: 'sendgrid', connected: true },
        ses: { platform: 'ses', connected: false },
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getEmailStatus();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/bots/email/status');
      expect(result.sendgrid.connected).toBe(true);
    });
  });

  // ===========================================================================
  // Zapier Integration
  // ===========================================================================

  describe('Zapier Integration', () => {
    it('should list Zapier apps', async () => {
      const mockApps = {
        apps: [{ id: 'app1', workspace_id: 'ws1', active: true, trigger_count: 3 }],
        count: 1,
      };
      mockClient.request.mockResolvedValue(mockApps);

      const result = await api.listZapierApps();

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/integrations/zapier/apps',
        undefined
      );
      expect(result.apps).toHaveLength(1);
    });

    it('should create Zapier app', async () => {
      const mockApp = {
        app: { id: 'app_new', workspace_id: 'ws1', api_key: 'key_123', api_secret: 'secret_123' },
        message: 'App created',
      };
      mockClient.request.mockResolvedValue(mockApp);

      const result = await api.createZapierApp({ workspace_id: 'ws1' });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/integrations/zapier/apps', {
        json: { workspace_id: 'ws1' },
      });
      expect(result.app.api_key).toBe('key_123');
    });

    it('should delete Zapier app', async () => {
      mockClient.request.mockResolvedValue({ deleted: true, app_id: 'app1' });

      const result = await api.deleteZapierApp('app1');

      expect(mockClient.request).toHaveBeenCalledWith(
        'DELETE',
        '/api/v1/integrations/zapier/apps/app1'
      );
      expect(result.deleted).toBe(true);
    });

    it('should subscribe to trigger', async () => {
      const mockTrigger = {
        trigger: { id: 'trig1', trigger_type: 'debate_completed', webhook_url: 'https://...' },
      };
      mockClient.request.mockResolvedValue(mockTrigger);

      const result = await api.subscribeZapierTrigger({
        app_id: 'app1',
        trigger_type: 'debate_completed',
        webhook_url: 'https://hooks.zapier.com/...',
      });

      expect(result.trigger.id).toBe('trig1');
    });
  });

  // ===========================================================================
  // n8n Integration
  // ===========================================================================

  describe('n8n Integration', () => {
    it('should list n8n credentials', async () => {
      const mockCredentials = {
        credentials: [{ id: 'cred1', workspace_id: 'ws1', active: true }],
        count: 1,
      };
      mockClient.request.mockResolvedValue(mockCredentials);

      const result = await api.listN8nCredentials();

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/integrations/n8n/credentials',
        undefined
      );
      expect(result.credentials).toHaveLength(1);
    });

    it('should create n8n credential', async () => {
      const mockCred = {
        credential: { id: 'cred_new', api_key: 'key_123', api_url: 'https://n8n.example.com' },
        message: 'Credential created',
      };
      mockClient.request.mockResolvedValue(mockCred);

      const result = await api.createN8nCredential({
        workspace_id: 'ws1',
        api_url: 'https://n8n.example.com',
      });

      expect(result.credential.api_key).toBe('key_123');
    });

    it('should get n8n nodes', async () => {
      const mockNodes = {
        node: { name: 'aragora', displayName: 'Aragora' },
        trigger: { name: 'aragoraTrigger', displayName: 'Aragora Trigger' },
        credential: {},
        events: {},
      };
      mockClient.request.mockResolvedValue(mockNodes);

      const result = await api.getN8nNodes();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/integrations/n8n/nodes');
      expect(result.node.name).toBe('aragora');
    });
  });

  // ===========================================================================
  // Integration Wizard
  // ===========================================================================

  describe('Integration Wizard', () => {
    it('should start wizard', async () => {
      const mockWizard = { session_id: 'sess_123', steps: ['configure', 'test', 'activate'] };
      mockClient.request.mockResolvedValue(mockWizard);

      const result = await api.startWizard('slack');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v2/integrations/wizard', {
        json: { type: 'slack' },
      });
      expect(result.steps).toHaveLength(3);
    });

    it('should list wizard providers', async () => {
      const mockProviders = {
        providers: [
          { type: 'slack', name: 'Slack', category: 'communication' },
          { type: 'teams', name: 'Teams', category: 'communication' },
        ],
      };
      mockClient.request.mockResolvedValue(mockProviders);

      const result = await api.listWizardProviders();

      expect(result.providers).toHaveLength(2);
    });

    it('should get wizard status', async () => {
      const mockStatus = {
        steps: [
          { step: 'configure', status: 'completed' },
          { step: 'test', status: 'in_progress' },
        ],
        current_step: 'test',
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getWizardStatus('sess_123');

      expect(result.current_step).toBe('test');
    });

    it('should validate wizard step', async () => {
      mockClient.request.mockResolvedValue({ valid: true });

      const result = await api.validateWizardStep('sess_123', 'configure', { api_key: 'key' });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v2/integrations/wizard/validate', {
        json: { session_id: 'sess_123', step: 'configure', data: { api_key: 'key' } },
      });
      expect(result.valid).toBe(true);
    });
  });

  // ===========================================================================
  // Integration Stats
  // ===========================================================================

  describe('Integration Stats', () => {
    it('should get integration stats', async () => {
      const mockStats = {
        total_integrations: 10,
        active_integrations: 8,
        by_type: { slack: 3, teams: 2, s3: 3 },
        total_syncs_24h: 150,
        failed_syncs_24h: 2,
      };
      mockClient.request.mockResolvedValue(mockStats);

      const result = await api.getStats();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v2/integrations/stats');
      expect(result.total_integrations).toBe(10);
    });
  });
});
