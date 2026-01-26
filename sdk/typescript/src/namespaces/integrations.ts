/**
 * Integrations Namespace API
 *
 * Provides a namespaced interface for external integration management.
 * Essential for connecting Aragora to external services.
 */

/**
 * Integration definition
 */
export interface Integration {
  id: string;
  type: string;
  name: string;
  description?: string;
  status: 'connected' | 'disconnected' | 'error' | 'pending';
  config: Record<string, unknown>;
  credentials_stored: boolean;
  last_sync_at?: string;
  error_message?: string;
  created_at: string;
  updated_at: string;
}

/**
 * Available integration type
 */
export interface AvailableIntegration {
  type: string;
  name: string;
  description: string;
  category: 'communication' | 'storage' | 'productivity' | 'analytics' | 'custom';
  icon_url?: string;
  oauth_supported: boolean;
  required_fields: string[];
  optional_fields: string[];
  documentation_url?: string;
  setup_time_minutes: number;
}

/**
 * Integration config schema
 */
export interface IntegrationConfigSchema {
  type: string;
  required_fields: {
    name: string;
    type: string;
    description: string;
    sensitive?: boolean;
  }[];
  optional_fields: {
    name: string;
    type: string;
    description: string;
    default?: unknown;
  }[];
}

/**
 * Integration sync status
 */
export interface IntegrationSyncStatus {
  integration_id: string;
  last_sync_at?: string;
  next_sync_at?: string;
  sync_frequency_minutes: number;
  items_synced: number;
  errors: string[];
  status: 'idle' | 'syncing' | 'error';
}

/**
 * Create integration request
 */
export interface CreateIntegrationRequest {
  type: string;
  name?: string;
  config: Record<string, unknown>;
}

/**
 * Update integration request
 */
export interface UpdateIntegrationRequest {
  name?: string;
  config?: Record<string, unknown>;
  enabled?: boolean;
}

/**
 * Bot status response
 */
export interface BotStatus {
  platform: string;
  connected: boolean;
  last_activity_at?: string;
  active_debates: number;
  error?: string;
}

/**
 * Zapier app configuration
 */
export interface ZapierApp {
  id: string;
  name: string;
  api_key: string;
  triggers_enabled: string[];
  actions_enabled: string[];
  created_at: string;
}

/**
 * Zapier trigger type
 */
export interface ZapierTrigger {
  id: string;
  name: string;
  description: string;
  sample_payload: Record<string, unknown>;
}

/**
 * Make connection
 */
export interface MakeConnection {
  id: string;
  name: string;
  status: 'active' | 'inactive' | 'error';
  last_used_at?: string;
  created_at: string;
}

/**
 * Make module definition
 */
export interface MakeModule {
  id: string;
  name: string;
  description: string;
  type: 'action' | 'trigger' | 'search';
  inputs: { name: string; type: string; required: boolean }[];
  outputs: { name: string; type: string }[];
}

/**
 * n8n credential
 */
export interface N8nCredential {
  id: string;
  name: string;
  type: string;
  created_at: string;
  updated_at: string;
}

/**
 * n8n node definition
 */
export interface N8nNode {
  name: string;
  displayName: string;
  description: string;
  version: number;
  inputs: string[];
  outputs: string[];
  properties: Record<string, unknown>[];
}

/**
 * Teams status
 */
export interface TeamsStatus {
  connected: boolean;
  tenant_id?: string;
  bot_id?: string;
  channels: number;
  error?: string;
}

/**
 * Wizard provider
 */
export interface WizardProvider {
  type: string;
  name: string;
  description: string;
  category: string;
  icon_url?: string;
  oauth_url?: string;
  fields: { name: string; type: string; required: boolean; description: string }[];
}

/**
 * Wizard step status
 */
export interface WizardStepStatus {
  step: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  message?: string;
  error?: string;
}

/**
 * Integration health status
 */
export interface IntegrationHealth {
  type: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency_ms?: number;
  last_check_at: string;
  details?: Record<string, unknown>;
}

/**
 * Integration stats
 */
export interface IntegrationStats {
  total_integrations: number;
  active_integrations: number;
  by_type: Record<string, number>;
  by_status: Record<string, number>;
  total_syncs_24h: number;
  failed_syncs_24h: number;
}

/**
 * Interface for the internal client used by IntegrationsAPI.
 */
interface IntegrationsClientInterface {
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
  put<T>(path: string, body?: unknown): Promise<T>;
  delete<T>(path: string): Promise<T>;
  request<T = unknown>(method: string, path: string, options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }): Promise<T>;
}

/**
 * Integrations API namespace.
 *
 * Provides methods for managing external integrations:
 * - Connect to external services (Slack, Teams, etc.)
 * - Configure integration settings
 * - Monitor sync status
 * - Test connections
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // List available integrations
 * const { integrations: available } = await client.integrations.listAvailable();
 *
 * // Get config schema for Slack
 * const schema = await client.integrations.getConfigSchema('slack');
 *
 * // Create a Slack integration
 * const integration = await client.integrations.create({
 *   type: 'slack',
 *   config: { workspace_id: 'T123456', channel: '#general' },
 * });
 *
 * // Test the connection
 * const result = await client.integrations.test(integration.id);
 *
 * // Trigger a sync
 * const status = await client.integrations.sync(integration.id);
 * ```
 */
export class IntegrationsAPI {
  constructor(private client: IntegrationsClientInterface) {}

  // ===========================================================================
  // Integration Discovery
  // ===========================================================================

  /**
   * List available integration types.
   */
  async listAvailable(): Promise<{ integrations: AvailableIntegration[] }> {
    return this.client.get('/api/integrations/available');
  }

  /**
   * Get configuration schema for an integration type.
   */
  async getConfigSchema(type: string): Promise<IntegrationConfigSchema> {
    return this.client.get(`/api/integrations/config/${type}`);
  }

  // ===========================================================================
  // Integration Management
  // ===========================================================================

  /**
   * List all configured integrations.
   */
  async list(): Promise<{ integrations: Integration[]; total: number }> {
    return this.client.get('/api/integrations');
  }

  /**
   * Get a specific integration by ID.
   */
  async get(id: string): Promise<Integration> {
    return this.client.get(`/api/integrations/${id}`);
  }

  /**
   * Create a new integration.
   */
  async create(body: CreateIntegrationRequest): Promise<Integration> {
    return this.client.post('/api/integrations', body);
  }

  /**
   * Update an existing integration.
   */
  async update(id: string, body: UpdateIntegrationRequest): Promise<Integration> {
    return this.client.put(`/api/integrations/${id}`, body);
  }

  /**
   * Delete an integration.
   */
  async delete(id: string): Promise<{ deleted: boolean }> {
    return this.client.delete(`/api/integrations/${id}`);
  }

  // ===========================================================================
  // Testing and Sync
  // ===========================================================================

  /**
   * Test an integration connection.
   */
  async test(id: string): Promise<{ success: boolean; message?: string; error?: string }> {
    return this.client.post(`/api/integrations/${id}/test`);
  }

  /**
   * Trigger a sync for an integration.
   */
  async sync(id: string): Promise<IntegrationSyncStatus> {
    return this.client.post(`/api/integrations/${id}/sync`);
  }

  /**
   * Get sync status for an integration.
   */
  async getSyncStatus(id: string): Promise<IntegrationSyncStatus> {
    return this.client.get(`/api/integrations/${id}/sync`);
  }

  // ===========================================================================
  // Bot Platforms
  // ===========================================================================

  /**
   * Get Slack bot status.
   */
  async getSlackStatus(): Promise<BotStatus> {
    return this.client.request('GET', '/api/v1/bots/slack/status');
  }

  /**
   * Get Telegram bot status.
   */
  async getTelegramStatus(): Promise<BotStatus> {
    return this.client.request('GET', '/api/v1/bots/telegram/status');
  }

  /**
   * Get WhatsApp bot status.
   */
  async getWhatsAppStatus(): Promise<BotStatus> {
    return this.client.request('GET', '/api/v1/bots/whatsapp/status');
  }

  /**
   * Get Discord bot status.
   */
  async getDiscordStatus(): Promise<BotStatus> {
    return this.client.request('GET', '/api/v1/bots/discord/status');
  }

  /**
   * Get Google Chat bot status.
   */
  async getGoogleChatStatus(): Promise<BotStatus> {
    return this.client.request('GET', '/api/v1/bots/google-chat/status');
  }

  /**
   * Get email webhook status.
   */
  async getEmailStatus(): Promise<{ sendgrid: BotStatus; ses: BotStatus }> {
    return this.client.request('GET', '/api/v1/bots/email/status');
  }

  // ===========================================================================
  // Teams Integration
  // ===========================================================================

  /**
   * Get Microsoft Teams integration status.
   */
  async getTeamsStatus(): Promise<TeamsStatus> {
    return this.client.request('GET', '/api/v1/integrations/teams/status');
  }

  /**
   * Install Microsoft Teams app.
   */
  async installTeams(tenantId: string): Promise<{ installed: boolean; bot_id: string }> {
    return this.client.request('POST', '/api/integrations/teams/install', { json: { tenant_id: tenantId } });
  }

  /**
   * Handle Teams OAuth callback.
   */
  async teamsCallback(code: string, state: string): Promise<{ success: boolean }> {
    return this.client.request('POST', '/api/integrations/teams/callback', { json: { code, state } });
  }

  /**
   * Refresh Teams token.
   */
  async refreshTeamsToken(): Promise<{ refreshed: boolean }> {
    return this.client.request('POST', '/api/integrations/teams/refresh');
  }

  /**
   * Send notification to Teams channel.
   */
  async notifyTeams(channelId: string, message: string, options?: { card?: Record<string, unknown> }): Promise<{ sent: boolean; message_id: string }> {
    return this.client.request('POST', '/api/v1/integrations/teams/notify', { json: { channel_id: channelId, message, ...options } });
  }

  // ===========================================================================
  // Discord Integration
  // ===========================================================================

  /**
   * Install Discord bot.
   */
  async installDiscord(guildId: string): Promise<{ installed: boolean; bot_id: string }> {
    return this.client.request('POST', '/api/integrations/discord/install', { json: { guild_id: guildId } });
  }

  /**
   * Handle Discord OAuth callback.
   */
  async discordCallback(code: string, state: string): Promise<{ success: boolean }> {
    return this.client.request('POST', '/api/integrations/discord/callback', { json: { code, state } });
  }

  /**
   * Uninstall Discord bot.
   */
  async uninstallDiscord(guildId: string): Promise<{ uninstalled: boolean }> {
    return this.client.request('POST', '/api/integrations/discord/uninstall', { json: { guild_id: guildId } });
  }

  // ===========================================================================
  // Zapier Integration
  // ===========================================================================

  /**
   * List Zapier apps.
   */
  async listZapierApps(): Promise<{ apps: ZapierApp[] }> {
    return this.client.request('GET', '/api/v1/integrations/zapier/apps');
  }

  /**
   * Create Zapier app.
   */
  async createZapierApp(name: string, triggers?: string[], actions?: string[]): Promise<ZapierApp> {
    return this.client.request('POST', '/api/v1/integrations/zapier/apps', { json: { name, triggers, actions } });
  }

  /**
   * Delete Zapier app.
   */
  async deleteZapierApp(appId: string): Promise<{ deleted: boolean }> {
    return this.client.request('DELETE', `/api/v1/integrations/zapier/apps/${appId}`);
  }

  /**
   * List available Zapier triggers.
   */
  async listZapierTriggers(): Promise<{ triggers: ZapierTrigger[] }> {
    return this.client.request('GET', '/api/v1/integrations/zapier/triggers');
  }

  /**
   * Subscribe to Zapier trigger.
   */
  async subscribeZapierTrigger(appId: string, triggerId: string, hookUrl: string): Promise<{ subscription_id: string }> {
    return this.client.request('POST', '/api/v1/integrations/zapier/triggers', { json: { app_id: appId, trigger_id: triggerId, hook_url: hookUrl } });
  }

  /**
   * Unsubscribe from Zapier trigger.
   */
  async unsubscribeZapierTrigger(subscriptionId: string): Promise<{ unsubscribed: boolean }> {
    return this.client.request('DELETE', `/api/v1/integrations/zapier/triggers/${subscriptionId}`);
  }

  // ===========================================================================
  // Make (Integromat) Integration
  // ===========================================================================

  /**
   * List Make connections.
   */
  async listMakeConnections(): Promise<{ connections: MakeConnection[] }> {
    return this.client.request('GET', '/api/v1/integrations/make/connections');
  }

  /**
   * Create Make connection.
   */
  async createMakeConnection(name: string, credentials: Record<string, unknown>): Promise<MakeConnection> {
    return this.client.request('POST', '/api/v1/integrations/make/connections', { json: { name, credentials } });
  }

  /**
   * Delete Make connection.
   */
  async deleteMakeConnection(connectionId: string): Promise<{ deleted: boolean }> {
    return this.client.request('DELETE', `/api/v1/integrations/make/connections/${connectionId}`);
  }

  /**
   * List Make modules.
   */
  async listMakeModules(): Promise<{ modules: MakeModule[] }> {
    return this.client.request('GET', '/api/v1/integrations/make/modules');
  }

  /**
   * Register Make webhook.
   */
  async registerMakeWebhook(connectionId: string, hookUrl: string, events: string[]): Promise<{ webhook_id: string }> {
    return this.client.request('POST', '/api/v1/integrations/make/webhooks', { json: { connection_id: connectionId, hook_url: hookUrl, events } });
  }

  /**
   * Unregister Make webhook.
   */
  async unregisterMakeWebhook(webhookId: string): Promise<{ unregistered: boolean }> {
    return this.client.request('DELETE', `/api/v1/integrations/make/webhooks/${webhookId}`);
  }

  // ===========================================================================
  // n8n Integration
  // ===========================================================================

  /**
   * List n8n credentials.
   */
  async listN8nCredentials(): Promise<{ credentials: N8nCredential[] }> {
    return this.client.request('GET', '/api/v1/integrations/n8n/credentials');
  }

  /**
   * Create n8n credential.
   */
  async createN8nCredential(name: string, type: string, data: Record<string, unknown>): Promise<N8nCredential> {
    return this.client.request('POST', '/api/v1/integrations/n8n/credentials', { json: { name, type, data } });
  }

  /**
   * Delete n8n credential.
   */
  async deleteN8nCredential(credentialId: string): Promise<{ deleted: boolean }> {
    return this.client.request('DELETE', `/api/v1/integrations/n8n/credentials/${credentialId}`);
  }

  /**
   * Get n8n node definitions.
   */
  async getN8nNodes(): Promise<{ nodes: N8nNode[] }> {
    return this.client.request('GET', '/api/v1/integrations/n8n/nodes');
  }

  /**
   * Register n8n webhook.
   */
  async registerN8nWebhook(credentialId: string, hookUrl: string, events: string[]): Promise<{ webhook_id: string }> {
    return this.client.request('POST', '/api/v1/integrations/n8n/webhooks', { json: { credential_id: credentialId, hook_url: hookUrl, events } });
  }

  /**
   * Unregister n8n webhook.
   */
  async unregisterN8nWebhook(webhookId: string): Promise<{ unregistered: boolean }> {
    return this.client.request('DELETE', `/api/v1/integrations/n8n/webhooks/${webhookId}`);
  }

  // ===========================================================================
  // Integration Wizard (v2)
  // ===========================================================================

  /**
   * Start integration wizard.
   */
  async startWizard(integrationType: string): Promise<{ session_id: string; steps: string[] }> {
    return this.client.request('POST', '/api/v2/integrations/wizard', { json: { type: integrationType } });
  }

  /**
   * List wizard providers.
   */
  async listWizardProviders(): Promise<{ providers: WizardProvider[] }> {
    return this.client.request('GET', '/api/v2/integrations/wizard/providers');
  }

  /**
   * Get wizard status.
   */
  async getWizardStatus(sessionId: string): Promise<{ steps: WizardStepStatus[]; current_step: string }> {
    return this.client.request('GET', '/api/v2/integrations/wizard/status', { params: { session_id: sessionId } });
  }

  /**
   * Validate wizard step.
   */
  async validateWizardStep(sessionId: string, step: string, data: Record<string, unknown>): Promise<{ valid: boolean; errors?: string[] }> {
    return this.client.request('POST', '/api/v2/integrations/wizard/validate', { json: { session_id: sessionId, step, data } });
  }

  // ===========================================================================
  // v2 Integration Management
  // ===========================================================================

  /**
   * List integrations (v2).
   */
  async listV2(options?: { type?: string; status?: string; limit?: number; offset?: number }): Promise<{ integrations: Integration[]; total: number }> {
    return this.client.request('GET', '/api/v2/integrations', { params: options });
  }

  /**
   * Get integration by type (v2).
   */
  async getByType(type: string): Promise<Integration> {
    return this.client.request('GET', `/api/v2/integrations/${type}`);
  }

  /**
   * Get integration health by type.
   */
  async getHealth(type: string): Promise<IntegrationHealth> {
    return this.client.request('GET', `/api/v2/integrations/${type}/health`);
  }

  /**
   * Test integration by type.
   */
  async testByType(type: string): Promise<{ success: boolean; latency_ms?: number; error?: string }> {
    return this.client.request('POST', `/api/v2/integrations/${type}/test`);
  }

  /**
   * Get integration stats.
   */
  async getStats(): Promise<IntegrationStats> {
    return this.client.request('GET', '/api/v2/integrations/stats');
  }

  /**
   * Test any platform integration.
   */
  async testPlatform(platform: string): Promise<{ success: boolean; error?: string }> {
    return this.client.request('POST', `/api/v1/integrations/${platform}/test`);
  }
}
