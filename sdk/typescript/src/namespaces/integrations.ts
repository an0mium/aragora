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
 * Zapier app summary (from list)
 */
export interface ZapierAppSummary {
  id: string;
  workspace_id: string;
  created_at: string;
  active: boolean;
  trigger_count: number;
  action_count: number;
}

/**
 * Zapier app with credentials (from create)
 */
export interface ZapierApp {
  id: string;
  workspace_id: string;
  api_key: string;
  api_secret: string;
  created_at: string;
}

/**
 * Zapier trigger subscription
 */
export interface ZapierTriggerSubscription {
  id: string;
  trigger_type: string;
  webhook_url: string;
  created_at: string;
}

/**
 * Zapier trigger type definition
 */
export interface ZapierTrigger {
  id: string;
  name: string;
  description: string;
  sample_payload: Record<string, unknown>;
}

/**
 * Create Zapier app request
 */
export interface CreateZapierAppRequest {
  workspace_id: string;
}

/**
 * Subscribe to Zapier trigger request
 */
export interface SubscribeZapierTriggerRequest {
  app_id: string;
  trigger_type: string;
  webhook_url: string;
  workspace_id?: string;
  debate_tags?: string[];
  min_confidence?: number;
}

/**
 * Make connection summary (from list)
 */
export interface MakeConnectionSummary {
  id: string;
  workspace_id: string;
  created_at: string;
  active: boolean;
  total_operations: number;
  webhooks_count: number;
}

/**
 * Make connection with credentials (from create)
 */
export interface MakeConnection {
  id: string;
  workspace_id: string;
  api_key: string;
  created_at: string;
}

/**
 * Make webhook
 */
export interface MakeWebhook {
  id: string;
  module_type: string;
  webhook_url: string;
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
 * Create Make connection request
 */
export interface CreateMakeConnectionRequest {
  workspace_id: string;
}

/**
 * Register Make webhook request
 */
export interface RegisterMakeWebhookRequest {
  connection_id: string;
  module_type: string;
  webhook_url: string;
  workspace_id?: string;
  event_filter?: Record<string, unknown>;
}

/**
 * n8n credential summary (from list)
 */
export interface N8nCredentialSummary {
  id: string;
  workspace_id: string;
  api_url: string;
  created_at: string;
  active: boolean;
  operation_count: number;
  webhooks_count: number;
}

/**
 * n8n credential with API key (from create)
 */
export interface N8nCredential {
  id: string;
  workspace_id: string;
  api_key: string;
  api_url: string;
  created_at: string;
}

/**
 * n8n webhook
 */
export interface N8nWebhook {
  id: string;
  webhook_path: string;
  events: string[];
  created_at: string;
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
 * n8n nodes response (includes node, trigger, credential definitions)
 */
export interface N8nNodesResponse {
  node: N8nNode;
  trigger: N8nNode;
  credential: Record<string, unknown>;
  events: Record<string, unknown>;
}

/**
 * Create n8n credential request
 */
export interface CreateN8nCredentialRequest {
  workspace_id: string;
  api_url?: string;
}

/**
 * Register n8n webhook request
 */
export interface RegisterN8nWebhookRequest {
  credential_id: string;
  events: string[];
  workflow_id?: string;
  node_id?: string;
  workspace_id?: string;
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
   * @route GET /api/v1/integrations/available
   */
  async listAvailable(): Promise<{ integrations: AvailableIntegration[] }> {
    return this.client.request('GET', '/api/v1/integrations/available') as Promise<{ integrations: AvailableIntegration[] }>;
  }

  /**
   * Get integration status summary.
   * @route GET /api/v1/integrations/status
   */
  async getStatus(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/integrations/status') as Promise<Record<string, unknown>>;
  }

  /**
   * Get global integration configuration.
   * @route GET /api/v1/integrations/config
   */
  async getConfig(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/integrations/config') as Promise<Record<string, unknown>>;
  }

  /**
   * Get configuration schema for an integration type.
   * @route GET /api/v1/integrations/config/{type}
   */
  async getConfigSchema(type: string): Promise<IntegrationConfigSchema> {
    return this.client.request('GET', `/api/v1/integrations/config/${type}`) as Promise<IntegrationConfigSchema>;
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
   *
   * @param workspaceId - Optional workspace ID to filter by
   * @returns List of Zapier apps
   */
  async listZapierApps(workspaceId?: string): Promise<{ apps: ZapierAppSummary[]; count: number }> {
    const params = workspaceId ? { params: { workspace_id: workspaceId } } : undefined;
    return this.client.request('GET', '/api/v1/integrations/zapier/apps', params);
  }

  /**
   * Create Zapier app.
   *
   * @param body - Create request with workspace_id
   * @returns Created app with API credentials (save these, they won't be shown again)
   */
  async createZapierApp(body: CreateZapierAppRequest): Promise<{ app: ZapierApp; message: string }> {
    return this.client.request('POST', '/api/v1/integrations/zapier/apps', { json: { ...body } });
  }

  /**
   * Delete Zapier app.
   *
   * @param appId - The app ID to delete
   * @returns Deletion confirmation
   */
  async deleteZapierApp(appId: string): Promise<{ deleted: boolean; app_id: string }> {
    return this.client.request('DELETE', `/api/v1/integrations/zapier/apps/${appId}`);
  }

  /**
   * List available Zapier triggers and actions.
   *
   * @returns Available trigger and action types
   */
  async listZapierTriggerTypes(): Promise<{ triggers: Record<string, unknown>; actions: Record<string, unknown> }> {
    return this.client.request('GET', '/api/v1/integrations/zapier/triggers');
  }

  /**
   * Subscribe to Zapier trigger.
   *
   * @param body - Subscribe request with app_id, trigger_type, webhook_url
   * @returns Created trigger subscription
   */
  async subscribeZapierTrigger(body: SubscribeZapierTriggerRequest): Promise<{ trigger: ZapierTriggerSubscription }> {
    return this.client.request('POST', '/api/v1/integrations/zapier/triggers', { json: { ...body } });
  }

  /**
   * Unsubscribe from Zapier trigger.
   *
   * @param triggerId - The trigger subscription ID
   * @param appId - The app ID the trigger belongs to
   * @returns Deletion confirmation
   */
  async unsubscribeZapierTrigger(triggerId: string, appId: string): Promise<{ deleted: boolean; trigger_id: string }> {
    return this.client.request('DELETE', `/api/v1/integrations/zapier/triggers/${triggerId}`, { params: { app_id: appId } });
  }

  // ===========================================================================
  // Make (Integromat) Integration
  // ===========================================================================

  /**
   * List Make connections.
   *
   * @param workspaceId - Optional workspace ID to filter by
   * @returns List of Make connections
   */
  async listMakeConnections(workspaceId?: string): Promise<{ connections: MakeConnectionSummary[]; count: number }> {
    const params = workspaceId ? { params: { workspace_id: workspaceId } } : undefined;
    return this.client.request('GET', '/api/v1/integrations/make/connections', params);
  }

  /**
   * Create Make connection.
   *
   * @param body - Create request with workspace_id
   * @returns Created connection with API key (save this, it won't be shown again)
   */
  async createMakeConnection(body: CreateMakeConnectionRequest): Promise<{ connection: MakeConnection; message: string }> {
    return this.client.request('POST', '/api/v1/integrations/make/connections', { json: { ...body } });
  }

  /**
   * Delete Make connection.
   *
   * @param connectionId - The connection ID to delete
   * @returns Deletion confirmation
   */
  async deleteMakeConnection(connectionId: string): Promise<{ deleted: boolean; connection_id: string }> {
    return this.client.request('DELETE', `/api/v1/integrations/make/connections/${connectionId}`);
  }

  /**
   * List Make modules.
   *
   * @returns Available module types
   */
  async listMakeModules(): Promise<{ modules: Record<string, unknown> }> {
    return this.client.request('GET', '/api/v1/integrations/make/modules');
  }

  /**
   * Register Make webhook.
   *
   * @param body - Register request with connection_id, module_type, webhook_url
   * @returns Created webhook
   */
  async registerMakeWebhook(body: RegisterMakeWebhookRequest): Promise<{ webhook: MakeWebhook }> {
    return this.client.request('POST', '/api/v1/integrations/make/webhooks', { json: { ...body } });
  }

  /**
   * Unregister Make webhook.
   *
   * @param webhookId - The webhook ID to unregister
   * @param connectionId - The connection ID the webhook belongs to
   * @returns Deletion confirmation
   */
  async unregisterMakeWebhook(webhookId: string, connectionId: string): Promise<{ deleted: boolean; webhook_id: string }> {
    return this.client.request('DELETE', `/api/v1/integrations/make/webhooks/${webhookId}`, { params: { connection_id: connectionId } });
  }

  // ===========================================================================
  // n8n Integration
  // ===========================================================================

  /**
   * List n8n credentials.
   *
   * @param workspaceId - Optional workspace ID to filter by
   * @returns List of n8n credentials
   */
  async listN8nCredentials(workspaceId?: string): Promise<{ credentials: N8nCredentialSummary[]; count: number }> {
    const params = workspaceId ? { params: { workspace_id: workspaceId } } : undefined;
    return this.client.request('GET', '/api/v1/integrations/n8n/credentials', params);
  }

  /**
   * Create n8n credential.
   *
   * @param body - Create request with workspace_id and optional api_url
   * @returns Created credential with API key (save this, it won't be shown again)
   */
  async createN8nCredential(body: CreateN8nCredentialRequest): Promise<{ credential: N8nCredential; message: string }> {
    return this.client.request('POST', '/api/v1/integrations/n8n/credentials', { json: { ...body } });
  }

  /**
   * Delete n8n credential.
   *
   * @param credentialId - The credential ID to delete
   * @returns Deletion confirmation
   */
  async deleteN8nCredential(credentialId: string): Promise<{ deleted: boolean; credential_id: string }> {
    return this.client.request('DELETE', `/api/v1/integrations/n8n/credentials/${credentialId}`);
  }

  /**
   * Get n8n node definitions.
   *
   * @returns Node, trigger, and credential definitions plus event types
   */
  async getN8nNodes(): Promise<N8nNodesResponse> {
    return this.client.request('GET', '/api/v1/integrations/n8n/nodes');
  }

  /**
   * Register n8n webhook.
   *
   * @param body - Register request with credential_id and events
   * @returns Created webhook with webhook_path
   */
  async registerN8nWebhook(body: RegisterN8nWebhookRequest): Promise<{ webhook: N8nWebhook }> {
    return this.client.request('POST', '/api/v1/integrations/n8n/webhooks', { json: { ...body } });
  }

  /**
   * Unregister n8n webhook.
   *
   * @param webhookId - The webhook ID to unregister
   * @param credentialId - The credential ID the webhook belongs to
   * @returns Deletion confirmation
   */
  async unregisterN8nWebhook(webhookId: string, credentialId: string): Promise<{ deleted: boolean; webhook_id: string }> {
    return this.client.request('DELETE', `/api/v1/integrations/n8n/webhooks/${webhookId}`, { params: { credential_id: credentialId } });
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
