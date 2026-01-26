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
 * Interface for the internal client used by IntegrationsAPI.
 */
interface IntegrationsClientInterface {
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
  put<T>(path: string, body?: unknown): Promise<T>;
  delete<T>(path: string): Promise<T>;
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
    return this.client.listAvailableIntegrations();
  }

  /**
   * Get configuration schema for an integration type.
   */
  async getConfigSchema(type: string): Promise<IntegrationConfigSchema> {
    return this.client.getIntegrationConfig(type);
  }

  // ===========================================================================
  // Integration Management
  // ===========================================================================

  /**
   * List all configured integrations.
   */
  async list(): Promise<{ integrations: Integration[]; total: number }> {
    return this.client.listIntegrations();
  }

  /**
   * Get a specific integration by ID.
   */
  async get(id: string): Promise<Integration> {
    return this.client.getIntegration(id);
  }

  /**
   * Create a new integration.
   */
  async create(body: CreateIntegrationRequest): Promise<Integration> {
    return this.client.createIntegration(body);
  }

  /**
   * Update an existing integration.
   */
  async update(id: string, body: UpdateIntegrationRequest): Promise<Integration> {
    return this.client.updateIntegration(id, body);
  }

  /**
   * Delete an integration.
   */
  async delete(id: string): Promise<{ deleted: boolean }> {
    return this.client.deleteIntegration(id);
  }

  // ===========================================================================
  // Testing and Sync
  // ===========================================================================

  /**
   * Test an integration connection.
   */
  async test(id: string): Promise<{ success: boolean; message?: string; error?: string }> {
    return this.client.testIntegration(id);
  }

  /**
   * Trigger a sync for an integration.
   */
  async sync(id: string): Promise<IntegrationSyncStatus> {
    return this.client.syncIntegration(id);
  }

  /**
   * Get sync status for an integration.
   */
  async getSyncStatus(id: string): Promise<IntegrationSyncStatus> {
    return this.client.getIntegrationSyncStatus(id);
  }
}
