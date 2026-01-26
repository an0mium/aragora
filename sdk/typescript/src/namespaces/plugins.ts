/**
 * Plugins Namespace API
 *
 * Provides a namespaced interface for plugin management.
 * Essential for extensibility and marketplace integrations.
 */

/**
 * Plugin definition
 */
export interface Plugin {
  id: string;
  name: string;
  description: string;
  version: string;
  author: string;
  category: string;
  tags: string[];
  icon_url?: string;
  documentation_url?: string;
  installed: boolean;
  enabled: boolean;
  config?: Record<string, unknown>;
  permissions: string[];
  created_at: string;
  updated_at: string;
}

/**
 * Plugin marketplace listing
 */
export interface PluginListing {
  id: string;
  name: string;
  description: string;
  version: string;
  author: string;
  category: string;
  tags: string[];
  icon_url?: string;
  rating: number;
  install_count: number;
  verified: boolean;
  featured: boolean;
}

/**
 * Plugin submission for marketplace review
 */
export interface PluginSubmission {
  id: string;
  plugin_name: string;
  version: string;
  status: 'pending' | 'reviewing' | 'approved' | 'rejected';
  submitted_at: string;
  reviewed_at?: string;
  reviewer_notes?: string;
}

/**
 * Plugin configuration schema
 */
export interface PluginConfigSchema {
  properties: Record<string, {
    type: string;
    description: string;
    required?: boolean;
    default?: unknown;
  }>;
}

/**
 * Install plugin request
 */
export interface InstallPluginRequest {
  config?: Record<string, unknown>;
}

/**
 * Plugin query request
 */
export interface PluginQueryRequest {
  query: string;
  category?: string;
  tags?: string[];
  limit?: number;
  offset?: number;
}

/**
 * Plugin validation request
 */
export interface PluginValidateRequest {
  manifest: Record<string, unknown>;
  code?: string;
}

/**
 * Interface for the internal client used by PluginsAPI.
 */
interface PluginsClientInterface {
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
  put<T>(path: string, body?: unknown): Promise<T>;
  delete<T>(path: string): Promise<T>;
}

/**
 * Plugins API namespace.
 *
 * Provides methods for managing plugins:
 * - Browse and search marketplace
 * - Install and configure plugins
 * - Run plugin functionality
 * - Submit plugins for review
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Browse marketplace
 * const { plugins, featured } = await client.plugins.getMarketplace();
 *
 * // Search plugins
 * const results = await client.plugins.query({ query: 'slack', category: 'integrations' });
 *
 * // Install a plugin
 * const plugin = await client.plugins.install('slack-connector', { config: { workspace: 'my-team' } });
 *
 * // Run plugin functionality
 * const result = await client.plugins.run('slack-connector', { action: 'send', channel: '#general' });
 * ```
 */
export class PluginsAPI {
  constructor(private client: PluginsClientInterface) {}

  // ===========================================================================
  // Plugin Discovery
  // ===========================================================================

  /**
   * List all available plugins.
   */
  async list(): Promise<{ plugins: Plugin[]; total: number }> {
    return this.client.get('/api/plugins');
  }

  /**
   * Get a specific plugin by name.
   */
  async get(name: string): Promise<Plugin> {
    return this.client.get(`/api/plugins/${name}`);
  }

  /**
   * Get marketplace listings with categories and featured plugins.
   */
  async getMarketplace(): Promise<{ plugins: PluginListing[]; categories: string[]; featured: PluginListing[] }> {
    return this.client.get('/api/plugins/marketplace');
  }

  /**
   * Query plugins with search and filters.
   */
  async query(body: PluginQueryRequest): Promise<{ plugins: PluginListing[]; total: number }> {
    return this.client.post('/api/plugins/query', body);
  }

  // ===========================================================================
  // Plugin Management
  // ===========================================================================

  /**
   * List installed plugins.
   */
  async listInstalled(): Promise<{ plugins: Plugin[]; total: number }> {
    return this.client.get('/api/plugins/installed');
  }

  /**
   * Install a plugin.
   */
  async install(name: string, body?: InstallPluginRequest): Promise<Plugin> {
    return this.client.post(`/api/plugins/${name}/install`, body);
  }

  /**
   * Uninstall a plugin.
   */
  async uninstall(name: string): Promise<{ uninstalled: boolean }> {
    return this.client.delete(`/api/plugins/${name}`);
  }

  /**
   * Run a plugin with the given parameters.
   */
  async run(name: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.post(`/api/plugins/${name}/run`, body);
  }

  // ===========================================================================
  // Plugin Submission
  // ===========================================================================

  /**
   * Submit a plugin for marketplace review.
   */
  async submit(body: Record<string, unknown>): Promise<PluginSubmission> {
    return this.client.post('/api/plugins/submit', body);
  }

  /**
   * List your plugin submissions.
   */
  async listSubmissions(): Promise<{ submissions: PluginSubmission[] }> {
    return this.client.get('/api/plugins/submissions');
  }

  /**
   * Validate a plugin manifest before submission.
   */
  async validate(body: PluginValidateRequest): Promise<{ valid: boolean; errors?: string[] }> {
    return this.client.post('/api/plugins/validate', body);
  }
}
