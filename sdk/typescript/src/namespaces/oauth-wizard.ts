/**
 * OAuth Wizard Namespace API
 *
 * Unified API for discovering, configuring, and managing OAuth provider
 * integrations through a wizard interface designed for SME onboarding.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get wizard configuration
 * const config = await client.oauthWizard.getConfig();
 *
 * // List all available providers
 * const providers = await client.oauthWizard.listProviders({ category: 'communication' });
 *
 * // Check integration status
 * const status = await client.oauthWizard.getStatus();
 *
 * // Validate configuration before connecting
 * const validation = await client.oauthWizard.validateConfig('slack');
 * ```
 */

/**
 * OAuth provider category.
 */
export type ProviderCategory =
  | 'communication'
  | 'development'
  | 'storage'
  | 'crm'
  | 'productivity'
  | 'analytics';

/**
 * Provider configuration status.
 */
export type ConfigStatus = 'configured' | 'partial' | 'not_configured' | 'error';

/**
 * OAuth provider information.
 */
export interface OAuthProvider {
  id: string;
  name: string;
  description: string;
  category: ProviderCategory;
  setup_time_minutes: number;
  features: string[];
  required_env_vars: string[];
  optional_env_vars: string[];
  oauth_scopes: string[];
  install_url: string | null;
  docs_url: string;
  icon_url?: string;
}

/**
 * Provider configuration status.
 */
export interface ProviderStatus {
  provider_id: string;
  name: string;
  status: ConfigStatus;
  configured_at?: string;
  missing_vars: string[];
  optional_missing_vars: string[];
  connected: boolean;
  last_sync_at?: string;
  error?: string;
}

/**
 * Wizard configuration.
 */
export interface WizardConfig {
  available_providers: string[];
  configured_providers: string[];
  recommended_order: string[];
  total_setup_time_minutes: number;
  completion_percent: number;
}

/**
 * Configuration validation result.
 */
export interface ValidationResult {
  provider_id: string;
  valid: boolean;
  missing_required: string[];
  missing_optional: string[];
  warnings: string[];
  errors: string[];
  can_proceed: boolean;
  suggested_actions: string[];
}

/**
 * Preflight check result.
 */
export interface PreflightCheck {
  provider_id: string;
  checks: Array<{
    name: string;
    status: 'passed' | 'failed' | 'warning' | 'skipped';
    message: string;
    required: boolean;
  }>;
  can_connect: boolean;
  estimated_connect_time_seconds: number;
}

/**
 * Integration status summary.
 */
export interface IntegrationStatusSummary {
  total_providers: number;
  configured: number;
  connected: number;
  errors: number;
  providers: ProviderStatus[];
  health_score: number;
  last_checked_at: string;
}

/**
 * Client interface for oauth wizard operations.
 */
interface OAuthWizardClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * OAuth Wizard API for managing integration configuration.
 */
export class OAuthWizardAPI {
  constructor(private client: OAuthWizardClientInterface) {}

  /**
   * Get wizard configuration.
   *
   * Returns the current state of the integration wizard including
   * available providers, configured providers, and recommended setup order.
   */
  async getConfig(): Promise<WizardConfig> {
    return this.client.request('GET', '/api/v2/integrations/wizard');
  }

  /**
   * List all available OAuth providers.
   *
   * @param options - Filter options
   * @param options.category - Filter by provider category
   * @param options.configured - Filter by configuration status
   */
  async listProviders(options?: {
    category?: ProviderCategory;
    configured?: boolean;
  }): Promise<{ providers: OAuthProvider[] }> {
    const params: Record<string, unknown> = {};
    if (options?.category) {
      params.category = options.category;
    }
    if (options?.configured !== undefined) {
      params.configured = options.configured;
    }

    return this.client.request('GET', '/api/v2/integrations/wizard/providers', { params });
  }

  /**
   * Get a specific provider's details.
   *
   * @param providerId - Provider identifier (e.g., 'slack', 'github')
   */
  async getProvider(providerId: string): Promise<OAuthProvider> {
    return this.client.request('GET', `/api/v2/integrations/wizard/providers/${providerId}`);
  }

  /**
   * Get integration status for all providers.
   *
   * Returns the current connection and configuration status
   * for all available integrations.
   */
  async getStatus(): Promise<IntegrationStatusSummary> {
    return this.client.request('GET', '/api/v2/integrations/wizard/status');
  }

  /**
   * Get status for a specific provider.
   *
   * @param providerId - Provider identifier
   */
  async getProviderStatus(providerId: string): Promise<ProviderStatus> {
    return this.client.request('GET', `/api/v2/integrations/wizard/status/${providerId}`);
  }

  /**
   * Validate provider configuration before connecting.
   *
   * Checks if all required environment variables are set
   * and validates the configuration format.
   *
   * @param providerId - Provider identifier
   */
  async validateConfig(providerId: string): Promise<ValidationResult> {
    return this.client.request('POST', '/api/v2/integrations/wizard/validate', {
      json: { provider_id: providerId },
    });
  }

  /**
   * Run preflight checks before connecting to a provider.
   *
   * Performs connection tests, scope validation, and other
   * checks to ensure successful integration.
   *
   * @param providerId - Provider identifier
   */
  async runPreflightChecks(providerId: string): Promise<PreflightCheck> {
    return this.client.request('POST', '/api/v2/integrations/wizard/preflight', {
      json: { provider_id: providerId },
    });
  }

  /**
   * Get the install URL for a provider.
   *
   * Returns the OAuth authorization URL to initiate the connection flow.
   *
   * @param providerId - Provider identifier
   * @param options - Install options
   * @param options.redirect_uri - Custom redirect URI
   * @param options.scopes - Override default scopes
   * @param options.state - Custom state parameter
   */
  async getInstallUrl(
    providerId: string,
    options?: {
      redirect_uri?: string;
      scopes?: string[];
      state?: string;
    }
  ): Promise<{ install_url: string; state: string; expires_at: string }> {
    return this.client.request('POST', `/api/v2/integrations/wizard/providers/${providerId}/install`, {
      json: options ?? {},
    });
  }

  /**
   * Disconnect a provider integration.
   *
   * Revokes OAuth tokens and removes the integration.
   *
   * @param providerId - Provider identifier
   */
  async disconnect(providerId: string): Promise<{ disconnected: boolean; message: string }> {
    return this.client.request('DELETE', `/api/v2/integrations/wizard/providers/${providerId}`);
  }

  /**
   * Refresh a provider's OAuth tokens.
   *
   * @param providerId - Provider identifier
   */
  async refreshTokens(providerId: string): Promise<{
    refreshed: boolean;
    expires_at: string;
  }> {
    return this.client.request('POST', `/api/v2/integrations/wizard/providers/${providerId}/refresh`);
  }

  /**
   * Get recommended providers based on current configuration.
   *
   * Returns providers that would benefit the user based on
   * their current setup and usage patterns.
   */
  async getRecommendations(): Promise<{
    recommendations: Array<{
      provider_id: string;
      reason: string;
      priority: 'high' | 'medium' | 'low';
    }>;
  }> {
    return this.client.request('GET', '/api/v2/integrations/wizard/recommendations');
  }

  /**
   * Test connection to a specific provider.
   *
   * @param providerId - Provider identifier (e.g., 'slack', 'github')
   */
  async testConnection(providerId: string): Promise<{ success: boolean; error?: string }> {
    return this.client.request('POST', `/api/v2/integrations/wizard/${providerId}/test`);
  }

  /**
   * List workspaces/tenants for a provider.
   *
   * @param providerId - Provider identifier
   */
  async listWorkspaces(providerId: string): Promise<{ workspaces: Array<{ id: string; name: string }> }> {
    return this.client.request('GET', `/api/v2/integrations/wizard/${providerId}/workspaces`);
  }

  /**
   * Disconnect a provider integration using the wizard endpoint.
   *
   * @param providerId - Provider identifier
   */
  async disconnectProvider(providerId: string): Promise<{ disconnected: boolean; message: string }> {
    return this.client.request('POST', `/api/v2/integrations/wizard/${providerId}/disconnect`);
  }
}
