/**
 * Teams Bot Namespace API
 *
 * Provides status, configuration, and debate integration for the Microsoft Teams bot.
 * The bot enables debates to be triggered and viewed from Teams channels.
 */

/**
 * Teams bot status response.
 */
export interface TeamsBotStatus {
  platform: string;
  enabled: boolean;
  app_id_configured: boolean;
  password_configured: boolean;
  sdk_available: boolean;
  sdk_error: string | null;
}

/**
 * Teams tenant configuration
 */
export interface TeamsTenant {
  tenant_id: string;
  name?: string;
  enabled: boolean;
  created_at: string;
  last_active?: string;
  channels_count?: number;
}

/**
 * Teams OAuth installation response
 */
export interface TeamsInstallResponse {
  authorization_url: string;
  state: string;
}

/**
 * Teams OAuth callback result
 */
export interface TeamsOAuthResult {
  success: boolean;
  tenant_id?: string;
  error?: string;
}

/**
 * Teams channel information
 */
export interface TeamsChannel {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  member_count?: number;
}

/**
 * Teams debate notification settings
 */
export interface TeamsNotificationSettings {
  tenant_id: string;
  channel_id: string;
  notifications_enabled: boolean;
  notify_on_debate_start: boolean;
  notify_on_consensus: boolean;
  notify_on_completion: boolean;
}

/**
 * Teams debate message (for sending to channels)
 */
export interface TeamsDebateMessage {
  debate_id: string;
  channel_id: string;
  message_id?: string;
  sent_at?: string;
}

/**
 * Interface for the internal client methods used by TeamsAPI.
 */
interface TeamsClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, string | number | boolean | undefined>; body?: unknown }
  ): Promise<T>;
  getBaseUrl(): string;
}

/**
 * Teams Bot API namespace.
 *
 * Provides comprehensive Microsoft Teams integration including bot status,
 * OAuth installation, tenant management, and debate notifications.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Check Teams bot status
 * const status = await client.teams.getStatus();
 * if (status.enabled) {
 *   console.log('Teams bot is active');
 * }
 *
 * // Get installation URL
 * const install = await client.teams.getInstallUrl();
 * console.log(`Install URL: ${install.authorization_url}`);
 *
 * // List configured tenants
 * const tenants = await client.teams.listTenants();
 * for (const tenant of tenants) {
 *   console.log(`${tenant.name}: ${tenant.enabled ? 'active' : 'inactive'}`);
 * }
 * ```
 */
export class TeamsAPI {
  constructor(private client: TeamsClientInterface) {}

  // ===========================================================================
  // Status & Configuration
  // ===========================================================================

  /**
   * Get Teams bot status and configuration.
   *
   * Returns information about whether the bot is enabled, credentials are configured,
   * and the Bot Framework SDK is available.
   *
   * @example
   * ```typescript
   * const status = await client.teams.getStatus();
   * if (!status.sdk_available) {
   *   console.error(`SDK error: ${status.sdk_error}`);
   * }
   * ```
   */
  async getStatus(): Promise<TeamsBotStatus> {
    return this.client.request('GET', '/api/v1/bots/teams/status');
  }

  /**
   * Check if Teams integration is healthy.
   *
   * @returns true if enabled and SDK available
   */
  async isHealthy(): Promise<boolean> {
    try {
      const status = await this.getStatus();
      return status.enabled && status.sdk_available;
    } catch {
      return false;
    }
  }

  // ===========================================================================
  // OAuth & Installation
  // ===========================================================================

  /**
   * Get the Teams installation/authorization URL.
   *
   * This URL redirects users to Microsoft's OAuth consent flow.
   *
   * @param options - Installation options
   *
   * @example
   * ```typescript
   * const install = await client.teams.getInstallUrl();
   * // Redirect user to install.authorization_url
   * ```
   */
  async getInstallUrl(options?: {
    redirect_uri?: string;
    state?: string;
  }): Promise<TeamsInstallResponse> {
    return this.client.request('GET', '/api/integrations/teams/install', {
      params: options as Record<string, string | number | boolean | undefined>,
    });
  }

  /**
   * Refresh OAuth tokens for a tenant.
   *
   * @param tenantId - The Microsoft 365 tenant ID
   *
   * @example
   * ```typescript
   * await client.teams.refreshTokens('tenant-123');
   * ```
   */
  async refreshTokens(tenantId: string): Promise<{ success: boolean }> {
    return this.client.request('POST', '/api/integrations/teams/refresh', {
      body: { tenant_id: tenantId },
    });
  }

  // ===========================================================================
  // Tenant Management
  // ===========================================================================

  /**
   * List all configured Teams tenants.
   *
   * @example
   * ```typescript
   * const tenants = await client.teams.listTenants();
   * const activeTenants = tenants.filter(t => t.enabled);
   * console.log(`${activeTenants.length} active tenants`);
   * ```
   */
  async listTenants(): Promise<TeamsTenant[]> {
    const response = await this.client.request<{ tenants: TeamsTenant[] }>(
      'GET',
      '/api/v1/teams/tenants'
    );
    return response.tenants;
  }

  /**
   * Get a specific tenant's configuration.
   *
   * @param tenantId - The Microsoft 365 tenant ID
   */
  async getTenant(tenantId: string): Promise<TeamsTenant> {
    return this.client.request('GET', `/api/v1/teams/tenants/${encodeURIComponent(tenantId)}`);
  }

  /**
   * Enable or disable a tenant.
   *
   * @param tenantId - The Microsoft 365 tenant ID
   * @param enabled - Whether to enable or disable
   */
  async setTenantEnabled(tenantId: string, enabled: boolean): Promise<TeamsTenant> {
    return this.client.request('PATCH', `/api/v1/teams/tenants/${encodeURIComponent(tenantId)}`, {
      body: { enabled },
    });
  }

  // ===========================================================================
  // Channel Management
  // ===========================================================================

  /**
   * List channels for a tenant.
   *
   * @param tenantId - The Microsoft 365 tenant ID
   *
   * @example
   * ```typescript
   * const channels = await client.teams.listChannels('tenant-123');
   * for (const channel of channels) {
   *   console.log(`#${channel.name}: ${channel.member_count} members`);
   * }
   * ```
   */
  async listChannels(tenantId: string): Promise<TeamsChannel[]> {
    const response = await this.client.request<{ channels: TeamsChannel[] }>(
      'GET',
      `/api/v1/teams/tenants/${encodeURIComponent(tenantId)}/channels`
    );
    return response.channels;
  }

  // ===========================================================================
  // Notification Settings
  // ===========================================================================

  /**
   * Get notification settings for a channel.
   *
   * @param tenantId - The Microsoft 365 tenant ID
   * @param channelId - The channel ID
   */
  async getNotificationSettings(
    tenantId: string,
    channelId: string
  ): Promise<TeamsNotificationSettings> {
    return this.client.request(
      'GET',
      `/api/v1/teams/tenants/${encodeURIComponent(tenantId)}/channels/${encodeURIComponent(channelId)}/notifications`
    );
  }

  /**
   * Update notification settings for a channel.
   *
   * @param tenantId - The Microsoft 365 tenant ID
   * @param channelId - The channel ID
   * @param settings - The notification settings to update
   *
   * @example
   * ```typescript
   * await client.teams.updateNotificationSettings('tenant-123', 'channel-456', {
   *   notifications_enabled: true,
   *   notify_on_consensus: true,
   * });
   * ```
   */
  async updateNotificationSettings(
    tenantId: string,
    channelId: string,
    settings: Partial<TeamsNotificationSettings>
  ): Promise<TeamsNotificationSettings> {
    return this.client.request(
      'PATCH',
      `/api/v1/teams/tenants/${encodeURIComponent(tenantId)}/channels/${encodeURIComponent(channelId)}/notifications`,
      { body: settings }
    );
  }

  // ===========================================================================
  // Debate Integration
  // ===========================================================================

  /**
   * Send a debate to a Teams channel.
   *
   * Posts a debate card to the specified channel with voting buttons.
   *
   * @param debateId - The debate ID
   * @param channelId - The Teams channel ID
   * @param options - Additional options
   *
   * @example
   * ```typescript
   * const message = await client.teams.sendDebateToChannel('debate-123', 'channel-456');
   * console.log(`Posted to Teams: ${message.message_id}`);
   * ```
   */
  async sendDebateToChannel(
    debateId: string,
    channelId: string,
    options?: {
      tenant_id?: string;
      include_voting?: boolean;
      include_summary?: boolean;
    }
  ): Promise<TeamsDebateMessage> {
    return this.client.request('POST', '/api/v1/teams/debates/send', {
      body: {
        debate_id: debateId,
        channel_id: channelId,
        ...options,
      },
    });
  }

  /**
   * Get debates sent to Teams channels.
   *
   * @param options - Filter options
   */
  async listDebateMessages(options?: {
    tenant_id?: string;
    channel_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<TeamsDebateMessage[]> {
    const response = await this.client.request<{ messages: TeamsDebateMessage[] }>(
      'GET',
      '/api/v1/teams/debates',
      { params: options as Record<string, string | number | boolean | undefined> }
    );
    return response.messages;
  }
}
