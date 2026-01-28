/**
 * Teams Bot Namespace API
 *
 * Provides status and configuration for the Microsoft Teams bot integration.
 * The bot enables debates to be triggered and viewed from Teams channels.
 */

/**
 * Teams bot status response.
 */
export interface TeamsBotStatus {
  enabled: boolean;
  app_id_configured: boolean;
  password_configured: boolean;
  sdk_available: boolean;
  sdk_error: string | null;
}

/**
 * Interface for the internal client methods used by TeamsAPI.
 */
interface TeamsClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Teams Bot API namespace.
 *
 * Provides status and configuration for Microsoft Teams integration.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Check Teams bot status
 * const status = await client.teams.getStatus();
 * if (status.enabled) {
 *   console.log('Teams bot is active');
 * } else if (!status.sdk_available) {
 *   console.log(`SDK error: ${status.sdk_error}`);
 * }
 * ```
 */
export class TeamsAPI {
  constructor(private client: TeamsClientInterface) {}

  /**
   * Get Teams bot status and configuration.
   */
  async getStatus(): Promise<TeamsBotStatus> {
    return this.client.request('GET', '/api/v2/teams/status');
  }
}
