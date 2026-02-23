/**
 * Channels Namespace API
 *
 * Provides methods for channel health monitoring.
 */

interface ChannelsClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class ChannelsAPI {
  constructor(private client: ChannelsClientInterface) {}

  /** Get overall channel health status. */
  async getHealth(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/channels/health');
  }

  /** Get health status for a specific channel. */
  async getChannelHealth(channelId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/channels/${channelId}/health`);
  }
}
