/**
 * Social Namespace API
 *
 * Provides social media integration for publishing debate results.
 */

export interface YouTubeAuthResponse {
  auth_url: string;
  state: string;
}

export interface YouTubeCallbackParams {
  code: string;
  state: string;
}

export interface YouTubeStatus {
  connected: boolean;
  channel_id?: string;
  channel_name?: string;
  expires_at?: string;
}

export interface PublishRequest {
  title?: string;
  description?: string;
  tags?: string[];
  visibility?: 'public' | 'unlisted' | 'private';
}

export interface PublishResult {
  success: boolean;
  platform: string;
  url?: string;
  post_id?: string;
  error?: string;
}

interface SocialClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown }
  ): Promise<T>;
}

export class SocialAPI {
  constructor(private client: SocialClientInterface) {}

  /**
   * Get YouTube OAuth authorization URL.
   */
  async getYouTubeAuthUrl(): Promise<YouTubeAuthResponse> {
    return this.client.request('GET', '/api/youtube/auth');
  }

  /**
   * Handle YouTube OAuth callback.
   */
  async handleYouTubeCallback(params: YouTubeCallbackParams): Promise<{ success: boolean }> {
    return this.client.request('GET', '/api/youtube/callback', {
      params: params as unknown as Record<string, unknown>,
    });
  }

  /**
   * Get YouTube connector status.
   */
  async getYouTubeStatus(): Promise<YouTubeStatus> {
    return this.client.request('GET', '/api/youtube/status');
  }

  /**
   * Publish a debate to Twitter/X.
   */
  async publishToTwitter(debateId: string, body?: PublishRequest): Promise<PublishResult> {
    return this.client.request('POST', `/api/debates/${debateId}/publish/twitter`, { body });
  }

  /**
   * Publish a debate to YouTube.
   */
  async publishToYouTube(debateId: string, body?: PublishRequest): Promise<PublishResult> {
    return this.client.request('POST', `/api/debates/${debateId}/publish/youtube`, { body });
  }
}
