/**
 * YouTube Namespace API
 *
 * Provides endpoints for publishing debates to YouTube and managing
 * YouTube OAuth authentication.
 */

import type { AragoraClient } from '../client';

/**
 * YouTube publish request
 */
export interface YouTubePublishRequest {
  title?: string;
  description?: string;
  tags?: string[];
  privacy?: 'public' | 'unlisted' | 'private';
  category_id?: string;
  playlist_id?: string;
  notify_subscribers?: boolean;
}

/**
 * YouTube publish response
 */
export interface YouTubePublishResponse {
  video_id: string;
  video_url: string;
  status: 'processing' | 'published' | 'failed';
  title: string;
  thumbnail_url?: string;
  publish_time?: string;
  error?: string;
}

/**
 * YouTube auth status
 */
export interface YouTubeAuthStatus {
  authenticated: boolean;
  channel_id?: string;
  channel_name?: string;
  expires_at?: string;
  scopes?: string[];
}

/**
 * YouTube OAuth URL response
 */
export interface YouTubeAuthUrl {
  auth_url: string;
  state: string;
}

/**
 * YouTube namespace for debate video publishing.
 *
 * @example
 * ```typescript
 * // Check if YouTube is connected
 * const auth = await client.youtube.getAuthStatus();
 * if (!auth.authenticated) {
 *   const { auth_url } = await client.youtube.getAuthUrl();
 *   // Redirect user to auth_url
 * }
 *
 * // Publish a debate to YouTube
 * const result = await client.youtube.publishDebate('debate-123', {
 *   title: 'AI Agents Debate: Climate Policy',
 *   privacy: 'public'
 * });
 * console.log(`Video URL: ${result.video_url}`);
 * ```
 */
export class YouTubeNamespace {
  constructor(private client: AragoraClient) {}

  /**
   * Get YouTube OAuth authorization URL.
   *
   * Redirect the user to this URL to authorize YouTube access.
   *
   * @param redirectUri - Optional custom redirect URI
   */
  async getAuthUrl(redirectUri?: string): Promise<YouTubeAuthUrl> {
    return this.client.request<YouTubeAuthUrl>('GET', '/api/v1/youtube/auth', {
      params: redirectUri ? { redirect_uri: redirectUri } : undefined,
    });
  }

  /**
   * Get current YouTube authentication status.
   */
  async getAuthStatus(): Promise<YouTubeAuthStatus> {
    return this.client.request<YouTubeAuthStatus>('GET', '/api/v1/youtube/auth/status');
  }

  /**
   * Publish a debate as a YouTube video.
   *
   * Generates a video from the debate transcript and uploads it to YouTube.
   *
   * @param debateId - The debate to publish
   * @param options - Publishing options (title, description, privacy, etc.)
   */
  async publishDebate(
    debateId: string,
    options?: YouTubePublishRequest
  ): Promise<YouTubePublishResponse> {
    return this.client.request<YouTubePublishResponse>(
      'POST',
      `/api/v1/debates/${encodeURIComponent(debateId)}/publish/youtube`,
      { body: options }
    );
  }

  /**
   * Get the status of a YouTube publish operation.
   *
   * @param debateId - The debate ID
   */
  async getPublishStatus(debateId: string): Promise<YouTubePublishResponse> {
    return this.client.request<YouTubePublishResponse>(
      'GET',
      `/api/v1/debates/${encodeURIComponent(debateId)}/publish/youtube/status`
    );
  }

  /**
   * Revoke YouTube OAuth access.
   */
  async revokeAuth(): Promise<{ success: boolean }> {
    return this.client.request<{ success: boolean }>('DELETE', '/api/v1/youtube/auth');
  }
}
