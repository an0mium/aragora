/**
 * Audio Namespace API
 *
 * Provides audio file serving and podcast feed management.
 */

export interface AudioFileInfo {
  id: string;
  debate_id?: string;
  format: string;
  duration_seconds?: number;
  size_bytes?: number;
  created_at?: string;
}

export interface PodcastEpisode {
  id: string;
  title: string;
  description?: string;
  audio_url: string;
  debate_id?: string;
  duration_seconds?: number;
  published_at?: string;
}

interface AudioClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown> }
  ): Promise<T>;
  getBaseUrl(): string;
}

export class AudioAPI {
  constructor(private client: AudioClientInterface) {}

  /**
   * Get the direct URL for an audio file.
   */
  getAudioUrl(id: string): string {
    return `${this.client.getBaseUrl()}/audio/${id}.mp3`;
  }

  /**
   * List podcast episodes.
   */
  async listEpisodes(params?: {
    limit?: number;
    offset?: number;
  }): Promise<{ episodes: PodcastEpisode[]; total?: number }> {
    return this.client.request('GET', '/api/v1/podcast/episodes', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get the podcast RSS feed URL.
   */
  getFeedUrl(): string {
    return `${this.client.getBaseUrl()}/api/v1/podcast/feed.xml`;
  }

  /**
   * Serve audio file.
   */
  async serveAudio(audioPath: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/audio/${audioPath}`) as Promise<Record<string, unknown>>;
  }
}
