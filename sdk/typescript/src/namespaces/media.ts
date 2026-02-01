/**
 * Media Namespace API
 *
 * Provides access to media assets including audio files and podcast episodes.
 */

export interface AudioFile {
  id: string;
  debate_id?: string;
  url: string;
  duration_seconds?: number;
  format: string;
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
  metadata?: Record<string, unknown>;
}

export interface PodcastFeed {
  feed_url: string;
  title: string;
  description?: string;
  episodes: PodcastEpisode[];
  total: number;
}

interface MediaClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown> }
  ): Promise<T>;
}

export class MediaAPI {
  constructor(private client: MediaClientInterface) {}

  /**
   * Get audio file metadata by ID.
   */
  async getAudio(id: string): Promise<AudioFile> {
    return this.client.request('GET', `/api/v1/media/audio/${id}`);
  }

  /**
   * Get the audio file URL for a debate.
   */
  getAudioUrl(id: string): string {
    return `/audio/${id}.mp3`;
  }

  /**
   * List podcast episodes.
   */
  async listPodcastEpisodes(params?: {
    limit?: number;
    offset?: number;
  }): Promise<{ episodes: PodcastEpisode[]; total: number }> {
    return this.client.request('GET', '/api/v1/podcast/episodes', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get the podcast RSS feed URL.
   */
  getFeedUrl(): string {
    return '/api/v1/podcast/feed.xml';
  }
}
