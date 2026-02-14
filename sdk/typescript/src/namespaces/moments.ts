/**
 * Moments Namespace API
 *
 * Provides access to debate moment detection and highlights.
 * Moments are key events or turning points in debates.
 */

/**
 * A debate moment representing a significant event
 */
export interface Moment {
  id: string;
  debate_id: string;
  type: 'breakthrough' | 'conflict' | 'consensus' | 'insight' | 'question' | 'evidence';
  timestamp: string;
  round: number;
  agent: string;
  content: string;
  significance_score: number;
  metadata?: Record<string, unknown>;
}

/**
 * Summary of moments in a debate
 */
export interface MomentsSummary {
  debate_id: string;
  total_moments: number;
  by_type: Record<string, number>;
  highlights: Moment[];
  key_turning_points: Moment[];
}

/**
 * Timeline of moments
 */
export interface MomentsTimeline {
  debate_id: string;
  moments: Moment[];
  duration_seconds: number;
}

/**
 * Trending moments across debates
 */
export interface TrendingMoments {
  period: string;
  moments: Moment[];
  top_debates: string[];
}

/**
 * Internal client interface
 */
interface MomentsClientInterface {
  get<T>(path: string): Promise<T>;
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Moments API namespace.
 *
 * Provides methods for accessing debate moments and highlights:
 * - Get moment summaries for debates
 * - View moment timelines
 * - Find trending moments
 * - Filter moments by type
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Get summary for a debate
 * const summary = await client.moments.getSummary('debate-123');
 *
 * // Get timeline
 * const timeline = await client.moments.getTimeline('debate-123');
 *
 * // Get trending moments
 * const trending = await client.moments.getTrending();
 * ```
 */
export class MomentsAPI {
  constructor(private client: MomentsClientInterface) {}

  /**
   * Get moment summary for debates.
   */
  async getSummary(debateId?: string): Promise<MomentsSummary> {
    return this.client.request('GET', '/api/v1/moments/summary', {
      params: debateId ? { debate_id: debateId } : undefined,
    });
  }

  /**
   * Get moment timeline for a debate.
   */
  async getTimeline(debateId?: string): Promise<MomentsTimeline> {
    return this.client.request('GET', '/api/v1/moments/timeline', {
      params: debateId ? { debate_id: debateId } : undefined,
    });
  }

  /**
   * Get trending moments across debates.
   */
  async getTrending(options?: { period?: string; limit?: number }): Promise<TrendingMoments> {
    return this.client.request('GET', '/api/v1/moments/trending', {
      params: options as Record<string, unknown> | undefined,
    });
  }

  /**
   * Get moments filtered by type.
   */
  async getByType(
    type: 'breakthrough' | 'conflict' | 'consensus' | 'insight' | 'question' | 'evidence',
    options?: { limit?: number; offset?: number }
  ): Promise<{ moments: Moment[]; total: number }> {
    return this.client.request('GET', `/api/v1/moments/by-type/${type}`, {
      params: options as Record<string, unknown> | undefined,
    });
  }
}
