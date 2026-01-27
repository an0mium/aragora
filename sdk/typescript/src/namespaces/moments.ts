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
    const params = debateId ? `?debate_id=${debateId}` : '';
    return this.client.get(`/api/v1/moments/summary${params}`);
  }

  /**
   * Get moment timeline for a debate.
   */
  async getTimeline(debateId?: string): Promise<MomentsTimeline> {
    const params = debateId ? `?debate_id=${debateId}` : '';
    return this.client.get(`/api/v1/moments/timeline${params}`);
  }

  /**
   * Get trending moments across debates.
   */
  async getTrending(options?: { period?: string; limit?: number }): Promise<TrendingMoments> {
    const params = new URLSearchParams();
    if (options?.period) params.set('period', options.period);
    if (options?.limit) params.set('limit', options.limit.toString());
    const query = params.toString() ? `?${params.toString()}` : '';
    return this.client.get(`/api/v1/moments/trending${query}`);
  }

  /**
   * Get moments filtered by type.
   */
  async getByType(
    type: 'breakthrough' | 'conflict' | 'consensus' | 'insight' | 'question' | 'evidence',
    options?: { limit?: number; offset?: number }
  ): Promise<{ moments: Moment[]; total: number }> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', options.limit.toString());
    if (options?.offset) params.set('offset', options.offset.toString());
    const query = params.toString() ? `?${params.toString()}` : '';
    return this.client.get(`/api/v1/moments/by-type/${type}${query}`);
  }
}
