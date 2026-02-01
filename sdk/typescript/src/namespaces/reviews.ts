/**
 * Reviews Namespace API
 *
 * Provides access to debate and decision reviews.
 */

export interface Review {
  id: string;
  debate_id?: string;
  reviewer?: string;
  status: 'pending' | 'approved' | 'rejected' | 'needs_revision';
  rating?: number;
  comments?: string;
  created_at: string;
  updated_at?: string;
  metadata?: Record<string, unknown>;
}

interface ReviewsClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown> }
  ): Promise<T>;
}

export class ReviewsAPI {
  constructor(private client: ReviewsClientInterface) {}

  /**
   * List recent reviews.
   */
  async list(params?: {
    limit?: number;
    offset?: number;
    status?: string;
  }): Promise<{ reviews: Review[]; total: number }> {
    return this.client.request('GET', '/api/v1/reviews', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get a specific review by ID.
   */
  async get(reviewId: string): Promise<Review> {
    return this.client.request('GET', `/api/v1/reviews/${reviewId}`);
  }
}
