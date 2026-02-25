/**
 * Search Namespace API
 *
 * Provides endpoints for searching across debates, knowledge,
 * agents, and other platform entities.
 */

import type { AragoraClient } from '../client';

/** Search result item */
export interface SearchResult {
  id: string;
  type: 'debate' | 'knowledge' | 'agent' | 'document' | 'workflow';
  title: string;
  snippet: string;
  score: number;
  metadata: Record<string, unknown>;
  created_at: string;
}

/** Search response with pagination */
export interface SearchResponse {
  results: SearchResult[];
  total: number;
  page: number;
  page_size: number;
  query: string;
}

/** Search facet for filtering */
export interface SearchFacet {
  field: string;
  values: Array<{ value: string; count: number }>;
}

/** Search query options */
export interface SearchOptions {
  query: string;
  type?: string;
  limit?: number;
  offset?: number;
  sort?: 'relevance' | 'date' | 'score';
  filters?: Record<string, string>;
}

/**
 * Search namespace for cross-platform search.
 *
 * @example
 * ```typescript
 * const results = await client.search.query({ query: 'rate limiting' });
 * console.log(`Found ${results.total} results`);
 * ```
 */
export class SearchNamespace {
  constructor(private client: AragoraClient) {}

  /** Execute a search query. */
  async query(options: SearchOptions): Promise<SearchResponse> {
    return this.client.request<SearchResponse>(
      'GET',
      '/api/v1/search',
      { params: options as unknown as Record<string, unknown> }
    );
  }

  /** Get search facets for filtering. */
  async getFacets(query: string): Promise<SearchFacet[]> {
    const response = await this.client.request<{ facets: SearchFacet[] }>(
      'GET',
      '/api/v1/search',
      { params: { query, facets: true } }
    );
    return response.facets;
  }

  /** Get search suggestions (autocomplete). */
  async suggest(prefix: string, options?: { limit?: number }): Promise<string[]> {
    const response = await this.client.request<{ suggestions: string[] }>(
      'GET',
      '/api/v1/search',
      { params: { query: prefix, suggest: true, ...options } }
    );
    return response.suggestions;
  }

  /** Reindex a specific entity type. */
  async reindex(type: string): Promise<{ success: boolean; indexed_count: number }> {
    return this.client.request<{ success: boolean; indexed_count: number }>(
      'POST',
      '/api/v1/search',
      { body: { type, action: 'reindex' } }
    );
  }
}
