/**
 * RLM (Recursive Language Models) Namespace API
 *
 * Provides API endpoints for RLM compression and query operations:
 * - Content compression with hierarchical abstraction
 * - Query operations on compressed contexts
 * - Context storage and retrieval
 * - Streaming with multiple modes
 */

/**
 * RLM decomposition strategies.
 */
export type RLMStrategy =
  | 'peek'
  | 'grep'
  | 'partition_map'
  | 'summarize'
  | 'hierarchical'
  | 'auto';

/**
 * Content source types for compression.
 */
export type SourceType = 'text' | 'code' | 'debate';

/**
 * Streaming modes.
 */
export type StreamMode = 'top_down' | 'bottom_up' | 'targeted' | 'progressive';

/**
 * Strategy description.
 */
export interface StrategyInfo {
  name: string;
  description: string;
  use_case: string;
  token_reduction: string;
}

/**
 * Compression result statistics.
 */
export interface CompressionResult {
  original_tokens: number;
  compressed_tokens: number;
  compression_ratio: number;
  levels: Record<string, { nodes: number; tokens: number }>;
  source_type: SourceType;
}

/**
 * Query result with metadata.
 */
export interface QueryResult {
  answer: string;
  metadata: {
    context_id: string;
    strategy: RLMStrategy;
    refined: boolean;
    confidence?: number;
    iterations?: number;
    tokens_processed?: number;
    sub_calls_made?: number;
  };
  timestamp: string;
}

/**
 * Stored context summary.
 */
export interface ContextSummary {
  id: string;
  source_type: SourceType;
  original_tokens: number;
  created_at: string;
}

/**
 * Context details.
 */
export interface ContextDetails extends ContextSummary {
  compressed_tokens: number;
  compression_ratio: number;
  levels: Record<
    string,
    {
      nodes: number;
      tokens: number;
      node_ids: string[];
    }
  >;
  summary_preview?: Array<{ id: string; content: string }>;
}

/**
 * Stream chunk.
 */
export interface StreamChunk {
  level: string;
  content: string;
  token_count: number;
  is_final: boolean;
  metadata?: Record<string, unknown>;
}

/**
 * RLM system stats.
 */
export interface RLMStats {
  cache: {
    hits?: number;
    misses?: number;
    size?: number;
    error?: string;
  };
  contexts: {
    stored: number;
    ids: string[];
  };
  system: {
    has_official_rlm: boolean;
    compressor_available: boolean;
    rlm_available: boolean;
  };
  timestamp: string;
}

/**
 * RLM API for recursive language model operations.
 */
export class RLMAPI {
  private baseUrl: string;
  private headers: HeadersInit;

  constructor(baseUrl: string, apiKey: string) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    };
  }

  /**
   * Get RLM compression statistics.
   */
  async getStats(): Promise<RLMStats> {
    const response = await fetch(`${this.baseUrl}/api/v1/rlm/stats`, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to get stats: ${response.statusText}`);
    return response.json();
  }

  /**
   * Get available decomposition strategies.
   */
  async getStrategies(): Promise<{
    strategies: Record<string, StrategyInfo>;
    default: string;
    documentation: string;
  }> {
    const response = await fetch(`${this.baseUrl}/api/v1/rlm/strategies`, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to get strategies: ${response.statusText}`);
    return response.json();
  }

  /**
   * Compress content and get a context ID.
   *
   * @param content - The content to compress
   * @param options - Compression options
   */
  async compress(
    content: string,
    options?: {
      source_type?: SourceType;
      levels?: number;
    }
  ): Promise<{
    context_id: string;
    compression_result: CompressionResult;
    created_at: string;
  }> {
    const response = await fetch(`${this.baseUrl}/api/v1/rlm/compress`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        content,
        source_type: options?.source_type ?? 'text',
        levels: options?.levels ?? 4,
      }),
    });
    if (!response.ok) throw new Error(`Compression failed: ${response.statusText}`);
    return response.json();
  }

  /**
   * Query a compressed context.
   *
   * @param contextId - ID of the compressed context
   * @param query - The question to answer
   * @param options - Query options
   */
  async query(
    contextId: string,
    query: string,
    options?: {
      strategy?: RLMStrategy;
      refine?: boolean;
      max_iterations?: number;
    }
  ): Promise<QueryResult> {
    const response = await fetch(`${this.baseUrl}/api/v1/rlm/query`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        context_id: contextId,
        query,
        strategy: options?.strategy ?? 'auto',
        refine: options?.refine ?? false,
        max_iterations: options?.max_iterations ?? 3,
      }),
    });
    if (!response.ok) throw new Error(`Query failed: ${response.statusText}`);
    return response.json();
  }

  /**
   * List stored compressed contexts.
   *
   * @param options - Pagination options
   */
  async listContexts(options?: {
    limit?: number;
    offset?: number;
  }): Promise<{
    contexts: ContextSummary[];
    total: number;
    limit: number;
    offset: number;
  }> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', options.limit.toString());
    if (options?.offset) params.set('offset', options.offset.toString());

    const url = `${this.baseUrl}/api/v1/rlm/contexts${params.toString() ? `?${params}` : ''}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to list contexts: ${response.statusText}`);
    return response.json();
  }

  /**
   * Get details of a specific context.
   *
   * @param contextId - Context ID
   * @param includeContent - Include summary preview content
   */
  async getContext(contextId: string, includeContent = false): Promise<ContextDetails> {
    const url = `${this.baseUrl}/api/v1/rlm/context/${contextId}${includeContent ? '?include_content=true' : ''}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to get context: ${response.statusText}`);
    return response.json();
  }

  /**
   * Delete a compressed context.
   *
   * @param contextId - Context ID to delete
   */
  async deleteContext(
    contextId: string
  ): Promise<{ success: boolean; context_id: string; message: string }> {
    const response = await fetch(`${this.baseUrl}/api/v1/rlm/context/${contextId}`, {
      method: 'DELETE',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to delete context: ${response.statusText}`);
    return response.json();
  }

  /**
   * Get available streaming modes.
   */
  async getStreamModes(): Promise<{
    modes: Array<{ mode: string; description: string; use_case: string }>;
  }> {
    const response = await fetch(`${this.baseUrl}/api/v1/rlm/stream/modes`, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to get stream modes: ${response.statusText}`);
    return response.json();
  }

  /**
   * Stream context with configurable modes.
   *
   * @param contextId - Context ID to stream
   * @param options - Streaming options
   */
  async stream(
    contextId: string,
    options?: {
      mode?: StreamMode;
      query?: string;
      level?: string;
      chunk_size?: number;
      include_metadata?: boolean;
    }
  ): Promise<{
    context_id: string;
    mode: string;
    query?: string;
    chunks: StreamChunk[];
    total_chunks: number;
    timestamp: string;
  }> {
    const response = await fetch(`${this.baseUrl}/api/v1/rlm/stream`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        context_id: contextId,
        mode: options?.mode ?? 'top_down',
        query: options?.query,
        level: options?.level,
        chunk_size: options?.chunk_size ?? 500,
        include_metadata: options?.include_metadata ?? true,
      }),
    });
    if (!response.ok) throw new Error(`Stream failed: ${response.statusText}`);
    return response.json();
  }
}
