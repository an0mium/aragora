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
 * Client interface for making HTTP requests.
 */
interface RLMClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * RLM API for recursive language model operations.
 */
export class RLMAPI {
  constructor(private client: RLMClientInterface) {}

  /**
   * Get RLM compression statistics.
   */
  async getStats(): Promise<RLMStats> {
    return this.client.request('GET', '/api/v1/rlm/stats');
  }

  /**
   * Get available decomposition strategies.
   */
  async getStrategies(): Promise<{
    strategies: Record<string, StrategyInfo>;
    default: string;
    documentation: string;
  }> {
    return this.client.request('GET', '/api/v1/rlm/strategies');
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
    return this.client.request('POST', '/api/v1/rlm/compress', {
      json: {
        content,
        source_type: options?.source_type ?? 'text',
        levels: options?.levels ?? 4,
      },
    });
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
    return this.client.request('POST', '/api/v1/rlm/query', {
      json: {
        context_id: contextId,
        query,
        strategy: options?.strategy ?? 'auto',
        refine: options?.refine ?? false,
        max_iterations: options?.max_iterations ?? 3,
      },
    });
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
    const params: Record<string, unknown> = {};
    if (options?.limit) params.limit = options.limit;
    if (options?.offset) params.offset = options.offset;

    return this.client.request('GET', '/api/v1/rlm/contexts', {
      params: Object.keys(params).length > 0 ? params : undefined,
    });
  }

  /**
   * Get details of a specific context.
   *
   * @param contextId - Context ID
   * @param includeContent - Include summary preview content
   */
  async getContext(contextId: string, includeContent = false): Promise<ContextDetails> {
    const params: Record<string, unknown> = {};
    if (includeContent) params.include_content = true;

    return this.client.request('GET', `/api/v1/rlm/context/${contextId}`, {
      params: Object.keys(params).length > 0 ? params : undefined,
    });
  }

  /**
   * Delete a compressed context.
   *
   * @param contextId - Context ID to delete
   */
  async deleteContext(
    contextId: string
  ): Promise<{ success: boolean; context_id: string; message: string }> {
    return this.client.request('DELETE', `/api/v1/rlm/context/${contextId}`);
  }

  /**
   * Get available streaming modes.
   */
  async getStreamModes(): Promise<{
    modes: Array<{ mode: string; description: string; use_case: string }>;
  }> {
    return this.client.request('GET', '/api/v1/rlm/stream/modes');
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
    return this.client.request('POST', '/api/v1/rlm/stream', {
      json: {
        context_id: contextId,
        mode: options?.mode ?? 'top_down',
        query: options?.query,
        level: options?.level,
        chunk_size: options?.chunk_size ?? 500,
        include_metadata: options?.include_metadata ?? true,
      },
    });
  }
}
