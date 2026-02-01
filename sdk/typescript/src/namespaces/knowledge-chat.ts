/**
 * Knowledge Chat Namespace API
 *
 * Provides methods for Knowledge + Chat bridge integration:
 * - Search knowledge from chat context
 * - Inject knowledge into conversations
 * - Store chat messages as knowledge
 * - Channel knowledge summaries
 *
 * Endpoints:
 *   POST /api/v1/chat/knowledge/search               - Search knowledge
 *   POST /api/v1/chat/knowledge/inject               - Inject knowledge
 *   POST /api/v1/chat/knowledge/store                - Store chat as knowledge
 *   GET  /api/v1/chat/knowledge/channel/:id/summary  - Channel summary
 */

/**
 * Search scope options.
 */
export type SearchScope = 'workspace' | 'channel' | 'user' | 'global';

/**
 * Search strategy options.
 */
export type SearchStrategy = 'hybrid' | 'semantic' | 'keyword' | 'exact';

/**
 * Chat message for injection/storage.
 */
export interface ChatMessage {
  author: string;
  content: string;
  timestamp?: string;
  message_id?: string;
}

/**
 * Knowledge search result item.
 */
export interface KnowledgeSearchResult {
  node_id: string;
  content: string;
  node_type: string;
  confidence: number;
  source?: string;
  created_at?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Knowledge context item for injection.
 */
export interface KnowledgeContextItem {
  node_id: string;
  content: string;
  relevance: number;
  node_type: string;
  source?: string;
}

/**
 * Channel knowledge summary item.
 */
export interface ChannelKnowledgeSummary {
  channel_id: string;
  total_nodes: number;
  recent_topics: string[];
  top_contributors: Array<{ user: string; contributions: number }>;
  knowledge_density: number;
  last_updated: string;
}

/**
 * Options for knowledge search.
 */
export interface KnowledgeSearchOptions {
  /** Search query text */
  query: string;
  /** Workspace to search in */
  workspaceId?: string;
  /** Optional channel ID for context */
  channelId?: string;
  /** Optional user ID for personalization */
  userId?: string;
  /** Search scope */
  scope?: SearchScope;
  /** Search strategy */
  strategy?: SearchStrategy;
  /** Filter by knowledge node types */
  nodeTypes?: string[];
  /** Minimum confidence threshold (default 0.3) */
  minConfidence?: number;
  /** Maximum results to return (default 10) */
  maxResults?: number;
}

/**
 * Options for knowledge injection.
 */
export interface KnowledgeInjectOptions {
  /** List of chat messages with author and content */
  messages: ChatMessage[];
  /** Workspace ID */
  workspaceId?: string;
  /** Optional channel ID */
  channelId?: string;
  /** Maximum knowledge items to return (default 5) */
  maxContextItems?: number;
}

/**
 * Options for storing chat as knowledge.
 */
export interface KnowledgeStoreOptions {
  /** List of chat messages (minimum 2 required) */
  messages: ChatMessage[];
  /** Workspace ID */
  workspaceId?: string;
  /** Channel ID */
  channelId?: string;
  /** Human-readable channel name */
  channelName?: string;
  /** Chat platform (slack, teams, discord, etc.) */
  platform?: string;
  /** Knowledge node type */
  nodeType?: string;
}

/**
 * Options for channel summary.
 */
export interface ChannelSummaryOptions {
  /** Channel ID */
  channelId: string;
  /** Workspace ID */
  workspaceId?: string;
  /** Maximum items in summary (default 10) */
  maxItems?: number;
}

/**
 * Client interface for making HTTP requests.
 */
interface KnowledgeChatClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Knowledge Chat API namespace.
 *
 * Provides methods for integrating knowledge with chat platforms:
 * - Search for relevant knowledge based on chat context
 * - Inject knowledge into ongoing conversations
 * - Store important chat exchanges as knowledge
 * - Get summaries of channel-related knowledge
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Search knowledge
 * const results = await client.knowledgeChat.search({
 *   query: "What's our vacation policy?",
 *   workspaceId: 'ws_123',
 *   channelId: 'C123456',
 * });
 *
 * // Inject knowledge into conversation
 * const context = await client.knowledgeChat.inject({
 *   messages: [
 *     { author: 'user1', content: 'How do we handle PTO?' },
 *     { author: 'user2', content: 'Good question' },
 *   ],
 *   workspaceId: 'ws_123',
 * });
 *
 * // Store chat as knowledge
 * const stored = await client.knowledgeChat.store({
 *   messages: [
 *     { author: 'user1', content: 'We decided to use Python 3.11' },
 *     { author: 'user2', content: 'Agreed, better performance' },
 *   ],
 *   workspaceId: 'ws_123',
 *   channelName: '#engineering',
 * });
 * ```
 */
export class KnowledgeChatAPI {
  constructor(private client: KnowledgeChatClientInterface) {}

  // =========================================================================
  // Search
  // =========================================================================

  /**
   * Search knowledge from chat context.
   *
   * @param options - Search options
   * @returns Search results with metadata
   */
  async search(options: KnowledgeSearchOptions): Promise<{
    results: KnowledgeSearchResult[];
    total: number;
    query: string;
    scope: SearchScope;
    strategy: SearchStrategy;
  }> {
    const data: Record<string, unknown> = {
      query: options.query,
      workspace_id: options.workspaceId ?? 'default',
    };

    if (options.channelId) data.channel_id = options.channelId;
    if (options.userId) data.user_id = options.userId;
    if (options.scope && options.scope !== 'workspace') data.scope = options.scope;
    if (options.strategy && options.strategy !== 'hybrid') data.strategy = options.strategy;
    if (options.nodeTypes) data.node_types = options.nodeTypes;
    if (options.minConfidence !== undefined && options.minConfidence !== 0.3) {
      data.min_confidence = options.minConfidence;
    }
    if (options.maxResults !== undefined && options.maxResults !== 10) {
      data.max_results = options.maxResults;
    }

    return this.client.request('POST', '/api/v1/chat/knowledge/search', {
      json: data,
    });
  }

  // =========================================================================
  // Inject
  // =========================================================================

  /**
   * Get relevant knowledge to inject into a conversation.
   *
   * Analyzes the conversation and returns relevant knowledge items
   * that could enhance the discussion.
   *
   * @param options - Injection options
   * @returns Context items with relevance scores
   */
  async inject(options: KnowledgeInjectOptions): Promise<{
    context_items: KnowledgeContextItem[];
    count: number;
  }> {
    const data: Record<string, unknown> = {
      messages: options.messages,
      workspace_id: options.workspaceId ?? 'default',
    };

    if (options.channelId) data.channel_id = options.channelId;
    if (options.maxContextItems !== undefined && options.maxContextItems !== 5) {
      data.max_context_items = options.maxContextItems;
    }

    return this.client.request('POST', '/api/v1/chat/knowledge/inject', {
      json: data,
    });
  }

  // =========================================================================
  // Store
  // =========================================================================

  /**
   * Store chat messages as persistent knowledge.
   *
   * @param options - Storage options
   * @returns Stored node information
   * @throws Error if fewer than 2 messages provided
   */
  async store(options: KnowledgeStoreOptions): Promise<{
    node_id: string;
    message_count: number;
  }> {
    if (options.messages.length < 2) {
      throw new Error('At least 2 messages required');
    }

    const data: Record<string, unknown> = {
      messages: options.messages,
      workspace_id: options.workspaceId ?? 'default',
    };

    if (options.channelId) data.channel_id = options.channelId;
    if (options.channelName) data.channel_name = options.channelName;
    if (options.platform && options.platform !== 'unknown') data.platform = options.platform;
    if (options.nodeType && options.nodeType !== 'chat_context') data.node_type = options.nodeType;

    return this.client.request('POST', '/api/v1/chat/knowledge/store', {
      json: data,
    });
  }

  // =========================================================================
  // Channel Summary
  // =========================================================================

  /**
   * Get a summary of knowledge related to a channel.
   *
   * @param options - Summary options
   * @returns Channel knowledge summary
   */
  async getChannelSummary(options: ChannelSummaryOptions): Promise<ChannelKnowledgeSummary> {
    const params: Record<string, unknown> = {
      workspace_id: options.workspaceId ?? 'default',
    };
    if (options.maxItems !== undefined && options.maxItems !== 10) {
      params.max_items = options.maxItems;
    }

    return this.client.request('GET', `/api/v1/chat/knowledge/channel/${options.channelId}/summary`, {
      params,
    });
  }
}
