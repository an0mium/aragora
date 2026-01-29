/**
 * Chat Namespace API
 *
 * Provides knowledge chat endpoints for search, inject, and store operations.
 * Bridges chat conversations with the knowledge management system.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Search knowledge from chat context
 * const results = await client.chat.searchKnowledge({
 *   query: "What's our vacation policy?",
 *   workspace_id: "ws_123",
 *   scope: "workspace",
 *   strategy: "hybrid",
 * });
 *
 * // Inject knowledge into a conversation
 * const context = await client.chat.injectKnowledge({
 *   messages: [{ author: "user1", content: "What's the policy?" }],
 *   workspace_id: "ws_123",
 * });
 *
 * // Store chat as knowledge
 * const stored = await client.chat.storeKnowledge({
 *   messages: [
 *     { author: "user1", content: "We decided to use Python 3.11" },
 *     { author: "user2", content: "Agreed, it has better performance" },
 *   ],
 *   workspace_id: "ws_123",
 *   channel_id: "C123456",
 *   platform: "slack",
 * });
 * ```
 */

/**
 * Search scope for knowledge queries.
 */
export type KnowledgeSearchScope = 'workspace' | 'channel' | 'user' | 'global';

/**
 * Relevance strategy for knowledge search.
 */
export type KnowledgeRelevanceStrategy = 'semantic' | 'keyword' | 'hybrid' | 'recency';

/**
 * Chat message structure for knowledge operations.
 */
export interface ChatMessage {
  author: string;
  content: string;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Knowledge search request parameters.
 */
export interface ChatKnowledgeSearchRequest {
  /** The search query */
  query: string;
  /** Workspace ID (default: "default") */
  workspace_id?: string;
  /** Channel ID for scoped search */
  channel_id?: string;
  /** User ID for personalized search */
  user_id?: string;
  /** Search scope */
  scope?: KnowledgeSearchScope;
  /** Relevance strategy */
  strategy?: KnowledgeRelevanceStrategy;
  /** Filter by node types */
  node_types?: string[];
  /** Minimum confidence threshold (0.0-1.0) */
  min_confidence?: number;
  /** Maximum results to return (1-100) */
  max_results?: number;
}

/**
 * Knowledge context item returned from search.
 */
export interface KnowledgeContextItem {
  /** Unique node identifier */
  node_id: string;
  /** Node type (e.g., "policy", "document", "chat_context") */
  node_type: string;
  /** Content of the knowledge item */
  content: string;
  /** Relevance score (0.0-1.0) */
  relevance_score: number;
  /** Confidence score (0.0-1.0) */
  confidence: number;
  /** Source metadata */
  source?: {
    channel_id?: string;
    platform?: string;
    created_at?: string;
  };
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Knowledge search response.
 */
export interface ChatKnowledgeSearchResponse {
  success: boolean;
  /** Search results */
  results: KnowledgeContextItem[];
  /** Total matching items */
  total: number;
  /** Search query used */
  query: string;
  /** Strategy used */
  strategy: string;
  /** Search metadata */
  search_metadata?: {
    execution_time_ms: number;
    nodes_searched: number;
  };
  /** Error message if not successful */
  error?: string;
}

/**
 * Knowledge inject request parameters.
 */
export interface ChatKnowledgeInjectRequest {
  /** Chat messages to analyze for knowledge injection */
  messages: ChatMessage[];
  /** Workspace ID (default: "default") */
  workspace_id?: string;
  /** Channel ID for context */
  channel_id?: string;
  /** Maximum context items to return (1-50) */
  max_context_items?: number;
}

/**
 * Knowledge inject response.
 */
export interface ChatKnowledgeInjectResponse {
  success: boolean;
  /** Context items to inject */
  context: KnowledgeContextItem[];
  /** Number of items returned */
  item_count: number;
  /** Error message if not successful */
  error?: string;
}

/**
 * Knowledge store request parameters.
 */
export interface ChatKnowledgeStoreRequest {
  /** Chat messages to store (minimum 2) */
  messages: ChatMessage[];
  /** Workspace ID (default: "default") */
  workspace_id?: string;
  /** Channel ID where the chat occurred */
  channel_id?: string;
  /** Channel name for display */
  channel_name?: string;
  /** Platform (e.g., "slack", "telegram", "discord") */
  platform?: string;
  /** Node type for storage (default: "chat_context") */
  node_type?: string;
}

/**
 * Knowledge store response.
 */
export interface ChatKnowledgeStoreResponse {
  success: boolean;
  /** ID of the created knowledge node */
  node_id?: string;
  /** Number of messages stored */
  message_count?: number;
  /** Error message if not successful */
  error?: string;
}

/**
 * Channel knowledge summary response.
 */
export interface ChatKnowledgeSummaryResponse {
  success: boolean;
  /** Channel ID */
  channel_id: string;
  /** Knowledge summary */
  summary: string;
  /** Total knowledge items for this channel */
  total_items: number;
  /** Top topics discussed */
  top_topics?: string[];
  /** Recent knowledge items */
  recent_items?: KnowledgeContextItem[];
  /** Error message if not successful */
  error?: string;
}

interface ChatClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown>; body?: unknown }
  ): Promise<T>;
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
}

/**
 * Chat API namespace.
 *
 * Provides methods for knowledge-chat integration:
 * - Search knowledge from chat context
 * - Inject relevant knowledge into conversations
 * - Store chat messages as knowledge
 * - Get channel knowledge summaries
 */
export class ChatAPI {
  constructor(private client: ChatClientInterface) {}

  /**
   * Search knowledge from chat context.
   *
   * Performs a semantic/keyword/hybrid search across the knowledge base
   * with optional scoping by workspace, channel, or user.
   *
   * @param body - Search parameters
   * @returns Search results with relevance scores
   */
  async searchKnowledge(body: ChatKnowledgeSearchRequest): Promise<ChatKnowledgeSearchResponse> {
    return this.client.post('/api/v1/chat/knowledge/search', body);
  }

  /**
   * Get relevant knowledge to inject into a conversation.
   *
   * Analyzes the provided messages and returns relevant knowledge items
   * that can be used to augment the conversation.
   *
   * @param body - Inject parameters including messages
   * @returns Context items to inject
   */
  async injectKnowledge(body: ChatKnowledgeInjectRequest): Promise<ChatKnowledgeInjectResponse> {
    return this.client.post('/api/v1/chat/knowledge/inject', body);
  }

  /**
   * Store chat messages as knowledge.
   *
   * Persists a chat conversation as a knowledge node that can be
   * retrieved and referenced in future searches.
   *
   * @param body - Store parameters including messages (minimum 2)
   * @returns Storage result with node ID
   */
  async storeKnowledge(body: ChatKnowledgeStoreRequest): Promise<ChatKnowledgeStoreResponse> {
    return this.client.post('/api/v1/chat/knowledge/store', body);
  }

  /**
   * Get a summary of knowledge related to a channel.
   *
   * @param channelId - The channel ID to get summary for
   * @param options - Optional parameters
   * @returns Channel knowledge summary
   */
  async getChannelSummary(
    channelId: string,
    options?: { workspace_id?: string; max_items?: number }
  ): Promise<ChatKnowledgeSummaryResponse> {
    const params = options ? { params: options } : undefined;
    return this.client.request(
      'GET',
      `/api/v1/chat/knowledge/channel/${channelId}/summary`,
      params
    );
  }
}
