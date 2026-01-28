/**
 * Chat Namespace API
 *
 * Provides knowledge chat endpoints for search, inject, and store operations.
 */

export interface ChatKnowledgeSearchRequest {
  query: string;
  workspace_id?: string;
  limit?: number;
}

export interface ChatKnowledgeSearchResponse {
  results: Array<Record<string, unknown>>;
  total?: number;
}

export interface ChatKnowledgeInjectRequest {
  channel_id: string;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface ChatKnowledgeStoreRequest {
  content: string;
  workspace_id?: string;
  metadata?: Record<string, unknown>;
}

export interface ChatKnowledgeSummaryResponse {
  channel_id: string;
  summary: string;
  message_count?: number;
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

export class ChatAPI {
  constructor(private client: ChatClientInterface) {}

  async searchKnowledge(body: ChatKnowledgeSearchRequest): Promise<ChatKnowledgeSearchResponse> {
    return this.client.post('/api/v1/chat/knowledge/search', body);
  }

  async injectKnowledge(body: ChatKnowledgeInjectRequest): Promise<{ stored: boolean }> {
    return this.client.post('/api/v1/chat/knowledge/inject', body);
  }

  async storeKnowledge(body: ChatKnowledgeStoreRequest): Promise<{ stored: boolean }> {
    return this.client.post('/api/v1/chat/knowledge/store', body);
  }

  async getChannelSummary(channelId: string): Promise<ChatKnowledgeSummaryResponse> {
    return this.client.get(`/api/v1/chat/knowledge/channel/${channelId}/summary`);
  }
}
