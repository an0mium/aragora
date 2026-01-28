/**
 * Outlook/Microsoft 365 Namespace API
 *
 * Provides a namespaced interface for Outlook/M365 email integration.
 * Enables message management, folder operations, and email sending.
 */

import type { PaginationParams } from '../types';

// =============================================================================
// Type Definitions
// =============================================================================

/**
 * OAuth initiation response.
 */
export interface OutlookOAuthResponse {
  auth_url: string;
  state: string;
}

/**
 * OAuth callback result.
 */
export interface OutlookAuthResult {
  success: boolean;
  email?: string;
  workspace_id?: string;
  user_id?: string;
  error?: string;
}

/**
 * Outlook connection status.
 */
export interface OutlookStatus {
  connected: boolean;
  email?: string;
  display_name?: string;
  error?: string;
}

/**
 * Mail folder.
 */
export interface MailFolder {
  id: string;
  display_name: string;
  unread_count: number;
  total_count: number;
  child_folder_count: number;
  is_hidden: boolean;
}

/**
 * Email message summary.
 */
export interface MessageSummary {
  id: string;
  thread_id?: string;
  subject: string;
  from_address: string;
  to_addresses: string[];
  date?: string;
  snippet?: string;
  is_read: boolean;
  is_starred: boolean;
  is_important: boolean;
}

/**
 * Full email message.
 */
export interface Message extends MessageSummary {
  cc_addresses?: string[];
  bcc_addresses?: string[];
  body_text?: string;
  body_html?: string;
  labels?: string[];
  attachments?: Attachment[];
}

/**
 * Email attachment.
 */
export interface Attachment {
  id: string;
  filename: string;
  mime_type: string;
  size: number;
}

/**
 * Email conversation/thread.
 */
export interface Conversation {
  id: string;
  subject: string;
  message_count: number;
  participants: string[];
  last_message_date?: string;
  snippet?: string;
  messages: Message[];
}

/**
 * Message list response.
 */
export interface MessageListResponse {
  messages: MessageSummary[];
  total: number;
  next_page_token?: string;
}

/**
 * Folder list response.
 */
export interface FolderListResponse {
  folders: MailFolder[];
  total: number;
}

/**
 * Search result item.
 */
export interface SearchResult {
  id: string;
  title: string;
  snippet: string;
  author?: string;
  url?: string;
}

/**
 * Search results response.
 */
export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total: number;
}

/**
 * Send message request.
 */
export interface SendMessageRequest {
  to: string[];
  subject: string;
  body: string;
  body_type?: 'text' | 'html';
  cc?: string[];
  bcc?: string[];
}

/**
 * Reply message request.
 */
export interface ReplyMessageRequest {
  message_id: string;
  body: string;
  body_type?: 'text' | 'html';
  reply_all?: boolean;
  cc?: string[];
}

/**
 * List messages params.
 */
export interface ListMessagesParams extends PaginationParams {
  folder_id?: string;
  max_results?: number;
  page_token?: string;
  filter?: string;
}

// =============================================================================
// Client Interface
// =============================================================================

interface OutlookClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

// =============================================================================
// Outlook API Class
// =============================================================================

/**
 * Outlook API namespace.
 *
 * Provides methods for Microsoft 365/Outlook email integration:
 * - OAuth connection flow
 * - Folder and message management
 * - Email sending and replying
 * - Message search
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Check connection status
 * const status = await client.outlook.getStatus();
 * if (!status.connected) {
 *   // Initiate OAuth
 *   const { auth_url, state } = await client.outlook.getOAuthUrl('https://app.com/callback');
 *   // Redirect user to auth_url...
 * }
 *
 * // List messages
 * const { messages } = await client.outlook.listMessages({ folder_id: 'Inbox', max_results: 25 });
 *
 * // Send email
 * await client.outlook.send({
 *   to: ['recipient@example.com'],
 *   subject: 'Hello',
 *   body: 'Message body'
 * });
 * ```
 */
export class OutlookAPI {
  constructor(private client: OutlookClientInterface) {}

  // ===========================================================================
  // Connection Management
  // ===========================================================================

  /**
   * Get OAuth authorization URL.
   *
   * @param redirectUri - Redirect URI after OAuth.
   * @returns OAuth URL and state parameter.
   */
  async getOAuthUrl(redirectUri: string): Promise<OutlookOAuthResponse> {
    return this.client.request('POST', '/api/v2/outlook/oauth/url', {
      json: { redirect_uri: redirectUri },
    });
  }

  /**
   * Complete OAuth flow with authorization code.
   *
   * @param code - Authorization code from callback.
   * @param state - State parameter from callback.
   * @param redirectUri - Optional redirect URI.
   */
  async completeOAuth(code: string, state: string, redirectUri?: string): Promise<OutlookAuthResult> {
    return this.client.request('POST', '/api/v2/outlook/oauth/callback', {
      json: { code, state, redirect_uri: redirectUri },
    });
  }

  /**
   * Get Outlook connection status.
   */
  async getStatus(): Promise<OutlookStatus> {
    return this.client.request('GET', '/api/v2/outlook/status');
  }

  // ===========================================================================
  // Folder Management
  // ===========================================================================

  /**
   * List mail folders.
   */
  async listFolders(): Promise<MailFolder[]> {
    const response = await this.client.request<FolderListResponse>('GET', '/api/v2/outlook/folders');
    return response.folders;
  }

  // ===========================================================================
  // Message Management
  // ===========================================================================

  /**
   * List messages with optional filtering.
   *
   * @param params.folder_id - Filter by folder.
   * @param params.max_results - Maximum results (default 50).
   * @param params.page_token - Pagination token.
   * @param params.filter - OData filter query.
   */
  async listMessages(params?: ListMessagesParams): Promise<MessageListResponse> {
    return this.client.request('GET', '/api/v2/outlook/messages', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get a message by ID.
   *
   * @param messageId - Message ID.
   * @param includeAttachments - Include attachment metadata.
   */
  async getMessage(messageId: string, includeAttachments = false): Promise<Message> {
    return this.client.request('GET', `/api/v2/outlook/messages/${messageId}`, {
      params: { include_attachments: includeAttachments },
    });
  }

  /**
   * Get a conversation/thread.
   *
   * @param conversationId - Conversation ID.
   * @param maxMessages - Maximum messages to include.
   */
  async getConversation(conversationId: string, maxMessages = 50): Promise<Conversation> {
    return this.client.request('GET', `/api/v2/outlook/conversations/${conversationId}`, {
      params: { max_messages: maxMessages },
    });
  }

  // ===========================================================================
  // Send & Reply
  // ===========================================================================

  /**
   * Send a new email.
   *
   * @example
   * ```typescript
   * await client.outlook.send({
   *   to: ['recipient@example.com'],
   *   subject: 'Meeting Request',
   *   body: '<h1>Hello</h1><p>Let\'s meet tomorrow.</p>',
   *   body_type: 'html'
   * });
   * ```
   */
  async send(request: SendMessageRequest): Promise<{ success: boolean; message?: string }> {
    return this.client.request('POST', '/api/v2/outlook/messages/send', {
      json: request,
    });
  }

  /**
   * Reply to a message.
   *
   * @example
   * ```typescript
   * await client.outlook.reply({
   *   message_id: 'msg-123',
   *   body: 'Thanks for the update!',
   *   reply_all: true
   * });
   * ```
   */
  async reply(request: ReplyMessageRequest): Promise<{ success: boolean; message?: string; in_reply_to?: string }> {
    return this.client.request('POST', '/api/v2/outlook/messages/reply', {
      json: request,
    });
  }

  // ===========================================================================
  // Search
  // ===========================================================================

  /**
   * Search messages.
   *
   * @param query - Search query.
   * @param maxResults - Maximum results (default 25).
   * @param folderId - Optional folder to search within.
   */
  async search(query: string, maxResults = 25, folderId?: string): Promise<SearchResponse> {
    return this.client.request('GET', '/api/v2/outlook/messages/search', {
      params: {
        query,
        max_results: maxResults,
        ...(folderId && { folder_id: folderId }),
      },
    });
  }

  // ===========================================================================
  // Message Actions
  // ===========================================================================

  /**
   * Mark a message as read or unread.
   */
  async markRead(
    messageId: string,
    isRead = true
  ): Promise<{ success: boolean; message_id: string; is_read: boolean }> {
    return this.client.request('PATCH', `/api/v2/outlook/messages/${messageId}/read`, {
      json: { is_read: isRead },
    });
  }

  /**
   * Move a message to a different folder.
   */
  async move(
    messageId: string,
    destinationFolderId: string
  ): Promise<{ success: boolean; message_id: string; destination_folder_id: string }> {
    return this.client.request('POST', `/api/v2/outlook/messages/${messageId}/move`, {
      json: { destination_folder_id: destinationFolderId },
    });
  }

  /**
   * Delete a message.
   *
   * @param messageId - Message ID.
   * @param permanent - Permanently delete (skip Deleted Items).
   */
  async delete(
    messageId: string,
    permanent = false
  ): Promise<{ success: boolean; message_id: string; deleted: boolean; permanent: boolean }> {
    return this.client.request('DELETE', `/api/v2/outlook/messages/${messageId}`, {
      params: { permanent },
    });
  }
}
