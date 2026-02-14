/**
 * Shared Inbox Namespace API
 *
 * Provides endpoints for managing shared inboxes with
 * message routing, assignment, and rule-based automation.
 */

import type { AragoraClient } from '../client';

/** Message status in the shared inbox */
export type SharedInboxMessageStatus = 'new' | 'assigned' | 'in_progress' | 'resolved' | 'archived';

/** Shared inbox message */
export interface SharedInboxMessage {
  id: string;
  inbox_id: string;
  subject: string;
  body: string;
  sender: string;
  status: SharedInboxMessageStatus;
  assigned_to?: string;
  tags: string[];
  created_at: string;
  updated_at: string;
}

/** Shared inbox definition */
export interface SharedInbox {
  id: string;
  name: string;
  description?: string;
  workspace_id: string;
  message_count: number;
  created_at: string;
}

/** Routing rule for the shared inbox */
export interface RoutingRule {
  id: string;
  name: string;
  description?: string;
  conditions: Array<{ field: string; operator: string; value: string }>;
  actions: Array<{ type: string; value: string }>;
  enabled: boolean;
  priority: number;
}

/** Request to create a shared inbox */
export interface CreateSharedInboxRequest {
  name: string;
  description?: string;
}

/**
 * Shared Inbox namespace for team inbox management.
 *
 * @example
 * ```typescript
 * const inboxes = await client.sharedInbox.list();
 * const messages = await client.sharedInbox.getMessages(inboxes[0].id);
 * ```
 */
export class SharedInboxNamespace {
  constructor(private client: AragoraClient) {}

  /** List shared inboxes. */
  async list(): Promise<SharedInbox[]> {
    const response = await this.client.request<{ inboxes: SharedInbox[] }>(
      'GET',
      '/api/v1/shared-inbox'
    );
    return response.inboxes;
  }

  /** Create a shared inbox. */
  async create(request: CreateSharedInboxRequest): Promise<SharedInbox> {
    return this.client.request<SharedInbox>(
      'POST',
      '/api/v1/shared-inbox',
      { body: request }
    );
  }

  /** Get a shared inbox by ID. */
  async get(inboxId: string): Promise<SharedInbox> {
    return this.client.request<SharedInbox>(
      'GET',
      `/api/v1/shared-inbox/${encodeURIComponent(inboxId)}`
    );
  }

  /** Get messages in a shared inbox. */
  async getMessages(
    inboxId: string,
    options?: { limit?: number; offset?: number; status?: string }
  ): Promise<SharedInboxMessage[]> {
    const response = await this.client.request<{ messages: SharedInboxMessage[] }>(
      'GET',
      `/api/v1/shared-inbox/${encodeURIComponent(inboxId)}/messages`,
      { params: options }
    );
    return response.messages;
  }

  /** Assign a message to a user. */
  async assignMessage(
    inboxId: string,
    messageId: string,
    assignee: string
  ): Promise<SharedInboxMessage> {
    return this.client.request<SharedInboxMessage>(
      'POST',
      `/api/v1/shared-inbox/${encodeURIComponent(inboxId)}/messages/${encodeURIComponent(messageId)}/assign`,
      { body: { assigned_to: assignee } }
    );
  }

  /**
   * List shared inbox items.
   * @route GET /api/v1/inbox/shared
   */
  async listShared(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/inbox/shared', { params }) as Promise<Record<string, unknown>>;
  }

  /**
   * List inbox routing rules.
   * @route GET /api/v1/inbox/routing/rules
   */
  async listRoutingRules(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/inbox/routing/rules') as Promise<Record<string, unknown>>;
  }
}
