/**
 * Support Namespace API
 *
 * Provides endpoints for support platform integration including
 * ticket management, knowledge base, and customer interactions.
 */

import type { AragoraClient } from '../client';

/** Support ticket priority */
export type TicketPriority = 'low' | 'medium' | 'high' | 'urgent';

/** Support ticket status */
export type TicketStatus = 'open' | 'pending' | 'in_progress' | 'resolved' | 'closed';

/** Support ticket */
export interface Ticket {
  id: string;
  subject: string;
  description: string;
  status: TicketStatus;
  priority: TicketPriority;
  requester_email: string;
  assigned_to?: string;
  tags: string[];
  created_at: string;
  updated_at: string;
  resolved_at?: string;
}

/** Ticket reply */
export interface TicketReply {
  id: string;
  ticket_id: string;
  author: string;
  body: string;
  is_internal: boolean;
  created_at: string;
}

/** Request to create a ticket */
export interface CreateTicketRequest {
  subject: string;
  description: string;
  priority?: TicketPriority;
  requester_email: string;
  tags?: string[];
}

/** Support metrics */
export interface SupportMetrics {
  open_tickets: number;
  avg_response_time_minutes: number;
  avg_resolution_time_hours: number;
  satisfaction_score: number;
  tickets_by_priority: Record<string, number>;
}

/**
 * Support namespace for customer support operations.
 *
 * @example
 * ```typescript
 * const tickets = await client.support.listTickets({ status: 'open' });
 * const metrics = await client.support.getMetrics();
 * ```
 */
export class SupportNamespace {
  constructor(private client: AragoraClient) {}

  /** List support tickets. */
  async listTickets(options?: {
    status?: string;
    priority?: string;
    limit?: number;
    offset?: number;
  }): Promise<Ticket[]> {
    const response = await this.client.request<{ tickets: Ticket[] }>(
      'GET',
      '/api/v1/support/tickets',
      { params: options }
    );
    return response.tickets;
  }

  /** Create a support ticket. */
  async createTicket(request: CreateTicketRequest): Promise<Ticket> {
    return this.client.request<Ticket>('GET', '/api/v1/support/tickets',
      { body: request }
    );
  }

  /** Get a ticket by ID. */
  async getTicket(ticketId: string): Promise<Ticket> {
    return this.client.request<Ticket>(
      'GET',
      `/api/v1/support/tickets/${encodeURIComponent(ticketId)}`
    );
  }

  /** Reply to a ticket. */
  async replyToTicket(
    ticketId: string,
    reply: { body: string; is_internal?: boolean }
  ): Promise<TicketReply> {
    return this.client.request<TicketReply>(
      'POST',
      `/api/v1/support/tickets/${encodeURIComponent(ticketId)}/replies`,
      { body: reply }
    );
  }

  /** Get support metrics. */
  async getMetrics(options?: { period?: string }): Promise<SupportMetrics> {
    return this.client.request<SupportMetrics>(
      'GET',
      '/api/v1/support/metrics',
      { params: options }
    );
  }
}
