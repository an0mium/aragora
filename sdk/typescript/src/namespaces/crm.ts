/**
 * CRM Namespace API
 *
 * Provides endpoints for CRM integration including
 * contact management, deal tracking, and activity logging.
 */

import type { AragoraClient } from '../client';

/** CRM contact */
export interface Contact {
  id: string;
  first_name: string;
  last_name: string;
  email: string;
  phone?: string;
  company?: string;
  title?: string;
  tags: string[];
  created_at: string;
  updated_at: string;
}

/** CRM deal */
export interface Deal {
  id: string;
  title: string;
  value: number;
  currency: string;
  stage: string;
  contact_id: string;
  probability: number;
  expected_close_date?: string;
  created_at: string;
  updated_at: string;
}

/** CRM activity */
export interface Activity {
  id: string;
  type: 'call' | 'email' | 'meeting' | 'note' | 'task';
  subject: string;
  description?: string;
  contact_id: string;
  deal_id?: string;
  completed: boolean;
  created_at: string;
}

/** Request to create a contact */
export interface CreateContactRequest {
  first_name: string;
  last_name: string;
  email: string;
  phone?: string;
  company?: string;
  title?: string;
  tags?: string[];
}

/**
 * CRM namespace for customer relationship management.
 *
 * @example
 * ```typescript
 * const contacts = await client.crm.listContacts();
 * const deals = await client.crm.listDeals({ stage: 'proposal' });
 * ```
 */
export class CRMNamespace {
  constructor(private client: AragoraClient) {}

  /** List contacts. */
  async listContacts(options?: {
    search?: string;
    tag?: string;
    limit?: number;
  }): Promise<Contact[]> {
    const response = await this.client.request<{ contacts: Contact[] }>(
      'GET',
      '/api/v1/crm/contacts',
      { params: options }
    );
    return response.contacts;
  }

  /** Create a contact. */
  async createContact(request: CreateContactRequest): Promise<Contact> {
    return this.client.request<Contact>(
      'POST',
      '/api/v1/crm/contacts',
      { body: request }
    );
  }

  /** Get a contact by ID. */
  async getContact(contactId: string): Promise<Contact> {
    return this.client.request<Contact>(
      'GET',
      `/api/v1/crm/contacts/${encodeURIComponent(contactId)}`
    );
  }

  /** List deals. */
  async listDeals(options?: {
    stage?: string;
    contact_id?: string;
    limit?: number;
  }): Promise<Deal[]> {
    const response = await this.client.request<{ deals: Deal[] }>(
      'GET',
      '/api/v1/crm/deals',
      { params: options }
    );
    return response.deals;
  }

  /** Log an activity. */
  async logActivity(activity: {
    type: string;
    subject: string;
    contact_id: string;
    deal_id?: string;
    description?: string;
  }): Promise<Activity> {
    return this.client.request<Activity>(
      'POST',
      '/api/v1/crm/activities',
      { body: activity }
    );
  }
}
