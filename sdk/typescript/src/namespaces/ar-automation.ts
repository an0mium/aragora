/**
 * AR Automation Namespace API
 *
 * Provides Accounts Receivable automation:
 * - Invoice creation and sending
 * - Payment reminders
 * - Aging reports
 * - Collection suggestions
 * - Customer balance tracking
 */

/**
 * AR Invoice status.
 */
export type ARInvoiceStatus = 'draft' | 'sent' | 'viewed' | 'paid' | 'partial' | 'overdue';

/**
 * Reminder escalation level.
 */
export type ReminderLevel = 1 | 2 | 3 | 4;

/**
 * Line item on an invoice.
 */
export interface ARLineItem {
  description: string;
  quantity?: number;
  unit_price?: number;
  amount: number;
  tax_rate?: number;
}

/**
 * AR Invoice.
 */
export interface ARInvoice {
  id: string;
  customer_id: string;
  customer_name: string;
  customer_email?: string;
  invoice_number: string;
  invoice_date: string;
  due_date: string;
  total_amount: number;
  tax_amount?: number;
  amount_paid: number;
  balance_due: number;
  payment_terms: string;
  status: ARInvoiceStatus;
  line_items: ARLineItem[];
  notes?: string;
  last_reminder_sent?: string;
  reminder_count: number;
  created_at: string;
  updated_at?: string;
}

/**
 * Create AR invoice request.
 */
export interface CreateARInvoiceRequest {
  customer_id: string;
  customer_name: string;
  customer_email?: string;
  invoice_date?: string;
  due_date: string;
  payment_terms?: string;
  line_items: ARLineItem[];
  notes?: string;
  send_immediately?: boolean;
}

/**
 * List AR invoices parameters.
 */
export interface ListARInvoicesParams {
  limit?: number;
  offset?: number;
  status?: ARInvoiceStatus;
  customer_id?: string;
  overdue_only?: boolean;
  due_before?: string;
  due_after?: string;
}

/**
 * Record AR payment request.
 */
export interface RecordARPaymentRequest {
  amount: number;
  payment_date?: string;
  payment_method?: string;
  reference_number?: string;
  notes?: string;
}

/**
 * Aging bucket.
 */
export interface AgingBucket {
  label: string;
  days_range: string;
  count: number;
  total_amount: number;
  invoices: Array<{
    invoice_id: string;
    customer_name: string;
    amount: number;
    days_overdue: number;
  }>;
}

/**
 * Aging report.
 */
export interface AgingReport {
  as_of_date: string;
  total_outstanding: number;
  total_overdue: number;
  buckets: {
    current: AgingBucket;
    days_1_30: AgingBucket;
    days_31_60: AgingBucket;
    days_61_90: AgingBucket;
    days_over_90: AgingBucket;
  };
  generated_at: string;
}

/**
 * Collection action type.
 */
export type CollectionActionType =
  | 'send_reminder'
  | 'phone_call'
  | 'escalate_to_collections'
  | 'offer_payment_plan'
  | 'final_notice';

/**
 * Collection suggestion.
 */
export interface CollectionSuggestion {
  invoice_id: string;
  customer_id: string;
  customer_name: string;
  amount_due: number;
  days_overdue: number;
  suggested_action: CollectionActionType;
  action_priority: 'high' | 'medium' | 'low';
  reason: string;
  previous_attempts: number;
}

/**
 * Add customer request.
 */
export interface AddARCustomerRequest {
  name: string;
  email?: string;
  phone?: string;
  billing_address?: {
    street?: string;
    city?: string;
    state?: string;
    zip?: string;
    country?: string;
  };
  payment_terms?: string;
  credit_limit?: number;
  notes?: string;
}

/**
 * Customer balance.
 */
export interface CustomerBalance {
  customer_id: string;
  customer_name: string;
  total_outstanding: number;
  total_overdue: number;
  invoice_count: number;
  oldest_invoice_date?: string;
  average_days_to_pay?: number;
  credit_limit?: number;
  available_credit?: number;
}

/**
 * Client interface for AR automation operations.
 */
interface ARAutomationClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * AR Automation API namespace.
 *
 * Provides methods for Accounts Receivable automation:
 * - Create and send invoices
 * - Send payment reminders
 * - Generate aging reports
 * - Get collection suggestions
 * - Track customer balances
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Create and send an invoice
 * const invoice = await client.arAutomation.createInvoice({
 *   customer_id: 'cust_123',
 *   customer_name: 'Acme Corp',
 *   customer_email: 'billing@acme.com',
 *   due_date: '2025-02-28',
 *   line_items: [{ description: 'Consulting', amount: 5000 }],
 *   send_immediately: true,
 * });
 *
 * // Get aging report
 * const aging = await client.arAutomation.getAgingReport();
 * ```
 */
export class ARAutomationAPI {
  constructor(private client: ARAutomationClientInterface) {}

  /**
   * Create a new AR invoice.
   */
  async createInvoice(
    request: CreateARInvoiceRequest
  ): Promise<{ invoice: ARInvoice; message: string }> {
    return this.client.request('POST', '/api/v1/accounting/ar/invoices', {
      json: request,
    });
  }

  /**
   * List AR invoices with filtering.
   */
  async listInvoices(
    params?: ListARInvoicesParams
  ): Promise<{ invoices: ARInvoice[]; total: number; limit: number; offset: number }> {
    return this.client.request('GET', '/api/v1/accounting/ar/invoices', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get a specific AR invoice.
   */
  async getInvoice(invoiceId: string): Promise<{ invoice: ARInvoice }> {
    return this.client.request('GET', `/api/v1/accounting/ar/invoices/${invoiceId}`);
  }

  /**
   * Send an invoice to the customer.
   */
  async sendInvoice(invoiceId: string): Promise<{ success: boolean; message: string }> {
    return this.client.request('POST', `/api/v1/accounting/ar/invoices/${invoiceId}/send`);
  }

  /**
   * Send a payment reminder for an invoice.
   */
  async sendReminder(
    invoiceId: string,
    escalationLevel?: ReminderLevel
  ): Promise<{ success: boolean; message: string; next_escalation?: ReminderLevel }> {
    return this.client.request('POST', `/api/v1/accounting/ar/invoices/${invoiceId}/reminder`, {
      json: escalationLevel ? { escalation_level: escalationLevel } : undefined,
    });
  }

  /**
   * Record a payment against an invoice.
   */
  async recordPayment(
    invoiceId: string,
    request: RecordARPaymentRequest
  ): Promise<{ invoice: ARInvoice; message: string }> {
    return this.client.request('POST', `/api/v1/accounting/ar/invoices/${invoiceId}/payment`, {
      json: request,
    });
  }

  /**
   * Get AR aging report.
   */
  async getAgingReport(): Promise<{ aging_report: AgingReport }> {
    return this.client.request('GET', '/api/v1/accounting/ar/aging');
  }

  /**
   * Get collection suggestions for overdue invoices.
   */
  async getCollectionSuggestions(): Promise<{ suggestions: CollectionSuggestion[] }> {
    return this.client.request('GET', '/api/v1/accounting/ar/collections');
  }

  /**
   * Add a new customer.
   */
  async addCustomer(
    request: AddARCustomerRequest
  ): Promise<{ success: boolean; customer_id: string; message: string }> {
    return this.client.request('POST', '/api/v1/accounting/ar/customers', {
      json: request,
    });
  }

  /**
   * Get customer balance summary.
   */
  async getCustomerBalance(customerId: string): Promise<CustomerBalance> {
    return this.client.request('GET', `/api/v1/accounting/ar/customers/${customerId}/balance`);
  }
}
