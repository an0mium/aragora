/**
 * Invoice Processing Namespace API
 *
 * Provides invoice processing and approval workflows:
 * - Document upload and OCR extraction
 * - Invoice approval/rejection
 * - PO matching
 * - Anomaly detection
 * - Payment scheduling
 */

/**
 * Invoice processing status.
 */
export type InvoiceProcessingStatus = 'pending' | 'approved' | 'rejected' | 'paid' | 'processing';

/**
 * Anomaly severity level.
 */
export type AnomalySeverity = 'critical' | 'high' | 'medium' | 'low';

/**
 * Invoice line item.
 */
export interface InvoiceLineItem {
  description: string;
  quantity?: number;
  unit_price?: number;
  amount: number;
  gl_code?: string;
  tax_rate?: number;
}

/**
 * Processed invoice.
 */
export interface ProcessedInvoice {
  id: string;
  vendor_name: string;
  vendor_id?: string;
  invoice_number?: string;
  invoice_date?: string;
  due_date?: string;
  total_amount: number;
  tax_amount?: number;
  subtotal?: number;
  line_items: InvoiceLineItem[];
  po_number?: string;
  status: InvoiceProcessingStatus;
  approver_id?: string;
  approved_at?: string;
  rejected_reason?: string;
  confidence_score?: number;
  extraction_method?: 'ocr' | 'manual' | 'api';
  created_at: string;
  updated_at?: string;
}

/**
 * Invoice anomaly.
 */
export interface InvoiceAnomaly {
  type: string;
  severity: AnomalySeverity;
  description: string;
  field?: string;
  expected_value?: string;
  actual_value?: string;
  recommendation?: string;
}

/**
 * Create invoice request.
 */
export interface CreateInvoiceRequest {
  vendor_name: string;
  vendor_id?: string;
  invoice_number?: string;
  invoice_date?: string;
  due_date: string;
  total_amount: number;
  tax_amount?: number;
  line_items?: InvoiceLineItem[];
  po_number?: string;
  notes?: string;
}

/**
 * List invoices parameters.
 */
export interface ListInvoicesParams {
  limit?: number;
  offset?: number;
  status?: InvoiceProcessingStatus;
  vendor_id?: string;
  approver_id?: string;
  date_from?: string;
  date_to?: string;
}

/**
 * PO match result.
 */
export interface POMatch {
  matched: boolean;
  po_number: string;
  po_amount: number;
  invoice_amount: number;
  variance: number;
  variance_percent: number;
  within_tolerance: boolean;
  line_matches: Array<{
    po_line: string;
    invoice_line: string;
    match_score: number;
  }>;
}

/**
 * Schedule payment request.
 */
export interface SchedulePaymentRequest {
  payment_date?: string;
  payment_method?: string;
  notes?: string;
}

/**
 * Scheduled payment.
 */
export interface ScheduledPayment {
  id: string;
  invoice_id: string;
  vendor_name: string;
  amount: number;
  scheduled_date: string;
  payment_method: string;
  status: 'scheduled' | 'processing' | 'completed' | 'failed';
  created_at: string;
}

/**
 * Invoice statistics.
 */
export interface InvoiceStats {
  total_invoices: number;
  pending_approval: number;
  approved: number;
  rejected: number;
  paid: number;
  total_amount_pending: number;
  total_amount_approved: number;
  average_processing_time_hours: number;
  anomaly_rate: number;
  period: string;
}

/**
 * Purchase order.
 */
export interface PurchaseOrder {
  id: string;
  po_number: string;
  vendor_id: string;
  vendor_name: string;
  total_amount: number;
  status: 'open' | 'partial' | 'closed';
  line_items: Array<{
    description: string;
    quantity: number;
    unit_price: number;
    amount: number;
  }>;
  created_at: string;
}

/**
 * Create PO request.
 */
export interface CreatePORequest {
  vendor_id: string;
  vendor_name: string;
  line_items: Array<{
    description: string;
    quantity: number;
    unit_price: number;
  }>;
  notes?: string;
}

/**
 * Client interface for invoice processing operations.
 */
interface InvoiceProcessingClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Invoice Processing API namespace.
 *
 * Provides methods for invoice processing and approval workflows:
 * - Upload and extract invoice data
 * - Approve or reject invoices
 * - Match invoices to POs
 * - Detect anomalies
 * - Schedule payments
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Upload an invoice document
 * const invoice = await client.invoiceProcessing.upload(base64Data, 'application/pdf');
 *
 * // Approve the invoice
 * await client.invoiceProcessing.approve(invoice.id);
 *
 * // Check for anomalies
 * const anomalies = await client.invoiceProcessing.getAnomalies(invoice.id);
 * ```
 */
export class InvoiceProcessingAPI {
  constructor(private client: InvoiceProcessingClientInterface) {}

  /**
   * Upload and process an invoice document.
   */
  async upload(
    documentData: string,
    contentType?: string,
    vendorHint?: string
  ): Promise<{ invoice: ProcessedInvoice; anomalies: InvoiceAnomaly[] }> {
    return this.client.request('POST', '/api/v1/accounting/invoices/upload', {
      json: {
        document_data: documentData,
        content_type: contentType,
        vendor_hint: vendorHint,
      },
    });
  }

  /**
   * Create an invoice manually.
   */
  async create(
    request: CreateInvoiceRequest
  ): Promise<{ invoice: ProcessedInvoice; message: string }> {
    return this.client.request('POST', '/api/v1/accounting/invoices', {
      json: request,
    });
  }

  /**
   * List invoices with filtering.
   */
  async list(
    params?: ListInvoicesParams
  ): Promise<{ invoices: ProcessedInvoice[]; total: number; limit: number; offset: number }> {
    return this.client.request('GET', '/api/v1/accounting/invoices', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get a specific invoice.
   */
  async get(invoiceId: string): Promise<{ invoice: ProcessedInvoice }> {
    return this.client.request('GET', `/api/v1/accounting/invoices/${invoiceId}`);
  }

  /**
   * Approve an invoice.
   */
  async approve(
    invoiceId: string,
    approverId?: string
  ): Promise<{ invoice: ProcessedInvoice; message: string }> {
    return this.client.request('POST', `/api/v1/accounting/invoices/${invoiceId}/approve`, {
      json: approverId ? { approver_id: approverId } : undefined,
    });
  }

  /**
   * Reject an invoice.
   */
  async reject(
    invoiceId: string,
    reason?: string
  ): Promise<{ invoice: ProcessedInvoice; message: string }> {
    return this.client.request('POST', `/api/v1/accounting/invoices/${invoiceId}/reject`, {
      json: reason ? { reason } : undefined,
    });
  }

  /**
   * Match an invoice to a purchase order.
   */
  async matchToPO(invoiceId: string): Promise<{ match: POMatch; invoice: ProcessedInvoice }> {
    return this.client.request('POST', `/api/v1/accounting/invoices/${invoiceId}/match`);
  }

  /**
   * Schedule payment for an invoice.
   */
  async schedulePayment(
    invoiceId: string,
    request?: SchedulePaymentRequest
  ): Promise<{ schedule: ScheduledPayment; message: string }> {
    return this.client.request('POST', `/api/v1/accounting/invoices/${invoiceId}/schedule`, {
      json: request,
    });
  }

  /**
   * Get anomalies detected in an invoice.
   */
  async getAnomalies(invoiceId: string): Promise<{ anomalies: InvoiceAnomaly[] }> {
    return this.client.request('GET', `/api/v1/accounting/invoices/${invoiceId}/anomalies`);
  }

  /**
   * Get invoices pending approval.
   */
  async getPendingApprovals(): Promise<{ invoices: ProcessedInvoice[]; count: number }> {
    return this.client.request('GET', '/api/v1/accounting/invoices/pending');
  }

  /**
   * Get overdue invoices.
   */
  async getOverdue(): Promise<{
    invoices: ProcessedInvoice[];
    count: number;
    total_amount: number;
  }> {
    return this.client.request('GET', '/api/v1/accounting/invoices/overdue');
  }

  /**
   * Get invoice processing statistics.
   */
  async getStats(period?: string): Promise<{ stats: InvoiceStats }> {
    return this.client.request('GET', '/api/v1/accounting/invoices/stats', {
      params: period ? { period } : undefined,
    });
  }

  /**
   * Create a purchase order.
   */
  async createPurchaseOrder(
    request: CreatePORequest
  ): Promise<{ purchase_order: PurchaseOrder; message: string }> {
    return this.client.request('POST', '/api/v1/accounting/purchase-orders', {
      json: request,
    });
  }

  /**
   * Get scheduled payments.
   */
  async getScheduledPayments(params?: {
    limit?: number;
    offset?: number;
    status?: string;
    date_from?: string;
    date_to?: string;
  }): Promise<{ payments: ScheduledPayment[]; total: number }> {
    return this.client.request('GET', '/api/v1/accounting/payments/scheduled', {
      params: params as Record<string, unknown>,
    });
  }
}
