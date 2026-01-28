/**
 * Expenses Namespace API
 *
 * Provides a namespaced interface for expense tracking and management.
 * Supports receipt processing, auto-categorization, approval workflows, and QBO sync.
 */
import type { PaginationParams } from '../types';
// =============================================================================
// Types
// =============================================================================

/** Expense categories */
export type ExpenseCategory =
  | 'travel'
  | 'meals'
  | 'office_supplies'
  | 'software'
  | 'equipment'
  | 'professional_services'
  | 'marketing'
  | 'utilities'
  | 'rent'
  | 'insurance'
  | 'other';

/** Expense status */
export type ExpenseStatus = 'pending' | 'approved' | 'rejected' | 'synced';

/** Payment methods */
export type PaymentMethod = 'credit_card' | 'debit_card' | 'cash' | 'check' | 'bank_transfer' | 'other';

/** Expense record */
export interface Expense {
  id: string;
  vendor_name: string;
  amount: number;
  currency: string;
  date: string;
  category: ExpenseCategory;
  payment_method: PaymentMethod;
  description: string;
  employee_id: string | null;
  is_reimbursable: boolean;
  tags: string[];
  status: ExpenseStatus;
  receipt_url: string | null;
  qbo_id: string | null;
  created_at: string;
  updated_at: string;
}

/** Receipt upload request */
export interface UploadReceiptRequest {
  receipt_data: string;  // Base64 encoded image
  content_type?: string; // image/png, image/jpeg, application/pdf
  employee_id?: string;
  payment_method?: PaymentMethod;
}

/** Create expense request */
export interface CreateExpenseRequest {
  vendor_name: string;
  amount: number;
  date?: string;
  category?: ExpenseCategory;
  payment_method?: PaymentMethod;
  description?: string;
  employee_id?: string;
  is_reimbursable?: boolean;
  tags?: string[];
}

/** Update expense request */
export interface UpdateExpenseRequest {
  vendor_name?: string;
  amount?: number;
  category?: ExpenseCategory;
  description?: string;
  status?: ExpenseStatus;
  is_reimbursable?: boolean;
  tags?: string[];
}

/** List expenses parameters */
export interface ListExpensesParams extends PaginationParams {
  category?: ExpenseCategory;
  vendor?: string;
  start_date?: string;
  end_date?: string;
  status?: ExpenseStatus;
  employee_id?: string;
}

/** Expense statistics */
export interface ExpenseStats {
  total_amount: number;
  expense_count: number;
  by_category: Record<ExpenseCategory, number>;
  by_status: Record<ExpenseStatus, number>;
  top_vendors: Array<{ vendor: string; amount: number; count: number }>;
  monthly_trend: Array<{ month: string; amount: number }>;
}

/** QBO sync result */
export interface SyncResult {
  success_count: number;
  failed_count: number;
  synced_ids: string[];
  errors: Array<{ expense_id: string; error: string }>;
}

/** Categorization result */
export interface CategorizeResult {
  categorized: Record<string, ExpenseCategory>;
  count: number;
}

// =============================================================================
// Expenses API
// =============================================================================

/**
 * Client interface for expenses operations.
 */
interface ExpensesClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Expenses namespace API for expense tracking and management.
 *
 * Provides comprehensive expense management:
 * - Receipt upload and OCR processing
 * - Auto-categorization using AI
 * - Approval workflows
 * - QuickBooks Online sync
 */
export class ExpensesAPI {
  constructor(private client: ExpensesClientInterface) {}

  // ===========================================================================
  // Receipt Processing
  // ===========================================================================

  /**
   * Upload and process a receipt image.
   *
   * Extracts vendor, amount, date, and category from the receipt
   * using OCR and AI analysis.
   *
   * @param request - Receipt data and metadata
   * @returns Created expense from receipt
   */
  async uploadReceipt(request: UploadReceiptRequest): Promise<{ expense: Expense; message: string }> {
    return this.client.request('POST', '/accounting/expenses/upload', {
      json: request as unknown as Record<string, unknown>,
    });
  }

  // ===========================================================================
  // CRUD Operations
  // ===========================================================================

  /**
   * Create an expense manually.
   *
   * @param request - Expense details
   * @returns Created expense
   */
  async create(request: CreateExpenseRequest): Promise<{ expense: Expense; message: string }> {
    return this.client.request('POST', '/accounting/expenses', {
      json: request as unknown as Record<string, unknown>,
    });
  }

  /**
   * List expenses with filters.
   *
   * @param params - Filtering and pagination options
   * @returns Paginated expense list
   */
  async list(params?: ListExpensesParams): Promise<{
    expenses: Expense[];
    total: number;
    limit: number;
    offset: number;
  }> {
    return this.client.request('GET', '/accounting/expenses', {
      params: params as unknown as Record<string, unknown>,
    });
  }

  /**
   * Get expense by ID.
   *
   * @param expenseId - Expense ID
   * @returns Expense details
   */
  async get(expenseId: string): Promise<{ expense: Expense }> {
    return this.client.request('GET', `/accounting/expenses/${expenseId}`);
  }

  /**
   * Update an expense.
   *
   * @param expenseId - Expense ID
   * @param request - Fields to update
   * @returns Updated expense
   */
  async update(expenseId: string, request: UpdateExpenseRequest): Promise<{ expense: Expense; message: string }> {
    return this.client.request('PUT', `/accounting/expenses/${expenseId}`, {
      json: request as unknown as Record<string, unknown>,
    });
  }

  /**
   * Delete an expense.
   *
   * @param expenseId - Expense ID
   * @returns Confirmation message
   */
  async delete(expenseId: string): Promise<{ message: string }> {
    return this.client.request('DELETE', `/accounting/expenses/${expenseId}`);
  }

  // ===========================================================================
  // Approval Workflow
  // ===========================================================================

  /**
   * Approve an expense for sync.
   *
   * @param expenseId - Expense ID
   * @returns Approved expense
   */
  async approve(expenseId: string): Promise<{ expense: Expense; message: string }> {
    return this.client.request('POST', `/accounting/expenses/${expenseId}/approve`);
  }

  /**
   * Reject an expense.
   *
   * @param expenseId - Expense ID
   * @param reason - Optional rejection reason
   * @returns Rejected expense
   */
  async reject(expenseId: string, reason?: string): Promise<{ expense: Expense; message: string }> {
    return this.client.request('POST', `/accounting/expenses/${expenseId}/reject`, {
      json: { reason },
    });
  }

  /**
   * Get expenses pending approval.
   *
   * @returns Pending expenses
   */
  async getPending(): Promise<{ expenses: Expense[]; count: number }> {
    return this.client.request('GET', '/accounting/expenses/pending');
  }

  // ===========================================================================
  // Categorization
  // ===========================================================================

  /**
   * Auto-categorize expenses using AI.
   *
   * @param expenseIds - Optional list of expense IDs (categorizes all uncategorized if empty)
   * @returns Categorization results
   */
  async categorize(expenseIds?: string[]): Promise<CategorizeResult & { message: string }> {
    return this.client.request('POST', '/accounting/expenses/categorize', {
      json: { expense_ids: expenseIds },
    });
  }

  // ===========================================================================
  // QBO Sync
  // ===========================================================================

  /**
   * Sync expenses to QuickBooks Online.
   *
   * @param expenseIds - Optional list of expense IDs (syncs all approved if empty)
   * @returns Sync results
   */
  async syncToQBO(expenseIds?: string[]): Promise<{ result: SyncResult; message: string }> {
    return this.client.request('POST', '/accounting/expenses/sync', {
      json: { expense_ids: expenseIds },
    });
  }

  // ===========================================================================
  // Statistics and Export
  // ===========================================================================

  /**
   * Get expense statistics.
   *
   * @param startDate - Optional start date filter
   * @param endDate - Optional end date filter
   * @returns Expense statistics
   */
  async getStats(startDate?: string, endDate?: string): Promise<{ stats: ExpenseStats }> {
    const params: Record<string, string> = {};
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    return this.client.request('GET', '/accounting/expenses/stats', { params });
  }

  /**
   * Export expenses to CSV or JSON.
   *
   * @param format - Export format ('csv' or 'json')
   * @param startDate - Optional start date filter
   * @param endDate - Optional end date filter
   * @returns Exported data
   */
  async export(format: 'csv' | 'json' = 'csv', startDate?: string, endDate?: string): Promise<{ data: string; format: string }> {
    const params: Record<string, string> = { format };
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    return this.client.request('GET', '/accounting/expenses/export', { params });
  }
}
