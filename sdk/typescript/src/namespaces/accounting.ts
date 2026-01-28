/**
 * Accounting Namespace API
 *
 * Provides a namespaced interface for QuickBooks Online and Gusto payroll integration.
 * Enables transaction sync, customer management, and financial reporting.
 */

import type { PaginationParams } from '../types';

// =============================================================================
// Type Definitions
// =============================================================================

/**
 * Company information.
 */
export interface Company {
  name: string;
  legalName: string;
  country: string;
  email: string;
}

/**
 * Financial statistics.
 */
export interface FinancialStats {
  receivables: number;
  payables: number;
  revenue: number;
  expenses: number;
  netIncome: number;
  openInvoices: number;
  overdueInvoices: number;
}

/**
 * Customer record.
 */
export interface Customer {
  id: string;
  displayName: string;
  companyName?: string;
  email?: string;
  balance: number;
  active: boolean;
}

/**
 * Transaction record (invoice or expense).
 */
export interface Transaction {
  id: string;
  type: 'Invoice' | 'Expense' | 'Payment';
  docNumber?: string;
  txnDate?: string;
  dueDate?: string;
  totalAmount: number;
  balance: number;
  customerName?: string;
  vendorName?: string;
  status: string;
}

/**
 * Accounting status response.
 */
export interface AccountingStatus {
  connected: boolean;
  company?: Company;
  stats?: FinancialStats;
  customers?: Customer[];
  transactions?: Transaction[];
  error?: string;
}

/**
 * Customer list response.
 */
export interface CustomerList {
  customers: Customer[];
  total: number;
}

/**
 * Transaction list response.
 */
export interface TransactionList {
  transactions: Transaction[];
  total: number;
}

/**
 * Report section.
 */
export interface ReportSection {
  name: string;
  items?: Array<{ name: string; amount: number }>;
  total: number;
}

/**
 * Financial report.
 */
export interface FinancialReport {
  title: string;
  period?: string;
  as_of?: string;
  sections: ReportSection[];
  netIncome?: number;
}

/**
 * Report response.
 */
export interface ReportResponse {
  report: FinancialReport;
  generated_at: string;
  mock?: boolean;
}

/**
 * List customers params.
 */
export interface ListCustomersParams extends PaginationParams {
  active?: boolean;
}

/**
 * List transactions params.
 */
export interface ListTransactionsParams extends PaginationParams {
  type?: 'all' | 'invoice' | 'expense';
  start_date?: string;
  end_date?: string;
}

/**
 * Report request.
 */
export interface ReportRequest {
  type: 'profit_loss' | 'balance_sheet' | 'ar_aging' | 'ap_aging';
  start_date: string;
  end_date: string;
}

// =============================================================================
// Gusto Payroll Types
// =============================================================================

/**
 * Gusto employee record.
 */
export interface Employee {
  id: string;
  first_name: string;
  last_name: string;
  email?: string;
  department?: string;
  job_title?: string;
  status: 'active' | 'terminated';
  hire_date?: string;
  termination_date?: string;
}

/**
 * Payroll run.
 */
export interface PayrollRun {
  id: string;
  pay_period_start: string;
  pay_period_end: string;
  check_date: string;
  status: 'pending' | 'processed' | 'paid';
  total_gross_pay: number;
  total_net_pay: number;
  total_taxes: number;
  employee_count: number;
}

/**
 * Payroll details.
 */
export interface PayrollDetails extends PayrollRun {
  employee_compensations: Array<{
    employee_id: string;
    employee_name: string;
    gross_pay: number;
    net_pay: number;
    taxes_withheld: number;
    deductions: number;
  }>;
}

/**
 * Journal entry for payroll.
 */
export interface JournalEntry {
  date: string;
  memo: string;
  lines: Array<{
    account: string;
    debit: number;
    credit: number;
    description: string;
  }>;
}

/**
 * Employee list response.
 */
export interface EmployeeList {
  employees: Employee[];
  total: number;
}

/**
 * Payroll list response.
 */
export interface PayrollList {
  payrolls: PayrollRun[];
  total: number;
}

// =============================================================================
// Client Interface
// =============================================================================

interface AccountingClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

// =============================================================================
// Accounting API Class
// =============================================================================

/**
 * Accounting API namespace.
 *
 * Provides methods for QuickBooks Online and Gusto payroll integration:
 * - OAuth connection flows
 * - Customer and transaction management
 * - Financial report generation
 * - Employee and payroll data
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Check QuickBooks status
 * const status = await client.accounting.getStatus();
 * if (status.connected) {
 *   console.log(`Connected to ${status.company?.name}`);
 *   console.log(`Open invoices: ${status.stats?.openInvoices}`);
 * }
 *
 * // List customers
 * const { customers } = await client.accounting.listCustomers({ active: true });
 *
 * // Generate P&L report
 * const report = await client.accounting.generateReport({
 *   type: 'profit_loss',
 *   start_date: '2025-01-01',
 *   end_date: '2025-01-31'
 * });
 * ```
 */
export class AccountingAPI {
  constructor(private client: AccountingClientInterface) {}

  // ===========================================================================
  // QuickBooks Connection
  // ===========================================================================

  /**
   * Get QuickBooks connection status and dashboard data.
   */
  async getStatus(): Promise<AccountingStatus> {
    return this.client.request('GET', '/api/v2/accounting/status');
  }

  /**
   * Initiate QuickBooks OAuth connection.
   *
   * @returns URL to redirect user for OAuth.
   */
  async connect(): Promise<{ auth_url: string }> {
    return this.client.request('POST', '/api/v2/accounting/connect');
  }

  /**
   * Disconnect QuickBooks integration.
   */
  async disconnect(): Promise<{ success: boolean; message?: string }> {
    return this.client.request('POST', '/api/v2/accounting/disconnect');
  }

  // ===========================================================================
  // Customers
  // ===========================================================================

  /**
   * List QuickBooks customers.
   *
   * @param params.active - Filter by active status.
   * @param params.limit - Maximum results.
   * @param params.offset - Pagination offset.
   */
  async listCustomers(params?: ListCustomersParams): Promise<CustomerList> {
    return this.client.request('GET', '/api/v2/accounting/customers', {
      params: params as Record<string, unknown>,
    });
  }

  // ===========================================================================
  // Transactions
  // ===========================================================================

  /**
   * List transactions (invoices, expenses).
   *
   * @param params.type - Filter by type (all, invoice, expense).
   * @param params.start_date - Filter from date (ISO format).
   * @param params.end_date - Filter to date (ISO format).
   */
  async listTransactions(params?: ListTransactionsParams): Promise<TransactionList> {
    return this.client.request('GET', '/api/v2/accounting/transactions', {
      params: params as Record<string, unknown>,
    });
  }

  // ===========================================================================
  // Reports
  // ===========================================================================

  /**
   * Generate a financial report.
   *
   * @param request.type - Report type (profit_loss, balance_sheet, ar_aging, ap_aging).
   * @param request.start_date - Report start date (ISO format).
   * @param request.end_date - Report end date (ISO format).
   *
   * @example
   * ```typescript
   * const { report } = await client.accounting.generateReport({
   *   type: 'profit_loss',
   *   start_date: '2025-01-01',
   *   end_date: '2025-03-31'
   * });
   *
   * for (const section of report.sections) {
   *   console.log(`${section.name}: $${section.total}`);
   * }
   * ```
   */
  async generateReport(request: ReportRequest): Promise<ReportResponse> {
    return this.client.request('POST', '/api/v2/accounting/reports', {
      json: request as unknown as Record<string, unknown>,
    });
  }

  // ===========================================================================
  // Gusto Payroll
  // ===========================================================================

  /**
   * Gusto payroll namespace.
   */
  gusto = {
    /**
     * Get Gusto connection status.
     */
    getStatus: async (): Promise<{ connected: boolean; company_name?: string }> => {
      return this.client.request('GET', '/api/v2/gusto/status');
    },

    /**
     * Initiate Gusto OAuth connection.
     */
    connect: async (): Promise<{ auth_url: string }> => {
      return this.client.request('POST', '/api/v2/gusto/connect');
    },

    /**
     * Disconnect Gusto integration.
     */
    disconnect: async (): Promise<{ success: boolean; message?: string }> => {
      return this.client.request('POST', '/api/v2/gusto/disconnect');
    },

    /**
     * List employees.
     */
    listEmployees: async (params?: PaginationParams): Promise<EmployeeList> => {
      return this.client.request('GET', '/api/v2/gusto/employees', {
        params: params as Record<string, unknown>,
      });
    },

    /**
     * List payroll runs.
     */
    listPayrolls: async (params?: PaginationParams): Promise<PayrollList> => {
      return this.client.request('GET', '/api/v2/gusto/payrolls', {
        params: params as Record<string, unknown>,
      });
    },

    /**
     * Get payroll run details.
     *
     * @param payrollId - Payroll run ID.
     */
    getPayroll: async (payrollId: string): Promise<PayrollDetails> => {
      return this.client.request('GET', `/api/v2/gusto/payrolls/${payrollId}`);
    },

    /**
     * Generate journal entry for a payroll run.
     *
     * Creates a journal entry that can be imported into QuickBooks
     * or other accounting software.
     *
     * @param payrollId - Payroll run ID.
     */
    generateJournalEntry: async (payrollId: string): Promise<JournalEntry> => {
      return this.client.request('POST', `/api/v2/gusto/payrolls/${payrollId}/journal-entry`);
    },
  };
}
