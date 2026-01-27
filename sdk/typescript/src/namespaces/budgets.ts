/**
 * Budgets Namespace API
 *
 * Provides a namespaced interface for budget management and cost control.
 * Critical for SME Starter Pack to manage spending limits and alerts.
 */

import type {
  Budget,
  BudgetList,
  BudgetAlert,
  BudgetAlertList,
  BudgetSummary,
  CreateBudgetRequest,
  UpdateBudgetRequest,
  PaginationParams,
  BudgetTransaction,
  BudgetTransactionList,
  BudgetTrendPoint,
  BudgetTrendsResponse,
} from '../types';

/**
 * Transaction filter options.
 */
export interface TransactionFilterOptions extends PaginationParams {
  date_from?: number;
  date_to?: number;
  user_id?: string;
}

/**
 * Trends options.
 */
export interface TrendsOptions {
  period?: 'hour' | 'day' | 'week' | 'month';
  limit?: number;
}

interface BudgetsClientInterface {
  listBudgets(params?: PaginationParams): Promise<BudgetList>;
  createBudget(body: CreateBudgetRequest): Promise<Budget>;
  getBudget(budgetId: string): Promise<Budget>;
  updateBudget(budgetId: string, body: UpdateBudgetRequest): Promise<Budget>;
  deleteBudget(budgetId: string): Promise<{ deleted: boolean }>;
  getBudgetAlerts(budgetId: string, params?: PaginationParams): Promise<BudgetAlertList>;
  acknowledgeBudgetAlert(budgetId: string, alertId: string): Promise<{ acknowledged: boolean }>;
  addBudgetOverride(budgetId: string, body: { user_id: string; limit: number; reason?: string }): Promise<{ added: boolean }>;
  removeBudgetOverride(budgetId: string, userId: string): Promise<{ removed: boolean }>;
  resetBudget(budgetId: string): Promise<{ reset: boolean; new_period_start: string }>;
  getBudgetSummary(): Promise<BudgetSummary>;
  checkBudget(body: {
    operation: string;
    estimated_cost: number;
    budget_id?: string;
  }): Promise<{ allowed: boolean; remaining_budget: number; warnings?: string[] }>;
  getBudgetTransactions(budgetId: string, params?: TransactionFilterOptions): Promise<BudgetTransactionList>;
  getBudgetTrends(budgetId: string, params?: TrendsOptions): Promise<BudgetTrendsResponse>;
  getOrgTrends(params?: TrendsOptions): Promise<BudgetTrendsResponse>;
}

/**
 * Budgets API namespace.
 *
 * Provides methods for managing budgets and cost control:
 * - Create and manage spending budgets
 * - Set up alert thresholds
 * - Check budget before operations
 * - Track spending across scopes
 *
 * Essential for SME cost management and preventing unexpected charges.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Create a monthly budget
 * const budget = await client.budgets.create({
 *   name: 'Monthly Debate Budget',
 *   limit_amount: 500,
 *   period: 'monthly',
 *   alert_threshold: 80
 * });
 *
 * // Check if operation is within budget
 * const { allowed, remaining_budget, warnings } = await client.budgets.check({
 *   operation: 'debate',
 *   estimated_cost: 2.50
 * });
 *
 * if (!allowed) {
 *   console.log('Budget exceeded! Cannot run debate.');
 * }
 *
 * // Get organization budget summary
 * const summary = await client.budgets.getSummary();
 * console.log(`${summary.exceeded_budgets} budgets exceeded`);
 * ```
 */
export class BudgetsAPI {
  constructor(private client: BudgetsClientInterface) {}

  // ===========================================================================
  // Budget CRUD
  // ===========================================================================

  /**
   * List all budgets for the organization.
   */
  async list(params?: PaginationParams): Promise<BudgetList> {
    return this.client.listBudgets(params);
  }

  /**
   * Create a new budget.
   *
   * @example
   * ```typescript
   * const budget = await client.budgets.create({
   *   name: 'Q1 AI Budget',
   *   limit_amount: 1000,
   *   period: 'quarterly',
   *   alert_threshold: 75
   * });
   * ```
   */
  async create(request: CreateBudgetRequest): Promise<Budget> {
    return this.client.createBudget(request);
  }

  /**
   * Get a budget by ID.
   */
  async get(budgetId: string): Promise<Budget> {
    return this.client.getBudget(budgetId);
  }

  /**
   * Update a budget.
   */
  async update(budgetId: string, request: UpdateBudgetRequest): Promise<Budget> {
    return this.client.updateBudget(budgetId, request);
  }

  /**
   * Delete a budget.
   */
  async delete(budgetId: string): Promise<{ deleted: boolean }> {
    return this.client.deleteBudget(budgetId);
  }

  // ===========================================================================
  // Budget Checks
  // ===========================================================================

  /**
   * Check if an operation is allowed within budget.
   *
   * Call this before expensive operations to ensure budget compliance.
   *
   * @example
   * ```typescript
   * const { allowed, warnings } = await client.budgets.check({
   *   operation: 'gauntlet',
   *   estimated_cost: 15.00
   * });
   *
   * if (warnings?.length) {
   *   console.log('Budget warnings:', warnings);
   * }
   * ```
   */
  async check(body: {
    operation: string;
    estimated_cost: number;
    budget_id?: string;
  }): Promise<{ allowed: boolean; remaining_budget: number; warnings?: string[] }> {
    return this.client.checkBudget(body);
  }

  /**
   * Get organization-wide budget summary.
   */
  async getSummary(): Promise<BudgetSummary> {
    return this.client.getBudgetSummary();
  }

  // ===========================================================================
  // Alerts
  // ===========================================================================

  /**
   * Get alerts for a budget.
   */
  async getAlerts(budgetId: string, params?: PaginationParams): Promise<BudgetAlertList> {
    return this.client.getBudgetAlerts(budgetId, params);
  }

  /**
   * Acknowledge a budget alert.
   */
  async acknowledgeAlert(budgetId: string, alertId: string): Promise<{ acknowledged: boolean }> {
    return this.client.acknowledgeBudgetAlert(budgetId, alertId);
  }

  // ===========================================================================
  // Overrides
  // ===========================================================================

  /**
   * Add a user-specific budget override.
   *
   * Useful for giving specific users higher limits for their work.
   */
  async addOverride(
    budgetId: string,
    body: { user_id: string; limit: number; reason?: string }
  ): Promise<{ added: boolean }> {
    return this.client.addBudgetOverride(budgetId, body);
  }

  /**
   * Remove a user-specific budget override.
   */
  async removeOverride(budgetId: string, userId: string): Promise<{ removed: boolean }> {
    return this.client.removeBudgetOverride(budgetId, userId);
  }

  // ===========================================================================
  // Period Management
  // ===========================================================================

  /**
   * Reset a budget period.
   *
   * Useful for manual resets or when changing budget parameters.
   */
  async reset(budgetId: string): Promise<{ reset: boolean; new_period_start: string }> {
    return this.client.resetBudget(budgetId);
  }

  // ===========================================================================
  // Transaction History
  // ===========================================================================

  /**
   * Get transaction history for a budget.
   *
   * Returns all recorded spending transactions with pagination and filtering.
   *
   * @example
   * ```typescript
   * // Get recent transactions
   * const { transactions, total } = await client.budgets.getTransactions('budget-123');
   *
   * // Filter by date range (unix timestamps)
   * const recent = await client.budgets.getTransactions('budget-123', {
   *   date_from: Date.now() / 1000 - 86400 * 7,  // Last 7 days
   *   limit: 100
   * });
   *
   * // Analyze spending
   * const totalSpent = transactions.reduce((sum, t) => sum + t.amount_usd, 0);
   * ```
   */
  async getTransactions(
    budgetId: string,
    params?: TransactionFilterOptions
  ): Promise<BudgetTransactionList> {
    return this.client.getBudgetTransactions(budgetId, params);
  }

  // ===========================================================================
  // Spending Trends
  // ===========================================================================

  /**
   * Get spending trends for a budget.
   *
   * Returns aggregated spending data for charts and analysis.
   *
   * @param budgetId - Budget ID to get trends for
   * @param params.period - Aggregation period: 'hour', 'day', 'week', 'month'
   * @param params.limit - Number of periods to return (default 30)
   *
   * @example
   * ```typescript
   * // Get daily spending for the last 30 days
   * const { trends } = await client.budgets.getTrends('budget-123', {
   *   period: 'day',
   *   limit: 30
   * });
   *
   * // Create a chart
   * trends.forEach(t => {
   *   console.log(`${t.period}: $${t.total_spent_usd} (${t.transaction_count} txns)`);
   * });
   *
   * // Weekly summary
   * const weekly = await client.budgets.getTrends('budget-123', { period: 'week' });
   * ```
   */
  async getTrends(budgetId: string, params?: TrendsOptions): Promise<BudgetTrendsResponse> {
    return this.client.getBudgetTrends(budgetId, params);
  }

  /**
   * Get organization-wide spending trends across all budgets.
   *
   * Useful for executive dashboards and org-level reporting.
   *
   * @example
   * ```typescript
   * // Get monthly org spending
   * const { trends } = await client.budgets.getOrgTrends({ period: 'month' });
   *
   * const totalYTD = trends.reduce((sum, t) => sum + t.total_spent_usd, 0);
   * console.log(`Year-to-date spend: $${totalYTD}`);
   * ```
   */
  async getOrgTrends(params?: TrendsOptions): Promise<BudgetTrendsResponse> {
    return this.client.getOrgTrends(params);
  }
}

// Re-export imported types for convenience
export type {
  Budget,
  BudgetAlert,
  BudgetSummary,
  CreateBudgetRequest,
  UpdateBudgetRequest,
  BudgetTransaction,
  BudgetTransactionList,
  BudgetTrendPoint,
  BudgetTrendsResponse,
};
