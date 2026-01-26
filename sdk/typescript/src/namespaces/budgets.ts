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
} from '../types';

/**
 * Interface for the internal client methods used by BudgetsAPI.
 */
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
}

// Re-export types for convenience
export type { Budget, BudgetAlert, BudgetSummary, CreateBudgetRequest, UpdateBudgetRequest };
