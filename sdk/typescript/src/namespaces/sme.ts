/**
 * SME Namespace API
 *
 * Provides a namespaced interface for SME (Small/Medium Enterprise) operations
 * including budgets, Slack integration, and workspace management.
 */

import type { AragoraClient } from '../client';

/**
 * SME API namespace.
 *
 * Provides methods for SME features:
 * - Budget management
 * - Slack integration (OAuth, workspaces, channels, subscriptions)
 */
export class SMEAPI {
  constructor(private client: AragoraClient) {}

  // ===========================================================================
  // Budgets
  // ===========================================================================

  /**
   * List budgets.
   * @route GET /api/v1/sme/budgets
   */
  async listBudgets(params?: { limit?: number; offset?: number }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/budgets', {
      params: params as Record<string, unknown>,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Create a budget.
   * @route POST /api/v1/sme/budgets
   */
  async createBudget(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/sme/budgets', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get a budget by ID.
   * @route GET /api/v1/sme/budgets/{budget_id}
   */
  async getBudget(budgetId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/sme/budgets/${encodeURIComponent(budgetId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Update a budget.
   * @route PATCH /api/v1/sme/budgets/{budget_id}
   */
  async updateBudget(budgetId: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'PATCH',
      `/api/v1/sme/budgets/${encodeURIComponent(budgetId)}`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Delete a budget.
   * @route DELETE /api/v1/sme/budgets/{budget_id}
   */
  async deleteBudget(budgetId: string): Promise<void> {
    return this.client.request(
      'DELETE',
      `/api/v1/sme/budgets/${encodeURIComponent(budgetId)}`
    ) as Promise<void>;
  }

  /**
   * Get budget alerts.
   * @route GET /api/v1/sme/budgets/{budget_id}/alerts
   */
  async getBudgetAlerts(budgetId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/sme/budgets/${encodeURIComponent(budgetId)}/alerts`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get budget transactions.
   * @route GET /api/v1/sme/budgets/{budget_id}/transactions
   */
  async getBudgetTransactions(budgetId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/sme/budgets/${encodeURIComponent(budgetId)}/transactions`
    ) as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Slack Integration
  // ===========================================================================

  /**
   * Start Slack OAuth flow.
   * @route GET /api/v1/sme/slack/oauth/start
   */
  async slackOAuthStart(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/slack/oauth/start') as Promise<Record<string, unknown>>;
  }

  /**
   * Slack OAuth callback.
   * @route GET /api/v1/sme/slack/oauth/callback
   */
  async slackOAuthCallback(params?: { code?: string; state?: string }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/slack/oauth/callback', {
      params: params as Record<string, unknown>,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Subscribe to Slack notifications.
   * @route POST /api/v1/sme/slack/subscribe
   */
  async slackSubscribe(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/sme/slack/subscribe', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * List Slack subscriptions.
   * @route GET /api/v1/sme/slack/subscriptions
   */
  async listSlackSubscriptions(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/slack/subscriptions') as Promise<Record<string, unknown>>;
  }

  /**
   * Delete a Slack subscription.
   * @route DELETE /api/v1/sme/slack/subscriptions/{subscription_id}
   */
  async deleteSlackSubscription(subscriptionId: string): Promise<void> {
    return this.client.request(
      'DELETE',
      `/api/v1/sme/slack/subscriptions/${encodeURIComponent(subscriptionId)}`
    ) as Promise<void>;
  }

  /**
   * List Slack workspaces.
   * @route GET /api/v1/sme/slack/workspaces
   */
  async listSlackWorkspaces(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/slack/workspaces') as Promise<Record<string, unknown>>;
  }

  /**
   * Create a Slack workspace connection.
   * @route POST /api/v1/sme/slack/workspaces
   */
  async createSlackWorkspace(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/sme/slack/workspaces', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get a Slack workspace by ID.
   * @route GET /api/v1/sme/slack/workspaces/{workspace_id}
   */
  async getSlackWorkspace(workspaceId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/sme/slack/workspaces/${encodeURIComponent(workspaceId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Update a Slack workspace.
   * @route PATCH /api/v1/sme/slack/workspaces/{workspace_id}
   */
  async updateSlackWorkspace(workspaceId: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'PATCH',
      `/api/v1/sme/slack/workspaces/${encodeURIComponent(workspaceId)}`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Delete a Slack workspace.
   * @route DELETE /api/v1/sme/slack/workspaces/{workspace_id}
   */
  async deleteSlackWorkspace(workspaceId: string): Promise<void> {
    return this.client.request(
      'DELETE',
      `/api/v1/sme/slack/workspaces/${encodeURIComponent(workspaceId)}`
    ) as Promise<void>;
  }

  /**
   * Get channels for a Slack workspace.
   * @route GET /api/v1/sme/slack/workspaces/{workspace_id}/channels
   */
  async getSlackWorkspaceChannels(workspaceId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/sme/slack/workspaces/${encodeURIComponent(workspaceId)}/channels`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Test a Slack workspace connection.
   * @route POST /api/v1/sme/slack/workspaces/{workspace_id}/test
   */
  async testSlackWorkspace(workspaceId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/sme/slack/workspaces/${encodeURIComponent(workspaceId)}/test`
    ) as Promise<Record<string, unknown>>;
  }
}
