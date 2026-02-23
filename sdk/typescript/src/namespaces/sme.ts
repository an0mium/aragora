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

  // ===========================================================================
  // Budget Check
  // ===========================================================================

  /**
   * Check if an action is within budget.
   * @route GET /api/v1/sme/budgets/check
   */
  async checkBudget(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/budgets/check', {
      params,
    }) as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Receipt Delivery
  // ===========================================================================

  /**
   * Get receipt delivery configuration.
   * @route GET /api/v1/sme/receipts/delivery/config
   */
  async getDeliveryConfig(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/receipts/delivery/config') as Promise<Record<string, unknown>>;
  }

  /**
   * Get receipt delivery history.
   * @route GET /api/v1/sme/receipts/delivery/history
   */
  async getDeliveryHistory(params?: { limit?: number; offset?: number }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/receipts/delivery/history', {
      params: params as Record<string, unknown>,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get receipt delivery statistics.
   * @route GET /api/v1/sme/receipts/delivery/stats
   */
  async getDeliveryStats(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/receipts/delivery/stats') as Promise<Record<string, unknown>>;
  }

  /**
   * Send a test receipt delivery.
   * @route GET /api/v1/sme/receipts/delivery/test
   */
  async testDelivery(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/receipts/delivery/test') as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Success Metrics
  // ===========================================================================

  /**
   * Get overall SME success metrics.
   * @route GET /api/v1/sme/success
   */
  async getSuccessOverview(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/success') as Promise<Record<string, unknown>>;
  }

  /**
   * Get CFO-oriented success metrics (cost savings, ROI).
   * @route GET /api/v1/sme/success/cfo
   */
  async getSuccessCfo(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/success/cfo') as Promise<Record<string, unknown>>;
  }

  /**
   * Get HR-oriented success metrics (time saved, adoption).
   * @route GET /api/v1/sme/success/hr
   */
  async getSuccessHr(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/success/hr') as Promise<Record<string, unknown>>;
  }

  /**
   * Get AI-generated success insights and recommendations.
   * @route GET /api/v1/sme/success/insights
   */
  async getSuccessInsights(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/success/insights') as Promise<Record<string, unknown>>;
  }

  /**
   * Get achieved and upcoming success milestones.
   * @route GET /api/v1/sme/success/milestones
   */
  async getSuccessMilestones(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/success/milestones') as Promise<Record<string, unknown>>;
  }

  /**
   * Get PM-oriented success metrics (decisions made, velocity).
   * @route GET /api/v1/sme/success/pm
   */
  async getSuccessPm(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/success/pm') as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Microsoft Teams Integration
  // ===========================================================================

  /**
   * List Microsoft Teams channels.
   * @route GET /api/v1/sme/teams/channels
   */
  async listTeamsChannels(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/teams/channels') as Promise<Record<string, unknown>>;
  }

  /**
   * Start Microsoft Teams OAuth flow.
   * @route GET /api/v1/sme/teams/oauth/start
   */
  async teamsOAuthStart(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/teams/oauth/start') as Promise<Record<string, unknown>>;
  }

  /**
   * Handle Microsoft Teams OAuth callback.
   * @route GET /api/v1/sme/teams/oauth/callback
   */
  async teamsOAuthCallback(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/teams/oauth/callback', {
      params,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Create a Microsoft Teams subscription for notifications.
   * @route POST /api/v1/sme/teams/subscribe
   */
  async teamsSubscribe(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/sme/teams/subscribe', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * List Microsoft Teams subscriptions.
   * @route GET /api/v1/sme/teams/subscriptions
   */
  async listTeamsSubscriptions(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/teams/subscriptions') as Promise<Record<string, unknown>>;
  }

  /**
   * List connected Microsoft Teams tenants.
   * @route GET /api/v1/sme/teams/tenants
   */
  async listTeamsTenants(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/teams/tenants') as Promise<Record<string, unknown>>;
  }

  /**
   * List Microsoft Teams workspaces.
   * @route GET /api/v1/sme/teams/workspaces
   */
  async listTeamsWorkspaces(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/teams/workspaces') as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Workflows
  // ===========================================================================

  /**
   * List SME-specific pre-built workflows.
   * @route GET /api/v1/sme/workflows
   */
  async listWorkflows(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/sme/workflows') as Promise<Record<string, unknown>>;
  }

  /**
   * Execute an SME workflow by name.
   * @route POST /api/v1/sme/workflows
   *
   * @param workflowName - Name of the workflow to execute
   * @param inputs - Input parameters for the workflow
   */
  async executeWorkflow(workflowName: string, inputs?: Record<string, unknown>): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = { workflow: workflowName };
    if (inputs) body.inputs = inputs;
    return this.client.request('POST', '/api/v1/sme/workflows', {
      body,
    }) as Promise<Record<string, unknown>>;
  }
}
