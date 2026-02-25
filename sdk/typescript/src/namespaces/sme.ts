/**
 * SME Namespace API
 *
 * Provides a namespaced interface for SME (Small/Medium Enterprise) operations
 * including budgets, Slack integration, and workspace management.
 */

import type { AragoraClient } from '../client';

type RequestOptions = {
  params?: Record<string, unknown>;
  body?: unknown;
};

type RequestMethod = (
  method: string,
  path: string,
  options?: RequestOptions
) => Promise<unknown>;

type LegacyMethod = (...args: unknown[]) => Promise<unknown>;

type CompatClient = AragoraClient & {
  request?: RequestMethod;
} & Record<string, unknown>;

/**
 * SME API namespace.
 *
 * Provides methods for SME features:
 * - Budget management
 * - Slack integration (OAuth, workspaces, channels, subscriptions)
 */
export class SMEAPI {
  constructor(private client: AragoraClient) {}

  private request<T>(
    method: string,
    path: string,
    options?: RequestOptions
  ): Promise<T> {
    const request = (this.client as CompatClient).request;
    if (typeof request !== 'function') {
      throw new TypeError('this.client.request is not a function');
    }
    return request.apply(this.client, [method, path, options]) as Promise<T>;
  }

  private invoke<T>(
    legacyMethod: string,
    legacyArgs: unknown[],
    method: string,
    path: string,
    options?: RequestOptions
  ): Promise<T> {
    const legacy = (this.client as CompatClient)[legacyMethod];
    if (typeof legacy === 'function') {
      return (legacy as LegacyMethod).apply(this.client, legacyArgs) as Promise<T>;
    }
    return this.request<T>(method, path, options);
  }

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
   * List channels for legacy Slack route compatibility.
   * @route GET /api/v1/sme/slack/channels
   * @route GET /api/v1/sme/slack/channels/{workspace_id}
   */
  async listSlackChannels(workspaceId?: string): Promise<Record<string, unknown>> {
    const path = workspaceId
      ? `/api/v1/sme/slack/channels/${encodeURIComponent(workspaceId)}`
      : '/api/v1/sme/slack/channels';
    return this.client.request('GET', path) as Promise<Record<string, unknown>>;
  }

  /**
   * List channels via legacy unversioned path.
   * @route GET /api/sme/slack/channels
   */
  async listSlackChannelsCompat(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/sme/slack/channels') as Promise<Record<string, unknown>>;
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
   * Get an SME workflow by ID.
   * @route GET /api/v1/sme/workflows/{workflow_id}
   */
  async getWorkflow(workflowId: string): Promise<Record<string, unknown>> {
    const legacy = (this.client as CompatClient).getSMEWorkflow;
    if (typeof legacy === 'function') {
      return (legacy as LegacyMethod).apply(this.client, [workflowId]) as Promise<Record<string, unknown>>;
    }
    return this.request<Record<string, unknown>>(
      'GET',
      `/api/v1/sme/workflows/${encodeURIComponent(workflowId)}`
    );
  }

  /**
   * Execute an SME workflow by name.
   * @route POST /api/v1/sme/workflows
   *
   * @param workflowName - Name of the workflow to execute
   * @param inputs - Input parameters for the workflow
   */
  async executeWorkflow(workflowName: string, inputs?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.invoke<Record<string, unknown>>(
      'executeSMEWorkflow',
      [workflowName, inputs],
      'POST',
      `/api/v1/sme/workflows/${encodeURIComponent(workflowName)}`,
      { body: inputs }
    );
  }

  /**
   * List SME workflows (supports filters and pagination).
   * @route GET /api/v1/sme/workflows
   */
  async listWorkflows(
    params?: { category?: string; industry?: string; limit?: number; offset?: number }
  ): Promise<Record<string, unknown>> {
    return this.invoke<Record<string, unknown>>(
      'listSMEWorkflows',
      [params],
      'GET',
      '/api/v1/sme/workflows',
      { params: params as Record<string, unknown> }
    );
  }

  // ===========================================================================
  // Onboarding
  // ===========================================================================

  /**
   * Get onboarding status.
   * @route GET /api/v1/onboarding/status
   */
  async getOnboardingStatus(): Promise<Record<string, unknown>> {
    return this.invoke<Record<string, unknown>>(
      'getOnboardingStatus',
      [],
      'GET',
      '/api/v1/onboarding/status'
    );
  }

  /**
   * Complete onboarding.
   * @route POST /api/v1/onboarding/complete
   */
  async completeOnboarding(request?: {
    first_debate_id?: string;
    template_used?: string;
  }): Promise<Record<string, unknown>> {
    return this.invoke<Record<string, unknown>>(
      'completeOnboarding',
      [request],
      'POST',
      '/api/v1/onboarding/complete',
      { body: request }
    );
  }

  // ===========================================================================
  // Quick Start Helpers
  // ===========================================================================

  async quickInvoice(options: {
    customerEmail: string;
    customerName: string;
    items: Array<{ name: string; price: number; quantity?: number }>;
    dueDate?: string;
  }): Promise<Record<string, unknown>> {
    return this.executeWorkflow('invoice', {
      inputs: {
        customer_email: options.customerEmail,
        customer_name: options.customerName,
        items: options.items.map((item) => ({
          name: item.name,
          unit_price: item.price,
          quantity: item.quantity ?? 1,
        })),
        due_date: options.dueDate,
      },
    });
  }

  async quickInventoryCheck(options: {
    productId: string;
    minThreshold: number;
    notificationEmail: string;
  }): Promise<Record<string, unknown>> {
    return this.executeWorkflow('inventory', {
      inputs: {
        product_id: options.productId,
        min_threshold: options.minThreshold,
        notification_email: options.notificationEmail,
      },
    });
  }

  async quickReport(options: {
    type: string;
    period: string;
    format?: string;
    email?: string;
  }): Promise<Record<string, unknown>> {
    return this.executeWorkflow('report', {
      inputs: {
        report_type: options.type,
        period: options.period,
        format: options.format ?? 'pdf',
        delivery_email: options.email,
      },
    });
  }

  async quickFollowup(options: {
    customerId: string;
    type: string;
    message?: string;
    delayDays?: number;
  }): Promise<Record<string, unknown>> {
    return this.executeWorkflow('followup', {
      inputs: {
        customer_id: options.customerId,
        followup_type: options.type,
        custom_message: options.message,
        delay_days: options.delayDays ?? 0,
      },
    });
  }
}
