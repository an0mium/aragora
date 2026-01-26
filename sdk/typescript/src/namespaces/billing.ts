/**
 * Billing Namespace API
 *
 * Provides a namespaced interface for billing and subscription management.
 * Essential for SME Starter Pack cost management.
 */

import type {
  BillingPlan,
  BillingPlanList,
  BillingUsage,
  Subscription,
  InvoiceList,
  UsageForecast,
  PaginationParams,
} from '../types';

/**
 * Interface for the internal client methods used by BillingAPI.
 */
interface BillingClientInterface {
  listBillingPlans(): Promise<BillingPlanList>;
  getBillingUsage(params?: { period?: string }): Promise<BillingUsage>;
  getBillingSubscription(): Promise<Subscription>;
  createCheckoutSession(body: {
    plan_id: string;
    success_url?: string;
    cancel_url?: string;
  }): Promise<{ session_id: string; checkout_url: string }>;
  createBillingPortalSession(returnUrl?: string): Promise<{ url: string }>;
  cancelSubscription(): Promise<{ cancelled: boolean; effective_date: string }>;
  resumeSubscription(): Promise<{ resumed: boolean }>;
  listBillingInvoices(params?: PaginationParams): Promise<InvoiceList>;
  getUsageForecast(): Promise<UsageForecast>;
  exportBillingUsage(params?: { format?: string; period?: string }): Promise<{ download_url: string }>;
}

/**
 * Billing API namespace.
 *
 * Provides methods for managing billing and subscriptions:
 * - View available plans and current subscription
 * - Track usage and costs
 * - Manage checkout and billing portal
 * - Access invoices and forecasts
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // View available plans
 * const { plans } = await client.billing.listPlans();
 *
 * // Check current usage
 * const usage = await client.billing.getUsage();
 * console.log(`${usage.debates_used} debates used this period`);
 *
 * // Get usage forecast
 * const forecast = await client.billing.getForecast();
 * console.log(`Projected cost: $${forecast.projected_monthly_cost}`);
 *
 * // Access billing portal
 * const { url } = await client.billing.getPortalUrl();
 * window.open(url);
 * ```
 */
export class BillingAPI {
  constructor(private client: BillingClientInterface) {}

  // ===========================================================================
  // Plans and Subscription
  // ===========================================================================

  /**
   * List available billing plans.
   */
  async listPlans(): Promise<BillingPlanList> {
    return this.client.listBillingPlans();
  }

  /**
   * Get current subscription details.
   */
  async getSubscription(): Promise<Subscription> {
    return this.client.getBillingSubscription();
  }

  /**
   * Create a checkout session for a new subscription.
   *
   * @example
   * ```typescript
   * const { checkout_url } = await client.billing.createCheckout({
   *   plan_id: 'pro-monthly',
   *   success_url: 'https://myapp.com/billing/success',
   *   cancel_url: 'https://myapp.com/billing/cancel'
   * });
   * window.location.href = checkout_url;
   * ```
   */
  async createCheckout(body: {
    plan_id: string;
    success_url?: string;
    cancel_url?: string;
  }): Promise<{ session_id: string; checkout_url: string }> {
    return this.client.createCheckoutSession(body);
  }

  /**
   * Get a URL to the billing portal for self-service management.
   */
  async getPortalUrl(returnUrl?: string): Promise<{ url: string }> {
    return this.client.createBillingPortalSession(returnUrl);
  }

  /**
   * Cancel the current subscription.
   * Cancellation takes effect at the end of the current billing period.
   */
  async cancel(): Promise<{ cancelled: boolean; effective_date: string }> {
    return this.client.cancelSubscription();
  }

  /**
   * Resume a cancelled subscription before the cancellation date.
   */
  async resume(): Promise<{ resumed: boolean }> {
    return this.client.resumeSubscription();
  }

  // ===========================================================================
  // Usage and Costs
  // ===========================================================================

  /**
   * Get current billing period usage.
   *
   * @param period - Optional period to query (e.g., '2024-01' for January 2024)
   */
  async getUsage(period?: string): Promise<BillingUsage> {
    return this.client.getBillingUsage(period ? { period } : undefined);
  }

  /**
   * Get usage forecast for the current period.
   */
  async getForecast(): Promise<UsageForecast> {
    return this.client.getUsageForecast();
  }

  /**
   * Export usage data.
   *
   * @param format - Export format ('csv' or 'json')
   * @param period - Optional period to export
   */
  async exportUsage(format?: string, period?: string): Promise<{ download_url: string }> {
    return this.client.exportBillingUsage({ format, period });
  }

  // ===========================================================================
  // Invoices
  // ===========================================================================

  /**
   * List invoices with optional pagination.
   */
  async listInvoices(params?: PaginationParams): Promise<InvoiceList> {
    return this.client.listBillingInvoices(params);
  }
}

// Re-export types for convenience
export type { BillingPlan, BillingUsage, Subscription, UsageForecast };
