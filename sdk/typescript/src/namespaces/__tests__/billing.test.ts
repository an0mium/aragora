/**
 * Billing Namespace Tests
 *
 * Comprehensive tests for the billing namespace API including:
 * - Plans and subscriptions
 * - Usage tracking
 * - Checkout and billing portal
 * - Invoices and forecasts
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { BillingAPI } from '../billing';

interface MockClient {
  listBillingPlans: Mock;
  getBillingUsage: Mock;
  getSubscription: Mock;
  createCheckoutSession: Mock;
  createBillingPortalSession: Mock;
  cancelSubscription: Mock;
  resumeSubscription: Mock;
  getInvoiceHistory: Mock;
  getUsageForecast: Mock;
  exportUsageData: Mock;
}

describe('BillingAPI Namespace', () => {
  let api: BillingAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      listBillingPlans: vi.fn(),
      getBillingUsage: vi.fn(),
      getSubscription: vi.fn(),
      createCheckoutSession: vi.fn(),
      createBillingPortalSession: vi.fn(),
      cancelSubscription: vi.fn(),
      resumeSubscription: vi.fn(),
      getInvoiceHistory: vi.fn(),
      getUsageForecast: vi.fn(),
      exportUsageData: vi.fn(),
    };
    api = new BillingAPI(mockClient as any);
  });

  // ===========================================================================
  // Plans and Subscription
  // ===========================================================================

  describe('Plans and Subscription', () => {
    it('should list available plans', async () => {
      const mockPlans = {
        plans: [
          { id: 'starter', name: 'Starter', price: 29, interval: 'month' },
          { id: 'pro', name: 'Professional', price: 99, interval: 'month' },
          { id: 'enterprise', name: 'Enterprise', price: 299, interval: 'month' },
        ],
      };
      mockClient.listBillingPlans.mockResolvedValue(mockPlans);

      const result = await api.listPlans();

      expect(mockClient.listBillingPlans).toHaveBeenCalled();
      expect(result.plans).toHaveLength(3);
      expect(result.plans[0].id).toBe('starter');
    });

    it('should get current subscription', async () => {
      const mockSubscription = {
        id: 'sub_123',
        plan_id: 'pro',
        status: 'active',
        current_period_start: '2024-01-01',
        current_period_end: '2024-02-01',
        cancel_at_period_end: false,
      };
      mockClient.getSubscription.mockResolvedValue(mockSubscription);

      const result = await api.getSubscription();

      expect(mockClient.getSubscription).toHaveBeenCalled();
      expect(result.plan_id).toBe('pro');
      expect(result.status).toBe('active');
    });

    it('should create checkout session', async () => {
      const mockCheckout = {
        session_id: 'sess_123',
        checkout_url: 'https://checkout.stripe.com/...',
      };
      mockClient.createCheckoutSession.mockResolvedValue(mockCheckout);

      const result = await api.createCheckout({
        plan_id: 'pro-monthly',
        success_url: 'https://app.example.com/success',
        cancel_url: 'https://app.example.com/cancel',
      });

      expect(mockClient.createCheckoutSession).toHaveBeenCalledWith({
        plan_id: 'pro-monthly',
        success_url: 'https://app.example.com/success',
        cancel_url: 'https://app.example.com/cancel',
      });
      expect(result.checkout_url).toContain('checkout.stripe.com');
    });

    it('should get billing portal URL', async () => {
      const mockPortal = { url: 'https://billing.stripe.com/session/...' };
      mockClient.createBillingPortalSession.mockResolvedValue(mockPortal);

      const result = await api.getPortalUrl('https://app.example.com/billing');

      expect(mockClient.createBillingPortalSession).toHaveBeenCalledWith(
        'https://app.example.com/billing'
      );
      expect(result.url).toContain('billing.stripe.com');
    });

    it('should cancel subscription', async () => {
      const mockCancel = {
        cancelled: true,
        effective_date: '2024-02-01',
      };
      mockClient.cancelSubscription.mockResolvedValue(mockCancel);

      const result = await api.cancel();

      expect(mockClient.cancelSubscription).toHaveBeenCalled();
      expect(result.cancelled).toBe(true);
    });

    it('should resume subscription', async () => {
      const mockResume = { resumed: true };
      mockClient.resumeSubscription.mockResolvedValue(mockResume);

      const result = await api.resume();

      expect(mockClient.resumeSubscription).toHaveBeenCalled();
      expect(result.resumed).toBe(true);
    });
  });

  // ===========================================================================
  // Usage and Costs
  // ===========================================================================

  describe('Usage and Costs', () => {
    it('should get current period usage', async () => {
      const mockUsage = {
        debates_used: 150,
        debates_limit: 500,
        agent_calls: 3000,
        storage_mb: 250,
        period_start: '2024-01-01',
        period_end: '2024-02-01',
      };
      mockClient.getBillingUsage.mockResolvedValue(mockUsage);

      const result = await api.getUsage();

      expect(mockClient.getBillingUsage).toHaveBeenCalledWith(undefined);
      expect(result.debates_used).toBe(150);
    });

    it('should get usage for specific period', async () => {
      const mockUsage = {
        debates_used: 200,
        debates_limit: 500,
        period: '2023-12',
      };
      mockClient.getBillingUsage.mockResolvedValue(mockUsage);

      const result = await api.getUsage('2023-12');

      expect(mockClient.getBillingUsage).toHaveBeenCalledWith({ period: '2023-12' });
      expect(result.debates_used).toBe(200);
    });

    it('should get usage forecast', async () => {
      const mockForecast = {
        projected_debates: 450,
        projected_agent_calls: 9000,
        projected_monthly_cost: 89.50,
        projection_confidence: 0.85,
      };
      mockClient.getUsageForecast.mockResolvedValue(mockForecast);

      const result = await api.getForecast();

      expect(mockClient.getUsageForecast).toHaveBeenCalled();
      expect(result.projected_monthly_cost).toBe(89.50);
    });

    it('should export usage data', async () => {
      const mockExport = { download_url: 'https://storage.example.com/exports/usage-2024.csv' };
      mockClient.exportUsageData.mockResolvedValue(mockExport);

      const result = await api.exportUsage('2024-01-01', '2024-01-31');

      expect(mockClient.exportUsageData).toHaveBeenCalledWith({
        start_date: '2024-01-01',
        end_date: '2024-01-31',
      });
      expect(result.download_url).toContain('usage-2024');
    });
  });

  // ===========================================================================
  // Invoices
  // ===========================================================================

  describe('Invoices', () => {
    it('should list invoices', async () => {
      const mockInvoices = {
        invoices: [
          { id: 'inv_1', amount: 9900, status: 'paid', created_at: '2024-01-01' },
          { id: 'inv_2', amount: 9900, status: 'paid', created_at: '2023-12-01' },
        ],
        total: 2,
      };
      mockClient.getInvoiceHistory.mockResolvedValue(mockInvoices);

      const result = await api.listInvoices();

      expect(mockClient.getInvoiceHistory).toHaveBeenCalled();
      expect(result.invoices).toHaveLength(2);
    });

    it('should list invoices with pagination', async () => {
      const mockInvoices = {
        invoices: [{ id: 'inv_3', amount: 9900, status: 'paid' }],
        total: 10,
        limit: 1,
        offset: 2,
      };
      mockClient.getInvoiceHistory.mockResolvedValue(mockInvoices);

      const result = await api.listInvoices({ limit: 1, offset: 2 });

      expect(mockClient.getInvoiceHistory).toHaveBeenCalledWith({ limit: 1, offset: 2 });
      expect(result.invoices).toHaveLength(1);
    });
  });
});
