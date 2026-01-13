/**
 * Aragora SDK Billing API Tests
 *
 * Tests for subscription, usage, and invoice management endpoints.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { AragoraClient, AragoraError } from '../src/client';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('BillingAPI', () => {
  let client: AragoraClient;

  beforeEach(() => {
    client = new AragoraClient({
      baseUrl: 'http://localhost:8080',
      apiKey: 'test-api-key',
    });
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  // ===========================================================================
  // Plans
  // ===========================================================================

  describe('plans()', () => {
    it('should list available subscription plans', async () => {
      const mockResponse = {
        plans: [
          {
            plan_id: 'free',
            name: 'Free',
            price_cents: 0,
            billing_period: 'monthly',
            features: ['5 debates/month', 'Basic agents'],
            limits: { debates_per_month: 5, api_calls_per_day: 100 },
          },
          {
            plan_id: 'pro',
            name: 'Pro',
            price_cents: 2900,
            billing_period: 'monthly',
            features: ['100 debates/month', 'All agents', 'Priority support'],
            limits: { debates_per_month: 100, api_calls_per_day: 10000 },
          },
          {
            plan_id: 'enterprise',
            name: 'Enterprise',
            price_cents: 19900,
            billing_period: 'monthly',
            features: ['Unlimited debates', 'Custom agents', 'SLA'],
            limits: { debates_per_month: -1, api_calls_per_day: -1 },
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.plans();
      expect(result).toHaveLength(3);
      expect(result[0].plan_id).toBe('free');
      expect(result[1].price_cents).toBe(2900);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/billing/plans',
        expect.objectContaining({
          method: 'GET',
        })
      );
    });
  });

  // ===========================================================================
  // Usage
  // ===========================================================================

  describe('usage()', () => {
    it('should return current usage metrics', async () => {
      const mockResponse = {
        debates_used: 45,
        debates_limit: 100,
        api_calls_used: 5000,
        api_calls_limit: 10000,
        storage_used_mb: 250,
        storage_limit_mb: 1000,
        period_start: '2024-01-01T00:00:00Z',
        period_end: '2024-01-31T23:59:59Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.usage();
      expect(result.debates_used).toBe(45);
      expect(result.debates_limit).toBe(100);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/billing/usage',
        expect.objectContaining({
          method: 'GET',
        })
      );
    });

    it('should throw on unauthorized access', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({
          error: 'Unauthorized',
          code: 'UNAUTHORIZED',
        }),
      });

      await expect(client.billing.usage()).rejects.toThrow(AragoraError);
    });
  });

  // ===========================================================================
  // Subscription
  // ===========================================================================

  describe('subscription()', () => {
    it('should return current subscription', async () => {
      const mockResponse = {
        subscription_id: 'sub_123abc',
        plan_id: 'pro',
        status: 'active',
        current_period_start: '2024-01-01T00:00:00Z',
        current_period_end: '2024-01-31T23:59:59Z',
        cancel_at_period_end: false,
        trial_end: null,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.subscription();
      expect(result.subscription_id).toBe('sub_123abc');
      expect(result.plan_id).toBe('pro');
      expect(result.status).toBe('active');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/billing/subscription',
        expect.objectContaining({
          method: 'GET',
        })
      );
    });

    it('should return subscription with trial', async () => {
      const mockResponse = {
        subscription_id: 'sub_trial123',
        plan_id: 'pro',
        status: 'trialing',
        current_period_start: '2024-01-01T00:00:00Z',
        current_period_end: '2024-01-31T23:59:59Z',
        cancel_at_period_end: false,
        trial_end: '2024-01-14T23:59:59Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.subscription();
      expect(result.status).toBe('trialing');
      expect(result.trial_end).toBe('2024-01-14T23:59:59Z');
    });

    it('should return subscription scheduled for cancellation', async () => {
      const mockResponse = {
        subscription_id: 'sub_cancel123',
        plan_id: 'pro',
        status: 'active',
        current_period_start: '2024-01-01T00:00:00Z',
        current_period_end: '2024-01-31T23:59:59Z',
        cancel_at_period_end: true,
        trial_end: null,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.subscription();
      expect(result.cancel_at_period_end).toBe(true);
    });
  });

  // ===========================================================================
  // Checkout
  // ===========================================================================

  describe('checkout()', () => {
    it('should create checkout session', async () => {
      const mockResponse = {
        checkout_url: 'https://checkout.stripe.com/pay/cs_123abc',
        session_id: 'cs_123abc',
        expires_at: '2024-01-01T01:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.checkout({
        plan_id: 'pro',
        success_url: 'https://app.aragora.ai/billing/success',
        cancel_url: 'https://app.aragora.ai/billing/cancel',
      });

      expect(result.checkout_url).toContain('checkout.stripe.com');
      expect(result.session_id).toBe('cs_123abc');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/billing/checkout',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            plan_id: 'pro',
            success_url: 'https://app.aragora.ai/billing/success',
            cancel_url: 'https://app.aragora.ai/billing/cancel',
          }),
        })
      );
    });

    it('should create checkout with annual billing', async () => {
      const mockResponse = {
        checkout_url: 'https://checkout.stripe.com/pay/cs_annual',
        session_id: 'cs_annual',
        expires_at: '2024-01-01T01:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.checkout({
        plan_id: 'pro',
        billing_period: 'annual',
        success_url: 'https://app.aragora.ai/billing/success',
        cancel_url: 'https://app.aragora.ai/billing/cancel',
      });

      expect(result.session_id).toBe('cs_annual');
    });

    it('should throw on invalid plan', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Invalid plan ID',
          code: 'INVALID_PLAN',
        }),
      });

      await expect(
        client.billing.checkout({
          plan_id: 'nonexistent',
          success_url: 'https://app.aragora.ai/success',
          cancel_url: 'https://app.aragora.ai/cancel',
        })
      ).rejects.toThrow(AragoraError);
    });
  });

  // ===========================================================================
  // Portal
  // ===========================================================================

  describe('portal()', () => {
    it('should create billing portal session', async () => {
      const mockResponse = {
        portal_url: 'https://billing.stripe.com/session/portal_123abc',
        return_url: 'https://app.aragora.ai/billing',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.portal();
      expect(result.portal_url).toContain('billing.stripe.com');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/billing/portal',
        expect.objectContaining({
          method: 'POST',
        })
      );
    });

    it('should throw if no subscription exists', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'No active subscription',
          code: 'NO_SUBSCRIPTION',
        }),
      });

      await expect(client.billing.portal()).rejects.toThrow(AragoraError);
    });
  });

  // ===========================================================================
  // Cancel
  // ===========================================================================

  describe('cancel()', () => {
    it('should cancel subscription at period end', async () => {
      const mockResponse = {
        subscription_id: 'sub_123abc',
        status: 'active',
        cancel_at_period_end: true,
        current_period_end: '2024-01-31T23:59:59Z',
        message: 'Subscription will be canceled at the end of the billing period',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.cancel();
      expect(result.cancel_at_period_end).toBe(true);
      expect(result.message).toContain('canceled at the end');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/billing/cancel',
        expect.objectContaining({
          method: 'POST',
        })
      );
    });

    it('should throw if already canceled', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Subscription is already canceled',
          code: 'ALREADY_CANCELED',
        }),
      });

      await expect(client.billing.cancel()).rejects.toThrow(AragoraError);
    });
  });

  // ===========================================================================
  // Resume
  // ===========================================================================

  describe('resume()', () => {
    it('should resume canceled subscription', async () => {
      const mockResponse = {
        subscription_id: 'sub_123abc',
        status: 'active',
        cancel_at_period_end: false,
        message: 'Subscription has been resumed',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.resume();
      expect(result.cancel_at_period_end).toBe(false);
      expect(result.message).toContain('resumed');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/billing/resume',
        expect.objectContaining({
          method: 'POST',
        })
      );
    });

    it('should throw if subscription not scheduled for cancellation', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Subscription is not scheduled for cancellation',
          code: 'NOT_CANCELED',
        }),
      });

      await expect(client.billing.resume()).rejects.toThrow(AragoraError);
    });
  });

  // ===========================================================================
  // Audit Log
  // ===========================================================================

  describe('auditLog()', () => {
    it('should return billing audit log', async () => {
      const mockResponse = {
        entries: [
          {
            id: 'log-1',
            action: 'subscription_created',
            timestamp: '2024-01-01T10:00:00Z',
            details: { plan_id: 'pro' },
          },
          {
            id: 'log-2',
            action: 'payment_succeeded',
            timestamp: '2024-01-01T10:01:00Z',
            details: { amount_cents: 2900 },
          },
        ],
        total: 2,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.auditLog();
      expect(result.entries).toHaveLength(2);
      expect(result.entries[0].action).toBe('subscription_created');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/billing/audit-log',
        expect.objectContaining({
          method: 'GET',
        })
      );
    });

    it('should return audit log with pagination', async () => {
      const mockResponse = {
        entries: [
          { id: 'log-3', action: 'usage_overage', timestamp: '2024-01-02T00:00:00Z', details: {} },
        ],
        total: 10,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.auditLog({ limit: 1, offset: 2 });
      expect(result.entries).toHaveLength(1);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/billing/audit-log?limit=1&offset=2',
        expect.objectContaining({
          method: 'GET',
        })
      );
    });
  });

  // ===========================================================================
  // Export Usage
  // ===========================================================================

  describe('exportUsage()', () => {
    it('should export usage as CSV', async () => {
      const mockResponse = {
        csv: 'date,debates,api_calls,storage_mb\n2024-01-01,5,500,50\n2024-01-02,3,300,55',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.exportUsage();
      expect(result).toContain('date,debates,api_calls');
      expect(result).toContain('2024-01-01');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/billing/usage/export',
        expect.objectContaining({
          method: 'GET',
        })
      );
    });
  });

  // ===========================================================================
  // Forecast
  // ===========================================================================

  describe('forecast()', () => {
    it('should return usage forecast', async () => {
      const mockResponse = {
        projected_debates: 95,
        projected_api_calls: 9500,
        projected_storage_mb: 450,
        debates_trend: 'stable',
        days_remaining: 15,
        overage_warning: false,
        projected_overage_amount_cents: 0,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.forecast();
      expect(result.projected_debates).toBe(95);
      expect(result.overage_warning).toBe(false);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/billing/usage/forecast',
        expect.objectContaining({
          method: 'GET',
        })
      );
    });

    it('should return forecast with overage warning', async () => {
      const mockResponse = {
        projected_debates: 150,
        projected_api_calls: 15000,
        projected_storage_mb: 1200,
        debates_trend: 'increasing',
        days_remaining: 15,
        overage_warning: true,
        projected_overage_amount_cents: 5000,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.forecast();
      expect(result.overage_warning).toBe(true);
      expect(result.projected_overage_amount_cents).toBe(5000);
    });
  });

  // ===========================================================================
  // Invoices
  // ===========================================================================

  describe('invoices()', () => {
    it('should return invoice history', async () => {
      const mockResponse = {
        invoices: [
          {
            invoice_id: 'inv_123',
            amount_cents: 2900,
            status: 'paid',
            paid_at: '2024-01-01T00:00:00Z',
            period_start: '2024-01-01T00:00:00Z',
            period_end: '2024-01-31T23:59:59Z',
            pdf_url: 'https://pay.stripe.com/invoice/inv_123/pdf',
          },
          {
            invoice_id: 'inv_122',
            amount_cents: 2900,
            status: 'paid',
            paid_at: '2023-12-01T00:00:00Z',
            period_start: '2023-12-01T00:00:00Z',
            period_end: '2023-12-31T23:59:59Z',
            pdf_url: 'https://pay.stripe.com/invoice/inv_122/pdf',
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.invoices();
      expect(result).toHaveLength(2);
      expect(result[0].invoice_id).toBe('inv_123');
      expect(result[0].status).toBe('paid');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/billing/invoices',
        expect.objectContaining({
          method: 'GET',
        })
      );
    });

    it('should return invoices with limit', async () => {
      const mockResponse = {
        invoices: [
          {
            invoice_id: 'inv_123',
            amount_cents: 2900,
            status: 'paid',
            paid_at: '2024-01-01T00:00:00Z',
            period_start: '2024-01-01T00:00:00Z',
            period_end: '2024-01-31T23:59:59Z',
            pdf_url: 'https://pay.stripe.com/invoice/inv_123/pdf',
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.invoices({ limit: 1 });
      expect(result).toHaveLength(1);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/billing/invoices?limit=1',
        expect.objectContaining({
          method: 'GET',
        })
      );
    });

    it('should handle unpaid invoices', async () => {
      const mockResponse = {
        invoices: [
          {
            invoice_id: 'inv_unpaid',
            amount_cents: 2900,
            status: 'open',
            paid_at: null,
            period_start: '2024-01-01T00:00:00Z',
            period_end: '2024-01-31T23:59:59Z',
            pdf_url: 'https://pay.stripe.com/invoice/inv_unpaid/pdf',
            hosted_invoice_url: 'https://invoice.stripe.com/pay/inv_unpaid',
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.billing.invoices();
      expect(result[0].status).toBe('open');
      expect(result[0].paid_at).toBeNull();
    });
  });

  // ===========================================================================
  // Error Handling
  // ===========================================================================

  describe('error handling', () => {
    // Use a client with retries disabled for error handling tests
    let noRetryClient: AragoraClient;

    beforeEach(() => {
      noRetryClient = new AragoraClient({
        baseUrl: 'http://localhost:8080',
        apiKey: 'test-api-key',
        retry: { maxRetries: 0 },
      });
    });

    it('should throw AragoraError with correct properties on payment failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 402,
        json: () => Promise.resolve({
          error: 'Payment failed',
          code: 'PAYMENT_FAILED',
          decline_code: 'insufficient_funds',
        }),
      });

      try {
        await noRetryClient.billing.checkout({
          plan_id: 'pro',
          success_url: 'https://app.aragora.ai/success',
          cancel_url: 'https://app.aragora.ai/cancel',
        });
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(AragoraError);
        if (error instanceof AragoraError) {
          expect(error.status).toBe(402);
          expect(error.code).toBe('PAYMENT_FAILED');
        }
      }
    });

    it('should handle quota exceeded error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: () => Promise.resolve({
          error: 'Debate quota exceeded',
          code: 'QUOTA_EXCEEDED',
          current_usage: 100,
          limit: 100,
        }),
      });

      try {
        await noRetryClient.billing.usage();
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(AragoraError);
        if (error instanceof AragoraError) {
          expect(error.code).toBe('QUOTA_EXCEEDED');
          expect(error.retryable).toBe(true); // 429 is retryable
        }
      }
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      await expect(noRetryClient.billing.plans()).rejects.toThrow(AragoraError);
    });
  });
});
