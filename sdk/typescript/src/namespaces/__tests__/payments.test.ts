/**
 * Payments Namespace Tests
 *
 * Comprehensive tests for the payments namespace API including:
 * - Payment operations (charge, authorize, capture, refund, void)
 * - Customer management
 * - Subscription management
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { PaymentsAPI } from '../payments';

interface MockClient {
  request: Mock;
}

describe('PaymentsAPI Namespace', () => {
  let api: PaymentsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new PaymentsAPI(mockClient as any);
  });

  // ===========================================================================
  // Payment Operations
  // ===========================================================================

  describe('Payment Operations', () => {
    it('should charge payment', async () => {
      const mockResult = {
        success: true,
        transaction: {
          transaction_id: 'txn_123',
          provider: 'stripe',
          status: 'approved',
          amount: '99.99',
          currency: 'USD',
          auth_code: 'ABC123',
          created_at: '2024-01-20T10:00:00Z',
        },
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.charge({
        amount: 99.99,
        currency: 'USD',
        customer_id: 'cus_123',
        description: 'Order #12345',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/payments/charge', {
        json: {
          amount: 99.99,
          currency: 'USD',
          customer_id: 'cus_123',
          description: 'Order #12345',
        },
      });
      expect(result.success).toBe(true);
      expect(result.transaction.status).toBe('approved');
    });

    it('should charge with payment method', async () => {
      const mockResult = { success: true, transaction: { transaction_id: 'txn_124' } };
      mockClient.request.mockResolvedValue(mockResult);

      await api.charge({
        amount: 50.00,
        payment_method: 'pm_123',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/payments/charge', {
        json: { amount: 50.00, payment_method: 'pm_123' },
      });
    });

    it('should authorize payment', async () => {
      const mockResult = {
        success: true,
        transaction_id: 'txn_125',
        transaction: {
          transaction_id: 'txn_125',
          status: 'approved',
          amount: '150.00',
        },
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.authorize({
        amount: 150.00,
        customer_id: 'cus_123',
        capture: false,
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/payments/authorize', {
        json: { amount: 150.00, customer_id: 'cus_123', capture: false },
      });
      expect(result.transaction_id).toBe('txn_125');
    });

    it('should capture authorized payment', async () => {
      const mockResult = {
        success: true,
        transaction: { transaction_id: 'txn_125', status: 'approved' },
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.capture('txn_125', 100.00);

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/payments/capture', {
        json: { transaction_id: 'txn_125', amount: 100.00, provider: undefined },
      });
      expect(result.success).toBe(true);
    });

    it('should refund payment', async () => {
      const mockResult = {
        success: true,
        refund_id: 'ref_123',
        transaction: { transaction_id: 'txn_123', status: 'refunded' },
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.refund({
        transaction_id: 'txn_123',
        amount: 50.00,
        reason: 'Customer request',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/payments/refund', {
        json: {
          transaction_id: 'txn_123',
          amount: 50.00,
          reason: 'Customer request',
        },
      });
      expect(result.refund_id).toBe('ref_123');
    });

    it('should void transaction', async () => {
      mockClient.request.mockResolvedValue({ success: true });

      const result = await api.void('txn_126');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/payments/void', {
        json: { transaction_id: 'txn_126', provider: undefined },
      });
      expect(result.success).toBe(true);
    });

    it('should get transaction details', async () => {
      const mockTransaction = {
        transaction: {
          transaction_id: 'txn_123',
          provider: 'stripe',
          status: 'approved',
          amount: '99.99',
          currency: 'USD',
          customer_id: 'cus_123',
          captured: true,
          captured_at: '2024-01-20T10:01:00Z',
        },
      };
      mockClient.request.mockResolvedValue(mockTransaction);

      const result = await api.getTransaction('txn_123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/payments/transaction/txn_123');
      expect(result.transaction.captured).toBe(true);
    });
  });

  // ===========================================================================
  // Customer Management
  // ===========================================================================

  describe('Customer Management', () => {
    it('should create customer', async () => {
      const mockResult = {
        success: true,
        customer_id: 'cus_new',
        customer: {
          id: 'cus_new',
          email: 'customer@example.com',
          name: 'John Doe',
          payment_methods: [],
          created_at: '2024-01-20T10:00:00Z',
        },
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.createCustomer({
        email: 'customer@example.com',
        name: 'John Doe',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/payments/customer', {
        json: { email: 'customer@example.com', name: 'John Doe' },
      });
      expect(result.customer_id).toBe('cus_new');
    });

    it('should create customer with payment method', async () => {
      const mockResult = { success: true, customer_id: 'cus_new2', customer: {} };
      mockClient.request.mockResolvedValue(mockResult);

      await api.createCustomer({
        email: 'customer@example.com',
        payment_method: 'pm_123',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/payments/customer', {
        json: { email: 'customer@example.com', payment_method: 'pm_123' },
      });
    });

    it('should get customer', async () => {
      const mockCustomer = {
        customer: {
          id: 'cus_123',
          email: 'customer@example.com',
          name: 'John Doe',
          default_payment_method: 'pm_123',
          payment_methods: [
            { id: 'pm_123', type: 'card', last_four: '4242', brand: 'visa', is_default: true },
          ],
        },
      };
      mockClient.request.mockResolvedValue(mockCustomer);

      const result = await api.getCustomer('cus_123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/payments/customer/cus_123');
      expect(result.customer.payment_methods).toHaveLength(1);
    });

    it('should update customer', async () => {
      const mockResult = {
        success: true,
        customer: { id: 'cus_123', name: 'Jane Doe' },
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.updateCustomer('cus_123', { name: 'Jane Doe' });

      expect(mockClient.request).toHaveBeenCalledWith('PUT', '/api/payments/customer/cus_123', {
        json: { name: 'Jane Doe' },
      });
      expect(result.customer.name).toBe('Jane Doe');
    });

    it('should delete customer', async () => {
      mockClient.request.mockResolvedValue({ success: true });

      const result = await api.deleteCustomer('cus_123');

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/api/payments/customer/cus_123');
      expect(result.success).toBe(true);
    });
  });

  // ===========================================================================
  // Subscription Management
  // ===========================================================================

  describe('Subscription Management', () => {
    it('should create subscription', async () => {
      const mockResult = {
        success: true,
        subscription_id: 'sub_123',
        subscription: {
          id: 'sub_123',
          customer_id: 'cus_123',
          name: 'Pro Plan',
          amount: '29.99',
          currency: 'USD',
          interval: 'month',
          interval_count: 1,
          status: 'active',
          current_period_start: '2024-01-20',
          current_period_end: '2024-02-20',
          cancel_at_period_end: false,
          created_at: '2024-01-20T10:00:00Z',
        },
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.createSubscription({
        customer_id: 'cus_123',
        name: 'Pro Plan',
        amount: 29.99,
        interval: 'month',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/payments/subscription', {
        json: {
          customer_id: 'cus_123',
          name: 'Pro Plan',
          amount: 29.99,
          interval: 'month',
        },
      });
      expect(result.subscription.status).toBe('active');
    });

    it('should create yearly subscription', async () => {
      const mockResult = { success: true, subscription_id: 'sub_124', subscription: {} };
      mockClient.request.mockResolvedValue(mockResult);

      await api.createSubscription({
        customer_id: 'cus_123',
        name: 'Enterprise',
        amount: 299.99,
        interval: 'year',
        interval_count: 1,
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/payments/subscription', {
        json: {
          customer_id: 'cus_123',
          name: 'Enterprise',
          amount: 299.99,
          interval: 'year',
          interval_count: 1,
        },
      });
    });

    it('should get subscription', async () => {
      const mockSubscription = {
        subscription: {
          id: 'sub_123',
          customer_id: 'cus_123',
          name: 'Pro Plan',
          status: 'active',
        },
      };
      mockClient.request.mockResolvedValue(mockSubscription);

      const result = await api.getSubscription('sub_123');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/payments/subscription/sub_123');
      expect(result.subscription.status).toBe('active');
    });

    it('should update subscription', async () => {
      const mockResult = {
        success: true,
        subscription: { id: 'sub_123', name: 'Enterprise Plan' },
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.updateSubscription('sub_123', { name: 'Enterprise Plan' });

      expect(mockClient.request).toHaveBeenCalledWith('PUT', '/api/payments/subscription/sub_123', {
        json: { name: 'Enterprise Plan' },
      });
    });

    it('should cancel subscription at period end', async () => {
      const mockResult = {
        success: true,
        subscription: { id: 'sub_123', cancel_at_period_end: true },
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.cancelSubscription('sub_123', true);

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/api/payments/subscription/sub_123', {
        json: { cancel_at_period_end: true },
      });
      expect(result.subscription.cancel_at_period_end).toBe(true);
    });

    it('should cancel subscription immediately', async () => {
      const mockResult = {
        success: true,
        subscription: { id: 'sub_123', status: 'cancelled' },
      };
      mockClient.request.mockResolvedValue(mockResult);

      await api.cancelSubscription('sub_123', false);

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/api/payments/subscription/sub_123', {
        json: { cancel_at_period_end: false },
      });
    });
  });
});
