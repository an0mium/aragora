/**
 * Payments Namespace API
 *
 * Provides payment processing capabilities for Stripe and Authorize.net:
 * - Payment processing (charge, authorize, capture, refund)
 * - Customer profile management
 * - Subscription management
 */

/**
 * Supported payment providers.
 */
export type PaymentProvider = 'stripe' | 'authorize_net';

/**
 * Payment transaction status.
 */
export type PaymentStatus = 'pending' | 'approved' | 'declined' | 'error' | 'void' | 'refunded';

/**
 * Subscription interval.
 */
export type SubscriptionInterval = 'day' | 'week' | 'month' | 'year';

/**
 * Billing address for payment.
 */
export interface BillingAddress {
  first_name?: string;
  last_name?: string;
  address?: string;
  city?: string;
  state?: string;
  zip?: string;
  country?: string;
}

/**
 * Payment method details.
 */
export interface PaymentMethodDetails {
  type: 'card';
  card_number: string;
  exp_month: string;
  exp_year: string;
  cvv: string;
  billing?: BillingAddress;
}

/**
 * Charge request.
 */
export interface ChargeRequest {
  /** Payment amount */
  amount: number;
  /** Currency code (default: USD) */
  currency?: string;
  /** Payment description */
  description?: string;
  /** Customer ID for saved payment methods */
  customer_id?: string;
  /** Payment method token or details */
  payment_method?: string | PaymentMethodDetails;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
  /** Payment provider (default: stripe) */
  provider?: PaymentProvider;
}

/**
 * Authorize request (for later capture).
 */
export interface AuthorizeRequest extends ChargeRequest {
  /** Whether to capture immediately (default: false for authorize) */
  capture?: boolean;
}

/**
 * Refund request.
 */
export interface RefundRequest {
  /** Original transaction ID to refund */
  transaction_id: string;
  /** Refund amount (defaults to full amount) */
  amount?: number;
  /** Reason for refund */
  reason?: string;
  /** Payment provider */
  provider?: PaymentProvider;
  /** Last 4 digits of card (for verification) */
  card_last_four?: string;
}

/**
 * Payment transaction result.
 */
export interface PaymentResult {
  transaction_id: string;
  provider: PaymentProvider;
  status: PaymentStatus;
  amount: string;
  currency: string;
  message?: string;
  auth_code?: string;
  avs_result?: string;
  cvv_result?: string;
  created_at: string;
  metadata?: Record<string, unknown>;
}

/**
 * Transaction details.
 */
export interface TransactionDetails extends PaymentResult {
  customer_id?: string;
  description?: string;
  refunded_amount?: string;
  captured?: boolean;
  captured_at?: string;
}

/**
 * Customer profile.
 */
export interface CustomerProfile {
  id: string;
  email: string;
  name?: string;
  phone?: string;
  default_payment_method?: string;
  payment_methods: PaymentMethodSummary[];
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at?: string;
}

/**
 * Payment method summary.
 */
export interface PaymentMethodSummary {
  id: string;
  type: 'card';
  last_four: string;
  brand: string;
  exp_month: number;
  exp_year: number;
  is_default: boolean;
}

/**
 * Create customer request.
 */
export interface CreateCustomerRequest {
  email: string;
  name?: string;
  phone?: string;
  payment_method?: string | PaymentMethodDetails;
  metadata?: Record<string, unknown>;
  provider?: PaymentProvider;
}

/**
 * Update customer request.
 */
export interface UpdateCustomerRequest {
  email?: string;
  name?: string;
  phone?: string;
  default_payment_method?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Subscription.
 */
export interface Subscription {
  id: string;
  customer_id: string;
  name: string;
  amount: string;
  currency: string;
  interval: SubscriptionInterval;
  interval_count: number;
  status: 'active' | 'paused' | 'cancelled' | 'past_due';
  current_period_start: string;
  current_period_end: string;
  cancel_at_period_end: boolean;
  created_at: string;
  cancelled_at?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Create subscription request.
 */
export interface CreateSubscriptionRequest {
  customer_id: string;
  name: string;
  amount: number;
  currency?: string;
  interval: SubscriptionInterval;
  interval_count?: number;
  price_id?: string;
  metadata?: Record<string, unknown>;
  provider?: PaymentProvider;
}

/**
 * Update subscription request.
 */
export interface UpdateSubscriptionRequest {
  name?: string;
  amount?: number;
  cancel_at_period_end?: boolean;
  metadata?: Record<string, unknown>;
}

/**
 * Client interface for payments operations.
 */
interface PaymentsClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Payments API namespace.
 *
 * Provides methods for payment processing, customer management, and subscriptions:
 * - Charge, authorize, capture, refund, and void payments
 * - Create and manage customer profiles
 * - Create and manage subscriptions
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Process a payment
 * const result = await client.payments.charge({
 *   amount: 99.99,
 *   currency: 'USD',
 *   customer_id: 'cus_123',
 *   description: 'Order #12345',
 * });
 *
 * // Create a subscription
 * const sub = await client.payments.createSubscription({
 *   customer_id: 'cus_123',
 *   name: 'Pro Plan',
 *   amount: 29.99,
 *   interval: 'month',
 * });
 * ```
 */
export class PaymentsAPI {
  constructor(private client: PaymentsClientInterface) {}

  // =========================================================================
  // Payment Operations
  // =========================================================================

  /**
   * Process a payment charge.
   */
  async charge(
    request: ChargeRequest
  ): Promise<{ success: boolean; transaction: PaymentResult }> {
    return this.client.request('POST', '/api/payments/charge', { json: request });
  }

  /**
   * Authorize a payment for later capture.
   */
  async authorize(
    request: AuthorizeRequest
  ): Promise<{ success: boolean; transaction_id: string; transaction: PaymentResult }> {
    return this.client.request('POST', '/api/payments/authorize', { json: request });
  }

  /**
   * Capture a previously authorized payment.
   */
  async capture(
    transactionId: string,
    amount?: number,
    provider?: PaymentProvider
  ): Promise<{ success: boolean; transaction: PaymentResult }> {
    return this.client.request('POST', '/api/payments/capture', {
      json: { transaction_id: transactionId, amount, provider },
    });
  }

  /**
   * Refund a payment.
   */
  async refund(
    request: RefundRequest
  ): Promise<{ success: boolean; refund_id?: string; transaction: PaymentResult }> {
    return this.client.request('POST', '/api/payments/refund', { json: request });
  }

  /**
   * Void a transaction.
   */
  async void(
    transactionId: string,
    provider?: PaymentProvider
  ): Promise<{ success: boolean }> {
    return this.client.request('POST', '/api/payments/void', {
      json: { transaction_id: transactionId, provider },
    });
  }

  /**
   * Get transaction details.
   */
  async getTransaction(transactionId: string): Promise<{ transaction: TransactionDetails }> {
    return this.client.request('GET', `/api/payments/transaction/${transactionId}`);
  }

  // =========================================================================
  // Customer Management
  // =========================================================================

  /**
   * Create a customer profile.
   */
  async createCustomer(
    request: CreateCustomerRequest
  ): Promise<{ success: boolean; customer_id: string; customer: CustomerProfile }> {
    return this.client.request('POST', '/api/payments/customer', { json: request });
  }

  /**
   * Get a customer profile.
   */
  async getCustomer(customerId: string): Promise<{ customer: CustomerProfile }> {
    return this.client.request('GET', `/api/payments/customer/${customerId}`);
  }

  /**
   * Update a customer profile.
   */
  async updateCustomer(
    customerId: string,
    request: UpdateCustomerRequest
  ): Promise<{ success: boolean; customer: CustomerProfile }> {
    return this.client.request('PUT', `/api/payments/customer/${customerId}`, {
      json: request,
    });
  }

  /**
   * Delete a customer profile.
   */
  async deleteCustomer(customerId: string): Promise<{ success: boolean }> {
    return this.client.request('DELETE', `/api/payments/customer/${customerId}`);
  }

  // =========================================================================
  // Subscription Management
  // =========================================================================

  /**
   * Create a subscription.
   */
  async createSubscription(
    request: CreateSubscriptionRequest
  ): Promise<{ success: boolean; subscription_id: string; subscription: Subscription }> {
    return this.client.request('POST', '/api/payments/subscription', { json: request });
  }

  /**
   * Get a subscription.
   */
  async getSubscription(subscriptionId: string): Promise<{ subscription: Subscription }> {
    return this.client.request('GET', `/api/payments/subscription/${subscriptionId}`);
  }

  /**
   * Update a subscription.
   */
  async updateSubscription(
    subscriptionId: string,
    request: UpdateSubscriptionRequest
  ): Promise<{ success: boolean; subscription: Subscription }> {
    return this.client.request('PUT', `/api/payments/subscription/${subscriptionId}`, {
      json: request,
    });
  }

  /**
   * Cancel a subscription.
   */
  async cancelSubscription(
    subscriptionId: string,
    cancelAtPeriodEnd = true
  ): Promise<{ success: boolean; subscription: Subscription }> {
    return this.client.request('DELETE', `/api/payments/subscription/${subscriptionId}`, {
      json: { cancel_at_period_end: cancelAtPeriodEnd },
    });
  }
}
