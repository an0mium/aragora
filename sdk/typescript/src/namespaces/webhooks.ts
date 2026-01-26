/**
 * Webhooks Namespace API
 *
 * Provides a namespaced interface for webhook management.
 * Essential for enterprise event-driven integrations.
 */

/**
 * Webhook configuration
 */
export interface Webhook {
  id: string;
  url: string;
  events: string[];
  secret: string;
  active: boolean;
  created_at: string;
  updated_at: string;
  last_triggered_at?: string;
  failure_count: number;
  metadata?: Record<string, unknown>;
}

/**
 * Webhook event type
 */
export interface WebhookEvent {
  name: string;
  description: string;
  category: string;
  payload_schema?: Record<string, unknown>;
}

/**
 * Webhook delivery status
 */
export interface WebhookDelivery {
  id: string;
  webhook_id: string;
  event: string;
  payload: Record<string, unknown>;
  status: 'pending' | 'delivered' | 'failed';
  response_code?: number;
  response_body?: string;
  delivered_at?: string;
  error?: string;
  retry_count: number;
}

/**
 * SLO status response
 */
export interface WebhookSLOStatus {
  delivery_rate: number;
  avg_latency_ms: number;
  p99_latency_ms: number;
  failed_deliveries_24h: number;
  total_deliveries_24h: number;
  status: 'healthy' | 'degraded' | 'critical';
}

/**
 * Create webhook request
 */
export interface CreateWebhookRequest {
  url: string;
  events: string[];
  secret?: string;
  active?: boolean;
  metadata?: Record<string, unknown>;
}

/**
 * Update webhook request
 */
export interface UpdateWebhookRequest {
  url?: string;
  events?: string[];
  secret?: string;
  active?: boolean;
  metadata?: Record<string, unknown>;
}

/**
 * Webhook delivery attempt
 */
export interface WebhookDeliveryAttempt {
  attempt_number: number;
  timestamp: string;
  response_code?: number;
  response_time_ms: number;
  error?: string;
}

/**
 * Webhook retry policy
 */
export interface WebhookRetryPolicy {
  max_retries: number;
  initial_delay_ms: number;
  max_delay_ms: number;
  backoff_multiplier: number;
}

/**
 * Interface for the internal client used by WebhooksAPI.
 */
interface WebhooksClientInterface {
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
  put<T>(path: string, body?: unknown): Promise<T>;
  delete<T>(path: string): Promise<T>;
  request<T = unknown>(method: string, path: string, options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }): Promise<T>;
}

/**
 * Webhooks API namespace.
 *
 * Provides methods for managing webhooks:
 * - Create, update, and delete webhooks
 * - List available webhook events
 * - Test webhook deliveries
 * - Monitor webhook SLO status
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Create a webhook
 * const webhook = await client.webhooks.create({
 *   url: 'https://your-app.com/webhooks',
 *   events: ['debate.completed', 'consensus.reached'],
 * });
 *
 * // List available events
 * const { events } = await client.webhooks.listEvents();
 *
 * // Test a webhook
 * const result = await client.webhooks.test(webhook.id);
 *
 * // Check SLO status
 * const slo = await client.webhooks.getSLOStatus();
 * console.log(`Delivery rate: ${slo.delivery_rate}%`);
 * ```
 */
export class WebhooksAPI {
  constructor(private client: WebhooksClientInterface) {}

  // ===========================================================================
  // Webhook Management
  // ===========================================================================

  /**
   * List all webhooks.
   */
  async list(): Promise<{ webhooks: Webhook[]; total: number }> {
    return this.client.get('/api/v1/webhooks');
  }

  /**
   * Get a specific webhook by ID.
   */
  async get(id: string): Promise<Webhook> {
    return this.client.get(`/api/v1/webhooks/${id}`);
  }

  /**
   * Create a new webhook.
   */
  async create(body: CreateWebhookRequest): Promise<Webhook> {
    return this.client.post('/api/v1/webhooks', body);
  }

  /**
   * Update an existing webhook.
   */
  async update(id: string, body: UpdateWebhookRequest): Promise<Webhook> {
    return this.client.put(`/api/v1/webhooks/${id}`, body);
  }

  /**
   * Delete a webhook.
   */
  async delete(id: string): Promise<{ deleted: boolean }> {
    return this.client.delete(`/api/v1/webhooks/${id}`);
  }

  /**
   * Test a webhook by sending a test event.
   */
  async test(id: string): Promise<{ success: boolean; response_code?: number; error?: string }> {
    return this.client.post(`/api/v1/webhooks/${id}/test`);
  }

  // ===========================================================================
  // Events
  // ===========================================================================

  /**
   * List all available webhook events.
   */
  async listEvents(): Promise<{ events: WebhookEvent[] }> {
    return this.client.get('/api/v1/webhooks/events');
  }

  // ===========================================================================
  // SLO Monitoring
  // ===========================================================================

  /**
   * Get webhook SLO status.
   */
  async getSLOStatus(): Promise<WebhookSLOStatus> {
    return this.client.get('/api/v1/webhooks/slo/status');
  }

  /**
   * Test webhook SLO by measuring delivery latency.
   */
  async testSLO(): Promise<{ success: boolean; latency_ms: number }> {
    return this.client.post('/api/v1/webhooks/slo/test');
  }

  // ===========================================================================
  // Delivery Management
  // ===========================================================================

  /**
   * List webhook deliveries.
   */
  async listDeliveries(webhookId: string, options?: { status?: string; limit?: number; offset?: number }): Promise<{ deliveries: WebhookDelivery[]; total: number }> {
    return this.client.request('GET', `/api/v1/webhooks/${webhookId}/deliveries`, { params: options });
  }

  /**
   * Get delivery details.
   */
  async getDelivery(webhookId: string, deliveryId: string): Promise<WebhookDelivery & { attempts: WebhookDeliveryAttempt[] }> {
    return this.client.request('GET', `/api/v1/webhooks/${webhookId}/deliveries/${deliveryId}`);
  }

  /**
   * Retry a failed delivery.
   */
  async retryDelivery(webhookId: string, deliveryId: string): Promise<{ retried: boolean; new_delivery_id: string }> {
    return this.client.request('POST', `/api/v1/webhooks/${webhookId}/deliveries/${deliveryId}/retry`);
  }

  /**
   * Get delivery stats for a webhook.
   */
  async getDeliveryStats(webhookId: string, options?: { days?: number }): Promise<{ success_rate: number; avg_latency_ms: number; total_deliveries: number; failed_deliveries: number }> {
    return this.client.request('GET', `/api/v1/webhooks/${webhookId}/stats`, { params: options });
  }

  // ===========================================================================
  // Retry Policy
  // ===========================================================================

  /**
   * Get webhook retry policy.
   */
  async getRetryPolicy(webhookId: string): Promise<WebhookRetryPolicy> {
    return this.client.request('GET', `/api/v1/webhooks/${webhookId}/retry-policy`);
  }

  /**
   * Update webhook retry policy.
   */
  async updateRetryPolicy(webhookId: string, policy: Partial<WebhookRetryPolicy>): Promise<WebhookRetryPolicy> {
    return this.client.request('PUT', `/api/v1/webhooks/${webhookId}/retry-policy`, { json: policy });
  }

  // ===========================================================================
  // Event Filtering
  // ===========================================================================

  /**
   * Get event categories.
   */
  async getEventCategories(): Promise<{ categories: { name: string; description: string; events: string[] }[] }> {
    return this.client.request('GET', '/api/v1/webhooks/events/categories');
  }

  /**
   * Subscribe to events for a webhook.
   */
  async subscribeEvents(webhookId: string, events: string[]): Promise<{ subscribed: string[] }> {
    return this.client.request('POST', `/api/v1/webhooks/${webhookId}/events`, { json: { events } });
  }

  /**
   * Unsubscribe from events for a webhook.
   */
  async unsubscribeEvents(webhookId: string, events: string[]): Promise<{ unsubscribed: string[] }> {
    return this.client.request('DELETE', `/api/v1/webhooks/${webhookId}/events`, { json: { events } });
  }

  // ===========================================================================
  // Signing and Verification
  // ===========================================================================

  /**
   * Rotate webhook secret.
   */
  async rotateSecret(webhookId: string): Promise<{ new_secret: string; old_secret_valid_until: string }> {
    return this.client.request('POST', `/api/v1/webhooks/${webhookId}/rotate-secret`);
  }

  /**
   * Get signing key info.
   */
  async getSigningInfo(webhookId: string): Promise<{ algorithm: string; header_name: string; format: string }> {
    return this.client.request('GET', `/api/v1/webhooks/${webhookId}/signing`);
  }

  // ===========================================================================
  // Bulk Operations
  // ===========================================================================

  /**
   * Pause all webhooks.
   */
  async pauseAll(): Promise<{ paused: number }> {
    return this.client.request('POST', '/api/v1/webhooks/pause-all');
  }

  /**
   * Resume all webhooks.
   */
  async resumeAll(): Promise<{ resumed: number }> {
    return this.client.request('POST', '/api/v1/webhooks/resume-all');
  }

  /**
   * Bulk delete webhooks.
   */
  async bulkDelete(webhookIds: string[]): Promise<{ deleted: number }> {
    return this.client.request('DELETE', '/api/v1/webhooks/bulk', { json: { ids: webhookIds } });
  }
}
