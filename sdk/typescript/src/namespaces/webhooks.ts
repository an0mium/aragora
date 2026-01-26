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
 * Interface for the internal client methods used by WebhooksAPI.
 */
interface WebhooksClientInterface {
  listWebhooks(): Promise<{ webhooks: Webhook[]; total: number }>;
  getWebhook(id: string): Promise<Webhook>;
  createWebhook(body: CreateWebhookRequest): Promise<Webhook>;
  updateWebhook(id: string, body: UpdateWebhookRequest): Promise<Webhook>;
  deleteWebhook(id: string): Promise<{ deleted: boolean }>;
  testWebhook(id: string): Promise<{ success: boolean; response_code?: number; error?: string }>;
  listWebhookEvents(): Promise<{ events: WebhookEvent[] }>;
  getWebhookSLOStatus(): Promise<WebhookSLOStatus>;
  testWebhookSLO(): Promise<{ success: boolean; latency_ms: number }>;
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
    return this.client.listWebhooks();
  }

  /**
   * Get a specific webhook by ID.
   */
  async get(id: string): Promise<Webhook> {
    return this.client.getWebhook(id);
  }

  /**
   * Create a new webhook.
   */
  async create(body: CreateWebhookRequest): Promise<Webhook> {
    return this.client.createWebhook(body);
  }

  /**
   * Update an existing webhook.
   */
  async update(id: string, body: UpdateWebhookRequest): Promise<Webhook> {
    return this.client.updateWebhook(id, body);
  }

  /**
   * Delete a webhook.
   */
  async delete(id: string): Promise<{ deleted: boolean }> {
    return this.client.deleteWebhook(id);
  }

  /**
   * Test a webhook by sending a test event.
   */
  async test(id: string): Promise<{ success: boolean; response_code?: number; error?: string }> {
    return this.client.testWebhook(id);
  }

  // ===========================================================================
  // Events
  // ===========================================================================

  /**
   * List all available webhook events.
   */
  async listEvents(): Promise<{ events: WebhookEvent[] }> {
    return this.client.listWebhookEvents();
  }

  // ===========================================================================
  // SLO Monitoring
  // ===========================================================================

  /**
   * Get webhook SLO status.
   */
  async getSLOStatus(): Promise<WebhookSLOStatus> {
    return this.client.getWebhookSLOStatus();
  }

  /**
   * Test webhook SLO by measuring delivery latency.
   */
  async testSLO(): Promise<{ success: boolean; latency_ms: number }> {
    return this.client.testWebhookSLO();
  }
}
