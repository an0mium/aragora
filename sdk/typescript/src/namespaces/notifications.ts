/**
 * Notifications Namespace API
 *
 * Provides access to notification settings and delivery for email, Telegram, and webhooks.
 */

/**
 * Notification channel types
 */
export type NotificationChannel = 'email' | 'telegram' | 'webhook' | 'slack';

/**
 * Notification event types
 */
export type NotificationEventType =
  | 'debate_started'
  | 'debate_completed'
  | 'consensus_reached'
  | 'consensus_failed'
  | 'agent_error'
  | 'budget_alert'
  | 'gauntlet_completed'
  | 'security_alert';

/**
 * Integration status for a notification channel
 */
export interface IntegrationStatus {
  channel: NotificationChannel;
  enabled: boolean;
  configured: boolean;
  last_delivery?: string;
  last_error?: string;
}

/**
 * Email configuration settings
 */
export interface EmailConfig {
  smtp_host?: string;
  smtp_port?: number;
  smtp_user?: string;
  smtp_password?: string;
  from_address?: string;
  use_tls?: boolean;
}

/**
 * Telegram configuration settings
 */
export interface TelegramConfig {
  bot_token?: string;
  chat_id?: string;
  parse_mode?: 'HTML' | 'Markdown' | 'MarkdownV2';
}

/**
 * Email recipient entry
 */
export interface EmailRecipient {
  email: string;
  name?: string;
  events?: NotificationEventType[];
  added_at?: string;
}

/**
 * Notification delivery record
 */
export interface NotificationDelivery {
  id: string;
  channel: NotificationChannel;
  event_type: NotificationEventType;
  recipient: string;
  subject?: string;
  delivered_at: string;
  success: boolean;
  error?: string;
}

/**
 * Interface for the internal client used by NotificationsAPI.
 */
interface NotificationsClientInterface {
  request<T>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Notifications API namespace.
 *
 * Provides methods for managing notification channels and delivery:
 * - Configure email, Telegram, and webhook integrations
 * - Manage recipients and subscriptions
 * - Send test notifications
 * - View delivery history
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Check notification status
 * const status = await client.notifications.getStatus();
 *
 * // Configure email notifications
 * await client.notifications.configureEmail({
 *   smtp_host: 'smtp.example.com',
 *   from_address: 'aragora@example.com'
 * });
 *
 * // Send a test notification
 * await client.notifications.sendTest({ channel: 'email' });
 * ```
 */
export class NotificationsAPI {
  constructor(private client: NotificationsClientInterface) {}

  /**
   * Get notification integration status for all channels.
   */
  async getStatus(): Promise<{ integrations: IntegrationStatus[] }> {
    return this.client.request('GET', '/api/notifications/status');
  }

  /**
   * Configure email notification settings.
   */
  async configureEmail(config: EmailConfig): Promise<{ success: boolean; message: string }> {
    return this.client.request('POST', '/api/notifications/email/config', {
      json: config as unknown as Record<string, unknown>,
    });
  }

  /**
   * Configure Telegram notification settings.
   */
  async configureTelegram(config: TelegramConfig): Promise<{ success: boolean; message: string }> {
    return this.client.request('POST', '/api/notifications/telegram/config', {
      json: config as unknown as Record<string, unknown>,
    });
  }

  /**
   * Add an email recipient for notifications.
   */
  async addEmailRecipient(recipient: {
    email: string;
    name?: string;
    events?: NotificationEventType[];
  }): Promise<{ success: boolean; recipient: EmailRecipient }> {
    return this.client.request('POST', '/api/notifications/email/recipient', { json: recipient });
  }

  /**
   * Remove an email recipient.
   */
  async removeEmailRecipient(email: string): Promise<{ success: boolean }> {
    return this.client.request('DELETE', '/api/notifications/email/recipient', {
      json: { email },
    });
  }

  /**
   * List all email recipients.
   */
  async listEmailRecipients(): Promise<{ recipients: EmailRecipient[] }> {
    return this.client.request('GET', '/api/notifications/email/recipients');
  }

  /**
   * Send a test notification to verify configuration.
   */
  async sendTest(options: {
    channel: NotificationChannel;
    recipient?: string;
  }): Promise<{ success: boolean; message: string }> {
    return this.client.request('POST', '/api/notifications/test', { json: options });
  }

  /**
   * Send a notification immediately.
   */
  async send(notification: {
    channel: NotificationChannel;
    event_type: NotificationEventType;
    recipient?: string;
    subject?: string;
    message: string;
    metadata?: Record<string, unknown>;
  }): Promise<{ success: boolean; delivery_id?: string }> {
    return this.client.request('POST', '/api/notifications/send', { json: notification });
  }

  /**
   * Get notification delivery history.
   */
  async getDeliveryHistory(options?: {
    channel?: NotificationChannel;
    event_type?: NotificationEventType;
    success?: boolean;
    limit?: number;
    offset?: number;
  }): Promise<{ deliveries: NotificationDelivery[]; total: number }> {
    return this.client.request('GET', '/api/notifications/history', { params: options });
  }
}
