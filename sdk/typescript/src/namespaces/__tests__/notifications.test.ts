/**
 * Notifications Namespace Tests
 *
 * Comprehensive tests for the notifications namespace API including:
 * - Notification status
 * - Email configuration
 * - Telegram configuration
 * - Recipient management
 * - Test and send notifications
 * - Delivery history
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { NotificationsAPI } from '../notifications';

interface MockClient {
  request: Mock;
}

describe('NotificationsAPI Namespace', () => {
  let api: NotificationsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new NotificationsAPI(mockClient as any);
  });

  // ===========================================================================
  // Notification Status
  // ===========================================================================

  describe('Notification Status', () => {
    it('should get integration status for all channels', async () => {
      const mockStatus = {
        integrations: [
          { channel: 'email', enabled: true, configured: true, last_delivery: '2024-01-20T10:00:00Z' },
          { channel: 'telegram', enabled: true, configured: true },
          { channel: 'webhook', enabled: false, configured: false },
          { channel: 'slack', enabled: true, configured: true },
        ],
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getStatus();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/notifications/status');
      expect(result.integrations).toHaveLength(4);
      expect(result.integrations.find((i) => i.channel === 'email')?.configured).toBe(true);
    });

    it('should show error status for failed channels', async () => {
      const mockStatus = {
        integrations: [
          { channel: 'email', enabled: true, configured: true, last_error: 'SMTP connection failed' },
        ],
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getStatus();

      expect(result.integrations[0].last_error).toBe('SMTP connection failed');
    });
  });

  // ===========================================================================
  // Email Configuration
  // ===========================================================================

  describe('Email Configuration', () => {
    it('should configure email settings', async () => {
      mockClient.request.mockResolvedValue({ success: true, message: 'Email configured' });

      const result = await api.configureEmail({
        smtp_host: 'smtp.example.com',
        smtp_port: 587,
        smtp_user: 'user@example.com',
        smtp_password: 'password123',
        from_address: 'aragora@example.com',
        use_tls: true,
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/notifications/email/config', {
        json: {
          smtp_host: 'smtp.example.com',
          smtp_port: 587,
          smtp_user: 'user@example.com',
          smtp_password: 'password123',
          from_address: 'aragora@example.com',
          use_tls: true,
        },
      });
      expect(result.success).toBe(true);
    });
  });

  // ===========================================================================
  // Telegram Configuration
  // ===========================================================================

  describe('Telegram Configuration', () => {
    it('should configure Telegram settings', async () => {
      mockClient.request.mockResolvedValue({ success: true, message: 'Telegram configured' });

      const result = await api.configureTelegram({
        bot_token: '123456:ABC-DEF',
        chat_id: '-1001234567890',
        parse_mode: 'HTML',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/notifications/telegram/config', {
        json: {
          bot_token: '123456:ABC-DEF',
          chat_id: '-1001234567890',
          parse_mode: 'HTML',
        },
      });
      expect(result.success).toBe(true);
    });
  });

  // ===========================================================================
  // Recipient Management
  // ===========================================================================

  describe('Recipient Management', () => {
    it('should add email recipient', async () => {
      const mockRecipient = {
        email: 'user@example.com',
        name: 'Test User',
        events: ['debate_completed', 'consensus_reached'],
        added_at: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue({ success: true, recipient: mockRecipient });

      const result = await api.addEmailRecipient({
        email: 'user@example.com',
        name: 'Test User',
        events: ['debate_completed', 'consensus_reached'],
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/notifications/email/recipient', {
        json: {
          email: 'user@example.com',
          name: 'Test User',
          events: ['debate_completed', 'consensus_reached'],
        },
      });
      expect(result.recipient.email).toBe('user@example.com');
    });

    it('should remove email recipient', async () => {
      mockClient.request.mockResolvedValue({ success: true });

      const result = await api.removeEmailRecipient('user@example.com');

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/api/notifications/email/recipient', {
        json: { email: 'user@example.com' },
      });
      expect(result.success).toBe(true);
    });

    it('should list email recipients', async () => {
      const mockRecipients = {
        recipients: [
          { email: 'user1@example.com', name: 'User 1' },
          { email: 'user2@example.com', name: 'User 2', events: ['debate_started'] },
        ],
      };
      mockClient.request.mockResolvedValue(mockRecipients);

      const result = await api.listEmailRecipients();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/notifications/email/recipients');
      expect(result.recipients).toHaveLength(2);
    });
  });

  // ===========================================================================
  // Test and Send Notifications
  // ===========================================================================

  describe('Test and Send Notifications', () => {
    it('should send test notification', async () => {
      mockClient.request.mockResolvedValue({ success: true, message: 'Test email sent' });

      const result = await api.sendTest({ channel: 'email' });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/notifications/test', {
        json: { channel: 'email' },
      });
      expect(result.success).toBe(true);
    });

    it('should send test notification to specific recipient', async () => {
      mockClient.request.mockResolvedValue({ success: true, message: 'Test sent' });

      const result = await api.sendTest({ channel: 'telegram', recipient: '123456' });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/notifications/test', {
        json: { channel: 'telegram', recipient: '123456' },
      });
    });

    it('should send notification immediately', async () => {
      mockClient.request.mockResolvedValue({ success: true, delivery_id: 'del_123' });

      const result = await api.send({
        channel: 'email',
        event_type: 'debate_completed',
        subject: 'Debate Complete',
        message: 'Your debate has been completed.',
        metadata: { debate_id: 'd_123' },
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/notifications/send', {
        json: {
          channel: 'email',
          event_type: 'debate_completed',
          subject: 'Debate Complete',
          message: 'Your debate has been completed.',
          metadata: { debate_id: 'd_123' },
        },
      });
      expect(result.delivery_id).toBe('del_123');
    });
  });

  // ===========================================================================
  // Delivery History
  // ===========================================================================

  describe('Delivery History', () => {
    it('should get delivery history', async () => {
      const mockHistory = {
        deliveries: [
          {
            id: 'd1',
            channel: 'email',
            event_type: 'debate_completed',
            recipient: 'user@example.com',
            subject: 'Debate Complete',
            delivered_at: '2024-01-20T10:00:00Z',
            success: true,
          },
          {
            id: 'd2',
            channel: 'telegram',
            event_type: 'security_alert',
            recipient: '123456',
            delivered_at: '2024-01-20T09:00:00Z',
            success: true,
          },
        ],
        total: 2,
      };
      mockClient.request.mockResolvedValue(mockHistory);

      const result = await api.getDeliveryHistory();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/notifications/history', {
        params: undefined,
      });
      expect(result.deliveries).toHaveLength(2);
    });

    it('should filter delivery history', async () => {
      const mockHistory = {
        deliveries: [{ id: 'd1', channel: 'email', success: false, error: 'SMTP error' }],
        total: 1,
      };
      mockClient.request.mockResolvedValue(mockHistory);

      const result = await api.getDeliveryHistory({
        channel: 'email',
        success: false,
        limit: 10,
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/notifications/history', {
        params: { channel: 'email', success: false, limit: 10 },
      });
      expect(result.deliveries[0].success).toBe(false);
    });

    it('should filter by event type', async () => {
      const mockHistory = { deliveries: [], total: 0 };
      mockClient.request.mockResolvedValue(mockHistory);

      await api.getDeliveryHistory({ event_type: 'budget_alert' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/notifications/history', {
        params: { event_type: 'budget_alert' },
      });
    });
  });

  // ===========================================================================
  // Notification Templates
  // ===========================================================================

  describe('Notification Templates', () => {
    it('should list notification templates', async () => {
      mockClient.request.mockResolvedValue({
        templates: [{ id: 'debate_completed', name: 'Debate Completed', subject: 'Done', body: 'Hi' }],
        count: 1,
      });

      const result = await api.listTemplates();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/notifications/templates');
      expect(result.count).toBe(1);
      expect(result.templates[0].id).toBe('debate_completed');
    });

    it('should get a notification template by id', async () => {
      mockClient.request.mockResolvedValue({
        template: { id: 'budget_alert', name: 'Budget Alert', subject: 'Alert', body: 'Hi' },
      });

      const result = await api.getTemplate('budget_alert');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/notifications/templates/budget_alert'
      );
      expect(result.template.id).toBe('budget_alert');
    });

    it('should update a notification template', async () => {
      mockClient.request.mockResolvedValue({
        template: { id: 'budget_alert', name: 'Budget Alert', subject: 'Updated', body: 'Hi' },
        updated: true,
      });

      const result = await api.updateTemplate('budget_alert', {
        subject: 'Updated',
        body: 'Hi',
      });

      expect(mockClient.request).toHaveBeenCalledWith(
        'PUT',
        '/api/notifications/templates/budget_alert',
        {
          json: {
            subject: 'Updated',
            body: 'Hi',
          },
        }
      );
      expect(result.updated).toBe(true);
    });

    it('should reset a notification template', async () => {
      mockClient.request.mockResolvedValue({
        template: { id: 'budget_alert', name: 'Budget Alert', subject: 'Default', body: 'Hi' },
        reset: true,
      });

      const result = await api.resetTemplate('budget_alert');

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/notifications/templates/budget_alert/reset'
      );
      expect(result.reset).toBe(true);
    });

    it('should preview a notification template with values', async () => {
      mockClient.request.mockResolvedValue({
        template_id: 'budget_alert',
        rendered_subject: 'Budget alert: 80% used',
        rendered_body: 'Hello Alex',
        values_used: { percent_used: '80', user_name: 'Alex' },
      });

      const result = await api.previewTemplate('budget_alert', {
        percent_used: '80',
        user_name: 'Alex',
      });

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/notifications/templates/budget_alert/preview',
        {
          json: {
            values: { percent_used: '80', user_name: 'Alex' },
          },
        }
      );
      expect(result.rendered_subject).toContain('80%');
    });
  });
});
